"""Attribute head clearance near the saved roll's first large support gap."""
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linear_sum_assignment

from s1_reference_inputs import ROOT, read, sha
from collect_roll_confirmation import prepare
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from roll_metrics import progress, summarize


OUT = ROOT / 'records/reports/roll-common-confirmation/rng-recovery'
LANDMARK_DEG = 160.
JOINTS = ('neck_pitch', 'head_pitch')
FEATURES = ('angle_deg', 'pitch_rate_rad_s', 'root_normal_mm_s', 'body_lateral')
MATCH_LIMITS = np.asarray([4., 1., 180., .15])
RESULTS = ('clearance_mm', 'normal_velocity_mm_s', 'base_contribution_mm_s', 'neck_contribution_mm_s',
           'head_pitch_contribution_mm_s', 'other_contribution_mm_s', 'next_neck_demand_deg', 'next_head_demand_deg')


class Geometry:
    def __init__(self, model):
        self.model, self.data = model, mujoco.MjData(model)
        self.head = model.geom('robot/head_collision').id
        self.floor = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).item()
        assert model.geom_type[self.head] == mujoco.mjtGeom.mjGEOM_MESH
        mesh = model.geom_dataid[self.head]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        self.vertices = model.mesh_vert[start:start+count].copy()
        root = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).item()
        assert model.jnt_qposadr[root] == model.jnt_dofadr[root] == 0
        self.joints = [model.joint('robot/' + name).id for name in JOINTS]
        self.dofs = model.jnt_dofadr[self.joints]
        self.addresses = model.jnt_qposadr[self.joints]
        self.ranges = model.jnt_range[self.joints]
        self.jac = np.zeros((3, model.nv))

    def measure(self, qpos, qvel):
        model, data = self.model, self.data
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        assert data.time == 0.
        normal = data.geom_xmat[self.floor].reshape(3, 3)[:, 2]
        rotation = data.geom_xmat[self.head].reshape(3, 3)
        world = self.vertices @ rotation.T + data.geom_xpos[self.head]
        distance = (world - data.geom_xpos[self.floor]) @ normal
        point = int(distance.argmin())
        mujoco.mj_jac(model, data, self.jac, None, world[point], int(model.geom_bodyid[self.head]))
        parts = 1000 * (normal @ self.jac) * qvel
        base = float(parts[:6].sum())
        neck, head = map(float, parts[self.dofs])
        total = float(parts.sum())
        return dict(clearance_mm=float(1000 * distance[point]), normal_velocity_mm_s=total,
            base_contribution_mm_s=base, neck_contribution_mm_s=neck, head_pitch_contribution_mm_s=head,
            other_contribution_mm_s=total-base-neck-head, root_normal_mm_s=float(1000 * (normal @ qvel[:3])),
            cpu_floor_contact=any(set(map(int, c.geom)) == {self.head, self.floor} and c.dist < 0 for c in data.contact))


def match(rows):
    successes = [row for row in rows if row['success'] and row['landmark']]
    failures = [row for row in rows if not row['success'] and row['landmark']]
    if not successes or not failures:
        return []
    values = lambda rows: np.array([[row['landmark'][name] for name in FEATURES] for row in rows])
    difference = abs(values(failures)[:, None] - values(successes)[None, :]) / MATCH_LIMITS
    legal = (difference <= 1.).all(axis=-1)
    left, right = linear_sum_assignment(np.where(legal, np.linalg.norm(difference, axis=-1), 1e6))
    return [dict(failure=failures[i]['world'], success=successes[j]['world'],
        differences={key: failures[i]['landmark'][key] - successes[j]['landmark'][key] for key in RESULTS})
        for i, j in zip(left, right, strict=True) if legal[i, j]]


def review(trace, handoff, geometry, order):
    geometric, _, rotation = progress(trace)
    pitch = np.rad2deg(geometric)
    columns = [order.index(name) for name in JOINTS]
    rows = []
    for row in handoff:
        if not row['gap'] or row['gap']['last_supported_type'] != 'head':
            continue
        world = row['world']
        head = round(row['head_seconds'] / .02)
        before = round(row['gap']['start_seconds'] / .02) - 1
        assert trace['head_contact'][before, world] and trace['active'][before, world]

        def at(k):
            qpos = trace['qpos'][k, world]
            velocity = trace['qvel'][k, world]
            measured = geometry.measure(qpos, velocity)
            joints = trace['joint_positions'][k, world, columns]
            np.testing.assert_array_equal(qpos[geometry.addresses], joints)
            targets = trace['applied_targets'][k+1, world, columns]
            demands = np.rad2deg(targets - joints)
            lower, upper = geometry.ranges.T
            margin = np.rad2deg(np.minimum(joints-lower, upper-joints))
            return dict(measured, time_s=float(trace['time'][k]), angle_deg=float(pitch[k, world]),
                pitch_rate_rad_s=float(trace['angular_velocity'][k, world, 1]),
                body_lateral=float(rotation[k, world, 2, 1]),
                next_neck_demand_deg=float(demands[0]), next_head_demand_deg=float(demands[1]),
                actuation={name: dict(actual_angle_deg=float(np.rad2deg(joints[i])),
                    actual_rate_deg_s=float(np.rad2deg(velocity[geometry.dofs[i]])),
                    next_target_deg=float(np.rad2deg(targets[i])), nearest_joint_limit_deg=float(margin[i]),
                    next_target_in_range=bool(lower[i] <= targets[i] <= upper[i]))
                    for i, name in enumerate(JOINTS)})

        candidates = np.arange(head, before+1)
        offsets = abs(pitch[candidates, world] - LANDMARK_DEG)
        supported = trace['head_contact'][candidates, world] & (offsets <= 8.)
        i = int(np.argmin(np.where(supported, offsets, np.inf)))
        rows.append(dict(world=world, success=row['success'], before_air=at(before),
                         landmark=at(int(candidates[i])) if supported.any() else None))
    pairs = match(rows)
    by_role = {}
    for success, label in ((True, 'success'), (False, 'failure')):
        selected = [row for row in rows if row['success'] == success]
        actuation = {}
        for name in JOINTS:
            samples = [row['before_air']['actuation'][name] for row in selected]
            actuation[name] = dict(inside_limit_requests=sum(row['next_target_in_range'] for row in samples),
                smallest_actual_margin_deg=min(row['nearest_joint_limit_deg'] for row in samples),
                median_actual_rate_deg_s=float(np.median([row['actual_rate_deg_s'] for row in samples])))
        by_role[label] = dict(eligible=len(selected), at_landmark=sum(row['landmark'] is not None for row in selected),
            before_air_medians={key: float(np.median([row['before_air'][key] for row in selected])) for key in RESULTS},
            joint_envelope_at_departure=actuation)
    return dict(rows=rows, matched_pairs=pairs, grouped=by_role,
        median_failure_minus_success={key: float(np.median([p['differences'][key] for p in pairs]))
                                      for key in RESULTS} if pairs else {})


def main():
    import onnx
    from mjlab.scene import Scene

    handoff_path = OUT / 'development-contact-handoff.json'
    handoff = read(handoff_path)
    assert handoff['status'] == 'passed-saved-development-data-only' and not handoff['heldout_used']
    cohorts, sources, model = {}, {handoff_path.relative_to(ROOT).as_posix(): sha(handoff_path)}, None
    for role, source in handoff['cohorts'].items():
        proof_path = OUT / f'{role}-analysis.json'
        proof = read(proof_path)
        assert not proof['heldout_used'] and role in ('development-a', 'development-b')
        folder = ROOT / proof['evaluator']
        plan_path, trace_path = folder / 'protocol.json', folder / 'forward-roll-trace.npz'
        scene_path = folder / 'scene-static-fields.npz'
        plan = read(plan_path)
        assert sha(trace_path) == proof['trace_sha256']
        if model is None:
            model = Scene(prepare(plan).scene, device='cpu').compile()
        with np.load(scene_path) as stored:
            compare_arrays(model_arrays(model), {key: stored[key] for key in stored.files})
        actor = ROOT / plan['model']
        assert sha(actor) == plan['model_sha256']
        order = next(x.value for x in onnx.load(actor).metadata_props if x.key == 'joint_names').split(',')
        with np.load(trace_path) as data:
            trace = {name: data[name] for name in data.files}
        score = summarize(trace, plan['criteria'])
        assert [row['success'] for row in score['episodes']] == [row['success'] for row in source['rows']]
        checked = cohorts[role] = review(trace, source['rows'], Geometry(model), order)
        print('ROLL_HEAD_SUPPORT_GEOMETRY', role, len(checked['matched_pairs']),
              checked['median_failure_minus_success'], checked['grouped'], flush=True)
        for path in (proof_path, plan_path, trace_path, scene_path, actor):
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
    write(OUT / 'development-head-support-geometry.json', dict(status='passed-saved-data-only', cohorts=cohorts,
        source_sha256=sources, script_sha256=sha(Path(__file__)), heldout_used=False, learning_updates=0, new_physical_steps=0,
        landmark_deg=LANDMARK_DEG, matching_limits=dict(zip(FEATURES, MATCH_LIMITS.tolist(), strict=True)),
        scope='CPU kinematics of the closest head-mesh point at exact archived qpos/qvel. Jacobian components are velocities, not contact forces.'))


if __name__ == '__main__':
    main()

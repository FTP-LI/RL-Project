"""Audit saved roll support transfer from the head to the feet on CPU."""
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linear_sum_assignment

from collect_roll_confirmation import prepare
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_roll_contact_handoff import OUT
from roll_metrics import progress, summarize
from s1_reference_inputs import ROOT, read, sha


FEATURES = ('phase_deg', 'pitch_rate_rad_s', 'lateral_projection', 'root_height_m')
MATCH_LIMITS = (12., 1.2, .15, .02)
LEGS = ('left', 'right')


class SoleGeometry:
    def __init__(self, model):
        self.model, self.data = model, mujoco.MjData(model)
        self.floor = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).item()
        self.feet = {}
        for side in LEGS:
            geom = model.geom(f'robot/{side}_foot_collision').id
            assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            leg_joints = [model.joint(f'robot/{side}_{name}').id for name in ('hip_pitch', 'knee', 'ankle')]
            self.feet[side] = geom, model.mesh_vert[start:start+count].copy(), model.jnt_dofadr[leg_joints]
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).item()
        assert model.jnt_qposadr[free] == model.jnt_dofadr[free] == 0
        self.jacobian = np.zeros((3, model.nv))

    def measure(self, qpos, qvel):
        m, d = self.model, self.data
        d.qpos[:], d.qvel[:] = qpos, qvel
        mujoco.mj_forward(m, d)
        np.testing.assert_array_equal(d.qpos, qpos)
        np.testing.assert_array_equal(d.qvel, qvel)
        assert d.time == 0.
        normal = d.geom_xmat[self.floor].reshape(3, 3)[:, 2]
        measurements = {}
        for side, (geom, vertices, dofs) in self.feet.items():
            world = vertices @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]
            clearance = (world-d.geom_xpos[self.floor]) @ normal
            lowest = int(clearance.argmin())
            mujoco.mj_jac(m, d, self.jacobian, None, world[lowest], int(m.geom_bodyid[geom]))
            parts = 1_000*(normal @ self.jacobian)*qvel
            measurements[side] = dict(clearance_mm=float(1_000*clearance[lowest]),
                vertical_velocity_mm_s=float(parts.sum()), floating_base_mm_s=float(parts[:6].sum()),
                hip_knee_ankle_mm_s=dict(zip(('hip_pitch', 'knee', 'ankle'), parts[dofs].tolist(), strict=True)))
        return measurements


def summarize_group(rows):
    def median(values):
        return float(np.median(values)) if len(values) else None

    found = [row for row in rows if row['first_foot'] is not None]
    return dict(cases=len(rows), first_return_head_only=sum(row['return_support'] == 'head' for row in rows),
        first_foot_observed_by_deadline=len(found), first_foot_within_120_ms=sum(row['first_foot']['delay_s'] <= .120001 for row in found),
        departure_nearest_sole_mm_median=median([row['departure']['nearest_sole_mm'] for row in rows]),
        first_return_nearest_sole_mm_median=median([row['return']['nearest_sole_mm'] for row in rows if row['return']]),
        first_foot_delay_s_median=median([row['first_foot']['delay_s'] for row in found]),
        first_foot_phase_deg_median=median([row['first_foot']['phase_deg'] for row in found]),
        first_foot_supported_progress_deg_median=median([row['first_foot']['supported_progress_deg'] for row in found]))


def match(rows):
    cases = [row for row in rows if not row['success'] and row['return_support'] == 'head']
    controls = [row for row in rows if row['success'] and row['return_support'] == 'head']
    if not cases or not controls:
        return []
    case_x, control_x = (np.array([[row['departure'][key] for key in FEATURES] for row in group])
                         for group in (cases, controls))
    distance = np.abs(case_x[:, None] - control_x[None, :])/MATCH_LIMITS
    allowed = np.all(distance <= 1., axis=-1)
    i, j = linear_sum_assignment(np.where(allowed, np.linalg.norm(distance, axis=-1), 1e6))
    pairs = []
    for failure, success in ((cases[a], controls[b]) for a, b in zip(i, j, strict=True) if allowed[a, b]):
        foot_a, foot_b = failure['first_foot'], success['first_foot']
        pairs.append(dict(failed_world=failure['world'], successful_world=success['world'],
            departure_deltas={k: failure['departure'][k]-success['departure'][k] for k in FEATURES},
            departure_nearest_sole_delta_mm=failure['departure']['nearest_sole_mm']-success['departure']['nearest_sole_mm'],
            first_return_nearest_sole_delta_mm=failure['return']['nearest_sole_mm']-success['return']['nearest_sole_mm'],
            first_foot_delay_delta_s=foot_a['delay_s']-foot_b['delay_s'] if foot_a and foot_b else None,
            first_foot_phase_delta_deg=foot_a['phase_deg']-foot_b['phase_deg'] if foot_a and foot_b else None))
    return pairs


def review(trace, plan, recorded, geometry):
    criteria = plan['criteria']
    checked = summarize(trace, criteria)
    geometric, supported, rotation = progress(trace)
    phase, valid_angle = np.rad2deg(geometric), np.rad2deg(supported)
    times, step_dt = trace['time'], criteria['step_dt']
    deadline = round(criteria['deadline_seconds']/step_dt)
    rows = []
    for previous, event in zip(recorded, checked['episodes'], strict=True):
        assert previous['world'] == event['env_index'] and previous['success'] == event['success']
        assert previous['head_seconds'] == event['head_contact_seconds']
        gap, world = previous['gap'], previous['world']
        if gap is None or gap['last_supported_type'] != 'head':
            continue
        begin = round(gap['start_seconds']/step_dt)
        departure, end = begin-1, begin+gap['air_steps']
        assert trace['head_contact'][departure, world] and not trace['foot_contact'][departure, world].any()
        assert not trace['ground_contact'][begin:end, world].any() and trace['active'][begin:end, world].all()
        assert abs(phase[departure, world]-gap['before_takeoff_deg']) < 1e-8

        def at(k):
            feet = geometry.measure(trace['qpos'][k, world], trace['qvel'][k, world])
            return dict(time_s=float(times[k]), phase_deg=float(phase[k, world]),
                supported_progress_deg=float(valid_angle[k, world]), root_height_m=float(trace['height'][k, world]),
                pitch_rate_rad_s=float(trace['angular_velocity'][k, world, 1]),
                lateral_projection=float(rotation[k, world, 2, 1]),
                foot_contact_count=int(trace['foot_contact'][k, world].sum()),
                nearest_sole_mm=min(foot['clearance_mm'] for foot in feet.values()), feet=feet)

        returned = None
        if gap['return_support_type'] != 'unobserved':
            assert trace['active'][end, world] and trace['ground_contact'][end, world]
            returned = at(end)
        foot_steps = np.flatnonzero(trace['active'][begin:deadline+1, world] &
                                    trace['foot_contact'][begin:deadline+1, world].any(axis=-1))
        first = None
        if len(foot_steps):
            k = begin+int(foot_steps[0])
            assert k >= end
            first = dict(at(k), delay_s=float(times[k]-times[departure]))
        rows.append({'world': world, 'success': event['success'], 'return_support': gap['return_support_type'],
                     'departure': at(departure), 'return': returned, 'first_foot': first})
    pairs = match(rows)
    deltas = {}
    for key in ('departure_nearest_sole_delta_mm', 'first_return_nearest_sole_delta_mm',
                'first_foot_delay_delta_s', 'first_foot_phase_delta_deg'):
        values = [pair[key] for pair in pairs if pair[key] is not None]
        deltas[key] = dict(pairs=len(values), median=float(np.median(values)) if values else None)
    groups = {name: summarize_group([row for row in rows if row['success'] == success])
              for name, success in (('success', True), ('failure', False))}
    return dict(original_attempts=checked['attempts'], original_successes=checked['successes'], groups=groups,
                matched_head_recontacts=pairs, paired_failure_minus_success=deltas, rows=rows)


def main():
    from mjlab.scene import Scene

    handoff_path = OUT/'development-contact-handoff.json'
    target = OUT/'development-foot-transfer.json'
    assert not target.exists(), 'Preserve the original review receipt'
    handoff = read(handoff_path)
    assert handoff['status'] == 'passed-saved-development-data-only' and not handoff['heldout_used']
    cohorts, sources = {}, {handoff_path.relative_to(ROOT).as_posix(): sha(handoff_path)}
    for role in ('development-a', 'development-b'):
        proof_path = OUT/f'{role}-analysis.json'
        proof = read(proof_path)
        assert proof['status'] == 'passed' and not proof['heldout_used']
        folder = ROOT/proof['evaluator']
        plan_path, trace_path = folder/'protocol.json', folder/'forward-roll-trace.npz'
        scene_path = folder/'scene-static-fields.npz'
        plan = read(plan_path)
        assert sha(trace_path) == proof['trace_sha256'] and plan['seed'] in (1020, 1021)
        model = Scene(prepare(plan).scene, device='cpu').compile()
        with np.load(scene_path) as archive:
            compare_arrays(model_arrays(model), {name: archive[name] for name in archive.files})
        keys = ('time', 'active', 'height', 'quaternion', 'head_quaternion', 'angular_velocity', 'foot_contact',
                'head_contact', 'ground_contact', 'head_penetration_m', 'joint_envelope_ok', 'qpos', 'qvel')
        with np.load(trace_path) as archive:
            trace = {name: archive[name] for name in keys}
        row = cohorts[role] = review(trace, plan, handoff['cohorts'][role]['rows'], SoleGeometry(model))
        assert row['original_attempts'] == 128 and row['original_successes'] == proof['successes']
        for path in (proof_path, plan_path, trace_path, scene_path):
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
        print('ROLL_FOOT_TRANSFER', role, row['groups'], row['paired_failure_minus_success'], flush=True)
    write(target, dict(status='passed-saved-development-data-only', cohorts=cohorts, heldout_used=False,
          original_scores_unchanged=True, new_physical_steps=0, new_learning_updates=0, source_sha256=sources,
          match_limits=dict(zip(FEATURES, MATCH_LIMITS, strict=True)), script_sha256=sha(Path(__file__)),
          scope='Closest sole collision-mesh point relative to the plane at archived poses. Contact times use the saved GPU sensors. CPU kinematics do not give contact force or a treatment effect.'))


if __name__ == '__main__':
    main()

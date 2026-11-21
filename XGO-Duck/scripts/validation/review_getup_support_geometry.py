"""Read support geometry and joint demand around the saved contact exit."""
from pathlib import Path

import mujoco
import numpy as np

from audit_getup_contact_entry import OUT, source
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from s1_reference_inputs import ROOT, read, sha


OFFSETS = (0, 5, 8, 10, 12, 15, 25)
METRICS = ('forward_com_mm', 'forward_com_velocity_mm_s', 'com_height_mm',
           'head_forward_mm', 'head_height_mm')


def geometry_reader(model):
    data = mujoco.MjData(model)
    root = int(model.jnt_bodyid[np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).item()])
    head = model.geom('robot/head_collision').id
    feet = [model.site(f'robot/{side}_foot').id for side in ('left', 'right')]
    foot_geoms = [model.geom(f'robot/{side}_foot_collision').id for side in ('left', 'right')]

    def measure(qpos, qvel):
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        assert data.time == 0.
        direction = data.xmat[root].reshape(3, 3)[:2, 0].copy()
        direction /= np.linalg.norm(direction)
        assert np.isfinite(direction).all()
        origin = data.site_xpos[feet].mean(axis=0)
        com = data.subtree_com[root] - origin
        head_center = data.geom_xpos[head] - origin
        support = [[], []]
        for contact in data.contact:
            if contact.dist >= 0:
                continue
            a, b = map(int, contact.geom)
            for i, foot in enumerate(foot_geoms):
                other = b if a == foot else a if b == foot else -1
                if other >= 0 and model.body_rootid[model.geom_bodyid[other]] != root:
                    support[i].append(float(1000 * ((contact.pos - origin)[:2] @ direction)))
        span = sorted(support[0] + support[1])
        return dict(forward_com_mm=float(1000 * (com[:2] @ direction)),
                    forward_com_velocity_mm_s=float(1000 * (data.subtree_linvel[root, :2] @ direction)),
                    com_height_mm=float(1000 * com[2]), head_forward_mm=float(1000 * (head_center[:2] @ direction)),
                    head_height_mm=float(1000 * head_center[2]),
                    cpu_foot_floor_contacts=[len(points) for points in support],
                    cpu_contact_span_mm=None if not span else [span[0], span[-1]])

    return measure


def sample(trace, cases, measure, limits):
    output = {}
    for offset in OFFSETS:
        records = []
        for pair in cases:
            sampled = {}
            for role, case in pair.items():
                k, i = case['first_head_step'] - 1 + offset, case['world']
                assert trace['active'][k, i]
                g = measure(trace['qpos'][k, i], trace['qvel'][k, i])
                # Demand is the request at the next control relative to this
                # saved pose. The feasible part is bounded by the joint limits.
                req = trace['requested_targets'][k + 1, i]
                q = trace['joint_positions'][k, i]
                bounded = np.clip(req, limits[:, 0], limits[:, 1])
                sampled[role] = dict(g, world=i, native_both_feet=bool(trace['foot_contact'][k, i].all()),
                    feasible_joint_demand_deg=np.rad2deg(bounded - q).tolist(),
                    request_beyond_limits_deg=np.rad2deg(req - bounded).tolist())
            records.append(sampled)
        output[f'{.02 * offset:.2f}'] = records
    return output


def summarize(records, joints):
    data = {role: [row[role] for row in records] for role in ('success', 'failure')}
    differences = {key: np.asarray([b[key] - a[key] for a, b in zip(*data.values(), strict=True)])
                   for key in METRICS}
    demands = np.array([b['feasible_joint_demand_deg'] for b in data['failure']]) - np.array(
        [a['feasible_joint_demand_deg'] for a in data['success']])
    return dict(median_failure_minus_success={k: float(np.median(v)) for k, v in differences.items()},
                pairs_with_positive_difference={k: int((v > 0).sum()) for k, v in differences.items()},
                median_feasible_joint_demand_diff_deg=dict(zip(joints, np.median(demands, axis=0).tolist(), strict=True)),
                cpu_both_feet={role: sum(all(row['cpu_foot_floor_contacts']) for row in rows) for role, rows in data.items()},
                native_both_feet={role: sum(row['native_both_feet'] for row in rows) for role, rows in data.items()})


def main():
    import torch
    from mjlab.scene import Scene
    from getup_knee_support_supply_recovery import shared

    exit_path = OUT / 'exit-analysis.json'
    exit_audit = read(exit_path)
    assert exit_audit['status'] == 'passed-saved-data-only'
    spec = read(ROOT / 'configs/experiments/getup_knee_support_supply_recovery.json')['environment']
    cfg, _, meta = shared.prepare(spec)
    model = Scene(cfg.env.scene, device='cpu').compile()
    joints = meta['joint_names'].split(',')
    limits = np.asarray(spec['joint_limits'])
    assert len(joints) == len(limits) == 14
    geometry = geometry_reader(model)
    cohorts = {}
    for seed, cohort in exit_audit['cohorts'].items():
        _, trace, _, _, scene, fingerprints = source(seed, 'reference-a' if seed == '1054' else 'original')
        with np.load(scene) as stored:
            compare_arrays(model_arrays(model), {k: stored[k] for k in stored.files})
        for path, digest in fingerprints.items():
            assert digest == exit_audit['source_sha256'][path.relative_to(ROOT).as_posix()]
        records = sample(trace, cohort['paired_events'], geometry, limits)
        summary = {offset: summarize(rows, joints) for offset, rows in records.items()}
        cohorts[seed] = dict(pairs=cohort['pairs'], summary=summary, pairs_by_offset=records)
        print('SUPPORT_GEOMETRY', seed, {t: summary[t]['median_failure_minus_success']
              for t in ('0.16', '0.20', '0.24')}, flush=True)
    assert not torch.cuda.is_initialized()
    write(OUT / 'support-geometry.json', dict(status='passed-saved-data-only', cohorts=cohorts,
        exit_analysis_sha256=sha(exit_path), script_sha256=sha(Path(__file__)),
        forward_axis='Pelvis +x projected to ground; origin is the mean of the two foot sites',
        geometry_scope='COM uses verified scene mass and archived qpos/qvel. CPU contact positions are geometric, not ground reaction or pressure.',
        demand_scope='Joint-limit-bounded next request minus archived joint angle. This is a descriptive quantity, not recorded torque or a proposed controller.',
        new_physical_steps=0, learning_updates=0, heldout_used=False))


if __name__ == '__main__':
    main()

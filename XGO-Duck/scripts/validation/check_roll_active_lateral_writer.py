"""Audit the roll-time horizontal velocity writer on the actual CPU plant."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from s1_reference_inputs import WORK, read, sha
from check_roll_active_lateral_geometry import DRAFT, OUT, ROOT, lateral_world, pinned
from record_io import write
from roll_entry_snapshot import RollEntrySnapshot
from roll_stand_walk_sequence import ROLL
from run_continuous_roll_stand_walk import configuration


def apply_roll_lateral_push(env, eligible, departure_quaternion, magnitude):
    """Use the heading before rolling so roll pitch cannot rotate the push axis."""
    mask = np.asarray(eligible)
    heading = np.asarray(departure_quaternion)
    assert mask.shape == (env.num_envs,) and mask.dtype == np.bool_
    assert heading.shape == (env.num_envs, 4) and abs(magnitude) <= .08
    ids = torch.from_numpy(np.flatnonzero(mask)).to(env.device)
    horizontal = env.scene['robot'].data.indexing.free_joint_v_adr[:2]
    qvel = env.sim.data.qvel
    before = qvel[ids[:, None], horizontal]
    _, delta = lateral_world(heading[mask], magnitude)
    target = torch.as_tensor(delta[:, :2], dtype=before.dtype, device=env.device)
    if magnitude:
        qvel[ids[:, None], horizontal] = before + target
    measured = qvel[ids[:, None], horizontal] - before
    torch.testing.assert_close(measured, target, rtol=0., atol=1e-7)
    return dict(world_ids=ids.cpu().numpy(), target_world_xy=target.cpu().numpy(), measured_world_xy=measured.cpu().numpy())


def main():
    from mjlab.entity.data import EntityData
    from mjlab.envs import ManagerBasedRlEnv

    target = OUT / 'writer-cpu.json'
    assert not target.exists()
    draft = read(DRAFT)
    geo_path = OUT / 'geometry-preflight.json'
    geo = read(geo_path)
    assert geo['status'] == 'passed-saved-roll-disturbance-geometry' and geo['physical_substeps'] == 0
    for name, expected in geo['sources'].items():
        assert sha(ROOT / name) == expected
    proof_path = pinned(draft['nominal_result'])
    proof = read(proof_path)
    run = ROOT / proof['record']
    p = read(run / 'protocol.json')
    prior_push = read(pinned(draft['previous_lateral_result']))
    prior_protocol = ROOT / prior_push['record'] / 'protocol.json'
    assert sha(prior_protocol) == prior_push['protocol_sha256']
    warp_limit = read(prior_protocol)['maximum_warp_snapshot_bytes']
    k, departure = geo['injection_control'], geo['standing_heading_control']
    with np.load(run / 'trace.npz', allow_pickle=False) as saved:
        qpos, qvel = (saved[key][k].copy() for key in ('qpos', 'qvel'))
        heading = saved['qpos'][departure, :, 3:7].copy()
        eligible = saved['active'][k] & (saved['phase_next'][k] == ROLL) & ~saved['head_contact'][departure:k+1].any(axis=0)
    assert int(eligible.sum()) == geo['eligible']
    library = Path(inspect.getsourcefile(EntityData))
    assert library.is_relative_to(WORK)
    assert 'torch.cat([velocity[:, :3], ang_vel_b]' in inspect.getsource(EntityData.write_root_velocity)
    torch.set_num_threads(4)
    env = ManagerBasedRlEnv(cfg=configuration(p), device='cpu', render_mode=None)
    checks = {}
    try:
        env.reset(seed=p['seed'])
        env.sim.data.qpos.copy_(torch.from_numpy(qpos))
        env.sim.data.qvel.copy_(torch.from_numpy(qvel))
        memory = SimpleNamespace(previous=torch.zeros(env.num_envs, 14), history=torch.zeros(env.num_envs, 14),
            previous_features=np.zeros((env.num_envs, 2), dtype=np.float32))
        snapshot = RollEntrySnapshot(env, memory, warp_limit)
        proof = snapshot.verify()
        assert proof['counters'] == [0, 0]
        addresses = env.scene['robot'].data.indexing.free_joint_v_adr[:2].tolist()
        untouched = np.ones(qvel.shape, dtype=bool)
        untouched[np.ix_(eligible, addresses)] = False
        for role, amount in draft['groups_horizontal_velocity_m_s'].items():
            pushed = apply_roll_lateral_push(env, eligible, heading, amount)
            np.testing.assert_array_equal(env.sim.data.qpos.numpy(), qpos)
            np.testing.assert_array_equal(env.sim.data.qvel.numpy()[untouched], qvel[untouched])
            changed = [key for key, value in snapshot.live_signatures().items() if value != snapshot.expected[key]]
            assert changed == (['warp.qvel'] if amount else [])
            assert snapshot.restore() == proof
            checks[role] = dict(touched_fields=changed, affected_worlds=len(pushed['world_ids']) if amount else 0,
                max_horizontal_error_m_s=float(abs(pushed['measured_world_xy']-pushed['target_world_xy']).max()))
        assert env.common_step_counter == env._sim_step_counter == 0
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    write(target, dict(status='passed-cpu-roll-lateral-write-and-restore', physical_substeps=0, hardware_io=False,
        heading_from_control=departure, fixture_from_control=k, eligible=int(eligible.sum()), cases=checks, snapshot=proof,
        geometry_sha256=sha(geo_path), writer_sha256=sha(Path(__file__)), library_sha256=sha(library)))
    print('ROLL_ACTIVE_LATERAL_WRITER_CPU', len(snapshot.expected), int(eligible.sum()), flush=True)


if __name__ == '__main__':
    main()

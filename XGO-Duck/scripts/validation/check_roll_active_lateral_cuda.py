"""Qualify the roll disturbance write and full-state restore on actual CUDA."""

from dataclasses import asdict
import fcntl
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from check_roll_active_lateral_writer import DRAFT, OUT, ROOT, ROLL, apply_roll_lateral_push, pinned, read, sha
from record_io import write
from roll_endpoint_readback_capture import EndpointReadback
from roll_entry_snapshot import RollEntrySnapshot
from run_continuous_roll_stand_walk import STARTUP, actors, configuration, cpu, install_guard
from run_getup_latched_head_corrective_ppo import require_stage


SEED = 1173


def sources():
    draft = read(DRAFT)
    assert not any(draft[key] for key in ('physical_substeps_authorized', 'heldout_used', 'hardware_io',
        'learning_updates', 'automatic_retry', 'automatic_extension'))
    nominal, old = (pinned(draft[key]) for key in ('nominal_result', 'previous_lateral_result'))
    prior, older = read(nominal), read(old)
    assert prior['passed_development_gate'] and all(row['passed_gate'] for row in older['branches'].values())
    run = ROOT / prior['record']
    protocol, archive = run / 'protocol.json', run / 'trace.npz'
    previous_protocol = ROOT / older['record'] / 'protocol.json'
    assert sha(protocol) == prior['protocol_sha256'] and sha(previous_protocol) == older['protocol_sha256']
    assert sha(archive) == prior['source_sha256'][archive.relative_to(ROOT).as_posix()]
    geometry, cpu_receipt = OUT / 'geometry-preflight.json', OUT / 'writer-cpu.json'
    g, c = read(geometry), read(cpu_receipt)
    assert c['status'] == 'passed-cpu-roll-lateral-write-and-restore' and c['geometry_sha256'] == sha(geometry)
    assert g['status'] == 'passed-saved-roll-disturbance-geometry' and g['eligible'] == c['eligible']
    for name, expected in g['sources'].items():
        assert sha(ROOT / name) == expected, name
    writer = Path(inspect.getsourcefile(apply_roll_lateral_push)).resolve()
    assert sha(writer) == c['writer_sha256']
    files = (DRAFT, nominal, old, protocol, previous_protocol, archive, geometry, cpu_receipt, writer, Path(__file__).resolve())
    return draft, read(protocol), archive, read(previous_protocol)['maximum_warp_snapshot_bytes'], c, files


def main():
    from mjlab.entity.data import EntityData
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    draft, parent, archive, warp_limit, cpu_receipt, files = sources()
    require_stage(DRAFT, draft['stage'])
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{SEED}_' in path.name for path in (ROOT / folder).iterdir())
    library = Path(inspect.getsourcefile(EntityData)).resolve()
    assert sha(library) == cpu_receipt['library_sha256']
    start, fork = draft['standing_heading_control'], draft['injection_control']
    assert parent['initial_stand_controls'] == start and 0 <= start < fork < parent['maximum_controls']
    with np.load(archive, allow_pickle=False) as saved:
        qpos, qvel = (saved[key][fork].copy() for key in ('qpos', 'qvel'))
        heading = saved['qpos'][start, :, 3:7].copy()
        eligible = saved['active'][fork] & (saved['phase_next'][fork] == ROLL) & ~saved['head_contact'][start:fork+1].any(axis=0)
    assert int(eligible.sum()) == cpu_receipt['eligible']
    config = dict(parent, seed=SEED, roll=dict(parent['roll'], seed=SEED))
    folder = OUT / 'zero'
    archived = {path.relative_to(ROOT).as_posix(): sha(path) for path in files}

    def forbid_physics(*args, **kwargs):
        raise AssertionError('CUDA disturbance qualification must use zero physical steps')

    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        folder.mkdir()
        write(folder / 'fixture-protocol.json', dict(status='frozen-before-zero-step', seed=SEED,
            heading_control=start, fixture_control=fork, expected_worlds=int(eligible.sum()), physical_substep_budget=0,
            sources=archived, roles=draft['groups_horizontal_velocity_m_s']))
        cfg = configuration(config)
        dump_yaml(folder / 'env.yaml', asdict(cfg))
        guard = install_guard(config['roll'], folder, STARTUP, 'rollout')
        env = guard(cfg=cfg, device='cuda:0', render_mode=None)
        try:
            with patch.object(env.sim, 'step', forbid_physics):
                env.seed(SEED)
                obs, _ = env.reset()
                actors(env, obs, config, folder)
                endpoint = EndpointReadback(env, folder)
                endpoint.capture()
                assert not endpoint.first_mismatches
                endpoint.save(1)
                home = env.action_manager.get_term('joint_pos')._processed_actions
                memory = SimpleNamespace(previous=torch.zeros_like(home), history=torch.zeros_like(home),
                    previous_features=np.zeros((env.num_envs, 2), dtype=np.float32))
                initial = RollEntrySnapshot(env, memory, warp_limit)
                initial_receipt = initial.verify()
                env.sim.data.qpos.copy_(torch.from_numpy(qpos))
                env.sim.data.qvel.copy_(torch.from_numpy(qvel))
                fixture = RollEntrySnapshot(env, memory, warp_limit)
                fixture_receipt = fixture.verify()
                coordinates = env.scene['robot'].data.indexing.free_joint_v_adr[:2].tolist()
                assert coordinates == [0, 1]
                untouched = np.ones(qvel.shape, dtype=bool)
                untouched[np.ix_(eligible, coordinates)] = False
                cases = {}
                for role, speed in draft['groups_horizontal_velocity_m_s'].items():
                    seen = apply_roll_lateral_push(env, eligible, heading, speed)
                    np.testing.assert_array_equal(cpu(env.sim.data.qpos), qpos)
                    np.testing.assert_array_equal(cpu(env.sim.data.qvel)[untouched], qvel[untouched])
                    after = fixture.live_signatures()
                    changed = [name for name in fixture.expected if after[name] != fixture.expected[name]]
                    assert changed == (['warp.qvel'] if speed else [])
                    saved = folder / f'{role}-velocity.npz'
                    np.savez_compressed(saved, **seen)
                    assert fixture.restore() == fixture_receipt
                    cases[role] = dict(changed_fields=changed, affected_worlds=len(seen['world_ids']) if speed else 0,
                        maximum_readback_error_m_s=float(np.abs(seen['target_world_xy']-seen['measured_world_xy']).max()),
                        archive_sha256=sha(saved))
                assert initial.restore() == initial_receipt
                assert env.common_step_counter == env._sim_step_counter == env.guard_steps == 0 and len(env.guard_resets) == 1
        finally:
            env.close()
    write(OUT / 'cuda-zero.json', dict(status='passed-actual-cuda-zero-step', seed=SEED, physical_substeps=0,
        fixture_control=fork, heading_control=start, eligible=int(eligible.sum()), cases=cases, fixture=fixture_receipt,
        initial=initial_receipt, source_sha256=archived, initial_archives={name:sha(folder/name) for name in (
            'startup.npz', 'endpoint-contact.npz', 'runtime-contract.json', 'fixture-protocol.json')},
        library_sha256=sha(library), heldout_used=False, hardware_io=False))
    print('ROLL_ACTIVE_LATERAL_CUDA_ZERO', int(eligible.sum()), fixture_receipt['fields'], cases, flush=True)


if __name__ == '__main__':
    main()

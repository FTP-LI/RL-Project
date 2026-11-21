"""Observe and independently score the one-use ramped-sequence holdout."""

from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch


PLAN = Path(__file__).resolve().parents[2] / 'configs/experiments/continuous_routed_getup_ramped_forward_holdout.json'


def observe():
    import verify_routed_forward_entry_ramp as windows

    with patch.object(windows, 'PLAN', PLAN):
        windows.observe(windows.read(PLAN))


def verify():
    import imageio_ffmpeg
    import numpy as np

    import run_routed_getup_ramped_forward_holdout as study
    from verify_routed_getup_ramped_forward import audit_targets
    from verify_routed_getup_sequence import score

    study.phase(PLAN)
    read, sha, write = study.read, study.sha, study.write
    spec = read(PLAN)
    checksum = sha(PLAN)
    source = study.route.pinned(spec['holdout']['development_protocol'])
    assert study.dynamics(spec) == study.dynamics(read(source))
    study.base.shared.assert_sources(spec)
    record = study.ROOT / spec['record']
    result_path = study.OUT / 'analysis.json'
    assert not result_path.exists()
    receipt = read(study.RECEIPT)
    state, view = read(record / 'evaluation.json'), read(record / 'windows-view-verification.json')
    assert receipt['status'] == 'claimed-before-physics' and receipt['seed'] == spec['environment']['seed'] == study.SEED
    assert not receipt['retry_allowed'] and not receipt['holdout_eligible_for_tuning']
    assert receipt['record'] == spec['record'] and receipt['protocol_sha256'] == checksum == sha(record / 'protocol.json')
    assert receipt['actual_before_claim'] == dict(controls=0, attempted_controls=0, substeps=0, attempted_substeps=0, resets=1)
    assert receipt['native_zero_sha256'] == sha(study.OUT / 'native-zero.json')
    assert receipt['observer_ready_sha256'] == sha(record / 'windows-observer-ready.json')
    assert state['holdout_consumption_sha256'] == sha(study.RECEIPT)
    assert state['data_scope'] == spec['data_scope'] == 'reserved-holdout' and state['heldout_used'] is True
    assert state['status'] == 'completed' and state['actor_weights_unchanged'] and state['protocol_sha256'] == checksum
    total, n = spec['maximum_unique_controls'], spec['environment']['num_envs']
    assert state['actual'] == dict(controls=total, attempted_controls=total, substeps=total*5,
        attempted_substeps=total*5, resets=1)
    assert state['actual_control_transitions'] == total*n and state['contact_peak'] < state['contact_capacity']
    assert view['status'] == 'passed' and view['running_snapshots_with_original_frame'] >= 3
    with np.load(record / 'initial.npz', allow_pickle=False) as saved:
        initial = {key: saved[key] for key in saved.files}
    trace, hashes, boundaries = audit_targets(spec, state, record, initial)
    results = score(spec, state, record, trace, initial)
    passed = results['passed_development_gate']
    clip = state['video']
    video = record / clip['file']
    assert clip['full_action'] and clip['complete_clip'] and not clip['extra_physics_steps'] and clip['environment_index'] == 0
    assert clip['common_steps'] == list(range(0, total+1, 2)) and clip['qpos_sha256_by_frame'] == hashes
    assert sha(video) == clip['sha256']
    frames = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    metadata = next(frames)
    assert metadata['fps'] == 25 and metadata['size'] == (640, 480)
    assert sum(1 for _ in frames) == len(hashes) == 801
    write(result_path, dict(status='passed-independent-verification', record=spec['record'], seed=study.SEED,
        protocol_sha256=checksum, consumption_sha256=sha(study.RECEIPT), dynamics_match_development=True,
        final_holdout_passed=passed, holdout_eligible_for_tuning=False, physical_resets=1,
        physical_substeps=total*5, transitions=total*n, policy=boundaries, score=results,
        video=dict(path=study.base.rel(video), sha256=clip['sha256'], frames=len(hashes), seconds=32., world=0),
        learning_updates=0, heldout_used=True, hardware_io=False))
    print('ROUTED_RAMPED_FINAL_HOLDOUT', results['getup']['overall']['timely_holds'], results['standing'],
          results['forward_passes'], results['complete'], passed, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('observe', 'verify'))
    (observe if parser.parse_args().mode == 'observe' else verify)()

"""Recheck matched training budgets, actual learning and all recorded frames."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np
import torch

from run_local_getup_ppo import PLAN, arm_plan
import verify_getup_supine_training as previous
from record_io import write
from s1_reference_inputs import ROOT, read, sha


AUDIT = ROOT/'configs/experiments/local_getup_ppo_verification.json'


def verify_clips(record, trace, clips):
    samples = read(record/'recording-states.json')
    cursor, result = 0, []
    assert clips and clips[0]['full_action']
    for clip in clips:
        video = record/clip['file']
        steps = clip['common_steps']
        start, end = steps[0], steps[-1]
        assert start % 400 == 0 and steps == list(range(start, end+1, 2))
        assert end <= len(trace['qpos'])
        full = end-start == 400
        assert clip['full_action'] == clip['complete_clip'] == full
        assert clip['frames'] == len(steps) and clip['extra_physics_steps'] == 0
        np.testing.assert_allclose(clip['episode_times_seconds'], (np.asarray(steps)-start)*.02, atol=1e-12, rtol=0)
        frames = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
        assert next(frames)['fps'] == 25 and sum(1 for _ in frames) == len(steps)
        for step in steps:
            qpos = trace['pre_qpos'][step, 0] if step == start else trace['qpos'][step-1, 0]
            assert samples[cursor] == dict(common_step=step, qpos_sha256=hashlib.sha256(qpos.tobytes()).hexdigest())
            cursor += 1
        result.append(dict(path=video.relative_to(ROOT).as_posix(), sha256=sha(video), start_control=start,
                           frames=len(steps), fps=25, physical_seconds=(end-start)*.02, full_action=full,
                           terminal_state_verified=full))
    assert cursor == len(samples)
    return result


def main():
    audit = read(AUDIT)
    assert audit['status'] == 'frozen-training-integrity-check' and audit['training_protocol_sha256'] == sha(PLAN)
    for filename, expected in audit['source_sha256'].items():
        assert sha(ROOT/filename) == expected, filename
    spec = read(PLAN)
    out = ROOT/spec['coordinator']
    coordinator = read(out/'coordinator.json')
    assert coordinator['status'] == 'collected-review-required' and coordinator['protocol_sha256'] == sha(PLAN)
    results = {}
    for role in spec['order']:
        p = arm_plan(spec, role)
        record = ROOT/'records/experiments'/p['run_id']
        zero = ROOT/spec['zero_step_record']/role
        receipt = read(record/'online-result.json')
        obs = read(record/'observation.json')
        raw = [json.loads(line) for line in (record/'learning-metrics.jsonl').read_text().splitlines()]
        assert sha(record/'observation.json') == receipt['observation_sha256']
        assert sha(record/'learning-metrics.jsonl') == receipt['samples_sha256']
        assert len(raw) == len(obs['rows']) == p['training_updates']
        assert all(all(row[key] == value for key, value in original.items())
                   for row, original in zip(obs['rows'], raw, strict=True))
        assert sum(row['active_policy_samples'] for row in raw) == spec['active_samples_per_arm']
        assert sum(row['completed_episodes'] for row in raw) == spec['complete_episodes_per_arm']
        assert sum(row['final_hold_samples'] for row in raw) == spec['final_hold_samples_per_arm']
        assert all(np.isfinite([row['environment_reward_mean'], row['policy_loss'], row['value_loss'],
                                row['ppo_seconds']]).all() and row['ppo_seconds'] > 0 for row in raw)
        telemetry = read(record/'gpu-telemetry.json')
        assert not telemetry['failure']
        assert all(row['used_mib']/row['total_mib'] <= spec['memory_fraction_limit'] for row in telemetry['samples'])
        with patch.object(previous, 'check_matched_start', lambda current, _: previous_start(current, zero)), \
             patch.object(previous, 'verify_clips', verify_clips):
            proof = previous.verify(p, out, sha(PLAN), spec['limits'][role])
        results[role] = dict(proof, peak_mib=max(row['used_mib'] for row in telemetry['samples']),
                             ppo_seconds=receipt['ppo_seconds'], wall_seconds=receipt['elapsed_seconds'],
                             mean_net_transitions_per_second=receipt['actual_transitions']/receipt['elapsed_seconds'])
    assert not torch.cuda.is_initialized()
    write(out/'training-verification.json', dict(status='passed-online-training-integrity', protocol_sha256=sha(PLAN),
          script_sha256=sha(Path(__file__)), audit_sha256=sha(AUDIT), arms=results,
          functional_evaluation_pending=True, heldout_used=False))


previous_start = previous.check_matched_start

if __name__ == '__main__':
    main()

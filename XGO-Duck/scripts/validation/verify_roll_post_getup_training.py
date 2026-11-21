"""Verify the two learned checkpoints and export only their final actors."""

import csv
import json
import math
from statistics import median

import imageio_ffmpeg
import numpy as np
import torch

from check_roll_friction_paired_reset import arrays
from export_roll_friction_pair import export
from observed_getup_entry import finalize_view
from record_io import write
from run_roll_post_getup_training import PLAN, PROJECT, REPORT, ROOT, ROLES, checked, read, rel, sha
from verify_roll_exposure_parent import finite


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def verify_role(role, plan, reset, parent):
    arm, recipe = plan['arms'][role], plan['recipe']
    n, total, stride = arm['environments'], arm['updates'], recipe['rollout_controls']
    warm = recipe['critic_only_updates']
    folder = REPORT / role
    run = ROOT / 'records/experiments' / arm['run_id']
    entry = read(folder / 'training-receipt.json')
    online = read(run / 'online-result.json', entry['result_sha256'])
    assert entry['status'] == online['status'] == 'collected-review-required'
    assert ROOT / entry['result'] == run / 'online-result.json' and not entry['authorization_monitor']['failure']
    assert read(run / 'protocol-arm.json')['protocol_sha256'] == sha(PLAN)
    zero = read(run / 'zero-step.json', online['zero_step_sha256'])
    assert zero['controls'] == zero['physics_substeps'] == zero['learning_updates'] == 0
    draft_reset = folder / 'zero' / 'startup-inputs.npz'
    assert sha(draft_reset) == reset['startup_sha256_by_role'][role]
    actual = run / 'startup-inputs.npz'
    assert sha(actual) == zero['startup_inputs_sha256']
    actual_arrays, admitted_arrays = arrays(actual), arrays(draft_reset)
    assert set(actual_arrays) == set(admitted_arrays)
    for key, value in actual_arrays.items():
        np.testing.assert_array_equal(value, admitted_arrays[key], err_msg=key)

    observed = read(run / 'observation.json')
    metrics = run / 'learning-metrics.jsonl'
    assert sha(metrics) == online['metrics_sha256']
    rows, drift = jsonl(metrics), jsonl(run / 'policy-drift.jsonl')
    assert observed['rows'] == rows and len(rows) == len(drift) == total
    for i, (row, probe) in enumerate(zip(rows, drift, strict=True), 1):
        assert row['update'] == probe['update'] == i and row['control'] == probe['control'] == i * stride
        assert row['phase'] == ('critic-only' if i <= warm else 'full-ppo') and probe['within_limits']
        assert all(math.isfinite(v) for v in row['loss'].values())
        assert math.isclose(row['net_transitions_per_second'], stride * n / row['measured_batch_wall_seconds'])
    late = [read(run / f'late-roll-control-{i}.json') for i in plan['training_checks']['late_control_steps']]
    assert all(item['passed'] for item in late)
    assert online['controls'] == observed['controls'] == stride * total
    assert online['completed_updates'] == observed['completed_updates'] == total
    assert online['transitions'] == arm['transitions'] == n * stride * total and online['physics_substeps'] == 5 * stride * total
    assert read(run / 'critic-only.json')['status'] == 'actor-byte-unchanged'
    checkpoint = run / online['checkpoint']
    assert sha(checkpoint) == online['checkpoint_sha256']
    trained = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert finite(trained) and trained['iter'] + 1 == total
    assert trained['infos']['env_state']['common_step_counter'] == total * stride
    assert set(trained['actor_state_dict']) == set(parent['actor_state_dict'])
    assert {int(v['step']) for v in trained['optimizer_state_dict']['state'].values()} == {20 * total, 20 * (total - warm)}
    for i in arm['extra_checkpoints']:
        saved = torch.load(run / f'checkpoint-update-{i}.pt', map_location='cpu', weights_only=True)
        assert saved['iter'] + 1 == i and saved['infos']['env_state']['common_step_counter'] == i * stride and finite(saved)

    clips = []
    for item in read(run / 'clips.json'):
        path = run / item['file']
        frames = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
        info = next(frames)
        count = sum(1 for _ in frames)
        assert info['fps'] == item['fps'] == 25 and count == item['frames'] and not item['extra_physics_steps']
        if item['full_action']:
            assert count == 176 and item['terminal_before_reset'] and item['complete_clip']
        clips.append(dict(file=rel(path), sha256=sha(path), frames=count, full_action=item['full_action']))
    assert any(clip['full_action'] for clip in clips)
    final = read(folder / 'final-view.json') if (folder / 'final-view.json').exists() else finalize_view(folder, arm['run_id'], entry)
    assert final['status'] == 'saved' and final['source_sha256'] == sha(run / 'observation.json')
    with (folder / 'final-view/native-metrics.csv').open(encoding='utf-8', newline='') as file:
        assert len(list(csv.DictReader(file))) == total

    rate = online['transitions'] / online['measured_wall_seconds']
    assert math.isclose(rate, online['measured_transitions_per_second'])
    device = online['device']
    assert all(row['used_mib'] < row['total_mib'] * recipe['hard_gpu_fraction'] for row in device['rows'])
    destination = ROOT / 'artifacts' / arm['run_id']
    destination.mkdir(exist_ok=True)
    model = destination / 'policy.onnx'
    temporary = destination / 'policy.partial.onnx'
    assert not model.exists()
    proof = export(plan, arm, checkpoint, temporary)
    temporary.replace(model)
    assert sha(model) == proof['model_sha256']
    return dict(role=role, run_id=arm['run_id'], transitions=online['transitions'], updates=total, actor_updates=total-warm,
        net_transitions_per_second=rate, actor_phase_median_rate=median(r['net_transitions_per_second'] for r in rows[warm:]),
        gpu_median_percent=device['median_utilization'], peak_mib=device['peak_used_mib'], wall_seconds=online['measured_wall_seconds'],
        late_stable_fractions=[item['qualified_stable_fraction'] for item in late], max_action_probe_rms_rad=max(r['rms_rad'] for r in drift),
        clips=clips, checkpoint=rel(checkpoint), checkpoint_sha256=sha(checkpoint), model=rel(model),
        model_sha256=proof['model_sha256'], export=proof, final_view=rel(folder / 'final-view/native-status.txt'))


def main():
    project, task, state, plan = checked()
    assert state['status'] == 'trained-review-required'
    assert not (task['learning_running'] or task['simulation_running'] or torch.cuda.is_initialized())
    torch.set_num_threads(2)
    reset = read(ROOT / state['matched_zero'], state['matched_zero_sha256'])
    parent = torch.load(ROOT / plan['recipe']['parent_checkpoint'], map_location='cpu', weights_only=True)
    results = {}
    for role in ROLES:
        record = REPORT / role / 'training-verification.json'
        if record.exists():
            results[role] = read(record)
            assert sha(ROOT / results[role]['model']) == results[role]['model_sha256']
            assert sha(ROOT / results[role]['checkpoint']) == results[role]['checkpoint_sha256']
            continue
        results[role] = verify_role(role, plan, reset, parent)
        write(record, results[role])
    assert not torch.cuda.is_initialized()
    out = REPORT / 'paired-training-results.json'
    write(out, dict(protocol_sha256=sha(PLAN), matched_reset_sha256=state['matched_zero_sha256'], results=results,
        new_physical_steps=0, learning_updates_in_verifier=0, action_evaluation_pending=True, hardware_io=False))
    state.update(status='action-evaluation-ready', training_results=rel(out), training_results_sha256=sha(out))
    task.update(current_protocol_execution='post-getup-paired-ppo-verified', current_evidence=rel(out))
    write(PROJECT, project)
    print('ROLL_POST_GETUP_TRAINING_VERIFIED', sha(out), flush=True)


if __name__ == '__main__':
    main()

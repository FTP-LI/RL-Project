"""Verify the restored turn map using saved inputs, with no physics or fitting."""

from pathlib import Path

import numpy as np
import torch

from routed_getup_sequence import FORWARD, STAND, ForwardFeedback
from routed_seven_command_controller import SevenCommandSequenceController
from routed_seven_turn_feedback_controller import SevenCommandTurnFeedbackController
from run_routed_seven_command import OUT, PLAN, ROOT, read, sha, write
from walking_command_controller import create_controller


def main():
    result_path = OUT/'turn-feedback-cpu.json'
    assert not result_path.exists() and not torch.cuda.is_initialized()
    spec, result = read(PLAN), read(OUT/'analysis.json')
    seven = spec['seven_command']
    record = ROOT/spec['record']
    state = read(record/'evaluation.json')
    assert result['protocol_sha256'] == state['protocol_sha256'] == sha(PLAN) and not result['heldout_used']
    delivery = ROOT/read(ROOT/'configs/experiments/continuous_routed_seven_command_draft.json')['walking_delivery'][0]
    frozen_walk = ROOT/read(delivery)['frozen_protocol']['path']
    walking_case, = read(frozen_walk)['cases'].values()
    assert walking_case['environment']['controller_module'] == 'walking_command_controller'
    for name in ('walking_command_controller.py', 'gyro_turn_controller.py', 'command_input_mapping.py'):
        path = f'scripts/validation/{name}'
        assert walking_case['environment']['source_sha256'][path] == spec['source_sha256'][path] == sha(ROOT/path)
    names = list(seven['commands'])
    reached = [min(set(ids)-set(result['score']['commands'][name]['missed_handoff']))
               for name, ids in seven['groups'].items()]
    skipped = result['score']['commands']['turn-left']['missed_handoff'][0]
    start = spec['sequence']['boundary_controls'][-1]
    chunk, = [row for row in state['record_chunks'] if row['first'] == start+1]
    path = record/chunk['file']
    assert sha(path) == chunk['sha256']
    with np.load(path, allow_pickle=False) as saved:
        observed = saved['observed'][:, reached+[skipped]].copy()
        policy_input = saved['actor_input'][:, reached+[skipped]].copy()
    assert len(observed) == 200
    w, p, s = spec['walking'], spec['environment'], spec['sequence']
    home = spec['switch_policies']['stand']['home']
    group = {name:[i] for i, name in enumerate(names)}
    group['turn-left'].append(7)
    arguments = dict(commands=seven['commands'], groups=group, routes=seven['routes'], ramp=spec['ramped_forward']['ramp'])
    baseline, restored = [cls(8, 'cpu', s['controller'], [p['model_home'], home, home], p['joint_limits'],
            ForwardFeedback(w, 8), **arguments) for cls in (SevenCommandSequenceController, SevenCommandTurnFeedbackController)]
    turning = [names.index(name) for name in ('turn-left', 'turn-right')]
    nonturning = [i for i in range(8) if i not in turning]
    desired = np.asarray([seven['commands'][name] for name in (*names,'turn-left')], np.float32)
    original = create_controller(8, .02)
    for ctrl in (baseline, restored):
        ctrl.mode.fill_(STAND)
        ctrl.dwell.fill_(ctrl.dwell_steps)
        ctrl.dwell[7] = 0
        ctrl.history[STAND] = torch.linspace(-.08, .08, 8*14).reshape(8, 14)
    changed = np.zeros(2, int)
    maximum = np.zeros(2)
    with torch.inference_mode():
        for k, raw in enumerate(observed):
            sensed = torch.from_numpy(raw)
            before, old_bridge = baseline.advance(start+k, sensed)
            after, new_bridge = restored.advance(start+k, sensed)
            assert restored.mode.tolist() == [FORWARD]*7 + [STAND]
            torch.testing.assert_close(old_bridge, new_bridge, atol=0, rtol=0)
            torch.testing.assert_close(before[nonturning], after[nonturning], atol=0, rtol=0)
            torch.testing.assert_close(before[turning, :50], after[turning, :50], atol=0, rtol=0)
            torch.testing.assert_close(before[turning, 51:], after[turning, 51:], atol=0, rtol=0)
            np.testing.assert_array_equal(before[turning, 50], policy_input[k, turning, 50])
            expected = original.map(desired, raw)[turning, 2]
            np.testing.assert_array_equal(after[turning, 50], expected)
            delta = abs(after[turning, 50].numpy()-before[turning, 50].numpy())
            changed += delta > 0
            maximum = np.maximum(maximum, delta)
    assert (changed > 0).all() and not torch.cuda.is_initialized()
    paths = (PLAN, OUT/'analysis.json', path, frozen_walk, delivery, Path(__file__).resolve(),
             Path(__file__).with_name('routed_seven_turn_feedback_controller.py'))
    write(result_path, dict(status='passed-restored-turn-feedback-cpu-readback', physical_substeps=0, heldout_used=False,
        archive_worlds=reached+[skipped], probe_controls=len(observed), verified_nonturning_routes=5,
        skipped_turn_unchanged=True, exact_original_gyro_map=True, changed_turn_samples=changed.tolist(),
        maximum_input_change_rad_s=maximum.tolist(), source_sha256={item.relative_to(ROOT).as_posix():sha(item) for item in paths}))
    print('ROUTED_TURN_FEEDBACK_CPU', changed.tolist(), maximum.tolist(), flush=True)


if __name__ == '__main__':
    main()

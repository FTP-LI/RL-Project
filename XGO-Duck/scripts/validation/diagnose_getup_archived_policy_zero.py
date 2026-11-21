"""Compare every reset snapshot field from two fresh zero-step CUDA processes."""

from argparse import ArgumentParser
from pathlib import Path
import sys
from unittest.mock import patch

import torch

import run_getup_archived_policy_comparison as trial
from record_io import write


def probe(seed, name):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    plan, evidence, _ = trial.files(seed)
    spec = trial.read(plan)
    trial.check_phase(plan)
    trial.shared.assert_sources(spec)
    record = trial.ROOT / spec['record']
    original = trial.read(record / 'evaluation.json')
    assert original['status'] == 'stopped-review-required' and original['actual']['attempted_substeps'] == 0
    target = evidence / 'reset-field-diagnosis' / name
    target.mkdir(parents=True)
    write(target / 'command.json', dict(argv=sys.argv, source_sha256=trial.sha(Path(__file__)), protocol_sha256=trial.sha(plan)))
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg, _, meta = trial.shared.prepare(spec['environment'])

    class Capture(trial.SceneReleaseSnapshot):
        def __init__(self, env, memory, cap):
            super().__init__(env, memory, cap)
            write(target / 'fields.json', self.expected)
            write(target / 'stable-inputs.json', {key: trial.shared.signature(value)
                for key, value in trial.shared.release_payload(env, memory).items()})

    env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
    try:
        obs, _ = env.reset(seed=seed)
        memory = trial.shared.make_memory(spec['environment'], meta, env.device)
        with patch.object(trial, 'SceneReleaseSnapshot', Capture):
            _, proof = trial.native_zero(env, obs, memory, spec, plan, target)
        assert env.common_step_counter == env._sim_step_counter == 0
        write(target / 'proof.json', proof)
        print('ARCHIVED_POLICY_RESET_FIELD_PROBE', name, proof['zero_readback']['initial']['state_sha256'], flush=True)
    finally:
        env.close()


def compare(seed):
    plan, evidence, _ = trial.files(seed)
    root = evidence / 'reset-field-diagnosis'
    diffs = {}
    for name in ('fields.json', 'stable-inputs.json'):
        a, b = (trial.read(root / role / name) for role in ('a', 'b'))
        assert a.keys() == b.keys()
        diffs[name] = sorted(key for key in a if a[key] != b[key])
    a, b = (trial.read(root / role / 'proof.json') for role in ('a', 'b'))
    write(root / 'analysis.json', dict(status='completed-reset-field-diagnosis', seed=seed, physical_substeps=0,
        protocol_sha256=trial.sha(plan), differences=diffs, import_errors_equal=a['model_max_error'] == b['model_max_error'],
        original_attempt_closed=True))
    print('ARCHIVED_POLICY_RESET_FIELD_DIFFERENCES', diffs, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=int)
    parser.add_argument('mode', choices=('a', 'b', 'compare'))
    args = parser.parse_args()
    if args.mode == 'compare':
        compare(args.seed)
    else:
        probe(args.seed, args.mode)

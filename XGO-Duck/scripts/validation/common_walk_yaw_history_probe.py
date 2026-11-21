"""Match one physical target pulse with and without the policy-history change."""
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import common_walk_yaw_pulse as pulse
import common_walk_forward_phase_screen as rollout
from check_walk_yaw_history_pulse import DRAFT
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_actuator_yaw_pulse import apply_projected_pulse
from walking_forward_yaw_pulse import ForwardYawPulse


PLAN = ROOT/'configs/experiments/common_walk_yaw_history_probe.json'
OUT = ROOT/'records/reports/common-walk-yaw-history-probe'
PAGE = ROOT/'records/walk-yaw-history-live.html'
STAGE = 'S2-forward-yaw-history-isolation'
ROLES = ('reference', 'command-pulse', 'actuator-pulse')


def freeze():
    from mjlab.envs import ManagerBasedRlEnv

    assert not PLAN.exists() and not OUT.exists()
    draft = read(DRAFT)
    proof_path = pulse.OUT/'history-probe-component-preflight.json'
    proof = read(proof_path)
    assert proof['status'] == 'passed-cpu-actuator-pulse-component' and proof['draft_sha256'] == sha(DRAFT)
    assert proof['controller_sha256'] == sha(ROOT/'scripts/validation/walking_actuator_yaw_pulse.py')
    spec = pulse.checked()
    p = spec['environment']
    seed = draft['development_seed']
    assert not any(f'_s{seed}_' in path.name for path in (ROOT/'records/evaluations').iterdir())
    rng = np.random.default_rng(70000+seed)
    n = p['parallel_worlds']
    permutation = rng.permutation(n)
    slot = list(p['commands']).index('forward')*32
    origin = np.flatnonzero(permutation == 0).item()
    permutation[slot], permutation[origin] = permutation[origin], permutation[slot]
    groups = {name: permutation[i*32:(i+1)*32].tolist() for i, name in enumerate(p['commands'])}
    signs = np.zeros(n, dtype=np.int8)
    signs[groups['forward']] = rng.permutation(np.repeat([-1, 1], 16))
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    p.update(seed=seed, assignment_seed=70000+seed, stage=STAGE, groups=groups,
             run_id=f'duck_walk_yaw-history_paired_s{seed}_{stamp}')
    spec['phase']['pulse_signs'] = signs.tolist()
    assert spec['phase']['warmup_controls']+1 == draft['pulse']['control']
    assert spec['phase']['peak_rad_s'] == draft['pulse']['command_rad_s']
    assert spec['maximum_control_transitions'] == draft['maximum_transitions']
    assert spec['maximum_physics_substeps'] == draft['maximum_batched_substeps']
    spec.update(stage=STAGE, roles=list(ROLES), history_probe=draft,
                scope='Three short branches match the applied first target and isolate command-pulse raw history.')
    for path in (pulse.PLAN, DRAFT, proof_path, pulse.OUT/'analysis.json', pulse.OUT/'action-response-review.json'):
        spec['reference_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (Path(__file__), Path(__file__).with_name('verify_walk_yaw_history_probe.py'),
                 Path(__file__).with_name('walking_actuator_yaw_pulse.py')):
        spec['source_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    rollout.check_sources(spec)
    _, desired = rollout.groups_and_commands(p)
    component = pulse.component_probe(p, spec['phase'], desired)
    torch.set_num_threads(4)
    cfg, _ = rollout.make_config(p)
    env = ManagerBasedRlEnv(cfg=cfg, device='cpu', render_mode=None)
    try:
        env.reset(seed=seed)
        env.command_manager.get_command('twist').copy_(torch.from_numpy(desired))
        snapshot = rollout.snapshot_probe(env, spec['maximum_warp_snapshot_bytes'])
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT/'preflight.json', dict(status='passed-cpu-zero-step', protocol_sha256=sha(PLAN),
          snapshot=snapshot, component=component, actuator_preflight_sha256=sha(proof_path),
          physical_substeps=0, learning_updates=0, heldout_used=False))
    print('WALK_YAW_HISTORY_FROZEN', sha(PLAN), p['run_id'], flush=True)


original_collect = rollout.collect


def collect(env, actor, spec, record, state, desired):
    term = env.action_manager.get_term('joint_pos')
    process = term.process_actions
    branch = 'shared-prefix'
    pending = None
    applied = 0
    maximum = spec['history_probe']['pulse']['maximum_applied_target_change_rad']
    pulse_control = spec['history_probe']['pulse']['control']

    def publish_fork(path, value):
        nonlocal branch
        write(path, value)
        # The frozen collector writes this receipt immediately after restoring
        # each branch, before calculating the first post-fork actor input.
        if path == record/'fork-readback.json':
            branch = next(reversed(value))
            assert branch in ROLES and pending is None

    class Controller(ForwardYawPulse):
        def map(self, observed, step, _candidate, active):
            nonlocal pending
            assert pending is None
            command = super().map(observed, step, branch == 'command-pulse', active)
            if branch == 'actuator-pulse' and step == pulse_control:
                alternate = observed.copy()
                alternate[:, 48:51] = command
                enabled = self.forward & active
                alternate[enabled, 50] = self.pulse[enabled]
                raw = actor(rollout.td(torch.from_numpy(alternate).to(env.device)))
                pending = alternate, raw, torch.as_tensor(enabled, device=env.device)
            return command

    def process_once(actions):
        nonlocal pending, applied
        process(actions)
        if pending is not None:
            observed, alternate, enabled = pending
            apply_projected_pulse(term, alternate, enabled, maximum)
            applied += 1
            np.savez_compressed(record/'actuator-pulse-readback.npz', observations=observed,
                                alternate_raw=rollout.cpu(alternate), enabled=rollout.cpu(enabled),
                                raw=rollout.cpu(term._raw_actions), applied=rollout.cpu(term._processed_actions))
            pending = None

    with patch.object(term, 'process_actions', process_once), patch.object(rollout, 'write', publish_fork), \
         patch.object(rollout, 'ForwardPhase', Controller):
        original_collect(env, actor, spec, record, state, desired)
    assert applied == 1 and pending is None


def sample():
    with patch.object(pulse, 'PLAN', PLAN), patch.object(pulse, 'OUT', OUT), \
         patch.object(pulse, 'PAGE', PAGE), patch.object(pulse, 'STAGE', STAGE), \
         patch.object(pulse, 'ROLES', ROLES), patch.object(rollout, 'collect', collect):
        spec = pulse.checked()
        try:
            pulse.sample(spec)
        finally:
            record = ROOT/'records/evaluations'/spec['environment']['run_id']
            readme = record/'README.md'
            if readme.exists():
                text = readme.read_text(encoding='utf-8')
                readme.write_text(text.replace('common_walk_yaw_pulse.json', PLAN.name).replace(
                    'common-walk-yaw-pulse/README.md', 'common-walk-yaw-history-probe/README.md'), encoding='utf-8')


if __name__ == '__main__':
    pulse.check_project()
    assert sys.argv[1:] in (['--freeze'], ['--sample'])
    if sys.argv[1:] == ['--freeze']:
        freeze()
    else:
        sample()

"""Preserve the zero-step failure and inherit the checkpoint's exact scale.

The report independently recomputed a double-precision RMS one ULP away from
the checkpoint. This is a provenance error in the new plan, not a new value
fit. A separately frozen plan reads the original checkpoint scalar exactly;
physics, actor, critic, RNG guards, budgets and accuracy gates stay unchanged.
Do not relax the equality assertion or alter any of the frozen parent files.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse, subprocess, sys
import numpy as np
import torch
import getup_anchor_value_check as base

ROOT = base.ROOT
PARENT_PLAN, PARENT_OUT = base.PLAN, base.OUT
PLAN = ROOT/'configs/experiments/getup_anchor_scale_recovery.json'
OUT = PARENT_OUT/'scale-recovery'
STAGE = 'S3-anchor-value-scale-recovery'


def freeze():
    from common_scene_contract import model_arrays, compare_arrays
    from check_getup_stand_handoff import real_processor
    from evaluate_roll_reset_pair import differences
    from mjlab.scene import Scene
    import yaml
    assert not PLAN.exists() and not OUT.exists()
    assert not torch.cuda.is_initialized()
    p = deepcopy(base.checked())
    failed = base.read(PARENT_OUT/'progress.json')
    record = ROOT/'records/experiments'/p['run_id']
    runtime = base.read(record/'runtime.json')
    assert failed['status'] == 'failed-preserved' and failed['control_steps'] == 0
    assert failed['actor_updates'] == failed['critic_updates'] == 0
    assert runtime['control_transitions'] == runtime['physics_substeps'] == 0
    log = (record/'execution.log').read_text()
    assert "assert warm['reward_scale']==p['reward_scale']" in log
    warm = torch.load(ROOT/p['critic_checkpoint'], map_location='cpu', weights_only=True)
    source, reported = warm['reward_scale'], p['reward_scale']
    assert isinstance(source, float) and isinstance(reported, float) and source > 0
    assert source != reported and abs(source-reported) == abs(np.spacing(source))
    assert np.float32(source) == np.float32(reported)
    before = asdict(base.prepare(p)[0])
    p['reward_scale'] = source
    assert warm['reward_scale'] == p['reward_scale']
    cfg, display, _ = base.prepare(p)
    assert not differences(before, asdict(cfg))
    fields = model_arrays(Scene(cfg.env.scene, device='cpu').compile())
    with np.load(PARENT_OUT/'cpu-scene-fields.npz') as f:
        compare_arrays(fields, {k:f[k] for k in f.files})
    processor = real_processor(cfg.env, p['model_home'], p['standing_home'])
    # Exercise the actual checkpoint serialization and value-network interface
    # on CPU before another live startup; this performs no optimizer updates.
    from getup_value_bootstrap import td
    from rsl_rl.models import MLPModel
    cpu_rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        critic = MLPModel(td('critic',torch.zeros(1,75)),{'critic':['critic']},'critic',1,
            hidden_dims=(512,256,128),activation='elu',obs_normalization=True)
    critic.load_state_dict(warm['critic_state_dict'])
    with torch.no_grad():
        prediction = critic(td('critic', torch.zeros(8,75)))
    assert prediction.shape == (8,1) and torch.isfinite(prediction).all()
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
    proof = dict(status='passed-cpu-only', checkpoint_scale=source,
        checkpoint_scale_hex=source.hex(), report_recomputed_scale=reported,
        report_scale_hex=reported.hex(), difference=source-reported,
        one_double_ulp=True, float32_scale_identical=True,
        checkpoint_scalar_exact=True, environment_config_exact=True,
        original_failure_controls=0, original_failure_transitions=0,
        original_failure_gradients=0, new_controls=400, new_transitions=409600,
        actor_updates=0, critic_updates=0, source_files_unchanged=True,
        existing_assertions_and_tolerances_unchanged=True)
    p['scale_recovery'] = dict(proof,
        parent_protocol=str(PARENT_PLAN.relative_to(ROOT)),
        parent_protocol_sha256=base.sha(PARENT_PLAN),
        parent_record=str(record.relative_to(ROOT)),
        change='Use the exact saved checkpoint scalar, not an independently recomputed report scalar',
        continuation_or_restore=False)
    p.update(stage=STAGE,
        run_id='duck_getup_anchor-scale_reference_s43_'+datetime.now().strftime('%Y%m%dT%H%M%S'),
        frozen_at=datetime.now(timezone.utc).isoformat())
    p['source_sha256'][str(Path(__file__).relative_to(ROOT))] = base.sha(Path(__file__))
    for path in (PARENT_PLAN, record/'execution.log', record/'runtime.json', record/'state.json'):
        p['reference_sha256'][str(path.relative_to(ROOT))] = base.sha(path)
    OUT.mkdir()
    np.savez_compressed(OUT/'cpu-scene-fields.npz', **fields)
    base.write(PLAN,p)
    pre = base.read(PARENT_OUT/'preflight.json')
    pre.update(proof, protocol_sha256=base.sha(PLAN), source_count=len(p['source_sha256']),
        processor=processor, config=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader),
        parent_scale_unchanged=source, display=display)
    base.write(OUT/'preflight.json',pre)
    base.write(PARENT_OUT/'failure-analysis.json',dict(proof,
        failure_record=str(record.relative_to(ROOT)),
        recovery_protocol=str(PLAN.relative_to(ROOT))))
    print('ANCHOR_SCALE_RECOVERY_FROZEN',base.sha(PLAN),len(p['source_sha256']),
          'sources;checkpoint scale exact;0 gradients',flush=True)


def activate():
    base.PLAN, base.OUT, base.STAGE = PLAN, OUT, STAGE


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.freeze:
        freeze()
    else:
        activate()
        if args.verify:
            from verify_getup_anchor_value import verify
            verify()
        else:
            base.run()
            subprocess.run([sys.executable,'-B',str(Path(__file__)),'--verify'],cwd=base.WORK,check=True)

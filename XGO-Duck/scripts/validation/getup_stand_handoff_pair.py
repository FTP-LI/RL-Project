"""Fresh bounded comparison: common interface alone vs getup -> standing.

Both arms share prospective walking-level observation noise and the same
post5s projected/slew-limited target processor. Only the standing-policy latch
is enabled in the candidate. The closed neck-relief rule is not reused.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import argparse, json, os, subprocess, sys
import numpy as np
import torch
import onnx
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/training'))
from s1_reference_inputs import WORK, read, sha
from record_io import write
from getup_release_coverage import prepare as original_prepare, actor_model, cpu_check
from getup_stand_handoff import apply_common_actor_noise, adapt_action, SETTINGS, NOISE

PLAN = ROOT/'configs/experiments/getup_stand_handoff_pair.json'
OUT = ROOT/'records/reports/getup-stand-handoff/fixed-evaluation'


def prepare(p):
    cfg, display, meta = original_prepare(p)
    cfg = replace(cfg,env=adapt_action(apply_common_actor_noise(cfg.env), p['standing_home'],
                                      p['joint_limits'], p['handoff_settings']))
    display.update(recorder='Four preselected complete11s continuous getup/stand episodes',
                   native_live_window=False)
    return cfg, display, meta


def freeze():
    from mjlab.scene import Scene
    from common_scene_contract import model_arrays, compare_arrays
    from getup_neck_relief_pair import checked as previous_checked
    from check_getup_stand_handoff import real_processor, check_slew
    from evaluate_roll_reset_pair import differences
    assert not PLAN.exists() and not OUT.exists(), 'Preserve every frozen attempt'
    previous = previous_checked()
    proof = read(ROOT/'records/reports/getup-stand-handoff/component-preflight.json')
    assert proof['status'] == 'passed-cpu-only'
    for name, digest in proof['source_files'].items():
        assert sha(ROOT/name) == digest, name
    walking = read(ROOT/'configs/experiments/common_plant/walk.json')
    assert sha(ROOT/walking['model']) == walking['model_sha256']
    smeta = {x.key:x.value for x in onnx.load(ROOT/walking['model']).metadata_props}
    seed = 1014
    codes = np.random.default_rng(seed).permutation(np.repeat(np.arange(4), [32,32,128,128]))
    names = ('prone','supine','left-side','right-side')
    groups = {name:np.flatnonzero(codes==i).tolist() for i,name in enumerate(names)}
    old = previous['arms']['control']
    required = ('model','model_sha256','task','num_envs','simulation_home','episode_seconds',
                'maximum_control_steps','maximum_control_transitions','initial_std',
                'training_source_sha256','library_source_sha256','criteria','success_rate_min',
                'actual_release_minimum','applied_joint_limit_tolerance_deg')
    template = {k:deepcopy(old[k]) for k in required}
    getup_meta = {x.key:x.value for x in onnx.load(ROOT/template['model']).metadata_props}
    template.update(status='frozen-before-dynamics', stage='S3-continuous-getup-stand', seed=seed,
        split=dict(pose_codes=codes.tolist(),groups=groups),
        standing_model=walking['model'], standing_model_sha256=walking['model_sha256'],
        standing_home=list(map(float,smeta['default_joint_pos'].split(','))),
        model_home=list(map(float,getup_meta['default_joint_pos'].split(','))),
        joint_limits=walking['joint_limits'], handoff_settings=deepcopy(SETTINGS),
        common_actor_noise=deepcopy(NOISE),learning_updates=0,actor_frozen=True,
        video_worlds={name:groups[name][0] for name in names})
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    arms = {}
    for role in ('control','candidate'):
        p = deepcopy(template)
        p.update(arm=role, handoff_enabled=role=='candidate',
                 run_id=f'duck_getup_stand-handoff_{role}_s{seed}_{stamp}')
        arms[role] = p
    cfg, display, meta = prepare(arms['control'])
    other = prepare(arms['candidate'])[0]
    assert not differences(asdict(cfg),asdict(other)), 'Both arms must use the same environment config'
    assert meta['joint_names'] == smeta['joint_names']
    models = model_arrays(Scene(cfg.env.scene,device='cpu').compile())
    with np.load(ROOT/'records/reports/getup-value-bootstrap/cpu-scene-fields.npz') as f:
        compare_arrays(models,{k:f[k] for k in f.files})
    source_home = list(map(float, meta['default_joint_pos'].split(',')))
    fixture = real_processor(cfg.env, source_home, template['standing_home'])
    numeric = cpu_check(dict(roles={'getup':template,
        'standing':dict(template,model=template['standing_model'],model_sha256=template['standing_model_sha256'])}))
    sources = deepcopy(previous['source_sha256'])
    for name in ('scripts/training/getup_stand_handoff.py',
                 'scripts/validation/check_getup_stand_handoff.py',
                 'scripts/validation/getup_stand_handoff_pair.py',
                 'scripts/validation/collect_getup_stand_handoff.py',
                 'scripts/validation/verify_getup_stand_handoff.py'):
        sources[name] = sha(ROOT/name)
    master = dict(status='frozen-before-dynamics',arms=arms,source_sha256=sources,stamp=stamp,
        frozen_at=datetime.now(timezone.utc).isoformat(),maximum_episodes=640,
        maximum_evaluation_transitions=352000,evaluation_seed=seed,learning_updates=0,
        heldout_used=False,hardware_io=False,automatic_retry=False,automatic_extension=False,
        component_preflight_sha256=sha(ROOT/'records/reports/getup-stand-handoff/component-preflight.json'),
        common_adaptation='Prospective walking-level actor noise; same raw-to-model HOME processor and post5s projected2rad/s target transition in both arms. Old out-of-range requests may converge gradually; no instantaneous bound claim.',
        only_difference='Candidate enables a time/observed-tilt one-way latch to the fixed standing model; both compute both actors and the same shadow trigger. No physical reset, actor fitting, neck-relief rule or new sensor.',
        candidate_start_gate='Skip if common-interface getup control already passes. Skip if actual pre2s release coverage is insufficient or unknown; no post5s rule can repair that coverage.',
        continue_gate='All requested groups, actual release groups and overall>=95percent timely recovery with each actual release class>=32 and zero unknowns; candidate overall successes>=control and at least one real handoff. Development lead only.',
        stop_rule='Nonfinite/interface/contact capacity/pause stops and preserves. Complete the declared first11s episodes; no replacement, retry or extension.',
        scope='One fresh development seed: continuous getup-to-zero-command-stand only, not full S5 walking/roll/robustness or hardware acceptance.')
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN,master)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
        identical_environment_configs=True,cpu_scene_fields_exact=True,
        real_processor=fixture,slew=check_slew(),model_import=numeric,
        requested_counts={k:len(v) for k,v in groups.items()},video_worlds=template['video_worlds'],
        config=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader),
        display=display,cuda_initialized=False,physics_steps=0))
    print('GETUP_STAND_HANDOFF_FROZEN',sha(PLAN),flush=True)


def checked():
    m = read(PLAN)
    assert sha(PLAN) == read(OUT/'preflight.json')['protocol_sha256']
    for name, digest in m['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    for p in m['arms'].values():
        assert sha(ROOT/p['model']) == p['model_sha256']
        assert sha(ROOT/p['standing_model']) == p['standing_model_sha256']
        for name, digest in p['training_source_sha256'].items():
            assert sha(WORK/name) == digest, name
        for name, digest in p['library_source_sha256'].items():
            assert sha(Path(name)) == digest, name
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--role',choices=('control','candidate'))
    a = parser.parse_args()
    if a.freeze:
        freeze(); return
    assert os.isatty(1), 'Use the original single visible VS Code terminal'
    m = checked()
    if a.role:
        from collect_getup_stand_handoff import collect
        collect(a.role,m); return
    assert not (OUT/'coordinator.json').exists(), 'No repeated evaluation'
    write(OUT/'coordinator.json',dict(status='running',pid=os.getpid()))
    try:
        for role in ('control','candidate'):
            if role == 'candidate':
                control = read(OUT/'control-analysis.json')
                if control['functional_coverage_gate']:
                    write(OUT/'candidate-skipped.json',dict(reason='Simpler common-interface control passes',new_physics_steps=0)); break
                if control['other_release_episodes'] or not all(x['coverage_passed'] for x in control['actual'].values()):
                    write(OUT/'candidate-skipped.json',dict(reason='Inadequate pre2s actual release coverage',new_physics_steps=0)); break
            subprocess.run([sys.executable,'-B',str(Path(__file__)),'--role',role],cwd=WORK,check=True)
            subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validation/verify_getup_stand_handoff.py'),'--role',role],cwd=WORK,check=True)
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validation/verify_getup_stand_handoff.py')],cwd=WORK,check=True)
        write(OUT/'coordinator.json',dict(status='completed-review-required',pid=os.getpid()))
    except BaseException as error:
        write(OUT/'coordinator.json',dict(status='stopped-review-required',pid=os.getpid(),error=repr(error)))
        raise


if __name__=='__main__':
    main()

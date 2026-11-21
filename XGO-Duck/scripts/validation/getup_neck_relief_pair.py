"""One bounded fixed-official-policy comparison of late neck-target relief.

No policy/critic fitting. The candidate changes only the enabled flag of the
new processed-target rule. Keep old studies closed; use a fresh development
seed, all first episodes, full contact, and complete eleven-second recordings.
"""
from copy import deepcopy
from dataclasses import asdict,fields
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,subprocess,sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import WORK,read,sha
from record_io import write
from getup_release_coverage import prepare as original_prepare,actor_model,cpu_check
from getup_neck_relief import NeckReliefActionCfg,SETTINGS
from fixed_getup_collection import collect

PLAN=ROOT/'configs/experiments/getup_neck_relief_pair.json'
OUT=ROOT/'records/reports/getup-neck-relief'


def prepare(p):
    cfg,display,meta=original_prepare(p);old=cfg.env.actions['joint_pos']
    values={f.name:deepcopy(getattr(old,f.name)) for f in fields(old) if f.init}
    cfg.env.actions['joint_pos']=NeckReliefActionCfg(**values,relief_enabled=p['arm']=='candidate',
        neck_index=meta['joint_names'].split(',').index('neck_pitch'),relief_settings=deepcopy(p['relief_settings']))
    display.update(recorder='Complete11s fixed-policy neck-relief comparison; four preselected worlds',native_live_window=False)
    return cfg,display,meta


def freeze():
    from mjlab.scene import Scene
    from common_scene_contract import model_arrays,compare_arrays
    from evaluate_roll_reset_pair import differences
    from check_getup_neck_relief import check
    assert not PLAN.exists() and not OUT.exists(),'Preserve existing frozen attempt'
    diagnosis=read(ROOT/'records/reports/getup-release-coverage/neck-clearance/analysis.json')
    admissible=[r['lower_bound_deg'] for r in diagnosis['sweep_summary'] if r['remaining_head_leg_cases']==r['new_self_contact_cases']==r['head_floor_penetration_over2mm']==0]
    assert admissible[0]==SETTINGS['lower_bound_deg']==-75
    old=read(ROOT/'configs/experiments/getup_release_coverage.json');p=deepcopy(old['roles']['original'])
    codes=np.random.default_rng(1013).permutation(np.repeat(np.arange(4),[32,32,128,128]))
    names=('prone','supine','left-side','right-side');groups={k:np.flatnonzero(codes==i).tolist() for i,k in enumerate(names)}
    p.update(seed=1013,split=dict(pose_codes=codes.tolist(),groups=groups),relief_settings=deepcopy(SETTINGS),
        video_worlds={pose:groups[pose][0] for pose in names})
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');arms={};configs={};models={}
    for role in ('control','candidate'):
        a=deepcopy(p);a.update(arm=role,run_id=f'duck_getup_neck-relief_{role}_s1013_{stamp}')
        cfg,display,meta=prepare(a);arms[role]=a;configs[role]=asdict(cfg);models[role]=Scene(cfg.env.scene,device='cpu').compile()
    changes=differences(configs['control'],configs['candidate'])
    assert len(changes)==1 and changes[0]['path']=='.env.actions.joint_pos.relief_enabled',changes
    compare_arrays(model_arrays(models['control']),model_arrays(models['candidate']))
    with np.load(ROOT/'records/reports/getup-value-bootstrap/cpu-scene-fields.npz') as f:compare_arrays(model_arrays(models['control']),{k:f[k] for k in f.files})
    fixture=check(arms['candidate'],prepare(arms['candidate'])[0],meta)
    imports=cpu_check(dict(roles=arms))
    sources=deepcopy(old['source_sha256'])
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name
    for name in ('scripts/training/getup_neck_relief.py','scripts/validation/getup_neck_relief_pair.py',
        'scripts/validation/fixed_getup_collection.py','scripts/validation/check_getup_neck_relief.py',
        'scripts/validation/verify_getup_neck_relief.py','scripts/validation/review_getup_neck_clearance.py'):
        sources[name]=sha(ROOT/name)
    master=dict(status='frozen-before-data',arms=arms,source_sha256=sources,stamp=stamp,
        frozen_at=datetime.now(timezone.utc).isoformat(),maximum_evaluation_transitions=352000,maximum_episodes=640,
        evaluation_seed=1013,learning_updates=0,heldout_used=False,hardware_io=False,automatic_retry=False,automatic_extension=False,
        geometry_evidence_sha256=sha(ROOT/'records/reports/getup-release-coverage/neck-clearance/analysis.json'),
        only_difference='Enable late processed-neck correction; actor, raw history, other targets, resets, noise, physical parameters and collisions unchanged',
        continue_gate='Candidate must pass all95percent requested/actual/overall functional gates with>=32actual releases per class and zero unknowns, plus overall successes>=control. Development lead only; independent confirmation still required.',
        candidate_start_gate='Skip if unchanged official control already passes. Stop if control actual release coverage inadequate or unknown releases: post5s intervention cannot solve pre2s coverage.',
        stop_rule='Nonfinite/interface/contact capacity/pause stops and preserves; no early resets or automatic extension',
        scope='One fresh development comparison of an explicitly new lightweight controller rule; zero weight learning, no hardware or continuous-switch claim')
    assert not torch.cuda.is_initialized();OUT.mkdir();write(PLAN,master)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),configuration_differences=changes,
        cpu_scene_exact=True,fixture=fixture,actor_import=imports,geometry_selected_lower_bound=-75,
        configs={k:yaml.load(yaml.dump(v),Loader=yaml.BaseLoader) for k,v in configs.items()},cuda_initialized=False))
    print('GETUP_NECK_RELIEF_FROZEN',sha(PLAN),flush=True)


def checked():
    m=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for p in m['arms'].values():
        assert sha(ROOT/p['model'])==p['model_sha256']
        for name,digest in p['training_source_sha256'].items():assert sha(WORK/name)==digest,name
        for name,digest in p['library_source_sha256'].items():assert sha(Path(name))==digest,name
    return m


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--role',choices=('control','candidate'));a=parser.parse_args()
    if a.freeze:freeze();return
    assert os.isatty(1),'Use the original one visible VS Code terminal'
    if a.role:collect(a.role,checked(),PLAN,OUT,prepare,actor_model);return
    checked();assert not (OUT/'coordinator.json').exists(),'No repeat'
    write(OUT/'coordinator.json',dict(status='running',pid=os.getpid()))
    try:
        for role in ('control','candidate'):
            if role=='candidate':
                control=read(OUT/'control-analysis.json')
                if control['functional_coverage_gate']:
                    write(OUT/'candidate-skipped.json',dict(reason='Unchanged official passes; prefer simpler controller',new_physics_steps=0));break
                if control['other_release_episodes'] or not all(r['coverage_passed'] for r in control['actual'].values()):
                    write(OUT/'candidate-skipped.json',dict(reason='Post5s correction cannot repair inadequate pre2s release coverage',new_physics_steps=0));break
            subprocess.run([sys.executable,'-B',str(Path(__file__)),'--role',role],cwd=WORK,check=True)
            subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validation/verify_getup_neck_relief.py'),'--role',role],cwd=WORK,check=True)
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validation/verify_getup_neck_relief.py')],cwd=WORK,check=True)
        write(OUT/'coordinator.json',dict(status='completed-review-required',pid=os.getpid()))
    except BaseException as error:
        write(OUT/'coordinator.json',dict(status='stopped-review-required',pid=os.getpid(),error=repr(error)));raise


if __name__=='__main__':main()

"""Freeze, visibly run and independently review one bounded getup reward pair.

Control/candidate differ only in the late head-leg cost within the declared
common learning adaptation. Three fixed models are evaluated on fresh dev
seed1011 with unchanged recovery/hold criteria. No heldout or hardware use.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import numpy as np
import torch
import yaml
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
from getup_contact_pilot import prepare,kinematic_stable
from check_getup_masked_ppo import settings
from common_scene_contract import model_arrays,compare_arrays,install_guard
from evaluate_roll_reset_pair import differences

PLAN=ROOT/'configs/experiments/getup_late_contact_pair.json'
OUT=ROOT/'records/reports/getup-late-contact'


def freeze():
    import mujoco,onnx
    from mjlab.scene import Scene
    assert not PLAN.exists() and not OUT.exists()
    proof=read(ROOT/'records/reports/getup-adaptation/masked-optimizer-preflight.json');assert proof['status']=='passed'
    admission=read(ROOT/'records/reports/getup-value-bootstrap/startup-recovery/analysis.json');assert admission['value_gate_passed']
    record=ROOT/admission['record'];warm=read(record/'protocol.json');draft=read(ROOT/'configs/experiments/getup_collision_adaptation_draft.json')
    p=deepcopy(warm);p.update(stage='S3-getup-late-contact-pilot',status='frozen-before-learning',
        critic_checkpoint=str((record/'critic-final.pt').relative_to(ROOT)),critic_sha256=sha(record/'critic-final.pt'),
        probe_dataset=str((record/'value-data.npz').relative_to(ROOT)),probe_sha256=sha(record/'value-data.npz'),
        reward_scale=admission['reward_scale'],learner=settings(),training_updates=50,control_steps=1200,
        maximum_control_transitions=1228800,exploration='Fixed0.05, no distribution optimizer',
        common_adaptation='Finite75dim critic warm start and Adam, fixed normalizers/positive reward scaling; masked actor loss; fresh separate actor Adam',
        scope='One matched development reward pilot; no heldout, continuous transition or hardware claim')
    sources=deepcopy(warm['source_sha256'])
    for name in ('scripts/training/getup_contact_pilot.py','scripts/training/getup_masked_ppo.py',
        'scripts/training/getup_ppo_math.py','scripts/training/action_video.py','scripts/training/recover_getup_value_startup.py',
        'scripts/validation/check_getup_masked_ppo.py','scripts/validation/verify_getup_contact_pilot.py',
        'scripts/validation/getup_contact_pair.py','configs/environments/action_recording.json'):
        sources[name]=sha(ROOT/name)
    for name,digest in warm['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in warm['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    p['source_sha256']=sources
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');arms={};configs={};models={}
    for arm in ('control','candidate'):
        a=deepcopy(p);a.update(arm=arm,run_id=f'duck_getup_late-contact_{arm}_s42_{stamp}',matching_control=f'duck_getup_late-contact_control_s42_{stamp}')
        cfg,display,meta=prepare(a);configs[arm]=asdict(cfg);models[arm]=Scene(cfg.env.scene,device='cpu').compile();arms[arm]=a
    changes=differences(configs['control'],configs['candidate'])
    assert len(changes)==1 and changes[0]['path']=='.env.rewards.late_head_leg_contact.params.enabled',changes
    compare_arrays(model_arrays(models['control']),model_arrays(models['candidate']))
    model=models['control'];metadata={x.key:x.value for x in onnx.load(ROOT/p['model']).metadata_props}
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+n) for n in metadata['joint_names'].split(',')]
    addresses=model.jnt_qposadr[ids].tolist();ranges=model.jnt_range[ids].tolist()
    with np.load(record/'value-data.npz') as f:
        stable=kinematic_stable(torch.from_numpy(f['qpos']),torch.from_numpy(f['qvel']),addresses,torch.tensor(ranges))
        reference=float(stable[300:].float().mean())
    for a in arms.values():a.update(qpos_addresses=addresses,joint_ranges=ranges,
        watch_reference_stable_fraction=reference,watch_stable_min=max(.3,reference-.25),
        watch_rule='All-world kinematic stable fraction below the32world fixed-actor late reference minus0.25 (floor0.30) for25consecutive controls at episode age>=7s; diagnostic only, not gait acceptance')
    assert not torch.cuda.is_initialized()
    OUT.mkdir();(OUT/'control').mkdir();(OUT/'candidate').mkdir()
    evaluation=deepcopy(read(ROOT/'configs/experiments/common_plant/getup.json'))
    evaluation.update(seed=1011,poses=['prone','supine'],source_sha256=sources,heldout_used=False,
        scope='Fresh common-full-contact development comparison with unchanged recovery and3s hold criteria')
    master=dict(status='frozen-before-learning',arms=arms,stamp=stamp,source_sha256=sources,
        bootstrap_admission_sha256=sha(ROOT/'records/reports/getup-value-bootstrap/startup-recovery/analysis.json'),
        draft_sha256=sha(ROOT/'configs/experiments/getup_collision_adaptation_draft.json'),
        source_getup_model_sha256=p['model_sha256'],training_seed=42,evaluation_seed=1011,
        maximum_learning_transitions=2457600,maximum_evaluation_transitions=105600,evaluation_template=evaluation,
        continue_gate=draft['proposed_fixed_evaluation']['continue_condition'],
        control_line_gate='Control prone/supine>=31/32 and supine>=unchanged official+2; possible common-adaptation lead only, not contact-cost improvement',
        automatic_retry=False,automatic_extension=False,heldout_used=False,hardware_io=False)
    write(PLAN,master);write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),configuration_differences=changes,
        configs={k:yaml.load(yaml.dump(v),Loader=yaml.BaseLoader) for k,v in configs.items()},
        cpu_physical_fields_exact=True,watch_reference_stable_fraction=reference,watch_min=max(.3,reference-.25),
        cuda_initialized=False,maximum_learning_transitions=2457600,maximum_evaluation_transitions=105600))
    print('GETUP_CONTACT_PAIR_FROZEN',sha(PLAN),'config_changes=',changes,flush=True)


def checked():
    p=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for arm in p['arms'].values():
        assert sha(ROOT/arm['critic_checkpoint'])==arm['critic_sha256']
        assert sha(ROOT/arm['probe_dataset'])==arm['probe_sha256']
        for name,digest in arm['training_source_sha256'].items():assert sha(WORK/name)==digest,name
        for name,digest in arm['library_source_sha256'].items():assert sha(Path(name))==digest,name
    return p


def evaluate(role):
    import evaluate_standup_reference as collector
    import common_plant_screen
    from PIL import ImageDraw
    from verify_standup_reference import main as verify
    import verify_standup_reference as verifier
    m=checked();p=deepcopy(m['evaluation_template'])
    if role!='original':
        path=ROOT/'artifacts'/m['arms'][role]['run_id']/'policy.onnx'
        assert sha(path)==read(path.parent/'manifest.json')['model_sha256']
        p.update(model=str(path.relative_to(ROOT)),model_sha256=sha(path))
    out=OUT/f'eval-{role}';out.mkdir();assert not (out/'progress.json').exists()
    run_id=f'duck_getup_late-contact-eval_{role}_s1011_'+m['stamp'];folder=ROOT/'records/evaluations'/run_id;folder.mkdir();(folder/'videos').mkdir()
    write(folder/'protocol.json',p)
    for name in p['source_sha256']:shutil.copy2(ROOT/name,folder/Path(name).name)
    state=dict(status='running',run_id=run_id,path=str(folder),pid=os.getpid(),conditions=[],model=str(ROOT/p['model']),
        model_sha256=p['model_sha256'],protocol_sha256=sha(folder/'protocol.json'),learning_updates=0,
        scope=p['scope'],started_at=datetime.now(timezone.utc).isoformat())
    write(out/'progress.json',state);write(folder/'evaluation.json',state)
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=run_id,pid=os.getpid()))
    collector.MODEL=ROOT/p['model'];collector.OUT=out;collector.config=lambda plan:common_plant_screen.prepare('getup',plan)
    install_guard(p,folder,ROOT/'records/reports/common-plant-screen/startup','rollout')
    text=ImageDraw.ImageDraw.text
    def labeled(self,xy,value,*args,**kwargs):
        if isinstance(value,str):value=value.replace('OFFICIAL GETUP','GETUP '+role.upper(),1)
        return text(self,xy,value,*args,**kwargs)
    ImageDraw.ImageDraw.text=labeled
    with (folder/'execution.log').open('x',buffering=1) as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=collector.Tee(stdout,log),collector.Tee(stderr,log)
        try:collector.evaluate(p,folder,state);state['status']='completed'
        except BaseException as error:state.update(status='failed',error=repr(error));raise
        finally:
            ImageDraw.ImageDraw.text=text;state['ended_at']=datetime.now(timezone.utc).isoformat()
            write(out/'progress.json',state);write(folder/'evaluation.json',state);sys.stdout,sys.stderr=stdout,stderr
    verifier.OUT=out;verify()


def analyze():
    m=checked();r={k:read(OUT/f'eval-{k}/analysis.json') for k in ('original','control','candidate')}
    counts={k:{c['condition']:c['successes'] for c in v['conditions']} for k,v in r.items()}
    c,n=counts['control'],counts['candidate'];admit=n['supine']>=31 and n['prone']>=31 and n['supine']>=c['supine']+2
    common=c['prone']>=31 and c['supine']>=31 and c['supine']>=counts['original']['supine']+2
    result=dict(status='passed',counts=counts,candidate_continue=admit,common_adaptation_lead=common,
        candidate_closed=not admit,actual_learning_transitions=2457600,actual_evaluation_transitions=105600,
        protocol_sha256=sha(PLAN),heldout_used=False,hardware_io=False,
        scope='Single training seed and one fresh development scene seed; no reliability or hardware claim')
    write(OUT/'analysis.json',result);print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--train',choices=('control','candidate'));parser.add_argument('--eval',choices=('original','control','candidate'));a=parser.parse_args()
    if a.freeze:freeze();return
    assert os.isatty(1),'Use the one visible shared VS Code terminal'
    if a.train:
        from getup_contact_pilot import run
        from verify_getup_contact_pilot import verify
        p=checked()['arms'][a.train];run(p,PLAN,OUT);verify(p,OUT)
    elif a.eval:evaluate(a.eval)
    else:
        checked();assert not (OUT/'coordinator.json').exists(),'Do not restart a closed attempt'
        write(OUT/'coordinator.json',dict(status='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat()))
        for flag,role in (('--eval','original'),('--train','control'),('--eval','control'),('--train','candidate'),('--eval','candidate')):
            print('VISIBLE_PAIR_STAGE',flag,role,flush=True)
            subprocess.run([sys.executable,'-B',str(Path(__file__)),flag,role],cwd=WORK,check=True)
        analyze();write(OUT/'coordinator.json',dict(status='completed-review-required',pid=os.getpid()))


if __name__=='__main__':main()

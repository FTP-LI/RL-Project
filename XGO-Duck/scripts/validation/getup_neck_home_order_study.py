"""Sequential fixed-policy recovery test, with the simplest passing arm kept.

A: common-interface getup; B: add processed neck-HOME target;
C: add standing only when observed neck/trunk are ready. Adjacent arms change
one feature. Fixed new seed/all first episodes, no policy fitting or extension.
"""
from copy import deepcopy
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,subprocess,sys
import numpy as np
import onnx
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import WORK,read,sha
from record_io import write
from getup_stand_handoff_pair import prepare as parent_prepare
from getup_release_coverage import actor_model,cpu_check
from getup_neck_home_order import adapt_action,NeckHomeState,SETTINGS

PLAN=ROOT/'configs/experiments/getup_neck_home_order.json'
OUT=ROOT/'records/reports/getup-neck-home-order'


def prepare(p):
    cfg,display,meta=parent_prepare(p)
    cfg=replace(cfg,env=adapt_action(cfg.env,p['neck_index'],p['model_home'][p['neck_index']]))
    display.update(recorder='Complete11s neck-HOME/order study, four preselected worlds',native_live_window=False)
    return cfg,display,meta


def make_memory(p,meta,device):
    return NeckHomeState(p['model_home'],p['standing_home'],p['num_envs'],
        enabled=p['handoff_enabled'],neck_enabled=p['neck_home_enabled'],neck_index=p['neck_index'],
        device=device,settings=p['handoff_settings'])


def startup(env,obs,actors,p,record):
    from collect_getup_stand_handoff import startup as parent_startup
    from common_scene_contract import compare_arrays
    cpu=lambda x:x.detach().cpu().numpy().copy()
    term=env.action_manager.get_term('joint_pos')
    assert not term.neck_home_mask.any()
    parent_startup(env,obs,actors,p,record)
    attrs=('previous_target','previous_valid','standing_mask','neck_home_mask','requested_target',
           'projected_destination','_raw_actions','_processed_actions')
    saved={k:getattr(term,k).clone() for k in attrs}
    histories={k:getattr(env.action_manager,k).clone() for k in ('_action','_prev_action','_prev_prev_action')}
    ages=env.episode_length_buf.clone()
    physical=dict(qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel))
    rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state(env.device).clone()
    probe=torch.full_like(histories['_action'],.01);probe[:,p['neck_index']]=-4.
    term.standing_mask[:]=False;term.neck_home_mask[:]=False
    env.episode_length_buf.fill_(249);env.action_manager.process_action(probe)
    old=cpu(term._processed_actions)
    env.episode_length_buf.fill_(250)
    mask=torch.arange(env.num_envs,device=env.device)%2==0
    term.neck_home_mask=mask
    env.action_manager.process_action(probe)
    requested=np.asarray(p['model_home'],np.float32)+cpu(probe)
    desired=requested.copy();desired[cpu(mask),p['neck_index']]=p['model_home'][p['neck_index']]
    limits=np.asarray(p['joint_limits'],np.float32);legal=np.clip(desired,limits[:,0],limits[:,1])
    expected=old+np.clip(legal-old,-.04,.04)
    np.testing.assert_allclose(cpu(term.requested_target),requested,atol=1e-7,rtol=0)
    np.testing.assert_allclose(cpu(term.projected_destination),legal,atol=1e-7,rtol=0)
    np.testing.assert_allclose(cpu(term._processed_actions),expected,atol=1e-7,rtol=0)
    torch.testing.assert_close(env.action_manager._action,probe,atol=0,rtol=0)
    for k,v in saved.items():setattr(term,k,v.clone())
    for k,v in histories.items():getattr(env.action_manager,k).copy_(v)
    env.episode_length_buf.copy_(ages)
    compare_arrays(physical,dict(qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel)))
    assert env.common_step_counter==0 and torch.equal(rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state(env.device))
    for k,v in saved.items():torch.testing.assert_close(getattr(term,k),v,atol=0,rtol=0)
    proof=read(record/'startup.json');proof.update(neck_home_before_slew_checked=True,
        raw_request_retained=True,all_new_memories_restored=True,physical_steps=0)
    write(record/'startup.json',proof)


def freeze():
    from mjlab.scene import Scene
    from getup_stand_handoff_pair import checked as parent_checked
    from common_scene_contract import model_arrays,compare_arrays
    from evaluate_roll_reset_pair import differences
    from check_getup_neck_home_order import check
    assert not PLAN.exists() and not OUT.exists(),'Preserve all frozen attempts'
    parent=parent_checked()
    draft=read(ROOT/'configs/experiments/getup_neck_home_order_draft.json')
    evidence=ROOT/draft['basis'];audit=read(evidence)
    assert audit['actual_inputs']==208 and audit['successful_recoveries_added']==0
    assert sha(ROOT/audit['data'])==audit['data_sha256']
    seed=draft['proposed_seed'];assert seed==1015 and seed!=parent['evaluation_seed']
    p=deepcopy(parent['arms']['control'])
    metadata={x.key:x.value for x in onnx.load(ROOT/p['model']).metadata_props}
    neck=metadata['joint_names'].split(',').index('neck_pitch')
    assert abs(p['model_home'][neck]-p['standing_home'][neck])<1e-7
    assert p['joint_limits'][neck][0]<p['model_home'][neck]<p['joint_limits'][neck][1]
    codes=np.random.default_rng(seed).permutation(np.repeat(np.arange(4),[32,32,128,128]))
    poses=('prone','supine','left-side','right-side')
    groups={name:np.flatnonzero(codes==i).tolist() for i,name in enumerate(poses)}
    p.update(seed=seed,stage='S3-neck-home-order',neck_index=neck,handoff_settings=deepcopy(SETTINGS),
        split=dict(pose_codes=codes.tolist(),groups=groups),
        video_worlds={pose:groups[pose][0] for pose in poses},
        video_title='NECK HOME ORDER',progress_label='GETUP_NECK_HOME_ORDER')
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');arms={};cfgs={}
    for role,head,stand in [('control',False,False),('neck_home',True,False),('ordered_stand',True,True)]:
        arm=deepcopy(p);arm.update(arm=role,neck_home_enabled=head,handoff_enabled=stand,
            run_id=f'duck_getup_neck-home-order_{role.replace("_","-")}_s{seed}_{stamp}')
        arms[role]=arm;cfgs[role]=prepare(arm)[0]
    for role in ('neck_home','ordered_stand'):
        assert not differences(asdict(cfgs['control']),asdict(cfgs[role])),'Environment configs must be identical'
    model=model_arrays(Scene(cfgs['control'].env.scene,device='cpu').compile())
    with np.load(ROOT/'records/reports/getup-value-bootstrap/cpu-scene-fields.npz') as f:
        compare_arrays(model,{k:f[k] for k in f.files})
    fixture=check(arms['control'],cfgs['control'])
    numeric=cpu_check(dict(roles=dict(getup=p,standing=dict(p,model=p['standing_model'],model_sha256=p['standing_model_sha256']))))
    sources=deepcopy(parent['source_sha256'])
    for name in ('scripts/training/getup_neck_home_order.py','scripts/validation/check_getup_neck_home_order.py',
        'scripts/validation/getup_neck_home_order_study.py','scripts/validation/collect_fixed_recovery.py',
        'scripts/validation/verify_getup_neck_home_order.py','scripts/validation/recovery_end_hold.py'):
        sources[name]=sha(ROOT/name)
    m=dict(status='frozen-before-dynamics',arms=arms,source_sha256=sources,stamp=stamp,
        frozen_at=datetime.now(timezone.utc).isoformat(),evaluation_seed=seed,
        execution_order=['control','neck_home','ordered_stand'],maximum_episodes=960,
        maximum_evaluation_transitions=528000,learning_updates=0,heldout_used=False,hardware_io=False,
        automatic_retry=False,automatic_extension=False,draft_sha256=sha(ROOT/'configs/experiments/getup_neck_home_order_draft.json'),
        hypothesis_evidence_sha256=sha(evidence),
        common_adaptation='Same full-contact physical plant, common walking-level noise, model-specific HOME, per-model raw history and one post5s2rad/s projected target transition. No physical reset.',
        adjacent_differences=['control -> neck_home: processed neck target held at metadata HOME after observed-time/tilt latch',
                              'neck_home -> ordered_stand: enable standing after observed neck within20deg of HOME and current observed tilt<=45deg'],
        admission_gate='Keep original>=95percent requested/actual/overall recovery gates with>=32actual releases per class and zero unknown; ALSO require same group gates throughout final8..11s. Changed arm must actually activate its feature.',
        selection_rule='Prefer simplest admitted arm. Stop after a passing arm; only continue when prior functional gates fail with adequate pre2s release coverage. No resampling or added budget.',
        stop_rule='Any interface/nonfinite/contact-capacity/pause stops all and preserves. No retry, physical reset at switch or extension.',
        scope='New development controller decomposition; no policy fitting or hardware. Any lead requires independent confirmation, new-interface seven-command regression and full S5 sequences/robustness.')
    assert not torch.cuda.is_initialized();OUT.mkdir()
    write(PLAN,m)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
        identical_environment_configs=True,cpu_scene_fields_exact=True,fixture=fixture,model_import=numeric,
        requested_counts={k:len(v) for k,v in groups.items()},video_worlds=p['video_worlds'],
        config=yaml.load(yaml.dump(asdict(cfgs['control'])),Loader=yaml.BaseLoader),
        cuda_initialized=False,physical_steps=0))
    print('GETUP_NECK_HOME_ORDER_FROZEN',sha(PLAN),flush=True)


def checked():
    m=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for p in m['arms'].values():
        assert sha(ROOT/p['model'])==p['model_sha256'] and sha(ROOT/p['standing_model'])==p['standing_model_sha256']
        for name,digest in p['training_source_sha256'].items():assert sha(WORK/name)==digest,name
        for name,digest in p['library_source_sha256'].items():assert sha(Path(name))==digest,name
    return m


def admitted(proof,p):
    return bool(proof['all_motion_gates_passed'] and
        (not p['neck_home_enabled'] or proof['neck_home_episodes']>0) and
        (not p['handoff_enabled'] or proof['handoff_episodes']>0))


def summarize(m):
    roles={role:read(OUT/f'{role}-analysis.json') for role in m['execution_order'] if (OUT/f'{role}-analysis.json').exists()}
    assert roles and all(x['status']=='passed' for x in roles.values())
    selected=next((role for role,x in roles.items() if admitted(x,m['arms'][role])),None)
    result=dict(status='passed',roles=roles,selected_development_arm=selected,
        development_lead=selected is not None,actual_evaluation_transitions=176000*len(roles),
        evaluated_arms=list(roles),skipped_arms=[r for r in m['execution_order'] if r not in roles],
        protocol_sha256=sha(PLAN),learning_updates=0,heldout_used=False,hardware_io=False,automatic_extension=False,
        scope='Evidence integrity only; any admitted development controller still needs independent confirmation and full S2-S6 work.')
    write(OUT/'analysis.json',result)
    print('GETUP_NECK_HOME_ORDER_RESULT',json.dumps({k:v for k,v in result.items() if k!='roles'}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--role',choices=('control','neck_home','ordered_stand'))
    parser.add_argument('--verify-role',choices=('control','neck_home','ordered_stand'))
    parser.add_argument('--summarize',action='store_true');a=parser.parse_args()
    if a.freeze:freeze();return
    m=checked()
    if a.verify_role:
        from verify_getup_neck_home_order import verify
        verify(a.verify_role);return
    if a.summarize:summarize(m);return
    assert os.isatty(1),'Use original one visible VS Code terminal'
    if a.role:
        from collect_fixed_recovery import collect
        collect(a.role,m,plan_path=PLAN,out=OUT,prepare=prepare,make_memory=make_memory,startup_check=startup);return
    assert not (OUT/'coordinator.json').exists(),'No repeated study'
    write(OUT/'coordinator.json',dict(status='running',pid=os.getpid()))
    try:
        for role in m['execution_order']:
            subprocess.run([sys.executable,'-B',str(Path(__file__)),'--role',role],cwd=WORK,check=True)
            subprocess.run([sys.executable,'-B',str(Path(__file__)),'--verify-role',role],cwd=WORK,check=True)
            proof=read(OUT/f'{role}-analysis.json')
            if admitted(proof,m['arms'][role]):
                write(OUT/'selection-stop.json',dict(reason='Simplest passing arm selected; remaining arms skipped',selected=role,new_physics_steps=0));break
            if proof['other_release_episodes'] or not all(x['coverage_passed'] for x in proof['actual'].values()):
                write(OUT/'selection-stop.json',dict(reason='Inadequate pre2s release coverage; later controller cannot fix it',selected=None,new_physics_steps=0));break
        summarize(m)
        write(OUT/'coordinator.json',dict(status='completed-review-required',pid=os.getpid()))
    except BaseException as error:
        write(OUT/'coordinator.json',dict(status='stopped-review-required',pid=os.getpid(),error=repr(error)));raise


if __name__=='__main__':main()

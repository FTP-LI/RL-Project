"""One visible zero-learning test of the predeclared early stall correction.

Fixed official weights and the common plant/noise/target interface in both
arms. Candidate enables one measured-input detector/correction rule. Fresh
seed1017,320first11s episodes per arm, complete videos, original and end-hold
gates, no resampling, no old-protocol continuation, no automatic extension.
"""
from copy import deepcopy
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,subprocess,sys
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import WORK,read,sha
from record_io import write
from getup_stand_handoff_pair import prepare as parent_prepare
from getup_early_guard_controller import adapt_action,EarlyGuardState
from getup_release_coverage import cpu_check

PLAN=ROOT/'configs/experiments/getup_early_guard_study.json'
OUT=ROOT/'records/reports/getup-early-guard'


def prepare(p):
    cfg,display,meta=parent_prepare(p)
    cfg=replace(cfg,env=adapt_action(cfg.env,p['neck_index'],p['guard_enabled'],p['guard_settings']))
    display.update(recorder='Four complete0..11s first episodes, predeclared early neck guard',native_live_window=False)
    return cfg,display,meta


def make_memory(p,meta,device):
    return EarlyGuardState(p['model_home'],p['standing_home'],p['num_envs'],neck_index=p['neck_index'],
        lower_limit=p['joint_limits'][p['neck_index']][0],guard_enabled=p['guard_enabled'],device=device,
        settings=p['guard_settings'],handoff_settings=p['handoff_settings'])


def startup(env,obs,actors,p,record):
    from collect_getup_stand_handoff import startup as parent_startup
    from common_scene_contract import compare_arrays,model_arrays
    term=env.action_manager.get_term('joint_pos')
    extra={k:getattr(term,k).clone() for k in ('neck_home_mask','guard_correction')}
    parent_startup(env,obs,actors,p,record)
    for k,v in extra.items():setattr(term,k,v.clone())
    attrs=('previous_target','previous_valid','standing_mask','neck_home_mask','guard_correction',
           'requested_target','projected_destination','_raw_actions','_processed_actions')
    saved={k:getattr(term,k).clone() for k in attrs}
    histories={k:getattr(env.action_manager,k).clone() for k in ('_action','_prev_action','_prev_prev_action')}
    ages=env.episode_length_buf.clone();rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state(env.device).clone()
    cpu=lambda x:x.detach().cpu().numpy().copy()
    physical=dict(qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel))
    probe=torch.full_like(histories['_action'],.01);probe[:,p['neck_index']]=-4.
    term.standing_mask[:]=False;term.neck_home_mask[:]=False;term.guard_correction[:]=0
    env.episode_length_buf.fill_(149);env.action_manager.process_action(probe)
    before=cpu(term._processed_actions)
    mask=torch.arange(env.num_envs,device=env.device)%2==0;term.neck_home_mask=mask
    env.episode_length_buf.fill_(150);env.action_manager.process_action(probe)
    expected=before.copy()
    if p['guard_enabled']:expected[cpu(mask),p['neck_index']]+=.04
    np.testing.assert_allclose(cpu(term._processed_actions),expected,atol=1e-6,rtol=0)
    torch.testing.assert_close(env.action_manager._action,probe,atol=0,rtol=0)
    # Exercise the same detector on the actual CUDA device, without touching env.
    memory=make_memory(dict(p,guard_enabled=True),{},env.device)
    sensed=torch.zeros_like(obs['actor']);sensed[:,3]=.5;sensed[:,5]=-np.sqrt(.75)
    neck=p['neck_index'];sensed[:,6+neck]=np.deg2rad(-130)-p['model_home'][neck]
    for age in range(100,116):
        clock=torch.full_like(env.episode_length_buf,age)
        memory.inputs(sensed,clock,env.step_dt);memory.select(probe,probe,clock,env.step_dt)
        assert bool(memory.guard_latched.all())==(age==115)
    for k,v in saved.items():setattr(term,k,v.clone())
    for k,v in histories.items():getattr(env.action_manager,k).copy_(v)
    env.episode_length_buf.copy_(ages)
    compare_arrays(physical,dict(qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel)))
    assert env.common_step_counter==0 and torch.equal(rng,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state(env.device))
    with np.load(OUT/'cpu-scene-fields.npz') as f:compare_arrays(model_arrays(env.sim.mj_model),{k:f[k] for k in f.files})
    for k,v in saved.items():torch.testing.assert_close(getattr(term,k),v,atol=0,rtol=0)
    proof=read(record/'startup.json');proof.update(early_guard_before5s_checked=True,
        all_new_memories_restored=True,elapsed_detector_cuda_checked=True,full_physical_fields_exact=True,physical_steps=0)
    write(record/'startup.json',proof)


def freeze():
    from mjlab.scene import Scene
    from common_scene_contract import model_arrays,compare_arrays
    from evaluate_roll_reset_pair import differences
    from check_getup_early_guard import check
    assert not PLAN.exists() and not OUT.exists(),'Keep frozen attempts immutable'
    parent=read(ROOT/'configs/experiments/getup_late_feasibility_pair.json')
    assert sha(ROOT/'configs/experiments/getup_late_feasibility_pair.json')==read(ROOT/'records/reports/getup-late-feasibility/analysis.json')['protocol_sha256']
    draft=read(ROOT/'configs/experiments/getup_early_guard_draft.json')
    screen=read(ROOT/'records/reports/getup-early-support/guard-screen.json')
    assert screen['status']=='passed-cpu-only' and screen['independent_numpy_matches']
    assert sha(ROOT/'scripts/training/getup_early_guard.py')==screen['detector_sha256']
    assert sha(ROOT/'configs/experiments/getup_early_guard_draft.json')==screen['definition_sha256']
    seed=draft['new_independent_seed'];assert seed==1017
    codes=np.random.default_rng(seed).permutation(np.repeat(np.arange(4),[32,32,128,128]))
    poses=('prone','supine','left-side','right-side');groups={name:np.flatnonzero(codes==i).tolist() for i,name in enumerate(poses)}
    base=deepcopy(parent['arms']['original']);base.update(seed=seed,stage='S3-early-guard',
        split=dict(pose_codes=codes.tolist(),groups=groups),guard_settings=deepcopy(draft['settings']),
        video_worlds={name:groups[name][0] for name in poses},video_title='EARLY NECK GUARD',progress_label='EARLY_GUARD_EVAL',
        neck_home_enabled=False,handoff_enabled=False)
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');arms={};configs={};models={}
    for role,enabled in [('control',False),('candidate',True)]:
        p=deepcopy(base);p.update(arm=role,guard_enabled=enabled,run_id=f'duck_getup_early-guard_{role}_s{seed}_{stamp}')
        arms[role]=p;cfg=prepare(p)[0];configs[role]=asdict(cfg);models[role]=Scene(cfg.env.scene,device='cpu').compile()
    changes=differences(configs['control'],configs['candidate'])
    assert len(changes)==1 and changes[0]['path']=='.env.actions.joint_pos.guard_enabled',changes
    fields=model_arrays(models['control']);compare_arrays(fields,model_arrays(models['candidate']))
    with np.load(ROOT/'records/reports/getup-late-feasibility/cpu-scene-fields.npz') as f:compare_arrays(fields,{k:f[k] for k in f.files})
    fixture=check(arms['candidate'],prepare(arms['candidate'])[0]);imports=cpu_check(dict(roles=arms))
    sources=deepcopy(parent['source_sha256'])
    for name in ('scripts/training/getup_early_guard.py','scripts/training/getup_early_guard_controller.py',
        'scripts/validation/audit_getup_early_support.py','scripts/validation/audit_getup_early_guard.py',
        'scripts/validation/check_getup_early_guard.py','scripts/validation/verify_getup_early_guard.py',
        'scripts/validation/getup_early_guard_study.py'):
        sources[name]=sha(ROOT/name)
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name
    for p in arms.values():p['source_sha256']=sources
    assert not torch.cuda.is_initialized()
    m=dict(status='frozen-before-dynamics',stamp=stamp,arms=arms,source_sha256=sources,
        parent_sha256=sha(ROOT/'configs/experiments/getup_late_feasibility_pair.json'),
        draft_sha256=sha(ROOT/'configs/experiments/getup_early_guard_draft.json'),
        screen_sha256=sha(ROOT/'records/reports/getup-early-support/guard-screen.json'),
        maximum_evaluation_transitions=352000,maximum_episodes=640,learning_updates=0,
        heldout_used=False,hardware_io=False,automatic_retry=False,automatic_extension=False,
        continue_gate='Candidate all original/final95percent gates,actual>=32perclass,zero unknowns,and>=control+2successes',
        candidate_start_gate='Skip when control passes or actual coverage inadequate; improvement must be reachable before candidate starts',
        scope='One fresh development fixed-official-policy early stall correction; no learning, fullS5 or hardware claim',full_goal_preserved=True)
    OUT.mkdir();np.savez_compressed(OUT/'cpu-scene-fields.npz',**fields);write(PLAN,m)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),source_count=len(sources),
        configuration_differences=changes,full_physical_fields_exact=True,processor=fixture,imports=imports,
        saved_screen_overlap={k:{key:v for key,v in x.items() if key!='rows'} for k,x in screen['roles'].items()},
        cuda_initialized=False,configs={k:yaml.load(yaml.dump(v),Loader=yaml.BaseLoader) for k,v in configs.items()}))
    print('EARLY_GUARD_FROZEN',sha(PLAN),len(sources),'sources',flush=True)


def checked():
    m=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for p in m['arms'].values():
        assert sha(ROOT/p['model'])==p['model_sha256'] and sha(ROOT/p['standing_model'])==p['standing_model_sha256']
        for base,key in ((WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
            for name,digest in p[key].items():assert sha(base/name)==digest,name
    assert sha(ROOT/'configs/experiments/getup_early_guard_draft.json')==m['draft_sha256']
    assert sha(ROOT/'records/reports/getup-early-support/guard-screen.json')==m['screen_sha256']
    return m


def covered(proof):return proof['other_release_episodes']==0 and all(x['coverage_passed'] for x in proof['actual'].values())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--evaluate',choices=('control','candidate'));parser.add_argument('--verify',choices=('control','candidate'));a=parser.parse_args()
    if a.freeze:freeze();return
    if a.verify:
        from verify_getup_early_guard import verify
        verify(a.verify);return
    if a.evaluate:
        from collect_fixed_recovery import collect
        collect(a.evaluate,checked(),plan_path=PLAN,out=OUT,prepare=prepare,make_memory=make_memory,startup_check=startup);return
    assert os.isatty(1),'Use the original one visible VS Code shared task terminal'
    checked();assert not (OUT/'coordinator.json').exists(),'No restart or extension'
    write(OUT/'coordinator.json',dict(status='running',pid=os.getpid()))
    project=read(ROOT/'project.json');project['staged_training_request'].update(simulation_running=True,learning_running=False);write(ROOT/'project.json',project)
    def step(*argv):subprocess.run([sys.executable,'-B',str(Path(__file__)),*argv],cwd=WORK,check=True)
    try:
        step('--evaluate','control');step('--verify','control');control=read(OUT/'control-analysis.json')
        gate=covered(control) and not control['all_motion_gates_passed'] and control['end_hold']['overall']['timely_holds']+2<=320
        if gate:step('--evaluate','candidate');step('--verify','candidate')
        proofs={role:read(OUT/f'{role}-analysis.json') for role in ('control','candidate') if (OUT/f'{role}-analysis.json').exists()}
        candidate=proofs.get('candidate');admitted=bool(candidate and covered(candidate) and candidate['all_motion_gates_passed'] and
            candidate['end_hold']['overall']['timely_holds']>=control['end_hold']['overall']['timely_holds']+2)
        lead='control' if control['all_motion_gates_passed'] else ('candidate' if admitted else None)
        result=dict(status='passed',roles=proofs,counts={k:v['overall']['timely_holds'] for k,v in proofs.items()},
            final_hold_counts={k:v['end_hold']['overall']['timely_holds'] for k,v in proofs.items()},
            selected_development_arm=lead,candidate_admitted=admitted,candidate_started=gate,
            actual_evaluation_transitions=len(proofs)*176000,learning_updates=0,protocol_sha256=sha(PLAN),
            heldout_used=False,hardware_io=False,automatic_extension=False,
            scope='Development correction gate only; independent confirmation and fullS2-S6 still required')
        write(OUT/'analysis.json',result);write(OUT/'coordinator.json',dict(status='completed-review-required',pid=os.getpid()))
        print('EARLY_GUARD_RESULT',json.dumps({k:v for k,v in result.items() if k!='roles'}),flush=True)
    except BaseException as error:
        write(OUT/'coordinator.json',dict(status='stopped-review-required',pid=os.getpid(),error=repr(error)));raise
    finally:
        project=read(ROOT/'project.json');project['staged_training_request'].update(simulation_running=False,learning_running=False);write(ROOT/'project.json',project)


if __name__=='__main__':main()

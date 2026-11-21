"""Fixed-policy reward decomposition after failed velocity-width tuning.

No learning. Compare the better parent with the trained candidate under the
same low-speed commands and model HOME. Capture actual reward-manager rates,
then recompute only the stateless velocity term at both declared widths on
the SAME states. This diagnoses incentives; it does not prove the cause of
PPO updates across the full randomized training distribution.
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
from s1_reference_inputs import ROOT, WORK, read, sha
from record_io import write

PLAN=ROOT/'configs/experiments/reward_incentive_audit.json'
OUT=ROOT/'records/reports/reward-incentives'


def freeze():
    assert not PLAN.exists()
    assert read(ROOT/'records/reports/tracking-precision/analysis.json')['decision']=='stop-and-review-hypothesis'
    template=deepcopy(read(ROOT/'configs/experiments/official_bootstrap_screen.json'))
    template.pop('model');template.pop('model_sha256');template.pop('source_sha256')
    template['evaluation'].update(episodes_per_condition=16,episode_seconds=10)
    template['evaluation']['commands']={k:template['evaluation']['commands'][k] for k in ('forward','backward')}
    template['scope']='Reward incentive diagnostic; not formal gait acceptance or heldout evaluation'
    sources=['scripts/validation/'+n for n in ('reward_incentive_audit.py','evaluate_fixed_policy.py',
        'evaluation_metrics.py','verify_fixed_evaluation.py')]+['scripts/training/official_actor.py']
    library=WORK/'.venv/lib/python3.12/site-packages/mjlab'
    models={}
    for role,source in dict(parent='records/reports/official-actor-bootstrap/training-verification.json',
        trained='records/reports/tracking-precision/candidate/training-verification.json').items():
        check=read(ROOT/source);assert check['status']=='passed'
        if role=='parent':
            model=ROOT/'artifacts/duck_walk_official-actor-bootstrap_candidate_s42_20261102T180647/policy.onnx'
        else:model=Path(check['model'])
        assert sha(model)==check['model_sha256']
        models[role]=dict(path=str(model.relative_to(ROOT)),sha256=sha(model),verification=source,verification_sha256=sha(ROOT/source))
    plan=dict(status='frozen',frozen_at=datetime.now(timezone.utc).isoformat(),models=models,
        source_sha256={p:sha(ROOT/p) for p in sources},
        library_sha256={str(p):sha(p) for p in (library/'managers/reward_manager.py',library/'tasks/velocity/mdp/rewards.py')},
        template=template,model_home_alignment=True,settling_seconds_excluded=1,
        reward_widths=[.25,.10],velocity_reward_weight=3.,learning_updates=0,
        interpretation='Descriptive fixed-policy reward comparison under flat play conditions with aligned HOME; not the full training distribution')
    OUT.mkdir(exist_ok=True);write(PLAN,plan)
    print(json.dumps(plan,indent=2))


def child(role,plan):
    assert os.isatty(1),'Use the existing visible task terminal'
    import torch
    import onnx
    import evaluate_fixed_policy as core
    from official_actor import align_home
    model=ROOT/plan['models'][role]['path'];assert sha(model)==plan['models'][role]['sha256']
    folder=OUT/role;folder.mkdir(exist_ok=False)
    protocol=deepcopy(plan['template']);protocol.update(model=str(model),model_sha256=sha(model))
    frozen=folder/'protocol.json';write(frozen,protocol)
    meta={p.key:p.value for p in onnx.load(model).metadata_props}
    factory=core.load_env_cfg
    def aligned(*a,**kw):return align_home(factory(*a,**kw),meta)
    core.load_env_cfg=aligned
    env_base=core.ManagerBasedRlEnv
    class Capture(env_base):
        def __init__(self,*a,**kw):
            self.audit_rows={};super().__init__(*a,**kw)
            self.audit_output=Path(read(ROOT/'records/evaluations/current-evaluation.json')['path'])
            self.audit_terms=list(self.reward_manager.active_terms)
            assert self.reward_manager.get_term_cfg('track_linear_velocity').weight==3.
            assert self.reward_manager.get_term_cfg('track_linear_velocity').params['std']==.25
        def step(self,action):
            result=super().step(action)
            cpu=lambda x:x.detach().cpu().numpy().copy()
            command=cpu(self.command_manager.get_command('twist'))
            keys=[k for k,v in protocol['evaluation']['commands'].items() if np.allclose(command,v,atol=1e-6)]
            assert len(keys)==1
            self.audit_rows.setdefault(keys[0],[]).append(dict(rate=cpu(self.reward_manager._step_reward),
                reward=cpu(result[1]),velocity3=cpu(self.scene['robot'].data.root_link_lin_vel_b)))
            return result
        def close(self):
            if getattr(self,'audit_rows',None):
                for condition,rows in self.audit_rows.items():
                    np.savez_compressed(self.audit_output/f'{condition}-reward.npz',
                        **{k:np.stack([r[k] for r in rows]) for k in rows[0]})
                write(self.audit_output/'reward-metadata.json',dict(terms=self.audit_terms,
                    scale_by_dt=self.reward_manager._scale_by_dt,step_dt=self.step_dt,
                    velocity_weight=3.,velocity_width=.25,read_only_instrumentation=True))
            super().close()
    core.ManagerBasedRlEnv=Capture;core.PROTOCOL=frozen
    before=set((ROOT/'records/evaluations').iterdir())
    sys.argv=[str(Path(core.__file__)),'--onnx',str(model)]
    try:core.main()
    except SystemExit as error:
        if error.code not in (0,None):raise
    created=[p for p in set((ROOT/'records/evaluations').iterdir())-before if (p/'evaluation.json').exists()]
    assert len(created)==1
    output=created[0];report=read(output/'evaluation.json');assert report['status']=='completed'
    shutil.copy2(output/'evaluation.json',output/'evaluation-before-scope-annotation.json')
    report.update(scope=plan['interpretation'],reward_audit=True,formal_acceptance_eligible=False,
        meets_all_reference_criteria=False,instrumentation_sha256=sha(Path(__file__)))
    for item in report['conditions']:item['formal_acceptance_eligible']=False
    write(output/'evaluation.json',report)
    shutil.copy2(Path(__file__),output/Path(__file__).name)
    import verify_fixed_evaluation as verifier
    sys.argv=[str(Path(verifier.__file__)),str(output)];verifier.main()
    write(folder/'result.json',dict(status='collected-and-verified',evaluation=str(output),
        evaluation_sha256=sha(output/'evaluation.json'),model_sha256=sha(model),learning_updates=0))


def analyze(plan):
    import yaml
    rows={};cfgs=[]
    for role in plan['models']:
        result=read(OUT/role/'result.json');assert result['status']=='collected-and-verified'
        folder=Path(result['evaluation']);report=read(folder/'evaluation.json');meta=read(folder/'reward-metadata.json')
        assert sha(folder/'evaluation.json')==result['evaluation_sha256']==read(folder/'verification.json')['report_sha256']
        assert meta['scale_by_dt'] is True and meta['step_dt']==.02
        cfgs.append(yaml.load((folder/'env.yaml').read_text(),Loader=yaml.BaseLoader));rows[role]={}
        names=meta['terms'];velocity_index=names.index('track_linear_velocity')
        for condition,command in plan['template']['evaluation']['commands'].items():
            trace_path=folder/f'{condition}-trace.npz';reward_path=folder/f'{condition}-reward.npz'
            with np.load(trace_path) as t,np.load(reward_path) as r:
                active=t['active'].copy();active[:50]=False
                assert np.all(active.sum(axis=0)>0) and r['rate'].shape[:2]==active.shape
                assert np.isfinite(r['rate'][active]).all()
                np.testing.assert_allclose(r['rate'].sum(axis=2)*.02,r['reward'],atol=2e-6,rtol=2e-5)
                err=np.square(r['velocity3']-np.asarray(command)).sum(axis=2)
                expected=3.*np.exp(-err/.25**2)
                np.testing.assert_allclose(r['rate'][:,:,velocity_index],expected,atol=2e-6,rtol=2e-5)
                mean=lambda x:(x*active).sum(axis=0)/active.sum(axis=0)
                rate=mean(r['rate'].sum(axis=2))
                narrower=mean(r['rate'].sum(axis=2)-expected+3.*np.exp(-err/.1**2))
                components={name:float(mean(r['rate'][:,:,i]).mean()) for i,name in enumerate(names)}
                rows[role][condition]=dict(episodes=len(rate),mean_reward_rate=float(rate.mean()),
                    narrower_width_same_states_reward_rate=float(narrower.mean()),
                    component_reward_rates=components,per_episode_rate=rate.tolist(),
                    mean_signed_speed=float(mean(t['velocity'][:,:,0]).mean()*np.sign(command[0])),
                    inward_fraction_scope='Use the separate verified gait evaluation; this audit measures rewards',
                    trace_sha256=sha(trace_path),reward_sha256=sha(reward_path))
    assert all(cfg==cfgs[0] for cfg in cfgs),'Audit simulation configurations differ'
    differences={condition:{name:rows['trained'][condition]['component_reward_rates'][name]-rows['parent'][condition]['component_reward_rates'][name]
        for name in rows['parent'][condition]['component_reward_rates']} for condition in ('forward','backward')}
    result=dict(status='passed',rows=rows,trained_minus_parent_components=differences,
        scope=plan['interpretation'],protocol_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),
        interpretation_limit='Reward preference on these trajectories is not a causal decomposition of training; no new learning authorized by this result')
    write(OUT/'analysis.json',result);print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--model',choices=('parent','trained'));parser.add_argument('--analyze',action='store_true');args=parser.parse_args()
    if args.freeze:freeze();return
    plan=read(PLAN)
    for path,digest in plan['source_sha256'].items():assert sha(ROOT/path)==digest,path
    for path,digest in plan['library_sha256'].items():assert sha(Path(path))==digest,path
    if args.model:child(args.model,plan);return
    if args.analyze:analyze(plan);return
    assert os.isatty(1)
    target=OUT/'progress.json';assert not target.exists(),'Existing attempt retained; no automatic retry'
    state=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PLAN),commands=[])
    write(target,state)
    try:
        for role in plan['models']:
            command=[sys.executable,'-B',str(Path(__file__)),'--model',role]
            item=dict(command=command,started_at=datetime.now(timezone.utc).isoformat());state['commands'].append(item);write(target,state)
            print('REWARD_AUDIT_COMMAND:',json.dumps(command),flush=True)
            code=subprocess.call(command,cwd=WORK);item.update(exit_code=code,ended_at=datetime.now(timezone.utc).isoformat());write(target,state)
            assert code==0,'Stop diagnostic on failure; no retry'
        analyze(plan);state.update(status='completed',ended_at=datetime.now(timezone.utc).isoformat(),learning_updates=0)
    except BaseException as error:state.update(status='failed',error=repr(error));raise
    finally:write(target,state)


if __name__=='__main__':main()

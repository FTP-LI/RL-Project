"""Reanalyze saved reward rates without repeating simulation.

The frozen collector saved velocities AFTER ManagerBasedRlEnv.sim.forward(),
whereas rewards use derived state one physics substep earlier. Therefore the
old pointwise velocity check is invalid. Keep that failed audit unchanged.
Use the actual reward-manager term for decomposition and the exact identity
exp(-e/new_std**2) = exp(-e/old_std**2)**(old_std/new_std)**2.
This counterfactual changes only the stateless term on the same reward states;
it is not a predicted trained policy or a claim that all reward causes are known.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import yaml
from s1_reference_inputs import ROOT, WORK, read, sha
from record_io import write

OUT=ROOT/'records/reports/reward-incentives'
PLAN=ROOT/'configs/experiments/reward_incentive_audit.json'


def main():
    plan=read(PLAN);previous=read(OUT/'progress.json')
    assert previous['status']=='failed' and 'Not equal to tolerance' in previous['error']
    for path,digest in plan['source_sha256'].items():assert sha(ROOT/path)==digest,path
    for path,digest in plan['library_sha256'].items():assert sha(Path(path))==digest,path
    env_source=WORK/'.venv/lib/python3.12/site-packages/mjlab/envs/manager_based_rl_env.py'
    source=env_source.read_text()
    assert source.index('self.reward_buf = self.reward_manager.compute(dt=self.step_dt)')<source.index('    self.sim.forward()',source.index('  def step('))
    assert 'stale by one physics' in source
    errors=np.linspace(0,10,10001)
    np.testing.assert_allclose(np.exp(-errors/.25**2)**(.25/.1)**2,
        np.exp(-errors/.1**2),atol=1e-14,rtol=1e-12)
    rows={};configs=[];maximum_sum_error=0.
    for role in plan['models']:
        checked=read(OUT/role/'result.json');folder=Path(checked['evaluation'])
        report=read(folder/'evaluation.json');metadata=read(folder/'reward-metadata.json')
        assert checked['status']=='collected-and-verified'
        assert sha(folder/'evaluation.json')==checked['evaluation_sha256']==read(folder/'verification.json')['report_sha256']
        assert report['reward_audit'] and not report['formal_acceptance_eligible']
        assert report['model_sha256']==plan['models'][role]['sha256']
        assert metadata['scale_by_dt'] is True and metadata['step_dt']==.02
        configs.append(yaml.load((folder/'env.yaml').read_text(),Loader=yaml.BaseLoader))
        names=metadata['terms'];velocity_index=names.index('track_linear_velocity');rows[role]={}
        for condition,command in plan['template']['evaluation']['commands'].items():
            tp=folder/f'{condition}-trace.npz';rp=folder/f'{condition}-reward.npz'
            with np.load(tp) as t,np.load(rp) as r:
                active=t['active'].copy();active[:50]=False
                assert active.shape==(500,16) and np.all(active.sum(axis=0)>0)
                assert r['rate'].shape[:2]==active.shape and np.isfinite(r['rate']).all()
                total=r['rate'].sum(axis=2)
                np.testing.assert_allclose(total*.02,r['reward'],atol=2e-6,rtol=2e-5)
                maximum_sum_error=max(maximum_sum_error,float(np.abs(total*.02-r['reward']).max()))
                velocity=r['rate'][:,:,velocity_index]
                assert np.all((velocity>=0)&(velocity<=3.000001))
                narrow=3.*(np.clip(velocity.astype(float)/3.,0,1)**6.25)
                mean=lambda value:((value*active).sum(axis=0)/active.sum(axis=0)).mean()
                components={name:float(mean(r['rate'][:,:,i])) for i,name in enumerate(names)}
                post_error=np.square(r['velocity3']-np.asarray(command)).sum(axis=2)
                rows[role][condition]=dict(episodes=16,mean_reward_rate=float(mean(total)),
                    narrower_width_same_reward_states_rate=float(mean(total-velocity+narrow)),
                    mean_signed_speed=float(mean(t['velocity'][:,:,0])*np.sign(command[0])),
                    component_reward_rates=components,
                    post_step_velocity_reward_mismatch_max=float(np.abs(velocity-3*np.exp(-post_error/.25**2)).max()),
                    trace_sha256=sha(tp),reward_sha256=sha(rp))
    assert configs[0]==configs[1]
    cfg=configs[0]['rewards']['track_linear_velocity']
    assert cfg['params']=={'command_name':'twist','std':'0.25'} and cfg['weight']=='3.0'
    differences={condition:dict(sorted(((name,rows['trained'][condition]['component_reward_rates'][name]-rows['parent'][condition]['component_reward_rates'][name])
        for name in rows['parent'][condition]['component_reward_rates']),key=lambda x:abs(x[1]),reverse=True)) for condition in ('forward','backward')}
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),rows=rows,
        trained_minus_parent_components=differences,maximum_total_reward_reconstruction_error=maximum_sum_error,
        plan_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),env_step_source_sha256=sha(env_source),
        prior_failed_check_preserved='progress.json',simulation_repeated=False,learning_updates=0,
        sampling_issue='Reward uses pre-forward derived state; collector velocity is post-forward. Do not equate these sampling instants.',
        counterfactual='Exact stateless exponential-term transformation on captured reward states, not on the separately sampled velocities.',
        scope=plan['interpretation'],causal_training_claim=False)
    write(OUT/'reanalysis.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

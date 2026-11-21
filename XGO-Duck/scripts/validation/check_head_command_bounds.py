"""Check the proposed head command change on the real registered CPU config.

No environment, GPU, optimizer or simulation steps. Compare the entire config
after restoring the two declared neck intervals; check every stage against
compiled soft/hard limits and retain source hashes for the later experiment.
"""
from dataclasses import asdict
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from head_command_bounds import prepare_reachable_head_commands,HEAD_NAMES


def main():
    import torch
    import yaml
    import mjlab.tasks
    from mjlab.tasks.registry import load_env_cfg
    torch.set_num_threads(1)
    cfg=load_env_cfg('Mjlab-Velocity-Flat-XgoDuck',play=False)
    plain=lambda c:yaml.load(yaml.dump(asdict(c)),Loader=yaml.BaseLoader)
    before=plain(cfg);patched,result=prepare_reachable_head_commands(cfg)
    assert plain(cfg)==before,'Source config mutated'
    after=plain(patched)
    assert [(r['joint'],r['location']) for r in result['changes']]==[
        ('neck_pitch','curriculum.head_pose_range.step_36000'),
        ('neck_pitch','curriculum.head_pose_range.step_48000')]
    neck=result['joint_bounds'][0]
    assert abs(neck['hard_range_rad'][1]-torch.pi/3)<1e-8
    expected_cap=neck['soft_range_rad'][1]-neck['home_rad']
    assert 0.52 < expected_cap < 0.54
    for i,stage in enumerate(patched.curriculum['head_pose_range'].params['range_stages']):
        for (lo,hi),bound in zip(stage['ranges'],result['joint_bounds'],strict=True):
            assert abs(lo+hi)<1e-8
            assert bound['home_rad']+lo>=bound['soft_range_rad'][0]-1e-8
            assert bound['home_rad']+hi<=bound['soft_range_rad'][1]+1e-8
        if stage['step'] in (36000,48000):
            after['curriculum']['head_pose_range']['params']['range_stages'][i]['ranges'][0]=before['curriculum']['head_pose_range']['params']['range_stages'][i]['ranges'][0]
    assert after==before,'Unexpected change outside the two neck intervals'
    # Exercise the real curriculum and command sampler on a CPU fixture. This
    # catches a changed config that never reaches the live CommandTerm cfg.
    # It creates no simulation; preserve this process's RNG around the check.
    fixture=SimpleNamespace(device='cpu',num_envs=4096,step_dt=.02,common_step_counter=0)
    command=deepcopy(patched.commands['head_pose']).build(fixture)
    fixture.command_manager=SimpleNamespace(get_term=lambda name:command if name=='head_pose' else None)
    curriculum=patched.curriculum['head_pose_range']; stages=curriculum.params['range_stages']
    points=sorted({0,120000,*[s['step'] for s in stages],*[s['step']-1 for s in stages if s['step']>0]})
    rng=torch.random.get_rng_state(); sampler_checks=[]
    try:
        torch.manual_seed(42); ids=torch.arange(fixture.num_envs)
        for step in points:
            fixture.common_step_counter=step
            curriculum.func(fixture,ids,**curriculum.params)
            selected=max((s for s in stages if s['step']<=step),key=lambda s:s['step'])
            assert command.cfg.ranges==selected['ranges']
            command._resample_command(ids)
            values=command.command
            lo=torch.tensor([r[0] for r in selected['ranges']]); hi=torch.tensor([r[1] for r in selected['ranges']])
            assert torch.isfinite(values).all() and (values>=lo).all() and (values<=hi).all()
            absolute=values+torch.tensor([b['home_rad'] for b in result['joint_bounds']])
            low=torch.tensor([b['soft_range_rad'][0] for b in result['joint_bounds']])
            high=torch.tensor([b['soft_range_rad'][1] for b in result['joint_bounds']])
            assert (absolute>=low-1e-6).all() and (absolute<=high+1e-6).all()
            sampler_checks.append(dict(common_step_counter=step,selected_stage=selected['step'],
                samples=fixture.num_envs,command_min_rad=values.min(dim=0).values.tolist(),
                command_max_rad=values.max(dim=0).values.tolist(),all_targets_inside_soft_limits=True))
    finally:torch.random.set_rng_state(rng)
    assert torch.equal(torch.random.get_rng_state(),rng)
    assert plain(cfg)==before
    assert not torch.cuda.is_initialized()
    work=Path.home()/'XGO-Duck/workspace/xgoduck_rl'
    paths=[Path(__file__),ROOT/'scripts/training/head_command_bounds.py',
        ROOT/'scripts/validation/run_head_command_check.sh',ROOT/'.vscode/tasks.json',
        work/'src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py',
        work/'src/mjlab_microduck/tasks/mdp.py',
        work/'src/mjlab_microduck/robot/xgoduck_constants.py',
        work/'src/mjlab_microduck/robot/xgoduck/robot_walk.xml']
    result.update(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        full_configuration_diff_verified=True,original_configuration_unchanged=True,
        real_curriculum_and_sampler_checked=True,sampler_checks=sampler_checks,
        real_training_environment_created=False,
        simulation_created=False,cuda_initialized=False,head_motion_improved_verified=False,
        scope='Prepared configuration repair only; no training or gait effect has been measured')
    folder=ROOT/'records/reports/head-command-bounds';folder.mkdir(exist_ok=True)
    (folder/'check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('HEAD_COMMAND_BOUNDS_CHECK_PASSED')
    print('Only changes:',json.dumps(result['changes']))
    print('Real curriculum/sampler checks:',len(sampler_checks),'x',fixture.num_envs,'CPU command samples')
    print('Simulation steps: 0; optimizer updates: 0; head-motion improvement: not yet tested')
    print('Evidence:',folder/'check.json')


if __name__=='__main__':main()

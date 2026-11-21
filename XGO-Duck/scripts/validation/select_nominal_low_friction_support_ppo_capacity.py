"""Freeze the training pair at the largest measured size below the memory ceiling."""

from datetime import datetime
import fcntl
from pathlib import Path
from zoneinfo import ZoneInfo

import recover_nominal_low_friction_support_ppo as verified
import run_nominal_low_friction_support_ppo as ppo


def freeze():
    project=ppo.read(ppo.PROJECT)
    task, goal=project['staged_training_request'],project['goal_execution']
    profile=goal[verified.PROFILE_KEY]
    assert ppo.parent.allowed(project,ppo.PROFILE_STAGE) and not task['simulation_running']
    assert profile['status']=='review-required' and set(profile['reviewed'])=={'3072','3328'}
    assert ppo.KEY not in goal and not ppo.PLAN.exists()
    spec=ppo.read(verified.PROFILE,profile['protocol_sha256'])
    ppo.original.sources_checked(spec)
    paths={int(k):ppo.ROOT/row['path'] for k,row in profile['reviewed'].items()}
    good, higher=(ppo.read(paths[n],profile['reviewed'][str(n)]['sha256']) for n in (3072,3328))
    assert good['status']=='completed' and good['controls']==150 and good['physics_substeps']==750
    assert good['learning_updates']==higher['learning_updates']==0
    assert higher['status']=='review-required' and 'Observed GPU memory ceiling:' in higher['reason']
    assert higher['total_gpu_mib']==good['total_gpu_mib']
    cap=spec['capacity']['gpu_cap']*good['total_gpu_mib']
    assert good['peak_gpu_mib']<=cap<higher['peak_gpu_mib']
    parity=paths[3072].with_name('native-clock-parity.json')
    checks=ppo.read(parity)
    assert len(checks['rows'])==10 and len(checks['reward_audit'])==27
    assert max(x['max_force_error_n'] for x in checks['rows'])<=5e-5
    assert sum(row['actuation_intervals']==5 for row in checks['reward_audit'])==26
    # The higher size has already crossed the frozen limit. Use the completed
    # lower size and leave the larger unstarted trial explicitly recorded.
    selection=verified.CAPACITY/'selection.json'
    assert not selection.exists()
    write_result=dict(environments=3072,conservative_transitions_per_s=good['conservative_transitions_per_s'],
        measured={str(n):{k:r[k] for k in ('status','controls','physics_substeps','actual_transitions','peak_gpu_mib')}
                  for n,r in ((3072,good),(3328,higher))},
        gpu_limit_mib=cap,unstarted_sizes=[3584],reason='The 3328-world trial exceeded the frozen GPU memory ceiling.',
        source_sha256={ppo.rel(path):ppo.sha(path) for path in (verified.PROFILE,*paths.values(),parity)},learning_updates=0)
    ppo.write(selection,write_result)
    fixed=ppo.resize(spec,3072)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    runs={role:f'duck-getup-support-credit-friction080-{role}-s{ppo.INPUT_SEED}-{stamp}' for role in ppo.ROLES}
    fixed.update(runs=runs)
    files=(Path(__file__).resolve(),verified.PROFILE,*paths.values(),parity,selection)
    fixed['integrity'].update({ppo.rel(path):ppo.sha(path) for path in files})
    ppo.write(ppo.PLAN,fixed)
    goal[ppo.KEY]=dict(status='cuda-zero-ready',protocol=ppo.rel(ppo.PLAN),protocol_sha256=ppo.sha(ppo.PLAN),
                      runs=runs,selected_capacity=ppo.rel(selection),hardware_io=False)
    profile.update(status='reviewed-at-memory-ceiling',selection=ppo.rel(selection),selection_sha256=ppo.sha(selection))
    ppo.selected(project,ppo.STAGE,ppo.PLAN,ppo.STAGE+'-zero-ready',selection,
                 'Verify identical CUDA initial state and train both arms with 3072 environments and equal transition budgets.')
    ppo.write(ppo.PROJECT,project)
    print('SUPPORT_PPO_TRAINING_CHOSEN_AT_MEMORY_CEILING',ppo.sha(ppo.PLAN),runs,flush=True)


if __name__=='__main__':
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze()

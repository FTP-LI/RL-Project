"""Check whether archived recovery leaves enough time to move the unloaded right foot."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from mjlab.scene import Scene
import numpy as np
import torch

from preflight_nominal_supine_unloaded_foot import DESIGN, FootPlacement
from preflight_nominal_timestep_low_friction import configured
from review_nominal_low_friction_right_support_contact_timing import ARMS, PROJECT, REPORT, ROOT, SOURCE, read, rel, sha, trace_rows, write
from run_nominal_skill_composition import allowed


KEY = 'nominal_low_friction_right_foot_window'
STAGE = 'nominal-low-friction-right-foot-window'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
OUTPUT = REPORT/'right-foot-window.json'


def freeze():
    project = read(PROJECT)
    task,goal = project['staged_training_request'],project['goal_execution']
    assert allowed(project,'nominal-route-voltage-delay') and not task['simulation_running']
    factor = goal['nominal_route_voltage_delay']
    assert factor['status']=='reviewed'
    read(ROOT/factor['result'],factor['result_sha256'])
    source = goal[SOURCE]
    paths = [source[name] for name in ('protocol','collection','motion_readback')]
    for name,path in zip(('protocol','collection','motion_readback'),paths,strict=True):
        assert sha(ROOT/path)==source[name+'_sha256']
    paths += [rel(DESIGN),rel(Path(__file__)),rel(REPORT/'foot-placement.json')]
    plan = read(ROOT/source['protocol'])
    motion = read(ROOT/source['motion_readback'])
    worlds = motion['still_missed_in_all_arms']
    left_turn = sorted(set(worlds)&set(plan['seven_command']['groups']['turn-left']))
    assert len(worlds)==8 and len(left_turn)==4 and motion['newly_admitted_in_both_comparisons']==[218]
    assert plan['environment']['seed']==1257 and plan['composition']['foot_friction']==.8 and KEY not in goal
    assert not PLAN.exists() and not OUTPUT.exists()
    write(PLAN,dict(source_paths=paths,source_sha256={path:sha(ROOT/path) for path in paths},
        original_roles=list(ARMS[:2]),checks_from_control=136,checks_to_control=156,
        persistent_worlds=worlds,witness_world=218,left_turn_persistent_worlds=left_turn,
        minimum_paired_left_turn_opportunities=2,maximum_offset_per_control_rad=.02,
        timing='Left contact and right no contact. Required consecutive endpoints equal the number of rate-limited control updates plus one endpoint.',
        holdout_seed=2246,physical_steps=0,learning_updates=0,hardware_io=False))
    info = dict(current_protocol_execution=STAGE+'-frozen', current_evidence=rel(PLAN))
    goal[KEY] = dict(status='frozen-archive-only',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Verify right foot placement geometry and remaining support time on both low-friction references.')
    project.update(info)
    write(PROJECT,project)
    print('LOW_FRICTION_RIGHT_FOOT_WINDOW_FROZEN',sha(PLAN),flush=True)


def measures(planner, rows, index, plan, limits):
    start,end = plan['checks_from_control'],plan['checks_to_control']
    mask = [tuple(rows[k]['foot_contact'][index])==(True,False) for k in range(start,end+1)]
    reasons,available = Counter(),[]
    longest = run = 0
    for k in range(end,start-1,-1):
        run = run+1 if mask[k-start] else 0
        longest = max(longest,run)
        if not run:
            continue
        q = rows[k]['qpos'][index]
        move,rejected = planner.estimate(q,rows[k]['foot_contact'][index])
        if move is None:
            reasons.update(rejected)
            continue
        assert move['foot']=='right'
        if any(not limits[name][0] <= q[planner.plant.joint('robot/'+name).qposadr.item()]+offset <= limits[name][1]
               for name,offset in move['joint_offsets_rad'].items()):
            reasons['policy-joint-limit'] += 1
            continue
        rate_steps = int(np.ceil(max(abs(x) for x in move['joint_offsets_rad'].values())/plan['maximum_offset_per_control_rad']))
        available.append(dict(control=k,remaining_endpoints=run,required_endpoints=rate_steps+1,
            rate_fits_window=bool(run>=rate_steps+1),placement=move))
    return dict(longest_left_contact_right_unloaded_endpoints=longest,rejected=dict(reasons),
        feasible_placement_states=available,has_rate_compatible_opportunity=any(row['rate_fits_window'] for row in available))


def score(check):
    project = read(PROJECT)
    goal,task = project['goal_execution'],project['staged_training_request']
    node = goal[KEY]
    assert allowed(project,STAGE) and not task['simulation_running']
    assert node['status']==('reviewed' if check else 'frozen-archive-only')
    spec = read(PLAN,node['protocol_sha256'])
    for name,digest in spec['source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    protocol,collection,*_ = (ROOT/x for x in spec['source_paths'])
    physical,raw = read(protocol),read(collection)
    assert physical['environment']['seed']==1257 and not raw['reserved_seed_used']
    worlds = np.asarray([spec['witness_world'],*spec['persistent_worlds']])
    start,end = spec['checks_from_control'],spec['checks_to_control']
    initial = trace_rows(collection.parent/'shared',raw['prefix']['chunks'],worlds,start,start)
    planner = FootPlacement(Scene(configured(physical).scene,device='cpu').compile(),read(DESIGN))
    limits = dict(zip(physical['walking']['joint_names'],physical['environment']['joint_limits'],strict=True))
    results = {}
    for role in spec['original_roles']:
        rows = initial|trace_rows(collection.parent/role,raw['branches'][role]['chunks'],worlds,start+1,end)
        results[role] = {str(world):measures(planner,rows,i,spec,limits) for i,world in enumerate(worlds)}
    paired = [int(world) for world in worlds if all(rows[str(world)]['has_rate_compatible_opportunity'] for rows in results.values())]
    left = sorted(set(paired)&set(spec['left_turn_persistent_worlds']))
    admitted = len(left)>=spec['minimum_paired_left_turn_opportunities']
    assert planner.data.time==0 and not torch.cuda.is_initialized()
    payload = dict(status='verified-right-foot-support-window',protocol_sha256=sha(PLAN),paired_opportunities=paired,
        paired_left_turn_opportunities=left,admitted_for_actuator_preflight=admitted,seed=1257,results=results,
        physical_steps=0,learning_updates=0,holdout_seed_used=False,hardware_io=False)
    if check:
        assert read(OUTPUT,node['result_sha256'])==payload
    else:
        assert not OUTPUT.exists()
        write(OUTPUT,payload)
        node.update(status='reviewed',result=rel(OUTPUT),result_sha256=sha(OUTPUT),actuator_preflight_admitted=admitted)
        info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(OUTPUT))
        task.update(info,next_gate='Audit actual actuator authority for the reachable right-foot windows.' if admitted else
            'Use archived bilateral support and trunk pitch to select an earlier load transfer; the unloaded-foot pulse is closed.')
        project.update(info)
        write(PROJECT,project)
    print('LOW_FRICTION_RIGHT_FOOT_WINDOW_VERIFIED',sha(OUTPUT),paired,left,admitted,flush=True)


if __name__=='__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','score','check'))
    option = parser.parse_args().action
    freeze() if option=='freeze' else score(option=='check')

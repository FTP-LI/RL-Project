"""Measure the change in support credit when contact timestamps differ."""

import json
from pathlib import Path

import numpy as np

from review_roll_forward_input_stop import ROOT, arrays, read, sha
from roll_metrics import progress
from roll_stand_walk_sequence import ROLL, STAND_LANDING


REPORT = ROOT/'records/reports/roll-forward-input-integration/support-window-force'


def reward_order(training):
    installed = training['installed_sha256']
    env,mdp = (next(Path(name) for name in installed if name.endswith(suffix)) for suffix in ('/manager_based_rl_env.py','/tasks/mdp.py'))
    for path in (env,mdp):
        assert sha(path)==installed[str(path)],str(path)
    code = env.read_text(encoding='utf-8').splitlines()
    step = next(i for i,line in enumerate(code) if line.lstrip().startswith('def step('))
    reward = next(i for i in range(step+1,len(code)) if 'reward_manager.compute(' in code[i])
    refresh = next(i for i in range(reward+1,len(code)) if 'self.sim.forward()' in code[i])
    observations = next(i for i in range(refresh+1,len(code)) if 'observation_manager.compute(' in code[i])
    signals = mdp.read_text(encoding='utf-8').splitlines()
    sensor = next(i for i,line in enumerate(signals) if line.startswith('def _sensor_any_contact('))
    found = next(i for i in range(sensor+1,sensor+9) if '.data.found' in signals[i])
    roll = next(i for i,line in enumerate(signals) if line.startswith('def _update_roulade_accum('))
    uses = next(i for i in range(roll+1,roll+50) if '_sensor_any_contact(env, _ROULADE_SUPPORT_SENSOR)' in signals[i])
    return dict(env=str(env),env_sha256=installed[str(env)],reward_compute_line=reward+1,post_reward_forward_line=refresh+1,
        observation_compute_line=observations+1,motion_source=str(mdp),motion_sha256=installed[str(mdp)],
        sensor_cache_field_line=found+1,rolling_support_use_line=uses+1)


def group_summary(positive, cached, fresh, angle, selected):
    preferred, missed = cached[:,selected]&~fresh[:,selected],~cached[:,selected]&fresh[:,selected]
    a,b = positive[:,selected],angle[:,selected]
    gain,loss = (a*preferred).sum(axis=0),(a*missed).sum(axis=0)
    ranges = b[preferred & (a>0.)]
    return dict(worlds=int(selected.sum()),cached_only_positive=int((preferred&(a>0.)).sum()),
        fresh_only_positive=int((missed&(a>0.)).sum()),
        fresh_excluded_positive=int((~fresh[:,selected]&(a>0.)).sum()),
        extra_credit_median_deg=float(np.median(gain)),missing_credit_median_deg=float(np.median(loss)),
        cached_minus_fresh_credit_median_deg=float(np.median(gain-loss)),
        excess_credit_geometric_angle_quantiles_deg=np.quantile(ranges,[.05,.5,.95]).tolist() if ranges.size else None)


def main():
    proof = REPORT/'force-readback.json'
    archived = read(proof)
    project = read(ROOT/'project.json')
    assert sha(proof)==project['goal_execution']['roll_support_window_probe']['result_sha256']
    assert all(sha(ROOT/name)==value for name,value in archived['source_sha256'].items())
    plan_path = ROOT/project['goal_execution']['roll_friction_paired']['protocol']
    assert sha(plan_path)==project['goal_execution']['roll_friction_paired']['sha256']
    execution = reward_order(read(plan_path))
    root = ROOT/'records/evaluations'/archived['run_id']
    plan = read(root/'protocol.json')
    assert sha(root/'protocol.json')==archived['protocol_sha256']
    endpoint = root/'endpoint-contact.npz'
    endpoint_receipt = read(root/'endpoint-contact.json')
    assert sha(endpoint)==endpoint_receipt['sha256'] and endpoint_receipt['complete'] and not endpoint_receipt['first_mismatches']
    contacts,trace,clock = (arrays(path) for path in (endpoint,root/'trace.npz',root/'contact-clock.npz'))
    first,last = (plan['force_capture'][name] for name in ('first_control','last_control'))
    start,n = plan['initial_stand_controls'],plan['worlds']
    sl=slice(first+1,last+1)
    assert first>=start and contacts['cache_valid'][sl,0].all()
    cached,fresh = (contacts[name][sl,:,0] for name in ('cached','fresh'))
    np.testing.assert_array_equal(fresh,trace['ground_contact'][sl])
    np.testing.assert_array_equal(fresh,contacts['constraint_points'][sl,:,0]>0)
    np.testing.assert_array_equal(clock['substep'][4::5],contacts['substep'][sl])
    last_points = clock['points'][4::5,:,0]>0
    angle,_,mat = progress({name:trace[name][start:last+1] for name in ('quaternion','ground_contact')})
    flat = np.clip((.866-np.abs(mat[1:,:,2,1]))/(.866-.5),0.,1.)
    eligible = trace['active'][sl] & (trace['phase_next'][sl]==ROLL)
    positive = np.rad2deg((np.maximum(np.diff(angle,axis=0),0.)*flat**2*(3.-2.*flat))[first-start:])*eligible
    roll = (trace['phase_next'][start:]==STAND_LANDING).any(axis=0)
    assert int(roll.sum())==archived['original_roll_success'] and positive.shape==(last-first,n)
    degrees = np.rad2deg(angle[first+1-start:])
    groups = {name:group_summary(positive,cached,fresh,degrees,mask) for name,mask in (('roll-pass',roll),('roll-fail',~roll))}
    result = dict(run_id=archived['run_id'],worlds=n,control_numbers=list(range(first+1,last+1)),
        cached_fresh_disagreements=int((cached!=fresh).sum()),cached_last_substep_contact_disagreements=int((cached!=last_points).sum()),
        installed_reward_order=execution,roll_groups=groups,scope='Fixed trajectory, same integrated angle with alternate recorded contact timestamps',
        source_sha256={file.relative_to(ROOT).as_posix():sha(file) for file in (proof,endpoint,plan_path,Path(__file__))},
        physical_steps_added=0,learning_updates=0,heldout_used=False,hardware_io=False)
    path=REPORT/'reward-contact-timing-audit.json'
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print('ROLL_REWARD_CONTACT_TIMING',result['cached_fresh_disagreements'],groups['roll-fail'],groups['roll-pass'])


if __name__=='__main__':
    main()

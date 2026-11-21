"""Read saved paired roll routes and localize the forward yaw errors."""

import argparse
from collections import Counter

import numpy as np

import evaluate_roll_friction_pair as evaluation


def spread(values):
    return dict(minimum=float(np.min(values)), median=float(np.median(values)), maximum=float(np.max(values)))


def read_condition(seed, condition):
    report = evaluation.REPORT / f's{seed}-{condition}-low-friction'
    proof = evaluation.read(report/'independent.json')
    folder = evaluation.ROOT / 'records/evaluations' / proof['run_id']
    plan, manifest, score = (evaluation.read(folder/name) for name in ('protocol.json','guarded-evaluation.json','raw-score.json'))
    trace, actions = folder/'trace.npz', folder/'commands.npz'
    evaluation.require(proof['seed']==seed and proof['condition']==condition and proof['full_route']
        and proof['model_sha256']==plan['roll']['model_sha256'] and proof['learning_updates']==0
        and manifest['trace_sha256']==evaluation.sha(trace) and manifest['command_sha256']==evaluation.sha(actions),
        'Forward yaw diagnosis requires verified, full matched routes')
    with np.load(trace,allow_pickle=False) as data:
        yaw = data['angular_velocity'][...,2]
        active = data['active']
    with np.load(actions,allow_pickle=False) as data:
        feedback = data['feedback_yaw']
    cap = plan['feedback']['maximum_yaw_input_rad_s']
    limit = plan['walking']['criteria']['yaw_error_max']
    duration = plan['forward_controls']
    evaluation.require(duration==1000 and plan['roll']['criteria']['step_dt']==.02, 'The forward timing contract changed')
    world_rows = {}
    failures = Counter()
    for row in score['forward']:
        if not row['entered_forward']:
            continue
        i, start = row['world'], row['forward_start']
        state = slice(start+1,start+duration+1)
        command = slice(start,start+duration)
        keep = active[state,i]
        values = yaw[state,i][keep]
        evaluation.require(keep.all() and np.isclose(np.mean(np.abs(values)),row['yaw_error_rad_s'],atol=1e-8),
                           'Forward window changed during diagnosis')
        known = feedback[command,i]
        failures.update(row['failed_checks'])
        world_rows[i] = dict(abs_yaw=float(np.abs(values).mean()), signed_yaw=float(values.mean()),
                             first_two_seconds_abs_yaw=float(np.abs(values[:100]).mean()),
                             last_five_seconds_abs_yaw=float(np.abs(values[-250:]).mean()),
                             feedback_cap_fraction=float(np.isclose(np.abs(known),cap,rtol=0,atol=1e-6).mean()),
                             signed_forward_speed_m_s=row['signed_speed_m_s'])
    evaluation.require(len(world_rows)==sum(row['entered_forward'] for row in score['forward']), 'Forward world count differs')
    receipt = dict(run_id=proof['run_id'], source_sha256=evaluation.sha(report/'independent.json'),
                   trace_sha256=manifest['trace_sha256'], command_sha256=manifest['command_sha256'],
                   entered_forward=len(world_rows), failures=dict(failures), worlds=world_rows)
    contract = dict(candidate_sha256=proof['model_sha256'], walk_sha256=plan['walking']['model_sha256'],
                    forward_command=plan['walking']['commands']['forward'], feedback=plan['feedback'],
                    forward_yaw_threshold_rad_s=limit)
    return receipt, contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed',type=int,choices=(1247,1248))
    seed = parser.parse_args().seed
    normal, common = read_condition(seed,'nominal')
    low, checked = read_condition(seed,'friction-low')
    evaluation.require(common==checked,'The fixed actor or feedback differs across the two evaluations')
    paired = sorted(normal['worlds'].keys() & low['worlds'].keys())
    evaluation.require(paired, 'No scene entered forward under both friction conditions')
    for run in (normal,low):
        run['entered_forward_spread'] = {field:spread([row[field] for row in run['worlds'].values()])
                                       for field in run['worlds'][paired[0]]}
        run['paired_forward_spread'] = {field:spread([run['worlds'][i][field] for i in paired])
                                       for field in run['worlds'][paired[0]]}
    result = dict(seed=seed, contract=common, matched_forward_worlds=len(paired), paired_worlds=paired,
                  nominal=normal, friction_low=low, learning_updates=0, new_physics_substeps=0,
                  scope='Matched candidate routes from disk. No low-friction forward observations from the other actors after their fixed gate stop.')
    output = evaluation.REPORT / f's{seed}-forward-yaw-diagnosis.json'
    evaluation.put(output,result)
    print('ROLL_FRICTION_YAW_DIAGNOSIS', seed, len(paired),
          normal['paired_forward_spread'],low['paired_forward_spread'],flush=True)


if __name__=='__main__':
    main()

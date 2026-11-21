"""Augment original independent metrics with common-scene runtime contracts.

Check actual no-step startup records, retained nominal primary parameters,
contact capacity, raw histories and model replay. Numeric gait failure remains
a failed gate even when all data-integrity checks pass.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnxruntime as ort
import yaml
from s1_reference_inputs import ROOT,read,sha
from record_io import write

OUT=ROOT/'records/reports/common-plant-screen'
PLAN=ROOT/'configs/experiments/common_plant_screen.json'


def get_record(skill):
    if skill=='roll':return ROOT/read(OUT/'roll/reference-analysis.json')['evaluator']
    return Path(read(OUT/skill/'progress.json')['path'])


def verify(skill):
    master=read(PLAN);folder=get_record(skill);p=read(folder/'protocol.json');report=read(folder/'evaluation.json')
    pre=read(OUT/f'{skill}-preflight.json')
    assert report['status']=='completed'
    assert yaml.load((folder/'env.yaml').read_text(),Loader=yaml.BaseLoader)==pre['config']
    assert p['skill']==skill and p['seed']==1009 and sha(ROOT/p['model'])==p['model_sha256']
    for name,digest in p['source_sha256'].items():assert sha(folder/Path(name).name)==digest,name
    runtime=read(folder/'runtime-contract.json');assert runtime['mode']=='rollout'
    assert runtime['control_transitions']<=master['budgets'][skill] and not runtime['capacity_reached']
    assert runtime['control_steps']>0 and runtime['extra_observation_or_render_steps']==0
    static=read(folder/'scene-static-contract.json');nominal=read(folder/'nominal-contract.json')
    assert static['status']==nominal['status']=='passed' and nominal['common_step_counter']==0
    assert sha(folder/'scene-static-fields.npz')==static['archive_sha256']
    assert sha(folder/'nominal-primary-fields.npz')==nominal['primary_sha256']
    assert sha(folder/'expanded-fields-before-physics.npz')==nominal['expanded_sha256']
    starts=read(folder/'interface-checks.json');assert len(starts)==(4 if skill=='getup' else 1)
    for i,r in enumerate(starts):
        assert r['status']=='passed' and r['rng_unchanged'] and r['initial_raw_history_zero']
        assert sha(folder/f'reset-{i}-interface.npz')==r['archive_sha256']
    startup=OUT/'startup'/skill;startup_runtime=read(startup/'runtime-contract.json')
    assert startup_runtime['control_steps']==0
    for name in ('scene-static-fields.npz','nominal-primary-fields.npz'):
        with np.load(folder/name) as a,np.load(startup/name) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    replayed=0;max_error=0.
    if skill=='getup':
        # The original recovery verifier checks geometry/holds and media;
        # independently replay released actions and all raw history here.
        options=ort.SessionOptions();options.intra_op_num_threads=1
        session=ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
        freeze=round(p['criteria']['freeze_seconds']/.02)
        for condition in report['conditions']:
            with np.load(folder/f'{condition["condition"]}-trace.npz') as t:
                valid=t['active'][1:]
                np.testing.assert_allclose(t['observations'][1:,:,34:48][valid],t['raw_policy_actions'][:-1][valid],atol=1e-7,rtol=0)
                for step in range(freeze,len(t['time']),40):
                    for index in np.flatnonzero(t['active'][step]):
                        action=session.run(None,{'obs':t['observations'][step,index:index+1]})[0][0]
                        expected=t['raw_policy_actions'][step,index]
                        np.testing.assert_allclose(action,expected,atol=2e-5,rtol=2e-5)
                        replayed+=1;max_error=max(max_error,float(abs(action-expected).max()))
    result=dict(status='passed',skill=skill,record=str(folder.relative_to(ROOT)),
        model_sha256=p['model_sha256'],protocol_sha256=sha(folder/'protocol.json'),runtime=runtime,
        complete_scene_startup_and_rollout_fields_exact=True,nominal_primary_all_worlds_equal=True,
        model_home_history_and_rng_checked=True,getup_replayed_inputs=replayed,
        getup_maximum_replay_error=max_error,interface_checks=starts,script_sha256=sha(Path(__file__)),
        gait_acceptance='Read original per-skill independent analysis; this file verifies integrity only')
    write(OUT/skill/'common-verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='interface_checks'},indent=2))
    return result


def analyze():
    master=read(PLAN);verified={k:read(OUT/k/'common-verification.json') for k in ('walk','getup','roll')}
    assert all(v['status']=='passed' for v in verified.values())
    walk=read(OUT/'walk/s1009/analysis.json');getup=read(OUT/'getup/analysis.json');roll=read(OUT/'roll/reference-analysis.json')
    assert walk['status']==getup['status']==roll['status']=='passed'
    gates=dict(walk=walk['functional_development_passed'],getup=getup['all_conditions_passed'],roll=roll['numeric_gate_passed'])
    result=dict(status='passed',stage='S5-common-plant-nominal-screen',functional_gates=gates,
        all_skill_gates_passed=all(gates.values()),continuous_switching_validated=False,
        walking_conditions=walk['rows'],getup_conditions=[{k:v for k,v in c.items() if k!='episodes'} for c in getup['conditions']],
        roll_successes=roll['successes'],roll_episodes=roll['episodes'],
        actual_control_transitions=sum(v['runtime']['control_transitions'] for v in verified.values()),
        maximum_control_transitions=master['maximum_control_transitions'],learning_updates=0,heldout_used=False,
        protocol_sha256=sha(PLAN),source_task_gates_preserved=True,
        decision='Prepare bounded continuous-transition protocol' if all(gates.values()) else 'Review failed skill regressions before continuous-transition acceptance',
        scope='One nominal shared physical plant and one development seed; no robustness, continuous-switch or hardware claim')
    assert result['actual_control_transitions']<=result['maximum_control_transitions']
    write(OUT/'analysis.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('skill',nargs='?',choices=['walk','getup','roll']);p.add_argument('--analyze',action='store_true');a=p.parse_args()
    analyze() if a.analyze else verify(a.skill)

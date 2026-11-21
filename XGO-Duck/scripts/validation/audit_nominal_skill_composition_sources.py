"""Read the original get-up, roll and commanded-action interfaces before composing them."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT/'records/reports/goal-review/nominal-skill-composition-sources.json'
ROLLOUT_PATH = 'records/reports/official-route-independent-s1235/analysis.json'
ROUTED_PATH = 'artifacts/delivery/routed-seven-command-simulation.json'
PLANT_PATH = 'records/reports/common-plant/source-contract.json'


def digest(path):
    with (ROOT/path).open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def read(path, expected=None):
    if expected is not None and digest(path)!=expected:
        raise ValueError(f'Source hash mismatch: {path}')
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def model(path, expected, joint_names, options):
    assert digest(path)==expected, path
    session=ort.InferenceSession(str(ROOT/path),sess_options=options,providers=['CPUExecutionProvider'])
    input_, = session.get_inputs()
    output, = session.get_outputs()
    meta=session.get_modelmeta().custom_metadata_map
    assert input_.shape==[1,61] and output.shape==[1,14]
    assert meta['joint_names'].split(',')==joint_names and float(meta['action_scale'])==1.
    home=np.asarray(meta['default_joint_pos'].split(','),dtype=np.float32)
    sample, = session.run(None,{input_.name:np.zeros((1,61),np.float32)})
    assert home.shape==(14,) and np.isfinite(home).all() and np.isfinite(sample).all()
    return dict(path=path,sha256=expected,input=input_.shape,output=output.shape,action_scale=1.,
                policy_home=[float(value) for value in meta['default_joint_pos'].split(',')],
                observation_terms=meta['observation_names'].split(','),command_names=meta['command_names'].split(','),
                cpu_finite_output=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    project=read('project.json')
    state=project['goal_execution']['nominal_skill_composition']
    inputs={path:digest(path) for path in (ROLLOUT_PATH,ROUTED_PATH,PLANT_PATH)}
    assert inputs==state['selected_source_sha256']
    nominal,routed,plant=[read(path,inputs[path]) for path in (ROLLOUT_PATH,ROUTED_PATH,PLANT_PATH)]
    roll_plan=read(nominal['protocol'],nominal['protocol_sha256'])
    getup_plan=read(routed['protocol']['path'],routed['protocol']['sha256'])
    read(routed['result']['path'],routed['result']['sha256'])
    assert plant['compiled_robot_fields_exact'] and plant['physical_config_exact']
    assert (plant['common_voltage_v'],plant['actuator_delay_seconds'])==(7.7,.004)
    assert all((task['physics_dt'],task['control_dt'])==(.004,.02) for task in plant['tasks'])
    assert {task['skill'] for task in plant['tasks']}=={'getup','roll','walk'}
    assert all(digest(src['path'])==src['sha256'] for src in routed['controller_sources'])
    names=roll_plan['walking']['joint_names']
    assert len(names)==14 and names==getup_plan['walking']['joint_names']
    settings=getup_plan['sequence']['controller']
    assert routed['control_hz']==50 and settings['no_reset_at_any_switch']
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    options.inter_op_num_threads=1
    policies={role:model(entry['onnx']['path'],entry['onnx']['sha256'],names,options)
              for role,entry in routed['policy_roles'].items()}
    roll=roll_plan['roll']
    assert digest(roll['model'])==roll['model_sha256']
    policies['official-forward-roll']=model('artifacts/delivery/official/xgoduck_roulade.onnx',roll['model_sha256'],names,options)
    plan=getup_plan['seven_command']
    payload=dict(source_sha256=inputs,script_sha256=digest(Path(__file__)),controller_period_s=.02,physics_period_s=.004,
        actuator_delay_s=.004,nominal_supply_v=7.7,joint_names=names,policies=policies,
        original_getup_standing_bridge_controls=settings['cross_model_bridge_controls'],
        original_roll_standing_bridge_controls=roll_plan['cross_model_bridge_controls'],
        roll_protocol_explanatory_text=roll_plan['action_boundary'],
        original_roll_route=dict(seed=nominal['seed'],worlds=nominal['worlds'],minimum_roll=roll_plan['minimum_roll_completion'],
            minimum_route=nominal['minimum_route_gate'],actual_roll=nominal['roll_landing_worlds'],actual_route=nominal['forward_success_worlds']),
        original_seven_commands=dict(worlds=routed['worlds'],completed=routed['complete'],
            worlds_by_command={command:len(worlds) for command,worlds in plan['groups'].items()},
            minimum_by_command=plan['minimum_by_command'],minimum_overall=plan['minimum_overall']),
        scope='Frozen component metadata and CPU inference; combined physical controller not sampled',
        new_physical_steps=0,learning_updates=0,reserved_seed_used=False,hardware_io=False)
    encoded=json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.check:
        assert RESULT.read_text(encoding='utf-8')==encoded
    else:
        with RESULT.open('x',encoding='utf-8',newline='\n') as out:
            out.write(encoded)
    print('NOMINAL_SKILL_SOURCES_VERIFIED',digest(RESULT),len(policies),len(names),flush=True)


if __name__=='__main__':
    main()

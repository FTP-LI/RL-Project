"""Audit the existing target-over-limit reward on saved real trajectories.

Executes the unchanged official reward function with the installed action
property getters and recorded targets. No environment construction, simulation,
optimizer or hardware. This verifies units/formula/accessors and measures an
unweighted diagnostic quantity; it does not select a training reward weight.
"""
import argparse
import ast
import __future__
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction

ROOT=Path(__file__).resolve().parents[2]
def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();assert output.is_relative_to(ROOT/'records')
    source=ROOT/'official_resources/xgoduck_rl/src/mjlab_microduck/tasks/mdp.py'
    function=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='action_over_limit_penalty')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec',flags=__future__.annotations.compiler_flag),namespace)
    penalty=namespace['action_over_limit_penalty']
    snapshot=ROOT/'records/experiments/baseline_extension_20261101T165647_s42/gait-diagnostics/20261101T090019523741Z'
    meta=read(snapshot/'live-status.json')
    xmlpath=ROOT/'official_resources/xgoduck_rl/src/mjlab_microduck/robot/xgoduck/robot_walk.xml'
    xml=ET.parse(xmlpath).getroot();assert xml.find('compiler').get('angle')=='radian'
    xmlranges={j.get('name'):np.fromstring(j.get('range'),sep=' ') for j in xml.findall('.//worldbody//joint') if j.get('name') in meta['joint_names']}
    ranges=np.asarray([xmlranges[name] for name in meta['joint_names']])
    np.testing.assert_allclose(ranges,meta['joint_ranges_rad'],atol=1e-10)
    env_text=(snapshot/'env.yaml').read_text()
    assert 'action_over_limit' not in env_text
    for getter in ('raw_action','scale','offset','target_ids'):
        assert isinstance(inspect.getattr_static(JointPositionAction,getter),property)
    run=ROOT/'records/evaluations/custom_onnx_reference_20261101T144436_s1001'
    report=read(run/'evaluation.json')
    assert report['model_metadata']['joint_names'].split(',')==meta['joint_names']
    assert np.all(np.asarray(report['action_filter_alpha'])==0)
    sim_home=np.asarray(report['simulation_home'],dtype=np.float32)
    # Use the actual class's property getters, without constructing an env.
    # The compact fixture has exactly the 14 named controlled joints; each
    # limit has already been matched by name to the archived XML and observer.
    term=object.__new__(JointPositionAction)
    term._scale=1.0
    term._target_ids=torch.arange(14)
    fixture=SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name:term),
        scene={'robot':SimpleNamespace(data=SimpleNamespace(joint_pos_limits=torch.as_tensor(ranges,dtype=torch.float32)[None]))})
    rows=[];inputs={}
    torch.set_num_threads(2)
    for item in report['conditions']:
        path=run/(item['condition']+'-trace.npz')
        with np.load(path,allow_pickle=False) as data:
            active=data['active']
            targets=data['requested_joint_targets'][active]
            homes=np.broadcast_to(sim_home,data['requested_joint_targets'].shape)[active].copy()
            np.testing.assert_array_equal(data['actions'],data['applied_actions'])
        # simulation_home is saved per environment, not just one 14-vector.
        # Keep its environment identity while selecting each active frame.
        term._offset=torch.as_tensor(homes)
        term._raw_actions=torch.as_tensor(targets-homes)
        torch_result=penalty(fixture,overshoot=0.3).detach().numpy()
        over=np.maximum(targets-(ranges[:,1]+0.3),0)+np.maximum((ranges[:,0]-0.3)-targets,0)
        expected=over.sum(axis=-1)
        np.testing.assert_allclose(torch_result,expected,rtol=1e-5,atol=2e-6)
        per_joint=over.mean(axis=0);roll=[meta['joint_names'].index(s+'_hip_roll') for s in ('left','right')]
        rows.append(dict(condition=item['condition'],active_frames=int(active.sum()),
            unweighted_penalty_mean_rad=float(expected.mean()),
            unweighted_penalty_median_rad=float(np.median(expected)),
            unweighted_penalty_max_rad=float(expected.max()),
            hip_roll_share_of_mean=float(per_joint[roll].sum()/per_joint.sum()),
            per_joint_mean_rad=dict(zip(meta['joint_names'],per_joint.tolist())),
            max_function_recompute_error=float(np.max(np.abs(torch_result-expected)))))
        inputs[str(path.relative_to(ROOT))]=sha(path)
    installed=Path(inspect.getfile(JointPositionAction))
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        official_function_sha256=sha(source),script_sha256=sha(Path(__file__)),
        installed_action_source=str(installed),installed_action_sha256=sha(installed),
        robot_xml_sha256=sha(xmlpath),observer_metadata_sha256=sha(snapshot/'live-status.json'),
        archived_training_config_sha256=sha(snapshot/'env.yaml'),
        walking_reward_currently_enabled=False,installed_action_getters_exercised=True,
        recorded_joint_order_and_radian_limits_verified=True,overshoot_rad=0.3,
        penalty_weight_selected=False,inputs_sha256=inputs,results=rows,
        scope='Saved evaluation frames, correlated within episodes; unweighted target excess, not measured training reward or learning improvement',
        actual_environment_constructed=False,simulation_started=False,training_started=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs_sha256','results')},indent=2))
    print(json.dumps([dict(condition=r['condition'],mean_rad=r['unweighted_penalty_mean_rad'],hip_roll_share=r['hip_roll_share_of_mean']) for r in rows],indent=2))

if __name__=='__main__':main()

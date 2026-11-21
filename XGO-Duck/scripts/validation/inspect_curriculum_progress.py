"""Audit official curriculum scalars from a frozen continuation snapshot.

Read saved YAML, log and TensorBoard bytes only; never import the environment
or run a simulator. The purpose is to explain changing training conditions,
not to assign the cause of reward dips or to certify gait quality. Logged
scalars can average values across resets during a transition iteration.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(stage, key):
    value = stage[key]
    if key == 'ranges':
        return max(abs(float(v)) for pair in value for v in pair)
    return float(value)


def active_scalar(counter, thresholds, values, comparison):
    eligible = [index for index,threshold in enumerate(thresholds)
                if counter > threshold or (comparison == '>=' and counter == threshold)]
    return values[eligible[-1] if eligible else 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_directory', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    evidence = args.evidence_directory.resolve()
    check = json.loads((evidence/'check.json').read_text())
    record = root/'records/experiments'/check['run_id']
    assert evidence.parent == record/'resume-verification'
    assert check['status'] == 'passed'
    for name, expected in check['input_hashes'].items():
        assert sha(evidence/name) == expected, name
    plan = json.loads((record/'resume-plan.json').read_text())
    log_dir = Path(check['log_dir'])
    assert log_dir.name.endswith('_'+check['run_id'])
    params_file = log_dir/'params/env.yaml'
    params = yaml.load(params_file.read_text(), Loader=yaml.BaseLoader)
    work = Path.home()/'XGO-Duck/workspace/xgoduck_rl'
    mdp_file = work/'src/mjlab_microduck/tasks/mdp.py'
    official = root/'official_resources/xgoduck_rl/src/mjlab_microduck/tasks/mdp.py'
    # A copied archive and Linux checkout can use different line endings.
    assert mdp_file.read_text() == official.read_text(), 'Curriculum source differs from archive'
    definitions = {
        'action_rate_weight': ('weight_stages', 'weight', 'reward_weight', '>'),
        'standing_envs': ('standing_stages', 'rel_standing_envs', 'standing_envs_curriculum', '>'),
        'head_pose_range': ('range_stages', 'ranges', 'pose_command_range_curriculum', '>='),
        'body_pose_range': ('range_stages', 'ranges', 'pose_command_range_curriculum', '>='),
        'com_range': ('range_stages', 'range', 'com_range_curriculum', '>'),
        'head_com_range': ('range_stages', 'range', 'com_range_curriculum', '>'),
        'head_pose_bias_weight': ('weight_stages', 'weight', 'reward_weight', '>'),
    }
    assert set(params['curriculum']) == set(definitions)
    ea = EventAccumulator(str(evidence/'tensorboard'), size_guidance={'scalars': 0}).Reload()
    blocks = re.split(r'Learning iteration\s+(\d+)/\d+', (evidence/'training-prefix.log').read_text())
    logs = {int(blocks[i]): blocks[i+1] for i in range(1,len(blocks),2)}
    labels = list(range(check['first_iteration'],check['last_verified_iteration']+1))
    rows = {label: {'iteration_label': label, 'completed_updates': label+1,
                    'counter_at_update_end': (label+1)*plan['num_steps_per_env']}
            for label in labels}
    summaries = {}
    for name, (stages_key,value_key,function,comparison) in definitions.items():
        cfg = params['curriculum'][name]
        stages = cfg['params'][stages_key]
        thresholds = [int(stage['step']) for stage in stages]
        assert thresholds == sorted(set(thresholds)) and thresholds[0] == 0
        expected = [scalar(stage,value_key) for stage in stages]
        tag = 'Curriculum/'+name
        values = {event.step:event.value for event in ea.Scalars(tag) if event.step in rows}
        assert sorted(values) == labels
        for label,value in values.items():
            match = re.search(re.escape(tag)+r':\s+([-+\d.eE]+)', logs[label])
            assert match and math.isfinite(value)
            assert abs(float(match[1])-value) <= 0.000051, (name,label)
            # A transition may be averaged. Check its scheduled endpoints,
            # rather than permitting any scalar from any future stage.
            start = label*plan['num_steps_per_env']
            end = (label+1)*plan['num_steps_per_env']
            bounds = [active_scalar(counter,thresholds,expected,comparison) for counter in (start,end)]
            assert min(bounds)-1e-6 <= value <= max(bounds)+1e-6, (name,label,'Inspect reset timing or stage mismatch')
            rows[label][name] = value
        final_counter = rows[labels[-1]]['counter_at_update_end']
        final_expected = active_scalar(final_counter,thresholds,expected,comparison)
        assert abs(values[labels[-1]]-final_expected) <= 1e-6, (name,'last value does not match stage')
        summaries[name] = dict(function=function, threshold_comparison=comparison,
            configured_stages=[dict(step=t, completed_updates_boundary=t/plan['num_steps_per_env'],
                logged_scalar=v, original_parameters=stage,
                first_matching_log_label=next((label for label in labels if abs(values[label]-v)<=1e-6),None))
                for t,v,stage in zip(thresholds,expected,stages)],
            first_observed=values[labels[0]], last_observed=values[labels[-1]],
            final_expected_scalar=final_expected,
            mixed_transition_iterations=[label for label in labels
                if all(abs(values[label]-v)>1e-6 for v in expected)])
    out = evidence/'curriculum'
    out.mkdir(exist_ok=True)
    shutil.copy2(params_file,out/'env.yaml')
    source = mdp_file.read_text()
    needed = {definition[2] for definition in definitions.values()}
    functions = [node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in needed]
    assert {node.name for node in functions} == needed
    excerpts = '\n\n'.join(ast.get_source_segment(source,node) for node in functions)+'\n'
    (out/'curriculum-functions.py').write_text(excerpts)
    manager_file = Path(sys.prefix)/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/mjlab/envs/manager_based_rl_env.py'
    manager_source = manager_file.read_text()
    manager_class = next(node for node in ast.parse(manager_source).body
                         if isinstance(node,ast.ClassDef) and node.name == 'ManagerBasedRlEnv')
    methods = [node for node in manager_class.body
               if isinstance(node,ast.FunctionDef) and node.name in ('step','_reset_idx')]
    assert len(methods) == 2
    (out/'env-reset-order.py').write_text('\n\n'.join(ast.get_source_segment(manager_source,node) for node in methods)+'\n')
    with (out/'curriculum.csv').open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[labels[0]]))
        writer.writeheader()
        writer.writerows(rows.values())
    result = dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        run_id=check['run_id'],evidence_directory=str(evidence),
        verified_iterations=len(labels),first_label=labels[0],last_label=labels[-1],
        scalars_checked=len(labels)*len(definitions),num_steps_per_env=plan['num_steps_per_env'],
        sources={str(p):sha(p) for p in (params_file,mdp_file,official,manager_file,evidence/'check.json')},
        outputs={p.name:sha(p) for p in (out/'env.yaml',out/'curriculum-functions.py',out/'env-reset-order.py',out/'curriculum.csv')},
        stage_endpoint_bounds_verified=True,
        curriculum=summaries,training_parameters_changed=False,
        reward_dip_causality_verified=False,stable_gait_verified=False,
        interpretation='Actual logged curriculum changes match the configured stages. Reset-time changes and averaged transition scalars prevent exact instant inference. Reward is measured under changing conditions; no causal or gait claim.')
    (out/'check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    shutil.copy2(__file__,out/Path(__file__).name)
    print(json.dumps({k:result[k] for k in ('status','verified_iterations','scalars_checked','first_label','last_label')},indent=2))
    print(json.dumps({name:{'first':value['first_observed'],'last':value['last_observed'],
        'stage_first_labels':[stage['first_matching_log_label'] for stage in value['configured_stages']],
        'mixed_iterations':value['mixed_transition_iterations']} for name,value in summaries.items()},indent=2))
    print(out/'check.json')


if __name__ == '__main__':
    main()

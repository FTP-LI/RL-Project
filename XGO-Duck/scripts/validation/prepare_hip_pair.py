"""Freeze one bounded diagnostic and inspect real configurations on CPU.

No environment, optimizer step, inference or CUDA context is created. The
first-action comparison is performed again inside the actual visible runs.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import yaml
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from guarded_training import prepare,sha,WORK
from training_watch import read,write,run_name

OUT=ROOT/'records/reports/s2-hip-repair'
PAIR=ROOT/'configs/experiments/hip_limit_pair.json'


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'progress.json').exists(),'Do not revise an attempted pair'
    decision=read(ROOT/'records/reports/s1-evaluation/next-repair-decision.json')
    gate=ROOT/'records/reports/s1-integration/completion-gate.json'
    assert read(gate)['bounded_learning_allowed']
    sources=['scripts/training/'+name for name in ('guarded_training.py','training_watch.py',
        'hip_limit_repair.py','head_command_start.py','resume_state.py','head_command_bounds.py','display_preferences.py')]
    sources += ['scripts/validation/'+name for name in ('prepare_hip_pair.py','run_hip_pair.py',
        'analyze_hip_pair.py','prepare_policy_export.py','verify_export_trace.py','validate_export.py',
        'export_and_validate.sh','run_fixed_evaluation.sh','evaluate_fixed_policy.py',
        'verify_fixed_evaluation.py','analyze_walking_suite.py','s1_observer_integration.py')]
    # A failed CPU preparation may be corrected before any run, preserving its
    # original frozen inputs. Once progress exists no implicit re-freeze occurs.
    if PAIR.exists():
        previous=OUT/('protocol-before-preparation-'+datetime.now().strftime('%H%M%S')+'.json')
        write(previous,read(PAIR))
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S')
    pair=dict(status='frozen',frozen_at=datetime.now(timezone.utc).isoformat(),run_stamp=stamp,
        parent_sha256=decision['parent_sha256'],source_sha256={file:sha(ROOT/file) for file in sources},
        arms=['control','candidate'],maximum_updates_per_arm=50,automatic_budget_extension=False,
        maximum_total_learning_updates=100,only_config_difference='rewards.hip_limit_proximity.weight: 0 -> -2',
        primary=dict(condition='forward',metric='equal-episode bilateral inward-over20deg fraction after first second',
            minimum_absolute_reduction=.10,comparisons=['paired-control','parent']),
        regression=dict(commands='all seven frozen development commands',minimum_survival_fraction=.95,
            survival_count_may_decrease=False,max_xy_error_increase_m_s=.01,max_yaw_error_increase_rad_s=.05,
            max_linear_signed_speed_drop_m_s=.01,max_turn_signed_speed_drop_rad_s=.05,stand_must_remain_passing=True),
        decision='Continue only after diagnostic posture improvement and all regression checks; no stable-walk claim from this single seed',
        evaluation_protocol_sha256=sha(ROOT/'configs/experiments/walking_reference_suite.json'),
        lateral_protocol_sha256=sha(ROOT/'configs/experiments/walking_strafe_supplement.json'))
    write(PAIR,pair)
    base=read(ROOT/'configs/experiments/next_training.json')
    for key in ('launch_blockers','stage_evidence','paired_control','matched_control_record','paired_control_sha256'):base.pop(key,None)
    base.update(status='frozen',scope='hip-limit-paired-diagnostic',gate_record=str(gate),gate_sha256=sha(gate),
        run_stamp=stamp,pair_protocol=str(PAIR),pair_protocol_sha256=sha(PAIR),visuals_enabled=True,
        repair_monitor={'inherited_faults':['inward','crossed'],'review_after_updates':50})
    cfgs={};plans={};checks=[]
    for arm in pair['arms']:
        plan=dict(base,arm=arm,reward_edits={'hip_limit_proximity':0. if arm=='control' else -2.})
        path=ROOT/f'configs/experiments/hip_limit_{arm}.json'
        if arm=='candidate':
            plan.update(matched_control_record=str(ROOT/'records/experiments'/run_name('walk','hip-limit','control',42,stamp)),
                paired_control_sha256=sha(ROOT/'configs/experiments/hip_limit_control.json'))
        write(path,plan);plans[arm]=plan
        cfg,watch,resume,changes,display=prepare(plan,'hip_pair_config_check')
        cfgs[arm]=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader)
        write(OUT/f'{arm}-preflight.json',dict(status='passed',resume=resume,changes=changes,display=display,
            protocol_sha256=sha(path),parent_sha256=plan['parent_sha256']))
    left,right=deepcopy(cfgs['control']),deepcopy(cfgs['candidate'])
    assert float(left['env']['rewards']['hip_limit_proximity']['weight'])==0.
    assert float(right['env']['rewards']['hip_limit_proximity']['weight'])==-2.
    right['env']['rewards']['hip_limit_proximity']['weight']=left['env']['rewards']['hip_limit_proximity']['weight']
    assert left==right,'Undeclared between-arm configuration difference'
    checks.append('All TrainConfig fields equal except one declared reward weight')
    for field,value in [('additional_updates',51),('reward_edits',{'hip_limit_proximity':-4}),
                        ('parent_sha256','0'*64),('pair_protocol_sha256','0'*64),('status','draft')]:
        bad=deepcopy(plans['candidate']);bad[field]=value
        try:prepare(bad,'rejected-config-check')
        except ValueError:checks.append('Rejected '+field)
        else:raise AssertionError('Invalid protocol accepted: '+field)
    # Independent scalar examples verify the installed official function and
    # joint selection, including asymmetry and an unrelated third joint.
    from types import SimpleNamespace as N
    from mjlab_microduck.tasks.mdp import joint_pos_limit_proximity
    joint=torch.tensor([[0.,0.,9.],[-.4,.4,9.],[-.3,.2,9.]],dtype=torch.float32)
    limits=torch.tensor([[[-.4,.4]]*3]*3)
    env=N(scene={'robot':N(data=N(joint_pos=joint,joint_pos_limits=limits))})
    cost=joint_pos_limit_proximity(env,N(name='robot',joint_ids=[0,1]),margin=.15)
    torch.testing.assert_close(cost,torch.tensor([0.,.30,.05]),atol=1e-7,rtol=1e-6)
    checks.append('Installed hard-limit function matches independent center/stop/asymmetric examples; other joint excluded')
    assert not torch.cuda.is_initialized()
    write(OUT/'preflight.json',dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        checks=checks,source_sha256=pair['source_sha256'],pair_sha256=sha(PAIR),
        protocols={arm:sha(ROOT/f'configs/experiments/hip_limit_{arm}.json') for arm in pair['arms']},
        live_first_action_comparison_pending=True,training_started=False))
    write(ROOT/'configs/experiments/next_training.json',plans['control'])
    print(json.dumps(dict(status='passed',checks=checks,run_stamp=stamp,training_started=False),indent=2))


if __name__=='__main__':main()

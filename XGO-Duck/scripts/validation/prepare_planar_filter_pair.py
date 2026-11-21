"""CPU preflight and immutable protocol for the planar-reward-statistic pair.

Both arms perform/reset the same EMA; only the returned XY reward statistic
differs. Freeze the progress test before learning and keep legacy gait flags.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from training_watch import read,run_name
from guarded_training import sha
from record_io import write
from planar_filter_training import prepare,OUT,PAIR


def main():
    torch.set_num_threads(2)
    assert not PAIR.exists(),'Never replace a frozen experiment'
    for file,source_key,target in (
        ('reward-check.json','verifier_sha256','scripts/validation/verify_planar_velocity_reward.py'),
        ('reward-check.json','source_sha256','scripts/training/planar_velocity_reward.py'),
        ('block-metric-check.json','script_sha256','scripts/validation/verify_planar_block_metrics.py'),
        ('block-metric-check.json','metric_sha256','scripts/validation/planar_block_metrics.py')):
        checked=read(OUT/file)
        assert checked['status']=='passed' and checked[source_key]==sha(ROOT/target),(file,source_key)
    assert read(ROOT/'records/reports/official-actor-bootstrap/training-verification.json')['status']=='passed'
    previous=ROOT/'records/experiments/duck_walk_official-actor-bootstrap_candidate_s42_20261102T180647'
    parent=read(previous/'last-checkpoint.json')
    assert parent['checkpoint_sha256']=='980d5797b544a76c0cd3bc8554ecfb5d1ce34dc1aa251ed7693a276de5fc2caf'
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S')
    training=('planar_filter_training.py','planar_velocity_reward.py','official_bootstrap.py','official_actor.py',
        'guarded_training.py','training_watch.py','resume_state.py','head_command_start.py',
        'hip_limit_repair.py','record_io.py','display_preferences.py','head_command_bounds.py','live_observer.py')
    validation=('prepare_planar_filter_pair.py','run_planar_filter_pair.py','verify_planar_filter_training.py',
        'evaluate_planar_filter.py','analyze_planar_filter_pair.py','planar_block_metrics.py',
        'verify_planar_velocity_reward.py','verify_planar_block_metrics.py','evaluate_official_bootstrap.py',
        'evaluate_fixed_policy.py','evaluation_metrics.py','verify_fixed_evaluation.py','s1_reference_inputs.py',
        's1_observer_integration.py')
    sources=['scripts/training/'+x for x in training]+['scripts/validation/'+x for x in validation]
    hashes={name:sha(ROOT/name) for name in sources}
    review=dict(status='reviewed-new-hypothesis',prior_protocols_closed=True,
        hypothesis='World-frame EMA with tau 0.2 seconds may discount periodic lateral sway without hiding persistent drift',
        observed_cyclic_sway_is_not_proof_of_cause=True,actor_inputs_unchanged=True,
        raw_body_z_penalty_retained=True,causal_claim_pending=True,automatic_extension=False,
        reward_check_sha256=sha(OUT/'reward-check.json'),block_check_sha256=sha(OUT/'block-metric-check.json'))
    write(OUT/'review.json',review)
    screen=read(ROOT/'configs/experiments/official_adaptation_screen.json')
    screen['source_sha256']={name:hashes[name] for name in (
        'scripts/validation/evaluate_official_bootstrap.py','scripts/validation/evaluate_planar_filter.py',
        'scripts/validation/evaluate_fixed_policy.py','scripts/validation/evaluation_metrics.py',
        'scripts/validation/verify_fixed_evaluation.py')}
    screen.pop('model');screen.pop('model_sha256')
    pair=dict(status='frozen',frozen_at=datetime.now(timezone.utc).isoformat(),stamp=stamp,
        arms={arm:f'configs/experiments/planar_filter_{arm}.json' for arm in ('control','candidate')},
        source_sha256=hashes,maximum_total_updates=100,updates_per_arm=50,
        only_difference='env.rewards.track_linear_velocity.params.use_filtered_xy: False -> True',
        common_adaptation='Both arms compute world-vector EMA, reset it per episode and retain raw body-Z penalty',
        tau_seconds=.2,review_sha256=sha(OUT/'review.json'),automatic_extension=False,evaluation_template=screen,
        screen=dict(commands=['stop','forward','backward'],episodes_per_command=32,seconds=20,seed=1001),
        block_metric=dict(block_steps=50,warmup_steps=50,missing_penalty=.5,
            definition='Equal episode mean of 19 one-second body-XY vector errors after first second; incomplete blocks penalized'),
        primary=dict(block_error_reduction_min=.005,signed_speed_gain_min=.01,
            comparison='At least one moving command improves BOTH block error and signed speed against matched control AND parent'),
        regression=dict(max_block_error_increase=.01,max_signed_speed_drop=.01,max_yaw_error_increase=.05,
            survival_min=.95,inward_fraction_max=.01,stand_must_pass=True),
        promotion='Development progress only; original instantaneous flags retained; four-command regression and independent confirmation required')
    write(PAIR,pair)
    cfgs={};protocols={};runs={}
    for arm in ('control','candidate'):
        name=run_name('walk','planar-filter',arm,42,stamp)
        plan=dict(status='frozen',task='Mjlab-Velocity-Flat-XgoDuck',arm=arm,stamp=stamp,
            additional_updates=50,use_filtered_xy=arm=='candidate',tau_seconds=.2,
            automatic_extension=False,source_sha256=hashes,checkpoint=parent['checkpoint'],
            checkpoint_sha256=parent['checkpoint_sha256'],parent_sha256=parent['checkpoint_sha256'])
        if arm=='candidate':
            plan.update(matched_control_record=str(ROOT/'records/experiments'/runs['control']),
                        paired_control_sha256=protocols['control'])
        path=ROOT/pair['arms'][arm];write(path,plan)
        cfg,display,resume=prepare(plan)
        cfgs[arm]=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader)
        protocols[arm]=sha(path);runs[arm]=name
        write(OUT/f'{arm}-preflight.json',dict(status='passed',resume=resume,display=display,
            protocol_sha256=sha(path),run_id=name,config=cfgs[arm]))
    left,right=deepcopy(cfgs['control']),deepcopy(cfgs['candidate'])
    right['agent']['run_name']=left['agent']['run_name']
    right['env']['rewards']['track_linear_velocity']['params']['use_filtered_xy']=left['env']['rewards']['track_linear_velocity']['params']['use_filtered_xy']
    assert left==right,'Undeclared between-arm difference'
    assert not torch.cuda.is_initialized()
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PAIR),protocols=protocols,runs=runs,
        only_declared_config_difference=True,cpu_reward_and_reset_checks=True,cpu_block_checks=True,
        parent_sha256=parent['checkpoint_sha256'],training_started=False,live_pair_matching_pending=True,cuda_initialized=False))
    print(json.dumps(read(OUT/'preflight.json'),indent=2))


if __name__=='__main__':main()

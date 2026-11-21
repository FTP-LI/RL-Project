"""Freeze the velocity-reward-width pair after a failed ordinary adaptation.

CPU-only: verify the actual task, parent tensors, single config difference and
installed reward formula. Live matching is repeated before physics in each arm.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as N
import json
import sys
import numpy as np
import torch
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/training'))
from training_watch import read, run_name
from guarded_training import sha
from record_io import write
from tracking_precision import prepare, OUT, PAIR


def main():
    torch.set_num_threads(2)
    assert not PAIR.exists(), 'Keep frozen protocols; review explicitly before a retry'
    decision = ROOT/'records/reports/official-adaptation/screen-analysis.json'
    assert read(decision)['decision'] == 'end-direct-continuation-route'
    boot = ROOT/'records/reports/official-actor-bootstrap/training-verification.json'
    assert read(boot)['status'] == 'passed'
    previous = ROOT/'records/experiments/duck_walk_official-actor-bootstrap_candidate_s42_20261102T180647'
    parent = read(previous/'last-checkpoint.json')
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    sources = ['scripts/training/'+x for x in ('tracking_precision.py','official_bootstrap.py','official_actor.py',
        'guarded_training.py','training_watch.py','resume_state.py','head_command_start.py',
        'hip_limit_repair.py','record_io.py','display_preferences.py','head_command_bounds.py','live_observer.py')]
    sources += ['scripts/validation/'+x for x in ('prepare_tracking_precision.py','run_tracking_precision_pair.py',
        'verify_tracking_precision.py','evaluate_tracking_precision.py','analyze_tracking_precision.py',
        'evaluate_official_bootstrap.py','evaluate_fixed_policy.py','evaluation_metrics.py','verify_fixed_evaluation.py',
        'analyze_official_bootstrap.py')]
    hashes = {x:sha(ROOT/x) for x in sources}
    OUT.mkdir(exist_ok=True)
    review = dict(status='reviewed-new-hypothesis', prior_direct_continuation_closed=True,
        prior_result_sha256=sha(decision), parent_evidence_sha256=sha(boot),
        hypothesis='Reduce velocity-reward width to penalize slow/stationary responses more strongly; do not change command sampling or weight',
        parent_choice='Earlier official bootstrap: reverse motion and forward speed better than final adaptation',
        causal_claim_pending=True, automatic_extension=False)
    write(OUT/'review.json', review)
    screen=read(ROOT/'configs/experiments/official_adaptation_screen.json')
    screen['source_sha256']={k:v for k,v in hashes.items() if k in (
        'scripts/validation/evaluate_official_bootstrap.py','scripts/validation/evaluate_tracking_precision.py',
        'scripts/validation/evaluate_fixed_policy.py','scripts/validation/evaluation_metrics.py',
        'scripts/validation/verify_fixed_evaluation.py')}
    # Model identity is filled from each verified final checkpoint after learning.
    screen.pop('model');screen.pop('model_sha256')
    pair = dict(status='frozen', frozen_at=datetime.now(timezone.utc).isoformat(), stamp=stamp,
        arms={arm:f'configs/experiments/tracking_precision_{arm}.json' for arm in ('control','candidate')},
        source_sha256=hashes, maximum_total_updates=100, updates_per_arm=50,
        only_difference='env.rewards.track_linear_velocity.params.std: 0.25 -> 0.10 m/s',
        review_sha256=sha(OUT/'review.json'), automatic_extension=False,evaluation_template=screen,
        screen=dict(commands=['stop','forward','backward'], episodes_per_command=32, seconds=20, seed=1001),
        primary=dict(error_reduction_min=.005, signed_speed_gain_min=.01,
            comparison='At least one moving command improves BOTH error and signed speed against matched control AND parent'),
        regression=dict(max_linear_error_increase=.01, max_signed_speed_drop=.01, max_yaw_error_increase=.05,
            survival_min=.95, inward_fraction_max=.01, stand_must_pass=True),
        promotion='Screen only; complete remaining four-command regression before extension or stage acceptance')
    write(PAIR, pair)
    cfgs = {}; protocols = {}; runs = {}
    for arm in ('control','candidate'):
        name = run_name('walk','velocity-precision',arm,42,stamp)
        plan = dict(status='frozen', task='Mjlab-Velocity-Flat-XgoDuck', arm=arm, stamp=stamp,
            additional_updates=50, velocity_reward_std=.25 if arm=='control' else .10,
            automatic_extension=False, source_sha256=hashes, checkpoint=parent['checkpoint'],
            checkpoint_sha256=parent['checkpoint_sha256'], parent_sha256=parent['checkpoint_sha256'])
        if arm=='candidate':
            plan.update(matched_control_record=str(ROOT/'records/experiments'/runs['control']),
                        paired_control_sha256=protocols['control'])
        path = ROOT/pair['arms'][arm]; write(path,plan)
        cfg, display, resume = prepare(plan)
        cfgs[arm] = yaml.load(yaml.dump(asdict(cfg)), Loader=yaml.BaseLoader)
        protocols[arm] = sha(path); runs[arm] = name
        write(OUT/f'{arm}-preflight.json',dict(status='passed',resume=resume,display=display,
            protocol_sha256=sha(path),run_id=name,config=cfgs[arm]))
    left, right = deepcopy(cfgs['control']), deepcopy(cfgs['candidate'])
    right['env']['rewards']['track_linear_velocity']['params']['std'] = left['env']['rewards']['track_linear_velocity']['params']['std']
    right['agent']['run_name'] = left['agent']['run_name']
    assert left == right, 'Undeclared between-arm difference'
    # Exercise the installed reward, including its z-velocity penalty.
    func = prepare(read(ROOT/pair['arms']['control']))[0].env.rewards['track_linear_velocity'].func
    command = torch.tensor([[.15,0,0],[-.1,0,0],[.15,0,0]], dtype=torch.float32)
    actual = torch.tensor([[0.,0,0],[0.,0,0],[.15,0,.1]], dtype=torch.float32)
    env = N(scene={'robot':N(data=N(root_link_lin_vel_b=actual))}, command_manager=N(get_command=lambda _:command))
    reward = {}
    for width in (.25,.10):
        value = func(env, std=width, command_name='twist')
        expected = np.exp(-np.array([.15**2,.1**2,.1**2])/width**2)
        np.testing.assert_allclose(value.numpy(), expected, atol=1e-7, rtol=1e-6)
        reward[str(width)] = value.tolist()
    assert not torch.cuda.is_initialized()
    write(OUT/'preflight.json', dict(status='passed', protocol_sha256=sha(PAIR), protocols=protocols,
        runs=runs, only_declared_config_difference=True, reward_formula_verified=True,
        normalized_reward_examples=reward, example_cases=['stationary-forward','stationary-backward','tracking-xy-but-vz0.1'],
        training_started=False, live_pair_matching_pending=True, cuda_initialized=False))
    print(json.dumps(read(OUT/'preflight.json'),indent=2))


if __name__=='__main__':main()

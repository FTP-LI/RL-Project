"""Freeze and evaluate the getup correction on exact shared releases."""
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
import evaluate_getup_early_support_ppo as evaluator
import getup_shared_release as shared
from getup_shared_release_terrain import cpu_scene_readback
from getup_upright_intervention import Trigger, pulse_request, run_collection
from record_io import write
from standup_metrics import classify_release, projected_gravity


PLAN = ROOT/'configs/experiments/getup_upright_recovery_pair.json'
DRAFT = PLAN.with_name('getup_upright_recovery_pair_draft.json')
OUT = ROOT/'records/reports/getup-upright-recovery-pair'
ROLES = ('reference-a','reference-b','bounded-legs')


def page(seed):
    return ROOT/f'records/getup-upright-recovery-s{seed}-live.html'


def fixture(trigger_proof,entry):
    windows = 0
    for name,cohort in trigger_proof['cohorts'].items():
        source = ROOT/'records/reports'/name/'analysis.json'
        proof = read(source)
        assert sha(source)==cohort['source_sha256']
        record = ROOT/cohort['record']
        prefix,branch = record/'trace-shared-prefix.npz',record/'trace-fixed-weighted.npz'
        assert sha(prefix)==proof['prefix_sha256'] and sha(branch)==proof['arms']['fixed-weighted']['trace_sha256']
        p = read(record/'protocol.json')['environment']
        with np.load(prefix) as data:
            pose = classify_release(projected_gravity(data['quaternion'][-1]),p['criteria']['release_alignment_min'])
        detector = Trigger(pose,entry)
        with np.load(branch) as data:
            q,feet,head = (data[key] for key in ('quaternion','foot_contact','native_head_leg_contact'))
            for k in range(101,round(entry['last_trigger_seconds']/.02)+1):
                detector.observe(k,q[k-101],feet[k-101],head[k-101])
        expected = cohort['results']['persistent_contact_and_low_rotation_progress']['trigger_time_seconds']
        np.testing.assert_array_equal(detector.step,[round(t/.02) if t is not None else -1 for t in expected])
        windows += len(pose)
    raw = torch.tensor([-1.2,-.97,0.,.96,1.1])
    delta = torch.tensor([.03,-.08,.02,.08,-.08])
    torch.testing.assert_close(pulse_request(raw,delta,torch.tensor(-1.),torch.tensor(1.)),
        torch.tensor([-1.2,-1.,.02,1.,1.1]),atol=0,rtol=0)
    return dict(offline_episode_triggers_exact=windows,project_only_baseline_legal=True)


def freeze():
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    draft = read(DRAFT)
    assert draft['status']=='draft-not-executable' and tuple(draft['roles'])==ROLES
    source,geometry,targets = (ROOT/draft[key] for key in ('source_trigger','source_geometry','source_actuator_projection'))
    trigger,geometric,actuator = read(source),read(geometry),read(targets)
    assert trigger['status']=='passed-saved-trigger-coverage' and not trigger['heldout_used']
    assert geometric['status']=='passed-saved-trigger-geometry-review' and not geometric['heldout_used']
    assert actuator['status']=='passed-static-target-projection' and not actuator['heldout_used']
    assert [r['feasible_original_failures'] for r in geometric['cohorts'].values()]==[5,7]
    assert sum(r['feasible_original_successes'] for r in geometric['cohorts'].values())==1
    weights = actuator['weights']
    assert len(weights)==draft['target']['controls']==22 and weights[0]==weights[-1]==0.
    assert all(not draft[k] for k in ('heldout_used','hardware_io','learning_updates','automatic_retry','automatic_extension'))
    preflight = fixture(trigger,draft['entry'])
    base_path = ROOT/draft['base_protocol']
    base = read(base_path)
    stages = [ROOT/'scripts/validation'/name for name in ('getup_upright_intervention.py',
              'verify_getup_upright_recovery.py','check_getup_supine_upright_geometry.py')]
    fixed = ROOT/base['environment']['model']
    assert sha(fixed)==base['environment']['model_sha256']
    paths = [DRAFT,base_path,source,geometry,targets,ROOT/draft['geometry_protocol'],Path(__file__),fixed,*stages]
    specs,scene_checks = {},{}
    for seed in draft['development_seeds']:
        assert not any(f'_s{seed}_' in path.name for path in (ROOT/'records/evaluations').iterdir())
        spec = deepcopy(base)
        p = spec['environment']
        codes = np.concatenate(([1],np.random.default_rng(seed).permutation(np.repeat(np.arange(4),[32,31,128,128]))))
        assert len(codes)==p['num_envs']==draft['worlds_per_seed']
        groups = {pose:np.flatnonzero(codes==i).tolist() for i,pose in enumerate(shared.POSES)}
        p.update(seed=seed,stage=draft['stage'],split=dict(pose_codes=codes.tolist(),groups=groups),
            run_id=f'duck_getup_upright-recovery_paired_s{seed}_{datetime.now(evaluator.TZ):%Y%m%dT%H%M%S}')
        policies = {role:dict(model=p['model'],model_sha256=p['model_sha256'],seed=seed) for role in ROLES}
        spec.update(stage=draft['stage'],branches=list(ROLES),policies=policies,observation=page(seed).relative_to(ROOT).as_posix(),
            cameras={role:{pose:groups[pose][0] for pose in shared.POSES} for role in ROLES},
            intervention=dict(geometry_protocol=draft['geometry_protocol'],entry=draft['entry'],target=draft['target'],
                              weights=weights,gate=draft['gate']),
            choice='One bounded correction and two identical frozen-policy references. Preserve all original pose and stable holds.')
        spec['references'].update({path.relative_to(ROOT).as_posix():sha(path) for path in paths})
        assert spec['prefix_controls']==draft['shared_release_controls']==100
        assert p['maximum_control_steps']==draft['episode_controls']==550
        assert spec['maximum_unique_controls']==draft['maximum_controls_per_seed']==1450
        assert spec['maximum_control_transitions']==draft['maximum_transitions_per_seed']==464000
        assert spec['maximum_physics_substeps']==draft['maximum_physics_substeps_per_seed']==7250
        shared.assert_sources(spec)
        scene_checks[str(seed)] = cpu_scene_readback(p,spec)
        specs[str(seed)] = spec
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN,dict(status='frozen-before-dynamics',stage=draft['stage'],seeds=draft['development_seeds'],
        specs=specs,second_seed_only_after_first_pass=True,learning_updates=0,heldout_used=False,hardware_io=False))
    write(OUT/'preflight.json',dict(status='passed-cpu-zero-step',protocol_sha256=sha(PLAN),
        physical_substeps=0,scene=scene_checks,fixture=preflight))
    print('GETUP_UPRIGHT_RECOVERY_FROZEN',sha(PLAN),list(specs),flush=True)


def checked(seed, requires_project=False):
    plan = read(PLAN)
    assert plan['status']=='frozen-before-dynamics' and not plan['heldout_used']
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(PLAN)
    spec = plan['specs'][str(seed)]
    shared.assert_sources(spec)
    if seed!=plan['seeds'][0]:
        prior = read(OUT/f'seed{plan["seeds"][0]}-analysis.json')
        assert prior['selection']['passed_paired_development_gate']
    if requires_project:
        assert read(OUT/'windows-entry.json')['protocol_sha256']==sha(PLAN)
    return spec,ROOT/'records/evaluations'/spec['environment']['run_id']


def sample(seed):
    spec,record = checked(seed,requires_project=True)
    assert not record.exists()

    def collect(env,actors,memory,meta,spec,path,state,publish,check_active):
        (path/'README.md').write_text('# 仰卧转正配对试验\n\n三路从同一物理释放开始，两路原策略作为参照。\n\n'
            f'[冻结方案](../../../configs/experiments/{PLAN.name}) · [状态](evaluation.json) · [日志](execution.log)\n',
            encoding='utf-8')
        run_collection(env,actors,memory,meta,spec,path,state,publish,check_active)

    with (patch.multiple(evaluator,PLAN=PLAN,OUT=OUT,PAGE=page(seed),STAGE=spec['stage']),
          patch.object(evaluator.evaluation,'collect',collect)):
        evaluator.run(spec,record)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','sample'))
    parser.add_argument('seed',type=int,nargs='?')
    args = parser.parse_args()
    if args.command=='freeze':
        assert args.seed is None
        freeze()
    else:
        assert args.seed is not None
        sample(args.seed)

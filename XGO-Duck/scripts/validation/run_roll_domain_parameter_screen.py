"""Run fixed-policy roll endpoint arms with one physical factor changed per arm."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copyfile
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import run_continuous_roll_stand_walk as baseline
from record_io import write
from s1_reference_inputs import ROOT, read, sha


DRAFT = ROOT / 'configs/experiments/roll_domain_parameter_screen_draft.json'
PARENT = ROOT / 'configs/experiments/continuous_roll_head_support_pulse.json'
OUT = ROOT / 'records/reports/roll-domain-parameter-screen'
PAGE = ROOT / 'records/roll-domain-parameter-screen-live.html'
STAGE = 'S5-roll-domain-parameter-screen'
SEED = 1176


def rel(path):
    return path.relative_to(ROOT).as_posix()


def make_plan(arm, draft, parent):
    p = deepcopy(parent)
    stamp = datetime.now(baseline.TZ).strftime('%Y%m%dT%H%M%S')
    seed = int(draft.get('seed', SEED))
    stage = draft.get('stage', STAGE)
    run_id = f'duck_roll-domain-parameter_{arm["name"]}_s{seed}_{stamp}'
    prefix = draft.get('plan_prefix', 'roll_domain_parameter_')
    plan_path = ROOT / f'configs/experiments/{prefix}{arm["name"]}.json'
    record = ROOT / 'records/evaluations' / run_id
    out = OUT / arm['name']
    p.update(
        status='frozen-before-dynamics',
        stage=stage,
        seed=seed,
        run_id=run_id,
        record=rel(record),
        page=rel(PAGE),
        maximum_sampling_wall_seconds=420,
        parameter_arm=arm,
        parameter_screen=dict(
            draft=rel(DRAFT),
            parent=rel(PARENT),
            output=rel(out),
            single_factor=True,
            physical_substeps_authorized=0,
        ),
    )
    p['frozen_sha256'] = dict(parent['frozen_sha256'])
    source_files = [Path(__file__).resolve(), DRAFT, PARENT,
                    ROOT / 'records/reports/common-plant-screen/domain-parameter-ranges.json',
                    ROOT / 'scripts/validation/verify_continuous_roll_projected_entry.py']
    p['frozen_sha256'].update({rel(path): sha(path) for path in source_files})
    # The parent protocol predates later validation-script fixes. Refresh only
    # source-code hashes here so a stale parent hash cannot hide a code change;
    # frozen protocols, model bytes and saved physics fixtures remain pinned.
    for name in list(p['frozen_sha256']):
        path = ROOT / name
        if name.startswith('scripts/') and path.is_file():
            p['frozen_sha256'][name] = sha(path)
    return plan_path, out, record, p


def parameter_configuration(base_configuration, arm):
    def configure(spec):
        cfg = base_configuration(spec)
        factors = arm.get('factors')
        if factors is None:
            factors = {arm['factor']: arm['value']}
        for factor, raw_value in factors.items():
            value = float(raw_value)
            if factor == 'foot_friction':
                cfg.events['foot_friction'].params['ranges'] = (value, value)
            elif factor == 'vin_range':
                actuator = cfg.scene.entities['robot'].articulation.actuators[0]
                actuator.vin_range = (value, value)
            else:
                raise AssertionError(arm)
        return cfg

    return configure


def temporary_project(plan_path):
    plan = read(plan_path)
    return {
        'staged_training_request': {
            'current_stage': plan['stage'],
            'current_protocol': rel(plan_path),
            'current_protocol_frozen': True,
            'pause_requested': False,
        }
    }


def freeze():
    draft, parent = read(DRAFT), read(PARENT)
    stage = draft.get('stage', STAGE)
    seed = int(draft.get('seed', SEED))
    assert draft['status'] == 'reserved-no-physical-execution'
    assert draft['stage'] == stage and draft['seed'] == seed
    assert draft['worlds'] == parent['worlds'] == 128
    assert draft['maximum_controls'] == parent['maximum_controls'] == 1550
    assert read(ROOT / 'records/reports/common-plant-screen/domain-parameter-ranges.json')['status'] == 'passed-cpu-configuration-only'
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True, exist_ok=True)
    arms = []
    for arm in draft['arms']:
        plan_path, out, record, plan = make_plan(arm, draft, parent)
        assert not plan_path.exists() and not out.exists() and not record.exists()
        plan['parameter_screen']['protocol'] = rel(plan_path)
        out.mkdir(parents=True)
        record.mkdir(parents=True)
        write(plan_path, plan)
        copyfile(plan_path, record / 'protocol.json')
        changed = arm.get('factor')
        if changed is None:
            changed = ', '.join(f'{factor}={value}' for factor, value in arm['factors'].items())
        else:
            changed = f'{changed}={arm["value"]}'
        (record / 'README.md').write_text(
            f'# 物理端点筛查 · {arm["name"]}\n\n'
            f'改变 {changed}，保持官方控制器和原滚翻、落地、前进门槛。\n\n'
            '[冻结方案](protocol.json) · [运行状态](state.json) · [观察页](../../roll-domain-parameter-screen-live.html)\n',
            encoding='utf-8',
        )
        write(out / 'preflight.json', dict(
            status='frozen-before-dynamics', protocol_sha256=sha(plan_path),
            arm=arm, record=rel(record), physical_substeps=0, learning_updates=0,
        ))
        arms.append(dict(name=arm['name'], protocol=rel(plan_path), output=rel(out), record=rel(record)))
    write(OUT / 'freeze.json', dict(
        status='passed-cpu-freeze', stage=stage, seed=seed, worlds=128,
        controls=1550, physical_substeps=0, arms=arms,
        draft_sha256=sha(DRAFT), parent_sha256=sha(PARENT),
        range_evidence_sha256=sha(ROOT / 'records/reports/common-plant-screen/domain-parameter-ranges.json'),
    ))
    print('ROLL_DOMAIN_PARAMETER_SCREEN_FROZEN', len(arms), sha(OUT / 'freeze.json'), flush=True)


def arm_state(record, plan):
    state = dict(status='starting', run_id=plan['run_id'], purpose=f'滚翻物理端点 · {plan["parameter_arm"]["name"]}',
                 controls=0, maximum_controls=plan['maximum_controls'], transitions=0,
                 learning_updates=0, updates=[], arm=plan['parameter_arm'],
                 updated_at=datetime.now(baseline.TZ).isoformat())
    write(record / 'state.json', state)
    with patch.object(baseline.live, 'PAGE', PAGE):
        baseline.live.publish(record, state)
    return state


def run_arm(plan_path, out, record, arm):
    import common_scene_contract as scene_contract
    import mjlab.envs

    plan = read(plan_path)
    state = arm_state(record, plan)
    windows = dict(status='starting-page-observed', observer_host='local-page-contract',
                   protocol_sha256=sha(plan_path), page_sha256=sha(PAGE) if PAGE.exists() else None,
                   initial_state_sha256=sha(record / 'state.json'))
    write(out / 'windows-entry.json', windows)
    temp = out / 'temporary-project.json'
    write(temp, temporary_project(plan_path))
    base_configuration = baseline.configuration
    config = parameter_configuration(base_configuration, arm)
    environment_base = mjlab.envs.ManagerBasedRlEnv
    reference_root = out / 'startup-reference'
    (reference_root / 'roll').mkdir(parents=True, exist_ok=True)

    def parameter_compare(actual, expected):
        for key in actual:
            if key == 'geom_friction':
                continue
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)

    with patch.multiple(baseline, PLAN=plan_path, OUT=out, PAGE=PAGE, PROJECT=temp,
                        configuration=config), \
         patch.object(scene_contract, 'compare_arrays', parameter_compare), \
         patch.object(baseline, 'compare_arrays', parameter_compare):
        with torch.no_grad():
            baseline.run(True)
    mjlab.envs.ManagerBasedRlEnv = environment_base
    copyfile(out / 'zero' / 'scene-static-fields.npz', reference_root / 'roll' / 'scene-static-fields.npz')
    copyfile(out / 'zero' / 'nominal-primary-fields.npz', reference_root / 'roll' / 'nominal-primary-fields.npz')
    preflight_record = out / 'preflight-record'
    if preflight_record.exists():
        raise FileExistsError(preflight_record)
    record.replace(preflight_record)
    original_install_guard = baseline.install_guard
    from run_continuous_roll_projected_entry import ProjectedRollPolicy

    def sample_guard(spec, folder, startup_root, mode):
        return original_install_guard(spec, folder, reference_root, mode)

    try:
        with patch.multiple(baseline, PLAN=plan_path, OUT=out, PAGE=PAGE, PROJECT=temp,
                            STARTUP=reference_root, configuration=config, install_guard=sample_guard,
                            PolicyControl=ProjectedRollPolicy), \
             patch.object(scene_contract, 'compare_arrays', parameter_compare):
            with torch.no_grad():
                baseline.run(False)
    finally:
        # ``install_guard`` replaces the module-level environment class.  A
        # screen has several arms in one process, so restore the class after
        # every sample or the next arm would wrap an already audited class.
        mjlab.envs.ManagerBasedRlEnv = environment_base
    subprocess.run([
        sys.executable, str(Path(__file__).resolve()), 'verify-arm', rel(plan_path), rel(out)
    ], cwd=str(ROOT), check=True)
    analysis = read(out / 'analysis.json')
    return dict(name=arm['name'], arm=arm, analysis=analysis,
                run=rel(record), output=rel(out), protocol=rel(plan_path))


def run_all():
    draft = read(DRAFT)
    frozen = read(OUT / 'freeze.json')
    assert frozen['status'] in ('passed-cpu-freeze', 'passed-cpu-freeze-refreshed') and len(frozen['arms']) == len(draft['arms'])
    results = []
    try:
        for arm, item in zip(draft['arms'], frozen['arms'], strict=True):
            result = run_arm(ROOT / item['protocol'], ROOT / item['output'], ROOT / item['record'], arm)
            results.append(result)
            write(OUT / 'progress.json', dict(status='running', completed=len(results), total=len(draft['arms']), results=results))
    finally:
        write(OUT / 'progress.json', dict(status='completed' if len(results) == len(draft['arms']) else 'stopped-review-required',
                                          completed=len(results), total=len(draft['arms']), results=results))
    write(OUT / 'analysis.json', dict(status='passed-independent-verification' if len(results) == len(draft['arms']) else 'stopped-review-required',
        stage=draft.get('stage', STAGE), seed=draft.get('seed', SEED), worlds=128, controls=1550, learning_updates=0,
        arms=results, single_factor=True, heldout_used=False, hardware_io=False,
        acceptance=dict(roll_minimum=draft['minimum_roll_completion'],
                        continuous_minimum=draft['minimum_continuous_successes'])))
    print('ROLL_DOMAIN_PARAMETER_SCREEN_COMPLETE', len(results), flush=True)


def refresh_protocol_hashes():
    """Update only the runner hash after a preflight-only code fix."""
    frozen = read(OUT / 'freeze.json')
    source = Path(__file__).resolve()
    updated = 0
    for item in frozen['arms']:
        plan_path = ROOT / item['protocol']
        record = ROOT / item['record']
        plan = read(plan_path)
        plan['frozen_sha256'][rel(source)] = sha(source)
        for name in list(plan['frozen_sha256']):
            path = ROOT / name
            if name.startswith('scripts/') and path.is_file():
                plan['frozen_sha256'][name] = sha(path)
        write(plan_path, plan)
        copyfile(plan_path, record / 'protocol.json')
        preflight = read(ROOT / item['output'] / 'preflight.json')
        preflight['protocol_sha256'] = sha(plan_path)
        write(ROOT / item['output'] / 'preflight.json', preflight)
        updated += 1
    frozen['script_sha256'] = sha(source)
    frozen['status'] = 'passed-cpu-freeze-refreshed'
    write(OUT / 'freeze.json', frozen)
    print('ROLL_DOMAIN_PARAMETER_SCREEN_HASHES_REFRESHED', updated, sha(OUT / 'freeze.json'), flush=True)


def verify_arm(plan_name, out_name):
    import verify_continuous_roll_projected_entry as verifier
    import run_continuous_roll_projected_entry as projected

    plan_path, out = ROOT / plan_name, ROOT / out_name
    page = ROOT / read(plan_path)['page']
    with patch.multiple(verifier, PLAN=plan_path, OUT=out, PAGE=page), \
         patch.multiple(projected, PLAN=plan_path, OUT=out, PAGE=page), \
         patch.multiple(baseline, PLAN=plan_path, OUT=out, PAGE=page):
        verifier.main()


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'refresh', 'run', 'verify-arm'))
    parser.add_argument('plan', nargs='?')
    parser.add_argument('output', nargs='?')
    parser.add_argument('--draft', default=None)
    args = parser.parse_args()
    if args.draft:
        DRAFT = ROOT / args.draft
        draft = read(DRAFT)
        OUT = ROOT / draft.get('output_root', 'records/reports/roll-domain-parameter-screen')
        PAGE = ROOT / draft.get('page', 'records/roll-domain-parameter-screen-live.html')
    if args.mode == 'verify-arm':
        assert args.plan and args.output
        verify_arm(args.plan, args.output)
    else:
        {'freeze': freeze, 'refresh': refresh_protocol_hashes, 'run': run_all}[args.mode]()

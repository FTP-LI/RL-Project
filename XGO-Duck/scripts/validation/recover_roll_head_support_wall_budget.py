"""Freeze a six-arm wall clock from measured runtime while preserving zero-step evidence."""

from argparse import ArgumentParser
from datetime import datetime
from math import ceil
from pathlib import Path
from shutil import copyfile
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'configs/experiments/continuous_roll_head_support_pulse_wall_budget.json'
OUT = ROOT/'records/reports/continuous-roll-head-support-pulse/wall-budget'
PAGE = ROOT/'records/continuous-roll-head-support-pulse-wall-budget-live.html'


def observe():
    import verify_roll_active_lateral as legacy
    import verify_walk_live_page as live

    main = live.main

    def wait_for_all_arms():
        with patch.object(sys,'argv',[*sys.argv[:-1],'1380']):
            main()

    with patch.object(legacy,'PLAN',PLAN), patch.object(live,'main',wait_for_all_arms):
        legacy.observe()


def freeze(run):
    run.lateral.require_stage(run.DRAFT,run.STAGE)
    read,sha,write,relative = run.read,run.sha,run.write,run.lateral.relative
    p, previous = read(run.PLAN), read(run.lateral.PLAN)
    record = ROOT/p['record']
    end = read(record/'runtime-contract.json')
    stop = read(record/'state.json')
    assert end['control_steps'] == end['control_transitions'] == end['physics_substeps'] == 0
    assert stop['status']=='stopped-review-required' and stop['controls']==0
    assert not any((record/name).exists() for name in ('evaluation.json','shared-trace.npz','shared-commands.npz'))
    launch = run.OUT/'sample-exit.json'
    assert read(launch)['exit_code']==1
    assert not any(path.exists() for path in (PLAN,OUT,PAGE))
    archive = ROOT/previous['record']
    telemetry_path, trial_path = archive/'gpu-telemetry.json', archive/'evaluation.json'
    telemetry, trial = read(telemetry_path), read(trial_path)
    elapsed = (datetime.fromisoformat(telemetry[-1]['at'])-datetime.fromisoformat(telemetry[0]['at'])).total_seconds()
    used = trial['controls']
    controls = p['perturbation']['maximum_unique_controls']
    # Include normal closure and video work. Round up to whole minutes.
    needed = 60*ceil(1.15*elapsed*controls/used/60)
    prior_limit = p['perturbation']['maximum_wall_seconds']
    assert needed == 1200 and needed > prior_limit and controls == 9010
    name = 'duck_switch_roll-head-support-pulse_wall-budget_s1176_'+datetime.now(run.lateral.baseline.TZ).strftime('%Y%m%dT%H%M%S')
    destination = ROOT/'records/evaluations'/name
    evidence = (run.PLAN,record/'protocol.json',record/'runtime-contract.json',record/'state.json',launch,telemetry_path,trial_path)
    fingerprints = {relative(path):sha(path) for path in (*evidence,Path(__file__).resolve())}
    p['frozen_sha256'].update(fingerprints)
    p['perturbation']['maximum_wall_seconds'] = needed
    p.update(run_id=name,record=relative(destination),page=relative(PAGE),wall_budget=dict(
        previous_limit_seconds=prior_limit,seconds=needed,measured_seconds=elapsed,measured_controls=used,
        forecast_controls=controls,padding_fraction=.15,previous_physical_substeps=0))
    destination.mkdir()
    OUT.mkdir()
    write(PLAN,p)
    copyfile(PLAN,destination/'protocol.json')
    (destination/'README.md').write_text('# 滚翻头部支撑对照\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../continuous-roll-head-support-pulse-wall-budget-live.html)\n',
        encoding='utf-8')
    with patch.object(run.lateral,'PAGE',PAGE):
        run.lateral.publish(destination,dict(status='starting',purpose='滚翻头部支撑对照',run_id=name,controls=0,
            maximum_controls=controls,transitions=0,updates=[],learning_updates=0))
    write(OUT/'preflight.json',dict(status='frozen-before-first-physics',protocol_sha256=sha(PLAN),record=p['record'],
        wall_budget=p['wall_budget'],physical_substeps=0))
    print('ROLL_HEAD_SUPPORT_WALL_BUDGET_FROZEN',p['record'],sha(PLAN),needed,flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze','observe','sample','verify'))
    mode = parser.parse_args().mode
    if mode == 'observe':
        return observe()
    import run_roll_head_support_pulse as run

    if mode == 'freeze':
        return freeze(run)
    with patch.multiple(run,PLAN=PLAN,PAGE=PAGE):
        if mode == 'sample':
            return run.sample()
        import verify_roll_head_support_run as audit

        with patch.object(run,'OUT',OUT):
            return audit.verify()


if __name__=='__main__':
    main()

"""Create a fresh six-arm roll run after a user pause, preserving the old run."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from shutil import copytree, copyfile
from unittest.mock import patch

import run_roll_head_support_pulse as run


ROOT = run.ROOT
OLD_PLAN = ROOT / 'configs/experiments/continuous_roll_head_support_pulse_wall_budget.json'
PLAN = ROOT / 'configs/experiments/continuous_roll_head_support_pulse_resume.json'
OUT = ROOT / 'records/reports/continuous-roll-head-support-pulse/resume-20261113'
PAGE = ROOT / 'records/continuous-roll-head-support-pulse-wall-budget-resume-live.html'
STAGE = run.STAGE


def relative(path):
    return path.relative_to(ROOT).as_posix()


def freeze():
    from record_io import write

    read, sha = run.read, run.sha
    assert not PLAN.exists() and not OUT.exists() and not PAGE.exists()
    old = read(OLD_PLAN)
    old_out = ROOT / 'records/reports/continuous-roll-head-support-pulse/wall-budget'
    old_record = ROOT / old['record']
    pause = read(old_out / 'pause.json')
    runtime = read(old_record / 'runtime-contract.json')
    assert pause['status'] == 'paused-by-user'
    assert runtime['control_steps'] == 2225 and runtime['physics_substeps'] == 11125
    assert pause['run'] == old['record']
    assert old['perturbation']['maximum_wall_seconds'] == 1200
    source_zero = ROOT / 'records/reports/continuous-roll-head-support-pulse/zero'
    assert (source_zero / 'startup.npz').exists() and (source_zero / 'endpoint-contact.npz').exists()
    run_id = 'duck_switch_roll-head-support-pulse_wall-budget_resume_s1176_' + datetime.now(
        run.lateral.baseline.TZ).strftime('%Y%m%dT%H%M%S')
    record = ROOT / 'records/evaluations' / run_id
    assert not record.exists()
    p = dict(old)
    p.update(
        status='frozen-before-dynamics',
        run_id=run_id,
        record=relative(record),
        page=relative(PAGE),
        resume_from={
            'status': 'fresh-run-after-user-pause',
            'previous_protocol': relative(OLD_PLAN),
            'previous_record': old['record'],
            'previous_pause': relative(old_out / 'pause.json'),
            'previous_physical_substeps': runtime['physics_substeps'],
            'previous_recorded_controls': pause['recorded_controls'],
            'new_run_restarts_physics': True,
        },
    )
    source = Path(__file__).resolve()
    p['frozen_sha256'] = dict(old['frozen_sha256'])
    p['frozen_sha256'][relative(source)] = sha(source)
    p['frozen_sha256'][relative(old_out / 'pause.json')] = sha(old_out / 'pause.json')
    p['frozen_sha256'][relative(old_record / 'runtime-contract.json')] = sha(old_record / 'runtime-contract.json')
    OUT.mkdir(parents=True)
    copytree(source_zero, OUT / 'zero')
    for path in sorted((OUT / 'zero').rglob('*')):
        if path.is_file():
            p['frozen_sha256'][relative(path)] = sha(path)
    record.mkdir(parents=True)
    write(PLAN, p)
    copyfile(PLAN, record / 'protocol.json')
    (record / 'README.md').write_text(
        '# 滚翻头部支撑对照续跑\n\n'
        '本运行从新的进程和新的物理初态开始，完整执行六组同初态对照。上一次用户暂停的部分运行单独保留。\n\n'
        f'[冻结方案](protocol.json) · [运行观察](../../{PAGE.name}) · '
        f'[上次暂停](../../../{relative(old_out / "pause.json")})\n',
        encoding='utf-8',
    )
    with patch.object(run.lateral, 'PAGE', PAGE):
        run.lateral.publish(
            record,
            dict(
                status='starting',
                purpose='滚翻头部支撑六组对照续跑',
                run_id=run_id,
                controls=0,
                maximum_controls=p['maximum_controls'],
                transitions=0,
                updates=[],
                learning_updates=0,
                recovery_from=p['resume_from'],
            ),
        )
    write(
        OUT / 'preflight.json',
        dict(
            status='frozen-before-dynamics-resume',
            protocol_sha256=sha(PLAN),
            record=p['record'],
            page=p['page'],
            physical_substeps=0,
            previous_pause_sha256=sha(old_out / 'pause.json'),
            copied_zero_sha256={relative(path): sha(path) for path in sorted((OUT / 'zero').rglob('*')) if path.is_file()},
        ),
    )
    print('ROLL_HEAD_SUPPORT_RESUME_FROZEN', p['record'], sha(PLAN), flush=True)


def observe():
    import verify_roll_active_lateral as legacy
    import verify_walk_live_page as live
    import sys

    def wait_for_run():
        with patch.object(sys, 'argv', [*sys.argv[:-1], '1380']):
            live.main()

    with patch.object(legacy, 'PLAN', PLAN), patch.object(live, 'main', wait_for_run):
        legacy.observe()


def sample():
    with patch.multiple(run, PLAN=PLAN, OUT=OUT, PAGE=PAGE):
        run.sample()


def verify():
    with patch.multiple(run, PLAN=PLAN, OUT=OUT, PAGE=PAGE):
        import verify_roll_head_support_run as audit

        audit.verify()


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'observe', 'sample', 'verify'))
    {'freeze': freeze, 'observe': observe, 'sample': sample, 'verify': verify}[parser.parse_args().mode]()

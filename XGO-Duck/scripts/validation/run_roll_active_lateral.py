"""Freeze and collect matched roll-time lateral perturbations."""

from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime
import fcntl
from pathlib import Path
from shutil import copyfile, disk_usage
import sys
from unittest.mock import patch

import numpy as np
import torch

from check_roll_active_lateral_cuda import DRAFT, OUT, ROOT, SEED, read, sha, sources
from collect_roll_active_lateral import baseline, collect
from getup_shared_release import project_running
from record_io import write
from roll_endpoint_readback_capture import EndpointReadback
from run_getup_latched_head_corrective_ppo import require_stage
from run_local_getup_ppo import MemoryWatch
from s1_reference_inputs import WORK


PLAN = DRAFT.with_stem('continuous_roll_early_lateral_push')
PAGE = ROOT / 'records/continuous-roll-early-lateral-push-live.html'
STAGE = 'S5-roll-active-lateral-disturbance'


def relative(path):
    return path.relative_to(ROOT).as_posix()


def publish(record, state):
    state['updated_at'] = datetime.now(baseline.TZ).isoformat()
    with patch.object(baseline.live, 'PAGE', PAGE):
        baseline.live.publish(record, state)


def checked():
    require_stage(PLAN, STAGE)
    plan = read(PLAN)
    assert plan['status'] == 'frozen-before-dynamics' and plan['stage'] == STAGE and plan['seed'] == SEED
    assert not any(plan[key] for key in ('hardware_io', 'heldout_used', 'learning_updates', 'automatic_retry', 'automatic_extension'))
    for root, field in ((ROOT, 'frozen_sha256'), (WORK, 'training_source_sha256'), (Path('/'), 'library_source_sha256')):
        for name, expected in plan[field].items():
            assert sha(root / name) == expected, name
    return plan


def freeze():
    draft, parent, _, warp_limit, cpu, source_files = sources()
    require_stage(DRAFT, draft['stage'])
    assert not PLAN.exists() and not PAGE.exists() and not torch.cuda.is_initialized()
    zero_path = OUT / 'cuda-zero.json'
    zero = read(zero_path)
    assert zero['status'] == 'passed-actual-cuda-zero-step' and zero['seed'] == SEED and zero['physical_substeps'] == 0
    assert zero['eligible'] == cpu['eligible'] and zero['fixture']['warp_bytes'] <= warp_limit
    for name, expected in zero['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in zero['initial_archives'].items():
        assert sha(OUT / 'zero' / name) == expected, name
    assert read(OUT / 'zero/runtime-contract.json')['physics_substeps'] == 0
    assert parent['worlds'] == draft['worlds'] == 128 and parent['initial_stand_controls'] == draft['standing_heading_control']
    assert parent['maximum_controls'] == draft['maximum_controls_each_trajectory'] == 1550
    assert parent['minimum_continuous_successes'] == parent['minimum_roll_completion'] == draft['minimum_complete_original_sequence'] == 116
    assert parent['forward_controls'] == 1000 and parent['feedback']['maximum_yaw_input_rad_s'] == .15
    assert parent['feedback']['maximum_joint_correction_rad'] == .03 and parent['walking']['criteria']['yaw_error_max'] == .2
    assert parent['projected_roll_without_slew']
    shared = draft['injection_control']
    roles = draft['groups_horizontal_velocity_m_s']
    assert roles == {'reference': 0., 'left-push': .08, 'right-push': -.08}
    total = shared + len(roles)*(parent['maximum_controls']-shared)
    assert total == draft['maximum_unique_controls'] and total*5 == draft['maximum_batched_physics_substeps']
    assert total*parent['worlds'] == draft['maximum_transitions']
    assert disk_usage(ROOT).free >= 12*1024**3
    assert not any(f'_s{SEED}_' in path.name for path in (ROOT / 'records/evaluations').iterdir())
    run_id = f'duck_switch_roll-early-lateral-push_development_s{SEED}_{datetime.now(baseline.TZ):%Y%m%dT%H%M%S}'
    record = ROOT / 'records/evaluations' / run_id
    names = ('collect_roll_active_lateral.py', 'verify_roll_active_lateral.py', 'verify_walk_live_page.py',
        'verify_continuous_lateral_push.py', 'verify_continuous_forward_yaw_pair.py', 'verify_continuous_roll_projected_entry.py',
        'getup_shared_release.py', 'run_local_getup_ppo.py')
    files = (*source_files, zero_path, Path(__file__).resolve(), *(Path(__file__).with_name(name).resolve() for name in names))
    frozen = dict(parent['frozen_sha256'])
    frozen.update({relative(path): sha(path) for path in files})
    frozen.update({relative(OUT / 'zero' / name): expected for name, expected in zero['initial_archives'].items()})
    study = dict(shared=shared, heading_control=draft['standing_heading_control'], minimum_eligible=draft['minimum_eligible_active_roll'],
        roles=roles, maximum_unique_controls=total, maximum_control_transitions=total*parent['worlds'],
        maximum_physics_substeps=total*5, maximum_wall_seconds=660, memory_fraction_limit=.85)
    plan = dict(parent, stage=STAGE, seed=SEED, roll=dict(parent['roll'], seed=SEED), run_id=run_id,
        perturbation=study, record=relative(record), page=relative(PAGE), maximum_warp_snapshot_bytes=warp_limit, frozen_sha256=frozen)
    record.mkdir()
    write(PLAN, plan)
    copyfile(PLAN, record / 'protocol.json')
    (record / 'README.md').write_text('# 滚翻阶段横向扰动\n\n'
        '三组从共同前 58 步分支。比较无扰动、左推和右推，保留原始滚翻、落地和完整前进判据。\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../continuous-roll-early-lateral-push-live.html)\n', encoding='utf-8')
    publish(record, dict(status='starting', purpose='滚翻阶段横向扰动', run_id=run_id,
        controls=0, maximum_controls=total, transitions=0, updates=[], learning_updates=0))
    write(OUT / 'preflight.json', dict(status='frozen-after-cuda-zero-step', protocol_sha256=sha(PLAN),
        record=relative(record), zero_sha256=sha(zero_path), physical_substeps=0))
    print('ROLL_ACTIVE_LATERAL_FROZEN', relative(record), sha(PLAN), flush=True)


def sample():
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    p = checked()
    record = ROOT / p['record']
    assert not (record / 'command.json').exists() and not (record / 'evaluation.json').exists()
    ready = read(record / 'windows-observer-ready.json')
    assert sha(PLAN) == sha(record / 'protocol.json') == ready['protocol_sha256']
    assert ready['status'] == 'starting-page-observed' and ready['page_sha256'] == sha(PAGE)
    assert ready['initial_state_sha256'] == sha(record / 'state.json')
    watch = MemoryWatch(p['perturbation']['memory_fraction_limit'])
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg = baseline.configuration(p)

    def check_active():
        require_stage(PLAN, STAGE)
        watch.check()

    def forbid_reset(*args, **kwargs):
        raise AssertionError('Matched roll branches permit only their common initial reset')

    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with watch.active():
            write(record / 'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), protocol_sha256=sha(PLAN)))
            dump_yaml(record / 'env.yaml', asdict(cfg))
            guard = baseline.install_guard(p['roll'], record, baseline.STARTUP, 'rollout')
            env = None
            try:
                env = guard(cfg=cfg, device='cuda:0', render_mode=None)
                env.seed(SEED)
                obs, _ = env.reset()
                networks, homes, sim_home, limits = baseline.actors(env, obs, p, record)
                endpoint = EndpointReadback(env, record)
                endpoint.capture()
                assert not endpoint.first_mismatches
                with np.load(OUT/'zero/startup.npz', allow_pickle=False) as expected, \
                     np.load(record/'startup.npz', allow_pickle=False) as current:
                    baseline.compare_arrays(dict(current), dict(expected))
                with np.load(OUT/'zero/endpoint-contact.npz', allow_pickle=False) as expected:
                    for key, value in endpoint.rows[0].items():
                        np.testing.assert_array_equal(value, expected[key][0], err_msg=key)
                endpoint.paired_start_verified = True
                check_active()
                project_running(True)
                with torch.no_grad(), patch.object(env, '_reset_idx', forbid_reset):
                    collect(env, obs, p, record, endpoint, networks, homes, sim_home, limits, page=PAGE,
                        check_active=check_active)
                check_active()
            except BaseException as error:
                state = read(record / 'state.json')
                state.update(status='stopped-review-required', purpose='滚翻扰动采样异常停止', error=repr(error))
                publish(record, state)
                raise
            finally:
                if env is not None:
                    env.close()
                project_running(False)
                write(record / 'gpu-telemetry.json', watch.rows)
                write(record / 'gpu-summary.json', watch.summary())
    state = read(record / 'state.json')
    if (record / 'evaluation.json').is_file() and state.get('latest_clip'):
        state.update(status='completed', purpose='滚翻扰动采样完成，等待独立评分')
    else:
        state.update(status='stopped-review-required', purpose='滚翻扰动按门槛停止，保留部分原片')
    publish(record, state)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'sample'))
    (freeze if parser.parse_args().mode == 'freeze' else sample)()

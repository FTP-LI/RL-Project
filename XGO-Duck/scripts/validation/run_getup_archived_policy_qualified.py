"""Run the same frozen getup comparison with process-local startup qualification."""

from argparse import ArgumentParser
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
import fcntl
from pathlib import Path
from shutil import copyfile
import sys
from time import perf_counter
from unittest.mock import patch
from zoneinfo import ZoneInfo

import torch

import run_getup_archived_policy_comparison as base


ROOT, read, sha, write = base.ROOT, base.read, base.sha, base.write
SEED = 1160
PARENT, PARENT_OUT, _ = base.files(SEED)
PLAN = PARENT.with_stem('getup_archived_policy_comparison_qualified_s1160')
OUT = PARENT_OUT.with_name('s1160-qualified')
PAGE = ROOT / 'records/getup-archived-policy-qualified-s1160-live.html'
EVIDENCE = PARENT_OUT / 'reset-field-diagnosis'
STAGE = 'S3-getup-archived-policy-same-process'


def ready_for(protocol):
    base.check_project()
    project = read(ROOT / 'project.json')
    assert ROOT / project['goal_execution']['current_protocol'] == protocol
    assert project['current_phase'] == STAGE


def freeze():
    ready_for(PARENT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE))
    spec = deepcopy(read(PARENT))
    base.shared.assert_sources(spec)
    stopped = ROOT / spec['record'] / 'evaluation.json'
    prior = read(stopped)
    assert prior['status'] == 'stopped-review-required' and prior['actual']['attempted_substeps'] == 0
    evidence = read(EVIDENCE / 'analysis.json')
    assert evidence['physical_substeps'] == 0 and evidence['protocol_sha256'] == sha(PARENT)
    assert evidence['original_attempt_closed'] and evidence['import_errors_equal']
    differences = evidence['differences']
    assert len(differences['fields.json']) == 25 and not differences['stable-inputs.json']
    witness = EVIDENCE / 'a'
    fields, stable = read(witness / 'fields.json'), read(witness / 'stable-inputs.json')
    assert fields.keys() == read(EVIDENCE / 'b' / 'fields.json').keys() and len(fields) == 335
    assert stable == read(EVIDENCE / 'b' / 'stable-inputs.json') and len(stable) == 14
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_archived-policies_qualified_s1160_{stamp}'
    record = ROOT / 'records/evaluations' / run_id
    assert not record.exists()
    spec.update(status='frozen-archived-getup-same-process', stage=STAGE, record=base.rel(record), page=base.rel(PAGE),
                allowed_cross_process_fields=differences['fields.json'], recovery_from=base.rel(stopped),
                within_process_release_exact=True, reused_development_seed_zero_prior_physics=True)
    spec['environment'].update(stage=STAGE, run_id=run_id)
    for path in (PARENT, stopped, EVIDENCE / 'analysis.json', PARENT_OUT / 'native-zero.json',
                 *(EVIDENCE / probe / name for probe in ('a', 'b') for name in ('fields.json', 'stable-inputs.json', 'proof.json'))):
        spec['references'][base.rel(path)] = sha(path)
    for path in (Path(__file__), Path(__file__).with_name('observe_getup_archived_policy_qualified.py'),
                 Path(__file__).with_name('diagnose_getup_archived_policy_zero.py')):
        spec['source_sha256'][base.rel(path.resolve())] = sha(path)
    base.shared.assert_sources(spec)
    record.mkdir()
    OUT.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    receipt = read(PARENT_OUT / 'freeze.json')
    write(OUT / 'freeze.json', dict(receipt, parent_protocol_sha256=sha(PARENT), protocol_sha256=sha(PLAN),
                                   stable_fields=len(stable), allowed_derived_fields=len(differences['fields.json'])))
    (record / 'README.md').write_text('# 归档起身策略同释放对照\n\n四路共享 2 秒物理跌落。初始 CUDA 读回、运行与配对使用同一进程。\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../getup-archived-policy-qualified-s1160-live.html) · [执行日志](execution.log)\n', encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='归档起身策略同释放对照', run_id=run_id, controls=0,
                                    maximum_controls=spec['maximum_unique_controls'], transitions=0, updates=[], learning_updates=0))
    print('ARCHIVED_POLICY_QUALIFIED_FROZEN', base.rel(record), sha(PLAN), flush=True)


def qualification(env, obs, memory, spec, record):
    from getup_release_fork_state import digest

    class FieldReadback(base.SceneReleaseSnapshot):
        def __init__(self, env, memory, cap):
            super().__init__(env, memory, cap)
            write(record / 'zero-fields.json', self.expected)
            write(record / 'zero-inputs.json', {key: base.shared.signature(value)
                for key, value in base.shared.release_payload(env, memory).items()})

    with patch.object(base, 'SceneReleaseSnapshot', FieldReadback):
        actors, proof = base.native_zero(env, obs, memory, spec, PLAN, record)
    observed = read(record / 'zero-fields.json')
    stable = read(record / 'zero-inputs.json')
    assert digest(observed) == proof['zero_readback']['initial']['state_sha256']
    allowed = set(spec['allowed_cross_process_fields'])
    differences = {}
    for witness in ('a', 'b'):
        prior = EVIDENCE / witness
        expected = read(prior / 'fields.json')
        assert observed.keys() == expected.keys()
        changes = {key for key in observed if observed[key] != expected[key]}
        assert changes <= allowed, sorted(changes - allowed)
        assert stable == read(prior / 'stable-inputs.json')
        assert proof['model_max_error'] == read(prior / 'proof.json')['model_max_error']
        differences[witness] = sorted(changes)
    write(record / 'zero-qualification.json', dict(status='passed-process-local-zero', protocol_sha256=sha(PLAN),
        physical_substeps=0, stable_inputs=len(stable), matching_snapshot_fields=len(observed),
        differences_from_diagnostic_processes=differences, frozen_original_untouched=True))
    return actors, proof


def sample():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from PIL import Image

    ready_for(PLAN)
    spec = read(PLAN)
    base.shared.assert_sources(spec)
    record = ROOT / spec['record']
    assert sha(PLAN) == sha(record / 'protocol.json') == read(OUT / 'freeze.json')['protocol_sha256']
    assert not (record / 'evaluation.json').exists() and not (OUT / 'native-zero.json').exists()
    state = read(record / 'state.json')
    ready = read(record / 'windows-observer-ready.json')
    assert ready['status'] == 'starting-page-observed' and ready['protocol_sha256'] == sha(PLAN)
    assert ready['page_sha256'] == sha(PAGE) and ready['initial_state_sha256'] == sha(record / 'state.json')
    p = spec['environment']
    assert state['status'] == 'starting' and p['seed'] == SEED and not spec['heldout_used']
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg, _, meta = base.shared.prepare(p)
    watch = base.MemoryWatch(spec['draft']['gpu_memory_fraction_cap'])
    state.update(phase='gpu-zero', conditions=[], actual=dict(controls=0, attempted_controls=0,
                 substeps=0, attempted_substeps=0, resets=0), protocol_sha256=sha(PLAN))
    last, timestamp = 0, perf_counter()

    def show():
        nonlocal last, timestamp
        controls = state['actual']['controls']
        if controls > last:
            now = perf_counter()
            state['updates'].append(dict(control=controls, transitions_per_second=(controls - last) * p['num_envs'] / (now - timestamp)))
            last, timestamp = controls, now
        state.update(controls=controls, transitions=controls * p['num_envs'])
        write(record / 'evaluation.json', state)
        base.publish(record, PAGE, state)

    def guard():
        ready_for(PLAN)
        watch.check()

    class LiveCamera(base.acquisition.FixedGetupCamera):
        def render(self):
            pixels = super().render()
            step, phase = self.env.common_step_counter, state['phase']
            active = spec['branches'][0] if phase == 'shared-prefix' else phase
            if step and step % 50 == 0 and self.role == active:
                name = f'frame-{self.role}-{step}.jpg'
                Image.fromarray(pixels).save(record / name, quality=88)
                state['frame'] = name
                show()
            return pixels

    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock, watch.active():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (record / 'execution.log').open('x', buffering=1) as log, \
                redirect_stdout(base.Tee(sys.stdout, log)), redirect_stderr(base.Tee(sys.stderr, log)):
            write(record / 'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), protocol_sha256=sha(PLAN)))
            env = None
            try:
                env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
                obs, _ = env.reset(seed=SEED)
                state['actual']['resets'] = 1
                memory = base.shared.make_memory(p, meta, env.device)
                actors, proof = qualification(env, obs, memory, spec, record)
                guard()
                watch.sample()
                write(OUT / 'native-zero.json', dict(status='passed-native-zero', proof=proof, gpu=watch.summary(),
                                                     qualification_sha256=sha(record / 'zero-qualification.json')))
                state['status'] = 'running'
                base.shared.project_running(True)
                show()
                with patch.object(base.acquisition, 'FixedGetupCamera', LiveCamera):
                    base.acquisition.collect(env, actors, memory, meta, spec, record, state, show, guard)
                assert state['qpos_joint_addresses'] == read(OUT / 'freeze.json')['cpu_joint_addresses']
                assert state['actual']['controls'] == spec['maximum_unique_controls']
                assert state['actual']['substeps'] == spec['maximum_physics_substeps']
                assert len(state['conditions']) == len(spec['branches'])
                guard()
                state.update(status='completed', phase='independent-score-pending', latest_clip=state['conditions'][-1]['video']['path'])
            except BaseException as error:
                state.update(status='stopped-review-required', error=repr(error))
                raise
            finally:
                if env is not None:
                    env.close()
                state['gpu'] = watch.summary()
                write(record / 'gpu-telemetry.json', watch.rows)
                base.shared.project_running(False)
                show()


def verify():
    import verify_getup_archived_policy_comparison as scoring

    record = ROOT / read(PLAN)['record']
    qual = read(record / 'zero-qualification.json')
    assert qual['status'] == 'passed-process-local-zero' and qual['physical_substeps'] == 0
    assert read(OUT / 'native-zero.json')['qualification_sha256'] == sha(record / 'zero-qualification.json')
    with patch.object(scoring, 'files', lambda seed: (PLAN, OUT, PAGE)):
        scoring.verify(SEED)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'sample', 'verify'))
    {'freeze': freeze, 'sample': sample, 'verify': verify}[parser.parse_args().mode]()

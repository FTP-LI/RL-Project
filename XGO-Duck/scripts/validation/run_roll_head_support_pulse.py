"""Qualify and freeze the six-arm roll head-support development comparison."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from shutil import copyfile, disk_usage
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import check_roll_active_lateral_cuda as native
import roll_entry_snapshot as snapshots
import run_roll_active_lateral as lateral
from collect_roll_head_support_pulse import collect
from roll_head_support_control import HeadSupportRollPolicy, pulse_snapshot_slots
from roll_head_support_pulse import HeadSupportPulse


ROOT, read, sha, write = lateral.ROOT, lateral.read, lateral.sha, lateral.write
DRAFT = ROOT/'configs/experiments/continuous_roll_head_support_pulse_draft.json'
PLAN = DRAFT.with_stem('continuous_roll_head_support_pulse')
OUT = ROOT/'records/reports/continuous-roll-head-support-pulse'
PAGE = ROOT/'records/continuous-roll-head-support-pulse-live.html'
STAGE = 'S5-roll-head-support-pulse-preparation'
SEED = 1176


def replay_inputs():
    proof = read(lateral.OUT/'analysis.json')
    record = ROOT/proof['record']
    paths = (record/'shared-commands.npz', record/'clips/reference/commands.npz')
    for file in paths:
        assert sha(file) == proof['source_sha256'][lateral.relative(file)]
    return paths


def sources():
    original, parent, archive, limit, cpu, files = native.sources()
    draft = read(DRAFT)
    assert draft['status']=='draft-no-physical-execution' and draft['development_seed']==SEED
    assert not any(draft[key] for key in ('physical_substeps_authorized','heldout_used','hardware_io','learning_updates'))
    for file,expected in draft['pinned'].items():
        assert sha(ROOT/file)==expected,file
    assert draft['worlds']==parent['worlds']==128 and draft['original_horizon_controls']==parent['maximum_controls']==1550
    assert draft['minimum_original_roll_and_complete']==parent['minimum_roll_completion']==parent['minimum_continuous_successes']
    assert draft['shared_controls']==original['injection_control'] and draft['perturbation_m_s']==original['groups_horizontal_velocity_m_s']
    reports = ROOT/'records/reports/continuous-roll-early-lateral-push'
    replay, overlay = reports/'head-roll-pulse-cpu.json', reports/'head-roll-overlay-cpu.json'
    assert read(replay)['draft_sha256']==sha(DRAFT) and read(overlay)['status']=='passed-cpu-control-overlay'
    for file,expected in read(overlay)['sha256'].items():
        assert sha(ROOT/file)==expected,file
    here = Path(__file__).resolve()
    extra = tuple(here.with_name(name) for name in ('roll_head_support_pulse.py','roll_head_support_control.py',
        'collect_roll_head_support_pulse.py','verify_roll_head_support_control.py','verify_roll_head_support_run.py'))
    initial_stop = OUT.with_name(OUT.name+'-zero-startup')
    contract = initial_stop/'zero/runtime-contract.json'
    assert read(contract)['control_steps'] == read(contract)['physics_substeps'] == 0
    replay_stop = OUT.with_name(OUT.name+'-zero-snapshot')
    replay_contract = replay_stop/'zero/runtime-contract.json'
    replay_pulse = replay_stop/'pulse-cuda.json'
    assert read(replay_contract)['control_steps'] == read(replay_contract)['physics_substeps'] == 0
    assert read(replay_pulse)['status'] == 'passed-native-pulse-fixture'
    settings = dict(original,stage=STAGE)
    return settings, parent, archive, limit, cpu, (
        *files,DRAFT,replay,overlay,here,*extra,contract,initial_stop/'first-preflight.log',
        replay_contract,replay_pulse,replay_stop/'second-preflight.log',*replay_inputs())


def zero():
    lateral.require_stage(DRAFT,STAGE)
    candidate = read(DRAFT)['candidate']
    prepared = sources()
    parent = prepared[1]
    slots, actors = snapshots.roll_slots, native.actors
    prefix, suffix = prepared[-1][-2:]
    with np.load(prefix,allow_pickle=False) as head, np.load(suffix,allow_pickle=False) as tail:
        observed, phases = (np.concatenate((head[key],tail[key]))[:220] for key in ('observed','phase_used'))
        shared = len(head['phase_used'])
    OUT.mkdir()

    def snapshot_slots(env,memory):
        if not hasattr(memory,'pulse'):
            memory.pulse = HeadSupportPulse(parent['worlds'],candidate['entry_degrees'],candidate['controls'])
        fields,generators = slots(env,memory)
        return pulse_snapshot_slots(fields,memory),generators

    def compare_actors(env,obs,spec,folder):
        networks, homes, sim_home, limits = actors(env,obs,spec,folder)
        p = dict(parent,head_support_pulse=candidate)
        off, on = (HeadSupportRollPolicy(p,networks,homes,sim_home,limits) for _ in range(2))
        phase = SimpleNamespace(phase=phases[0].copy(),phase_changed=np.zeros(env.num_envs,dtype=int))
        selected, changed = [], []
        with torch.inference_mode():
            for i, (values, mode) in enumerate(zip(observed,phases,strict=True),1):
                phase.phase_changed[mode != phase.phase] = i-1
                phase.phase = mode
                on.pulse_enabled = i > shared
                actor_input = torch.from_numpy(values).to(env.device)
                original,a = off.next(i,actor_input,phase)
                updated,b = on.next(i,actor_input,phase)
                np.testing.assert_array_equal(a['actor_input'],b['actor_input'])
                np.testing.assert_array_equal(a['raw_policy_actions'],b['raw_policy_actions'])
                expected = original.clone()
                mask = torch.from_numpy(b['pulse_selected'] & on.pulse_enabled).to(env.device)
                expected[mask,on.joint] = torch.clamp(original[mask,on.joint]+on.increment,min=limits[on.joint,0])
                torch.testing.assert_close(updated,expected,atol=0,rtol=0)
                selected.append(b['pulse_selected'])
                changed.append(b['pulse_applied_delta_rad'])
        selected, changed = np.asarray(selected), np.asarray(changed)
        assert not selected[:shared+1].any() and (selected.sum(axis=0)==2).all()
        assert np.count_nonzero(changed) > 0 and np.abs(changed).max() <= .0200001
        assert env.common_step_counter == env._sim_step_counter == 0
        write(OUT/'pulse-cuda.json',dict(status='passed-native-pulse-fixture',physical_substeps=0,
            actor_inputs=len(observed)*env.num_envs,selected_targets=int(selected.sum()),
            changed_targets=int(np.count_nonzero(changed)),maximum_target_change_rad=float(np.abs(changed).max()),
            actor_history_unchanged=True))
        return networks,homes,sim_home,limits

    with patch.multiple(native,DRAFT=DRAFT,OUT=OUT,SEED=SEED,sources=lambda:prepared,actors=compare_actors), \
         patch.object(snapshots,'roll_slots',snapshot_slots):
        native.main()


def freeze():
    lateral.require_stage(DRAFT,STAGE)
    settings, parent, _, limit, cpu, files = sources()
    old, draft = read(lateral.PLAN), read(DRAFT)
    cuda_path, pulse_path = OUT/'cuda-zero.json', OUT/'pulse-cuda.json'
    cuda, pulse = read(cuda_path), read(pulse_path)
    assert cuda['status']=='passed-actual-cuda-zero-step' and pulse['status']=='passed-native-pulse-fixture'
    assert not cuda['physical_substeps'] and not pulse['physical_substeps'] and cuda['seed']==SEED
    assert cuda['eligible']==cpu['eligible'] and cuda['fixture']['warp_bytes']<=limit
    assert all(sha(ROOT/name)==value for name,value in cuda['source_sha256'].items())
    assert all(sha(OUT/'zero'/name)==value for name,value in cuda['initial_archives'].items())
    assert not PLAN.exists() and not PAGE.exists() and not torch.cuda.is_initialized()
    assert disk_usage(ROOT).free >= 12*1024**3
    assert not any(f'_s{SEED}_' in path.name for path in (ROOT/'records/evaluations').iterdir())
    control = draft['perturbation_m_s']
    roles = {**control, **{'pulse-'+role:delta for role,delta in control.items()}}
    shared, end = draft['shared_controls'], draft['original_horizon_controls']
    total = shared+len(roles)*(end-shared)
    study = dict(old['perturbation'],roles=roles,maximum_unique_controls=total,
        maximum_control_transitions=total*parent['worlds'],maximum_physics_substeps=total*5)
    assert study['shared']==settings['injection_control'] and study['heading_control']==settings['standing_heading_control']
    name = 'duck_switch_roll-head-support-pulse_development_s'+str(SEED)+datetime.now(lateral.baseline.TZ).strftime('_%Y%m%dT%H%M%S')
    record = ROOT/'records/evaluations'/name
    frozen = dict(old['frozen_sha256'])
    frozen.update({lateral.relative(file):sha(file) for file in (*files,cuda_path,pulse_path)})
    frozen.update({lateral.relative(OUT/'zero'/file):value for file,value in cuda['initial_archives'].items()})
    p = dict(parent,stage=STAGE,seed=SEED,roll=dict(parent['roll'],seed=SEED),run_id=name,perturbation=study,
        record=lateral.relative(record),page=lateral.relative(PAGE),maximum_warp_snapshot_bytes=limit,
        head_support_pulse=draft['candidate'],frozen_sha256=frozen)
    record.mkdir()
    write(PLAN,p)
    copyfile(PLAN,record/'protocol.json')
    (record/'README.md').write_text('# 滚翻头部支撑对照\n\n'
        '从同一滚翻状态比较三个扰动方向的固定对照和两步头部脉冲。\n\n'
        '[观察](../../continuous-roll-head-support-pulse-live.html) · [冻结方案](protocol.json)\n',encoding='utf-8')
    with patch.object(lateral,'PAGE',PAGE):
        lateral.publish(record,dict(status='starting',purpose='滚翻头部支撑对照',run_id=name,controls=0,
            maximum_controls=total,transitions=0,updates=[],learning_updates=0))
    write(OUT/'preflight.json',dict(status='frozen-after-cuda-zero-step',protocol_sha256=sha(PLAN),record=p['record'],
        physical_substeps=0))
    print('ROLL_HEAD_SUPPORT_PULSE_FROZEN',p['record'],sha(PLAN),flush=True)


def sample():
    def invoke(*args,**kwargs):
        collect(*args,plan=PLAN,**kwargs)

    with patch.multiple(lateral,PLAN=PLAN,OUT=OUT,PAGE=PAGE,SEED=SEED,STAGE=STAGE,collect=invoke):
        lateral.sample()


if __name__=='__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('zero','freeze','sample'))
    dict(zero=zero,freeze=freeze,sample=sample)[parser.parse_args().mode]()

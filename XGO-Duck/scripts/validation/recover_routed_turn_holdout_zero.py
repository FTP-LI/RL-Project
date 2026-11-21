"""Preserve the unconsumed holdout and audit cdof_dot by value before launch."""

from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import run_routed_turn_feedback_holdout as hold
import verify_routed_turn_feedback_holdout as audit


ROOT, read, sha, write = hold.ROOT, hold.read, hold.sha, hold.write
PLAN = hold.PLAN.with_stem(hold.PLAN.stem+'_zero_recovery')
OUT = hold.OUT.with_name(hold.OUT.name+'-zero-recovery')
PAGE = hold.PAGE.with_stem(hold.PAGE.stem.removesuffix('-live')+'-zero-recovery-live')
FIELD = 'warp.cdof_dot'


def previous_stop():
    spec = read(hold.PLAN)
    record = ROOT/spec['record']
    stop = read(record/'evaluation.json')
    assert stop['status'] == 'stopped-review-required' and stop['phase'] == 'gpu-zero'
    assert stop['actual'] == dict(controls=0, attempted_controls=0, substeps=0, attempted_substeps=0, resets=1)
    assert stop['error'] == "AssertionError(['warp.cdof_dot'])" and not stop['heldout_used']
    assert not any((record/name).exists() for name in ('holdout-consumption.json','fork.json',hold.ARM))
    assert not (hold.OUT/'native-zero.json').exists()
    zero, launch = hold.OUT/'independent-zero', hold.OUT/'sample-exit.json'
    assert read(launch)['exit_code'] == 1
    checksum = sha(hold.PLAN)
    assert checksum == stop['protocol_sha256'] == sha(record/'protocol.json') == read(zero/'proof.json')['protocol_sha256']
    first, second = (read(path/'zero-fields.json') for path in (zero, record))
    assert first.keys() == second.keys()
    assert read(zero/'zero-inputs.json') == read(record/'zero-inputs.json')
    changed = {name for name in first if first[name] != second[name]}
    assert changed - set(spec['allowed_cross_process_fields']) == {FIELD}
    paths = (hold.PLAN,record/'protocol.json',record/'evaluation.json',record/'zero-fields.json',
             record/'zero-inputs.json',zero/'proof.json',zero/'zero-fields.json',zero/'zero-inputs.json',launch)
    return spec, {hold.base.rel(path):sha(path) for path in paths}


def freeze():
    hold.phase(PLAN)
    assert not any(path.exists() for path in (PLAN,OUT,PAGE))
    spec, archived = previous_stop()
    run_id = 'duck_switch_routed-turn-gyro_holdout-zero-recovery_s2175_'+datetime.now(
        ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    record = ROOT/'records/evaluations'/run_id
    spec['environment']['run_id'] = run_id
    spec.update(record=hold.base.rel(record), page=hold.base.rel(PAGE), zero_recovery=dict(
        first_attempt=archived,field=FIELD, acceptance='Exact finite numeric equality. Permit only zero sign differences.',
        plant_writes=0, extra_physics_steps=0))
    spec['references'].update(archived)
    source = Path(__file__).resolve()
    spec['source_sha256'][hold.base.rel(source)] = sha(source)
    hold.base.shared.assert_sources(spec)
    OUT.mkdir()
    record.mkdir()
    write(PLAN,spec)
    copyfile(PLAN,record/'protocol.json')
    write(OUT/'freeze.json',dict(read(hold.OUT/'freeze.json'),protocol_sha256=sha(PLAN),record=spec['record'],
                                old_protocol_sha256=sha(hold.PLAN),physical_substeps=0))
    (record/'README.md').write_text('# 七指令留出初态复核\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../continuous-routed-turn-feedback-holdout-zero-recovery-live.html)\n',
        encoding='utf-8')
    hold.base.publish(record,PAGE,dict(status='starting',purpose='七指令独立留出',run_id=run_id,
        controls=0,maximum_controls=spec['maximum_unique_controls'],transitions=0,updates=[],learning_updates=0))
    print('ROUTED_TURN_HOLDOUT_RECOVERY_FROZEN',spec['record'],sha(PLAN),flush=True)


def snapshot(env,obs,memory,spec,target):
    actors,proof,fields,inputs = hold.snapshot(env,obs,memory,spec,target)
    array = env.sim.wp_data.cdof_dot.numpy()
    assert np.isfinite(array).all() and sha256(array.tobytes()).hexdigest() == fields[FIELD][-1]
    assert env.common_step_counter == env._sim_step_counter == 0
    np.save(target/'cdof-dot.npy',array,allow_pickle=False)
    return actors,proof,fields,inputs


def verify_comparison(witness,target):
    with (witness/'cdof-dot.npy').open('rb') as f:
        first = np.load(f,allow_pickle=False)
    with (target/'cdof-dot.npy').open('rb') as f:
        second = np.load(f,allow_pickle=False)
    assert first.shape == second.shape == (3072,20,6) and first.dtype == second.dtype == np.float32
    assert np.isfinite(first).all() and np.isfinite(second).all()
    np.testing.assert_array_equal(first,second,err_msg='Numerical drift in cdof_dot')
    bit_changed = first.view(np.uint32) != second.view(np.uint32)
    assert np.all(first[bit_changed] == 0.) and np.all(second[bit_changed] == 0.)
    for directory,values in ((witness,first),(target,second)):
        assert sha256(values.tobytes()).hexdigest() == read(directory/'zero-fields.json')[FIELD][-1]
    return dict(field=FIELD,bit_changes=int(bit_changed.sum()),values_identical=True,
                maximum_absolute_difference=0.,witness_sha256=sha(witness/'cdof-dot.npy'),
                sample_sha256=sha(target/'cdof-dot.npy'),plant_writes=0)


def qualify(env,obs,memory,spec,target):
    witness = OUT/'independent-zero'
    prior = read(witness/'proof.json')
    assert prior['protocol_sha256'] == sha(PLAN) and prior['physical_substeps'] == 0
    for name,expected in prior['hashes'].items():
        assert sha(witness/name) == expected
    actors,proof,fields,inputs = snapshot(env,obs,memory,spec,target)
    old = read(witness/'zero-fields.json')
    assert old.keys() == fields.keys() and inputs == read(witness/'zero-inputs.json')
    changed = {name for name in fields if fields[name] != old[name]}
    comparison = verify_comparison(witness,target)
    assert changed <= set(spec['allowed_cross_process_fields']) | {FIELD}
    assert prior['model_max_error'] == proof['model_max_error']
    write(target/'zero-qualification.json',dict(status='passed-process-local-zero',protocol_sha256=sha(PLAN),
        physical_substeps=0,stable_inputs=len(inputs),matching_snapshot_fields=len(fields),
        differing_derived_fields=sorted(changed),cdof_dot=comparison,independent_zero_sha256=sha(witness/'proof.json')))
    return actors,proof


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze','zero','observe','sample','verify'))
    mode = parser.parse_args().mode
    if mode == 'freeze':
        return freeze()
    _, archived = previous_stop()
    spec = read(PLAN)
    assert spec['zero_recovery']['first_attempt'] == archived
    record = ROOT/spec['record']
    if mode == 'observe':
        with patch.object(audit,'PLAN',PLAN):
            return audit.observe()
    hold.phase(PLAN)
    with patch.multiple(hold,PLAN=PLAN,OUT=OUT,PAGE=PAGE):
        if mode == 'verify':
            assert read(record/'zero-qualification.json')['cdof_dot'] == verify_comparison(OUT/'independent-zero',record)
            with patch.object(audit,'PLAN',PLAN):
                return audit.verify()
        assert not (record/'holdout-consumption.json').exists()
        seed = spec['environment']['seed']
        pair,base,route,original = hold.pair,hold.base,hold.route,hold.original
        with patch.multiple(pair.pair_runner,PLAN=PLAN,OUT=OUT,PAGE=PAGE,SEED=seed,phase=hold.phase,collect=hold.collect), \
             patch.multiple(route.prior,PLAN=PLAN,OUT=OUT,PAGE=PAGE,SEED=seed,phase=hold.phase,snapshot=snapshot,qualify=qualify), \
             patch.object(base.shared,'prepare',original.prepare), \
             patch.object(base.shared,'actor_model',route.guarded_actor), \
             patch.object(original,'switch_models',pair.prior.route_models):
            (route.prior.zero if mode == 'zero' else pair.pair_runner.sample)()


if __name__ == '__main__':
    main()

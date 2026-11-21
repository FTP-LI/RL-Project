"""Reconcile a zero-step voltage comparison using the measured reset torque."""

import numpy as np

from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, permitted, read, rel, sha, write


KEY = 'nominal_route_voltage_delay'
STAGE = 'nominal-route-voltage-delay'
REPORT = ROOT/'records/reports/nominal-skill-composition/closed-loop-voltage-delay'
STOP = REPORT.with_name(REPORT.name+'-entry-command.log')
REFERENCE = 'nominal-7p7v-4ms'
LOW = 'voltage-7p4v'
LATE = 'delay-8ms'


def main():
    project = read(PROJECT)
    permitted(project,STAGE)
    goal,task = project['goal_execution'],project['staged_training_request']
    node = goal[KEY]
    assert node['status']=='zero-ready'
    plan = read(ROOT/node['protocol'],node['protocol_sha256'])
    assert (plan['seed'],plan['worlds'],tuple(plan['arms']))==(2258,896,(REFERENCE,LOW,LATE))
    failure = STOP.read_text(encoding='utf-8-sig')
    assert f'{LOW}/ctrl' in failure and 'ZERO_EXIT=1' in failure
    for name,digest in plan['original_sources_sha256'].items():
        assert sha(ROOT/name)==digest,name
    files,reset,fields,response = {},{},{},{}
    for name,(v,ticks) in zip(plan['arms'],((7.7,1),(7.4,1),(7.7,2)),strict=True):
        key = KEY+'_'+name.replace('-','_')
        child = goal[key]
        assert child['status']=='sample-ready' and child['protocol_sha256']==plan['arms'][name]['protocol_sha256']
        zero = REPORT/name/'zero'
        proof = read(zero/'proof.json',child['zero_sha256'])
        assert proof['physical_substeps']==0 and proof['protocol_sha256']==child['protocol_sha256']
        observed = read(zero/'factor-witness.json')
        assert observed['physics_steps']==0 and observed['factor']['supply_v']==v
        assert observed['factor']['actuator_delay_ticks']==ticks
        hashes = {'reset.npz':proof['reset_sha256'],'factor-state.npz':observed['arrays_sha256'],
                  'factor-witness.json':sha(zero/'factor-witness.json'),'proof.json':child['zero_sha256']}
        for file,digest in hashes.items():
            assert sha(zero/file)==digest
            files[name+'/'+file]=digest
        with (np.load(zero/'reset.npz',allow_pickle=False) as a,
              np.load(zero/'factor-state.npz',allow_pickle=False) as b):
            reset[name],fields[name] = dict(a),dict(b)
        r = reset[name]
        assert tuple(r)==('qpos','qvel','ctrl','actor') and not r['qvel'].any()
        np.testing.assert_array_equal(fields[name]['configured_voltage'],np.full((896,1),v,np.float32))
        for field in ('qpos','qvel','actor'):
            np.testing.assert_array_equal(r[field],reset[REFERENCE][field],err_msg=f'{name}/{field}')
        for field,x in fields[name].items():
            if field!='configured_voltage':
                np.testing.assert_array_equal(x,fields[REFERENCE][field],err_msg=f'{name}/{field}')
        ratio = fields[name]['configured_voltage']/fields[REFERENCE]['configured_voltage']
        predicted = reset[REFERENCE]['ctrl']*ratio
        np.testing.assert_allclose(r['ctrl'],predicted,atol=4e-8,rtol=0)
        if name!=LOW:
            np.testing.assert_array_equal(r['ctrl'],reset[REFERENCE]['ctrl'])
        else:
            assert np.count_nonzero(r['ctrl']!=reset[REFERENCE]['ctrl'])==8960
        response[name] = dict(supply_v=v,actuator_delay_ticks=ticks,
            maximum_abs_ctrl_error_vs_voltage_ratio=float(np.abs(r['ctrl']-predicted).max()),
            maximum_abs_ctrl_difference_nm=float(np.abs(r['ctrl']-reset[REFERENCE]['ctrl']).max()))
    receipt = REPORT/'zero-comparison.json'
    assert not receipt.exists()
    write(receipt,dict(status='verified-matched-cuda-zero',protocol_sha256=node['protocol_sha256'],seed=plan['seed'],
        same_reset_qpos_qvel_and_actor_observation=True,same_compiled_and_runtime_physics_arrays=True,
        reset_torque_definition='Zero-speed torque scales with the measured nominal supply. The delay arm retains the original reset torque.',
        maximum_torque_ratio_error_nm=max(x['maximum_abs_ctrl_error_vs_voltage_ratio'] for x in response.values()),
        initial_torque_by_arm=response,files=files,physical_steps=0,hardware_io=False,
        failed_equality_log=rel(STOP),failed_equality_log_sha256=sha(STOP),
        reconciliation_source=rel(ROOT/'scripts/validation/verify_nominal_route_voltage_delay_zero.py'),
        reconciliation_source_sha256=sha(ROOT/'scripts/validation/verify_nominal_route_voltage_delay_zero.py')))
    node.update(status='sample-ready',zero=rel(receipt),zero_sha256=sha(receipt))
    info = dict(current_protocol_execution=STAGE+'-zero-verified',current_evidence=rel(receipt))
    project.update(info)
    task.update(info,next_gate='Run the three fixed routes and independently verify measured supply and delay.')
    write(PROJECT,project)
    print('MATCHED_ZERO_RECOVERED',sha(receipt),response,flush=True)


if __name__=='__main__':
    main()

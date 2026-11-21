"""Check archived runtime math against saved evaluation observations and ONNX.

This uses synthetic feedback reconstructed from saved observations, which
already contain simulator noise/delay. It checks software interface parity,
not physical sensor correctness or closed-loop effects of runtime filters.
No training, GPU simulation, serial I/O or writes into the official archive.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
import numpy as np
import onnx
import yaml
from onnx import numpy_helper


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--onnx', type=Path, help='Explicit model for non-default records; its hash must match the saved evaluation')
    args=parser.parse_args()
    run=args.run.resolve()
    root=Path(__file__).resolve().parents[2]
    report=json.loads((run/'evaluation.json').read_text())
    # Historical WSL absolute paths are provenance, not portable file locators.
    # Resolve the current project's archive (or an explicit model) and still
    # require the exact recorded bytes. Never fall back to an unrelated model.
    model=(args.onnx or root/'official_resources/xgoduck_runtime_arduino/python/xgoduck_walk.onnx').resolve()
    assert digest(model)==report['model_sha256'], 'Model hash does not match saved evaluation'
    runtime=root/'official_resources/xgoduck_runtime_arduino/python'
    sys.path.insert(0,str(runtime))
    import rl_core
    from policy import Policy
    assert Path(rl_core.__file__).parent==runtime
    matched=Policy(); default=Policy()
    ok,msg=matched.load(str(model),use_model_home=True); assert ok,msg
    ok,msg=default.load(str(model)); assert ok,msg
    meta=matched.session.get_modelmeta().custom_metadata_map
    home=np.array([float(x) for x in meta['default_joint_pos'].split(',')],dtype=np.float32)
    names=meta['joint_names'].split(',')
    assert float(meta['action_scale'])==matched.action_scale==1
    assert meta['observation_names'].split(',')==['base_ang_vel','projected_gravity','joint_pos','joint_vel','actions','command','head_command','body_command']
    graph=onnx.load(model)
    onnx.checker.check_model(graph)
    # Verify the graph actually includes input normalization before the MLP.
    sub,div=graph.graph.node[:2]
    initial={v.name:numpy_helper.to_array(v) for v in graph.graph.initializer}
    assert sub.op_type=='Sub' and sub.input[0]==matched.in_name
    assert div.op_type=='Div' and div.input[0]==sub.output[0]
    mean=initial[sub.input[1]].reshape(61)
    divisor=initial[div.input[1]].reshape(61)
    assert np.isfinite(mean).all() and np.isfinite(divisor).all() and (divisor>0).all()
    snapshot=yaml.load((run/'env.yaml').read_text(),Loader=yaml.BaseLoader)
    actuator=snapshot['scene']['entities']['robot']['articulation']['actuators'][0]
    header=runtime.parent/'sketch/duck_config.h'
    firmware=header.read_text()
    kp=int(re.search(r'TEMP_KP_RUN\s*=\s*(\d+)',firmware).group(1))
    kd=int(re.search(r'TEMP_KD_RUN\s*=\s*(\d+)',firmware).group(1))
    probes=[]
    all_default_delta=[]
    sources={str(Path(__file__).resolve()):digest(Path(__file__).resolve()),str(model):digest(model),str(runtime/'rl_core.py'):digest(runtime/'rl_core.py'),
             str(runtime/'policy.py'):digest(runtime/'policy.py')}
    for source in (run/'env.yaml',header,runtime.parent/'sketch/scs_bus.h',runtime.parent/'sketch/sketch.ino'):
        sources[str(source)]=digest(source)
    for condition in report['conditions']:
        file=run/f'{condition["condition"]}-trace.npz'
        data=np.load(file); obs=data['observations']; actions=data['actions']; active=data['active']
        assert obs.shape[1:]==(report['num_envs'],61)
        # All saved live steps must preserve raw action history, not the filtered target.
        np.testing.assert_allclose(obs[0,:,34:48],0,atol=0,rtol=0)
        live_pairs=active[1:] & active[:-1]
        np.testing.assert_allclose(obs[1:,:,34:48][live_pairs],actions[:-1][live_pairs],atol=0,rtol=0)
        samples=0; max_input=0.; max_raw=0.; deltas=[]
        for step in (0,100,500,999):
            if step>=len(obs): continue
            for i in np.flatnonzero(active[step]):
                x=obs[step,i]
                angles=np.zeros(15,dtype=np.float32)
                vels=np.zeros(15,dtype=np.float32)
                angles[rl_core.POLICY_INDEX]=np.rad2deg(x[6:20]+home)
                vels[rl_core.POLICY_INDEX]=np.rad2deg(x[20:34])
                matched.prev_action[:]=x[34:48]; default.prev_action[:]=x[34:48]
                rebuilt=matched.build_obs(x[3:6],x[:3],angles,vels,x[48:51],x[51:55]).copy()
                err=float(np.max(np.abs(rebuilt-x)))
                assert err<2e-6,err
                raw=matched.session.run(matched._outputs,{matched.in_name:rebuilt[None]})[0][0]
                raw_error=float(np.max(np.abs(raw-actions[step,i])))
                assert raw_error<2e-5,raw_error
                alt=default.build_obs(x[3:6],x[:3],angles,vels,x[48:51],x[51:55]).copy()
                np.testing.assert_allclose(alt[6:20]-rebuilt[6:20],home-np.deg2rad(default.home_deg),atol=2e-7,rtol=0)
                alt_raw=default.session.run(default._outputs,{default.in_name:alt[None]})[0][0]
                deltas.append(alt_raw-raw)
                max_input=max(max_input,err); max_raw=max(max_raw,raw_error); samples+=1
        delta=np.asarray(deltas); all_default_delta.append(delta)
        probes.append(dict(condition=condition['condition'],samples=samples,
            max_observation_error=max_input,max_raw_action_error=max_raw,
            default_home_action_delta_rms=float(np.sqrt(np.mean(delta**2))),
            default_home_action_delta_max=float(np.max(np.abs(delta))),
            normalized_command=((np.asarray(condition['command'])-mean[48:51])/divisor[48:51]).tolist()))
        sources[str(file)]=digest(file)
    # Validate the actual runtime filter implementation for a recorded sequence.
    # Previous raw action is forced to saved history at every step to isolate
    # the filter comparison from a hypothetical, unexecuted closed-loop rollout.
    data=np.load(run/'forward-trace.npz')
    matched.reset_action_filter()
    alpha=np.array([matched.action_alpha]*5+[matched.action_alpha_head]*5+[matched.action_alpha]*4,dtype=np.float32)
    expected=np.zeros(14,dtype=np.float32); filter_err=0.; filtered_delta=[]
    for x,raw,live in zip(data['observations'][:,0],data['actions'][:,0],data['active'][:,0]):
        if not live: break
        angles=np.zeros(15,dtype=np.float32); vels=np.zeros(15,dtype=np.float32)
        angles[rl_core.POLICY_INDEX]=np.rad2deg(x[6:20]+home)
        vels[rl_core.POLICY_INDEX]=np.rad2deg(x[20:34])
        matched.prev_action[:]=x[34:48]
        filtered=matched.infer(x[3:6],x[:3],angles,vels,x[48:51],x[51:55])
        expected=alpha*expected+(1-alpha)*raw
        filter_err=max(filter_err,float(np.max(np.abs(filtered-expected))))
        filtered_delta.append(filtered-raw)
    assert filter_err<2e-5,filter_err
    delta=np.vstack(all_default_delta)
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        model_used=str(model),recorded_model_path=report['model'],
        analysis_command=[sys.executable,*sys.argv],exit_code=0,
        scope='Stored-observation synthetic runtime feedback; interface and conditional output differences only',
        root_cause_resolved=False,hardware_verified=False,training_started=False,
        simulation_actuator=dict(kp_fw=float(actuator['kp_fw']),
            delay_lag_steps=[int(actuator['delay_min_lag']),int(actuator['delay_max_lag'])],
            physics_dt=float(snapshot['sim']['mujoco']['timestep']),decimation=int(snapshot['decimation']),
            vin_range=[float(v) for v in actuator['vin_range']]),
        firmware_run_gains=dict(kp=kp,kd=kd,source='Constants used by setRunKpKd at startup; hardware readback not performed'),
        model_training_provenance=dict(run_path=meta.get('run_path'),
            training_commit_known=False,training_seed_known=False,training_budget_known=False),
        source_sha256=sources,normalization=dict(embedded=True,command_mean=mean[48:51].tolist(),
            command_divisor=divisor[48:51].tolist(),external_normalization_required=False),
        home=[dict(joint=name,model_degrees=float(np.rad2deg(home[i])),
                   runtime_default_degrees=float(default.home_deg[i]),
                   delta_degrees=float(np.rad2deg(home[i])-default.home_deg[i])) for i,name in enumerate(names)],
        samples=sum(p['samples'] for p in probes),probes=probes,
        default_home_conditional_action_delta_rms=float(np.sqrt(np.mean(delta**2))),
        default_home_conditional_action_delta_max=float(np.max(np.abs(delta))),
        filter_replay=dict(steps=len(filtered_delta),alpha_per_joint=alpha.tolist(),max_formula_error=filter_err,
            conditional_filtered_vs_raw_action_rms=float(np.sqrt(np.mean(np.asarray(filtered_delta)**2)))),
        limitations=['Reconstructed feedback is not raw or real sensor data.',
            'Normalization statistics do not recover the model training commit, task, actuator, seed or budget.',
            'Home/filter output differences are conditional on recorded states, not new closed-loop outcomes.',
            'IMU fusion, firmware gains, communication and actuator response are not reproduced by this audit.'])
    (run/'interface-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','samples','root_cause_resolved','normalization',
        'default_home_conditional_action_delta_rms','default_home_conditional_action_delta_max','filter_replay']},indent=2))
    print('maximum reconstructed input error',max(p['max_observation_error'] for p in probes))
    print('maximum raw action replay error',max(p['max_raw_action_error'] for p in probes))


if __name__=='__main__': main()

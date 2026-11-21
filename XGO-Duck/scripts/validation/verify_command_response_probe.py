"""Independent saved-trace accounting for the bounded response diagnostic.

No physics or learning. Recompute each first episode, replay selected ONNX
inputs, check raw-history/target contracts, decode every recorded short clip,
and preserve all nineteen responses without inventing a gait pass threshold.
"""
from pathlib import Path
import json
import numpy as np
import yaml
import onnxruntime as ort
import imageio_ffmpeg
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from planar_block_metrics import block_statistic

OUT=ROOT/'records/reports/command-response'


def verify(folder):
    folder=Path(folder);plan=read(ROOT/'configs/experiments/command_response_probe.json')
    pre=read(OUT/'probe-preflight.json');report=read(folder/'evaluation.json')
    assert pre['protocol_sha256']==sha(folder/'protocol.json')
    assert report['status']=='completed' and (report['num_envs'],report['episode_seconds'],report['seed'])==(16,12,1004)
    assert report['model_sha256']==plan['model_sha256']==sha(ROOT/plan['model'])
    assert yaml.load((folder/'env.yaml').read_text(),Loader=yaml.BaseLoader)==pre['config']
    assert [x['condition'] for x in report['conditions']]==list(plan['commands'])
    for name,digest in plan['source_sha256'].items():assert sha(folder/Path(name).name)==digest,name
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/plan['model']),sess_options=options,providers=['CPUExecutionProvider'])
    names=report['model_metadata']['joint_names'].split(',');hips=[names.index(s+'_hip_roll') for s in ('left','right')]
    home=np.asarray(report['policy_home']);scale=float(report['model_metadata']['action_scale'])
    checks=[];rows=[];initial=None;total_steps=0;replayed=0;maximum_parity_error=0.
    for item in report['conditions']:
        condition=item['condition'];command=np.asarray(plan['commands'][condition]);path=folder/f'{condition}-trace.npz'
        assert sha(path)==item['trace_sha256'] and sha(folder/f'{condition}-initial.npz')==item['initial_sha256']
        with np.load(folder/f'{condition}-initial.npz') as value:start={k:value[k].copy() for k in value.files}
        if initial is None:initial={k:start[k] for k in ('qpos','qvel')}
        else:assert all(np.array_equal(start[k],initial[k]) for k in initial)
        with np.load(path) as value:t={k:value[k].copy() for k in value.files}
        active=t['active'];T,n=active.shape;assert n==16 and 0<T<=600 and active[0].all()
        total_steps+=T;assert not np.any(np.diff(active.astype(int),axis=0)>0)
        assert all(np.isfinite(value[active]).all() for name,value in t.items() if name!='active')
        np.testing.assert_allclose(t['observations'][:,:,48:51],np.broadcast_to(command,(T,16,3)),atol=1e-6,rtol=0)
        np.testing.assert_allclose(t['observations'][:,:,51:61],0.,atol=1e-6,rtol=0)
        np.testing.assert_allclose(t['observations'][0,:,34:48],0.,atol=0,rtol=0)
        np.testing.assert_allclose(t['observations'][1:,:,34:48][active[1:]],t['actions'][:-1][active[1:]],atol=0,rtol=0)
        np.testing.assert_allclose(t['requested_joint_targets'],home+scale*t['actions'],atol=1e-6,rtol=0)
        np.testing.assert_allclose(t['applied_actions'],t['actions'],atol=0,rtol=0)
        for step in range(0,T,40):
            for index in np.flatnonzero(active[step]):
                a=session.run(None,{'obs':t['observations'][step,index:index+1]})[0][0]
                b=t['actions'][step,index];np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-5)
                maximum_parity_error=max(maximum_parity_error,float(abs(a-b).max()));replayed+=1
        outcomes=[]
        for index,episode in enumerate(item['episodes']):
            valid=np.flatnonzero(active[:,index]);last=valid[-1];assert len(valid)==episode['steps']
            flags={k:bool(t[k][last,index]) for k in ('nan_state','fell_over','out_of_bounds','terminated','truncated')}
            reason=('nan_state' if flags['nan_state'] else 'fell_over' if flags['fell_over'] else
                'out_of_terrain_bounds' if flags['out_of_bounds'] else 'other_termination' if flags['terminated'] else
                'early_timeout' if last+1<600 else 'horizon_complete')
            assert episode['outcome']==reason;outcomes.append(reason)
            linear=np.linalg.norm(t['velocity'][valid,index]-command[:2],axis=1).mean(dtype=np.float64)
            yaw=np.abs(t['yaw_rate'][valid,index]-command[2]).mean(dtype=np.float64)
            displacement=float(np.linalg.norm(t['xy'][last,index]-start['xy'][index]))
            for key,value in (('linear_xy_mae_m_s',linear),('yaw_mae_rad_s',yaw),('final_displacement_m',displacement)):
                np.testing.assert_allclose(episode[key],value,atol=1e-6,rtol=1e-6)
        survival=float(np.mean(np.asarray(outcomes)=='horizon_complete'))
        np.testing.assert_allclose(survival,item['full_episode_survival'],atol=0,rtol=0)
        after=active.copy();after[:50]=False;counts=after.sum(axis=0)
        axis=int(np.argmax(abs(command)));sign=float(np.sign(command[axis]))
        values=t['yaw_rate'] if axis==2 else t['velocity'][:,:,axis]
        signed=np.divide((values*after).sum(axis=0),counts,out=np.zeros(16),where=counts>0)*sign
        padded_v=np.zeros((600,16,2),dtype=t['velocity'].dtype);padded_v[:T]=t['velocity']
        padded_a=np.zeros((600,16),bool);padded_a[:T]=active
        blocks=block_statistic(padded_v,padded_a,command)
        angles=np.rad2deg(t['joint_positions'][:,:,hips]);inward=(angles[:,:,0]<-20)&(angles[:,:,1]>20)
        fractions=np.divide((inward*after).sum(axis=0),counts,out=np.ones(16),where=counts>0)
        movie=None
        if item['video']:
            value=item['video'];video=folder/value['path'];assert sha(video)==value['sha256']
            reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);frames=sum(1 for _ in reader)
            times=np.asarray(value['frame_times_seconds']);assert frames==len(times) and frames<=76 and times[0]==0 and times[-1]<=3.
            assert meta['fps']==25 and meta['size']==(640,480)
            assert value['extra_physics_steps']==0 and value['source']=='actual-simulation-state'
            if survival==1:assert frames==76 and times[-1]==3.
            movie=dict(path=value['path'],sha256=sha(video),frames=frames,simulation_span_seconds=float(times[-1]))
        assert bool(item['video'])==(condition in plan['video_conditions'])
        rows.append(dict(condition=condition,command=command.tolist(),episodes=16,survivors=outcomes.count('horizon_complete'),
            signed_axis_speed_after1s=float(signed.mean()),axis_unit='rad/s' if axis==2 else 'm/s',
            linear_error=item['linear_xy_mae_m_s'],yaw_error=item['yaw_mae_rad_s'],
            block_error=float(blocks['mean_error_per_episode'].mean()),
            inward_fraction_after1s=float(fractions.mean()),missing_postwarmup_episodes=int((counts==0).sum()),
            valid_blocks_per_episode=blocks['valid_blocks_per_episode'].tolist(),
            median_displacement=item['final_displacement_m_median'],wall_seconds=item['wall_seconds'],
            host_profile_seconds=item['host_profile_seconds'],trace_sha256=sha(path)))
        checks.append(dict(condition=condition,steps=T,episodes=16,trace_sha256=sha(path),video=movie))
    assert total_steps<=plan['maximum_control_steps']
    result=dict(status='passed',report_sha256=sha(folder/'evaluation.json'),protocol_sha256=sha(folder/'protocol.json'),
        script_sha256=sha(Path(__file__)),conditions=checks,episodes=304,actual_control_steps=total_steps,
        maximum_transitions=plan['maximum_transitions'],exact_initial_physics_matched=True,
        replayed_inputs=replayed,maximum_inference_error=maximum_parity_error,formal_acceptance=False,
        scope='Independent first-episode, interface, model replay and full clip decode; no gait certification')
    write(folder/'verification.json',result);write(OUT/'probe-verification.json',result)
    analysis=dict(status='passed',rows=rows,model_sha256=plan['model_sha256'],evaluation=str(folder.relative_to(ROOT)),
        verification_sha256=sha(folder/'verification.json'),script_sha256=sha(Path(__file__)),
        decision='review-response-map-before-any-learning',automatic_learning=False,heldout_used=False,
        original_gait_verdicts_unchanged=True,learning_updates=0,
        profile_scope='Exclusive host sections; env stepping may enqueue GPU work whose wait appears in later reads; not GPU kernel timings',
        scope='Development response map: 16 episodes x 12 s differ from original 32 x 20 s; no retroactive pass or selected calibration acceptance')
    write(OUT/'probe-analysis.json',analysis)
    print(json.dumps(dict(status='passed',episodes=304,actual_control_steps=total_steps,replayed_inputs=replayed,
        maximum_inference_error=maximum_parity_error,decision=analysis['decision']),indent=2))
    for row in rows:print(row['condition'],row['survivors'],'/16',round(row['signed_axis_speed_after1s'],5),row['axis_unit'],flush=True)
    return result


if __name__=='__main__':
    import sys
    verify(sys.argv[1])

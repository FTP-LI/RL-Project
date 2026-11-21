"""Recompute observer throughput from all prespecified saved updates.

Check identical starts/configurations, actual head telemetry, uninterrupted
display, finite final training state, source hashes and full video decoding.
Keep both run orders and all timing spikes. No simulation, GUI or learning.
"""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import statistics

from verify_parallel_benchmark import read,sha,jsonlines,tensors,quantile

ROOT=Path(__file__).resolve().parents[2]


def verify(folder):
    import torch
    import numpy as np
    import yaml
    import imageio_ffmpeg
    from PIL import Image
    torch.set_num_threads(1)
    assert folder.parent==ROOT/'records/benchmarks'
    batch,protocol=read(folder/'batch.json'),read(folder/'protocol.json')
    assert batch['status']=='completed-awaiting-verification'
    assert batch['protocol_sha256']==sha(folder/'protocol.json')
    order=sum(protocol['ordered_passes'],[])
    assert len(batch['trials'])==len(order)==6
    parent=ROOT/'artifacts/recovery'/protocol['parent_run_id']
    original={k:yaml.load((parent/'params'/f'{k}.yaml').read_text(),Loader=yaml.BaseLoader) for k in ('env','agent')}
    rows=[];inputs={};sources=None;initial=None;first_action=None
    for index,(entry,name) in enumerate(zip(batch['trials'],order,strict=True)):
        trial=folder/entry['folder'];request,result=read(trial/'request.json'),read(trial/'result.json')
        assert entry==read(trial/'process.json') and entry['exit_code']==0
        assert entry['case']==request['case']==name and request['trial_index']==index
        assert request['settings']==protocol['cases'][name]
        assert request['protocol_sha256']==batch['protocol_sha256']
        assert result['status']=='completed' and result['additional_updates']==40 and result['cuda_graph_enabled']
        assert not result['observer_closed_early'] and not result['gait_selection_allowed']
        assert not result['formal_model_changed']
        assert sha(Path(request['plan']['checkpoint']))==protocol['checkpoint_sha256']
        if sources is None:sources=request['source_sha256']
        assert sources==request['source_sha256']
        for key,value in sources.items():assert sha(trial/'sources'/f'{key}.txt')==value,key
        reset=read(trial/'loaded-reset.json');correction=reset['correction']
        assert correction['next_iteration_label']==5000 and correction['remaining_updates']==40
        assert correction['common_step_counter']==120000
        assert correction['ppo_learning_rate_after']==request['plan']['restored_learning_rate']
        action=read(trial/'first-action.json')['sha256']
        if initial is None:initial=reset['observation_sha256'];first_action=action
        assert reset['observation_sha256']==initial and action==first_action
        config={k:yaml.load((trial/f'{k}.yaml').read_text(),Loader=yaml.BaseLoader) for k in ('env','agent')}
        assert config['agent']['run_name']==request['run_name'] and config['agent']['max_iterations']=='40'
        assert config['env']['scene']['num_envs']==config['env']['scene']['terrain']['num_envs']=='4096'
        for key in ('run_name','max_iterations'):config['agent'][key]=original['agent'][key]
        config['env']['scene']['num_envs']=original['env']['scene']['num_envs']
        config['env']['scene']['terrain']['num_envs']=original['env']['scene']['terrain']['num_envs']
        assert config==original,'Unexpected environment or agent configuration change'
        checkpoint=Path(result['checkpoint']);assert sha(checkpoint)==result['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['iter']==5039 and saved['infos']['env_state']['common_step_counter']==120960
        assert {int(v['step'].item()) for v in saved['optimizer_state_dict']['state'].values()}=={100800}
        for group in ('actor','critic'):assert int(saved[group+'_state_dict']['obs_normalizer.count'])==126812160
        assert all(torch.isfinite(t).all().item() for t in tensors(saved))
        timing=jsonlines(trial/'iteration-timing.jsonl');live=jsonlines(trial/'live-view.jsonl')
        status=read(trial/'live-view-status.json')
        assert status['status']=='observer-finished' and 'display_closed_at' not in status
        assert status['max_wall_fps']==protocol['cases'][name]['observer_max_hz']
        assert status['frames']==len(live)==timing[-1]['observer_samples_total']
        assert [r['iteration'] for r in timing]==list(range(5000,5040))
        for i,r in enumerate(timing):
            assert all(math.isfinite(v) for v in r.values()) and r['wall_seconds']>0 and r['viewer_running']
            assert abs(r['wall_seconds']-(r['end_monotonic_seconds']-r['start_monotonic_seconds']))<1e-8
            assert r['common_step_counter']==120000+(i+1)*24
            if i:assert r['start_monotonic_seconds']==timing[i-1]['end_monotonic_seconds']
            assert r['observer_seconds']>=0 and r['observer_samples']>=0
        names=status['joint_names'];indices=[names.index(j) for j in status['head_joint_names']]
        assert live[0]['common_step_counter']==120000
        steps=np.array([r['common_step_counter'] for r in live])
        assert (np.diff(steps)>0).all() and 120960-steps[-1]<=24
        for r in live:
            actual=np.array(r['head_joint_positions_rad']);target=np.array(r['head_pose_target_rad'])
            command=np.array(r['head_pose_command_rad']);velocity=np.array(r['head_joint_velocities_rad_s'])
            assert actual.shape==target.shape==command.shape==velocity.shape==(4,)
            assert np.isfinite(np.r_[actual,target,command,velocity]).all()
            np.testing.assert_allclose(actual,np.array(r['joint_positions_rad'])[indices],atol=1e-7)
            np.testing.assert_allclose(actual-target,r['head_pose_error_actuator_rad'],atol=1e-7)
        homes=np.array([np.array(r['head_pose_target_rad'])-np.array(r['head_pose_command_rad']) for r in live])
        np.testing.assert_allclose(homes,np.broadcast_to(homes[0],homes.shape),atol=1e-6)
        selected=timing[20:];wall=sum(r['wall_seconds'] for r in selected)
        assert [selected[0]['iteration'],selected[-1]['iteration']]==protocol['measured_iteration_labels']
        assert abs(wall-(selected[-1]['end_monotonic_seconds']-selected[0]['start_monotonic_seconds']))<1e-7
        observed=sum(r['observer_samples'] for r in selected)
        assert observed==timing[-1]['observer_samples_total']-timing[19]['observer_samples_total']
        telemetry=jsonlines(trial/'telemetry.jsonl')
        active=[r for r in telemetry if r['start_monotonic_seconds']>=selected[0]['start_monotonic_seconds'] and r['end_monotonic_seconds']<=selected[-1]['end_monotonic_seconds']]
        assert len(active)>=2 and all(r.get('gpu_exit_code')==0 and 'gpu_error' not in r for r in telemetry)
        graphics=read(trial/'graphics.json');renderer=graphics['egl_renderer']
        assert ('D3D12' in renderer and 'NVIDIA' in renderer) if name.startswith('gpu') else 'llvmpipe' in renderer
        video=trial/'training-env0.mp4';assert sha(video)==result['video_sha256']
        stream=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(stream);count=0
        assert meta['size']==(960,720) and meta['fps']==50
        for count,pixels in enumerate(stream,1):
            if count==101:Image.frombytes('RGB',meta['size'],pixels).save(trial/'frame-100.png')
        assert count==200
        rows.append(dict(index=index,case=name,pass_index=index//3,measured_updates=20,
            measured_transitions=20*24*4096,wall_seconds=wall,transitions_per_second=20*24*4096/wall,
            median_update_seconds=statistics.median(r['wall_seconds'] for r in selected),
            p95_update_seconds=quantile([r['wall_seconds'] for r in selected],.95),
            collection_seconds=sum(r['collection_seconds'] for r in selected),
            learning_seconds=sum(r['learning_seconds'] for r in selected),
            cpu_seconds=sum(r['cpu_seconds'] for r in selected),
            observer_seconds=sum(r['observer_seconds'] for r in selected),
            actual_state_samples_per_second=observed/wall,actual_state_samples=observed,
            simulation_seconds_per_wall_second=(20*24*.02)/wall,
            gpu_medians={k:statistics.median(r['gpu'][k] for r in active) for k in active[0]['gpu']},
            gpu_telemetry_samples=len(active),process_wall_seconds=entry['process_wall_seconds'],
            renderer=renderer,video_decoded_frames=count,head_telemetry_verified=True,
            observer_coverage_verified=True))
        for file in trial.iterdir():
            if file.is_file():inputs[str(file.relative_to(folder))]=sha(file)
        print('OBSERVER_TRIAL_VERIFIED:',json.dumps(rows[-1]),flush=True)
    rule=protocol['selection_rule'];driver_ratios=[];cap_ratios=[];cap_fps=[]
    for repeat in (0,1):
        part={r['case']:r for r in rows if r['pass_index']==repeat}
        driver_ratios.append(part['gpu4']['transitions_per_second']/part['default4']['transitions_per_second'])
        cap_ratios.append(part['gpu15']['transitions_per_second']/part['gpu4']['transitions_per_second'])
        cap_fps.append(part['gpu15']['actual_state_samples_per_second'])
    geometric=math.sqrt(math.prod(driver_ratios))
    gpu_ok=min(driver_ratios)>rule['gpu4_both_pass_ratios_above'] and geometric>=rule['gpu4_min_geometric_mean_throughput_ratio_to_default4']
    cap_ok=min(cap_ratios)>=rule['gpu15_min_each_pass_throughput_ratio_to_gpu4'] and min(cap_fps)>=rule['gpu15_min_each_pass_actual_state_samples_per_second']
    chosen='gpu15' if gpu_ok and cap_ok else ('gpu4' if gpu_ok else 'default4')
    assert sum(r['measured_transitions'] for r in rows)==protocol['measured_transitions']
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),trials=rows,
        driver_pass_ratios=driver_ratios,driver_geometric_mean_ratio=geometric,
        cap_pass_throughput_ratios=cap_ratios,cap_pass_actual_state_fps=cap_fps,
        gpu_driver_rule_passed=gpu_ok,higher_cap_rule_passed=cap_ok,rule_selected_case=chosen,
        matched_initial_observations_and_actions=True,source_sha256=sources,inputs_sha256=inputs,
        protocol_sha256=batch['protocol_sha256'],batch_sha256=sha(folder/'batch.json'),
        actual_native_present_rate_measured=False,actual_state_sample_rate_measured=True,
        cuda_initialized=False,formal_training_resumed=False,gait_selection_allowed=False,
        scope='Two ordered passes on one WSL host; actual observer state updates, not native display refresh. Initial video excluded from steady timing; no gait or hardware acceptance.')
    write=folder/'verification.json';write.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    shutil.copy2(__file__,folder/Path(__file__).name)
    print('OBSERVER_BENCHMARK_CHECK_PASSED:',json.dumps({k:v for k,v in result.items() if k not in ('trials','source_sha256','inputs_sha256')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    verify(p.parse_args().folder.resolve())

"""Verify and summarize a completed parallel-environment screen from saved data.

CPU-only: checkpoint counters/finite values, source hashes, strict configuration
comparison, all 40 timestamps, full video decode, measured telemetry and paired
throughput. Diagnostic models never enter gait selection. Retain all outliers.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def tensors(value):
    import torch
    if isinstance(value,torch.Tensor):
        yield value
    elif isinstance(value,dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value,(list,tuple)):
        for child in value:
            yield from tensors(child)


def quantile(values,p):
    ordered=sorted(values)
    position=(len(ordered)-1)*p
    lower=math.floor(position)
    return ordered[lower]+(ordered[min(lower+1,len(ordered)-1)]-ordered[lower])*(position-lower)


def verify(folder):
    import torch
    import yaml
    import imageio_ffmpeg
    torch.set_num_threads(1)
    batch=read(folder/'batch.json')
    protocol=read(folder/'protocol.json')
    assert batch['status']=='completed-awaiting-verification'
    assert sha(folder/'protocol.json')==batch['protocol_sha256']
    assert sha(folder/'benchmark_parallel_envs.py')==batch['entry_sha256']
    sizes=sum(protocol['ordered_passes'],[])
    assert len(batch['trials'])==len(sizes)==6
    root=Path(__file__).resolve().parents[2]
    parent=root/'artifacts/recovery'/protocol['parent_run_id']
    parent_config={name:yaml.load((parent/'params'/f'{name}.yaml').read_text(),Loader=yaml.BaseLoader)
                   for name in ('env','agent')}
    rows=[]
    source_hashes=None
    input_sha256={}
    for index,(entry,num_envs) in enumerate(zip(batch['trials'],sizes)):
        trial=folder/entry['folder']
        assert trial.name==f'trial-{index+1:02d}-n{num_envs}'
        assert entry['exit_code']==0 and entry==read(trial/'process.json')
        request,result=read(trial/'request.json'),read(trial/'result.json')
        assert request['trial_index']==index and request['plan']['num_envs']==num_envs
        assert request['protocol_sha256']==batch['protocol_sha256'] and request['protocol']==protocol
        assert result['status']=='completed' and result['additional_updates']==40
        assert result['num_envs']==num_envs and result['cuda_graph_enabled']
        assert not result['formal_model_changed'] and not result['gait_selection_allowed']
        assert sha(Path(request['plan']['checkpoint']))==protocol['checkpoint_sha256']
        assert request['source_sha256']['benchmark_entry']==batch['entry_sha256']
        if source_hashes is None:
            source_hashes=request['source_sha256']
        assert request['source_sha256']==source_hashes
        for key,expected in source_hashes.items():
            assert sha(trial/'sources'/f'{key}.txt')==expected
        reset=read(trial/'loaded-reset.json')
        assert reset['next_iteration_label']==5000 and reset['remaining_updates']==40
        assert reset['common_step_counter']==120000
        assert reset['ppo_learning_rate_after']==request['plan']['restored_learning_rate']
        actual={name:yaml.load((trial/f'{name}.yaml').read_text(),Loader=yaml.BaseLoader) for name in ('env','agent')}
        assert actual['env']['scene']['num_envs']==actual['env']['scene']['terrain']['num_envs']==str(num_envs)
        assert actual['agent']['run_name']==request['run_name'] and actual['agent']['max_iterations']=='40'
        assert actual['agent']['seed']=='42'
        # Normalize only the declared changes and compare the entire remaining
        # saved config, rather than checking a small subset of reward weights.
        actual['env']['scene']['num_envs']=parent_config['env']['scene']['num_envs']
        actual['env']['scene']['terrain']['num_envs']=parent_config['env']['scene']['terrain']['num_envs']
        for field in ('run_name','max_iterations'):
            actual['agent'][field]=parent_config['agent'][field]
        assert actual==parent_config,'Undeclared configuration change'
        checkpoint=Path(result['checkpoint'])
        assert sha(checkpoint)==result['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['iter']==5039 and saved['infos']['env_state']['common_step_counter']==5040*24
        assert {int(v['step'].item()) for v in saved['optimizer_state_dict']['state'].values()}=={100800}
        count=122880000+40*24*num_envs
        for group in ('actor','critic'):
            assert int(saved[group+'_state_dict']['obs_normalizer.count'].item())==count
        assert all(torch.isfinite(t).all().item() for t in tensors(saved))
        timing=jsonlines(trial/'iteration-timing.jsonl')
        assert [row['iteration'] for row in timing]==list(range(5000,5040))
        for i,row in enumerate(timing):
            assert all(math.isfinite(v) for v in row.values()) and row['wall_seconds']>0
            assert abs(row['wall_seconds']-(row['end_monotonic_seconds']-row['start_monotonic_seconds']))<1e-8
            if i:
                assert row['start_monotonic_seconds']==timing[i-1]['end_monotonic_seconds']
        selected=timing[20:]
        wall=sum(row['wall_seconds'] for row in selected)
        phase_start,phase_end=selected[0]['start_monotonic_seconds'],selected[-1]['end_monotonic_seconds']
        assert abs(wall-(phase_end-phase_start))<1e-7
        telemetry=jsonlines(trial/'telemetry.jsonl')
        assert all(t.get('gpu_exit_code')==0 and 'gpu' in t and 'gpu_error' not in t for t in telemetry)
        assert all(all(math.isfinite(v) for v in t['gpu'].values()) for t in telemetry)
        active=[t for t in telemetry if t['start_monotonic_seconds']>=phase_start and t['end_monotonic_seconds']<=phase_end]
        assert len(active)>=3,'Insufficient telemetry inside the measured interval'
        assert all(t['gpu']['memory_total_mib']>0 and t['available_ram_kib']>0 for t in telemetry)
        cpu_delta=(active[-1]['cpu_user_seconds']+active[-1]['cpu_system_seconds']
                   -active[0]['cpu_user_seconds']-active[0]['cpu_system_seconds'])
        cpu_wall=active[-1]['start_monotonic_seconds']-active[0]['start_monotonic_seconds']
        assert cpu_delta>=0 and cpu_wall>0
        video=trial/'training-env0.mp4'
        assert sha(video)==result['video_sha256']
        # Use the locked environment's existing decoder, as in the gait audit.
        # A system ffprobe/ffmpeg installation is not required. Consume every
        # decoded frame rather than relying on a container's declared count.
        reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
        metadata=next(reader)
        decoded_frames=sum(1 for _ in reader)
        assert metadata['size']==(960,720) and abs(metadata['fps']-50)<1e-6
        assert decoded_frames==200
        row=dict(trial_index=index,pass_index=index//3,num_envs=num_envs,
                 measured_updates=20,measured_transitions=20*24*num_envs,wall_seconds=wall,
                 transitions_per_second=20*24*num_envs/wall,
                 median_update_seconds=statistics.median(t['wall_seconds'] for t in selected),
                 p95_update_seconds=quantile([t['wall_seconds'] for t in selected],.95),
                 collection_seconds=sum(t['collection_seconds'] for t in selected),
                 learning_seconds=sum(t['learning_seconds'] for t in selected),
                 process_wall_seconds=entry['process_wall_seconds'],
                 measured_telemetry_samples=len(active),cpu_logical_core_percent=100*cpu_delta/cpu_wall,
                 gpu_medians={k:statistics.median(t['gpu'][k] for t in active) for k in active[0]['gpu']},
                 max_observed_gpu_memory_fraction=max(t['gpu']['memory_used_mib']/t['gpu']['memory_total_mib'] for t in telemetry),
                 minimum_observed_available_ram_kib=min(t['available_ram_kib'] for t in telemetry),
                 video_frames=200,video_sha256=sha(video),checkpoint_sha256=sha(checkpoint))
        rows.append(row)
        for name in ('request.json','result.json','process.json','loaded-reset.json','env.yaml','agent.yaml',
                     'iteration-timing.jsonl','telemetry.jsonl','training-env0.mp4','console.log'):
            input_sha256[str((trial/name).relative_to(folder))]=sha(trial/name)
        print(f'VERIFIED trial {index+1}/6: {num_envs} envs, {row["transitions_per_second"]:.2f} transitions/s',flush=True)
    comparisons=[]
    rule=protocol['selection_rule']
    for n in (1024,2048,4096):
        ratios=[]
        matched=[row for row in rows if row['num_envs']==n]
        for row in matched:
            reference=next(v for v in rows if v['num_envs']==1024 and v['pass_index']==row['pass_index'])
            ratios.append(row['transitions_per_second']/reference['transitions_per_second'])
        mean=math.exp(statistics.mean(math.log(r) for r in ratios))
        memory_ok=all(row['max_observed_gpu_memory_fraction']<=rule['maximum_observed_gpu_memory_fraction'] for row in matched)
        eligible=(n!=1024 and min(ratios)>rule['both_pass_ratios_must_exceed'] and
                  mean>=rule['minimum_geometric_mean_throughput_ratio'] and memory_ok)
        comparisons.append(dict(num_envs=n,pass_ratios=ratios,geometric_mean_ratio=mean,memory_condition_passed=memory_ok,eligible=eligible))
    candidates=[row for row in comparisons if row['eligible']]
    chosen=max(candidates,key=lambda row:row['geometric_mean_ratio'])['num_envs'] if candidates else 1024
    assert sum(row['measured_transitions'] for row in rows)==protocol['measured_transitions_all_runs']
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),trials=rows,comparisons=comparisons,
                rule_selected_num_envs=chosen,configuration_and_checkpoint_checks_passed=True,
                source_sha256=source_hashes,input_sha256=input_sha256,protocol_sha256=batch['protocol_sha256'],
                batch_sha256=sha(folder/'batch.json'),parent_checkpoint_sha256=protocol['checkpoint_sha256'],
                media_decoder=imageio_ffmpeg.get_ffmpeg_exe(),media_decoder_sha256=sha(Path(imageio_ffmpeg.get_ffmpeg_exe())),
                cuda_initialized=False,formal_budget_changed=False,gait_selection_allowed=False,
                interpretation='Two ordered passes on one WSL host; inspect clock/power/order differences. Larger batches are not equal gait-training budgets; no OS speedup or gait improvement is inferred.')
    (folder/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    shutil.copy2(__file__,folder/Path(__file__).name)
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','source_sha256','trials')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    verify(parser.parse_args().folder.resolve())

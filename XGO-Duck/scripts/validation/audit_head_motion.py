"""Quantify head motion from verified saved evaluations, without simulation.

Why: a training video can mix random head-pose commands with balance motion.
Compare the archived official reference and the completed paired repair trial
under zero head commands. The early official trace contains noisy/delayed actor
observations only: compare that same observation channel across all policies,
and report measured joint/target quantities separately where actually saved.
Keep both the full episode and a declared 1-second startup exclusion. These
descriptive diagnostics add no retrospective pass threshold or causal claim.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
HEAD = ('neck_pitch', 'head_pitch', 'head_yaw', 'head_roll')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_clip(out, times, angles):
    """Show the recorded training interval, without inferring head commands."""
    os.environ.setdefault('MPLCONFIGDIR', str(Path.home()/'XGO-Duck/cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from figure_style import apply_figure_style
    apply_figure_style(sizes=(11,10,9),font='DejaVu Sans')
    fig,axes=plt.subplots(2,1,figsize=(8.8,5.8),sharex=True)
    fig.subplots_adjust(left=.13,right=.97,bottom=.13,top=.92,hspace=.38)
    for ax,columns,labels,title in zip(axes,[(0,1),(2,3)],
            [('Neck pitch','Head pitch'),('Head yaw','Head roll')],['Pitch motion','Yaw and roll']):
        for column,label,color in zip(columns,labels,('#0072B2','#D55E00')):
            ax.plot(times,angles[:,column],label=label,color=color,linewidth=1.4)
        ax.set_title(title,pad=9)
        ax.set_ylabel('Joint angle (deg)')
        ax.legend(loc='best',ncol=2)
        ax.grid(axis='y',alpha=.15)
        ax.margins(x=.02,y=.20)
    axes[-1].set_xlabel('Time within recorded control-step interval (s)')
    axes[-1].set_xticks(np.arange(0,4.01,1))
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
           if t.get_visible() and t.get_text().strip()]
    collisions=[]
    for i,(a,ab) in enumerate(texts):
        assert fig.bbox.contains(ab.x0,ab.y0) and fig.bbox.contains(ab.x1,ab.y1),a.get_text()
        for b,bb in texts[i+1:]:
            if ab.overlaps(bb):collisions.append([a.get_text(),b.get_text()])
    for ax in axes:
        own=set(ax.get_xticklabels(which='both')+ax.get_yticklabels(which='both'))
        for spine in ax.spines.values():
            if spine.get_visible():
                box=spine.get_window_extent(renderer)
                for t,tbox in texts:
                    if t not in own and tbox.overlaps(box):collisions.append([t.get_text(),'spine'])
    assert not collisions,collisions
    png=out/'training-head-motion.png'
    with plt.rc_context({'savefig.bbox':None}):fig.savefig(png,dpi=300,bbox_inches=None,facecolor='white')
    image=Image.open(png);fig.canvas.draw();renderer=fig.canvas.get_renderer();scale=image.width/fig.bbox.width
    for ax,name in zip(axes,('pitch-panel.png','yaw-roll-panel.png')):
        box=ax.get_tightbbox(renderer).expanded(1.02,1.08)
        pixels=(max(0,int(box.x0*scale)),max(0,int(image.height-box.y1*scale)),
                min(image.width,int(np.ceil(box.x1*scale))),min(image.height,int(np.ceil(image.height-box.y0*scale))))
        image.crop(pixels).save(out/name)
    plt.close(fig)
    (out/'plot-check.json').write_text(json.dumps(dict(status='passed',text_collisions=collisions,
        png_sha256=sha(png),samples=len(times),smoothing_applied=False,visual_inspection='pending'),indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--extra-reference', type=Path,
                        help='Add one completed current official evaluation; preserve the original analysis')
    args = parser.parse_args()
    batch = args.batch.resolve()
    assert batch.parent == ROOT/'records/experiments'
    post = read(batch/'postprocessing.json')
    assert post['status'] == 'completed', 'Finish the existing fixed evaluations first'
    assert read(batch/'verification.json')['status'] == 'passed'
    reference = ROOT/'records/evaluations/official_walk_reference_20261101T005022_s1001'
    # This preserved independent check binds the early report and its traces.
    reference_check = ROOT/'records/evaluations/custom_onnx_reference_20261101T144436_s1001/reference-comparison/reference-verification.json'
    runs = [('archived_official', reference, reference_check)]
    for arm in ('control', 'target_envelope'):
        folder = Path(post['arms'][arm]['evaluation_record'])
        runs.append((arm, folder, folder/'verification.json'))
    if args.extra_reference:
        extra = args.extra_reference.resolve()
        assert extra.parent == ROOT/'records/evaluations'
        runs.append(('current_official', extra, extra/'verification.json'))
    source_record = ROOT/'records/experiments'/post['arms']['control']['run_id']
    limits_source = source_record/'live-view-status.json'
    limits = read(limits_source)
    training_cfg_path = source_record/'params/env.yaml'
    cfg = yaml.load(training_cfg_path.read_text(), Loader=yaml.BaseLoader)
    stages = cfg['curriculum']['head_pose_range']['params']['range_stages']
    selected = max((s for s in stages if int(s['step']) <= 120000), key=lambda s:int(s['step']))
    ranges = np.array(selected['ranges'], dtype=float)
    assert ranges.shape == (4, 2)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in
              (batch/'postprocessing.json', batch/'verification.json', limits_source, training_cfg_path)}
    # The current VS Code breadcrumb identified this control-arm saved clip.
    # Observer samples cover the same control-step interval, but no head-command
    # values were recorded there. Do not invent per-frame command attribution.
    media_path = source_record/'media-check/check.json'
    media = read(media_path)
    clip = next(c for c in media['clips'] if c['relative_control_step'] == 4800)
    assert media['status'] == 'passed' and clip['complete_clip']
    assert sha(ROOT/clip['file']) == clip['sha256']
    parent_step = read(source_record/'resume-plan.json')['common_step_counter']
    live_path = source_record/'live-view.jsonl'
    live = [json.loads(line) for line in live_path.read_text().splitlines()]
    clip_rows = [x for x in live if parent_step+4800 <= x['common_step_counter'] <= parent_step+5000]
    assert clip_rows
    clip_indices = [limits['joint_names'].index(name) for name in HEAD]
    clip_angles = np.rad2deg(np.array([x['joint_positions_rad'] for x in clip_rows])[:,clip_indices])
    clip_summary = dict(video=clip['file'],video_sha256=clip['sha256'],
        decoded_frames=clip['decoded_frames'],fps=clip['fps'],
        observer_samples=len(clip_rows),common_step_range=[clip_rows[0]['common_step_counter'],clip_rows[-1]['common_step_counter']],
        head_joint_names=HEAD,absolute_angle_min_median_max_deg=np.quantile(clip_angles,[0,.5,1],axis=0).tolist(),
        simultaneous_head_commands_saved=False,frame_exact_synchronization_verified=False,
        scope='Control-arm training env 0, same control-step interval as the reported saved clip; not the fixed final policy')
    for p in (media_path, ROOT/clip['file'], live_path, source_record/'resume-plan.json'):
        inputs[str(p.relative_to(ROOT))] = sha(p)
    summaries, episodes, configs = [], [], set()
    head_range_check = None
    for label, folder, checked in runs:
        report, verification = read(folder/'evaluation.json'), read(checked)
        assert report['status'] == 'completed' and verification['status'] == 'passed'
        assert verification['report_sha256'] == sha(folder/'evaluation.json')
        assert report['seed'] == 1001 and report['num_envs'] == 32 and report['episode_seconds'] == 20
        assert report['model_metadata']['observation_names'].split(',') == [
            'base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel',
            'actions', 'command', 'head_command', 'body_command']
        names = report['model_metadata']['joint_names'].split(',')
        assert names == limits['joint_names']
        index = [names.index(name) for name in HEAD]
        home = np.array([float(v) for v in report['model_metadata']['default_joint_pos'].split(',')])
        if 'policy_home' in report:
            np.testing.assert_allclose(home, report['policy_home'], atol=1e-7)
        hard = np.array(limits['joint_ranges_rad'])[index]
        configs.add(sha(folder/'env.yaml'))
        for p in (folder/'evaluation.json', checked, folder/'env.yaml', folder/'evaluate_fixed_policy.py'):
            inputs[str(p.relative_to(ROOT))] = sha(p)
        if label == 'control':
            sim_home = np.asarray(report['simulation_home'])
            assert np.all(sim_home == sim_home[0])
            desired = sim_home[0,index,None] + ranges
            head_range_check = dict(joint_names=HEAD,
                curriculum_step=int(selected['step']),
                command_delta_ranges_deg=np.rad2deg(ranges).tolist(),
                desired_joint_ranges_deg=np.rad2deg(desired).tolist(),
                compiled_hard_limits_deg=np.rad2deg(hard).tolist(),
                command_upper_excess_deg=np.rad2deg(np.maximum(desired[:,1]-hard[:,1],0)).tolist(),
                scope='Configured command endpoints versus compiled actuator-joint limits; not recovered commands for the displayed video')
        for condition in report['conditions']:
            name = condition['condition']
            trace = folder/f'{name}-trace.npz'
            bound = next(c for c in verification['checks'] if c['condition'] == name)
            assert bound['trace_sha256'] == sha(trace)
            inputs[str(trace.relative_to(ROOT))] = sha(trace)
            with np.load(trace, allow_pickle=False) as data:
                active = data['active']
                assert active.dtype == bool and active.shape[1] == 32
                np.testing.assert_allclose(data['observations'][:,:,51:61], 0, atol=1e-6)
                offsets = np.rad2deg(data['observations'][:,:,6:20][:,:,index])
                observed_speed = np.rad2deg(data['observations'][:,:,20:34][:,:,index])
                actual_available = all(k in data for k in ('joint_positions','joint_velocities','requested_joint_targets'))
                if actual_available:
                    actual = np.rad2deg(data['joint_positions'][:,:,index] - home[index])
                    actual_speed = np.rad2deg(data['joint_velocities'][:,:,index])
                    target = data['requested_joint_targets'][:,:,index]
                    qpos = data['joint_positions'][:,:,index]
                    if 'applied_actions' in data:
                        np.testing.assert_allclose(data['requested_joint_targets'],
                            home+float(report['model_metadata']['action_scale'])*data['applied_actions'], atol=1e-5)
                # Observations are saved before each step; measured joints after it.
                # Excluding 50 complete steps gives observation times 1..19.98 s
                # and measured times 1.02..20 s at this 50-Hz control rate.
                for window, start in [('full_episode',0), ('after_first_second',round(1/report['step_dt']))]:
                    mask = active.copy(); mask[:start] = False
                    for column, joint in enumerate(HEAD):
                        values = offsets[:,:,column][mask]
                        speeds = observed_speed[:,:,column][mask]
                        assert len(values) and np.isfinite(values).all() and np.isfinite(speeds).all()
                        per_episode = []
                        for env in range(active.shape[1]):
                            selected_mask = mask[:,env]
                            if not selected_mask.any(): continue
                            q = offsets[selected_mask,env,column]
                            row = dict(model=label,condition=name,window=window,joint=joint,env=env,
                                samples=len(q),mean_offset_deg=float(q.mean()),
                                rms_offset_deg=float(np.sqrt(np.mean(q*q))),
                                oscillation_rms_about_episode_mean_deg=float(q.std()),
                                absolute_speed_p95_deg_s=float(np.quantile(abs(observed_speed[selected_mask,env,column]),.95)))
                            episodes.append(row); per_episode.append(row)
                        summary = dict(model=label,condition=name,window=window,joint=joint,
                            episodes=len(per_episode),active_samples=int(mask.sum()),
                            observed_offset_min_median_max_deg=np.quantile(values,[0,.5,1]).tolist(),
                            observed_absolute_speed_p95_deg_s=float(np.quantile(abs(speeds),.95)),
                            episode_mean_offset_median_deg=float(np.median([x['mean_offset_deg'] for x in per_episode])),
                            episode_oscillation_rms_median_deg=float(np.median([x['oscillation_rms_about_episode_mean_deg'] for x in per_episode])),
                            actual_joint_measurements_available=actual_available)
                        if actual_available:
                            measured = actual[:,:,column][mask]
                            measured_speed = actual_speed[:,:,column][mask]
                            near = (qpos[:,:,column] <= hard[column,0]+np.deg2rad(1)) | (qpos[:,:,column] >= hard[column,1]-np.deg2rad(1))
                            outside = (target[:,:,column] < hard[column,0]) | (target[:,:,column] > hard[column,1])
                            summary.update(actual_offset_min_median_max_deg=np.quantile(measured,[0,.5,1]).tolist(),
                                actual_absolute_speed_p95_deg_s=float(np.quantile(abs(measured_speed),.95)),
                                actual_within_one_degree_of_limit_fraction=float(near[mask].mean()),
                                requested_target_outside_hard_limit_fraction=float(outside[mask].mean()))
                        summaries.append(summary)
    assert len(configs) == 1, 'Evaluation environments differ'
    out = (extra if args.extra_reference else batch)/'head-motion'
    out.mkdir(exist_ok=not bool(args.extra_reference))
    times=np.array([x['common_step_counter']-(parent_step+4800) for x in clip_rows])*0.02
    if not args.extra_reference:
        np.savetxt(out/'training-clip-joints.csv',np.column_stack((times,clip_angles)),delimiter=',',
                   header='time_s,'+','.join(HEAD),comments='')
        plot_clip(out,times,clip_angles)
    with (out/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(episodes[0]));writer.writeheader();writer.writerows(episodes)
    shutil.copy2(__file__,out/Path(__file__).name)
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)),inputs_sha256=inputs,
        episode_csv_sha256=sha(out/'episodes.csv'),head_joint_names=HEAD,
        observation_channel_includes_noise_and_velocity_delay=True,
        first_second_exclusion_is_additional_not_a_replacement_for_full_episode=True,
        head_and_body_commands_verified_zero=True,matching_environment_sha256=next(iter(configs)),
        training_command_range_check=head_range_check,displayed_training_clip=clip_summary,results=summaries,
        head_motion_acceptance_threshold_defined=False,balance_causality_verified=False,
        simulation_started=False,training_started=False,
        scope='Saved descriptive head-motion comparison; archived official training history differs; not a reward ablation or proof that head motion is necessary for balance')
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed',output=str(out),rows=len(summaries),episode_rows=len(episodes),training_command_range_check=head_range_check),indent=2))
    for item in summaries:
        if item['condition']=='forward' and item['window']=='after_first_second': print(json.dumps(item))


if __name__ == '__main__':
    main()

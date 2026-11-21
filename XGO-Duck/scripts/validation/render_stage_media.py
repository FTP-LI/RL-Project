"""Incremental authentic-video atlas; preserve the earlier 131-clip review.

Inventory closed, published project recordings absent from the historical manifest.
Fully decode each source, select two declared temporal frames, and keep source
hashes, frame indices, labels and raw sizes. No simulated or generated frames.
The overview covers every included clip; full-size paired cards remain in the
HTML index. The actual selected environment never replaces all-episode metrics.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import html
import json
import math
import os
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'records/reports/stage-media'
DOC=ROOT/'docs/reports'
WALK_YAW_CLIPS = {
    'duck_walk_yaw-pulse_': ('前进脉冲', {'pulse-yaw': '命令脉冲'}),
    'duck_walk_yaw-history_': ('同目标偏航', {'command-pulse': '命令脉冲', 'actuator-pulse': '执行器脉冲'}),
    'duck_walk_actuator-feedback_': ('前进闭环', {'actuator-feedback': '执行器反馈'}),
}
PULSE_PROOFS = {
    'duck_walk_yaw-pulse_': 'common-walk-yaw-pulse',
    'duck_walk_yaw-history_': 'common-walk-yaw-history-probe',
}
GETUP_VIDEO_PROOFS = {
    ('duck_getup_local-ppo_', '_s1061_'): 'getup-local-ppo-20261110T011523/evaluation-analysis.json',
    ('duck_getup_early-support-ppo_', '_s1084_'): 'getup-early-support-ppo-functional/analysis.json',
    ('duck_getup_side-to-supine_', '_s1087_'): 'getup-side-to-supine-replay/evaluation/analysis.json',
    ('duck_getup_right-upright-ppo_', '_s1102_'): 'getup-right-upright-ppo/evaluation/analysis.json',
    ('duck_getup_pose-route_', '_s1062_'): 'getup-pose-routing/seed1062-analysis.json',
    ('duck_getup_pose-route_', '_s1063_'): 'getup-pose-routing/seed1063-analysis.json',
    ('duck_getup_pose-route_', '_s2062_'): 'getup-pose-routing/heldout/seed2062-analysis.json',
    ('duck_getup_upright-recovery_', '_s1093_'): 'getup-upright-recovery-pair/seed1093-analysis.json',
    ('duck_getup_upright-recovery_', '_s1094_'): 'getup-upright-recovery-pair/seed1094-analysis.json',
}
ROLL_VIDEO_PROOFS = {
    'duck_roll_endpoint-readback_development_s1025_': ('滚翻接触读数', 'roll-endpoint-readback/analysis.json'),
    'duck_roll_endpoint-confirmation_development-b_s1026_': ('滚翻开发确认', 'roll-endpoint-confirmation/development-b-analysis.json'),
    'duck_roll_endpoint-confirmation_heldout_s2026_': ('滚翻独立留出', 'roll-endpoint-confirmation/heldout-analysis.json'),
}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(ROOT))


def short(run):
    if 'shared-release-terrain' in run:return '起身同一释放·官方重复'
    if 'roll_contact-clock' in run:return '官方滚翻·接触计时'
    if 'recovery-retention' in run:
        return '起身动作保持'+('·对照组' if '_control_' in run else '·候选组')
    if 'supine-weighted' in run:
        return '仰卧释放训练'+('·等权对照' if '_control_' in run else '·加权候选')
    if 'contact-duration-value' in run:return '起身持续接触·价值准备'
    if 'persistent-contact' in run:
        if '_original_' in run:return '起身持续接触·官方模型'
        if '_control_' in run:return '起身持续接触·原奖励对照'
        return '起身持续接触·惩罚候选'
    if 'terminal-reset' in run:return '共同接口七指令·首次采样修复'
    if 'common-projection' in run:return '共同接口七指令·合法目标'
    if 'anchor-scale' in run or 'anchor-value' in run:
        return '起身新种子·零更新'
    if 'success-anchor' in run:
        if '_original_' in run:return '起身成功动作保持·固定官方模型'
        if '_control_' in run:return '起身成功动作保持·原奖励训练'
        return '起身成功动作保持·保持约束训练'
    if 'rng-preserved' in run:
        if '_development-a_' in run:return '官方滚翻独立确认·开发A'
        if '_development-b_' in run:return '官方滚翻独立确认·开发B'
        return '官方滚翻独立确认·留出'
    if 'early-guard' in run:
        return '起身早期解困'+('·官方原样对照' if '_control_' in run else '·持续卡住修正')
    if 'late-feasibility' in run:
        if '_original_' in run:return '起身末段可达目标·官方模型'
        if '_control_' in run:return '起身末段可达目标·原奖励对照'
        return '起身末段可达目标·辅助损失候选'
    if 'interface-value' in run:
        return '共同接口价值预检·行为固定'+('·读取修复' if 'readback' in run else '·接口中止')
    if 'neck-home-order' in run:
        if '_neck-home_' in run:return '颈部归位顺序·颈部归位起身'
        if '_ordered-stand_' in run:return '颈部归位顺序·归位后站立'
        return '颈部归位顺序·共同接口起身'
    if 'stand-handoff' in run:return '起身接力站立'+('·共同接口对照' if '_control_' in run else '·候选')
    if 'neck-relief' in run:return '起身末段颈部解困'+('·对照' if '_control_' in run else '·候选')
    if 'release-coverage' in run:return '起身姿态覆盖'+('·官方模型' if '_original_' in run else '·原奖励对照')
    if 'late-contact-eval_original' in run:return '起身接触奖励·官方模型'
    if 'late-contact' in run:return '起身接触奖励'+('·对照' if '_control_' in run else '·候选')
    if 'value-bootstrap' in run:return '起身价值预检·策略固定'
    if 'contact-isolation' in run:
        return '起身碰撞诊断·全身接触' if '_full_' in run else '起身碰撞诊断·原掩码'
    if 'common-plant' in run:
        if run.startswith('duck_walk_'):return '共同环境·行走'
        if run.startswith('duck_getup_'):return '共同环境·起身'
        if run.startswith('duck_roll_'):return '共同环境·滚翻'
    if run.startswith('duck_roll_official-reference_'):return '官方滚翻'
    if 'top-contact-eval_original' in run:return '滚翻同条件官方模型'
    if 'adaptation-screen_original' in run:return '滚翻新场景·官方模型'
    if 'adaptation-screen_adapted' in run:return '滚翻新场景·原奖励短训'
    if 'completion-frontier-eval_original' in run:return '滚翻完成门槛·官方模型'
    if 'completion-frontier-recovery' in run:return '滚翻完成门槛·恢复候选'
    if 'stand-preparation_prepared' in run:return '站立一秒后滚翻'
    if 'stand-preparation_control' in run:return '直接触发滚翻'
    if 'angle-feature-eval_original' in run:return '滚翻角度对照·官方模型'
    changes={'hip-limit':'髋约束','observer-stop':'观察停止','observer-resume':'观察恢复','observer-video':'录像开销',
        'observer-control':'无录像对照','official-actor-bootstrap':'官方初始化','official-adaptation-recovery':'适应恢复',
        'official-adaptation':'直接适应','velocity-precision':'速度奖励宽度','planar-filter':'速度时间统计','command-response':'指令响应',
        'command-mapping':'输入映射','straight-parallel':'前后走独立确认','gyro-turn-screen':'转向反馈与平移',
        'seven-command':'七指令独立确认','user-':'用户练习','action-memory':'滚翻动作滤波','top-contact':'滚翻头顶奖励',
        'completion-frontier':'滚翻完成门槛',
        'angle-feature':'滚翻角度输入',
        'latched-corrective-ppo':'起身纠正回放','official-reference':'官方起身','lateral-coverage':'侧卧覆盖','random-drop':'随机跌落'}
    changes.update({'reachable_head':'可达颈部指令','head_command_repair_control':'头部原样对照','reset-pose':'滚翻起点'})
    for key,label in changes.items():
        if key in run:return label+('·对照' if '_control_' in run else ('·候选' if '_candidate_' in run else ''))
    return run.replace('custom_onnx_','').replace('_s1001','')


def inventory():
    old=read(ROOT/'records/reports/training-review/video-inventory.json')
    known={r['path'] for r in old['sources']};clips=[];excluded=[]
    for folder in sorted((ROOT/'records/experiments').iterdir()):
        if not folder.is_dir():continue
        state=read(folder/'state.json') if (folder/'state.json').exists() else {}
        listed={x['file']:x for x in state.get('clips',[]) if x.get('frames',0)>1}
        closed=read(folder/'clips.json') if (folder/'clips.json').exists() else []
        backward_value = folder.name.startswith('duck_walk_shared-value_') and (folder/'verification.json').exists()
        if backward_value:
            proof = read(folder/'verification.json')
            assert proof['status']=='passed-independent-native-backward-value-readback'
            checked, = [clip for clip in closed if clip['complete_clip'] and sha(folder/clip['file'])==proof['clip_sha256']]
            listed[checked['file']] = checked
        backward_pair = folder.name.startswith('duck_walk_backward-focus-ppo_') and (folder/'verification.json').exists()
        if backward_pair:
            proof = read(folder/'verification.json')
            assert proof['status'] in ('passed-native-backward-paired-arm',
                                       'passed-native-backward-paired-arm-with-exit-status-gap')
            checked, = [clip for clip in closed if clip['complete_clip'] and sha(folder/clip['file'])==proof['video_sha256']]
            listed[checked['file']] = checked
        support_value = folder.name.startswith('duck_walk_right-support-shared-value_')
        right_value = folder.name.startswith(('duck_walk_right-startup-shared-value_',
            'duck_walk_right-support-shared-value_')) and (folder/'verification.json').is_file()
        support_pair = folder.name.startswith('duck_walk_right-support-ppo_')
        right_pair = folder.name.startswith(('duck_walk_right-startup-ppo_',
            'duck_walk_right-support-ppo_')) and (folder/'verification.json').is_file()
        if right_value or right_pair:
            proof = read(folder/'verification.json')
            if right_value:
                assert proof['status']=='passed-independent-native-right-turn-value-readback'
                digest = proof['clip_sha256']
            else:
                assert proof['status']=='passed-native-right-turn-ppo-arm-readback'
                digest = proof['video_sha256']
            checked, = [clip for clip in closed if clip['complete_clip'] and sha(folder/clip['file'])==digest]
            listed[checked['file']] = checked
        observation = folder.name.startswith('duck_getup_observation_s3061_') and state.get('status') == 'completed'
        if observation:
            proof = read(folder/'windows-view-verification.json')
            assert proof['status'] == 'completed'
            observed = {row.get('clip') for row in proof['windows_reader_observations']}
            listed.update({clip['file']:clip for clip in closed if clip['complete_clip'] and clip['file'] in observed})
        support_stage = next((stage for stage in ('clock', 'value', 'native-reward')
                              if folder.name.startswith(f'duck_getup_early-support-{stage}_')), None)
        if support_stage and state.get('status') == 'completed':
            proof = read(ROOT/f'records/reports/getup-early-support-{support_stage}/analysis.json')
            assert proof['status'] == {'clock': 'passed-actual-cuda-reward-clock',
                                       'value': 'passed-integrity',
                                       'native-reward': 'passed-actual-native-reward-preflight'}[support_stage]
            verified = proof['video']
            assert proof['record'] == rel(folder) and verified['frames'] == 201 and verified['seconds'] == 8.
            assert sha(ROOT/verified['path']) == verified['sha256']
            selected = [clip for clip in closed if clip['complete_clip'] and rel(folder/clip['file']) == verified['path']]
            assert len(selected) == 1 and selected[0]['frames'] == verified['frames']
            listed[selected[0]['file']] = selected[0]
        right_probe = next((stage for stage,seed in (('clock',1098),('value',1099),('value-coverage',1100))
                            if folder.name.startswith(f'duck_getup_right-upright-{stage}_development_s{seed}_')),None)
        if right_probe and state.get('status')=='completed':
            proof = read(ROOT/f'records/reports/getup-right-upright-{right_probe}/analysis.json')
            assert proof['record']==rel(folder) and proof['video']['seconds']==8.
            checked = proof['video']
            clip, = [item for item in closed if item['complete_clip'] and rel(folder/item['file'])==checked['path']]
            assert clip['frames']==checked['frames']==201 and sha(ROOT/checked['path'])==checked['sha256']
            listed[clip['file']] = clip
        single={x['file']:x for x in closed if x.get('frames')==1}
        for video in sorted(folder.rglob('*.mp4')):
            if str(video) in known:continue
            clip=listed.get(video.name)
            if not clip or state.get('status') in ('running','training','initializing','ready','starting','stop-requested'):
                if video.name in single and state.get('status')=='failed-preserved':
                    excluded.append(dict(path=rel(video),reason='仅t=0单帧启动文件；没有动作过程，保留为失败证据',
                        frames=1,source_sha256=sha(video),kind='startup-single-frame'))
                else:excluded.append(dict(path=rel(video),reason='not a published closed clip from a stopped run'))
                continue
            complete=bool(clip.get('complete_clip'))
            label=short(folder.name)
            if not complete:label+=f'·部分片段 {clip["frame_times_seconds"][-1]:.2f} s'
            kind='固定价值检查' if ('anchor-scale' in folder.name or 'anchor-value' in folder.name or 'contact-duration-value' in folder.name) else '训练'
            action_frames = (dict(event_times=[2.,8.], event_labels=['释放状态','重置前终态'])
                             if any(name in folder.name for name in ('supine-weighted', 'recovery-retention', 'latched-corrective-ppo'))
                             and complete else {})
            if backward_value:
                kind, label = '固定价值检查', '后退共享价值·策略固定'
                action_frames = dict(event_times=[2.,4.], event_labels=['停滞标记时点','记录终态'])
            if backward_pair:
                label = '后退 PPO·' + ('停滞加权' if '_candidate_' in folder.name else '等权对照')
                action_frames = dict(event_times=[2.,4.], event_labels=['停滞标记时点','记录终态'])
            if right_value:
                kind = '固定价值检查'
                label = '右转支撑奖励·共享价值' if support_value else '右转共享价值·策略固定'
            if right_pair:
                if support_pair:
                    label = '右转支撑 PPO·' + ('启动奖励' if '_candidate_' in folder.name else '原奖励')
                else:
                    label = '右转启动 PPO·' + ('欠速惩罚' if '_candidate_' in folder.name else '原奖励')
            if right_value or right_pair:
                action_frames = dict(event_times=[1.,4.], event_labels=['启动观察','启动窗口终态'])
            if observation:
                kind, label = '固定评估', '起身运行观察'
                action_frames = dict(event_times=[2.,11.], event_labels=['释放状态','记录终态'])
            if support_stage:
                kind, label = {'clock': ('固定评估', '起身奖励时钟'),
                               'value': ('固定价值检查', '起身价值起点'),
                               'native-reward': ('固定评估', '起身奖励读回')}[support_stage]
                action_frames = dict(event_times=[2.,8.], event_labels=['释放状态','重置前终态'])
            if right_probe:
                kind,label = {'clock':('固定评估','右侧起身奖励时钟'),
                              'value':('固定价值检查','右侧起身价值覆盖'),
                              'value-coverage':('固定价值检查','右侧起身扩展价值覆盖')}[right_probe]
                action_frames = dict(event_times=[2.,8.],event_labels=['释放状态','重置前终态'])
            if folder.name.startswith(('duck_getup_early-support-ppo-readback_', 'duck_getup_early-support-ppo-native_',
                                       'duck_getup_side-to-supine-ppo_', 'duck_getup_right-upright-ppo_')):
                role = state['arm']
                title = ('右侧起身 PPO' if folder.name.startswith('duck_getup_right-upright-ppo_') else
                         '起身回放' if folder.name.startswith('duck_getup_side-to-supine-ppo_') else '起身 PPO')
                label = f'{title} '+{'control':'对照','candidate':'候选'}[role]
                if complete:
                    action_frames = dict(event_times=[2.,8.],event_labels=['释放状态','重置前终态'])
                else:
                    label += ' · 中途停止'
                    action_frames = dict(event_times=[2.,clip['frame_times_seconds'][-1]],event_labels=['释放状态','最后保存帧'])
            if 'recovery-retention' in folder.name and state.get('status')=='stopped-review-required':
                stop = read(ROOT/'records/reports/getup-recovery-retention/stop-verification.json')
                assert stop['status']=='verified-guard-stop-saved-data-only'
                assert stop['control_record']==rel(folder) and any(
                    item['path']==rel(video) and item['sha256']==sha(video) for item in stop['videos'])
                label+='·停止前首个动作'
            clips.append(dict(path=video,kind=kind,run=folder.name,label=label,complete_clip=complete,
                times=clip['frame_times_seconds'],source_steps=clip['common_steps'],condition=None,**action_frames))
    for folder in sorted((ROOT/'records/evaluations').iterdir()):
        if not folder.is_dir():continue
        if folder.name.startswith(('duck_getup_archived-policies_qualified_', 'duck_getup_archived-policies_replication_',
                                   'duck_getup_teacher-head_development_', 'duck_getup_teacher-head-latch_development_s1163_',
                                   'duck_getup_latched-corrective-ppo_development_s1164_', 'duck_getup_dual-pose-route_development_s1166_',
                                   'duck_getup_dual-pose-route_development_s1167_', 'duck_getup_dual-pose-route_holdout_s2064_',
                                   'duck_getup_right-robustness_development_s1168_', 'duck_getup_right-robustness_holdout_s2065_')):
            review = {
                'duck_getup_archived-policies_qualified_': ('归档起身', 'getup-archived-policy-comparison/s1160-qualified'),
                'duck_getup_archived-policies_replication_': ('起身复验', 'getup-archived-policy-comparison/s1161'),
                'duck_getup_teacher-head_development_': ('起身末层对照', 'getup-teacher-head-evaluation/s1162'),
                'duck_getup_teacher-head-latch_development_s1163_': ('起身仰卧锁存', 'getup-teacher-head-evaluation/s1163-latch'),
                'duck_getup_latched-corrective-ppo_development_s1164_rng_': ('起身纠正回放', 'getup-latched-head-corrective-ppo/evaluation/rng-guarded'),
                'duck_getup_latched-corrective-ppo_development_s1164_': ('起身纠正回放', 'getup-latched-head-corrective-ppo/evaluation'),
                'duck_getup_dual-pose-route_development_s1166_': ('双姿态起身分流', 'getup-dual-pose-route'),
                'duck_getup_dual-pose-route_development_s1167_': ('双姿态起身复验', 'getup-dual-pose-route/replication'),
                'duck_getup_dual-pose-route_holdout_s2064_': ('双姿态起身留出', 'getup-dual-pose-route/holdout'),
                'duck_getup_right-robustness_development_s1168_': ('右侧起身开发', 'getup-right-robustness/development'),
                'duck_getup_right-robustness_holdout_s2065_': ('右侧起身独立留出', 'getup-right-robustness/holdout'),
            }
            title,evidence = next(value for prefix,value in review.items() if folder.name.startswith(prefix))
            receipt = ROOT/'records/reports'/evidence/'analysis.json'
            if not receipt.is_file():continue
            proof = read(receipt)
            assert proof['status']=='passed-independent-archived-policy-verification' and ROOT/proof['record']==folder
            names = {'reference-a':'参照 A','reference-b':'参照 B','control':'训练对照',
                     'prior-reward-continuation':'奖励续训','side-to-supine-replay':'侧卧到仰卧回放','candidate':'末层候选'}
            if title in ('双姿态起身分流', '双姿态起身复验', '双姿态起身留出', '右侧起身开发', '右侧起身独立留出'):
                names['candidate'] = '右侧专用策略'
            for row in read(folder/'evaluation.json')['conditions']:
                role,condition,clip = (row[k] for k in ('role','condition','video'))
                arm = proof['arms'][role]
                checked = arm['video']
                video,world,times = folder/clip['path'],checked['world'],clip['frame_times_seconds']
                assert checked['source_states_verified'] and ROOT/checked['path']==video and sha(video)==checked['sha256']
                assert world==clip['environment_index'] and clip['complete_clip']
                pose = arm['timely']['episodes'][world]['release_pose']
                clips.append(dict(path=video,kind='固定评估',run=folder.name,label=f'{title}·{names[role]}',
                    complete_clip=True,times=times,source_steps=[round(t/.02) for t in times],condition=condition,
                    release_pose=pose,event_times=[2.,11.],event_labels=[f'共同释放 · {pose}','持稳窗口结束'],
                    environment_index=world,run_status='completed'))
            continue
        if folder.name.startswith('duck_getup_hard-start_'):
            receipt = ROOT/'records/reports/getup-hard-start-screen/analysis.json'
            if not receipt.is_file():continue
            proof = read(receipt)
            assert proof['status']=='passed-independent-hard-start-screen' and ROOT/proof['record']==folder
            state = read(folder/'evaluation.json')
            names = {'challenge':'困难初态','matched':'匹配初态','nominal':'原始分布'}
            for role,title in names.items():
                checked,clip = proof['videos'][role],state['videos'][role]
                video = folder/clip['path']
                assert checked['source_states_verified'] and sha(video)==checked['sha256']
                assert ROOT/checked['path']==video and checked['world']==clip['environment_index']
                times,world = clip['frame_times_seconds'],checked['world']
                clips.append(dict(path=video,kind='固定评估',run=folder.name,label=f'仰卧筛查·{title}',
                    complete_clip=True,times=times,source_steps=[round(t/.02) for t in times],condition='supine',
                    release_pose=checked['release_pose'],event_times=[2.,11.],event_labels=['释放状态','持稳窗口结束'],
                    environment_index=world,run_status='completed'))
            continue
        if folder.name.startswith('duck_getup_second-attempt_'):
            receipt = ROOT/'records/reports/getup-second-attempt-trial/analysis.json'
            if not receipt.is_file():continue
            proof = read(receipt)
            assert proof['status']=='passed-independent-second-attempt-trial' and ROOT/proof['record']==folder
            state = read(folder/'evaluation.json')
            names = {'reference-a':'参照 A','reference-b':'参照 B','neck-retreat':'短程回撤'}
            for role,title in names.items():
                checked = proof['branches'][role]
                clip = state['branches'][role]['video']
                video = folder/clip['path']
                assert checked['video']['source_states_verified'] and sha(video)==checked['video']['sha256']
                assert ROOT/checked['video']['path']==video
                times,world = clip['frame_times_seconds'],clip['environment_index']
                clips.append(dict(path=video,kind='固定评估',run=folder.name,label=f'二次起身·{title}',
                    complete_clip=True,times=times,source_steps=[round(t/.02) for t in times],condition='supine',
                    release_pose=checked['timely']['episodes'][world]['release_pose'],
                    event_times=[3.2,11.],event_labels=['最早触发时点','持稳窗口结束'],
                    environment_index=world,run_status='completed'))
            continue
        if folder.name.startswith('duck_getup_left-hip-pulse_'):
            receipt = ROOT/'records/reports/getup-supine-left-hip-pulse-trial/analysis.json'
            if not receipt.is_file():continue
            proof = read(receipt)
            assert proof['status']=='passed-independent-left-hip-pulse-trial' and ROOT/proof['record']==folder
            state = read(folder/'evaluation.json')
            names = {'reference-a':'参照 A','reference-b':'参照 B',
                     'toward-target':'沿跟踪误差方向','away-from-target':'反向'}
            for role,title in names.items():
                checked = proof['branches'][role]
                clip = state['branches'][role]['video']
                video = folder/clip['path']
                assert checked['video']['source_states_verified'] and sha(video)==checked['video']['sha256']
                assert ROOT/checked['video']['path']==video
                times,world = clip['frame_times_seconds'],clip['environment_index']
                clips.append(dict(path=video,kind='固定评估',run=folder.name,label=f'仰卧左髋·{title}',
                    complete_clip=True,times=times,source_steps=[round(t/.02) for t in times],condition='supine',
                    release_pose=checked['timely']['episodes'][world]['release_pose'],
                    event_times=[2.32,11.],event_labels=['共同状态后首帧','持稳窗口结束'],
                    environment_index=world,run_status='completed'))
            continue
        if folder.name.startswith('duck_getup_left-hip-load_'):
            receipt = ROOT/'records/reports/getup-supine-left-hip-load-capture/analysis.json'
            if not receipt.is_file():continue
            proof = read(receipt)
            if ROOT/proof['record'] != folder:continue
            checked = proof['video']
            assert proof['status']=='passed-independent-left-hip-load-readback' and checked['source_states_verified']
            clip = read(folder/'evaluation.json')['video']
            video = folder/clip['path']
            assert ROOT/checked['path']==video and sha(video)==checked['sha256']
            times, world = clip['frame_times_seconds'],clip['environment_index']
            clips.append(dict(path=video,kind='固定评估',run=folder.name,label='仰卧起身左髋受力',
                complete_clip=True,times=times,source_steps=[round(t/.02) for t in times],
                condition='supine',release_pose=proof['original_timing']['episodes'][world]['release_pose'],
                event_times=[2.32,3.08],event_labels=['受力采集开始附近','受力采集结束附近'],
                environment_index=world,run_status='completed'))
            continue
        if folder.name.startswith(('duck_switch_routed-getup-stand-walk_development_s1170_',
                                   'duck_switch_routed-getup-ramped-forward_development_s1172_',
                                   'duck_switch_routed-getup-ramped-forward_holdout_s2172_',
                                   'duck_switch_routed-seven-command_development_s1174_')):
            ramped = 'ramped-forward' in folder.name
            seven = 'routed-seven-command' in folder.name
            heldout = '_holdout_' in folder.name
            name = ('continuous-routed-seven-command' if seven else
                    'continuous-routed-getup-ramped-forward' if ramped else 'continuous-routed-getup-stand-walk')
            report = ROOT/'records/reports'/name
            receipt = (report/'holdout' if heldout else report)/'analysis.json'
            if not receipt.exists():continue
            proof = read(receipt)
            assert proof['status']=='passed-independent-verification' and ROOT/proof['record']==folder
            clip = read(folder/'evaluation.json')['video']
            video = folder/clip['file']
            checked = proof['video']
            assert ROOT/checked['path']==video and sha(video)==checked['sha256']
            world = checked['world']
            with np.load(folder/'release-poses.npz', allow_pickle=False) as saved:
                pose = str(saved['poses'][world])
            label = '起身接站立与渐升前进' if ramped else '起身接站立与前进'
            missed = False
            if seven:
                assert checked['terminal_command'] == 'turn-left'
                missed = world in proof['score']['commands']['turn-left']['missed_handoff']
                label = '起身接七指令 · 左转组未接入' if missed else '起身接七指令 · 左转原片'
            if heldout:
                label += '留出'
            clips.append(dict(path=video,kind='固定评估',run=folder.name,label=label,
                complete_clip=True,times=clip['frame_times_seconds'],source_steps=clip['common_steps'],
                condition='连续切换',release_pose=pose,event_times=[12.,16.] if seven else [11.,32.],
                event_labels=(['接入判定点','仍未接入左转'] if missed else ['站立结束','指令接入后 4 秒'])
                             if seven else ['起身窗口结束','连续回合终点'],
                environment_index=world,run_status='completed'))
            continue
        if folder.name.startswith('duck_switch_routed-turn-gyro_pair_s1175_'):
            receipt = ROOT/'records/reports/continuous-routed-turn-feedback-pair/analysis.json'
            if not receipt.exists():continue
            proof = read(receipt)
            assert proof['status']=='passed-independent-verification' and ROOT/proof['record']==folder
            state = read(folder/'evaluation.json')
            with np.load(folder/'release-poses.npz',allow_pickle=False) as saved:
                pose = str(saved['poses'][0])
            for role,title in (('static-turn','固定映射'),('restored-gyro','恢复陀螺反馈')):
                clip, checked = state['branches'][role]['video'], proof['videos'][role]
                video = folder/clip['path']
                assert ROOT/checked['path']==video and sha(video)==checked['sha256'] and checked['world']==0
                entered = checked['entered']
                label = f'同站立状态转向 · {title}' + ('' if entered else ' · 未接入左转')
                clips.append(dict(path=video,kind='固定评估',run=folder.name,label=label,
                    complete_clip=True,times=clip['frame_times_seconds'],source_steps=clip['common_steps'],
                    condition='连续切换',release_pose=pose,event_times=[12.,16.],
                    event_labels=['接入判定点','左转 4 秒' if entered else '仍未接入左转'],
                    environment_index=0,run_status='completed'))
            continue
        if folder.name.startswith('duck_switch_routed-turn-gyro_holdout-zero-recovery_s2175_'):
            receipt = ROOT/'records/reports/continuous-routed-turn-feedback-holdout-zero-recovery/analysis.json'
            if not receipt.exists():continue
            proof = read(receipt)
            assert proof['status']=='passed-independent-verification' and ROOT/proof['record']==folder
            state = read(folder/'evaluation.json')
            clip,checked = state['branches']['restored-gyro']['video'],proof['video']
            video = folder/clip['path']
            assert ROOT/checked['path']==video and sha(video)==checked['sha256'] and checked['world']==0
            with np.load(folder/'release-poses.npz',allow_pickle=False) as saved:
                pose = str(saved['poses'][0])
            entered = checked['entered']
            clips.append(dict(path=video,kind='固定评估',run=folder.name,label='七指令独立留出 · 左转',
                complete_clip=True,times=clip['frame_times_seconds'],source_steps=clip['common_steps'],
                condition='连续切换',release_pose=pose,event_times=[12.,16.],
                event_labels=['接入判定点','左转 4 秒' if entered else '未接入左转'],
                environment_index=0,run_status='completed'))
            continue
        if folder.name.startswith('duck_switch_forward-entry-ramp_development_s1171_'):
            receipt = ROOT/'records/reports/continuous-routed-forward-entry-ramp/analysis.json'
            if not receipt.exists():continue
            proof = read(receipt)
            assert proof['status']=='passed-independent-verification' and ROOT/proof['record']==folder
            state = read(folder/'evaluation.json')
            with np.load(folder/'release-poses.npz', allow_pickle=False) as saved:
                pose = str(saved['poses'][0])
            for role, label in (('step-command','立即指令'), ('ramped-command','半秒渐升')):
                clip = state['branches'][role]['video']
                video, checked = folder/clip['path'], proof['videos'][role]
                assert ROOT/checked['path']==video and sha(video)==checked['sha256']
                clips.append(dict(path=video,kind='固定评估',run=folder.name,label=f'前进起步·{label}',
                    complete_clip=True,times=clip['frame_times_seconds'],source_steps=clip['common_steps'],
                    condition='同站立状态',release_pose=pose,event_times=[12.,14.],
                    event_labels=['共同站立终态','起步窗口终点'],environment_index=0,run_status='completed'))
            continue
        if folder.name.startswith('duck_switch_roll-entry-isolation_development_s1091_'):
            receipt = ROOT/'records/reports/continuous-roll-entry-isolation/analysis.json'
            if not receipt.exists():continue
            proof = read(receipt)
            assert proof['status']=='passed-independent-verification' and ROOT/proof['record']==folder
            state = read(folder/'evaluation.json')
            names = {'original-roll-target':'原始目标','projection-only':'仅目标投影','projection-and-slew':'投影加入口限速'}
            for role,title in names.items():
                checked = proof['branches'][role]
                clip = state['branches'][role]['video']
                video = folder/clip['path']
                assert sha(video)==checked['video']['sha256'] and ROOT/checked['video']['path']==video
                world = clip['environment_index']
                first = checked['outcomes'][world]['head_contact_seconds']
                end = clip['frame_times_seconds'][-1]
                head = np.ceil(round((1.+first)/.02)/2)*.04 if first is not None else 1.
                has_head = first is not None and head<end-1e-7
                clips.append(dict(path=video,kind='固定评估',run=folder.name,
                    label=f'滚翻入口同初态·{title}'+('·早停部分原片' if not clip['complete_clip'] else ''),
                    complete_clip=clip['complete_clip'],times=clip['frame_times_seconds'],
                    source_steps=clip['common_steps'],condition='滚翻入口',release_pose=None,
                    event_times=[head if has_head else 1.,end],
                    event_labels=['头部支撑' if has_head else '站立结束快照',
                                  '完整滚翻窗口结束' if clip['complete_clip'] else '达标已不可能，自动停止'],
                    environment_index=world,run_status='completed' if clip['complete_clip'] else 'stopped'))
            continue
        projected = folder.name.startswith('duck_switch_roll-projected-entry_development_s1092_')
        yaw_retest = folder.name.startswith('duck_switch_forward-yaw-replication_development_s1096_')
        if projected or yaw_retest:
            report = 'continuous-roll-projected-entry' if projected else 'continuous-forward-yaw-replication'
            receipt = ROOT/'records/reports'/report/'analysis.json'
            if not receipt.exists():continue
            proof = read(receipt)
            assert proof['status']=='passed-independent-verification' and ROOT/proof['record']==folder
            clip, = read(folder/'clips.json')
            video = folder/clip['file']
            assert sha(video)==proof['video']['sha256'] and ROOT/proof['video']['path']==video
            world = clip['environment_index']
            roll,forward = proof['roll_switch_at'][world],proof['forward_at'][world]
            end = clip['frame_times_seconds'][-1]
            handoff = np.ceil(roll/2)*.04 if roll>=0 else 1.
            walking = np.ceil(forward/2)*.04+3. if forward>=0 else end
            label = '连续切换·投影后进入滚翻' if projected else '连续前进偏航·独立复验'
            clips.append(dict(path=video,kind='固定评估',run=folder.name,
                label=label+('·早停部分原片' if not clip['complete_clip'] else ''),
                complete_clip=clip['complete_clip'],times=clip['frame_times_seconds'],source_steps=clip['common_steps'],
                condition='滚翻连续切换',release_pose=None,event_times=[min(handoff,max(0.,end-.04)),min(walking,end)],
                event_labels=['滚翻后切换站立' if roll>=0 else '滚翻入口',
                              '前进阶段' if forward>=0 else '记录终态'],
                environment_index=world,run_status='completed' if clip['complete_clip'] else 'stopped'))
            continue
        if folder.name.startswith('duck_switch_forward-yaw-pair_development_s1095_'):
            receipt = ROOT/'records/reports/continuous-forward-yaw-pair/analysis.json'
            if not receipt.exists():continue
            proof = read(receipt)
            assert proof['status']=='passed-independent-verification' and ROOT/proof['record']==folder
            state = read(folder/'evaluation.json')
            for role,title in (('reference','0.12 rad/s'),('wider-input','0.15 rad/s')):
                checked = proof['branches'][role]
                clip = state['branches'][role]['video']
                video = folder/clip['path']
                assert sha(video)==checked['video']['sha256'] and ROOT/checked['video']['path']==video
                world = clip['environment_index']
                start,end = np.ceil(proof['shared_controls']/2)*.04,clip['frame_times_seconds'][-1]
                forward = checked['forward_at'][world]
                walking = np.ceil(forward/2)*.04+3. if forward>=0 else end
                clips.append(dict(path=video,kind='固定评估',run=folder.name,label=f'连续前进偏航·{title}',
                    complete_clip=clip['complete_clip'],times=clip['frame_times_seconds'],source_steps=clip['common_steps'],
                    condition='连续前进偏航',release_pose=None,event_times=[min(start,max(0.,end-.04)),min(walking,end)],
                    event_labels=['共同场景分支' if start<end else '结束前状态','前进阶段' if forward>=0 else '记录终态'],
                    environment_index=world,run_status='completed' if clip['complete_clip'] else 'stopped'))
            continue
        if folder.name.startswith(('duck_switch_forward-lateral-push_development_s1097_',
                                   'duck_switch_roll-early-lateral-push_development_s1173_')):
            active_roll = folder.name.startswith('duck_switch_roll-early-lateral-push_')
            name = 'continuous-roll-early-lateral-push' if active_roll else 'continuous-forward-lateral-push'
            label = '滚翻抗扰' if active_roll else '连续前进抗扰'
            interval = .48 if active_roll else 2.
            receipt = ROOT/'records/reports'/name/'analysis.json'
            if not receipt.exists():continue
            proof = read(receipt)
            assert ROOT/proof['record']==folder
            ended_before_fork = proof['status']=='verified-shared-stop'
            if ended_before_fork:
                state = read(folder/'shared-stop.json')['clips']
                verified = proof['clips']
            else:
                assert proof['status']=='passed-independent-verification'
                state = {role:row['video'] for role,row in read(folder/'evaluation.json')['branches'].items()}
                verified = {role:row['video'] for role,row in proof['branches'].items()}
            for role,title in (('reference','无扰动'),('left-push','左推'),('right-push','右推')):
                clip = state[role]
                video = folder/clip['path']
                assert sha(video)==verified[role]['sha256'] and ROOT/verified[role]['path']==video
                end = clip['frame_times_seconds'][-1]
                center = end if ended_before_fork else proof['shared_controls']*.02
                before = min(center,max(0.,end-.04))
                after = min(center+interval,end)
                after_label = '分支后 0.48 秒' if active_roll else '恢复窗口终点'
                clips.append(dict(path=video,kind='固定评估',run=folder.name,label=f'{label}·{title}',
                    complete_clip=clip['complete_clip'],times=clip['frame_times_seconds'],source_steps=clip['common_steps'],
                    condition='滚翻抗扰' if active_roll else '前进抗扰',release_pose=None,event_times=[before,after],
                    event_labels=['共同轨迹' if ended_before_fork else '共同分支',
                                  after_label if after==center+interval else '记录终态'],
                    environment_index=clip['environment_index'],run_status='completed' if clip['complete_clip'] else 'stopped'))
            continue
        if folder.name.startswith('duck_walk_right-support-pulse_'):
            receipt = folder/'verification.json'
            if not receipt.exists():continue
            proof = read(receipt)
            assert proof['status']=='passed-independent-short-support-readback' and proof['record']==rel(folder)
            report = read(folder/'evaluation.json')
            labels = {'reference-a':'参考一', 'reference-b':'参考二', 'candidate':'支撑修正'}
            for condition in report['conditions']:
                role, info = condition['role'], condition['video']
                camera = proof['roles'][role]['video']
                video, times = ROOT/camera['path'], info['frame_times_seconds']
                assert camera['physical_source_hashes_verified'] and sha(video)==camera['sha256']
                clips.append(dict(path=video,kind='固定评估',run=folder.name,
                    label=f'右转支撑短对照·后退留档·{labels[role]}',
                    complete_clip=camera['complete_window'],times=times,source_steps=[round(t/.02) for t in times],
                    condition='backward',release_pose=None,event_times=[min(1.52,times[-1]),times[-1]],
                    event_labels=['分支开始','记录终态'],environment_index=camera['environment'],run_status='completed'))
            continue
        feedback_trials = {
            'duck_walk_stance-response_': ('stance-response','支撑脉冲',2.),
            'duck_walk_yaw-direction-pilot_': ('yaw-direction-pilot','四秒方向反馈',4.),
            'duck_walk_yaw-direction-development_': ('yaw-direction-development','七指令方向反馈',4.),
            'duck_walk_yaw-direction-fresh_': ('yaw-direction-fresh','七指令独立复验',4.),
            'duck_walk_yaw-direction-holdout_': ('yaw-direction-holdout','七指令最终留出',4.),
        }
        kind = next((value for prefix,value in feedback_trials.items() if folder.name.startswith(prefix)),None)
        if kind:
            stage, title, end = kind
            receipt = ROOT/'records/reports/common-walk-right-support-yaw-training'/stage/'analysis.json'
            if not receipt.exists():continue
            proof = read(receipt)
            if proof['record']!=rel(folder):continue
            report = read(folder/'evaluation.json')
            labels = {'reference-a':'原反馈一', 'reference-b':'原反馈二', 'positive':'正向脉冲',
                      'negative':'负向脉冲','causal-feedback':'方向反馈'}
            for condition in report['conditions']:
                role, info = condition['role'], condition['video']
                camera = proof['roles'][role]['video']
                video, times = ROOT/camera['path'], info['frame_times_seconds']
                assert camera['physical_source_hashes_verified'] and sha(video)==camera['sha256']
                assert camera['complete_window'] == (end == 4.) and times[-1]==end
                clips.append(dict(path=video,kind='固定评估',run=folder.name,
                    label=f'{title}·后退录像·{labels[role]}', complete_clip=camera['complete_window'],times=times,
                    source_steps=[round(t/.02) for t in times], condition='backward',release_pose=None,
                    event_times=[1.68,end],event_labels=['分支状态','诊断结束' if end == 2. else '录像终点'],
                    environment_index=camera['environment'],run_status='completed'))
            continue
        if folder.name.startswith('duck_walk_forward-repeatability_'):
            receipt = ROOT/'records/reports/common-walk-right-support-yaw-training/forward-repeatability/analysis.json'
            if not receipt.exists():continue
            proof = read(receipt)
            assert proof['record']==rel(folder)
            report = read(folder/'evaluation.json')
            labels = {'reference-a':'父策略一', 'reference-b':'父策略二',
                      'shadow-right':'仅计算候选', 'applied-right':'实际使用候选'}
            for condition in report['conditions']:
                role, info = condition['role'], condition['video']
                camera = proof['arms'][role]['media']
                video, times = ROOT/camera['path'], info['frame_times_seconds']
                assert camera['physical_source_hashes_verified'] and sha(video)==camera['sha256']
                assert not camera['complete_window'] and times[-1]==2.
                clips.append(dict(path=video,kind='固定评估',run=folder.name,
                    label=f'前进偏航短对照·后退录像·{labels[role]}',
                    complete_clip=False,times=times,source_steps=[round(t/.02) for t in times],
                    condition='backward',release_pose=None,event_times=[1.,2.],
                    event_labels=['第 1 秒','诊断结束'],environment_index=camera['environment'],run_status='completed'))
            continue
        if folder.name.startswith(('duck_walk_shared-prefix-repeat_', 'duck_walk_right-leg-recovery_')):
            leg_recovery = folder.name.startswith('duck_walk_right-leg-recovery_')
            receipt = (folder/'verification.json' if leg_recovery else
                       ROOT/'records/reports/common-walk-shared-prefix-repeat/analysis.json')
            if not receipt.exists():continue
            proof = read(receipt)
            assert proof['record']==rel(folder)
            camera = proof['roles']['reference-a']['video']
            report = read(folder/'evaluation.json')
            info, = [row['video'] for row in report['conditions'] if row['role']=='reference-a']
            video = ROOT/camera['path']
            assert camera['physical_source_hashes_verified'] and sha(video)==camera['sha256']
            times = info['frame_times_seconds']
            label = '右转腿部对照·共享前缀' if leg_recovery else '参考重复·共享前缀'
            clips.append(dict(path=video,kind='固定评估',run=folder.name,label=label,
                complete_clip=camera['complete_window'],times=times,source_steps=[round(t/.02) for t in times],
                condition='backward',release_pose=None,event_times=[min(1.,times[-1]/2),times[-1]],
                event_labels=['启动观察','共享前缀终态'],environment_index=camera['environment'],
                run_status='completed'))
            continue
        if folder.name.startswith(('duck_walk_paired-seven_', 'duck_walk_forward-routing_', 'duck_walk_right-onset_',
                                   'duck_walk_right-startup-eval_', 'duck_walk_right-support-eval_', 'duck_walk_forward-margin_',
                                   'duck_walk_early-official-right_')):
            receipt = folder/'verification.json'
            if not receipt.exists():continue
            proof = read(receipt)
            camera = proof['camera']
            assert camera['physical_source_hashes_verified']
            report = read(folder/'evaluation.json')
            condition, = report['conditions']
            assert condition['condition']=='backward'
            info = condition['video']
            video = folder/info['path']
            assert sha(video)==camera['sha256'] and rel(video)==camera['path']
            times = info['frame_times_seconds']
            if folder.name.startswith('duck_walk_right-onset_'):
                series = '右转对照的后退复验'
                title = {'reference':'参考组', 'candidate':'修正组'}[report['arm']]
            elif folder.name.startswith('duck_walk_right-startup-eval_'):
                assert proof['status']=='passed-independent-right-turn-seven-command-readback'
                series = '右转训练评估·后退留档'
                title = {'routed-parent':'父策略', 'ppo-control':'原奖励', 'ppo-candidate':'欠速'}[report['arm']]
            elif folder.name.startswith('duck_walk_right-support-eval_'):
                assert proof['status']=='passed-independent-right-turn-seven-command-readback'
                series = '右转支撑奖励评估·后退留档'
                title = {'routed-parent':'父策略', 'ppo-control':'原奖励', 'ppo-candidate':'支撑奖励'}[report['arm']]
            elif folder.name.startswith('duck_walk_forward-margin_'):
                assert proof['status']=='passed-independent-right-turn-seven-command-readback'
                series = '前进反馈评估·后退留档'
                title = {'routed-parent':'父策略 0.12', 'ppo-control':'候选 0.12', 'ppo-candidate':'候选 0.15'}[report['arm']]
            elif folder.name.startswith('duck_walk_early-official-right_'):
                assert proof['status']=='passed-independent-right-turn-seven-command-readback'
                series = '右转早期切换·后退留档'
                title = {'routed-parent':'父策略', 'shadow-official':'影子计算', 'early-official':'实际切换'}[report['arm']]
            else:
                title = {'official':'官方', 'control':'等权', 'candidate':'停滞加权', 'routed':'按指令选模型'}[report['arm']]
                series = '模型选择后退' if folder.name.startswith('duck_walk_forward-routing_') else '七指令后退'
            complete = camera['complete_window']
            clips.append(dict(path=video,kind='固定评估',run=folder.name,label=f'{series}·{title}',
                complete_clip=complete,times=times,source_steps=[round(t/.02) for t in times],
                condition='backward',release_pose=None,event_times=[min(1.,times[-1]/2),times[-1]],
                event_labels=['启动观察','记录终态'],environment_index=camera['environment'],
                run_status='completed'))
            continue
        if folder.name.startswith(('duck_walk_native-rollout-probe_', 'duck_walk_ppo-headroom_')):
            receipt = folder/'verification.json'
            if not receipt.exists():continue
            proof = read(receipt)
            if folder.name.startswith('duck_walk_ppo-headroom_'):
                analysis = folder/'analysis.json'
                assert proof['status']=='passed-independent-native-ppo-headroom'
                assert proof['analysis_sha256']==sha(analysis)
                source_digest = read(analysis)['video_sha256']
                title = '后退 PPO·权重固定'
            else:
                assert proof['status']=='passed-completed-native-probe-readback'
                source_digest = proof['clip_sha256']
                title = '原生后退·固定官方策略'
            clip, = read(folder/'clips.json')
            video = folder/clip['file']
            assert sha(video)==source_digest and clip['frames']==proof['original_frames']
            clips.append(dict(path=video,kind='固定评估',run=folder.name,label=title,
                complete_clip=True,times=clip['frame_times_seconds'],source_steps=clip['common_steps'],
                condition='backward',release_pose=None,event_times=[2.,4.],
                event_labels=['前两段速度观察结束','记录终态'],environment_index=clip['environment_index'],
                run_status='completed'))
            continue
        evaluation = folder/'evaluation.json'
        stopped_sequence = (folder.name.startswith('duck_switch_roll-stand-walk_development_s1090_')
                            and not evaluation.exists())
        if stopped_sequence:
            stopped = ROOT/'records/reports/continuous-roll-stand-walk/stopped-review.json'
            if not stopped.exists():continue
            evidence = read(stopped)
            assert evidence['status']=='verified-stopped-development-attempt' and ROOT/evidence['record']==folder
            report = dict(status='stopped',video=read(folder/'clips.json'))
        elif evaluation.exists():
            report = read(evaluation)
        else:
            continue
        getup_clips, getup_arms = {}, {}
        getup_proof = next((name for (prefix, seed), name in GETUP_VIDEO_PROOFS.items()
                            if folder.name.startswith(prefix) and seed in folder.name), None)
        new_upright = folder.name.startswith('duck_getup_upright-recovery_')
        if getup_proof and (report['status']=='collected-review-required'
                            or (new_upright and report['status']=='completed')):
            proof = read(ROOT/'records/reports'/getup_proof)
            assert proof['status'] == 'passed-independent-verification' and ROOT/proof['record'] == folder
            getup_clips = {row['path']:row for row in proof['videos']}
            getup_arms = proof['arms']
            assert len(getup_clips) == len(report['conditions'])
            for case in report['conditions']:
                video = case['video']
                checked = getup_clips[rel(folder/video['path'])]
                assert checked['sha256'] == video['sha256'] == sha(folder/video['path'])
                assert checked['source_states_verified'] and checked['frames'] == len(video['frame_times_seconds'])
            report = dict(report, status='completed')
        if folder.name.startswith('duck_roll_neck-support_log-recovery_') and report['status']=='stopped-review-required':
            recovered=ROOT/'records/reports/roll-neck-support/log-recovery/readback'
            if (recovered/'verification.json').exists():
                proof=read(recovered/'verification.json')
                restored=read(recovered/'analysis.json')
                if ROOT/restored['record']==folder:
                    assert proof['original_receipts_preserved'] and proof['analysis_sha256']==sha(recovered/'analysis.json')
                    assert proof['archived_sha256'][rel(folder/'evaluation.json')]==sha(folder/'evaluation.json')
                    report=dict(report,status='completed')  # Only the atlas reading view changes.
        # A stopped protocol may still publish closed, hashed condition clips.
        # Include those as failure evidence, never an open writer or unlisted file.
        if report['status'] not in ('completed','failed','stopped','interrupted'):
            # The early pulse verifier wrote an independent receipt after
            # collection, while its frozen raw evaluator retained that status.
            proof_name = next((name for prefix, name in PULSE_PROOFS.items() if folder.name.startswith(prefix)), None)
            if proof_name and report['status'] == 'collected-review-required':
                proof = read(ROOT/'records/reports'/proof_name/'analysis.json')
                assert proof['status'] == 'passed-integrity' and ROOT/proof['record'] == folder
                assert proof['protocol_sha256'] == sha(folder/'protocol.json')
                for case in report['conditions']:
                    video = case['video']
                    checked = proof['videos'][case['role']]
                    assert ROOT/checked['path'] == folder/video['path']
                    assert checked['sha256'] == video['sha256'] == sha(ROOT/checked['path'])
                    assert checked['complete_window'] and checked['frames'] == len(video['frame_times_seconds'])
                report = dict(report, status='completed')
            else:
                for video in folder.rglob('*.mp4'):
                    if str(video) not in known:excluded.append(dict(path=rel(video),reason='evaluation not complete'))
                continue
        continuous = None
        if folder.name.startswith('duck_switch_roll-stand-walk_development_s1090_'):
            continuous = evidence if stopped_sequence else read(ROOT/'records/reports/continuous-roll-stand-walk/analysis.json')
            assert ROOT/continuous['record']==folder
            clip, = report['video']
            assert clip['full_action'] is (not stopped_sequence) and clip['frames']==continuous['video']['frames']
            assert sha(folder/clip['file'])==continuous['video']['sha256']
            conditions = [dict(condition='continuous-roll-stand-walk',video=dict(path=clip['file'],
                sha256=continuous['video']['sha256'],environment_index=clip['environment_index'],
                complete_clip=clip['complete_clip'],frame_times_seconds=clip['frame_times_seconds']))]
        else:
            conditions=report['conditions']
        endpoint_proof = next((item for prefix, item in ROLL_VIDEO_PROOFS.items() if folder.name.startswith(prefix)), None)
        endpoint_score = None
        if endpoint_proof and report.get('video'):
            endpoint_label, receipt = endpoint_proof
            proof = read(ROOT/'records/reports'/receipt)
            video = dict(report['video'])
            assert proof['complete_video'] and video['complete_horizon']
            assert proof['video_sha256'] == video['sha256'] == sha(folder/video['path'])
            assert len(video['frame_steps']) == 226 and len(conditions) == 1
            video.update(complete_clip=True, frame_times_seconds=[step*.02 for step in video['frame_steps']])
            conditions = [dict(conditions[0], condition='forward-roll', video=video)]
            endpoint_score = next((row for row in proof.get('rows', []) if row['env_index'] == video['environment_index']), {})
        if folder.name.startswith('duck_roll_rng-preserved_') and report.get('video'):
            # This frozen collector stores its one camera beside conditions.
            # Normalize only the reading view; preserve the original record.
            assert len(conditions)==1 and not conditions[0].get('video')
            conditions=[dict(conditions[0],condition='forward-roll',video=report['video'])]
        contact_clock = None
        if folder.name.startswith('duck_roll_contact-clock_') and report.get('video'):
            parent = ROOT/'records/reports/roll-contact-clock'
            coordinator, proof = read(parent/'coordinator.json'), parent/'analysis.json'
            assert coordinator['status']=='completed' and coordinator['analysis_sha256']==sha(proof)
            verification = read(proof)
            assert verification['status']=='passed-independent-diagnostic-verification' and ROOT/verification['record']==folder
            assert verification['complete_video'] and verification['decoded_frames']==226 and len(conditions)==1
            video = dict(report['video'], complete_clip=True,
                         frame_times_seconds=[step*report['step_dt'] for step in report['video']['frame_steps']])
            assert video['frame_times_seconds'][-1]==9.
            conditions = [dict(conditions[0], condition='forward-roll', video=video)]
            contact_clock = read(folder/'verification.json')
            assert contact_clock['video_sha256']==video['sha256']
        shift=None
        anchor=None
        release_labels=None
        release_title=None
        if folder.name.startswith('duck_getup_contact-shift_') and report['status']=='completed':
            shift=read(ROOT/'records/reports/getup-contact-shift/analysis.json')
            assert shift['status']=='passed-integrity' and shift['protocol_sha256']==report['protocol_sha256']
            assert ROOT/shift['record']==folder
        if folder.name.startswith('duck_getup_anchor-same-release_') and report['status']=='completed':
            anchor=read(ROOT/'records/reports/getup-anchor-same-release/analysis.json')
            assert anchor['status']=='passed-independent-verification' and ROOT/anchor['record']==folder
            assert anchor['protocol_sha256']==report['protocol_sha256']
            release_labels={'reference-a':'官方一','reference-b':'官方二','success-anchor':'已训练模型'}
            release_title='起身同一释放'
        if folder.name.startswith('duck_getup_supine-weighted_') and report['status']=='completed':
            proof=ROOT/'records/reports/getup-supine-weighted-learning/analysis.json'
            anchor=read(proof)
            assert anchor['status']=='passed-independent-verification' and ROOT/anchor['record']==folder
            assert report['analysis_sha256']==sha(proof) and anchor['evaluation_protocol_sha256']==sha(folder/'protocol.json')
            release_labels={'fixed-parent':'父模型','unweighted-continuation':'等权对照','supine-weighted-continuation':'加权候选'}
            release_title='仰卧同一释放'
        for condition in conditions:
            if not condition.get('video'):continue
            video=folder/condition['video']['path']
            if str(video) in known:continue
            assert sha(video)==condition['video']['sha256']
            label=short(folder.name)
            phase_time=None
            phase_label=None
            event_times=None
            event_labels=None
            world=condition['video'].get('environment_index',0)
            if continuous is not None:
                label = '连续切换·站立、滚翻、站立、前进'
                if stopped_sequence:
                    failure = read(ROOT/'records/reports/continuous-roll-stand-walk/roll-failure-stage.json')
                    event = failure['continuous_episodes'][world]['head_contact_seconds']
                    event_times = [np.ceil(round((1.+event)/.02)/2)*.04 if event is not None else 1.,10.]
                    event_labels = ['滚翻头部支撑' if event is not None else '滚翻起始边界','滚翻窗口末未切换']
                else:
                    switched = continuous['roll_switch_at'][world]
                    forward = continuous['forward_at'][world]
                    event_times = [np.ceil(switched/2)*.04 if switched>=0 else 1.,
                                   min(continuous['video']['seconds'],np.ceil(forward/2)*.04+3.) if forward>=0 else 10.]
                    event_labels = ['滚翻后切换站立' if switched>=0 else '滚翻起始边界',
                                    '前进指令持续 3 秒' if forward>=0 else '本镜头未进入前进']
            if endpoint_score is not None:
                label = endpoint_label
                head, hold = (endpoint_score.get(key) for key in ('head_contact_seconds', 'hold_end_seconds'))
                event_times = [np.ceil(round(t/.02)/2)*.04 for t in (head or .28, hold or 9.)]
                event_labels = [f'头部支撑（事件 {head:.2f} s）' if head is not None else '动作早期',
                                f'持稳结束（事件 {hold:.2f} s）' if hold is not None else '记录终态']
            if getup_clips:
                proof = getup_clips[rel(video)]
                arm = {'fixed-weighted':'父模型', 'n1536':'较小并行组', 'n3072':'较大并行组',
                       'pose-routed':'姿态分流', 'control':'原奖励对照', 'candidate':'早期支撑候选'}
                if new_upright:
                    arm = {'reference-a':'原策略','reference-b':'原策略复测','bounded-legs':'腿部转正修正'}
                label = f'同次释放起身·{arm[proof["role"]]}'
                event_times = [2.,11.]
                event_labels = ['释放状态','记录终态']
                if '_s1087_' in folder.name:
                    arm = {'fixed-weighted':'父模型','control':'原覆盖对照','candidate':'侧倒转仰卧回放'}
                    label = f'同次释放起身·{arm[proof["role"]]}'
                if '_s1102_' in folder.name:
                    arm = {'fixed-weighted':'父模型','control':'原奖励对照','candidate':'右侧直立奖励'}
                    label = f'同次释放起身·{arm[proof["role"]]}'
                if any(seed in folder.name for seed in ('_s1084_','_s1087_','_s1102_')) or new_upright:
                    names = {'prone':'俯卧', 'supine':'仰卧', 'left-side':'左侧', 'right-side':'右侧', 'other':'其他'}
                    actual = names[proof['actual_release_pose']]
                    requested = names[proof['requested_pose']]
                    stable = getup_arms[proof['role']]['end_hold']['stable_by_environment'][world]
                    event_labels = [f'指定{requested} · 实际{actual}释放', '末段站稳通过' if stable else '末段站稳未通过']
            if 'knee-support-supply' in folder.name:
                arm={'reference-a':'官方一','reference-b':'官方二','knee-pulse':'伸膝候选'}[condition['condition']]
                label=f'起身同状态·{arm}'
                phase_time=report['fork_control']*.02 if 'fork_control' in report else None
            if shift:
                arm={'reference-a':'官方一','reference-b':'官方二','hip-knee-shift':'髋膝候选'}[condition['condition']]
                label=f'起身髋膝协同·{arm}'
                score=shift['scores'][condition['condition']]
                trigger=score['trigger_by_world'][world]
                release=score['rows'][world]['release_pose']
                phase_time=(trigger+5)*.02 if trigger>=0 else report['fork_control']*.02
                phase_label=f'实际释放：{release} · '+('本环境达到触发条件' if trigger>=0 else '本环境未触发')
            if folder.name.startswith('duck_walk_forward-phase_'):
                arm={'reference-a':'官方一','reference-b':'官方二','phase-yaw':'相位候选'}[condition['role']]
                label=f'前进偏航·{arm}'
            if folder.name.startswith('duck_walk_backward-onset_'):
                label = f'后退启动·开发 {report["seed"]}'
                event_times = [1.,3.]
                event_labels = ['启动观察','第 3 秒状态']
            if folder.name.startswith('duck_walk_backward-load-clock_') and report['status']=='completed':
                trial = read(ROOT/f'records/reports/common-walk-backward-load-clock/seed{report["seed"]}-analysis.json')
                assert trial['status']=='passed-integrity' and ROOT/trial['record']==folder
                assert trial['video']['sha256']==sha(video)
                label = '后退受力·原策略'
                event_times = [1.64,3.]
                event_labels = ['受力采集中','第 3 秒状态']
            if folder.name.startswith('duck_walk_backward-enrichment_') and report['status']=='completed':
                arm = 'candidate' if '_candidate-' in folder.name else 'control'
                trial = read(ROOT/f'records/reports/common-walk-backward-start-enrichment/recovery/{arm}/seed{report["seed"]}-analysis.json')
                assert trial['status']=='passed-integrity' and ROOT/trial['record']==folder
                assert trial['video']['sha256']==sha(video)
                label = '后退初态·' + ('困难重置' if arm=='candidate' else '普通重置')
                event_times = [1.,2.]
                event_labels = ['首秒状态','第 2 秒状态']
            if folder.name.startswith('duck_walk_backward-stall-capture_') and report['status']=='completed':
                trial = read(ROOT/'records/reports/common-walk-backward-stall-capture/capacity/analysis.json')
                role = condition['role']
                assert trial['status']=='passed-stall-capture-integrity' and ROOT/trial['record']==folder
                assert trial['videos'][role]['sha256']==sha(video)
                label = '后退状态·' + ('原续跑' if role=='backward' else '磁盘恢复')
                event_times = [2.,3.]
                event_labels = ['共享状态','续跑第 1 秒']
            if folder.name.startswith('duck_walk_backward-pulse_'):
                arm = {'backward':'参照一','reference-repeat':'参照二','start-pulse':'助推'}[condition['role']]
                label = f'后退助推·{arm}'
                event_times = [2.,3.]
                event_labels = ['共同状态','分支后第 1 秒']
            if folder.name.startswith('duck_walk_backward-lateral_') and report['status']=='completed':
                names = {'backward':'原策略','reference-repeat':'原策略复测','body-left':'左向侧移','body-right':'右向侧移'}
                role = condition['role']
                trial = read(ROOT/f'records/reports/common-walk-backward-lateral-probe/seed{report["seed"]}-analysis.json')
                assert trial['status']=='passed-integrity' and ROOT/trial['record']==folder
                assert trial['videos'][role]['sha256']==sha(video)
                label = f'后退双向侧移·{names[role]}'
                event_times = [2.,3.]
                changed = role in ('body-left','body-right') and world in trial['triggered']
                event_labels = ['共同状态', '第 3 秒 · 本镜头已侧移' if changed else '第 3 秒 · 本镜头无侧移']
            if folder.name.startswith(('duck_walk_backward-actuator-support_','duck_walk_backward-roll_')) and report['status']=='completed':
                roll_target = folder.name.startswith('duck_walk_backward-roll_')
                names = {'backward':'原策略','reference-repeat':'原策略复测','body-left':'左向关节目标','body-right':'右向关节目标'}
                role = condition['role']
                study = 'common-walk-backward-roll-pair' if roll_target else 'common-walk-backward-actuator-support'
                trial = read(ROOT/f'records/reports/{study}/seed{report["seed"]}-analysis.json')
                assert trial['status']=='passed-integrity' and ROOT/trial['record']==folder
                assert trial['videos'][role]['sha256']==sha(video)
                title = '后退侧倾' if roll_target else '后退支撑转移'
                label = f'{title}·{names[role]}'
                event_times = [2.,3.]
                changed = role in ('body-left','body-right') and world in trial['triggered']
                event_labels = ['共同状态', '第 3 秒 · 本镜头已修正' if changed else '第 3 秒 · 本镜头无修正']
            for prefix, (title, labels) in WALK_YAW_CLIPS.items():
                if folder.name.startswith(prefix):
                    arm = {'reference': '参照', 'reference-a': '参照一', 'reference-b': '参照二', **labels}
                    label = f'{title}·{arm[condition["role"]]}'
                    event_times = [1.04, 3.00]
                    event_labels = ['分支后首个录像帧', '前进指令持续阶段']
                    break
            if folder.name.startswith('duck_roll_neck-support_') and report['status']=='completed':
                role=condition['role']
                arm={'reference-a':'官方一','reference-b':'官方二','neck-support':'颈部候选'}[role]
                label=f'滚翻颈部支撑·{arm}'
                case=read(folder/f'original-score-{role}.json')['episodes'][world]
                first=case['head_contact_seconds']
                final=case['hold_end_seconds']
                event_times=[np.ceil(round(t/.02)/2)*.04
                             for t in (first if first is not None else .3, final if final is not None else 9.)]
                event_labels=[f'头部支撑（事件 {first:.2f} s）' if first is not None else '动作早期',
                              f'持稳结束（事件 {final:.2f} s）' if final is not None else '观察结束']
            if contact_clock:
                case = contact_clock['rows'][world]
                first, last = case['head_contact_seconds'], case['hold_end_seconds']
                event_times = [float(np.ceil(round(t/.02)/2)*.04) for t in (first or .3, last or 9.)]
                event_labels = [f'头部支撑（事件 {first:.2f} s）' if first is not None else '动作早期',
                                f'持稳结束（事件 {last:.2f} s）' if last is not None else '观察结束']
            if anchor:
                role=condition['role']
                label=f'{release_title}·{release_labels[role]}'
                case=next(item for item in anchor['videos']
                          if item['role']==role and item['requested_pose']==condition['condition'])
                assert case['world']==world and case['sha256']==sha(video)
                release=case['actual_release_pose']
                pose_names={'prone':'俯卧','supine':'仰卧','left-side':'左侧','right-side':'右侧','other':'其他'}
                row=anchor['arms'][role]
                result='通过' if row['end_hold']['stable_by_environment'][world] else '未通过'
                event_times=[2.,11.]
                event_labels=[f'指定{pose_names[case["requested_pose"]]} · 实际{pose_names[release]}释放',
                              f'末段持稳{result}']
            if report.get('model') and not folder.name.startswith(('duck_getup_','duck_roll_')) and not any(s in folder.name for s in ('command-response','command-mapping','straight-parallel','gyro-turn-screen','seven-command','user-','action-memory','top-contact','adaptation-screen','completion-frontier','stand-preparation','angle-feature','common-plant','common-projection','terminal-reset')):
                model_parent=Path(report['model']).parent.name
                label=short(model_parent) if model_parent.startswith(('duck_','head_command')) else '官方行走参照'
            if report['status']!='completed':label+='·试验中止'
            complete=bool(condition['video'].get('complete_clip',True))
            if not complete:label+=f'·部分片段 {condition["video"]["frame_times_seconds"][-1]:.2f} s'
            if not shift and not anchor:
                release = (getup_clips[rel(video)]['actual_release_pose'] if getup_clips else
                           condition.get('episodes',[{}])[0].get('release_pose'))
            if release is None and (folder/'coverage.json').exists():
                release=next(row['release_pose'] for row in read(folder/'coverage.json')['episodes'] if row['env_index']==world)
            clips.append(dict(path=video,kind='固定评估',run=folder.name,label=label,
                complete_clip=complete,
                times=condition['video']['frame_times_seconds'],condition=condition['condition'],release_pose=release,
                fork_seconds=phase_time, phase_label=phase_label, event_times=event_times, event_labels=event_labels,
                environment_index=world,run_status=report['status']))
        published={str(folder/c['video']['path']) for c in conditions if c.get('video')}
        for video in folder.rglob('*.mp4'):
            if str(video) not in known and str(video) not in published:
                excluded.append(dict(path=rel(video),reason='not a published closed condition clip'))
    benchmarks = ROOT/'records/benchmarks'
    probes = (*benchmarks.glob('duck_walk_right-startup-probe_*'), *benchmarks.glob('duck_walk_right-support-yaw-probe_*'),
              *benchmarks.glob('duck_walk_right-support-reward-probe_*'))
    for folder in sorted(probes):
        receipt = folder/'verification.json'
        if not receipt.is_file():
            continue
        proof = read(receipt)
        clip, = read(folder/'clips.json')
        video = folder/clip['file']
        assert proof['status']=='passed-native-right-turn-probe-readback' and proof['record']==rel(folder)
        assert sha(video)==proof['video_sha256'] and clip['complete_clip'] and clip['frames']==proof['original_frames']
        contact_probe = folder.name.startswith('duck_walk_right-support-yaw-probe_')
        reward_probe = folder.name.startswith('duck_walk_right-support-reward-probe_')
        if reward_probe:
            label = '右转原生奖励接线'
        else:
            label = '原生右转接触采样' if contact_probe else '原生右转短测'
            label += '·首遍' if '_measure-a_' in folder.name else '·复测'
        times = [1.52,3.] if contact_probe or reward_probe else [1.,4.]
        events = ['触发边界','反馈期限'] if contact_probe or reward_probe else ['启动观察','启动奖励窗口终态']
        clips.append(dict(path=video,kind='固定评估',run=folder.name,label=label,complete_clip=True,
            times=clip['frame_times_seconds'],source_steps=clip['common_steps'],condition='turn-right',
            release_pose=None,event_times=times,event_labels=events,
            environment_index=clip['environment_index'],run_status='completed'))
    return clips,excluded


def release_annotation(clip):
    if clip.get('phase_label'):
        return clip['phase_label']
    pose=clip.get('release_pose')
    if not pose:
        return None
    if 'knee-support-supply' in clip['run']:
        return f'同状态分支 · 2 s 实际姿态：{pose}'
    if 'shared-release-terrain' in clip['run']:
        return f'共同释放；2 s 实际姿态：{pose}'
    return f'落下方向：{clip["condition"]}；2 s 实际姿态：{pose}'


def main():
    OUT.mkdir(exist_ok=True);cards=OUT/'cards';cards.mkdir(exist_ok=True)
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',20)
    small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',15)
    heading=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',30)
    clips,excluded=inventory();assert clips
    prior=read(OUT/'manifest.json') if (OUT/'manifest.json').exists() else {}
    cached={r['source']:r for r in prior.get('entries',[])}
    # Stable card identities prevent a newly sorted run from renumbering and
    # re-decoding all later videos. Old rows retain order; new sources append.
    order={r['source']:i for i,r in enumerate(prior.get('entries',[]))}
    clips.sort(key=lambda c:order.get(rel(c['path']),len(order)))
    next_id=max((int(r['id'][1:]) for r in prior.get('entries',[])),default=0)+1
    rows=[];tiles=[]
    for number,clip in enumerate(clips,1):
        times=np.asarray(clip['times']);assert len(times)>1
        # Recovery: release and deadline. Walking: early and established gait.
        # Short training clips: quarter and three-quarter positions.
        wanted=([.3,1.] if clip['run'].startswith('duck_roll_') else
                ([2.,8.] if times[-1]>=8 else [float(times[-1]*.25),float(times[-1]*.75)]))
        if 'shared-release-terrain' in clip['run']:
            wanted=[2.,11.]
        if clip.get('fork_seconds') is not None and clip['complete_clip']:
            wanted=[clip['fork_seconds'],11.]
        events = clip.get('event_times')
        use_events = events is not None and (clip['complete_clip'] or
            (times[0] <= min(events) and max(events) <= times[-1]+1e-9))
        if use_events:
            wanted=events
        indices=[int(np.argmin(abs(times-t))) for t in wanted]
        source=rel(clip['path']);old=cached.get(source)
        code=old['id'] if old else f'N{next_id:03d}'
        if not old:next_id+=1
        target=cards/f'{code}.jpg';digest=sha(clip['path'])
        caption=clip['condition'] or ('固定价值检查' if clip['kind']=='固定价值检查' else '实际训练')
        frame_labels=(clip.get('event_labels') if use_events else None) or [release_annotation(clip)]*len(indices)
        expected=[]
        for index, frame_label in zip(indices,frame_labels,strict=True):
            label=f'{caption} · t={times[index]:.2f} s · 环境 {clip.get("environment_index",0)}'
            if clip.get('source_steps'):label+=f' · 步 {clip["source_steps"][index]}'
            expected.append(label)
            if frame_label:expected.append(frame_label)
        old=cached.get(source)
        # Reuse only a previously decoded card whose bytes, source bytes,
        # identity, selected times and full annotation text still match.
        # Changed labels or ordering force a new decode/render for that card.
        if (old and old['id']==code and old['kind']==clip['kind'] and old['label']==clip['label'] and
            old['condition']==clip['condition'] and old.get('release_pose')==clip.get('release_pose') and
            old['source_sha256']==digest and old['decoded_frames']==len(times) and
            old['frame_indices']==indices and old['frame_times_seconds']==times[indices].tolist() and
            old['annotations']==expected and target.exists() and sha(target)==old['image_sha256']):
            row=dict(old,cache_reused=True,
                complete_clip=clip.get('complete_clip',True),
                recorded_interval_seconds=[float(times[0]),float(times[-1])],
                render_script_sha256=old.get('render_script_sha256',prior['script_sha256']))
            rows.append(row)
            with Image.open(target) as card:tiles.append(card.resize((600,293),Image.Resampling.LANCZOS))
            continue
        reader=imageio_ffmpeg.read_frames(str(clip['path']),pix_fmt='rgb24');meta=next(reader)
        width,height=meta['size'];frames={};count=0
        for i,raw in enumerate(reader):
            if i in indices:frames[i]=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(height,width,3).copy())
            count+=1
        assert count==len(times) and meta['fps']==25 and len(frames)==2
        title=f'{code}  {clip["kind"]} · {clip["label"]}'
        card=Image.new('RGB',(1220,595),'white');draw=ImageDraw.Draw(card)
        draw.text((12,8),title,font=heading,fill='#15334b')
        annotations=[]
        for column,(index,frame_label) in enumerate(zip(indices,frame_labels,strict=True)):
            x=10+column*610;card.paste(frames[index].resize((600,450),Image.Resampling.LANCZOS),(x,55))
            label=f'{caption} · t={times[index]:.2f} s · 环境 {clip.get("environment_index",0)}'
            if clip.get('source_steps'):label+=f' · 步 {clip["source_steps"][index]}'
            draw.text((x,515),label,font=font,fill='#15334b');annotations.append(label)
            if frame_label:
                draw.text((x,550),frame_label,font=font,fill='#15334b');annotations.append(frame_label)
        card.save(target,quality=90)
        tile=card.resize((600,293),Image.Resampling.LANCZOS);tiles.append(tile)
        rows.append(dict(id=code,kind=clip['kind'],run_id=clip['run'],label=clip['label'],condition=clip['condition'],
            source=rel(clip['path']),source_sha256=sha(clip['path']),decoded_frames=count,source_size=meta['size'],
            fps=meta['fps'],frame_indices=indices,frame_times_seconds=times[indices].tolist(),annotations=annotations,
            recorded_interval_seconds=[float(times[0]),float(times[-1])],
            release_pose=clip.get('release_pose'),image=rel(target),image_sha256=sha(target),
            complete_clip=clip.get('complete_clip',True),environment_index=clip.get('environment_index',0),
            run_status=clip.get('run_status'),cache_reused=False,render_script_sha256=sha(Path(__file__))))
    sheet=Image.new('RGB',(1840,95+math.ceil(len(tiles)/3)*310),'#edf2f5');draw=ImageDraw.Draw(sheet)
    draw.text((20,10),'阶段录像索引',font=heading,fill='#15334b')
    draw.text((20,52),f'{len(rows)} 段原始录像，每段 2 帧。完整图册可放大和播放；单个画面不代表动作通过。',font=font,fill='#15334b')
    for i,tile in enumerate(tiles):sheet.paste(tile,(10+(i%3)*610,90+(i//3)*310))
    overview=DOC/'assets/stage-video-overview.jpg';sheet.save(overview,quality=90)
    recovery=[r for r in rows if r['run_id'].startswith('duck_getup_official-reference_')]
    if recovery:
        detail=Image.new('RGB',(1220,595*len(recovery)),'white')
        for i,r in enumerate(recovery):detail.paste(Image.open(ROOT/r['image']),(0,i*595))
        detail.save(DOC/'assets/standup-keyframes.jpg',quality=92)
    body=['<h1>阶段录像索引</h1><p>每段标明实际环境编号和原片时间范围。新起身录像覆盖完整动作；旧训练短片可能包含重置，固定评估仅首次 episode。试验中止的片段单独标明。学习效果见主报告的全样本指标。</p>',
        '<p><a href="development_report.html">全流程报告</a> · <a href="training-gait-gallery.html">较早的 131 段图册</a></p>']
    for row in rows:
        img=Path(os.path.relpath(ROOT/row['image'],DOC)).as_posix();video=Path(os.path.relpath(ROOT/row['source'],DOC)).as_posix()
        start,end=row['recorded_interval_seconds']
        timing=f'原片物理时间 {start:.2f}–{end:.2f} s · {row["decoded_frames"]}帧 · {row["fps"]:g}fps'
        body.append(f'<section><h2>{html.escape(row["id"]+" · "+row["label"])}</h2><p>{html.escape(row["run_id"])}<br>{html.escape(row["source"])}<br>{timing}</p><img loading="lazy" src="{html.escape(img)}"><details><summary>播放原始录像</summary><video controls preload="none" src="{html.escape(video)}"></video></details></section>')
    singles=[r for r in excluded if r.get('kind')=='startup-single-frame']
    if singles:
        body.append('<section><h2>启动中止证据</h2><p>以下文件只有启动单帧，单列保留，不计入动作录像或评估次数。</p>')
        for item in singles:
            link=Path(os.path.relpath(ROOT/item['path'],DOC)).as_posix()
            body.append(f'<p><a href="{html.escape(link)}">{html.escape(item["path"])}</a> · {html.escape(item["reason"])}</p>')
        body.append('</section>')
    page=DOC/'stage-video-gallery.html'
    page.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>阶段录像索引</title><style>body{max-width:1240px;margin:30px auto;font:17px/1.7 system-ui;color:#15334b}img,video{width:100%;height:auto}section{margin:35px 0}p{overflow-wrap:anywhere}</style>'+''.join(body)+'</html>')
    manifest=dict(status='decoded-rendered-awaiting-visual-inspection',created_at=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)),videos=len(rows),training_videos=sum(r['kind']=='训练' for r in rows),
        evaluation_videos=sum(r['kind']=='固定评估' for r in rows),
        value_check_videos=sum(r['kind']=='固定价值检查' for r in rows),
        partial_training_videos=sum(r['kind']=='训练' and not r['complete_clip'] for r in rows),
        frames=len(rows)*2,entries=rows,excluded=excluded,
        historical_manifest_sha256=sha(ROOT/'records/reports/training-review/video-inventory.json'),
        overview=rel(overview),overview_sha256=sha(overview),gallery=rel(page),gallery_sha256=sha(page),
        overview_size=sheet.size,generated_robot_images=False,simulation_started=False)
    manifest.update(cache_reused_videos=sum(r['cache_reused'] for r in rows),
        newly_decoded_videos=sum(not r['cache_reused'] for r in rows),
        cache_rule='Source/card SHA-256, selected times, identity and full labels must match')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('entries',)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

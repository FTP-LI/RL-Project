"""Refresh the ongoing continuation's report from already verified artifacts.

This only edits documentation and project status. Run the checkpoint, log,
curriculum, plot and media checks first, including actual visual inspection.
The last checkpoint remains a recovery point, not a gait acceptance result.
Final training completion and fixed-policy evaluation need their own update.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_one(text, pattern, replacement):
    result, count = re.subn(pattern, lambda _: replacement, text, flags=re.M)
    assert count == 1, (pattern, count)
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    project = read(ROOT/'project.json')
    formal = project['formal_training']
    assert formal['status'] == 'continuation-running'
    run_id = formal['continuation_run_id']
    assert re.fullmatch(r'baseline_continuation_\d{8}T\d{6}_s(?:42|43|44)', run_id)
    record = ROOT/'records/experiments'/run_id
    live = read(record/'resume-live-check.json')
    snapshot = Path(live['evidence_directory'])
    assert snapshot.is_relative_to(record) and live['status'] == 'passed'
    assert len(live['active_training_pids']) == 1
    command = (Path('/proc')/str(live['active_training_pids'][0])/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    assert any(Path(arg).name == 'resume_official.py' for arg in command)
    assert command[command.index('--run-id')+1] == run_id
    for name, expected in live['input_hashes'].items():
        assert sha(snapshot/name) == expected
    plot = read(snapshot/'figures/plot-check.json')
    curriculum = read(snapshot/'curriculum/check.json')
    assert plot['status'] == curriculum['status'] == 'passed'
    assert plot['visual_inspection'] == 'passed'
    assert plot['source_check_sha256'] == sha(snapshot/'check.json')
    assert plot['figure_sha256'] == sha(snapshot/'figures/continuation-progress.png')
    assert plot['data_sha256'] == sha(snapshot/'figures/progress.csv')
    assert plot['points'] == live['new_updates_verified'] == curriculum['verified_iterations']
    checks = sorted((read(path) for path in (record/'checkpoint-checks').glob('model_*.json')), key=lambda item:item['saved_iteration'])
    assert checks and all(item['status'] == 'passed' and item['run_id'] == run_id for item in checks)
    latest = checks[-1]
    assert not latest['final_budget_verified'] and latest['saved_iteration'] <= live['last_verified_iteration']
    assert sha(ROOT/latest['backup']) == latest['checkpoint_sha256']
    assert [item['saved_iteration'] for item in checks] == list(range(500, latest['saved_iteration']+1, 250))
    assert all(item['finite_tensors_checked'] == 76 and item['changed_mlp_tensors'] == {'actor':8,'critic':8} for item in checks)
    media = read(record/'media-manifest.json')
    media_check = read(record/'live-media-check.json')
    steps = [int(item['file'].rsplit('-',1)[1][:-4]) for item in media]
    assert steps == list(range(0, steps[-1]+1, 6000))
    assert media_check['full_decode'] == media and media_check['preview_images_visually_inspected']
    assert media_check['latest_preview_inspected'] == media[-1]['frame_preview']
    for item in media:
        assert (item['frames'], item['fps'], item['duration_seconds']) == (200,50,4)
        assert sha(record/'videos'/item['file']) == media_check['source_hashes_verified'][item['file']] == item['sha256']

    time = datetime.fromisoformat(live['checked_at']).astimezone(timezone(timedelta(hours=8)))
    first, last, added, total = (live[key] for key in ('first_iteration','last_verified_iteration','new_updates_verified','cumulative_completed_updates'))
    label, saved = latest['saved_iteration'], latest['cumulative_completed_updates']
    scalar_count = curriculum['scalars_checked']
    link = f'../../records/experiments/{run_id}'
    span = f'{first}–{last}'
    report_path = ROOT/'docs/reports/development_report.md'
    report = report_path.read_text()
    report = replace_one(report, r'\d\d:\d\d 的进度快照已交叉核验编号 \d+–\d+ 的 \d+ 次新更新，累计完成 \d+ 次；最近完成核验的恢复点 `model_\d+\.pt` 实际包含 \d+ 次更新',
                         f'{time:%H:%M} 的进度快照已交叉核验编号 {span} 的 {added} 次新更新，累计完成 {total} 次；最近完成核验的恢复点 `model_{label}.pt` 实际包含 {saved} 次更新')
    head, section = report.split('## 18. 官方基线续训', 1)
    section = re.sub(r'resume-verification/\d{8}T\d{12}Z', 'resume-verification/'+snapshot.name, section)
    substitutions = [
        (r'^\| 新迭代编号 \| .* \|$', f'| 新迭代编号 | {time:%H:%M:%S} 快照中为 {span}，连续 {added} 次 |'),
        (r'^\| 累计预算 \| .* \|$', f'| 累计预算 | 日志快照已验证 {total} 次，目标 5000 次；最终预算尚未验收 |'),
        (r'^\| 新恢复点 \| .* \|$', f'| 新恢复点 | `model_{label}.pt` 已核验累计 {saved} 次，包含续训新增的 {latest["new_updates_in_checkpoint"]} 次更新 |'),
        (r'本段编号 \d+–\d+ 的共同前缀', f'本段编号 {span} 的共同前缀'),
        (r'(?:[一二三四五六七八九十]+|\d+)\s*个新恢复点各有 76 个保存张量', f'{len(checks)} 个新恢复点各有 76 个保存张量'),
        (r'`model_\d+\.pt` 的 SHA-256 为 `[a-f0-9]{64}`', f'`model_{label}.pt` 的 SHA-256 为 `{latest["checkpoint_sha256"]}`'),
        (r'以下曲线来自已核验的 \d+ 次新更新', f'以下曲线来自已核验的 {added} 次新更新'),
        (r'第一点为 \d+，最后一点为 \d+', f'第一点为 {first+1}，最后一点为 {total}'),
        (r'对编号 \d+–\d+ 的冻结快照，已逐项核对 7 个课程标量，共 \d+ 个值', f'对编号 {span} 的冻结快照，已逐项核对 7 个课程标量，共 {scalar_count} 个值'),
    ]
    for pattern, replacement in substitutions:
        section = replace_one(section, pattern, replacement)
    rows = []
    for item in checks:
        number = item['saved_iteration']
        rows.append(f'| model_{number}.pt | {item["cumulative_completed_updates"]} | {item["common_step_counter"]} | {item["normalizer_counts"]["actor"]} | {item["optimizer_steps_per_parameter"][0]} | [检查]({link}/checkpoint-checks/model_{number}.json) |')
    section = replace_one(section, r'(?:^\| model_\d+\.pt \|[^\n]+\n)+', '\n'.join(rows)+'\n')
    earlier = '、'.join(f'[{step}]({link}/videos/rl-video-step-{step}.mp4)' for step in steps[1:-1])
    media_text = f'已归档本次采样步数 {"、".join(map(str, steps))} 的 {len(media)} 段录像，每段均完整解码为 200 帧、4 秒，副本与原文件哈希一致。下方展示步数 {steps[-1]} 的录像，步数 {earlier} 的中间录像仍保留。各段采用真实随机训练条件，不能把画面变化直接算成同条件评估收益。'
    section = replace_one(section, r'^已归档本次采样步数[^\n]+', media_text)
    section = replace_one(section, r'<video controls preload="metadata" src="[^"\n]+/rl-video-step-[1-9]\d*\.mp4"></video>', f'<video controls preload="metadata" src="{link}/videos/{media[-1]["file"]}"></video>')
    report = head+'## 18. 官方基线续训'+section

    goal_path = ROOT/'docs/current_goal.md'
    goal = goal_path.read_text()
    goal = replace_one(goal, r'^当前恢复操作：[^\n]+', f'当前恢复操作：`{run_id}` 在 VS Code 可见任务终端持续运行，checkpoint 加载与 reset 已通过。{time:%H:%M:%S} 的只读快照核验编号 {span} 的 {added} 次新更新，日志与 TensorBoard 相符，累计完成 {total} 次。主报告第 18 节保留初始快照与阶段快照，更新曲线和真实录像。原生桌面工具的 Raise 动作会连带还原窗口，因此停止窗口激活动作，后续以日志和文件监测为主。')
    next_label = min(label+250, 4999)
    goal = replace_one(goal, r'^已核验从 `model_500\.pt`[^\n]+', f'已核验从 `model_500.pt` 到 `model_{label}.pt`、每隔 250 次更新保存的 {len(checks)} 个新恢复点。后者包含累计 {saved} 次更新、新增 {latest["new_updates_in_checkpoint"]} 次，采样计数 {latest["common_step_counter"]}、归一化计数 {latest["normalizer_counts"]["actor"]}、Adam 计数 {latest["optimizer_steps_per_parameter"][0]}，76 个张量均为有限值；恢复副本与原件哈希一致。{"、".join(map(str, steps))} 步的 {len(media)} 段训练录像已完整解码，归档清单按数字步数排序。未来启动用的页面监督脚本同步修正，当前运行副本未重载。下一项检查为 `model_{next_label}.pt`，最终 5000 次预算、导出和固定指令步态仍待验收。')
    goal = replace_one(goal, r'P2 课程解读已核验：对编号 \d+–\d+ 的冻结快照，7 个课程标量共 \d+ 个值', f'P2 课程解读已核验：对编号 {span} 的冻结快照，7 个课程标量共 {scalar_count} 个值')
    base = f'records/experiments/{run_id}'
    formal.update(next_checkpoint_check=f'After model_{next_label}.pt is fully saved, verify counts and archive; final model_4999 requires --final after exit code 0',
                  checkpoint_check_status=f'model_{label}_verified_{saved}_total_updates', latest_verified_checkpoint=latest['backup'],
                  latest_verified_checkpoint_sha256=latest['checkpoint_sha256'], checkpoint_evidence=f'{base}/checkpoint-checks/model_{label}.json',
                  checkpoint_cumulative_updates=saved,
                  curriculum_analysis_status=f'{scalar_count} logged scalars across labels {first}-{last} match TensorBoard and configured stage bounds, including mixed boundary values; no training parameter changes or reward-dip causality claim',
                  curriculum_analysis_evidence=f'{base}/resume-verification/{snapshot.name}/curriculum/check.json')
    # Prepare every replacement before writing; unexpected prose structure fails
    # without partially updating the report. Historical evidence stays intact.
    report_path.write_text(report)
    goal_path.write_text(goal)
    (ROOT/'project.json').write_text(json.dumps(project, ensure_ascii=False, indent=2)+'\n')
    shutil.copy2(__file__, snapshot/'refresh_continuation_report.py')
    result = dict(status='updated-from-verified-artifacts', checked_at=datetime.now(timezone.utc).isoformat(),
                  snapshot=str(snapshot.relative_to(ROOT)), checkpoint=label, cumulative_checkpoint_updates=saved,
                  cumulative_log_updates=total, media_clips=len(media),
                  output_sha256={str(path.relative_to(ROOT)):sha(path) for path in (report_path,goal_path,ROOT/'project.json')},
                  stable_gait_verified=False, scope='Ongoing training documentation; HTML render/media checks run separately')
    (record/'progress-report-refresh.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

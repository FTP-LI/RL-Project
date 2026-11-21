"""Close the first continuation's training record using independently checked data.

Run after final checkpoint/log/curriculum checks and actual plot inspection.
This edits records and documentation only; it never starts compute or marks
gait acceptance. Keep the ongoing-report helper's live-process guard intact.
"""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(text, pattern, replacement):
    text, count = re.subn(pattern, lambda _: replacement, text, flags=re.M)
    assert count == 1, (pattern, count)
    return text


def main():
    project = read(ROOT/'project.json')
    formal = project['formal_training']
    assert formal['status'] == 'continuation-running', 'Training closure already recorded or unexpected phase'
    run = formal['continuation_run_id']
    assert re.fullmatch(r'baseline_continuation_\d{8}T\d{6}_s42', run)
    base = f'records/experiments/{run}'
    record = ROOT/base
    plan = read(record/'resume-plan.json')
    result = read(record/'training-result.json')
    log = Path(result['log_dir'])
    live = read(record/'resume-live-check.json')
    snapshot = Path(live['evidence_directory'])
    assert snapshot.is_relative_to(record/'resume-verification')
    assert result['run_id'] == run and result['status'] == 'training-finished'
    assert result['exit_code'] == 0 and (record/'exit-code.txt').read_text().strip() == '0'
    assert plan['total_updates'] == 5000 and plan['remaining_updates'] == 4749
    assert live['status'] == 'passed' and live['active_training_pids'] == []
    assert (live['first_iteration'], live['last_verified_iteration'], live['new_updates_verified'],
            live['cumulative_completed_updates']) == (251, 4999, 4749, 5000)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            parts = (proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        assert not (run.encode() in parts and any(p.endswith(b'/resume_official.py') for p in parts))
    for name, expected in live['input_hashes'].items():
        assert sha(snapshot/name) == expected
    assert sha(record/'training.log') == sha(snapshot/'training-prefix.log')
    checks = sorted((read(p) for p in (record/'checkpoint-checks').glob('model_*.json')),
                    key=lambda item: item['saved_iteration'])
    assert [c['saved_iteration'] for c in checks] == [*range(500, 4751, 250), 4999]
    for check in checks:
        assert check['status'] == 'passed' and check['run_id'] == run
        assert check['finite_tensors_checked'] == 76 and check['changed_mlp_tensors'] == {'actor': 8, 'critic': 8}
        assert sha(ROOT/check['backup']) == check['checkpoint_sha256']
    final = checks[-1]
    assert final['final_budget_verified'] and final['cumulative_completed_updates'] == 5000
    assert final['common_step_counter'] == 120000 and final['optimizer_steps_per_parameter'] == [100000]
    assert final['normalizer_counts'] == {'actor': 122880000, 'critic': 122880000}
    assert sha(Path(final['checkpoint'])) == final['checkpoint_sha256']
    assert [item['sha256'] for item in result['checkpoints'] if item['path'] == final['checkpoint']] == [final['checkpoint_sha256']]
    plot = read(snapshot/'figures/plot-check.json')
    curriculum = read(snapshot/'curriculum/check.json')
    assert plot['status'] == curriculum['status'] == 'passed' and plot['visual_inspection'] == 'passed'
    assert plot['points'] == curriculum['verified_iterations'] == 4749 and curriculum['scalars_checked'] == 33243
    assert plot['source_check_sha256'] == sha(snapshot/'check.json')
    assert plot['figure_sha256'] == sha(snapshot/'figures/continuation-progress.png')
    assert plot['data_sha256'] == sha(snapshot/'figures/progress.csv')
    media = read(record/'media-manifest.json')
    media_check = read(record/'live-media-check.json')
    assert media == media_check['full_decode'] and media_check['latest_preview_inspected'] == media[-1]['frame_preview']
    assert media_check['preview_images_visually_inspected']
    assert [int(m['file'].rsplit('-', 1)[1][:-4]) for m in media] == list(range(0, 108001, 6000))
    assert {p.name for p in (log/'videos/train').glob('*.mp4')} == {m['file'] for m in media}
    for item in media:
        assert (item['frames'], item['fps'], item['duration_seconds']) == (200, 50, 4)
        assert sha(log/'videos/train'/item['file']) == sha(record/'videos'/item['file']) == item['sha256']
        assert media_check['source_hashes_verified'][item['file']] == item['sha256']
    export = read(record/'post-training-export-readiness.json')
    assert export['status'] == 'inputs-verified-only' and export['cumulative_updates'] == 5000
    assert export['checkpoint_sha256'] == final['checkpoint_sha256']
    finished = datetime.fromisoformat(result['finished_at']).astimezone(timezone(timedelta(hours=8)))
    started = datetime.fromisoformat(result['started_at'])
    elapsed = (finished-started).total_seconds()
    completed = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(), run_id=run,
                     finished_at=result['finished_at'], supervisor_wall_seconds=elapsed,
                     exit_code=0, cumulative_updates=5000, resumed_updates=4749,
                     cumulative_transitions=122880000, resumed_transitions=116711424,
                     final_checkpoint_sha256=final['checkpoint_sha256'], active_training_pids=[],
                     training_completed=True, final_budget_verified=True, stable_gait_verified=False,
                     media_clips=len(media), media_unchanged_since_full_decode=True,
                     evidence_snapshot=str(snapshot.relative_to(ROOT)),
                     input_sha256={str(p.relative_to(ROOT)): sha(p) for p in (
                         record/'training-result.json', record/'exit-code.txt', snapshot/'check.json',
                         record/'checkpoint-checks/model_4999.json', snapshot/'figures/plot-check.json',
                         snapshot/'curriculum/check.json', record/'media-manifest.json', record/'live-media-check.json',
                         record/'post-training-export-readiness.json')},
                     scope='Completed training and unchanged decoded media; fixed-policy gait and export remain separate')

    # Prepare all prose before writing. A changed report structure aborts here.
    link = '../../'+base
    report_path = ROOT/'docs/reports/development_report.md'
    report = report_path.read_text()
    intro = (f'2026-11-01 恢复的 WSL 官方基线已于 {finished:%H:%M:%S} 正常结束，退出码为 0。'
             f'从 `model_250.pt` 保存的 251 次更新继续新增 4749 次，最终 `model_4999.pt` 累计完成 5000 次；'
             f'日志、TensorBoard、checkpoint 和课程计数已独立核验，恢复副本哈希一致。'
             f'本次可见任务从 10:18:09 到结束共约 {elapsed/3600:.2f} 小时，包含启动和收尾，不能作为纯训练内核耗时。'
             f'完整曲线与 19 段真实训练录像见第 18 节，[完成核验]({link}/training-completion-check.json)保留具体证据。'
             f'导出预检已通过，ONNX 一致性和固定条件步态仍待验收。本文第 12、15–17 节的暂停描述保留当时状态。'
             f'用户已自行安装 Arduino App Lab，目前先处理训练及评估，应用和真机操作后置。')
    report = one(report, r'^2026-11-01 用户已明确恢复 WSL 官方基线续训。[^\n]+', intro)
    head, section = report.split('## 18. 官方基线续训', 1)
    section = re.sub(r'resume-verification/\d{8}T\d{12}Z', 'resume-verification/'+snapshot.name, section)
    section = one(section, r'当前训练仍在运行。', f'本阶段训练于 {finished:%H:%M:%S} 正常结束，退出码为 0。')
    substitutions = [
        (r'^\| 新迭代编号 \|[^\n]+', '| 新迭代编号 | 251–4999，连续 4749 次，已核对完整日志 |'),
        (r'^\| 累计预算 \|[^\n]+', '| 累计预算 | 5000 次更新、122880000 条环境转移；本次续训新增 116711424 条 |'),
        (r'^\| 新恢复点 \|[^\n]+', '| 新恢复点 | `model_4999.pt` 已核验累计 5000 次，包含续训新增的 4749 次更新 |'),
        (r'^\| 进程 \|[^\n]+', '| 进程 | 训练与监督进程已退出，成功退出记录已保存 |'),
        (r'本段编号 \d+–\d+ 的共同前缀', '本段编号 251–4999 的完整日志'),
        (r'\d+ 个新恢复点各有', '19 个新恢复点各有'),
        (r'`model_\d+\.pt` 的 SHA-256 为 `[0-9a-f]{64}`', f'`model_4999.pt` 的 SHA-256 为 `{final["checkpoint_sha256"]}`'),
        (r'日志比 checkpoint 更新更靠后，尚未保存的更新不能作为恢复点。', '最终日志与 checkpoint 均达到累计 5000 次更新，完成预算核验。'),
        (r'以下曲线来自已核验的 \d+ 次新更新', '以下曲线来自已核验的 4749 次新更新'),
        (r'第一点为 252，最后一点为 \d+', '第一点为 252，最后一点为 5000'),
        (r'对编号 \d+–\d+ 的冻结快照，已逐项核对 7 个课程标量，共 \d+ 个值', '对编号 251–4999 的冻结快照，已逐项核对 7 个课程标量，共 33243 个值'),
        (r'^下一阶段先核对新 checkpoint[^\n]+', '最终 checkpoint 的累计采样计数、Adam 更新数、76 个有限张量与恢复副本均已核验，训练预算达到 5000 次。接下来导出 ONNX 并做数值一致性检查，再用第 7 节冻结的五条指令、32 环境和 20 秒时长评估；先判断存活、跌倒、速度误差和停止漂移，再决定是否进入奖励修改组。训练奖励和退出码不能代替步态验收。'),
        (r'本次没有启动这两项计算。', '两项计算的实际结果随后单独核验。'),
        (r'当前训练未完成时，已实际返回', '在训练尚未完成时，入口检查曾实际返回'),
        (r'当前真实入口已因训练未结束而拒绝启动，未创建真实留出选择或尝试记录；这些准备结果不代表留出评估通过。', '准备阶段的真实入口曾因训练未结束而拒绝启动，未创建留出选择或尝试记录。当前应先完成导出及开发评估，开发条件通过后再执行留出评估。'),
    ]
    for pattern, replacement in substitutions:
        section = one(section, pattern, replacement)
    rows = []
    for item in checks:
        n = item['saved_iteration']
        rows.append(f'| model_{n}.pt | {item["cumulative_completed_updates"]} | {item["common_step_counter"]} | {item["normalizer_counts"]["actor"]} | {item["optimizer_steps_per_parameter"][0]} | [检查]({link}/checkpoint-checks/model_{n}.json) |')
    section = one(section, r'(?:^\| model_\d+\.pt \|[^\n]+\n)+', '\n'.join(rows)+'\n')
    section = section.replace('### 18.4 下一步验收\n', f'### 18.4 下一步验收\n\n[完成记录]({link}/training-completion-check.json)将正常退出、完整日志、最终 checkpoint、课程与媒体来源绑定在一起。[导出预检]({link}/post-training-export-readiness.json)已通过，只读取文件，没有导出模型或运行仿真。\n')
    report = head+'## 18. 官方基线续训'+section
    goal_path = ROOT/'docs/current_goal.md'
    goal = goal_path.read_text()
    goal = one(goal, r'^当前恢复操作：[^\n]+', f'当前恢复操作：`{run}` 已于 {finished:%H:%M:%S} 在原 VS Code 任务终端正常结束，退出码为 0。完整日志编号 251–4999 共 4749 次新更新与 TensorBoard 相符，最终累计 5000 次；checkpoint 加载、reset、课程与最终预算核验通过。完成证据为 `{base}/training-completion-check.json`。窗口激活动作继续停用，后续导出和评估通过现有共享任务终端执行。')
    goal = one(goal, r'^已核验从 `model_500\.pt`[^\n]+', f'已核验从 `model_500.pt` 到 `model_4750.pt` 的 18 个定期恢复点及最终 `model_4999.pt`，共 19 个。最终模型累计 5000 次更新，采样计数 120000、归一化计数 122880000、Adam 计数 100000，76 个张量均为有限值；恢复副本与原件哈希一致。19 段训练录像均已完整解码，结束后复核其源文件、副本与解码清单一致。任务 19 的只读导出预检通过；接下来在同一可见终端顺序完成导出和固定指令评估，稳定步态仍未验收。')
    goal = one(goal, r'P2 课程解读已核验：对编号 \d+–\d+ 的冻结快照，7 个课程标量共 \d+ 个值', 'P2 课程解读已核验：对编号 251–4999 的冻结快照，7 个课程标量共 33243 个值')
    goal = goal.replace('本轮仍按冻结的 5000 次完成后评估；', '本轮已完成冻结的 5000 次，接下来执行固定条件评估；')
    goal = goal.replace('当前未完成训练的两个入口均在只读预检中拒绝启动', '训练未完成期间，两个入口均在只读预检中拒绝启动')
    goal = goal.replace('实际导出、数值一致性和固定指令仿真尚未执行，须等当前训练结束后在同一可见终端顺序执行', '最终模型的导出预检现已通过，实际导出、数值一致性和固定指令仿真仍须在同一可见终端顺序完成')
    goal = goal.replace('当前真实入口因训练未结束而拒绝启动，没有创建真实选择或尝试记录；', '准备阶段真实入口因训练未结束而拒绝启动，没有创建真实选择或尝试记录；')
    goal = goal.replace('1. 继续现有 WSL 官方基线，复用单个可见终端，核查日志、配置、checkpoint 与最终累计预算；不重复启动训练，也不为窗口截图改变布局。', '1. 保留已完成的 WSL 官方基线、最终 checkpoint、日志与预算核验；继续复用单个可见终端，不为截图改变窗口布局。')
    project['current_phase'] = 'wsl-official-baseline-training-completed-postprocessing'
    project['current_objective'] = project['current_objective'].replace('it is running, and current work is limited to training and its evidence', 'its first 5000-update stage has completed, and current work is limited to training, export, evaluation and their evidence')
    project['goal_execution'].update(status='active-validation', evidence=base+'/training-completion-check.json',
                                     reason='First 5000-update stage completed and verified; visible export and fixed-command evaluation follow; full P0-P7 scope remains incomplete')
    formal.update(status='continuation-training-completed', final_budget_verified=True,
                  finished_at=result['finished_at'], completed_iterations=5000,
                  resume_semantics_status='actual-load-reset-rollouts-and-final-budget-verified',
                  next_checkpoint_check='Final model_4999 verified; proceed to visible export and fixed-condition evaluation',
                  checkpoint_check_status='model_4999_verified_5000_total_updates',
                  latest_verified_checkpoint=final['backup'], latest_verified_checkpoint_sha256=final['checkpoint_sha256'],
                  checkpoint_evidence=base+'/checkpoint-checks/model_4999.json', checkpoint_cumulative_updates=5000,
                  training_completion_evidence=base+'/training-completion-check.json',
                  post_training_entry_status='Final 5000-update checkpoint passed export check-only; actual export and evaluation require visible tasks 19/20',
                  post_training_export_readiness=base+'/post-training-export-readiness.json',
                  curriculum_analysis_status='33243 logged scalars across labels 251-4999 verified against TensorBoard and configured bounds',
                  curriculum_analysis_evidence=str((snapshot/'curriculum/check.json').relative_to(ROOT)))
    assert not formal['stable_gait_verified']
    # Task 19 reruns the checkpoint verifier. Preserve its current output and
    # the other closure inputs separately so later checks cannot replace them.
    frozen = snapshot/'training-completion-inputs'
    frozen.mkdir(exist_ok=False)
    labels = ['training-result.json', 'exit-code.txt', 'log-tensorboard-check.json',
              'final-checkpoint-check.json', 'plot-check.json', 'curriculum-check.json',
              'media-manifest.json', 'live-media-check.json', 'export-readiness.json']
    original = completed['input_sha256']
    assert len(original) == len(labels)
    frozen_hashes, mapping = {}, {}
    for (source, expected), name in zip(original.items(), labels):
        target = frozen/name
        assert sha(ROOT/source) == expected
        shutil.copy2(ROOT/source, target)
        assert sha(target) == expected
        relative = str(target.relative_to(ROOT))
        frozen_hashes[relative], mapping[source] = expected, relative
    completed.update(input_sha256=frozen_hashes, input_original_paths=mapping,
                     evidence_frozen_at=datetime.now(timezone.utc).isoformat())
    (record/'training-completion-check.json').write_text(json.dumps(completed, indent=2)+'\n')
    report_path.write_text(report)
    goal_path.write_text(goal)
    (ROOT/'project.json').write_text(json.dumps(project, ensure_ascii=False, indent=2)+'\n')
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    refresh = dict(status='updated-from-verified-final-artifacts', checked_at=datetime.now(timezone.utc).isoformat(),
                   completed_updates=5000, stable_gait_verified=False,
                   output_sha256={str(p.relative_to(ROOT)): sha(p) for p in (report_path, goal_path, ROOT/'project.json')})
    (record/'completion-report-refresh.json').write_text(json.dumps(refresh, indent=2)+'\n')
    print(json.dumps({**refresh, 'completion_evidence': str((record/'training-completion-check.json').relative_to(ROOT))}, indent=2))


if __name__ == '__main__':
    main()

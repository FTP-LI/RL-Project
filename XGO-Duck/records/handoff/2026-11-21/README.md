# XGO-Duck阶段交付与恢复入口

训练与新增仿真已暂停。资料覆盖环境部署、官方基线、行走、起身、前滚翻、连续动作、算力测试、候选对照与真机准备。当前实物工作从原厂程序和零点标定开始。[当前目标](../../../docs/current_goal.md) · [全流程报告](../../../docs/reports/development_report.md) · [首次部署、标定与飞智黑武士4Pro手册](../../../docs/runbooks/xgo_duck_first_deployment_and_vader4pro.md)

## 阶段证据

| 阶段 | 核对内容与入口 |
| --- | --- |
| 环境和可视观察 | Windows、WSL、Ubuntu、镜像、显卡验证和VS Code入口。按运行编号核对曲线、日志和原片。[环境与训练手册](../../../docs/runbooks/pc_training.md) · [VS Code观察入口](../../training-current.md) |
| 官方基线与单项动作 | 七指令、起身、前滚翻以及各动作的验收矩阵。[动作阶段与证据](../../reports/stage-acceptance-matrix/README.md) · [官方及自训权重与迭代记录](../../../artifacts/delivery/iteration_record.md) |
| 标称连续路线 | 种子2256、摩擦1.0，42.5秒内完成起身、站立、前滚翻、再站立及分配指令的场景为872/896。逐指令、模型、原片及协议见该次归档。[标称连续路线报告](../../reports/nominal-skill-composition/closed-loop-timestep-confirmation/README.md) · [模型交付清单](../../../artifacts/delivery/nominal-continuous-simulation.json) |
| 局部物理扰动 | 种子2257的滚翻横向扰动，以及种子2258的供电和舵机时延对照。各组环境和统计分别保留。[横向扰动](../../reports/nominal-skill-composition/closed-loop-roll-lateral-disturbance/README.md) · [供电和时延](../../reports/nominal-skill-composition/closed-loop-voltage-delay/README.md) |
| 低摩擦配对学习 | 种子1257，摩擦0.8，同释放的父策略、原奖励和承重转正奖励候选滚翻前准入分别为880/896、882/896、881/896。前进分组分别为125/128、126/128、125/128。两训练组各17次更新、1,253,376条转移。[逐项比较、算力与训练曲线](../../reports/nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/README.md) |
| 暂停前最后的复核 | 同一完整轨迹配对的第一遍4ms参照为13/15，父策略为10/13。保存逐场景联合信号，独立种子2246保留。[最后一次配对核验](../../reports/nominal-skill-composition/low-friction-forward-support/pre-support-geometry/README.md) |
| 真机准备 | 核对已有板端程序和IMU情况，记录机械零点、飞智黑武士4Pro兼容性与断联结果，最后做候选模型对照。[首次部署与标定手册](../../../docs/runbooks/xgo_duck_first_deployment_and_vader4pro.md) |

## 下载与查看

在GitHub仓库`main`的本目录下载各分类ZIP。每个ZIP独立可打开。CSV可在VS Code或电子表格中按原项目路径、运行编号或归档名搜索。

| 文件或前缀 | 内容 |
| --- | --- |
| [inventory.json](inventory.json) | 分组数量、大小、分包名称及SHA-256 |
| [published-files.csv](published-files.csv) | 可发布文件的原路径、所在ZIP、字节数及SHA-256 |
| [archive-sha256.txt](archive-sha256.txt) | ZIP的SHA-256清单 |
| [local-raw-index.csv](local-raw-index.csv) | 本地原始张量、中间检查点、重复打包和安装会话等文件的相对路径与大小 |
| `documents-and-reproduction-*.zip` | 项目文档、环境与实验配置、可复现脚本、来源清单 |
| `stage-reports-and-evaluations-*.zip` | 每阶段报告、逐场景结果、执行日志与历史观察页 |
| `run-logs-and-results-*.zip`、`run-curves-and-snapshots-*.zip` | 原训练和评估记录、逐次曲线、TensorBoard事件 |
| `action-videos-*.zip`、`stage-figures-*.zip` | 训练和评估原片、关键帧、阶段图和截图 |
| `model-weights-*.zip` | 官方ONNX、候选ONNX、结案训练权重与交付清单引用的保存点 |

先在ZIP内按类查看内容。需要重现报告中的相对链接时，将所需ZIP解压到**同一个新建的XGO-Duck目录**中，保留内部相对路径。Windows可在该目录的PowerShell中执行。

~~~powershell
$downloadDir = 'D:\Downloads\XGO-Duck-handoff'
$reviewRoot = (Get-Location).Path
Get-ChildItem -LiteralPath $downloadDir -File -Filter '*.zip' | ForEach-Object {
    Expand-Archive -LiteralPath $_.FullName -DestinationPath $reviewRoot
}
$rawRoot = Join-Path $reviewRoot 'records\training-archive'
foreach ($name in 'evaluations', 'experiments', 'benchmarks', 'profiling') {
    $link = Join-Path $reviewRoot "records\$name"
    $target = Join-Path $rawRoot $name
    if ((Test-Path -LiteralPath $target) -and -not (Test-Path -LiteralPath $link)) {
        New-Item -ItemType Junction -Path $link -Target $target | Out-Null
    }
}
~~~

在解压目录内建立`records/handoff/2026-11-21/`，把本页和四个清单文件从下载目录放进去，报告中的归档链接即可打开。在VS Code打开`docs/reports/development_report.html`或`records/training-current.md`回看过程。需要逐个核对ZIP时，可运行`Get-FileHash -Algorithm SHA256 -LiteralPath '<ZIP路径>'`并与校验清单比对。旧报告若引用高频轨迹或中间快照，可在本地文件索引查到原路径。大批高频张量和冗余快照留在原位置。

仿真恢复从`configs/experiments/nominal_low_friction_full_trace_progress.json`及其[完整轨迹分析](../../reports/nominal-skill-composition/low-friction-forward-support/pre-support-geometry/progress-qualification/full-trace/analysis.json)开始。恢复后先核对候选协议、真实观察入口和分组条件，再使用CUDA；种子2246在开发协议确认后使用。

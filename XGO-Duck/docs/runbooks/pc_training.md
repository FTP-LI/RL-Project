# 电脑环境与训练

本机环境已部署：Windows 11 与 NVIDIA GPU，经 WSL2 Ubuntu 24.04 使用归档的训练源码。首次环境复现见[验证报告](../reports/pc_pipeline_validation.md)和[部署记录](../../records/deployment/2026-10-31/README.md)。下面安装步骤用于换机复现，日常无需重新安装。

Ubuntu 24.04 与 WSL2 是本机选择。归档的[官方安装说明](https://github.com/LuwuDynamics/xgoduck_rl/blob/fa621fc77ad9a05d69da13270e86118bdd3603fe/README.md#install)要求 CUDA GPU、Python 3.12 和 uv，没有指定 Ubuntu 24.04。依赖库文档构建配置不能代表官方训练主机；本机兼容性以实际 CUDA、训练、回放和导出结果判断。

## 查看最近训练

低摩擦承重转正批次索引含两组17次更新的学习曲线、8秒训练原片和三组同释放12秒动作原片。打开关键帧可对照头部接触、双足支撑和倾角。[最近一轮配对训练与动作复核](../../records/reports/nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/README.md)

在VS Code打开[当前训练记录](../../records/training-current.md)可查看实际运行状态、每次更新的曲线、执行日志与完整原片。起身后滚翻课程对照的两组使用同一个官方父保存点与训练预算。对比入口附有重置代码、配置、运行编号与评估原片。[起身后滚翻启动状态训练对照](../../records/reports/nominal-skill-composition/post-getup-roll-curriculum/README.md) · [滚翻接触奖励对照](../../records/reports/roll-forward-input-integration/endpoint-progress-training/README.md) · [低摩擦前进偏航奖励配对](../../records/reports/forward-low-friction-yaw-reward/README.md)

此前的滚翻PPO对照用真实仿真和更新测量本机吞吐。每轮保留运行编号、每次更新曲线和完整7秒训练录像。[滚翻容量对照和原片](../../records/reports/roll-friction-ppo-capacity-live/README.md)

此前起身同步检查组和监控候选组各使用3,072环境，完成17次更新和1,253,376条转移，实际训练总耗时分别为172.19秒和133.13秒。候选把重复检查移到独立线程，并检查结果的新鲜度。修改原因、命令、动作评估和完整录像见[起身两组对照](../../records/reports/getup-monitored-authorization-20261116/README.md)。[起身固定路径对照](../../records/reports/getup-canonical-path-20261116/README.md) · [起身规模对照](../../records/reports/getup-observed-throughput-20261116/README.md)

在已打开本项目的VS Code按**Ctrl+P**，输入`records/training-current.md`并回车，再按**Ctrl+Shift+V**预览。链接可打开实际曲线、`.mp4`原片和日志。已完成记录标为历史数据。每段连续路线原片覆盖实际采样时长，并标出结束时刻和对应判据。

运行中的新任务在其报告目录保存 `native-view/native-status.txt`、`native-curve.png` 和媒体路径；状态约每 2 秒更新，曲线约每 15 秒更新。结束后看 `final-view/` 和独立核验结果。真实训练已验证这一流程；查看器只负责显示，停止由训练回调和外层监督器执行，不能仅凭页面心跳判定工作器健康。

起身训练按完整 8 秒动作录像，评价按完整 11 秒；滚翻和连续切换读取实际场景时长，站立与行走短片为 4 秒。约每 180 秒等待下一次动作起点再拍摄，重置前保存终态，不为录像追加物理步。提前停止片段单独标记。实际规则见[录像配置](../../configs/environments/action_recording.json)。

在项目根目录的PowerShell可独立复查低摩擦起身对照的姿态路由、原始文件哈希和七指令失分阶段。复查读取现有仿真记录。[冻结方案与三组原片](../../records/reports/nominal-skill-composition/low-friction-release-integrator/README.md)

```powershell
python -X utf8 -B scripts/validation/review_nominal_low_friction_release_outcomes.py --check
```

## 启动下一次训练

先在项目根目录只读检查实际状态：

```powershell
$projectState = Get-Content -LiteralPath 'project.json' -Raw | ConvertFrom-Json
$projectState.staged_training_request | Select-Object current_stage, pause_requested, learning_running, simulation_running, next_gate
```

每次新训练按以下顺序准备：

1. 更新当前目标，写清要解决的问题、修改及原因、父保存点、种子、样本预算、独立动作门槛和停止条件。每组使用新的可辨识运行编号，旧协议不重启、不自动追加。
2. 固定源码、依赖、配置、模型和观察器哈希。先做 CPU 接口检查，再做与本次运行一致的实际 CUDA 零步；核对网络、优化器、随机状态和执行器字段。
3. 启动本次监督入口，实际打开对应的运行中状态并核对编号。查看确认必须来自当前画面；不能复制历史确认文件。通过后工作器才导入训练依赖并开始采样。
4. 在固定预算内训练，保存每次更新指标、资源、命令、退出原因和完整原片。观察失败、无进度、来源变更或异常时停止并保留实际保存点；故障注入的合成测试不等于真实 GPU 卡死恢复已验证。
5. 结束后读回网络与 Adam、核对样本和导出一致性、解码原片。用独立场景评价最终策略，再决定保留、修改或晋级；训练中的持稳比例不能替代独立评分。

滚翻配对组使用[训练入口](../../scripts/validation/run_roll_friction_pair.py)，先核对[当前目标清单](../current_goal.md)。此前的起身方案可以参照[起身冻结协议](../../configs/experiments/observed_getup_monitor_pair_20261116.json)、[起身零步核验](../../records/reports/getup-monitored-authorization-20261116/zero-verification.json)和[起身训练入口](../../scripts/training/observed_getup_monitor_entry.py)。以下章节中的历史短训练命令用于复现早期环境。[入口审计](../../records/reports/native-ppo-observation-preflight/entry-audit.json)

保存点的 `iter` 含义取决于具体训练器，不能一律当作完成次数。当前五个主要组件已通过 CPU 网络、优化器和 ONNX 读回，但没有验证原物理现场精确续跑；行走价值网络输入为 77 维，起身为 75 或 76 维，不能直接混用。[保存点边界](../../records/reports/native-ppo-observation-preflight/README.md#保存点边界)

## 1. 路径与记录原则

Windows 项目根目录为 `C:\Users\atlas-win\Desktop\RL-Project\XGO-Duck`。在 PowerShell 中先进入该目录：

```powershell
Set-Location -LiteralPath 'C:\Users\atlas-win\Desktop\RL-Project\XGO-Duck'
```

环境安装和部署保持可见窗口，并记录命令、输出与退出码。训练可以不占用可见 VS Code 终端，但启动前必须验收用户能打开的真实进度、曲线和录像入口，运行中持续留档并能在异常时停止。每一步检查退出码和产物再继续。安装脚本将输出写到 `records/deployment/<date>/`；实验另建 `records/experiments/<run-id>/`。截图使用 `screenshots/01_environment.png`、`02_simulation.png` 等含义明确的文件名，配简短说明。

Windows 归档保持不变；Linux 的唯一训练工作副本为 `~/XGO-Duck/workspace/xgoduck_rl`。不要在 Windows 和 Linux 两处分别修改同一个训练任务。

## 2. WSL 与 Ubuntu

环境恢复记录：2026-10-31 已完成 WSL 与 Ubuntu 安装、系统重启及用户初始化。默认 Linux 用户为 atlas（UID 1000），Ubuntu 24.04.5，Python 3.12.3；RTX 3080 Laptop 16 GB。23:00 环境检查通过，官方锁定依赖、CUDA 张量运算及任务入口均已核验，见 `records/deployment/2026-10-31/linux-launch-225954-result.json`。官方与修改组各完成 5 次迭代、导出与 1000 帧 ONNX 闭环检查；短训练结束时工作副本为 3.3，随后正式基线已恢复官方 3.0 权重。最新训练、评估和性能诊断状态以 [主报告](../reports/development_report.md) 与 `project.json` 为准。

本次初检未找到已安装 WSL 和 Ubuntu。已启动的脚本是：

```powershell
.\scripts\setup\install_wsl.ps1
```

它执行微软官方 `wsl.exe --install --no-distribution --web-download`，记录日志，不自动重启。若安装长时间无输出，应先检查日志、网络及原安装进程，不并行启动多个安装器。必要时使用微软文档给出的官方 MSI 路径，记录来源、签名、SHA-256 和安装退出码。

出现重启要求时，保存其他工作后由用户重启。重启后先在可见 PowerShell 验证：

```powershell
wsl --version
wsl --status
wsl --list --verbose
```

重启后可在可见 PowerShell 运行安装脚本。脚本会检查启动时间、保留已有发行版，并将命令、时间和退出码保存到当天记录目录；本次已启动，完成情况须读取实际结果文件。

```powershell
.\scripts\setup\install_ubuntu.ps1
```

本次在线安装在下载阶段报 `WININET_E_INVALID_SERVER_RESPONSE` 并退出，保留该次日志后改用以下可见本地包流程：

```powershell
.\scripts\setup\install_ubuntu.ps1 -UseLocalPackage
```

此模式从 Ubuntu 官方服务器下载 `ubuntu-24.04.5-wsl-amd64.wsl`，支持断点续传，按微软发行清单核验 SHA-256 与文件大小，再调用 `wsl --install --from-file`。下载包放在 `workspace/downloads/`，来源、哈希、进度及安装退出码保存到当天部署记录；不能将开始下载视为安装成功。

其中安装命令依据 [微软 WSL 命令说明](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)，只在没有注册 `Ubuntu-24.04` 时执行。安装日志结束后再单独启动发行版：

```powershell
.\scripts\setup\initialize_ubuntu.ps1
```

首次启动脚本在可见窗口执行 `wsl -d Ubuntu-24.04`，不启用终端录制。需要由用户设置 Linux 用户名和密码；密码不写进脚本、记录或聊天。初始化完成后，确认发行版使用 WSL2，并在 Ubuntu 内运行：

```bash
cat /etc/os-release
uname -r
nvidia-smi
python3 --version
df -h ~
free -h
```

若 NVIDIA GPU 不可见，先解决 WSL/驱动通路；不要继续用 CPU 跑本轮 CUDA 训练以冒充成功，也不要在 WSL 内重复安装 Linux 显卡内核驱动。

### 本机网络恢复

本机存在 FlClash 隧道与本机 7890 代理。用户切换节点后，Windows 经该代理成功下载 PyPI 索引，WSL NAT 通路仍连接超时。因此使用 `scripts/setup/repair_wsl_network.ps1`，在可见窗口停止本项目 uv 同步，创建此前不存在的 `C:\Users\atlas-win\.wslconfig`，设置 `networkingMode=mirrored`、`autoProxy=true`、`dnsTunneling=true`，仅重启 WSL，再测试本机代理。依据为 [微软 WSL 网络说明](https://learn.microsoft.com/en-us/windows/wsl/networking#mirrored-mode-networking)。实际通过与否以 `network-recovery-*-result.json` 为准。

脚本发现已有 `.wslconfig` 会拒绝覆盖。配置副本、SHA-256 与原先是否存在的状态保存在部署记录中。需要回退时，先结束 WSL 中的工作；核对当前文件仍与保存副本一致后，将本次新增文件重命名为 `.wslconfig.xgo-duck-disabled`，再执行 `wsl --shutdown`，下次启动恢复默认网络。若文件后来增加了其他设置，应仅撤销本次三项，不能覆盖后续修改。

## 3. 固定来源与安装依赖

本轮训练源码固定到 `fa621fc77ad9a05d69da13270e86118bdd3603fe`。归档依赖为 Python 3.12、mjlab 1.3.0、warp-lang 1.12.0、torch 2.9.1，实际传递依赖以 `uv.lock` 为准。使用 `uv sync --locked`，不更新锁文件来掩盖安装失败。

本轮仿真入口固定为注册任务 `Mjlab-Velocity-Flat-XgoDuck` 及对应的 `train`、`play`、`scripts/export.py`。仓库中的 `scripts/infer_policy.py` 默认加载 MicroDuck XML，`scripts/testbench_sim2real.py` 面向 XL330 单关节台架；这两个遗留脚本的默认参数不等于 XGO-Duck 的完整仿真配置，后续复用前需另行适配和验证。

Ubuntu 已初始化后，在可见 PowerShell 执行：

```powershell
.\scripts\setup\run_linux_bootstrap.ps1 -UseMirror
```

脚本会依次：核验 Linux 和 GPU；按逐文件哈希验证官方源码；建立专属 Linux 开发副本；按需安装 uv；同步锁定依赖；执行 CUDA 张量计算；列举任务；保存训练/回放帮助和依赖列表。它不会启动训练。

用户要求优先镜像下载。`-UseMirror` 使用 `sync_locked_mirror.py` 生成仅替换下载地址的临时锁文件，将 PyPI 文件下载指向清华镜像；保留每个包的版本、哈希、registry 和平台条件，解析后逐项比对一致才执行。临时文件与报告保存在部署记录中；使用 `uv sync --frozen --default-index https://pypi.tuna.tsinghua.edu.cn/simple` 下载，其中 index 参数用于构建依赖。结束或可捕获中断后恢复官方锁文件，再执行 `uv sync --locked --offline` 核验。若终端或电脑被强制关闭，应先核对工作副本锁文件；必要时按记录哈希恢复归档中的原文件，再继续。

2026-10-31 对相同 PyTorch 文件前 8 MiB 做只读对照：原连接 25 秒仅收到约 1.28 MiB，Windows 镜像约 0.65 秒完成，Linux 经本机代理约 1.07 秒完成。该结果是小样本测速，不是完整安装耗时。清华镜像的使用地址见 [官方帮助](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)。

后续自动操作集中在 `scripts/setup/visible_console.ps1` 的单个可见窗口。该控制台顺序执行项目命令并记录结果，不启动隐藏任务；当前命令完成后在同一窗口等待下一条。关闭该窗口即停止控制台，勿重复开启。

重复执行时已有开发副本必须仍与官方基线一致；若已经修改实验，脚本会拒绝覆盖不一致文件。此时进入既有副本单独运行 `uv sync --locked`，不要删除实验来强行重装。

预期结果：日志中 CUDA 为可用；GPU 型号符合本机；任务列表包含 `Mjlab-Velocity-Flat-XgoDuck`；脚本最后打印检查通过且退出码为 0。检查通过后仍需实际训练验证 Warp 编译、物理仿真和反向传播。

首次同步因 PyPI 的 setuptools 索引连接超时退出。22:32 已在进程结束后将 84 MiB 缓存移入项目目录，迁移退出码为 0；随后恢复锁定同步。连接超时设为 60 秒，读取超时保持 600 秒，含义见 [uv 环境变量说明](https://docs.astral.sh/uv/reference/environment/#uv_http_connect_timeout)。包来源、锁文件和 TLS 验证保持原配置。恢复脚本先执行 `--move-only`，不把缓存迁移当作依赖安装成功。

依赖完整安装后，可在可见 Ubuntu 终端执行以下离线核验；整理脚本拒绝在 uv 进程活动时移动缓存：

```bash
bash /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/setup/relocate_linux_cache.sh
source /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/setup/project_environment.sh
```

整理后 uv、CUDA、Torch、Triton 缓存统一指向 `~/XGO-Duck/cache/`；后续训练和验证终端先加载 `project_environment.sh`。

## 4. 官方短训练的历史流程

本节命令保留 2026-10-31 的复现方法。历史脚本会启动学习，并不具备上面列出的当前观察与暂停准入；新训练应等待新的受监督入口和冻结方案。

短流程中曾应用修改组补丁，后续正式基线已经恢复官方线速度奖励 3.0。独立复现时先检查工作副本实际状态；只有仍为修改组 3.3 时，才在 Linux 副本用 `git apply -R --check /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/configs/experiments/velocity_tracking.patch` 检查，再用同一路径执行 `git apply -R`。已经是官方基线时不反向应用补丁。记录脚本会核对基线源码哈希，防止误把修改组当作基线。

环境检查及缓存整理通过后，可以在可见 PowerShell 使用已准备的记录脚本：

```powershell
.\scripts\training\run_linux_smoke.ps1 -Mode baseline
```

脚本读取实验定义中的并行数、迭代数与 seed，检查任务源码哈希，创建唯一运行名称，并把请求配置、源码及锁文件哈希、终端日志与退出码写入 `records/experiments/<run-name>/`。两组已实际执行，CLI 参数与保存配置均已核验。它不自动应用修改补丁；`-Mode candidate` 要求源码已经匹配候选哈希。完成命令后还必须核验实际 checkpoint 和参数文件。

以下是已核验的 Ubuntu 命令行流程示例。`max-iterations`、`run-name` 等参数与保存的 `train-help.txt` 对照。新的正式运行先按开发路线验收运行中观察，VS Code 任务终端可选。 [观察要求](../development_roadmap.md)

```bash
source /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/setup/project_environment.sh
cd "$HOME/XGO-Duck/workspace/xgoduck_rl"
set -o pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/XGO-Duck/records"
uv run --locked train Mjlab-Velocity-Flat-XgoDuck \
  --env.scene.num-envs 256 \
  --agent.max-iterations 5 \
  --agent.seed 42 \
  --agent.run-name official_smoke \
  --agent.logger tensorboard \
  2>&1 | tee "$HOME/XGO-Duck/records/official-smoke.log"
printf 'Training exit: %s\n' "${PIPESTATUS[0]}"
```

先检查 5 次迭代，必要时延长到 20。首次 Warp 编译耗时单独记录，不把首次编译计入稳定吞吐。发生显存不足时逐步降到 128/64 并记录；出现 NaN、仿真异常或 CUDA 错误时先排查，不能直接忽略。

输出通常位于 `logs/rsl_rl/xgoduck_velocity/<run>/`，检查保存的 `params/env.yaml`、`params/agent.yaml` 与 `model_*.pt`。记录完整路径，不用“最新文件”代替实验标识。短训练 checkpoint 只用于流程验证，不作为实机候选。

## 5. 回放与可视化

把 `<run>` 和 `<checkpoint>` 替换为刚刚核实存在的路径：

```bash
uv run --locked play Mjlab-Velocity-Flat-XgoDuck \
  --checkpoint-file 'logs/rsl_rl/xgoduck_velocity/<run>/<checkpoint>' \
  --num-envs 1 --viewer viser
```

以终端实际打印的地址打开浏览器，保存机器人、地面及可见状态的截图。Viser 端口以实际输出为准。若浏览器连不上，先确认可见终端中的 viewer 仍在运行，再检查 localhost 转发；不要直接扩大对外网络暴露范围。

回放时记录是否能加载、是否出现 NaN 或异常坠落、指令变化时的动作；短训练无法行走是可能结果，需要如实记录。若渲染依赖缺失，根据错误安装所需包并记录，不预先安装整套桌面环境。

## 6. 修改训练

在官方短训练的配置、checkpoint 和日志保存完成后，按照 [实验定义](../../configs/experiments/velocity_tracking.json) 修改 Linux 开发副本。目标是 `src/mjlab_microduck/tasks/__init__.py` 的 `_xgoduck_velocity_cfg`：

```python
cfg.rewards["track_linear_velocity"].weight = 3.3
```

原值为 3.0，只改这一处；保存统一 diff 到本项目实验记录中。相同 seed、环境数和迭代数再次训练，仅将 `--agent.run-name` 改为 `tracking_weight_smoke`。核验保存配置中的权重确实为 3.3，并回放与导出。不要自动把短训练模型发送到机器人。

已准备唯一改动的 [补丁](../../configs/experiments/velocity_tracking.patch)，实验定义同时记录原文件与修改后文件的 SHA-256。在 Linux 开发副本中，完成官方组后执行：

```bash
git apply --check /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/configs/experiments/velocity_tracking.patch
git apply /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/configs/experiments/velocity_tracking.patch
```

补丁检查不通过时先比对源码来源和已有修改；不能强行套用。2026-10-31 的修改组短流程曾应用此补丁并完成 5 次更新，保存配置确认权重为 3.3；随后正式基线已恢复 3.0。该短流程不能证明修改有效，当前基线运行期间不应用补丁。

正式比较前明确训练预算，先筛查有希望的方案再扩展到 3 个 seed；评估指令、初始状态与环境扰动集合固定。基线和修改组分别从头训练，或明确标注相同的微调来源，不能混用。指标和拟定验收方案见 [开发路线](../development_roadmap.md#evaluation-delivery)。

## 7. 导出与离线核验

等待 Linux 环境时，可以独立核验官方随附模型。在可见 PowerShell 执行下列脚本，它将创建项目隔离环境并安装固定的 CPU 依赖，输出 `official-onnx-cpu.json`：

```powershell
.\scripts\setup\check_official_onnx.ps1
```

该检查覆盖 ONNX 图、形状、dtype、128 组固定 seed 的合成观测和 PC CPU 推理计时，不包括运动稳定性、轨迹一致性或 UNO Q 性能。检查候选文件时可调用 `scripts/validation/check_onnx.py --models <model.onnx> --report <report.json>`，使用对应隔离环境的 Python。

官方导出显式指定环境数为 1，避免为导出构建大量环境：

```bash
uv run --locked python scripts/export.py Mjlab-Velocity-Flat-XgoDuck \
  --checkpoint-file 'logs/rsl_rl/xgoduck_velocity/<run>/<checkpoint>' \
  --num-envs 1 \
  --onnx-file 'logs/rsl_rl/xgoduck_velocity/<run>/walk_candidate.onnx'
sha256sum 'logs/rsl_rl/xgoduck_velocity/<run>/walk_candidate.onnx'
```

需检查 ONNX 图完整性、输入 `[1,61]`、输出 `[1,14]`、dtype、默认关节姿态、动作比例、关节顺序、IR/opset 和有限输出。再使用实际仿真观测比较 PyTorch 与 ONNX，并以 ONNX 闭环回放。归档 `docs/policy_contract_report.json` 是旧官方模型的静态报告，不是新模型报告。

数值对比和 ONNX 仿真闭环使用项目辅助脚本，两组均已完成 1000 帧并保存观测/动作轨迹。可用以下命令模板复现：

```bash
uv run --locked python \
  /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/validation/validate_export.py \
  --checkpoint 'logs/rsl_rl/xgoduck_velocity/<run>/<checkpoint>' \
  --onnx 'logs/rsl_rl/xgoduck_velocity/<run>/walk_candidate.onnx' \
  --output-dir 'logs/rsl_rl/xgoduck_velocity/<run>/export_validation' \
  --steps 1000 --seed 42
```

该脚本让 ONNX 动作驱动官方环境，同时在同一原始观测上计算 PyTorch 参考输出；保存 `export-validation.json`、`observation-action-trace.npz` 和评估配置。默认容差为 `atol=1e-5, rtol=1e-4`，不通过即记录失败。每次使用新的输出目录保留证据；reset 次数包括超时，不能直接解释为跌倒次数。

板端兼容检查使用独立 `onnxruntime==1.22.1` 环境，避免降低训练环境的依赖。PC 上加载成功仅说明对应版本能运行该图；UNO Q 的 ARM 支持、实际频率和延迟仍须上板测量。

## 8. 记录与恢复

每个实验摘要包含：任务、官方提交、修改 diff、锁文件哈希、命令、seed、环境数、预算、实际迭代数、退出码、checkpoint/ONNX 绝对路径与 SHA-256、评估条件、结果和失败项。大 checkpoint 留在 Linux 副本，摘要同步回 Windows 项目；准备交付时将明确选定的模型和契约放入 `artifacts/<run-id>/`。

中断后先查看日志与 checkpoint，再决定重跑或恢复；恢复参数以安装后的 `train --help` 为准，记录恢复前后的 run-id。不要删除官方归档、系统 Python 或其他项目环境。

## 9. 参考

- [Microsoft WSL 安装](https://learn.microsoft.com/en-us/windows/wsl/install)
- [Microsoft 文件系统建议](https://learn.microsoft.com/en-gb/windows/wsl/filesystems)
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [官方 XGO-Duck 训练 README](../../official_resources/xgoduck_rl/README.md)
- [mjlab 1.3.0 训练入口](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/scripts/train.py)
- [mjlab 1.3.0 回放入口](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/scripts/play.py)
- [官方导出实现](../../official_resources/xgoduck_rl/scripts/export.py)






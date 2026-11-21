# 电脑环境、仿真与导出复现

本机在Windows 11、WSL2 Ubuntu 24.04和RTX 3080 Laptop GPU上复现。官方训练要求CUDA、Python 3.12和uv，项目源码锁定到`fa621fc77ad9a05d69da13270e86118bdd3603fe`。本轮训练处于暂停状态，先完成[实机部署链路验收](hardware_acceptance_checklist.md)。[官方训练来源](../../official_resources/xgoduck_rl/README.md) · [固定源码清单](../../official_resources/source_manifest.json)

## 1. 核对来源和环境

换机复现时可用`git clone --depth 1 --branch main https://github.com/FTP-LI/RL-Project.git`获取当前仓库文件，再打开其中的`XGO-Duck`。

1. 在VS Code打开`RL-Project/XGO-Duck`。Windows首次安装WSL和Ubuntu时，在现有可见PowerShell执行`scripts/setup/install_wsl.ps1`和`scripts/setup/install_ubuntu.ps1`，根据系统提示完成重启、账户初始化。运行`wsl --list --verbose`确认Ubuntu使用WSL2。
2. 在项目目录运行以下只读检查。脚本先校验归档文件和依赖锁，项目运行副本位于Linux用户目录`~/XGO-Duck/workspace/xgoduck_rl`。

   ~~~bash
   python3 scripts/setup/prepare_training_copy.py --verify-only
   bash scripts/setup/bootstrap_linux.sh --plan-only
   ~~~

3. 首次部署环境时在可见终端运行`bash scripts/setup/bootstrap_linux.sh --mirror`。保留命令日志和退出码。安装结束后运行`nvidia-smi`，并在工作副本中用`uv run --locked --offline python -c "import torch; print(torch.cuda.is_available())"`确认CUDA能被训练解释器读取。镜像变更记录保存于本机`records/deployment/`，官方依赖锁与源码清单保留。

## 2. 小规模基线和单变量修改

实机部署验收完成并恢复训练后，用`bash scripts/training/run_smoke.sh baseline`复现官方小规模基线。该命令核验源码哈希，读取已冻结的256场景、种子42、5次迭代配方，写入原始运行状态、退出码和TensorBoard记录。用同一硬件、并行度、种子、迭代次数和地图比较修改组。[小规模冻结配置](../../configs/experiments/velocity_tracking.json)

在Linux工作副本备份原任务文件，使用`git apply --check`检查[线速度奖励单变量补丁](../../configs/experiments/velocity_tracking.patch)。补丁只将线速度跟踪权重从3.0改为3.3。其余参数保持一致，回到项目目录运行`bash scripts/training/run_smoke.sh candidate`。保存对应改动、保存点、实测曲线与评估指标。VS Code可直接查看落盘状态、曲线和原始MP4。[代表性曲线和录像](../../records/handoff/2026-11-22/README.md)

完整训练采用原项目冻结配方和独立种子。接续低摩擦阶段从[完整轨迹协议](../../configs/experiments/nominal_low_friction_full_trace_progress.json)开始。该阶段所需的高频物理状态和完整训练过程保存在本机归档。

## 3. 导出与模型预检

从Linux工作副本导出所需策略。填写实际检查点与输出文件名。

~~~bash
uv run --locked python scripts/export.py Mjlab-Velocity-Flat-XgoDuck \
  --checkpoint-file 'logs/rsl_rl/xgoduck_velocity/<run>/<checkpoint>' \
  --num-envs 1 \
  --onnx-file 'logs/rsl_rl/xgoduck_velocity/<run>/walk_candidate.onnx'
~~~

用相同采样观测对比训练网络与ONNX的动作输出，并运行独立仿真闭环。CLI参数见[导出对比脚本](../../scripts/validation/validate_export.py)。本项目的发布模型可用现有含NumPy和ONNX Runtime的解释器离线核验。下例在项目目录启动。

~~~bash
"$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" \
  scripts/validation/verify_deployment_bundle.py
~~~

部署先使用原厂程序及其权重完成站立、走停和手柄验收。候选模型按实际板端版本核对关节顺序、动作尺度和姿态过渡后再进入台架对照。[板端首次部署与回退](xgo_duck_first_deployment_and_vader4pro.md)

# 脚本入口

日常从 VS Code 的 XGO-Duck 任务菜单进入，共享一个可见终端。任务统一调用 [training/training_workflow.sh](training/training_workflow.sh)。

- `setup/`：首次部署、镜像下载和环境核验；已安装环境不重复部署。
- `training/`：训练、恢复、显示与监控。`guarded_training.py` 是新入口；`experiments/` 和旧专用脚本保留用于历史实验复查。
- `validation/`：模型核验、导出、固定评估、保存轨迹复算、图像与报告。S1 使用 `s1_reference_inputs.py`、`s1_reference_evaluation.py`；S2 髋姿态诊断使用 `prepare_hip_pair.py`、`run_hip_pair.py`、`analyze_hip_pair.py`，共同复用 `analyze_walking_suite.py` 的指标。

完整路径与用途见[工程导航](../docs/project_structure.md)。训练与评估脚本必须在可见终端运行，不能作为隐藏任务启动。具体执行前检查冻结协议及阶段；旧脚本存在不代表当前允许恢复它。

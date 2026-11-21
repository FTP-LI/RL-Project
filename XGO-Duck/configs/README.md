# 配置入口

- [动作课程](experiments/motion_curriculum.json)：九类动作的阶段、评估与依赖。
- [下一次训练](experiments/next_training.json)：尚未冻结时必须拒绝启动学习。
- [七指令筛查](experiments/walking_reference_suite.json)：新筛查指标，保留旧五指令成绩。
- [横移补充条件](experiments/walking_strafe_supplement.json)：只采集缺少的左右平移数据。
- [观察工程验收](experiments/observer_integration.json)：预先限定的停止、恢复和开销短测，不选择步态模型。
- [髋姿态成对协议](experiments/hip_limit_pair.json)：已完成的两组各 50 次单因素诊断；协议保留，未达到继续条件，当前不重复运行。
- [随机跌落参照](experiments/random_recovery_reference.json)、[滚翻参照](experiments/roll_reference.json)：官方固定模型、有界评估，均已执行并独立复算。
- [滚翻初始姿态对照](experiments/roll_reset_pair.json)：两组各 32 次已完成，候选未满足继续条件，不重复或追加。
- [连续切换草案](experiments/skill_transition_protocol.json)：尚未冻结，需要先核对共同物理环境与独立技能，不作为启动指令。
- [训练监控](experiments/training_watch.json)：采样、保存、短片和停止设置。
- [显示偏好](environments/training_display.json)：单个可见终端、关闭原生实时窗口、保持窗口布局。
- [名称映射](experiments/run_catalog.json)：历史显示名与原始路径。
- [历史任务](environments/vscode_tasks_history.json)：原编号任务的完整配置。

已执行实验的协议以对应 `records/experiments/` 中保存的快照和哈希为准。当前配置改动不覆盖旧证据；新增实验先冻结输入和验收条件。


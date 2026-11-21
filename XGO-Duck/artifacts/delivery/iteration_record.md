# XGO-Duck 权重与迭代记录

当前目录用于仿真复现。`.onnx` 用于推理，`.pt` 是对应训练保存点。行走组件从最终留出实际使用的来源逐字节核对。早期对照、起身和滚翻权重另有来源校验。 [行走清单](walking-simulation.json) · [历史权重校验](../../records/training-archive/delivery-weights.json) · [来源重查](../../records/training-archive/delivery-weight-recheck.json)

标称连续仿真清单对应独立种子2256的42.5秒路线。后续种子2258复用其中模型，分别确认7.4V供电及8ms舵机时延的单因素动作表现，两组完整通过881/896和891/896。该批学习更新为0，模型用途与工况对应关系见原始评分。[标称连续仿真交付清单](nominal-continuous-simulation.json) · [供电与舵机时延对照](../../records/reports/nominal-skill-composition/closed-loop-voltage-delay/README.md)

同一权重的结论须按场景阅读：`walk-common-base.onnx` 是七指令分流组合的一部分，但该权重在滚翻后前进路线对照中未通过。起身候选的实际右侧相对两组参照分别净增 4、5 例，满足原留出门槛；总体相对参照分别少 4、5 例。 [门槛与模型用途核对](../../records/reports/getup-gate-clarification/README.md)

五个当前仿真组件的策略、价值网络和优化器状态已在 CPU 加载，并用保存的观测核对 ONNX 输出；价值网络输入分别为行走 77 维、起身回退和仰卧 76 维、起身右侧 75 维。这不等于已验证从中断物理现场续训。 [保存点内容与恢复边界](../../records/reports/native-ppo-observation-preflight/README.md#保存点边界)

2026-11-20复核起身监控候选：等样本吞吐提高29.3%，独立留出总体2,737/2,816，请求仰卧244/264，低于251/264；该分支权重与保存点保留作复现，现有交付模型不变。低摩擦滚翻使用相同官方权重，按原滚翻门槛停止，无模型更新。 [候选训练与保存点](../../records/reports/getup-monitored-authorization-20261116/README.md) · [独立留出](../../records/reports/getup-monitored-pair-holdout-s4268/README.md) · [低摩擦对照](../../records/reports/roll-matched-factor-low-continuation-s1242/README.md)

滚翻配对训练的共用初始保存点已归档，其策略部分与官方ONNX通过288条实际输入核验。此保存点用于下面的两组等样本训练。[共用初始保存点](../training-parents/roll-official-actor-ppo-start.pt) · [来源及复算](../../records/reports/roll-friction-exposure-preflight/common-ppo-parent.json) · [冻结训练方案](../../configs/experiments/roll_friction_paired_training.json)

## 权重入口

| 动作 | 官方权重 | 自训权重 | 使用范围 |
| --- | --- | --- | --- |
| 站立、前后走、左右转、左右移 | [运行包权重](official/xgoduck_walk.onnx) · [仿真前进权重](official/walk-simulation-reference.onnx) | [共同接口 ONNX](trained/walk-common-base.onnx) · [保存点](trained/walk-common-base.pt) · [右转 ONNX](trained/walk-right-support.onnx) · [保存点](trained/walk-right-support.pt) | 按指令选用策略和反馈，最终留出七项各 32/32。 [控制参数和评分](walking-simulation.json) |
| 跌倒后起身 | [起身参照](official/xgoduck_getup.onnx) | [共同回退、仰卧、右侧及对应保存点](getup-simulation.json) | 独立留出总体 2,983/3,072，实际右侧 179/183，原门槛通过。 [评分与录像](../../records/reports/getup-right-robustness/holdout/README.md) |
| 前滚翻 | [滚翻参照](official/xgoduck_roulade.onnx) | [角度对照 ONNX](trained/roll-angle-control.onnx) · [最终保存点](trained/roll-angle-control.pt) | 自训对照用于复盘 29/32 的开发结果。后续可靠性验证继续使用官方权重。 [角度对照](../../records/reports/roll-angle-feature/startup-recovery/analysis.json) |

## 迭代记录

2026-11-21，右脚持续承重后80ms的双足承重转正奖励完成配对训练，两组各17次更新、1,253,376条转移。固定释放种子1257的原奖励组与候选滚翻前准入为882和881/896，前进分组由126/128至125/128。两组ONNX和可恢复保存点保留在实验目录，标称交付仍对应独立种子2256的确认。[实验权重、指标及曲线](../../records/reports/nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/README.md)

2026-11-21，从标称仿真交付清单复用全部实际权重，完成种子2257的滚翻横向扰动回归。无扰动、左向和右向的完整路线通过874、867和873/896。测试采用原推理接口和动作门槛。[交付配置](nominal-continuous-simulation.json) · [三组完整动作与核验](../../records/reports/nominal-skill-composition/closed-loop-roll-lateral-disturbance/README.md)

2026-11-21，0.8足底滑动摩擦下的三组固定权重对照共采集2,844,800条场景转移。仅调整起身首1.2秒的仿真积分步长，候选前进入口达到122/128，左转入口为119/128。该对照用于选择低摩擦验证方案，沿用已确认的标称连续模型组。[低摩擦步长对照、原片和源码](../../records/reports/nominal-skill-composition/low-friction-release-integrator/README.md)

2026-11-21，低摩擦承重交接奖励与原奖励各完成17次更新及1,253,376条转移。种子1257同释放快照的滚翻前准入，父策略885/896、原奖励878/896、承重交接奖励880/896。三组前进分组均为126/128。两组ONNX和可恢复保存点归入实验记录，父策略用于后续连续路线。[训练记录、导出、评分和原片](../../records/reports/nominal-skill-composition/low-friction-support-effort/support-credit-ppo/README.md)

2026-11-21，标称2ms滚翻连续路线在两遍开发评估均完成873/896后，通过种子2256的独立确认，完整动作872/896，四种实际释放姿态和七项指令逐项过线。清单链接七个实际使用的策略及五份包含网络和优化器的可恢复保存点。报告保留原训练权重、官方运行包、配置与完整录像。[标称仿真交付清单](nominal-continuous-simulation.json) · [独立确认结果](../../records/reports/nominal-skill-composition/closed-loop-timestep-confirmation/README.md)

2026-11-21，仰卧起身原奖励组与新增站稳带奖励组均用相同父点完成17次更新。种子1257同释放快照的滚翻前总准入为父策略877/896、原奖励882/896、候选884/896。候选相对原奖励的前进分组准入净增1个，晋级要求2个。两组归入实验权重，交付路线使用现行仰卧父策略。[两组ONNX、保存点和同释放评估](../../records/reports/nominal-skill-composition/early-getup-load/support-progress/README.md)

2026-11-21，起身后滚翻启动状态课程与官方站立重置训练组各完成30次更新及3,317,760条环境转移。开发种子1257的滚翻与落地通过数为官方853/896、原始站立重置训练组848/896、课程训练组847/896。两份自训练权重及其保存点用于实验复现。[课程组ONNX](../duck-roll-post-getup-start-curriculum-e4608-s1258-20261121T081905/policy.onnx) · [课程组保存点](../../records/experiments/duck-roll-post-getup-start-curriculum-e4608-s1258-20261121T081905/profile-final.pt) · [两组导出与动作评估](../../records/reports/nominal-skill-composition/post-getup-roll-curriculum/README.md)

2026-11-21，核对交付目录里的行走、起身、滚翻官方运行包ONNX，三份均与本地官方来源逐字节一致。七指令连续控制清单涉及的11个ONNX及保存点也通过SHA-256核对。滚翻末端接触奖励训练的ONNX和对应最终保存点保留于实验来源。新完成的颈部相位控制对照在种子1247、1248为108/128、115/128，未通过原滚翻门槛。[候选权重及保存点](../../records/reports/roll-forward-input-integration/endpoint-progress-training/candidate/independent.json) · [颈部对照评分及原片](../../records/reports/roll-forward-input-integration/endpoint-progress-training/observed-neck-support/current-evaluation.md)

2026-11-21，使用官方行走ONNX完成定时髋俯仰闭环和右脚支撑时长对照。完整闭环的两组标称摩擦通过128/128，两组低摩擦偏航仍高于0.20rad/s。随后按同快照对比追加一拍和持续修正，两个低摩擦种子的持续组均过160毫秒门槛。[20秒闭环与原片](../../records/reports/forward-phase-hip-pitch/README.md) · [同快照时长与整段预检](../../records/reports/right-support-hip-pitch-duration/README.md)

2026-11-21，比较前进策略的全层更新与末端输出层更新。两组各12次更新及884,736条转移，使用相同偏航代价及固定探索。种子1247原20秒门槛下，标称分别通过46/128和12/128，低摩擦均为0/128。两份训练权重保留用于复现。[独立评分、权重和保存点](../../records/reports/forward-output-layer-pair/README.md)

2026-11-21，使用官方仿真权重比较仅首段脉冲和右脚支撑持续修正。增加持续修正后，两组低摩擦均为0/128，两组标称分别为128/128和126/128。现用官方仿真参考以及前进全层和末端输出层配对训练的ONNX、保存点均已核对SHA-256。[完整动作对照](../../records/reports/forward-right-support-continuation/README.md) · [权重来源核验](../../records/reports/forward-right-support-continuation/weight-source-check.json)

2026-11-21，官方行走ONNX使用0.115m/s前进模型输入完成四组20秒同快照开发对照，候选四组各128/128。外部指令为0.15m/s，实际均速中位数约0.134m/s。此轮作为官方权重的输入配置记录，学习更新为0。下一步与低摩擦自训滚翻模型组成连续路线并检查独立留出。[配置、场景评分与原片](../../records/reports/forward-input-full-route/README.md)

2026-11-21，低摩擦加权自训滚翻ONNX接官方行走ONNX。种子1247、脚底摩擦0.8的完整路线由0/128提高到120/128，两条路线滚翻均为120/128。种子1248实际滚翻112/128，低于原门槛116/128。候选权重和带优化器保存点已归档，连续路线进入滚翻支撑复核。[候选ONNX](../duck_roll_friction_low-friction_e4608_s1246_20261120T213457/policy.onnx) · [对应训练保存点](../../records/experiments/duck_roll_friction_low-friction_e4608_s1246_20261120T213457/profile-final.pt) · [对照成绩和原片](../../records/reports/roll-forward-input-integration/README.md)

2026-11-21，保留同一次低摩擦训练的第30次更新保存点并导出ONNX。第15次保存点的策略权重与训练父点逐位一致。第30次保存点在288条固定观测上的导出最大误差为7.75×10⁻⁷rad，滚翻实测使用单独的开发对照。[更早训练保存点](../../records/experiments/duck_roll_friction_low-friction_e4608_s1246_20261120T213457/checkpoint-update-30.pt) · [该保存点的ONNX](../duck_roll_friction_low-friction_e4608_s1246_20261120T213457/policy-after-30-updates.onnx) · [来源与输出核验](../../records/reports/roll-friction-paired-training/low-friction/checkpoint-comparison/assessment.json)

同场景滚翻对照中，第30次保存点在种子1247和1248分别为121/128与111/128。第50次保存点的两次参照分别为116／113和117／111。第30次保存点用于复现训练与滚翻差异，模型交付状态以模型选择清单为准。[两份权重的动态对照](../../records/reports/roll-forward-input-integration/checkpoint-selection/README.md) · [仿真模型选择](simulation-model-selection.json)

2026-11-21，末端接触奖励候选完成等样本50次更新，种子1247与1248的动作结果为111/128和112/128。两次原奖励参照在两个种子分别为119／116和110／114。候选ONNX与三个保存点归档用于复现，每个文件的哈希见训练复核。[候选ONNX](../duck_roll_endpoint_progress_e4608_s1246_20261121T050751/policy.onnx) · [最终保存点](../../records/experiments/duck_roll_endpoint_progress_e4608_s1246_20261121T050751/profile-final.pt) · [训练、评价和来源校验](../../records/reports/roll-forward-input-integration/endpoint-progress-training/README.md)

2026-11-20，低摩擦前进偏航奖励配对。两组各完成24次更新、1,769,472条转移，候选增加封顶的物理偏航代价。摩擦0.8与1.0的自训两组均为0/128，候选在两种摩擦下的偏航中位数比同样本对照低0.02342和0.02358 rad/s。两组ONNX及带优化器的保存点作为实验记录保留。[配对训练与完整录像](../../records/reports/forward-low-friction-yaw-reward/README.md) · [对照组ONNX](../duck_forward_yaw_reward_control_e3072_s1250_20261120T232536/policy.onnx) · [对照组保存点](../../records/experiments/duck_forward_yaw_reward_control_e3072_s1250_20261120T232536/model-final.pt) · [候选组ONNX](../duck_forward_yaw_reward_candidate_e3072_s1250_20261120T232536/policy.onnx) · [候选组保存点](../../records/experiments/duck_forward_yaw_reward_candidate_e3072_s1250_20261120T232536/model-final.pt)

2026-11-21，固定前进候选权重，核对确定性推理、训练标准差下的高斯探索和训练幅值的闭环观测噪声。种子1247、摩擦0.8的20秒评估均为0/128。三组20秒平均绝对偏航速度的场景中位数分别为0.29468、0.28916和0.28219rad/s。此候选权重和对应保存点用于复现实验。[推理对照和归档环境](../../records/reports/forward-action-sampling/README.md) · [闭环对照和早期窗口复算](../../records/reports/forward-observation-noise-trial/README.md)

2026-11-20，低摩擦前进。两次低摩擦直接前进均为0/128，标称直接前进均为127/128。以种子1247配对，提高关节修正上限和替换为归档教师约束权重的通过数均为0/128。归档权重的20秒平均绝对偏航速度组内中位数从0.23030降到0.22656 rad/s，原上限为0.20 rad/s。此批为同场景策略评估，学习更新为0次。[直接前进配对](../../records/reports/roll-friction-stand-forward/README.md) · [关节修正记录](../../records/reports/forward-correction-limit-probe/README.md) · [权重对照与原片](../../records/reports/forward-teacher-actor-probe/README.md)

2026-11-20，滚翻低摩擦配对。两组各训练50次更新、5,529,600条环境转移。候选将脚底摩擦低于0.85的抽样概率从25%增至50%；两个开发种子的低摩擦滚翻分别为118/128和116/128，完整路线均为0/128。候选权重归档用于复现，低摩擦前进继续定位。[配对流程与视频](../../records/reports/roll-friction-paired-training/README.md) · [动作评分](../../records/reports/roll-friction-paired-development/README.md) · [均匀组ONNX](../duck_roll_friction_uniform_e4608_s1246_20261120T213457/policy.onnx) · [均匀组保存点](../../records/experiments/duck_roll_friction_uniform_e4608_s1246_20261120T213457/profile-final.pt) · [低摩擦候选ONNX](../duck_roll_friction_low-friction_e4608_s1246_20261120T213457/policy.onnx) · [低摩擦候选保存点](../../records/experiments/duck_roll_friction_low-friction_e4608_s1246_20261120T213457/profile-final.pt)

最新起身监控候选与同步组各训练 17 次更新；共同跌落开发对照分别为 1,006／1,024 和 998／1,024，父模型为 1,003／1,024。候选虽通过该开发门槛，但独立留出请求仰卧为 244／264，低于 251／264 门槛，未通过完整留出；两组权重保留为实验记录，不替换现有交付组件。[候选 ONNX](../duck_getup_auth-monitor_e3072_s3266_20261116T095241/policy.onnx) · [候选保存点](../../records/experiments/duck_getup_auth-monitor_e3072_s3266_20261116T095241/model-final.pt) · [同步组 ONNX](../duck_getup_auth-sync_e3072_s3266_20261116T095241/policy.onnx) · [同步组保存点](../../records/experiments/duck_getup_auth-sync_e3072_s3266_20261116T095241/model-final.pt) · [开发对照](../../records/reports/getup-monitored-pair-action-s3267/README.md) · [独立留出](../../records/reports/getup-monitored-pair-holdout-s4268/README.md)

2026-11-16：监控开销对照两组各完成 17 次更新、1,253,376 条转移；总耗时 172.19→133.13 秒，吞吐提高 29.3%。两组保存点、ONNX 和完整录像均已核验；单次顺序对照仍需复测，最终动作未独立评价，不替换本目录中的官方或自训组件。诊断权重保留在各自运行目录。[改动、模型与结果](../../records/reports/getup-monitored-authorization-20261116/README.md)

早期行走续训保留用于导出流程复盘。 [历史基线 ONNX](trained/walk-baseline-5000.onnx) · [历史基线保存点](trained/walk-baseline-5000.pt)

| 阶段 | 修改和训练量 | 核验结果 |
| --- | --- | --- |
| 起身固定路径开销 | 固定路径在进程内仅解析一次，每次暂停读取、协议及源码检查保留；同父点、种子和 3,072 环境，每组 17 次更新、1,253,376 条转移。 | 总耗时 209.16→153.18 秒，吞吐提高 36.5%；各 2,075 次回调累计 80.20→32.50 秒。首步已有数值微差，最终模型不同，未独立评价动作，不替换交付权重。两份诊断模型、完整原片和日志均保留。 [方案与结果](../../records/reports/getup-canonical-path-20261116/README.md) |
| 起身观察与吞吐 | 接入本机状态、曲线、完整原片和监督入口；独立修复 JSON 表示。1,536 和 3,072 环境各采集 1,253,376 条转移。 | 两组保存点、导出和 8 秒原片均通过。较大规模总过程吞吐提高 9.7%，显存多 5.16 GiB；尚未独立评价动作，不替换本表交付模型。首次失败另留 73,728 条转移。 [方案与结果](../../records/reports/getup-observed-throughput-20261116/README.md) |
| 官方标称路线与本机观察 | 固定官方策略，接入本机状态、曲线及按动作的完整录像；没有学习更新。 | 123/128 达到路线门槛；速度曲线与真实画面在 VS Code 自动刷新。该批只验证固定策略评估；真实训练的单独记录见上行，不同参数组仍需验收。 [评分与录像](../../records/reports/duck_roll-native_nominal_s1241_20261115T230228/README.md) |
| 标称限力类型修复 | 另建零步入口，检查时将布尔限力标志复制为原守卫接受的类型；原模型数组和权重不变，零学习更新。 | 真实模型的 14 个限力标志均已启用，标称编译限力为 ±0.97816 N·m；保存的 32 个接口样本最大输出差为 5.96×10⁻⁷。仅核验零步，不表示动态效果改善。 [结果与日志](../../records/reports/duck_roll-factor_nominal_s1241_20261115T224902/README.md) |
| 电机力限值检查 | 在零步入口增加编译力限值和源码冻结检查；不修改模型、不进行采样。 | 43 项入口假数据、8 项原守卫假环境和 3 项原入口来源检查通过；当时真实模型与 CUDA 零步尚未验证。 [检查](../../records/reports/force-limit-guard/README.md) |
| 参数对照初态审计 | 只读保存数据与现存执行器源码，不修改模型、不运行仿真。 | 七组构造种子 1176、重置种子 1240；执行器内部状态和编译电机力限值未保存，电压对力限值的影响仅有源码推算。 [证据与待补内容](../../records/reports/matched-start-source-audit/README.md) |
| 单因素零步接线 | 将配置与场景守卫接入独立零步入口；不改权重、不进行学习或物理采样。 | 34 项入口假数据检查和 6 项原守卫假环境检查通过；真实场景、观察器、CUDA 零步及动态运行仍待验证。 [检查与复现](../../records/reports/roll-factor-zero-entry/README.md) |
| 有界入口离线接线 | 新增独立名义入口、收集器和逐步停止判定，不修改历史模型或运行脚本。 | 原轨迹与假环境验证门槛不可达、异常、暂停和保存；该名义入口不支持非名义参数，多臂入口、真实观察与 CUDA 零步仍未完成。 [入口记录](../../records/reports/roll-domain-parameter-random-s20/guarded-entry.md) |
| 参数随机化筛查 | 固定官方模型和该批控制器，不学习；对 0.8、1.05、1.3 摩擦和 7.4、7.7、8.0 电压组合做 128 环境筛查。该批落地过渡与 S17 不同。 | 六臂已完整评分：低摩擦三臂为 110/128、108/128、112/128，中摩擦/低电压 125/128，高摩擦/低电压 124/128，中摩擦/高电压连续路线 25/128。用户要求暂停时高摩擦/名义电压臂为 1400/1550 步，未完成区间准入。 [暂停记录](../../records/reports/roll-domain-parameter-random-s20/README.md) |
| 离线真机准备 | 不训练、不连接实体设备；固定官方模型哈希、ONNX 接口、策略关节顺序、软件映射、15 行标定模板和回退材料。 | 软件准备审计通过。实体舵机 ID、机械中心、方向、IMU、通信和供电仍待现场记录，官方参照保持为后续硬件基线。 [审计结果](../../records/hardware/deployment-preparation/s19-readiness.md) |
| 行走基线 | 复现官方 PPO 配置，累计 5,000 次更新并导出。 | 导出一致性通过，固定动作跟踪低于门槛。 [导出记录](../baseline_continuation_20261101T101809_s42/manifest.json) |
| 行走修复 | 对照目标越界惩罚与可达头部指令，成对试验各 250 次更新。 | 减少了部分越界请求，早期并脚与动作跟踪仍需修复。 [过程与结果](../../docs/reports/development_report.md) |
| 官方行走控制 | 固定权重，加入指令映射与转向反馈，零学习更新。 | 原行走环境七指令 672/672。共同接口留出前进 30/32，低于门槛。 [七指令](../../records/reports/seven-command/confirmation-summary.json) · [共同接口](../../records/reports/common-walk-confirmation/README.md) |
| 前进执行器反馈 | 固定官方权重，按陀螺仪和双髋速度修正前进执行器目标，单步限于 0.03 rad。 | 两批开发七项各 224/224。留出前进 32/32，配对偏航收益 0.01749–0.01854 rad/s。后退 30/32，整组未晋级。 [开发](../../records/reports/common-walk-actuator-yaw-feedback/README.md) · [复验](../../records/reports/common-walk-actuator-yaw-feedback-replication/README.md) · [留出](../../records/reports/common-walk-actuator-yaw-feedback-heldout/README.md) |
| 后退启动助推 | 固定官方权重，在低髋膝活动时段施加 0.4 秒限幅目标修正，零学习。 | 两批共同停滞挽回 0/6、1/7，低于双批次门槛。 [诊断与回放](../../records/reports/common-walk-backward-onset/README.md) · [助推结果](../../records/reports/common-walk-backward-start-pulse/README.md) |
| 后退双向侧移 | 同初态固定策略的四路对照，两个开发种子，403,200 条转移。 | ±0.015 m/s 均未挽回 15 个共同停滞，140 毫秒侧移为 0.123–0.146 mm。 [报告与原片](../../records/reports/common-walk-backward-lateral-probe/README.md) |
| 后退支撑转移 | 双脚支撑时平滑修正关节目标，限幅 0.03 rad，原权重与历史不变。 | 右向挽回 3 次、丢失 1 次，三次均未形成规定窗口内的单脚支撑。关闭该方案。 [评分与观察](../../records/reports/common-walk-backward-actuator-support/README.md) |
| 滚翻接前进 | 固定官方权重，偏航反馈乘数由 0.12 提高到 0.15 rad/s，关节修正仍限 0.03 rad。 | 同初态完整序列由 110/128 提高到 122/128，新开发种子 124/128，均达到原门槛。扰动单独验收。 [配对评分](../../records/reports/continuous-forward-yaw-pair/README.md) · [独立复验与原片](../../records/reports/continuous-forward-yaw-replication/README.md) |
| 前进抗扰 | 同初态比较无扰动与一次左右 0.08 m/s 水平速度增量，零学习。 | 完整序列和恢复分别通过 125/128、124/128、125/128。左推一例偏航略超限。 [评分与原片](../../records/reports/continuous-forward-lateral-push/README.md) |
| 起身接前进 | 从同一站立终态比较立即指令与半秒渐升，使用相同的已归档模型，零学习。 | 起步严重姿态异常由 185 降到 0。完整回合的开发与独立留出分别为 2,976/3,072、2,990/3,072，均通过原门槛。 [起步对照](../../records/reports/continuous-routed-forward-entry-ramp/README.md) · [留出原片](../../records/reports/continuous-routed-getup-ramped-forward/holdout/README.md) · [连续仿真交付](ramped-getup-standing-forward-simulation.json) |
| 起身接七指令 | 恢复左右转使用的原陀螺仪反馈。固定模型，零学习。 | 同站立状态配对从 2,923 提高到 2,977/3,072。独立留出 2,992/3,072，七项均通过原门槛。 [留出评分与原片](../../records/reports/continuous-routed-turn-feedback-holdout-zero-recovery/README.md) · [连续仿真交付](routed-seven-command-simulation.json) |
| 起身仰卧加权 | 由同一父模型续训。等权组和加权组各 50 次，每组 614,400 条转移。 | 父模型 303/320，等权 302/320，加权 310/320。指定仰卧 30/32、实际右侧 37/41。 [评估](../../records/reports/getup-supine-weighted-learning/analysis.json) |
| 起身动作保持 | 约束原有成功动作的输出漂移。新对照完成 32 次更新后触发停止线。 | 保留停止点及复盘输入，候选组未启动。 [停止核验](../../records/reports/getup-recovery-retention/stop-verification.json) |
| 起身自适应回放 | 从成功观测中动态选择偏差最大的状态与关节。1,536 和 3,072 环境各采集 1,253,376 条转移。 | 训练与导出通过核验。新种子同次释放评估为固定父模型 312/320、1,536 环境 303/320、3,072 环境 306/320，候选未通过分姿态门槛。 [本轮训练](../../records/reports/getup-local-ppo-20261110T011523/README.md) |
| 起身姿态分流 | 首次有效重力观测锁定固定父模型或右侧专用模型，零新训练。 | 开发 313/320、316/320，留出 314/320。指定仰卧 30/32 低于门槛，未晋级。 [记录与原片](../../records/reports/getup-pose-routing/README.md) |
| 仰卧早期支撑 | 保持相同初态和动作保护，两组各 17 次更新、1,253,376 条转移。 | 新场景父策略 312/320、对照 306/320、候选 310/320。候选通过四姿态动作门槛，实际仰卧净增 3 次低于晋级门槛，权重留作开发复盘。 [评分与原片](../../records/reports/getup-early-support-ppo-functional/README.md) |
| 起身回放覆盖 | 两组同奖励与采样预算，只替换候选 180 条成功回放。各 17 次更新，Windows 观察与权重独立核验。 | 固定父策略 311/320、对照 311/320、候选 315/320。指定仰卧 30/32，实际仰卧净增 3 次，均未过晋级线。 [独立评分与原片](../../records/reports/getup-side-to-supine-replay/evaluation/README.md) |
| 右侧起身进展奖励 | 两组共享四姿态回放与价值起点，各 17 次更新、1,253,376 条转移。仅候选获得右侧受阻后的有符号直立进展奖励。 | 父模型 311/320、对照 316/320、候选 308/320。候选比对照少通过 1 个实际右侧和 6 个实际仰卧场景。三组指定仰卧均未过线，权重列为开发证据。 [训练记录](../../records/reports/getup-right-upright-ppo/README.md) · [独立评分与原片](../../records/reports/getup-right-upright-ppo/evaluation/README.md) |
| 滚翻角度输入 | 两组各 50 次更新。共用原奖励与角度任务，仅候选允许策略学习角度输入。 | 官方 28/32，对照 29/32，候选 27/32，关闭本轮。 [评估](../../records/reports/roll-angle-feature/startup-recovery/analysis.json) |
| 滚翻连续切换 | 在同一全接触环境从站立切换到官方滚翻，比较真实支撑和目标过渡。 | 128 个场景均超过头部穿透限值，后续站立与前进未启动。部分原片和完整评分已归档。 [连续试验](../../records/reports/continuous-roll-stand-walk/README.md) |
| 滚翻入口隔离 | 共享站立结束快照，只改变滚翻目标的投影与入口限速。 | 原始目标 119/128，仅投影 124/128，投影加入口限速第 22 步自动停止。 [三路结果](../../records/reports/continuous-roll-entry-isolation/README.md) |
| 投影入口连续试验 | 保留滚翻入口目标投影，去掉入口限速，保持滚翻后转站立过渡。 | 126 个完成前进 20 秒，111 个通过全部判据，15 个仅偏航未通过。 [独立结果与原片](../../records/reports/continuous-roll-projected-entry/README.md) |
| 仰卧腿部转正 | 在同释放状态比较两路原策略与 22 步有界腿部修正。 | 6 个实际修正场景均未达到原起身与持稳判据。指定仰卧 31/32，右侧和配对门槛未通过。 [首批结果与原片](../../records/reports/getup-upright-recovery-pair/README.md) |
| 官方滚翻确认 | 同一轨迹对照接触读取时点，零学习更新。 | 新开发和独立留出在对齐接触读数后各 117/128，通过单动作门槛。 [独立结果与原片](../../records/reports/roll-endpoint-confirmation/README.md) |
| 后退侧倾启动 | 四路共享初态，比较原策略与左右两组限幅关节目标。固定权重，共 201,600 条实际转移。 | 左右均挽回 0/5，右组另丢失 1 个原启动场景。保持结束时侧倾仅增加 0.070°、0.040°，本轮停止。 [核验与原片](../../records/reports/common-walk-backward-roll-pair/README.md) |
| 后退启动时序 | 对齐首次持续单脚支撑，逐步复算质心、机身、膝角和下发目标。两次从同一原始档案复算一致。 | 59 个自然启动和 325 个停滞窗口中，四批均未通过固定动作顺序门槛。左右脚法向力通道已在 8 个保存姿态通过 CPU 零步核验。 [时序](../../records/reports/common-walk-backward-natural-timing/README.md) · [力通道](../../records/reports/common-walk-backward-load-clock/README.md) |
| 后退脚底受力 | 原策略 CUDA 采集，独立核对最后子步的实际力与保存状态。 | 承重标记仅提前覆盖 5/15 个自然迟启动场景，未晋级。保留力读数、观察曲线和原片。 [受力图与原片](../../records/reports/common-walk-backward-load-clock/README.md) |
| 后退初态课程 | 从十批开发档案重新计算原策略的停滞标签，核对并去重初始状态。 | 2,048 个场景中得到 63 个不同的困难初态，进入重置读回准备。原批次状态按索引取用。 [困难初态](../../records/reports/common-walk-backward-start-curriculum/README.md) |
| 后退初态重置 | 在原生重置阶段写入困难初态，复核整批、部分重置和六指令保留样本。 | CPU 与 CUDA 零步均通过。3,072 条官方参考动作复算一致。初态差异集中在平面位置和偏航，需先通过新随机流复验。 [读回与复验方案](../../records/reports/common-walk-backward-start-curriculum/README.md) |
| 后退初态复验 | 固定策略比较普通重置与困难初态，两组共 76,800 条实际转移。 | 困难初态净增 3 个停滞，低于 4 个门槛。前两秒筛选在旧轨迹选出 163 个候选，其中 63 个持续停滞，进入完整状态采集准备。 [配对复验](../../records/reports/common-walk-backward-start-enrichment/README.md) · [轨迹筛选](../../records/reports/common-walk-backward-stall-prefix/README.md) |
| 后退状态恢复 | 896 个 CUDA 环境采集完整停滞状态，并从同一保存点续跑两路。 | 448,000 条实际转移。309 项磁盘恢复状态一致，首步位置误差超限，关闭该路线。 [核验与原片](../../records/reports/common-walk-backward-stall-capture/README.md) |
| 后退在线加权 | 从真实轨迹复算停滞识别、重置清除和 PPO 策略损失，再接入原生环境。 | 离线两端一致。原生 CUDA 采样 393,216 条转移，独立核验自动重置和价值目标。首档净吞吐约 13,279 条转移/秒。 [复算、计时与原片](../../records/reports/common-walk-backward-online-focus/README.md) |
| 后退并行规模 | 同一 CUDA 采样、录像和写盘链路，按相反顺序比较并行规模。 | 四次计时完成 2,359,296 条实际转移。2,048 档两遍约 2.1 万转移/秒，4,096 档显存超限于零步停止。 [结果、原片与核验](../../records/benchmarks/duck_walk_native-scale_reference_s1134_20261110T085345/README.md) |
| 后退规模细测 | 在 2,048 到 3,584 环境之间按相反顺序测八次，保留录像和日志开销。 | 3,072 档约 2.64–2.67 万转移/秒，较 2,048 档快 23% 和 30%。3,584 档额外收益低于门槛。 [计时、显存与原片](../../records/benchmarks/duck_walk_native-scale_reference_s1135_20261110T090325/README.md) |
| 后退 PPO 核验 | 3,072 环境实际采样并执行 20 个梯度小批量，预留 Adam 状态。 | 价值目标复算通过，显存距预设上限约 1,450 MiB，策略及价值权重哈希均一致。 [梯度、显存与原片](../../records/evaluations/duck_walk_ppo-headroom_reference_s1136_20261110T091415/README.md) |
| 后退共享价值 | 固定官方策略采集 3,072 个首回合，按环境划分拟合与验证，执行 100 次价值更新。 | 独立回读解释方差 0.830，归一化均方误差 0.00757。两组可共享价值参数和优化器状态。 [训练、核验与检查点](../../records/experiments/duck_walk_shared-value_reference_s1137_20261110T092228/README.md) |
| 后退配对 PPO | 同起点、同奖励和预算，两组各 42 次更新和 3,096,576 条转移。 | 候选加权 5,800 条停滞样本，检查点和 ONNX 已独立回读，两组末批价值损失突升。 [对照](../../records/experiments/duck_walk_backward-focus-ppo_control_s1138_20261110T092909/README.md) · [候选](../../records/experiments/duck_walk_backward-focus-ppo_candidate_s1138_20261110T092909/README.md) |
| 后退七指令对照 | 两个新种子比较官方、等权和停滞加权，保留同初态、前进反馈和原动作门槛。 | 一批后退均为 32/32，另一批两组自训均从 30/32 提高到 32/32。两组前进速度损失超限，未晋级。下一步核验按指令保留官方前进。 [逐项评分与原片](../../records/reports/common-walk-backward-paired-eval/README.md) |
| 七指令模型选择 | 外部前进选官方，其余六项选等权。反馈、原始动作历史和物理限幅一致。 | 两批前进速度恢复到官方水平，后退均为 32/32。首批右转 30/32，低于门槛，暂未进入同场景切换。 [评分、录像和右转图](../../records/reports/common-walk-forward-routing/README.md) |
| 右转启动限幅 | 同一批次形状，两组计算同一候选目标，修正组仅在低角速度时短时下发。 | 旧种子均为 30/32，新种子均为 32/32。首步目标相同但物理轨迹分歧，未通过配对验收，关闭本次修正。 [评分、分歧与回放](../../records/reports/common-walk-routed-right-onset/README.md) |
| 同进程参考重复 | 共享前 200 步，从同一保存状态重复两个 50 步参考后缀。另核对第一个控制的五个物理子步。 | 恢复与首个子步输入逐位一致，积分后分歧。右转一秒平均观测角速度最大差为 0.00459 rad/s。 [复算与录像](../../records/reports/common-walk-shared-prefix-repeat/README.md) |
| 右转腿部修正 | 双参考共享前 200 步。候选在低角速度时仅修正十个腿关节，目标限幅 0.03 rad，头颈与原始动作历史保留名义路径。 | 旧批两例触发均未挽回，三组均为 30/32。新批未触发，三组均为 32/32。关闭该方案。 [评分与录像](../../records/reports/common-walk-right-leg-recovery/README.md) |
| 右转启动训练准备 | 保留原偏航反馈和全程奖励，在原生重置前记录前 4 秒的右转欠速项。 | 两遍固定权重短测共 1,474,560 条转移，净吞吐约 22,700 条/秒，峰值显存 12,167 MiB。重置时序、价值目标与原片已核验。 [短测与录像](../../records/reports/common-walk-right-startup-ppo/README.md) |
| 右转共享价值 | 从同一批原生首回合重建两路折扣回报，以均值拟合共同价值。 | 3,072 个完整首回合，100 次价值更新。独立验证解释方差 0.836、归一化均方误差 0.00728，策略权重逐位一致。 [价值检查点与原片](../../records/experiments/duck_walk_right-startup-shared-value_reference_s1147_20261110T113806/README.md) |
| 右转启动 PPO | 同策略、价值和样本数，只在候选中加入系数 0.25 的前 4 秒欠速惩罚。 | 两组各 14 次更新和 1,032,192 条转移，检查点和 ONNX 已通过独立读回。 [原奖励组](../../records/experiments/duck_walk_right-startup-ppo_control_s1148_20261110T114313/README.md) · [欠速组](../../records/experiments/duck_walk_right-startup-ppo_candidate_s1148_20261110T114313/README.md) |
| 右转训练动态评估 | 固定前进与其他五条指令的策略，比较原路由与两种训练导出。 | 两批六组共 1,344,000 条评估转移。右转三组同为 30/32 和 32/32。欠速奖励未增加成功数，模型留作实验参考。 [评分、原片和回放](../../records/reports/common-walk-right-startup-seven-command/README.md) |
| 右转早期切换 | 在第 76 步比较原策略、只计算官方动作和实际切换。 | 三组共 672,000 条转移。两例真实切换均未挽回，三组右转各 30/32，其余六项各 32/32。 [评分和复查](../../records/reports/common-walk-early-official-right/README.md) |
| 右转支撑短对照 | 两条参考和右脚目标共享前 75 步，只给试验组下发不超过 0.03 rad 的 0.6 秒修正。 | 134,400 条转移。场景 13 无持续支撑，场景 198 的试验组与未修正参考一分别在 2.60 秒、2.58 秒形成支撑。关闭固定修正。 [评分和两段回放](../../records/reports/common-walk-right-support-transfer/README.md) |
| 右转原生接触采样 | 固定权重，两遍各 3,072 环境，增加重置前的接触记录。 | 1,474,560 条转移，第 3 秒前分别确认 29、30 个事件。一次性奖励 0.07 的在线 CPU 重放通过。 [采样与复现](../../records/reports/common-walk-right-support-yaw-probe/README.md) |
| 右转原生奖励接线 | 在真实 CUDA 奖励管理器中加入每回合最多一次的 0.07 支撑启动奖励，保持固定策略。 | 737,280 条转移，独立匹配 26 个事件。终止步增量为零，实际总奖励的最大复算误差为 6.24×10⁻⁸。 [接线与回报](../../records/reports/common-walk-right-support-yaw-training/README.md) |
| 右转共享价值 | 将原奖励与候选奖励取均值，按完整首回合划分独立验证组。 | 3,072,000 条转移、100 次价值更新。35 次奖励事件中 9 次在验证集。验证解释方差为 0.828，策略权重保持一致。 [独立核验](../../records/reports/common-walk-right-support-yaw-training/value-zero-step-recovery/shared-value-analysis.json) |
| 右转支撑奖励 PPO | 两组共用初态、价值与策略起点，分别完成 14 次更新、1,032,192 条转移。 | 候选组实际获得 4 次启动奖励。两个保存点、终点 ONNX 和逐步回报通过独立核验。 [训练核验](../../records/reports/common-walk-right-support-yaw-training/paired-ppo-analysis.json) · [曲线与原片](../../records/walk-right-support-paired-ppo-live.html) |
| 右转支撑奖励评估 | 两个开发种子、三种固定策略、原七指令判据，共 1,344,000 条转移。 | 支撑奖励组右转两批均为 32/32。两组训练在新种子均丢一例前进，均未晋级。 [成绩与证据](../../records/reports/common-walk-right-support-yaw-training/README.md) |
| 前进偏航重复 | 一次重置后按相同状态恢复，比较两次父策略、只计算右转候选、实际使用右转候选。 | 89,600 条转移、零学习。两秒最大偏航误差配对差为 0.02654、0.01507 和 0.02684 rad/s。 [诊断与复算](../../records/reports/common-walk-right-support-yaw-training/README.md) |
| 前进反馈幅度 | 沿用连续动作的 0.15 rad/s 候选上限，保持 0.03 rad 关节修正。 | 保存轨迹 121,600 条输入通过投影。完整评估 672,000 条转移，扩大反馈的前进 31/32、右转 32/32，偏航改善低于继续门槛，新种子未启动。 [投影核验](../../records/reports/common-walk-right-support-yaw-training/forward-feedback-margin-cpu.json) · [独立评分](../../records/reports/common-walk-right-support-yaw-training/forward-margin/known/analysis.json) |
| 单脚支撑识别 | 以现有关节与 IMU 观测识别支撑侧，仿真接触用作离线标签。 | 不同种子的 127,872 条观测通过识别门槛，进入分相位反馈预检。 [支撑时序](../../records/reports/common-walk-right-support-yaw-training/forward-yaw-phase-readback.json) · [识别核验](../../records/reports/common-walk-right-support-yaw-training/forward-stance-observation-cpu.json) |
| 单脚支撑脉冲 | 第 85 步前共用状态，比较两次原反馈、正脉冲和负脉冲，原关节限制不变。 | 33,152 条转移。两侧即时响应方向通过，固定左侧正脉冲使误差增加 0.00179 rad/s，方案未晋级。 [评分与原片](../../records/reports/common-walk-right-support-yaw-training/stance-response/analysis.json) |
| 偏航方向拟合 | 保留四帧和原网络大小，训练目标从下一步幅度改为方向。 | 四批开发准确率 96.2%–97.4%，覆盖 73.4%–77.6%。两侧保存脉冲重选均降低首步偏航，进入连续四秒试验。 [拟合与重选](../../records/reports/common-walk-right-support-yaw-training/forward-yaw-direction-fit-cpu.json) |
| 连续偏航反馈 | 从第 84 步同一状态恢复，在一个分支中仅对高置信场景加入至多 0.02 rad/s。 | 96,768 条转移，首步两侧均改善，随后 2.32 秒偏航较两组改善约 0.007 rad/s，准入原七指令评估。 [独立复算](../../records/reports/common-walk-right-support-yaw-training/yaw-direction-pilot/analysis.json) |
| 方向反馈七指令 | 固定候选反馈，恢复同一状态做三路 20 秒完整评估。 | 634,368 条转移。候选左转 31/32，其余各 32/32，前进偏航比两路参照各改善 0.00577、0.00555 rad/s，准入新开发种子。 [完整评分](../../records/reports/common-walk-right-support-yaw-training/yaw-direction-development/analysis.json) |
| 方向反馈独立复验 | 固定反馈与原门槛，仅更换预留种子及场景分组。 | 634,368 条转移，三路七项各 32/32。候选前进偏航比两路参照各改善 0.00720、0.00618 rad/s，准入最终留出。 [独立评分](../../records/reports/common-walk-right-support-yaw-training/yaw-direction-fresh/analysis.json) |
| 方向反馈最终留出 | 冻结三套策略组件、方向规则和限幅，一次性评估种子 2153。 | 634,368 条转移，七项各 32/32。前进偏航较两路参照各下降 0.00603、0.00524 rad/s。实际使用的权重与保存点已逐字节核验。 [最终留出](../../records/reports/common-walk-right-support-yaw-training/yaw-direction-holdout/analysis.json) |
| 四姿态起身分流 | 固定仰卧策略与共同回退，只在实际右侧加用已有右侧策略。两批开发和一批留出共 4,454,400 条评估转移，零学习。 | 两批开发均过线。留出总体 993/1,024，实际右侧 69/73，差 1 例，候选未晋级。 [最终评分与原片](../../records/reports/getup-dual-pose-route/holdout/README.md) |
| 右侧起身扩大覆盖 | 沿用分流与固定权重，3,072 个开发场景包含 960 个实际右侧释放。 | 候选总体 3,010/3,072，实际右侧 936/960，通过原门槛，进入新留出准备。 [评分和完整回放](../../records/reports/getup-right-robustness/development/README.md) |
| 四姿态起身独立留出 | 恢复原姿态请求比例，单次运行新种子并独立核对实际姿态、持稳及分流。 | 候选总体 2,983/3,072，实际右侧 179/183。归档三份训练 ONNX 和对应保存点，保留历史仰卧加权研究。 [独立评分与录像](../../records/reports/getup-right-robustness/holdout/README.md) · [历史评价](../../records/reports/getup-supine-weighted-learning/README.md) |
| 滚翻头部支撑脉冲 | 在同一滚翻状态首次跨过 120° 时，对头部横滚目标施加两步、每步 −0.02 rad 的有界修正。 | 原控制完整序列为 122/128、117/128、126/128。脉冲为 119/128、119/128、124/128，标称分支退化，候选关闭。 [六组评分与原片](../../records/reports/continuous-roll-head-support-pulse/resume-20261113/README.md) |
| 物理端点筛查 | 单因素配置脚底摩擦 0.7、1.3 和供电 7.4、8.0 V，保留原控制和 116/128 门槛。 | 四组均完成独立核验：低摩擦 106/128、高摩擦 120/128、低电压 123/128、高电压滚翻 118/128 但连续前进 0/128。通过端点进入交叉组合；失败尝试和修复记录已归档。 [端点报告](../../records/reports/roll-domain-parameter-screen/README.md) · [失败记录](../../records/reports/roll-domain-parameter-screen/preflight-failures/README.md) |
| 物理端点交叉复验 | 新 seed 1190 复测标称、高摩擦、低电压和高摩擦低电压组合，保持原控制与 116/128 门槛。 | 四组分别为 123/128、125/128、124/128、125/128，连续前进分别为 122/128、125/128、124/128、125/128，全部通过。候选随机化区间暂定为脚底摩擦 1.0–1.3、供电 7.4–7.7 V。 [交叉报告](../../records/reports/roll-domain-parameter-cross-check/README.md) |
| 参数随机化短对照 | 同父起身训练接口，标称范围与脚底摩擦 1.0–1.3、供电 7.4–7.7 V 候选范围，8 次更新、每组 196,608 条转移。 | candidate/control 均完成。末次稳定率 93.75%/91.89%，最大 drift 0.1366/0.1432 rad，最大显存 30.73%/34.34%。短预算尚未通过原动作回归，保留末次 PT、ONNX、曲线和真实片段。 [训练报告](../../records/reports/parameter-randomized-training/README.md) |
| 参数随机化原动作回归 | 同 seed 1202 对两个末次 ONNX 做四姿态起身、官方滚翻和七指令 20 秒屏幕，零学习。 | 起身 prone/supine 32/32、侧躺 19/32；滚翻 0/32；七指令完整运行但前后进退、转向和平移误差超门槛。两组均保留完整轨迹和视频，未晋级。 [回归报告](../../records/reports/parameter-randomized-regression/README.md) |

完整源码快照、日志、轨迹、中间保存点和录像按原批次查找。 [训练归档](../../records/training-archive/README.md) · [全流程报告](../../docs/reports/development_report.html)
| 滚翻 airborne-support 配对 | 同一官方父模型、seed 1210、控制组与候选组各 50 次更新；候选只增加 30–240° 空中前向旋转惩罚。 | 独立 seed 1211 评估官方 108/128、控制组 112/128、候选 111/128。候选无新增头部穿透或关节限位失败，但未过 116/128 和相对控制组 +4 门槛，关闭且不晋级。 [失败分析](../../records/reports/roll-airborne-support/analysis.json) · [训练记录](../../records/reports/roll-airborne-support/README.md) |
| 官方组件连续路由 seed-1212 | CPU 场景冻结、GPU 零步后执行 `stand → roll → stand → forward`，无学习更新。 | 746 控制步、95,488 条转移、3,730 个物理子步后早停；滚翻→站立 0/128、前进 0/128。保留状态、日志、关键帧和部分原片，不重试同 seed。 [停止复盘](../../records/reports/continuous-roll-stand-walk-s1212/stop-review.json) |
| 官方组件 router 边界与 bridge 消融 | seed-1213/1214 做起始站立和 phase/latch 筛查；seed-1215 只改变跨模型 bridge。 | 起始站立第 50 步 128/128；seed-1214 进入 ROLL 128/128、latch 1/128；bridge 保留时头部穿透 127/128，关闭时 0/128。均为零学习诊断。 [消融分析](../../records/reports/continuous-roll-bridge-ablation-s1215/analysis.json) |
| bridge 关闭完整滚翻边界 seed-1216 | 同一官方组件，50 步站立加 450 步滚翻，关闭跨模型 bridge。 | 121/128 达到滚翻→落地 latch，超过 116/128；最大头部穿透 0.0032455 m，关节限位和头部穿透超限均为 0。7 个失败保留，尚未进入站立接力。 [独立分析](../../records/reports/continuous-roll-bridge-disabled-boundary-s1216/analysis.json) |
| 前进反馈上限固定控制探针 seed-1220 | 只将 `maximum_yaw_input_rad_s` 从 0.12 提到 0.15，其他路由组件、评分和奖励冻结，零学习。 | 完整路线滚翻入口 128/128、落地 127/128、前进入口 127/128、前进成功 127/128；前进偏航失败为 0。通过开发门槛，等待独立 seed 确认，不作为模型晋级。 [探针记录](../../records/reports/router-forward-feedback-probe-s1220/README.md) |
| 前进反馈独立确认 seed-1221 | 复用 0.15 rad/s 固定反馈上限，只更换随机流，零学习。 | 完整路线滚翻入口 128/128、落地 121/128、前进入口 121/128、前进成功 121/128；前进偏航失败为 0。与 seed 1220 一起通过开发门槛，进入留出确认和轻量 PPO 设计。 [确认记录](../../records/reports/router-forward-feedback-confirmation-s1221/README.md) |
| 前进反馈一次性留出 seed-1222 | 复用 0.15 rad/s 固定反馈上限，只更换随机流，不调参，零学习。 | 完整路线滚翻入口 128/128、落地 125/128、前进入口 125/128、前进成功 124/128；超过 116/128。固定参数进入候选路由冻结和轻量 PPO 设计。 [留出记录](../../records/reports/router-forward-feedback-holdout-s1222/README.md) |
| 官方与自训练行走权重连续路由 seed-1223 | 同 seed、同路由和同评分比较官方 walk ONNX 与已归档 `walk-common-base.onnx`，零学习。 | 官方前进 119/128，通过连续门槛；自训练前进 111/128，速度中位数 0.1452 m/s 且偏航失败 7 个，未晋级。下一步冻结滚翻后前进状态的轻量 PPO 方案。 [对比记录](../../records/reports/router-self-trained-walk-compare-s1223/README.md) |
| 滚翻后前进状态课程 seed-1224 | 从 seed 1220–1222 的真实连续轨迹提取前进入口和随后 50 步，零物理、零学习；以官方 actor 为父点编写 5 次更新试运行草案。 | 373 个入口状态通过有限值、形状和 phase 回读；尚未启动 PPO，等待状态恢复和 GPU 零步。 [课程报告](../../records/reports/route-aware-walk-training/README.md) |
| bridge 关闭滚翻接力 seed-1217 | 延长落地站立并记录前进相位入口，零学习、无重置。 | 116/128 完成滚翻→站立，116/128 进入前进相位，冻结完整路由前置接口。 [接力分析](../../records/reports/continuous-roll-stand-handoff-s1217/analysis.json) |
| 完整无重置路由 seed-1218 | 50 步站立、450 步滚翻、50 步落地站立、1,000 步前进，bridge 关闭，恢复原 20 秒前进评分。 | 滚翻入口 128/128、落地 123/128、前进入口 123/128；完整前进 111/128，低于 116/128。5 个接力失败、12 个偏航失败，暂不晋级。 [完整分析](../../records/reports/continuous-roll-full-sequence-s1218/analysis.json) |
| 滚翻后前进状态课程 pilot 预检 | 官方 walk actor 为父点，373 个真实前进入口课程，4096 环境，critic warmup 1 次，pilot 5 次更新；入口只构造配置。 | Ubuntu-24.04 CUDA 配置预检通过，`cuda_initialized=false`，物理步和学习更新均为 0；尚未形成新权重。 [入口预检](../../records/reports/route-aware-walk-training/pilot-config-preflight.json) · [冻结协议](../../configs/experiments/route_aware_walk_training_s1224.json) |
| 滚翻后前进状态课程入口修复 | 首次启动只完成环境构造和 `model_init.pt` 保存，未进入学习；复用官方导入器时补充 route-aware 的扁平兼容字段。 | 入口在物理步 0、学习更新 0 处停止，失败证据保留，修复后重新预检。 [失败记录](../../records/reports/route-aware-walk-training/entry-failures/initialization-missing-flat-fields.json) |
| 滚翻后前进状态课程五次更新 | 官方 walk actor 初始化，373 个真实前进入口课程，4096 环境，1 次 critic warmup、4 次 actor 更新，共 491,520 个采样步。 | 末次采样约 23,232 steps/s，均值奖励 1.63→18.60，guard 无异常；导出候选等待 seed 1225 原始完整路线评分。 [训练状态](../../records/experiments/duck_walk_route-aware-entry_candidate_s1224_20261114T045600/state.json) · [导出清单](../../artifacts/duck_walk_route-aware-entry_candidate_s1224_20261114T045600/manifest.json) |
| route-aware pilot 同 seed 路线对照 | 同一 seed 1225、bridge-disabled 路由、0.15 rad/s 反馈和原门槛；只比较官方父模型与五次更新 pilot 行走 actor。 | 官方 125/128，pilot 28/128；pilot 偏航失败 95、官方 0，速度中位数相近。pilot 拒绝晋级，停止追加学习。 [对照报告](../../records/reports/route-aware-pilot-parent-compare-s1225/README.md) |
| route-aware actor 漂移恢复探针 | 低学习率、1 次 critic warmup 加 1 次 actor 更新的受保护短测；每次更新比较 64 个课程入口的确定性输出。 | 两次更新均通过输出漂移门禁，候选保留等待独立路线评估。 [冻结协议](../../configs/experiments/route_aware_actor_drift_probe_s1226.json) · [门禁记录](../../records/experiments/duck_walk_route-aware-drift-probe_candidate_s1226_20261114T061500/output-drift-update-2.json) |
| route-aware seed 1228 开发对照 | 同一固定路由和原门槛，比较 guarded probe 与官方父模型，零学习。 | guarded probe 124/128、官方 123/128，偏航均为 0；仅作为开发种子通过，进入独立种子复验。 [对照报告](../../records/reports/route-aware-probe-parent-compare-s1228/README.md) |
| route-aware seed 1229 独立对照 | 保持 seed 1228 的路由、反馈上限、门槛和模型，比较候选与官方父模型，零学习。 | 候选 119/128、官方 121/128，双方均通过原路线和安全门槛；候选没有优势，关闭该学习线，保留完整复现证据。 [对照报告](../../records/reports/route-aware-probe-parent-compare-s1229/README.md) |
| route-aware seed 1227 留出对照 | 按冻结探针协议复用相同路由、反馈上限、门槛和模型，比较候选与官方父模型，零学习。 | 候选 124/128、官方 121/128；与 seed 1228、1229 合并为 367/384 对 365/384，但逐种子收益不稳定，关闭当前学习线，保留完整留出证据。 [对照报告](../../records/reports/route-aware-probe-parent-compare-s1227/README.md) |
| route-aware 教师约束短训 | 在 373 个真实滚翻后前进入口上加入 detached 官方动作锚点；4096 环境、1 次 critic warmup、2 次 actor 锚点步骤，三次更新封顶。 | CPU admission、GPU 零步、输出漂移和 ONNX 回放均通过；实际教师锚点损失与训练视频留档。短训权重只作开发证据。 [训练协议](../../configs/experiments/route_aware_teacher_anchor_s1230.json) · [导出复核](../../records/reports/route-aware-teacher-anchor-s1230/export-verification.json) |
| route-aware 教师约束开发/留出 | 固定教师约束导出，seed 1231 开发、seed 1232 留出，与官方父模型使用同一路由和门槛。 | 候选 121/128、122/128；官方 122/128、121/128；两组均无偏航和安全失分，合计 243/256 对 243/256。候选不晋级，官方保留为后续仿真和真机准备基线。 [开发对照](../../records/reports/route-aware-teacher-anchor-parent-compare-s1231/README.md) · [留出对照](../../records/reports/route-aware-teacher-anchor-parent-compare-s1232/README.md) |

## 最新滚翻与路由审计

S14b 在 seed-1173 保存轨迹上对 109 个因果特征做留一环境筛查，左右扰动均未达到失败覆盖和成功误触发门槛。S15 对 seed-1218 的完整滚翻→站立→前进切换做保存数据复核，确认 123/128 个世界进入前进，但偏航失败前 50 步没有稳定早期信号。S16 在新 seed-1234 上采集官方参考、左推和右推完整轨迹，独立评分为 123/128、119/128、127/128；三臂通过原门槛，但留一审计仍未形成可迁移触发器。S16 首次左推采集的 guard 边界失败、独立进程重跑日志、三臂视频和轨迹全部保留。由于没有选择性控制信号，未追加零步、动力学或 PPO；官方权重继续作为唯一后续真机准备基线，自训教师约束权重仅作研究复现材料。 [S14b 审计](../../records/reports/roll-phase-perturbation-review/s14b-causal-feature-audit.json) · [S15 审计](../../records/reports/continuous-roll-full-sequence-s1218/route-transition-causal-s15.json) · [S16 评分](../../records/reports/roll-phase-perturbation-review/s16-score.json) · [S16 时序审计](../../records/reports/roll-phase-perturbation-review/s16-temporal-audit.json) · [S16 失败记录](../../records/reports/roll-phase-perturbation-review/s16-run-failure.json)

S17 在 seed-1235 上对官方固定路线完成 zero-step 和完整 1,550 控制步复核。滚翻落地 120/128、前进入口 120/128、完整路线 119/128，头部最大穿透 3.317 mm、关节越限为 0，官方路线门槛通过。官方权重仍为后续真机准备基线，S17 没有学习更新或模型导出。 [S17 评分](../../records/reports/official-route-independent-s1235/analysis.json) · [S17 关键帧](../../records/reports/official-route-independent-s1235/keyframes.jpg)

S18 汇总了四组参数端点、S16 三臂名义横向扰动和 S17 官方路线的固定策略证据。四组参数连续成功为 122/128、125/128、124/128、125/128；三臂扰动和 S17 路线均过原门槛。这些条件各自通过当时门槛，但参数端点的落地过渡与 S17 不同，不能合并证明统一控制器的参数鲁棒性；随机化训练收益与可迁移扰动触发器仍未准入。 [控制器复核](../../records/reports/roll-domain-parameter-random-s20/offline-review.md) [S18 汇总](../../records/reports/robustness-evidence-s18.md) · [S18 机器记录](../../records/reports/robustness-evidence-s18.json)

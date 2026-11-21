# 当前目标与验收条件

**当前目标是整理交付并准备真机。** 训练与新增动态仿真暂停。阶段报告、配置、日志、原片、官方权重与自训保存点通过现有GitHub仓库`main`交付。实物为已装配的XGO-Duck官方套机，零点标定待做，首轮现场验收按通信、舵机ID、机械中位、IMU、原厂站姿、网页走停和飞智黑武士4Pro依次进行。之后才进入同条件模型对照。完整仿真路线保留。[阶段归档与复现入口](../records/handoff/2026-11-21/README.md) · [首次部署和遥控手册](runbooks/xgo_duck_first_deployment_and_vader4pro.md)

## 仿真暂停时的状态

**0.8低摩擦连续路线保留为仿真恢复项。** 四路同释放4ms采集已验证，场景439和779都具备早段双足承重，候选在头部卸载确认后分别64ms、76ms重新承重。承重前约40ms开始的两种局部扭矩调整已完成短回放，最大转正增益为0.30°，门槛3°。后续沿用15°与25步持稳、七指令逐项122/128和总体852/896。开发种子完成全路线后，使用尚未调用的2246独立确认。[4ms接触、关节扭矩与短回放](../records/reports/nominal-skill-composition/low-friction-forward-support/README.md)

四路足底与质心复算、释放后360ms和420ms的有界扭矩回放均已归档。420ms调整的最大转正增益为1.99°。同批完整轨迹的400–640ms联合信号在第一遍4ms参照中得到13/15对正向排列，父策略为10/13，各自准入要求为14/15和12/13。该信号保留为诊断证据，实时采样和训练待重新授权及协议审核。[首次承重前复算与完整轨迹配对](../records/reports/nominal-skill-composition/low-friction-forward-support/pre-support-geometry/README.md)

400–480ms固定状态筛查已复核19种调整。左髋横滚候选在26个同初态短回放中最多让目标场景的右脚承重提前4ms，原准入样本11/13满足保护判据，该候选已结案。右脚首次持续承重后80ms的双足承重转正量，通过13对归档场景、本机CUDA计分预检及实际求解载荷复算。[舵机扭矩、短回放与承重转正指标](../records/reports/nominal-skill-composition/low-friction-early-transfer-authority/README.md)

实时采样三档均已完成，3,072场景的保守采样吞吐为24,645条/秒。两组各完成17次更新、1,253,376条转移。种子1257同释放的父策略、原奖励组和候选准入为880、882和881/896，前进分组为125、126和125/128，左转为122、123和123/128。候选距前进净增门槛分别差2和3例，权重与保存点已进入实验归档。[采样、训练与正式评价](../records/reports/nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/README.md)

**供电和舵机时延对照已完成。** 在摩擦系数1.0、独立种子2258中，7.4V组和8ms组分别通过881/896和891/896条完整路线。标称参照的右转入口为121/128，结束于第1125步。保留两个因素组的逐项评分与原片；低摩擦优先定位支撑和逐指令入口。[综合评分、曲线与录像](../records/reports/nominal-skill-composition/closed-loop-voltage-delay/README.md)

**标称连续路线的滚翻横向扰动已通过。** 独立种子2257从第608步分支，885个场景符合扰动条件。无扰动、左向0.08m/s和右向0.08m/s的完整路线分别通过874、867和873/896，三组各跑满42.5秒。左向组右转122/128，达到单项门槛。[三组评分与42.5秒原片](../records/reports/nominal-skill-composition/closed-loop-roll-lateral-disturbance/README.md)

**低摩擦首次承重的髋膝扭矩已复算。** 固定姿态下，13对失分样本在至少一个时点通过转正加速度检验，两个时点均满足成功保护判据的配对为8/13，门槛12/13。保留方向和位移读数，关闭当前髋膝方案的动态准入。[髋膝扭矩响应](../records/reports/nominal-skill-composition/low-friction-hip-knee-authority/README.md) · [首次承重对齐复算](../records/reports/nominal-skill-composition/low-friction-first-support-effort/README.md)

**低摩擦起身步长对照已复核。** 两遍4ms参照和首1.2秒2ms候选的滚翻完成及落地持稳依次为855、859和859/896。候选按请求仰卧的起身结果达到213/224，前进入口达到122/128，左转入口为119/128，距原门槛差3例。三组在22.50秒的七指令入口结束。[评分、曲线与原片](../records/reports/nominal-skill-composition/low-friction-release-integrator/README.md) · [七批失分复算](../records/reports/nominal-skill-composition/low-friction-failure-recurrence/README.md)

**低摩擦起身承重交接奖励已完成配对训练和动作对照。** 两组采用3,072个并行场景，各完成17次更新和1,253,376条场景转移。种子1257同释放快照的父策略、原奖励和候选滚翻前准入依次为885、878和880/896。三组前进分组均为126/128，候选进入实验归档。候选12个仰卧未准入场景在末25步均持续头部与双足接地。[容量实测、模型、训练过程与动作结果](../records/reports/nominal-skill-composition/low-friction-support-effort/support-credit-ppo/README.md)

**低摩擦起身载荷与奖励预检。** 种子1257固定策略的起身评分和滚翻前准入均为880/896。13个实际仰卧未准入场景在第7–8秒持续三点承重，头部载荷占标称机重的场景中位数为20.4%，固件占空比最高为限值的42.9%。新增承重交接奖励在三批归档上分别有6/6、11/14及13/13对由原准入场景得分更高。基于保存姿态重算的五组活跃接触中，GPU与CPU法向合力的最大绝对差为1.91×10⁻⁶N。[实测、候选预检与原片](../records/reports/nominal-skill-composition/low-friction-support-effort/README.md)

低摩擦起身恢复两轮对照已完成。29个归档姿态中，25个可在只调腿部时达到12°倾角并双足支撑。第550步交还队列三组滚翻前准入均为881/896。第450步交还队列的两遍参照为878/896，候选879/896，净增1个左平移场景。三组原起身评分均为878/896，左转准入均为123/128。两个候选的全部31个实际调整场景在各自交还前末25步持续头部接地。下一阶段核验原第8秒起身期限前的足底与头部荷载转移，再冻结控制或学习改动。保留低摩擦独立种子2246及原动作门槛。[目标可达性](../records/reports/nominal-skill-composition/low-friction-upright-target/README.md) · [第550步交还对照](../records/reports/nominal-skill-composition/low-friction-upright-recovery/README.md) · [第450步交还对照](../records/reports/nominal-skill-composition/low-friction-upright-early-return/README.md)

**仰卧起身奖励训练已完成。** 候选与原奖励组各17次更新及1,253,376条训练转移，训练曲线、权重、可恢复保存点和原片已重新核验。标称同快照前进分组的滚翻前准入由124/128至125/128，低摩擦对照两组均为125/128，左转由124/128至122/128。候选留作实验权重。[候选训练与两种工况的对照](../records/reports/nominal-skill-composition/early-getup-load/support-progress/low-friction-action-comparison/README.md)

低摩擦闭环步长对照已完成。种子1257的共同入口为881/896，两遍2ms参照及首1.20秒使用1ms的候选完成滚翻和落地853、862和870/896。候选前进分组为124/128，左转为120/128，左转的6个滚翻前失分均为实际仰卧释放且持续前倾接地，另外2个在滚翻中有效支撑转角不足。用带噪声的控制输入复算，保存完整观测的三个开发轨迹共识别42/42个滚翻入口失分，原通过的2,646个场景未触发。下一阶段先核验恢复动作的关节可达性和足底、头部接触，再做有界同快照对照。种子2246留作独立确认。[闭环评分与原片](../records/reports/nominal-skill-composition/closed-loop-low-friction-roll-window/README.md) · [控制观测复算](../records/reports/nominal-skill-composition/closed-loop-low-friction-roll-window/observed-tilt-readback.json)

**2ms连续路线已通过独立标称确认。** 开发种子1257的两遍完整动作各873/896。保留种子2256的滚翻及后续完整动作通过872/896，四种实际释放姿态和七项指令均逐项过线。已归档七个实际策略、五份可恢复保存点、冻结配置及原片。[独立确认与证据](../records/reports/nominal-skill-composition/closed-loop-timestep-confirmation/README.md) · [标称仿真交付清单](../artifacts/delivery/nominal-continuous-simulation.json)

低摩擦开发评估和准入复算已完成。种子1257仅将两脚滑动摩擦降至0.8，滚翻并落地859/896。前进和左转入口为120/128和121/128，均低于单项122/128。前进8个失分中5个有效支撑转角不足，左转7个失分中6个在滚翻前。下一阶段优先用这些轨迹核对低摩擦的滚翻前站稳与滚翻触地，按相同动作判据选择后续对照。种子2246保留，扰动阶段待低摩擦开发结论后推进。[零步核验](../records/reports/nominal-skill-composition/closed-loop-timestep-low-friction-preflight/README.md) · [低摩擦动作评分与原片](../records/reports/nominal-skill-composition/closed-loop-timestep-low-friction-development/README.md)

低摩擦逐帧复算中，9个滚翻前失分场景在第576–600步只超出倾角上限，末步为32.04°至34.24°。6个滚翻中失分场景的20ms采样漏计转角为34.55°至43.43°，发生在滚翻最初1.20秒。以两遍同种子标称轨迹提供重复波动参照，并按原动作评分限制后续对照。[站稳和接触复算](../records/reports/nominal-skill-composition/closed-loop-timestep-low-friction-development/standing-and-roll-mechanisms.json)

仰卧起身站稳带奖励训练、同释放快照对照和旧2ms滚翻失分对账均已完成。父策略、原奖励组和候选的滚翻前准入为877/896、882/896和884/896。前进分组分别为122/128、124/128和125/128，候选相对原奖励净增1个，晋级门槛为2个。候选进入实验归档，父策略用于标称重复评估及独立确认。[训练、动作结果及关键帧](../records/reports/nominal-skill-composition/early-getup-load/support-progress/README.md) · [旧2ms路线对账](../records/reports/nominal-skill-composition/early-getup-load/support-progress/action-comparison/timestep-forward-readiness.json)

仰卧起身4ms足底载荷已补采并复算。两批实际仰卧进入滚翻者持续双足承重的中位数均为596ms，未准入者分别为696ms和656ms。按相近释放状态配对后，未准入者在承重确认前80ms的倾角分别高5.42°和13.87°，确认后80ms的差异扩大到16.98°和21.31°。下一项在原仰卧策略及保存点上预检交接窗口内的持续转正训练信号，再确定等样本训练对照。最终留出种子2256和2246继续保留。[两批载荷、配对结果与原片](../records/reports/nominal-skill-composition/early-getup-load/README.md) · [训练信号预检草案](../configs/experiments/nominal_supine_support_progress_preflight.json)

仰卧起身足部实际响应对照已完成。同状态的两条原策略和候选均为880/896通过第600步滚翻前站稳判据，前进分组各122/128。57个场景实际改变动作目标，仅1个场景在共同离地窗口内前移比参照重复差多出至少1mm。三份归档中513、535和533个实际仰卧释放在第108步或更早均曾读到足地接触。滚翻段2ms积分保留作后续连续评估，滚翻、转场和七指令沿用冻结判据。[归档复算、足部响应与三段原片](../records/reports/nominal-skill-composition/supine-pre-bilateral-support/README.md)

仰卧起身早段策略接手已复核。两组滚翻准入分别为885/896和884/896，前进分组分别为126/128和125/128。18个实际接手场景中，原策略10个准入，接手组8个准入。原15°、连续25步及后续七指令门槛继续执行。[同状态短测与早段支撑](../records/reports/nominal-skill-composition/supine-early-support/README.md)

仰卧起身末段三组短测均有881/896进入滚翻，12个触发场景的滚翻准入均为0/12。第400步共用快照中，11/12个触发场景的重心已位于足地接触区域前缘之外。联合候选有10/12先失去双足接触，再开始腿部修正。[起身末段对照](../records/reports/nominal-skill-composition/supine-terminal-stance/README.md)

种子1257的同状态闭环步长对照已完成。两条4ms参照与滚翻段2ms候选分别完成855/896、851/896与872/896次滚翻并落地持稳。候选前进准入120/128，原门槛122/128。5个未进入滚翻的仰卧场景在第600步倾角为32.11°–33.54°，末25步均超过15°。另有3个场景有效支撑转角不足，种子2256和2246保留作最终确认。[评分、录像和VS Code观察](../records/reports/nominal-skill-composition/closed-loop-timestep/README.md) · [起身末段姿态读回](../records/reports/nominal-skill-composition/closed-loop-timestep/forward-pre-roll-admission.json)

仰卧起身后滚翻已完成4ms载荷对齐与数值短回放。对称腿部动作估算下压量最高为头部间隙中位数的3.76%，头部姿态补偿方案为2.24%。在4ms回放能复现原支撑顺序的仰卧失分事件中，改用2ms积分后，同一中间时刻卸载数由34/34变为26/34，12ms内卸载时间的配对变化中位数为0ms。CUDA的16场景预先验证两次2ms求解共用同一份4ms电机输入。[支撑取样、短回放与CUDA核验](../records/reports/nominal-skill-composition/support-sampling-audit/README.md)

更新于2026-11-21。起身后滚翻启动状态课程完成了两组各4,608个并行环境、各30次更新的训练。种子1257同状态评估中，官方模型、原始站立重置训练组和课程训练组滚翻并落地分别为853/896、848/896和847/896。候选入口合计847/896，原门槛852/896。两份自训权重进入实验归档。下一项以官方权重为参照，核对仰卧起身后头足支撑交接。候选已进入滚翻的32个失分场景均为有效支撑转角不足，其中实际仰卧19个。种子2256和2246留作最终确认。[训练曲线、分姿态评分与原片](../records/reports/nominal-skill-composition/post-getup-roll-curriculum/README.md)

仰卧滚翻后0.54–0.92秒的20ms边界复算已完成。候选相对官方的无接触正向转角配对差，在新增通过的15个场景中为−22.59°，在转为失分的19个场景中为+25.46°，均为场景中位数。下一项将用已有4ms载荷证据与本轮20ms结果核对支撑交接过程，再选定最小动作调整及原门槛复评。[匹配场景的支撑窗口](../records/reports/nominal-skill-composition/post-getup-roll-curriculum/action-comparison/supine-support-window.json)

此前颈部下调对照的两次原策略为855/896和859/896，候选为856/896。候选的前进、左转和右转入口分别为121、120和121/128，每项门槛122/128。该对照提供了官方滚翻在相同物理起点的重复波动范围。[颈部对照、原片与启动状态溯源](../records/reports/nominal-skill-composition/neck-descent-pair/README.md)

此前低摩擦滚翻的相位颈部候选在两个开发种子分别为108/128和115/128，均低于116/128。头部到足部接棒的1,536条归档轨迹已复核。[相位对照、完整原片及曲线](../records/reports/roll-forward-input-integration/endpoint-progress-training/observed-neck-support/current-evaluation.md) · [接棒复算](../records/reports/roll-forward-input-integration/endpoint-progress-training/head-to-foot-handoff.json)

4,608环境曾在本机容量对照中取得4.94%的保守净吞吐增益。滚翻配对两组各完成50次更新、5,529,600条环境转移及ONNX核验。12轮同场景评估已复核，其中当时的低摩擦加权训练候选两个种子均达到滚翻完成门槛，随后前进均触发偏航门槛。[容量对照](../records/reports/roll-friction-ppo-capacity-live/README.md) · [正式配对](../records/reports/roll-friction-paired-training/README.md) · [动作结果与偏航复算](../records/reports/roll-friction-paired-development/README.md)

站立接前进四组已复核。种子1247、1248的标称直接前进均通过127/128，低摩擦直接前进均为0/128，256个场景均仅因偏航失分。种子1247的关节修正上限从0.030 rad提高到0.035 rad后仍为0/128。仅替换为归档的行走教师约束权重后仍为0/128，20秒偏航组内中位数为0.22656 rad/s，原上限为0.20 rad/s。[直接前进与滚翻后对照](../records/reports/roll-friction-stand-forward/README.md) · [关节修正上限检查](../records/reports/forward-correction-limit-probe/README.md) · [教师约束权重评分](../records/reports/forward-teacher-actor-probe/README.md)

低摩擦前进偏航奖励配对使用同一官方行走父点、训练种子、场景、样本数及动作约束。候选增加前进偏航代价。奖励先用保存的运动轨迹定标，再核对CUDA零步初态及训练规模。两组以相同20秒仿真、原偏航和速度门槛、关节与接触检查和通过数对照。VS Code可查看进度、曲线、原片及逐轮记录。独立留出用于最终确认。

前进偏航代价已用512,000条保存样本定标。4,096环境初始化峰值15,760 MiB，触发85%显存上限。3,072环境两组的初态逐项一致，四轮短测保守净吞吐增益为14.39%。正式配对各完成24次更新、1,769,472条转移，ONNX、保存点和四段训练原片均已复核。种子1247的四组动作评价均完整，摩擦0.8与1.0的自训两组均为0/128，唯一失分项为偏航。候选在两种摩擦下分别比同样本对照减小0.02342和0.02358 rad/s。[训练过程与完整评分](../records/reports/forward-low-friction-yaw-reward/README.md) · [容量选择](../records/reports/forward-low-friction-yaw-reward/capacity/selection.json)

补充对照已完成。候选前进从官方站立起步后仍为0/128，20秒偏航组内中位数由0.29516变为0.29477 rad/s。[官方站立对照与原片](../records/reports/forward-official-stance-probe/README.md)

前进探索对照已复核。种子1247、脚底摩擦0.8。官方站立1秒后，候选的确定性推理和高斯探索组各完成128次20秒前进，均为0/128，唯一失分项为偏航。两组20秒平均绝对偏航速度的场景中位数分别为0.29468和0.28916rad/s，原上限为0.20rad/s。第8次更新的actor与官方行走父点相同，其原低摩擦评估为0/128。[配对复算与原片](../records/reports/forward-action-sampling/README.md)

观测输入对照已完成。20秒确定性轨迹按32个时点和128个场景取得4,096组输入。固定归档偏航反馈时，训练观测噪声引起候选关节目标RMS变化的样本中位数为0.01307rad，同一批干净输入上候选与官方权重的关节目标RMS差中位数为0.01353rad。[归档环境核对](../records/reports/forward-action-sampling/environment-contract.json) · [固定状态读回](../records/reports/forward-action-sampling/observation-noise-readback.json)

闭环观测噪声对照已复核。种子1247、摩擦0.8，从第51个控制步起向策略和反馈共用的观测加入训练噪声，动作仍取确定性均值。新增组完整采样1,050步，128/128进入前进，原门槛通过0/128。确定性参照和加噪组的20秒平均绝对偏航速度场景中位数分别为0.29468和0.28219rad/s。复算核验128,000组实际加噪输入、动作历史、电机目标及21.04秒原片。[闭环对照与原片](../records/reports/forward-observation-noise-trial/README.md)

早期筛查已用11批、1,408个归档场景校准。前8秒偏航筛查和20秒完整验收的批次判定一致，均保留2批标称官方参照并标记其余9批；个体场景存在2处判定差。下一轮以官方仿真权重为共同起点，准备同预算候选和对照，按保存点检查确定性前进。用8秒筛查标记明显退化，再按原20秒动作、速度、姿态和接触门槛评估标称及低摩擦场景。[早期窗口复算](../records/reports/forward-observation-noise-trial/early-screen-calibration.json)

前进输出层配对共用官方actor与同一价值网络、3,072并行环境、种子1250、物理偏航代价和初始0.05rad固定探索幅度。前8次只更新价值网络，随后4次同时更新策略与价值网络。对照更新全部策略层，候选仅更新最后一个线性输出层，两组各最多884,736条训练转移。先验证策略冻结、实际CUDA零步及VS Code观察入口，再运行两组。在首个策略保存点评价标称1.0和低摩擦0.8下的确定性前进，固定官方站立起步。记录首8秒偏航筛查及原20秒联合评分，独立留出保留给最终确认。

两组训练各完成12次更新及884,736条转移。实际CUDA零步的根关节状态、策略和价值观测、目标及随机状态逐项一致。保存点复算确认候选只变更最后一层，两个ONNX已通过独立动作读回。两组第12次保存点的原20秒评分已复核。标称下官方、全层组和输出层组分别通过127、46和12/128，低摩擦下均为0/128，四组自训评分的唯一失分项为偏航。两份训练权重用于复现实验。[训练记录、独立评分与原片](../records/reports/forward-output-layer-pair/README.md)

下一项对照从官方策略出发。512个归档场景均具备左右脚单独接触期间的髋偏航±0.01rad目标余量。先核对同一物理、执行器延迟和观测状态，再分别施加零脉冲、正脉冲和负脉冲，比较后续物理偏航速度；三组沿用原关节区间及0.03rad整体修正上限。[脉冲范围草案](../configs/experiments/forward_hip_yaw_pulse_draft.json) · [实际目标余量](../records/reports/forward-output-layer-pair/hip-yaw-pulse-bounds.json)

四个种子与摩擦组合各使用128场景和5秒共同路线。左右侧各在首个至少64场景同时满足单脚接触和正负指令余量的时刻分支，从同一仿真状态采零、正、负及第二次零脉冲，每条分支跟踪8步。以最初60毫秒的平均绝对偏航变化为主要指标，用双零差异核对数值重复性。预算各314个批量控制步，不进入学习和独立留出。

四组首个支撑脉冲已复核。两个低摩擦种子下，负脉冲在前60毫秒降低左侧平均绝对偏航0.00409至0.00423rad/s，右侧降低0.00063至0.00065rad/s。以上为各场景降幅的中位数。160毫秒窗口的降幅低于双零差第95百分位。旧识别器在整段归档轨迹上达到原门槛，最早左侧分支的识别覆盖为57%至67%。[首个支撑脉冲复算](../records/reports/forward-hip-yaw-pulse/paired-response.json) · [传感器选向复算](../records/reports/forward-hip-yaw-pulse/pulse-adaptive-sign-readback.json)

前进2秒后，识别器与实际接触一致的四组脉冲已复核。低摩擦左脚在正脉冲下的前60毫秒降幅为0.000092至0.000098rad/s，右脚在负脉冲下为0.000760至0.000891rad/s。四组左右侧在160毫秒窗口的降幅均低于各自的双零差第95百分位。两项对照合计321,536条物理转移，使用官方权重，保存8段共同路线原片。[可识别支撑对照](../records/reports/forward-observed-hip-yaw-pulse/README.md)

前进共同支撑关节对照已完成四组，共177,152条物理转移。两个低摩擦种子的左脚负脉冲、右脚正脉冲均达到髋俯仰的160毫秒筛选门槛。全部脉冲场景在传感器开始识别相应单脚支撑后的前3步内。[前进共同支撑关节对照](../records/reports/forward-support-joint-pulse/README.md)

髋俯仰规则已通过四条官方20秒轨迹的离线预检。闭环对照沿用种子1247、1248的标称与低摩擦场景和重置初态，各运行128个场景，站立1秒、前进20秒。前进满2秒后，按编码器与IMU识别、陀螺仪符号和关节余量触发20毫秒髋俯仰脉冲。保留原平均绝对偏航上限0.20rad/s、每组至少116/128场景通过及原速度、姿态与接触条件。每组保存全部物理与目标轨迹、独立评分及完整录像。[髋俯仰方案和离线预检](../records/reports/forward-phase-hip-pitch/README.md) · [运行记录](../records/reports/forward-phase-hip-pitch/current-evaluation.md)

闭环采样和独立复算已完成四组。两个标称组均通过128/128，两个低摩擦组均为0/128，唯一失分项是20秒偏航上限。右脚支撑中段归档预检在两个低摩擦种子中分别匹配118和124个场景，均超过64个门槛。下一项从同一物理、执行器和控制器状态比较追加单步与持续到第8步的右髋俯仰修正。[闭环评分](../records/reports/forward-phase-hip-pitch/current-evaluation.md) · [右脚支撑修正时长预检](../records/reports/right-support-hip-pitch-duration/README.md)

右脚支撑时长对照共80,384条物理转移，两个低摩擦种子均达到160毫秒筛选门槛。随后完成两组低摩擦和两组标称摩擦的20秒同快照闭环对照，共998,400条完整采样转移。低摩擦两条路线均为0/128，标称两组候选分别为128/128、126/128。当前连续路线仍使用已验收的仿真参考。下一阶段在同一快照加入重复参考轨迹，以左右各完整支撑周期的平均绝对偏航变化和双参考差值复核新的修正条件。整段验收沿用每场景0.20rad/s上限、原速度和姿态接触条件，以及每组至少116/128通过。[时长对照](../records/reports/right-support-hip-pitch-duration/README.md) · [整段对照评分、原片与观察记录](../records/reports/forward-right-support-continuation/README.md)

完整步态周期对照共256,000条环境转移。0.130m/s和0.115m/s模型输入均通过两个低摩擦种子的3秒筛选，0.115m/s按配对降幅入选。完整20秒同快照评估四组共947,200条环境转移，候选四组各128/128通过原门槛。两个低摩擦组的参照均为0/128，两个标称组的参照均为128/128。外部前进指令为0.15m/s，候选各组实际20秒速度中位数约0.134m/s。[完整步态周期对照](../records/reports/forward-stride-yaw-response/README.md) · [完整20秒独立评分与原片](../records/reports/forward-input-full-route/README.md)

滚翻→站立→前进的连续路线已冻结。逐场景前进计时已与四组20秒候选的实际ONNX输入逐项对齐，自训滚翻模型和对应保存点已核对。两条路线恢复第300控制步的同一CUDA状态及滚翻落地锁存，沿用第500步滚翻至少116/128与每组完整路线至少116/128的原门槛。独立留出种子2246按原方案保留。[连续路线运行记录](../records/reports/roll-forward-input-integration/README.md)

首组种子1247、脚底摩擦0.8的两条路线均完成120/128次滚翻。完整路线从0/128提高到120/128，按场景20秒平均绝对偏航速度取中位数为0.230056→0.137129rad/s。第二组种子1248在第500步达到112/128次滚翻，低于116/128的门槛，顺序评估结束。16个失分场景均达到至少330°几何转角，有效支撑转角为312.58°至329.46°。下一步依据既存轨迹定位头部离地与脚部接地之间的支撑损失，再为滚翻建立针对性对照。独立留出种子2246保留。[首组复算](../records/reports/roll-forward-input-integration/s1247-friction-low-low-friction/trajectory-readback.json) · [第二组逐场景复算](../records/reports/roll-forward-input-integration/remaining/s1248-friction-low-low-friction/roll-prefix-readback.json)

同种子4毫秒载荷核验已完成。原滚翻评分111/128。失分场景中，83个评分点无接触的正向转动区间各有4或5个子步承重，汇总正向子步转角的90.86%发生在承重时。保存姿态上，颈部俯仰目标增加0.02rad在68个头部承重区间中的54个同时出现间隙缩小和预期方向力矩变化。下一阶段先核对小幅颈部俯仰干预的目标范围、首步恢复和停止条件，再与两条同状态参照比较原500步滚翻评分。[逐场景转角](../records/reports/roll-forward-input-integration/support-window-force/contact-sample-timing.json) · [关节与力矩核验](../records/reports/roll-forward-input-integration/support-window-force/head-contact-authority.json) · [后续对照方案](../configs/experiments/roll_forward_input_integration/neck_support_probe_draft.json)

支撑交接对照已按原门槛结束。种子1247的两条参照均为118/128，候选114/128。种子1248的参照均为110/128，候选108/128。候选在两组均没有达到原116/128和相对双参照净增4个的要求，累计348,160条环境转移及六段原片已复核。[支撑交接对照](../records/reports/roll-forward-input-integration/neck-support-pair/README.md)

同次低摩擦训练第30次更新的保存点已完成CPU导出，以及与第50次保存点在768条保存滚翻观测上的本机CUDA输入和关节输出核验。后续从第49控制步同状态恢复，切换两个训练保存点的滚翻策略，沿用两次参照和原滚翻门槛。独立留出种子2246保留。[保存点导出](../records/reports/roll-friction-paired-training/low-friction/checkpoint-comparison/assessment.json) · [CUDA输出核验](../records/reports/roll-friction-paired-training/low-friction/checkpoint-comparison/cuda-policy-readback.json) · [配对方案](../configs/experiments/roll_forward_input_integration/roll_checkpoint_selection_draft.json)

低摩擦保存点对照已完成358,912条仿真转移。第30次更新保存点在种子1247通过121/128，两次第50次保存点参照为116/128和113/128。种子1248候选为111/128，两次参照为117/128和111/128。候选只通过一组开发门槛，两份权重继续用于复现。[逐组评分、VS Code观察记录和六段原片](../records/reports/roll-forward-input-integration/checkpoint-selection/README.md)

下一阶段只比较滚翻进度奖励的接触取样时点。固定轨迹复算中，改用末端接触后，17个滚翻未通过场景的该项奖励全部下降，场景配对变化中位数为−0.382；111个通过场景的配对变化中位数为+0.117。候选和对照共用官方策略初始化、低摩擦抽样和训练预算。先用CUDA测量奖励差值、场景重置和净吞吐，再按原门槛评估两份模型。独立留出种子2246继续保留。[奖励复算](../records/reports/roll-forward-input-integration/support-window-force/endpoint-progress-payment-replay.json) · [下一组训练方案](../configs/experiments/roll_forward_input_integration/endpoint_progress_training_draft.json)

CUDA固定策略核验完成46,080条转移，128次自动重置全部通过。候选正式训练沿用已实测的4,608环境，并使用已完成的同种子、同行为接口、同采样预算的归档原奖励组作对照。正式训练的逐次指标、曲线和完整录像写入VS Code可打开的工作区文件。[阶段训练与评估记录](../records/reports/roll-forward-input-integration/endpoint-progress-training/README.md)

候选已完成50次更新和5,529,600条训练转移，保存点、最终ONNX和两段7秒原片已复核。共同站立快照下，种子1247的两次原奖励自训参照为119/128和116/128，候选为111/128。种子1248的两次参照为110/128和114/128，候选为112/128。候选两组均低于116/128，归档供复现。[训练、对照评分和原片](../records/reports/roll-forward-input-integration/endpoint-progress-training/README.md)

当前复盘两组各12个场景的支撑时序。这些场景在两次原奖励参照中均通过、在末端接触候选中失分。按首次达到330°几何前翻角及原有效支撑峰值对齐，分别核对离地、侧倾扣减、反转和已施加的关节位置目标，逐场景复现原角度账目。全样本和两次原奖励参照用于对照波动。当前只读取保存的开发数据，独立留出种子2246继续保留。[场景编号和逐组测量](../records/reports/roll-forward-input-integration/endpoint-progress-training/action-evaluation/paired-roll-support-analysis.json) · [复盘口径](../configs/experiments/roll_forward_input_integration/endpoint-progress-evaluation/support-loss-localization.json)

768条保存轨迹已复算。两个种子各12个配对失分场景的主要端点离地缺口落在180°–270°，候选比两次参照多扣的角度中位数为16.79°至20.49°。24个候选场景中，21个最大离地段前后均是头部单独触地。下一项冻结同快照的4ms子步载荷与力矩实测，区分控制步端点采样和实际失去承重，并保留原动作评分。[支撑角度复算](../records/reports/roll-forward-input-integration/endpoint-progress-training/action-evaluation/support-loss-localization.json) · [分阶段差值图](../records/reports/roll-forward-input-integration/endpoint-progress-training/action-evaluation/support-loss-phases.png)

4ms载荷和力矩测量已结束。两个开发种子记录142,080条场景子步证据，候选沿用原判据取得115/128和109/128。24个新测配对失分场景的94个未计分间隔中，93个在末次子步求解测到承重，之后一次求解均未测到高于阈值的载荷。下一步先在这些时间边界检查头部俯仰和颈部关节目标的实际可达性，寻找能延长真实承重的控制改动，再按原滚翻、落地与连续路线门槛验证。独立留出种子2246继续保留。[逐子步复算](../records/reports/roll-forward-input-integration/endpoint-progress-training/head-recontact-load-recovery/physical-contact-clock-audit.json) · [修复与停止记录](../records/reports/roll-forward-input-integration/endpoint-progress-training/head-recontact-load-recovery-admission.json)

关节目标预检复算了1,045条卸载记录。实际力矩复算的最大绝对误差为1.89×10⁻⁷N·m。颈部俯仰目标上调0.02rad时，候选配对失分组分别有37/45和40/48条同时满足有效力矩变化、静态头部离地间隙减小和无接触约束下的法向加速度向下。该计算限于归档姿态，接触载荷变化由原滚翻判据下的动态对照测量。[冻结预检方案](../configs/experiments/roll_forward_input_integration/head-recontact-load-recovery/target-authority-review.json) · [逐关节复算](../records/reports/roll-forward-input-integration/endpoint-progress-training/head-recontact-load-recovery/target-authority-review.json)

基于观测相位的颈部对照已复核。候选与两次参照使用同一低摩擦自训模型及共同站立快照。归档两次参照的674条事件均能在卸载前一控制步收到至少0.01rad的目标变化。动态候选在两个种子的配对净增分别为−5／−9和+1／0，均未达到各净增至少4个的门槛。全体场景的180°–270°头部承重时长配对中位变化为0、0、−2、0ms。[冻结动态协议](../configs/experiments/roll_forward_input_integration/observed-neck-support/master.json) · [逐组评分和原片](../records/reports/roll-forward-input-integration/endpoint-progress-training/observed-neck-support/current-evaluation.md)

按实际4ms载荷复算接棒，两个方案各两个种子，全部128场景的头足交接间隔配对中位变化均为0ms。两次参照均通过而候选失分的场景中，间隔的配对中位变化介于−2和+4ms。现有证据未显示系统性足部接棒延迟。[冻结接棒口径](../configs/experiments/roll_forward_input_integration/head_to_foot_handoff_review.json) · [接棒逐场景复算](../records/reports/roll-forward-input-integration/endpoint-progress-training/head-to-foot-handoff.json)

标称连续路线已完成模型来源检查。七份ONNX均通过CPU读取和推理，输入61维、输出14维，关节顺序一致。切换时按目标策略的关节默认位姿和原始动作历史构造观测。来源协议数值为起身接站立过渡25步、官方滚翻路线过渡0步。组合控制器编写完成后，先核对实际CUDA初态和切换入口，再做同场景动态验证。[模型、门槛和切换参数](../records/reports/goal-review/nominal-skill-composition-sources.json)

标称路线拟采用896个场景，四类请求起身姿态分别为112、224、224和336个。每条路线在同一物理回合依次经过跌倒、起身、站立、官方前滚翻和落地站立，再执行一条预分配的指令，七条指令各128场景。保留起身接站立25步过渡和官方滚翻的0步过渡。滚翻落地后先按原条件站立，再用25步换回七指令的站立策略。末段每条指令保持原20秒及动作门槛。[连续动作方案](../configs/experiments/nominal_skill_composition_design.json)

组合接口通过CPU和实际CUDA零步。首次CUDA检查在0个物理步退出，随后隔离了方向预测器加载对CPU随机数状态的影响。实际初态每份模型用32条输入复算，七份ONNX的最大输出差为2.15×10⁻⁶rad。动态采集沿用恢复协议及原起身、滚翻和七指令门槛。[组合运行记录](../records/reports/nominal-skill-composition/README.md)

第1125步停止后，原轨迹复盘确认，31个未进入七指令的场景中有10个未进入滚翻，21个有效支撑转角不足。落地后换站立模型的865个场景均达到指令前的站立条件。下一项同快照试验保持每条指令128场景，单独对比滚翻前的站立模型。[丢失场景归因](../records/reports/nominal-skill-composition/zero-recovery/pre-command-handoff-analysis.json)

896场景的CUDA零步预检确认336项物理与随机数状态可精确恢复。第575步的实测快照包含346项状态，三条分支的物理和控制器读回均一致。候选右转组的5个场景达到了几何转动，未达到330°有效支撑转角。继续用原站立策略作参照，后续按原动作门槛确定是否消除了各组的滚翻失分。[CUDA预检](../records/reports/nominal-skill-composition/pre-roll-stand-snapshot-preflight.json) · [逐场景失分](../records/reports/nominal-skill-composition/pre-roll-stand-pair/handoff-review.json)

[候选组训练与同步组对照](../records/reports/getup-monitored-authorization-20261116/README.md) · [独立起身留出](../records/reports/getup-monitored-pair-holdout-s4268/README.md) · [滚翻两次标称对照](../records/reports/roll-matched-factor-report-capture-s1242/README.md) · [低摩擦组停止记录](../records/reports/roll-matched-factor-report-capture-s1242/friction-low/result.json) · [VS Code中的训练记录](../records/training-current.md) · [全流程报告](reports/development_report.html) · [目标历史](current_goal_history.md)

首组在采集前因观察确认路径不一致而停止：`records/evaluations` 是指向归档目录的联接，状态页显示了解析后的目录，而确认程序要求原始路径。没有物理步或学习更新；两种路径指向同一文件，确认检查已通过，新运行方案已单独冻结，旧记录保留。[停止记录](../records/reports/roll-matched-factor-dynamic-s1242/nominal-a/result.json) · [路径检查](../records/reports/roll-matched-factor-dynamic-s1242/view-confirmation-path-check.json) · [新运行入口](../records/reports/roll-matched-factor-observed-s1242/README.md)

新首组通过观察确认后，在零步初态指纹写入目录检查处停止：指纹函数要求报告目录，动态入口传入了评估目录。已保存的启动与执行器记录均显示零物理步；报告优先、哈希核对的适配通过离线检查，新对照已单独冻结。[停止与适配检查](../records/reports/roll-matched-factor-observed-s1242/README.md) · [新对照入口](../records/reports/roll-matched-factor-report-capture-s1242/README.md)

同种子滚翻对照在第500步的完成数分别为118/128、123/128和112/128。低摩擦组未达到116/128，按原规则结束，保存64,000条环境转移和10.04秒原片；完整路线成绩只报告两次标称。 [低摩擦对照](../records/reports/roll-matched-factor-low-continuation-s1242/README.md) · [标称对照](../records/reports/roll-matched-factor-report-capture-s1242/README.md)

起身共同跌落开发评估中，父模型、同步组和监控候选组分别为 1,003、998、1,006／1,024；候选通过开发门槛。独立留出中候选为 2,737／2,816，但请求仰卧 244／264，低于 251／264 门槛，因此完整留出未通过，不替换现有交付模型，也不利用留出挑参数。改善与退化案例、12 段留出原片和两份训练权重已归档。[开发记录与案例](../records/reports/getup-monitored-pair-action-s3267/README.md) · [留出结果](../records/reports/getup-monitored-pair-holdout-s4268/README.md)

3,072环境留出因观察器读取错误在1,275/1,450控制步停止。修复后使用新种子完成2,816环境独立留出，复算结果见上文。两批记录分别保留。 [当时的停止与修复检查](../records/reports/getup-monitored-pair-holdout-s4267/README.md)

## 总目标

在可复现的仿真环境中完成摔倒后站起、站立、前进、后退、左转、右转、左平移、右平移和前滚翻，并验证动作之间的连续切换与有界扰动。以官方模型为参照，保留有价值的自训模型和对应可恢复保存点；每个模型按实际动作、场景和门槛说明用途，不把多个组件说成一个已通过全部门槛的模型。

交付内容包括官方原始权重、自训 ONNX 与对应 `.pt`、可复现环境与命令、训练曲线和真实录像、每次修改与结果、独立评价、源码和模型哈希。仿真达标且具备现场条件后，再依次完成舵机顺序与方向、零位、IMU、通信、官方基线、自训策略迁移和回退准备。 [权重与迭代记录](../artifacts/delivery/iteration_record.md) · [真机标定手册](runbooks/hardware_commissioning.md)

## 最近进展

本机网页无法访问，现已在原 VS Code 窗口实际核验本地训练状态、曲线和录像；窗口大小和位置保持不变。网页访问不再是查看训练记录的前提。

最新监控开销对照两组均完成 17 次更新、各 1,253,376 条转移；训练过程总耗时 172.19→133.13 秒，吞吐提高 29.3%，GPU 利用率中位数 35%→45%。保存点、ONNX 和完整 8 秒原片均通过独立核验。两组间隔约三小时，性能结果为单次顺序对照，尚不能证明稳定收益；动作开发与留出结果见上文。[对照分析与关键帧](../records/reports/getup-monitored-authorization-20261116/README.md) · [本地训练记录](../records/training-current.md)

两组均已完成，每组采集 1,253,376 条转移。3,072 环境组的训练过程总耗时为 209.13 秒，1,536 环境组为 229.33 秒，包含初始化、导出和收尾；按该范围的吞吐约为 5,993 和 5,465 转移/秒，前者提高约 9.7%，多用约 5.16 GiB 峰值显存。从采样开始至最后一次指标发布的窗口另算，吞吐提高约 11.3%；两种计时范围分别保留。两组保存点、ONNX 和完整 8 秒录像均通过独立复核；GPU 利用率中位数仅 33% 和 31%，尚未达到充分利用算力的目标。[本轮方案、结果与修复](../records/reports/getup-observed-throughput-20261116/README.md)

首次运行因 Python 元组写入 JSON 后成为列表而误报观察数据不一致，在 24 控制步后关闭；只发生 120 次价值网络小批量更新，策略梯度更新为 0。独立入口只统一序列化表示，仍严格检查数值、进度和来源；旧失败、部分保存点和不完整片段保留。[复现检查](../records/reports/getup-observed-throughput-20261116/publication-verification.json)

此前的环境规模对照用于核验真实训练记录与吞吐，其对应模型没有另做动作改善的独立评估，不替换现有交付模型。此前官方标称连续路线为 123/128；多参数配对初态、扰动和最终技能组合仍需继续。[官方路线录像](../records/reports/duck_roll-native_nominal_s1241_20261115T230228/README.md) · [此前观察准备](current_goal_history.md)

固定路径对照现已完成：同父点、种子、3,072 环境，每组 17 次更新、1,253,376 条转移。训练过程总耗时从 209.16 秒降为 153.18 秒，对应吞吐提高 36.5%；两组各 2,075 次检查回调，累计耗时从 80.20 秒降为 32.50 秒。每次暂停判断、文件读取及哈希校验保留，118 份冻结来源未变。GPU 利用率中位数为 33% 和 33.5%，仍未充分利用本机算力。[方案、代码与结果](../records/reports/getup-canonical-path-20261116/README.md)

两组保存的初态一致，但第一控制步已出现约 10⁻⁷ 的数值差异，最终模型也不同；差异来源尚未确定。本轮只有一次顺序对照，尚未独立评价最终动作效果，两份模型作为诊断材料保留，不替换交付权重。暂停后仅重新读取模型、日志并完整解码两段 8 秒录像，没有新增学习或物理步。[暂停复核](../records/reports/pause-archive-20261116/saved-evidence-readback.json)

## 全部动作与阶段

| 阶段 | 已有仿真结果 | 剩余工作 |
| --- | --- | --- |
| 来源与流程 | 环境、官方参照、导出、过程观察和停止流程已有验证记录。 | 新方案需重新核对冻结来源、观察入口和运行条件。 |
| 七种行走指令 | 共同接口的最终留出中，站立、前进、后退、左右转和左右平移各 **32/32**；方案使用多个策略组件和反馈。 | 在共同场景继续验证连续切换、扰动和最终模型组合。 [行走清单](../artifacts/delivery/walking-simulation.json) |
| 四姿态起身 | 独立留出 **2,983/3,072**，实际右侧 **179/183**，右侧相对两组参照分别净增 **4、5** 例，原留出门槛通过。 | 总体相对参照分别少 4、5 例；保留参照与失败样本，继续验证完整技能组合。 [原评分与原片](../records/reports/getup-right-robustness/holdout/README.md) · [门槛核对](../records/reports/getup-gate-clarification/README.md) |
| 前滚翻 | 官方策略在独立开发和留出中均为 **117/128**，单动作门槛通过。 | 分别完成连续切换和扰动验收。 [滚翻评分](../records/reports/roll-endpoint-confirmation/README.md) |
| 连续动作与扰动 | 标称种子2256完整通过872/896。种子2257的滚翻横向扰动对照中，三组完整路线为874、867和873/896，七指令逐项过线。 | 完成低摩擦完整路线并验证其他供电、时延工况。 [标称确认](../records/reports/nominal-skill-composition/closed-loop-timestep-confirmation/README.md) · [滚翻横向扰动](../records/reports/nominal-skill-composition/closed-loop-roll-lateral-disturbance/README.md) · [低摩擦起身步长对照](../records/reports/nominal-skill-composition/low-friction-release-integrator/README.md) |
| 真机准备 | 已完成离线模型接口、软件关节映射、标定模板和回退材料。 | 实物 ID、机械零位、方向、IMU、通信、供电和使能尚未现场验证。 [准备审计](../records/hardware/deployment-preparation/s19-readiness.md) |

已有自训权重按各自用途保留。路线感知短训的候选在同种子对照中为 **28/128**，官方父模型为 **125/128**，因此停止该学习线；教师约束候选与官方父模型两种子合计均为 **243/256**，差异不稳定，候选未晋级。 [短训对照](../records/reports/route-aware-pilot-parent-compare-s1225/README.md) · [教师约束对照](../records/reports/route-aware-teacher-anchor-parent-compare-s1232/README.md) · [模型选择](../artifacts/delivery/simulation-model-selection.json)

## 后续顺序

起身候选的开发和留出对照已完成，现按以下顺序继续；不重启已经关闭的试验。新评估仍须事先核验模型接口、预算、本机观察和停止条件，训练记录不能替代独立评分。

1. **核验持续前倾时的支撑转移。** 已用陀螺仪和投影重力确认持续前倾，已检验两种腿部目标交还时点。下一项用有效电机输入和物理接触读数核对头部、双足与躯干的荷载转移，按原第8秒起身时限以及第600步滚翻准入完成同快照试验。按原门槛决定何时使用低摩擦留出种子。[目标跟踪复算](../records/reports/nominal-skill-composition/low-friction-upright-early-return/target-tracking.json) · [实际控制输入核验](../records/reports/nominal-skill-composition/closed-loop-low-friction-roll-window/observed-tilt-readback.json)
2. **逐步评估并停止无效运行。** 先进行名义与单因素有界对照，按原动作门槛、头部穿透与关节范围检查，并在每臂结束后检查能否进入下一臂。异常、无效数值、预算耗尽或门槛已不可达时保留证据并停止，不自动重启或追加预算。若需要完整失败诊断，事先单列条件和预算。
3. **继续定位性能与数值差异。** 本轮同步检查回调耗时 34.18→0.93 秒；候选监控线程另有并行开销。候选环境控制步计时 56.87 秒，计时范围存在嵌套，不能相加。首步物理输出已有微小差异，最终网络也有差异；不把单次训练内持稳当作动作改善。后续复测负载及不同种子，并优先定位控制步和同步等待。[分析与来源](../records/reports/getup-runtime-review-20261116/README.md) · [本轮结果](../records/reports/getup-monitored-authorization-20261116/README.md)
4. **再决定训练修改。** 从同一父保存点做等样本对照，记录改动原因、配置与影响。小规模学习对照每组最多 50 次更新；单次运行不超过 250 次，有证据支持的分段累计最多 1,000 次，具体范围以新冻结协议为准。原官方 ONNX 是推理参照，不等于包含优化器、价值网络和历史的 PPO 保存点。
5. **独立复验并交付。** 保持各动作原门槛，用未用于挑参的留出场景和多个训练种子核验，记录成功数、比例及失败原因；再验证共同场景切换和扰动。达标后交付官方参照、自训 ONNX 与对应 `.pt`，核对导出一致性、恢复能力和来源哈希。
6. **真机后置。** 达到仿真门槛并具备现场条件后，再按标定手册逐项核对。舵机只调整顺序时先验证软件映射；型号、几何或动力学变化需要重新评估。实际写入、使能和部署当前不进行。

## 本机训练与观察

已完成的同父点规模短测中，行走 1,024、2,048、4,096 环境分别约为 **9.86–10.05k、16.08–16.30k、23.98–24.58k 转移/秒**，4,096 档最高显存约 4,630 MiB、CPU 约 100%。这是已测范围内的结果，并不证明更大规模一定更快。观察器 D3D12 + 4 Hz 的两遍吞吐约 **18.35–18.38k 转移/秒**，实际状态采样约 2.25–2.29 Hz。 [规模和观察器记录](../records/reports/throughput-selection/README.md)

恢复后优先使用已核验的 WSL、本机 CUDA 和上述行走配方；起身使用已记录的独立配方，并在固定父点、样本量和动作门槛下比较。分开记录采样、更新、录像、CPU、显存和磁盘开销，以达到同一动作门槛的总时间选择方案。若 WSL 成为瓶颈，再与原生 Ubuntu 做同机同预算对照；两遍均提速、综合提升至少 15% 且动作指标稳定时再考虑迁移。

训练过程须在用户可打开的页面或工具中显示真实运行名、状态、进度和曲线，并按动作保存完整原片；VS Code 终端可选。默认不打开卡顿的持续实时窗口，约每三分钟在动作起点安排录像：起身训练约 8 秒、评估约 11 秒，滚翻和切换覆盖实际完整过程，异常或中断片段明确标记。不得为录像额外推进物理步。 [训练手册](runbooks/pc_training.md) · [录像规则](../configs/environments/action_recording.json)

## 文件与证据入口

- [全流程报告](reports/development_report.html)：环境部署、训练、修改原因、结果、截图、视频和真机准备；[完整过程记录](reports/development_history.html)保留逐次历史。
- [本次暂停进度归档](../records/reports/pause-archive-20261116-monitored/README.md)：当前结果、过程文件、保留模型、暂停核验及快照说明；[此前固定路径归档](../records/reports/pause-archive-20261116/README.md)保持原字节。
- [权重与迭代记录](../artifacts/delivery/iteration_record.md)：官方参照、自训组件及保存点；[模型选择](../artifacts/delivery/simulation-model-selection.json)注明哪些候选没有晋级。
- [训练记录归档](../records/training-archive/README.md)和[暂停进度归档](../records/reports/progress-archive-20261114/README.md)：按批定位方案、日志、轨迹、原片和快照。快照不是脱离工程目录的完整备份。
- [此前暂停进度归档](../records/reports/pause-archive-20261115/README.md)：当时的暂停状态、阶段结论、评估、模型入口和快照；权重、轨迹和原片仍按原路径保留。[更早快照](../records/reports/progress-archive-20261115/README.md)保留本机观察、零步失败、修复和独立核验。
- [观察与监督准备归档](../records/reports/pause-archive-20261115/supervision-preparation.zip)：暂停后整理的相关源码、记录和当前文档；大文件仍按原路径保留。[归档清单](../records/reports/pause-archive-20261115/supervision-preparation-manifest.json)
- [起身接七指令完整原片](../records/evaluations/duck_switch_routed-turn-gyro_holdout-zero-recovery_s2175_20261110T230520/restored-gyro/clip-step-0.mp4) · [左转关键帧](../records/reports/stage-media/cards/N628.jpg) · [全部录像总览](reports/assets/all-video-overview.jpg)。它们是已保存的评估画面，不是当前直播。
- [工程导航](project_structure.md) · [目标历史记录](current_goal_history.md) · [原文保全记录](../records/reports/goal-review/goal-organization-source.json)：历史内容仅供追溯，以本页和实际证据为准。

清理前先核对绝对路径、来源哈希、引用、保留价值和恢复依赖；保留官方归档、最终或已评价模型、恢复父点、代表性失败与审计输入。安装优先使用可用镜像源并记录过程，保持官方依赖锁定；代理断线时先检查现有进程与缓存。新运行必须先冻结对应方案并通过观察及运行准入，不自动重启中断评估或追加旧预算。

# 当前训练记录

**训练已暂停。** 在VS Code打开下面的历史曲线、日志和真实录像，可按运行编号回看。最新续接位置为种子1257低摩擦完整轨迹配对核验，实际训练与独立仿真留出保持封存。[阶段归档](handoff/2026-11-21/README.md) · [逐分支完整轨迹配对](reports/nominal-skill-composition/low-friction-forward-support/pre-support-geometry/README.md) · [真机首次部署手册](../docs/runbooks/xgo_duck_first_deployment_and_vader4pro.md)

**低摩擦前进起身的4ms载荷采集与短回放已完成。** 四组同释放共448,000条转移。两个前进场景均出现头部重新承重，近承重时点的两种扭矩调整已完成回放。[结果与四段原片](reports/nominal-skill-composition/low-friction-forward-support/README.md) · [最终状态](reports/nominal-skill-composition/low-friction-forward-support/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/low-friction-forward-support/final-view/native-curve.png) · [承重、倾角与颈部扭矩](reports/nominal-skill-composition/low-friction-forward-support/forward-support-timing.png) · [执行日志](reports/nominal-skill-composition/low-friction-forward-support/sample-command.log)

**低摩擦承重转正配对训练和动作对照已完成。** 两组各17次更新、1,253,376条转移。父策略、原奖励对照和候选的滚翻前准入为880、882和881/896，前进分组为125、126和125/128。打开历史状态、逐次曲线和原片均可回看。[全部结果与权重](reports/nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/README.md) · [候选逐次训练曲线](reports/nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/candidate/final-view/native-curve.png) · [动作对照历史状态](reports/nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/action-comparison/final-view/native-status.txt) · [动作对照曲线](reports/nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/action-comparison/final-view/native-curve.png) · [标注关键帧](reports/nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/action-comparison/support-credit-keyframes.jpg)

**前进起身时序已复算。** 同权重两遍参照的滚翻前准入各123/128，场景283和727交换判定。场景439与779在五组均未准入，随后按再接触和首次承重分别检查。[五组逐场景时序](reports/nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/action-comparison/forward-getup-readback.json)

**低摩擦承重转正指标通过计分复核。** 右脚首次持续承重后的80ms中，13对归档里有12对原准入场景的双足承重转正量较高，配对中位差4.76°。3,072场景短测达到24,645条转移/秒，两组正式训练实测净吞吐为13,673和13,611条/秒。[实际读数、对照图与原始结果](reports/nominal-skill-composition/low-friction-early-transfer-authority/README.md) · [三档本机实测](reports/nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/measured-capacity/selection.json)

**供电与舵机时延对照已完成。** 7.4V和8ms组完整通过881/896和891/896，原片各覆盖42.5秒。标称组滚翻并站稳872/896，右转入口121/128，按原门槛结束于22.5秒。[三组状态、曲线、执行日志及原片](reports/nominal-skill-composition/closed-loop-voltage-delay/README.md) · [7.4V组曲线](reports/nominal-skill-composition/closed-loop-voltage-delay/voltage-7p4v/final-view/native-curve.png) · [8ms组曲线](reports/nominal-skill-composition/closed-loop-voltage-delay/delay-8ms/final-view/native-curve.png) · [标注关键帧](reports/nominal-skill-composition/closed-loop-voltage-delay/keyframes.jpg)

**低摩擦双足接地时序已复算。** 新增准入场景的三组均在第139步确认双足接地，候选到第156步确认转正和头部离地。第136–156步的迈脚几何和时长复核中，左转失分场景满足执行要求的为0/4。[逐场景承重、姿态及关节时长](reports/nominal-skill-composition/low-friction-right-support-trial/support-timing/README.md)

**低摩擦右髋膝位置对照已完成。** 两遍参照及候选获得884、884和885/896滚翻前准入。左转分组均为124/128。21个触发场景中有20个实际达到两关节扭矩方向和幅度要求。[VS Code最终状态](reports/nominal-skill-composition/low-friction-right-support-trial/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/low-friction-right-support-trial/final-view/native-curve.png) · [运行中读回](reports/nominal-skill-composition/low-friction-right-support-trial/live-readback.json) · [逐项评分、三段12秒原片及标注帧](reports/nominal-skill-composition/low-friction-right-support-trial/README.md)

**独立标称滚翻扰动对照已完成。** 无扰动、左向扰动和右向扰动的完整路线分别通过874、867和873/896。三组都跑满42.5秒。[VS Code最终状态](reports/nominal-skill-composition/closed-loop-roll-lateral-disturbance/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/closed-loop-roll-lateral-disturbance/final-view/native-curve.png) · [最终实景](reports/nominal-skill-composition/closed-loop-roll-lateral-disturbance/final-view/native-latest.jpg) · [运行中读回](reports/nominal-skill-composition/closed-loop-roll-lateral-disturbance/live-readback.json) · [三组原片、关键帧与逐项评分](reports/nominal-skill-composition/closed-loop-roll-lateral-disturbance/README.md)

**低摩擦仰卧起身早段观测已复算。** 释放后720ms，固定俯仰窗口在四条实际策略观测归档分别识别15/15、15/15、12/14和9/9个实际仰卧未准入场景。后续核对右髋、右膝的位置目标与固件实际输出。[观测与覆盖复算](reports/nominal-skill-composition/low-friction-right-support-signal/README.md)

**首次双足承重前后的扭矩复算已完成。** 基于保存的4ms扭矩与载荷数据核对13对实际仰卧轨迹。7对重复失分场景在首次持续双足承重前160–80ms少转正14.24°。本轮复算采用归档数据。[关节分组、原始读数与复算日志](reports/nominal-skill-composition/low-friction-first-support-effort/README.md)

**低摩擦起身步长对照已完成。** 三组共2,844,800条固定权重仿真转移，滚翻完成及落地持稳依次为855、859和859/896。候选前进入口122/128，左转入口119/128，三组均结束于指令下发前。[VS Code最终状态](reports/nominal-skill-composition/low-friction-release-integrator/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/low-friction-release-integrator/final-view/native-curve.png) · [最终实景](reports/nominal-skill-composition/low-friction-release-integrator/final-view/native-latest.jpg) · [运行中读回](reports/nominal-skill-composition/low-friction-release-integrator/live-readback.json) · [逐项评分、关键帧和三段原片](reports/nominal-skill-composition/low-friction-release-integrator/README.md)

**低摩擦起身两组训练已完成。** 同父点两组各17次更新和1,253,376条场景转移。各组实际曲线、权重和完整8秒原片已核验，动作对照从共同释放开始。[原奖励训练曲线](reports/nominal-skill-composition/low-friction-support-effort/support-credit-ppo/control/final-view/native-curve.png) · [承重交接奖励训练曲线](reports/nominal-skill-composition/low-friction-support-effort/support-credit-ppo/candidate/final-view/native-curve.png) · [原奖励8秒原片](experiments/duck-getup-support-credit-friction080-control-s5417-20261121T154357/clip-step-0.mp4) · [承重交接奖励8秒原片](experiments/duck-getup-support-credit-friction080-candidate-s5417-20261121T154357/clip-step-0.mp4) · [容量实测和训练核验](reports/nominal-skill-composition/low-friction-support-effort/support-credit-ppo/README.md)

**共同释放动作对照已完成。** 父策略、原奖励对照及承重交接奖励候选的滚翻前准入依次为885、878和880/896。候选相对原奖励净增2个，较父策略少5个。[逐场景和分组评分](reports/nominal-skill-composition/low-friction-support-effort/support-credit-ppo/action-comparison/analysis.json) · [同快照关键帧](reports/nominal-skill-composition/low-friction-support-effort/support-credit-ppo/action-comparison/support-credit-keyframes.jpg) · [候选12秒评估原片](evaluations/duck-getup-support-credit-action-friction080-s1257-20261121T155133/support-credit-reward/clip-step-0.mp4) · [评估采样曲线](reports/nominal-skill-composition/low-friction-support-effort/support-credit-ppo/action-comparison/final-view/native-curve.png)

**低摩擦起身载荷核验已完成。** 896个场景的第600步准入为880/896。承重交接奖励通过归档预检，并完成GPU接触读回和单次计算耗时核验。下一阶段测量接入后的实际采样吞吐。本轮为固定策略采样。[本轮结果与奖励准入](reports/nominal-skill-composition/low-friction-support-effort/README.md) · [最终状态](reports/nominal-skill-composition/low-friction-support-effort/verified-entry/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/low-friction-support-effort/verified-entry/final-view/native-curve.png) · [运行中读回](reports/nominal-skill-composition/low-friction-support-effort/verified-entry/live-readback.json) · [12秒原片](evaluations/duck-getup-support-effort-friction080-s1257-20261121T145447/clip-step-0.mp4)

**起身目标提前交还对照已完成。** 两遍原策略及第450步交还候选的第600步准入分别为878、878和879/896，原起身评分同为878/896，左转准入同为123/128。[逐项评分、姿态关键帧和三段12秒原片](reports/nominal-skill-composition/low-friction-upright-early-return/README.md) · [最终状态](reports/nominal-skill-composition/low-friction-upright-early-return/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/low-friction-upright-early-return/final-view/native-curve.png) · [运行中读回](reports/nominal-skill-composition/low-friction-upright-early-return/live-readback.json)

**低摩擦起身恢复对照已完成。** 同一第250步快照的三组第600步准入均为881/896。候选实改15个场景，3个在第550步交还后才抬头。[评分、复算和三段12秒原片](reports/nominal-skill-composition/low-friction-upright-recovery/README.md) · [最终状态](reports/nominal-skill-composition/low-friction-upright-recovery/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/low-friction-upright-recovery/final-view/native-curve.png)

**低摩擦滚翻早段步长对照已完成。** 两遍2ms参照及首1.20秒使用1ms的候选分别有853、862和870/896个场景滚翻并落地持稳。候选左转入口120/128，原门槛122/128。[逐项评分、关键帧及三段22.50秒原片](reports/nominal-skill-composition/closed-loop-low-friction-roll-window/README.md) · [最终状态](reports/nominal-skill-composition/closed-loop-low-friction-roll-window/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/closed-loop-low-friction-roll-window/final-view/native-curve.png) · [运行中读回](reports/nominal-skill-composition/closed-loop-low-friction-roll-window/live-readback.json) · [CUDA采样命令和输出](reports/nominal-skill-composition/closed-loop-low-friction-roll-window/sample-command.log)

**低摩擦起身奖励对照已完成。** 仰卧父策略、原奖励组和站稳带奖励候选共用同一批896个释放快照，滚翻前准入为880、882和881/896。三组前进分组均为125/128，左转分组分别为123、124和122/128。两组各17次更新的训练档案已重新核验。[逐项评分和三段12秒原片](reports/nominal-skill-composition/early-getup-load/support-progress/low-friction-action-comparison/README.md) · [最终进度](reports/nominal-skill-composition/early-getup-load/support-progress/low-friction-action-comparison/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/early-getup-load/support-progress/low-friction-action-comparison/final-view/native-curve.png) · [仿真帧](reports/nominal-skill-composition/early-getup-load/support-progress/low-friction-action-comparison/final-view/native-latest.jpg) · [CUDA采样日志](reports/nominal-skill-composition/early-getup-load/support-progress/low-friction-action-comparison/sample-command.log)

**0.8足底摩擦连续开发评估已完成。** 滚翻并落地859/896。前进和左转入口为120/128和121/128，采样按原规则结束于第1125步。[最终状态](reports/nominal-skill-composition/closed-loop-timestep-low-friction-development/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/closed-loop-timestep-low-friction-development/final-view/native-curve.png) · [仿真帧](reports/nominal-skill-composition/closed-loop-timestep-low-friction-development/final-view/native-latest.jpg) · [22.50秒原片](evaluations/duck-roll-fixed-policy-2ms-friction-low-s1257-20261121T124136/lower-friction/clip-step-0.mp4) · [逐场景失分和配置](reports/nominal-skill-composition/closed-loop-timestep-low-friction-development/README.md)

**2ms连续路线独立确认已完成。** 种子2256的完整序列通过872/896，实际四种跌倒姿态和七项后续指令逐项达到原门槛。保存完整42.5秒原片。[最终状态](reports/nominal-skill-composition/closed-loop-timestep-confirmation/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/closed-loop-timestep-confirmation/final-view/native-curve.png) · [仿真帧](reports/nominal-skill-composition/closed-loop-timestep-confirmation/final-view/native-latest.jpg) · [原片](evaluations/duck-roll-fixed-policy-2ms-heldout-s2256-20261121T121926/heldout-nominal/clip-step-0.mp4) · [方案与逐项评分](reports/nominal-skill-composition/closed-loop-timestep-confirmation/README.md)

**2ms连续滚翻重复评估已完成。** 种子1257的两遍各873/896完整通过，七条指令逐项达到原门槛。两段42.5秒原片和运行过程可在VS Code回看。[最终状态](reports/nominal-skill-composition/closed-loop-timestep-repeat/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/closed-loop-timestep-repeat/final-view/native-curve.png) · [最终仿真帧](reports/nominal-skill-composition/closed-loop-timestep-repeat/final-view/native-latest.jpg) · [逐项评分、命令与原片](reports/nominal-skill-composition/closed-loop-timestep-repeat/README.md)

**仰卧起身站稳带奖励配对。** 两组各完成17次更新、1,253,376条训练转移。训练期间核验过本机观察入口，以下链接可在VS Code回看完整记录。

| 训练组 | 状态、真实曲线和仿真帧 | 命令输出与原片 |
| --- | --- | --- |
| 原奖励对照组 | [17次更新状态](reports/nominal-skill-composition/early-getup-load/support-progress/observed-training/control/final-view/native-status.txt) · [曲线](reports/nominal-skill-composition/early-getup-load/support-progress/observed-training/control/final-view/native-curve.png) · [仿真帧](reports/nominal-skill-composition/early-getup-load/support-progress/observed-training/control/final-view/native-latest.jpg) | [原奖励对照组命令日志](reports/nominal-skill-composition/early-getup-load/support-progress/observed-training/control-training-command.log) · [8秒原片](experiments/duck-getup-upright-band-control-observed-s5117-20261121T112216/clip-step-0.mp4) |
| 新增站稳带奖励候选组 | [17次更新状态](reports/nominal-skill-composition/early-getup-load/support-progress/observed-training/authorization-recovery/candidate/final-view/native-status.txt) · [曲线](reports/nominal-skill-composition/early-getup-load/support-progress/observed-training/authorization-recovery/candidate/final-view/native-curve.png) · [仿真帧](reports/nominal-skill-composition/early-getup-load/support-progress/observed-training/authorization-recovery/candidate/final-view/native-latest.jpg) | [候选组命令日志](reports/nominal-skill-composition/early-getup-load/support-progress/observed-training/authorization-recovery/candidate-training-command.log) · [8秒原片](experiments/duck-getup-upright-band-candidate-direct-s5117-20261121T112756/clip-step-0.mp4) |

[奖励源码、CUDA初态及训练核验](reports/nominal-skill-composition/early-getup-load/support-progress/README.md)

**同释放快照动作对照已完成。** 仰卧父策略、原奖励组和候选前进分组的滚翻前准入分别为122/128、124/128和125/128。候选按实验权重留档。[最终采样状态](reports/nominal-skill-composition/early-getup-load/support-progress/action-comparison/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/early-getup-load/support-progress/action-comparison/final-view/native-curve.png) · [标注关键帧](reports/nominal-skill-composition/early-getup-load/support-progress/action-comparison/comparison-keyframes.jpg) · [共同状态核验](evaluations/duck-getup-upright-band-action-comparison-s1257-20261121T113723/fork.json) · [评分与三组12秒原片](reports/nominal-skill-composition/early-getup-load/support-progress/README.md#同释放快照对照)

**仰卧起身早段足底载荷已采样并复算。** 两批分别为886/896与879/896通过滚翻前站稳。实际仰卧场景持续双足承重的中位数在两批准入组均为596ms，在两批未准入组分别为696和656ms。[种子1256最终状态](reports/nominal-skill-composition/early-getup-load/verified-entry/seed-1256/final-view/native-status.txt) · [种子1257最终状态](reports/nominal-skill-composition/early-getup-load/verified-entry/seed-1257/final-view/native-status.txt) · [载荷图、曲线和两段原片](reports/nominal-skill-composition/early-getup-load/README.md)

**仰卧起身足部实际响应对照已完成。** 两次原策略与未承重脚前移组均为880/896通过原滚翻前站稳判据。候选的57个实际修正场景中，有1个场景在共同离地窗口内的前移量比参照重复差多出至少1mm。[最终进度](reports/nominal-skill-composition/supine-pre-bilateral-support/foot-authority/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/supine-pre-bilateral-support/foot-authority/final-view/native-curve.png) · [最终实景](reports/nominal-skill-composition/supine-pre-bilateral-support/foot-authority/final-view/native-latest.jpg) · [三段原片、关键帧和评分](reports/nominal-skill-composition/supine-pre-bilateral-support/README.md)

**仰卧起身双足确认前的归档复算。** 第130步倾角规则在两份种子1257参照各提前识别三分之二的未准入场景。第122步的迈脚几何在原未准入场景的覆盖为0/7、0/12和0/9。[逐步复算、轨迹来源和命令日志](reports/nominal-skill-composition/supine-pre-bilateral-support/README.md)

**仰卧起身早段策略接手对照已完成。** 原策略和接手组的滚翻准入分别为885/896与884/896。18个实际接手场景分别为10/18和8/18通过准入。[最终进度](reports/nominal-skill-composition/supine-early-support/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/supine-early-support/final-view/native-curve.png) · [归档支撑曲线、评分和两段原片](reports/nominal-skill-composition/supine-early-support/README.md)

**仰卧起身末段姿态对照已完成。** 三组的滚翻前站立准入均为881/896。两组处理各解除12个触发场景的头脚接触，触发场景准入均为0/12。[最终状态](reports/nominal-skill-composition/supine-terminal-stance/final-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/supine-terminal-stance/final-view/native-curve.png) · [关键帧、原片和复算](reports/nominal-skill-composition/supine-terminal-stance/README.md)

**滚翻闭环积分步长对照已完成。** 4ms两条参照分别855/896、851/896，2ms候选滚翻并落地持稳872/896。候选前进准入120/128，原门槛122/128。[最终状态](reports/nominal-skill-composition/closed-loop-timestep/final-view/native-status.txt) · [全程采样曲线](reports/nominal-skill-composition/closed-loop-timestep/final-view/native-curve.png) · [最终画面](reports/nominal-skill-composition/closed-loop-timestep/final-view/native-latest.jpg) · [评分、关键帧和完整原片](reports/nominal-skill-composition/closed-loop-timestep/README.md)

**仰卧滚翻支撑诊断已完成。** 已核对五条归档的4ms载荷与20ms评分边界，复算关节小幅动作，完成4,464个CPU短回放积分步。16个CUDA场景核验两次2ms积分共用一份4ms电机输入，为闭环对照提供了时钟凭据。[支撑取样图、力矩复算及CUDA时钟](reports/nominal-skill-composition/support-sampling-audit/README.md)

**起身后滚翻启动状态配对训练和同场景对照已完成。** 每组4,608个环境、30次更新和3,317,760条环境转移。官方滚翻、训练对照和课程训练候选分别通过853/896、848/896和847/896，三组均在七指令入口结束。[训练方案、曲线及原片](reports/nominal-skill-composition/post-getup-roll-curriculum/README.md) · [候选训练记录](reports/nominal-skill-composition/post-getup-roll-curriculum/post-getup-start-curriculum/final-view/native-status.txt) · [动作评估复算](reports/nominal-skill-composition/post-getup-roll-curriculum/action-comparison/analysis.json)

**滚翻颈部目标下调对照已完成。** 两次原策略滚翻并落地855/896、859/896，候选856/896。候选的前进、左转和右转停在七指令入口。下一阶段以种子1256的886个起身后真实启动状态构建训练课程。[采样状态](reports/nominal-skill-composition/neck-descent-pair/release-retry/native-view/native-status.txt) · [采样曲线](reports/nominal-skill-composition/neck-descent-pair/release-retry/native-view/native-curve.png) · [实际画面](reports/nominal-skill-composition/neck-descent-pair/release-retry/native-view/native-latest.jpg) · [完整评估、三段原片与逐时刻标注](reports/nominal-skill-composition/neck-descent-pair/README.md)

**仰卧恢复后滚翻支撑交接已完成。** 种子1257补录860/896达到滚翻与落地站立，站立指令入口121/128。已核对4ms载荷、实际力矩和颈部目标下调候选的归档覆盖。[两组比较、载荷图和原片](reports/nominal-skill-composition/support-transfer/README.md) · [采样曲线](reports/nominal-skill-composition/support-transfer/native-view/native-curve.png) · [最近实景](reports/nominal-skill-composition/support-transfer/native-view/native-latest.jpg)

**种子1257独立连续动作评估已完成。** 四姿态起身均过线，滚翻与落地站立848/896，低于七指令总入口852/896，评估在22.50秒结束。[独立评分与原片](reports/nominal-skill-composition/independent-development/README.md) · [采样曲线](reports/nominal-skill-composition/independent-development/native-view/native-curve.png) · [最近实景](reports/nominal-skill-composition/independent-development/native-view/native-latest.jpg)

**标称滚翻接触与载荷核验已完成。** 原站立路线完整通过866/896，七条指令逐项过线。支撑转角不足的20个场景中，91个相关区间都在末子步承重、下一次求解卸载。[载荷结果及42.50秒原片](reports/nominal-skill-composition/contact-load/README.md) · [采样曲线](reports/nominal-skill-composition/contact-load/native-view/native-curve.png)

**滚翻前站立策略对照已完成。** 同快照的两条原站立参照分别完成864/896和866/896条路线，第二条七指令均过门槛。官方站立候选的右转入口为121/128，采样停在指令下发前。[对照过程与结果](reports/nominal-skill-composition/pre-roll-stand-pair/README.md) · [采样曲线](reports/nominal-skill-composition/pre-roll-stand-pair/native-view/native-curve.png) · [标注关键帧](reports/nominal-skill-composition/pre-roll-stand-pair/keyframes.jpg)

**标称连续动作已完成本轮采样和复算。** 共同场景起身与持稳886/896，官方滚翻865/896。后退和左转进入末段的场景数低于各122/128，采样停在指令下发前。已经保存从跌倒到滚翻落地站立的22.50秒原片。[标称连续动作](reports/nominal-skill-composition/README.md) · [采样曲线](reports/nominal-skill-composition/zero-recovery/native-view/native-curve.png) · [标注关键帧](reports/nominal-skill-composition/zero-recovery/keyframes.jpg)

**颈部目标按观测相位调节的CUDA对照已完成。** 第49步共同站立快照后的三组共用同一个模型，候选按当前投影重力和俯仰角速度上调颈部目标，最高0.02rad。两个种子的候选分别通过108/128和115/128，原门槛为116/128。两组共358,912条仿真转移。[逐组评分与六段10秒原片](reports/roll-forward-input-integration/endpoint-progress-training/observed-neck-support/current-evaluation.md) · [标注关键帧](reports/roll-forward-input-integration/endpoint-progress-training/observed-neck-support/neck-support-keyframes.jpg) · [种子1247状态](reports/roll-forward-input-integration/endpoint-progress-training/observed-neck-support/seed-1247/native-view/native-status.txt) · [种子1248采样曲线](reports/roll-forward-input-integration/endpoint-progress-training/observed-neck-support/seed-1248/native-view/native-curve.png)

**滚翻关节目标预检已完成。** 颈部俯仰上调0.02rad在两个候选配对失分组中分别有37/45和40/48条记录通过力矩、静态间隙和无约束加速度方向的联合检查。已保存1,045条原力矩复算及全部候选设置的结果。[关节目标预检与命令日志](reports/roll-forward-input-integration/endpoint-progress-training/README.md#关节目标预检)

**低摩擦滚翻末端接触奖励候选已完成50次更新。** 本机RTX 3080 Laptop GPU使用4,608个并行场景，采集5,529,600条训练转移。最终ONNX、保存点、曲线及两段完整训练原片已复核。[VS Code训练状态](reports/roll-forward-input-integration/endpoint-progress-training/candidate/final-view/native-status.txt) · [完整训练曲线](reports/roll-forward-input-integration/endpoint-progress-training/candidate/final-view/native-curve.png) · [训练与评估记录](reports/roll-forward-input-integration/endpoint-progress-training/README.md)

**滚翻动作对照已完成。** 末端接触奖励候选在种子1247和1248分别通过111/128和112/128，两组均低于116/128门槛。动作评分、接触复盘及关键帧已保存。[逐组评分与六段10秒原片](reports/roll-forward-input-integration/endpoint-progress-training/action-evaluation/current-evaluation.md) · [标注关键帧](reports/roll-forward-input-integration/endpoint-progress-training/action-evaluation/endpoint-reward-keyframes.jpg)

**滚翻支撑时序复盘已完成。** 两组各12个配对失分场景的主要端点离地缺口集中在180°–270°，候选较两次参照多扣16.79°至20.49°。下一项测量该区间的4ms子步承重与电机力矩。[分阶段差值图](reports/roll-forward-input-integration/endpoint-progress-training/action-evaluation/support-loss-phases.png) · [原打分和逐场景账目](reports/roll-forward-input-integration/endpoint-progress-training/action-evaluation/support-loss-localization.json)

**头部再接触载荷测量已完成。** 两个种子共保存142,080条4ms场景记录。两次参照都通过而候选失分的场景中，未计分正向转角的头部承重比例为85.57%和87.54%。[逐组评分与六段原片](reports/roll-forward-input-integration/endpoint-progress-training/head-recontact-load-recovery/current-evaluation.md) · [载荷和关节力矩](reports/roll-forward-input-integration/endpoint-progress-training/head-recontact-load-recovery/physical-load-summary.json) · [首次退出记录](reports/roll-forward-input-integration/endpoint-progress-training/head-recontact-load-recovery-admission.json)

**低摩擦滚翻保存点对照已完成。** 第30次更新保存点在种子1247和1248分别通过121/128和111/128，两次第50次更新参照分别为116／113和117／111。候选通过首组门槛。[VS Code状态、曲线、关键帧与原片](reports/roll-forward-input-integration/checkpoint-selection/README.md) · [逐场景评分](reports/roll-forward-input-integration/checkpoint-selection/comparison.json)

**滚翻进度奖励已完成固定轨迹复算。** 在种子1248的17个未通过场景中，改用末端接触后该项奖励全部下降。下一阶段准备CUDA训练前检查。[原接触取样审计](reports/roll-forward-input-integration/support-window-force/reward-contact-timing-audit.json) · [两种接触时点的奖励复算](reports/roll-forward-input-integration/support-window-force/endpoint-progress-payment-replay.json)

**低摩擦滚翻颈部俯仰对照已完成。** 种子1247的两条参照与候选为118、118、114/128，种子1248为110、110、108/128。候选按原门槛关闭。[六段原片和逐场景评分](reports/roll-forward-input-integration/neck-support-pair/current-evaluation.md) · [关键帧、实际响应与执行日志](reports/roll-forward-input-integration/neck-support-pair/README.md)

**低摩擦滚翻支撑载荷已复核。** 种子1248在第500步完成111/128次滚翻。17个未通过场景中，83个评分点无接触的正向转动区间各有4或5个子步承重。[物理载荷与时序](reports/roll-forward-input-integration/support-window-force/contact-sample-timing.json) · [采样曲线](reports/roll-forward-input-integration/support-window-force/s1248-friction-low-low-friction/native-view/native-curve.png) · [实景终帧](reports/roll-forward-input-integration/support-window-force/s1248-friction-low-low-friction/native-view/native-latest.jpg) · [10秒原片](evaluations/duck_roll_support_force_s1248_20261121T041038/clip-step-0.mp4) · [命令日志](reports/roll-forward-input-integration/support-force-sample.log)

**自训滚翻接站立与前进开发对照已复算。** 脚底摩擦0.8下，种子1247的完整路线由0/128提升至120/128。种子1248在第500步完成112/128次滚翻，低于原门槛116/128，本批次到此结束。[对照表、采样状态和原片](reports/roll-forward-input-integration/current-evaluation.md) · [流程与复算](reports/roll-forward-input-integration/README.md) · [正式训练更新、曲线和录像](reports/roll-friction-paired-training/README.md)

**0.115m/s前进模型输入的完整20秒对照已复核4/4组。** 候选在两组低摩擦和两组标称摩擦中均为128/128，通过原动作门槛。两组低摩擦参照均为0/128。[逐组评分和八段21秒原片](reports/forward-input-full-route/current-evaluation.md) · [VS Code状态、采样曲线和标注图](reports/forward-input-full-route/README.md)

**完整步态周期偏航对照已复核。** 两个低摩擦种子中，0.115m/s模型输入的3秒偏航中位数分别为0.119657和0.120378rad/s，后段速度中位数均约0.128m/s。按冻结规则入选完整20秒对照。[逐组评分与十段8秒原片](reports/forward-stride-yaw-response/current-evaluation.md) · [方案、曲线与复算](reports/forward-stride-yaw-response/README.md)

**右脚支撑持续修正整段对照已复核4/4组。** 两个低摩擦种子下，两条路线均为0/128。两个标称种子的仅首段脉冲／持续修正分别为128/128／128/128和127/128／126/128。[逐组评分和八段原片](reports/forward-right-support-continuation/current-evaluation.md) · [VS Code状态、日志和采样曲线](reports/forward-right-support-continuation/README.md)

**右脚支撑髋俯仰时长对照已复核。** 两个低摩擦种子的持续修正分别降低160毫秒偏航0.003989和0.003583rad/s，均达到门槛。整段持续触发已通过四条20秒轨迹的目标限幅预检。[逐组评分](reports/right-support-hip-pitch-duration/current-evaluation.md) · [分支曲线与原片](reports/right-support-hip-pitch-duration/README.md) · [20秒持续修正预检](reports/forward-right-support-continuation/README.md)

**髋俯仰闭环前进已复核。** 四组各完成20秒前进。两个标称组均通过128/128，两个低摩擦组的唯一失分项为偏航。四段完整原片与本机采样曲线已归档。[逐组评分](reports/forward-phase-hip-pitch/current-evaluation.md) · [累计偏航图、曲线和原片](reports/forward-phase-hip-pitch/README.md)

**前进共同支撑关节对照已复核。** 四组共177,152条物理转移。两个低摩擦种子的左脚负、右脚正髋俯仰脉冲均达到160毫秒筛选门槛。[逐组评分](reports/forward-support-joint-pulse/current-evaluation.md) · [采样曲线、原片与复现步骤](reports/forward-support-joint-pulse/README.md)

**可识别支撑下的髋偏航对照已复核。** 四组从前进2秒后选择传感器与物理接触一致的场景，共160,768条物理转移。低摩擦左脚的前60毫秒最佳降幅不足0.00010rad/s。下一项准备在同一批支撑场景比较髋侧摆和髋俯仰。[逐组评分](reports/forward-observed-hip-yaw-pulse/current-evaluation.md) · [完整结果与四段原片](reports/forward-observed-hip-yaw-pulse/README.md) · [下一轮方案](../configs/experiments/forward_support_joint_pulse_draft.json)

**首个支撑时刻的髋偏航对照已复核。** 四组使用官方权重，左右接触分别比较双零和正负单步脉冲，共160,768条物理转移。[独立读回、曲线和共同路线原片](reports/forward-hip-yaw-pulse/README.md)

**前进输出层训练与动作评分已复核。** 两组各3,072环境、12次更新、884,736条训练转移。种子1247下，全层对照和输出层候选的标称通过数为46/128和12/128，低摩擦均为0/128。[候选曲线](reports/forward-output-layer-pair/candidate/native-view/native-curve.png) · [候选训练原片](experiments/duck_forward_output_layer_candidate_e3072_s1250_20261121T002624/clip-step-0.mp4) · [8秒筛查和20秒评分](reports/forward-output-layer-pair/development/current-evaluation.md) · [配对说明和权重](reports/forward-output-layer-pair/README.md)

**前进闭环观测噪声对照已独立复核。** 新增组在128场景完成原20秒前进，平均绝对偏航速度的场景中位数为0.28219rad/s，原门槛通过0/128。另用1,408个归档场景校准了8秒早期筛查。[评分、曲线与原片](reports/forward-observation-noise-trial/README.md) · [早期筛查复算](reports/forward-observation-noise-trial/early-screen-calibration.json)

**前进采样配对已独立复核。** 种子1247、摩擦0.8的确定性和高斯探索组均为0/128，20秒平均绝对偏航速度的场景中位数分别为0.29468和0.28916rad/s。两组各保留21.04秒完整录像。[配对评分与原片](reports/forward-action-sampling/README.md) · [采样曲线与日志](reports/forward-action-sampling/current-evaluation.md)

**官方站立接自训前进已复核。** 两组均为0/128，官方站立组的低摩擦前进偏航中位数为0.29477 rad/s。[官方站立对照与原片](reports/forward-official-stance-probe/README.md) · [训练配对最终评分](reports/forward-low-friction-yaw-reward/pair/development/current-evaluation.md)

**正式偏航奖励配对和四组动作评价已复核。** 两组各3,072环境、24次更新、1,769,472条转移。种子1247的两种摩擦下，自训两组动作通过数均为0/128，偏航奖励候选的偏航低于同预算原奖励对照。官方在标称摩擦条件下通过127/128。[独立动作评价](reports/forward-low-friction-yaw-reward/pair/development/current-evaluation.md) · [对照组训练曲线](reports/forward-low-friction-yaw-reward/pair/control/native-view/native-curve.png) · [候选组训练曲线](reports/forward-low-friction-yaw-reward/pair/candidate/native-view/native-curve.png) · [权重和录像清单](reports/forward-low-friction-yaw-reward/pair/training-check.json)

**低摩擦前进PPO容量已复核。** 3,072环境相对2,560环境的保守净吞吐增益为14.39%，显存峰值13,176 MiB。以3,072环境执行正式偏航奖励配对。[容量记录与曲线入口](reports/forward-low-friction-yaw-reward/capacity/current-training.md) · [实时训练截图](reports/forward-low-friction-yaw-reward/capacity/p1-e2560/running-in-vscode.jpg) · [配对方案与初始化复算](reports/forward-low-friction-yaw-reward/README.md)

**归档行走权重对照已复核。** 种子1247、摩擦0.8的128个同初态场景，官方和教师约束权重均为0/128，唯一失分项为偏航。两种权重的20秒偏航组内中位数分别为0.23030和0.22656 rad/s。[VS Code评分](reports/forward-teacher-actor-probe/current-evaluation.md) · [对照方案和完整原片](reports/forward-teacher-actor-probe/README.md)

**站立接前进对照已复核。** 两个标称组均通过127/128，两个低摩擦组均为0/128。256个低摩擦直接前进场景均仅因偏航失分。[直接前进结果与原片](reports/roll-friction-stand-forward/README.md) · [VS Code对照表](reports/roll-friction-stand-forward/current-evaluation.md)

**关节修正上限检查已复核。** 种子1247的同一批低摩擦场景，最大单步关节修正由0.03提高到0.035 rad后仍为0/128，平均绝对偏航速度仍超门槛。[VS Code对照表](reports/forward-correction-limit-probe/current-evaluation.md) · [配对结论与原片](reports/forward-correction-limit-probe/README.md)

**滚翻低摩擦配对训练已复核。** 低摩擦加权候选组与均匀摩擦对照组各完成50次更新、5,529,600条转移。两组保存点、更新曲线和各自两段完整原片均通过核对，进入同场景动作评估。[正式配对记录](reports/roll-friction-paired-training/README.md) · [实际初态核对](reports/roll-friction-paired-training/reset-admission/paired-check.json) · [候选组日志](reports/roll-friction-paired-training/low-friction/execution.log)

**连续路线对照已复核12/12轮。** 候选两次低摩擦滚翻分别为118/128和116/128，完整路线均为0/128。接续前进的234个场景均因偏航失分。[VS Code评估总表](reports/roll-friction-paired-development/current-evaluation.md) · [配对分析与原片](reports/roll-friction-paired-development/README.md) · [对照组ONNX](../artifacts/duck_roll_friction_uniform_e4608_s1246_20261120T213457/policy.onnx) · [候选组ONNX](../artifacts/duck_roll_friction_low-friction_e4608_s1246_20261120T213457/policy.onnx)

[候选最终曲线](reports/roll-friction-paired-training/low-friction/final-view/native-curve.png) · [候选预热原片](experiments/duck_roll_friction_low-friction_e4608_s1246_20261120T213457/clip-step-0.mp4) · [候选学习后原片](experiments/duck_roll_friction_low-friction_e4608_s1246_20261120T213457/clip-step-700.mp4) · [VS Code候选录像截图](reports/roll-friction-paired-training/low-friction/video-in-vscode.jpg)

[对照组最终曲线](reports/roll-friction-paired-training/uniform/final-view/native-curve.png) · [对照组预热原片](experiments/duck_roll_friction_uniform_e4608_s1246_20261120T213457/clip-step-0.mp4) · [对照组学习后原片](experiments/duck_roll_friction_uniform_e4608_s1246_20261120T213457/clip-step-700.mp4) · [VS Code运行截图](reports/roll-friction-paired-training/uniform/running-view.jpg)

[容量对照与选档](reports/roll-friction-ppo-capacity-live/README.md) · [4,608环境首遍曲线](reports/roll-friction-ppo-capacity-live/phase-2-pass-2-e4608/final-view/native-curve.png) · [完整原片](experiments/duck_roll_ppo-capacity_p2t2_e4608_s1249_20261120T210547/clip-step-0.mp4)

**本机滚翻采样短测已完成。** 3,584环境两遍净吞吐为26,336和26,743条转移/秒，在三个测试规模中最佳。六轮短测均保留完整7秒仿真录像和GPU记录；正式选档还要加入PPO更新。

[短测状态](reports/roll-friction-sampling-capacity-bound/native-view/native-status.txt) · [三档结果、曲线和六段原片](reports/roll-friction-sampling-capacity-bound/README.md) · [执行日志](reports/roll-friction-sampling-capacity-bound/execution.log) · [VS Code准入截图](reports/roll-friction-sampling-capacity-bound/current-view.jpg) · [低摩擦训练配方零步核验](reports/roll-friction-exposure-preflight/README.md)

**本轮采样已结束：低摩擦滚翻对照。** VS Code已核验实时自动刷新，并记录第350步截图。第500步按原门槛停止，低摩擦组为112/128。当前入口包含最后状态、过程曲线和真实视频；起身训练候选的日志和原片在下方。

[最后状态](reports/roll-matched-factor-low-continuation-s1242/friction-low/native-view/native-status.txt) · [采样曲线](reports/roll-matched-factor-low-continuation-s1242/friction-low/native-view/native-curve.png) · [真实原片](evaluations/duck_roll-friction-low-continuation_s1242_20261120T195158/clip-step-0.mp4) · [运行中截图](reports/roll-matched-factor-low-continuation-s1242/friction-low/running-view.jpg) · [执行日志](reports/roll-matched-factor-low-continuation-s1242/friction-low/execution.log) · [独立复算与失稳对照图](reports/roll-matched-factor-low-continuation-s1242/README.md)

**独立留出已完成复算，候选未通过完整四姿态门槛。** 候选总体 2,737／2,816，请求仰卧 244／264，低于 251／264 的门槛；保留现有交付模型。[留出结果与完整原片](reports/getup-monitored-pair-holdout-s4268/README.md) · [最后画面](reports/getup-monitored-pair-holdout-s4268/native-view/native-latest.jpg) · [采样速度](reports/getup-monitored-pair-holdout-s4268/native-view/native-curve.png) · [运行截图](reports/getup-monitored-pair-holdout-s4268/running-view.png)

**上一次留出评估曾中断。** 观察器读取状态文件报错后，采样器在 1,275／1,450 个控制步停止；部分数据不能作为留出成绩。观察入口的有界读取修复已检查，随后完成了上面的新种子留出。[停止与修复记录](reports/getup-monitored-pair-holdout-s4267/README.md) · [运行截图](reports/getup-monitored-pair-holdout-s4267/running-view.png)

**起身开发对照已完成独立复算。** 父模型、同步组和监控候选组分别为 1,003、998、1,006／1,024；候选通过该开发门槛，但上面的独立留出未通过。三组共用跌落状态，各保存四段完整 11 秒录像。[结果与原片](reports/getup-monitored-pair-action-s3267/README.md) · [运行记录](reports/getup-monitored-pair-action-s3267/native-view/native-status.txt) · [最后画面](reports/getup-monitored-pair-action-s3267/native-view/native-latest.jpg) · [采样速度](reports/getup-monitored-pair-action-s3267/native-view/native-curve.png) · [运行截图](reports/getup-monitored-pair-action-s3267/running-view.png)

[实际右侧起身关键帧](reports/getup-monitored-pair-action-s3267/actual-right-review/actual-right-keyframes.jpg) · [改善案例](reports/getup-monitored-pair-action-s3267/actual-right-review/candidate-right-env57-replay.mp4) · [退化案例](reports/getup-monitored-pair-action-s3267/actual-right-review/candidate-right-env370-replay.mp4)。这两段是从保存状态重建的完整回放，按结果选取案例，不能作为总体成功率。

状态文件在运行中定期更新；最后画面来自本次视频录制时实际渲染的帧。观察器提示自身不能停止训练：本轮采样器另行检查观察器健康、项目暂停状态和显存上限，异常时停止。

**最新两组均已完成。** 各完成 17 次更新、1,253,376 条转移，保存点、ONNX 和完整 8 秒原片已通过独立核验。训练期间状态约每 2 秒更新，曲线约每 15 秒更新；下面的最终文件为历史快照。[本轮对照分析](reports/getup-monitored-authorization-20261116/README.md) · [起身关键帧](reports/getup-monitored-authorization-20261116/training-key-frames.jpg)

| 内容 | 同步检查组 | 监控候选组 |
| --- | --- | --- |
| 最终状态 | [状态](reports/getup-monitored-authorization-20261116/training/control/final-view/native-status.txt) | [状态](reports/getup-monitored-authorization-20261116/training/throughput/final-view/native-status.txt) |
| 完整曲线 | [曲线](reports/getup-monitored-authorization-20261116/training/control/final-view/native-curve.png) | [曲线](reports/getup-monitored-authorization-20261116/training/throughput/final-view/native-curve.png) |
| 逐轮指标 | [指标](reports/getup-monitored-authorization-20261116/training/control/final-view/native-metrics.csv) | [指标](reports/getup-monitored-authorization-20261116/training/throughput/final-view/native-metrics.csv) |
| 8 秒起身原片 | [视频](experiments/duck_getup_auth-sync_e3072_s3266_20261116T095241/clip-step-0.mp4) | [视频](experiments/duck_getup_auth-monitor_e3072_s3266_20261116T095241/clip-step-0.mp4) |
| 日志与退出 | [日志](reports/getup-monitored-authorization-20261116/training/control/execution.log) · [退出](reports/getup-monitored-authorization-20261116/training/control/supervisor-result.json) | [日志](reports/getup-monitored-authorization-20261116/training/throughput/execution.log) · [退出](reports/getup-monitored-authorization-20261116/training/throughput/supervisor-result.json) |
| 独立核验 | [核验](reports/getup-monitored-authorization-20261116/training/control/independent-verification.json) | [核验](reports/getup-monitored-authorization-20261116/training/throughput/independent-verification.json) |

下表为上轮已完成的固定路径对照，属于历史记录。两组各完成 17 次更新、1,253,376 条转移，工作进程已退出。[上轮归档](reports/pause-archive-20261116/README.md) · [上轮结果](reports/getup-canonical-path-20261116/README.md)

| 内容 | 原检查组 | 固定路径组 |
| --- | --- | --- |
| 最终状态 | [状态](reports/getup-canonical-path-20261116/training/control/final-view/native-status.txt) | [状态](reports/getup-canonical-path-20261116/training/throughput/final-view/native-status.txt) |
| 完整曲线 | [曲线](reports/getup-canonical-path-20261116/training/control/final-view/native-curve.png) | [曲线](reports/getup-canonical-path-20261116/training/throughput/final-view/native-curve.png) |
| 8 秒训练原片 | [播放](experiments/duck_getup_auth-original_e3072_s3265_20261116T021829/clip-step-0.mp4) | [播放](experiments/duck_getup_auth-canonical_e3072_s3265_20261116T021829/clip-step-0.mp4) |
| 完整命令与退出 | [记录](reports/getup-canonical-path-20261116/training/control/supervisor-result.json) | [记录](reports/getup-canonical-path-20261116/training/throughput/supervisor-result.json) |
| 标准输出与错误 | [日志](reports/getup-canonical-path-20261116/training/control/execution.log) | [日志](reports/getup-canonical-path-20261116/training/throughput/execution.log) |
| 模型与导出复核 | [核验](reports/getup-canonical-path-20261116/training/control/independent-verification.json) | [核验](reports/getup-canonical-path-20261116/training/throughput/independent-verification.json) |

VS Code 按 **Ctrl+P** 输入 `records/training-current.md`，回车后按 **Ctrl+Shift+V** 预览，再点击表格链接。视频也可直接用系统播放器打开，无需访问本机网页。若预览器不播放视频，复制对应 `.mp4` 的本地路径到播放器。

训练中的状态约每 2 秒更新，曲线约每 15 秒更新；完整片段在动作结束后发布。已结束组的 `native-view` 包含观察器退出状态，复盘结果优先看表中的 `final-view` 和独立核验。

本轮总耗时 172.19→133.13 秒，吞吐提高 29.3%；两组间隔约三小时且只有一次顺序对照，尚需复测。训练内持稳不等于独立动作成绩，两份模型只供诊断。上一批固定路径对照的 36.5% 是单独结果，不能与本轮直接相加。

滚翻参数零步诊断已完成标称和低摩擦的数值复核。初始观测与动作一致，零步没有接触；现已冻结动态对照，详细数据见[参数检查记录](reports/roll-matched-factor-preflight-20261116/README.md)。

[上一批规模对照](reports/getup-observed-throughput-20261116/README.md) · [官方与自训模型](../artifacts/delivery/iteration_record.md) · [全流程报告](../docs/reports/development_report.html) · [全部目标](../docs/current_goal.md)

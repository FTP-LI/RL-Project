# 阶段验收矩阵

这份矩阵把 S2–S6 的动作门槛、连续路线、扰动和真机前离线准备放到一个入口。每个数字都来自对应的源 JSON 或同批 README；`matrix.json` 同时记录源文件 SHA-256，便于复核。矩阵只整理既有结果，没有启动新的学习或物理评估。

本次核验原矩阵的 26 条来源记录：23 条仍与原哈希一致；3 条指向后来更新过的两份说明文档，原哈希与当前哈希均保留，相关历史文档快照尚未定位。数值结果的 JSON 来源未发现哈希偏差。 [来源核验](source-hash-audit.json) · [修改前矩阵](matrix-before-s20-offline-review.json)

## 当前结论

2026-11-21，0.8低摩擦前进起身完成四路共同释放和4ms载荷复算。两个目标场景在双足承重后均再次用头部承重，短回放的最大局部转正增益为0.30°，原门槛3°。下一阶段先核验早于首次持续双足承重的动态姿态与接触几何，再进入原起身和全路线门槛。[同释放采集、图与短回放](../nominal-skill-composition/low-friction-forward-support/README.md)

2026-11-21，0.8低摩擦承重转正配对边界已评分。种子1257的父策略、原奖励组与候选滚翻前准入为880、882和881/896。候选七个分组的入口各达到122/128，前进相对两参照分别净变0和−1例，距原净增门槛差2和3例。候选作为实验归档，完整路线后续按原门槛检验，独立种子2246保留。[判据、曲线与12秒原片](../nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/README.md)

2026-11-21，S5标称2ms连续路线通过独立确认。保留种子2256从跌倒到分配指令结束的42.5秒完整路线通过872/896，四种实际释放姿态起身和七项指令逐项达到原门槛。两遍开发种子1257各完成873/896，已归档实际使用的权重和配置。[独立确认](../nominal-skill-composition/closed-loop-timestep-confirmation/README.md) · [标称仿真清单](../../../artifacts/delivery/nominal-continuous-simulation.json)

2026-11-21，独立种子2257在相同模型和标称物理下完成滚翻横向扰动。无扰动、左向0.08m/s和右向0.08m/s的完整路线分别通过874、867和873/896，三组七指令均达到原单项门槛。[滚翻横向扰动评分及原片](../nominal-skill-composition/closed-loop-roll-lateral-disturbance/README.md)

2026-11-21，独立种子2258在足底摩擦1.0下完成供电及时延单因素试验。7.4V组与8ms组全路线分别通过881/896、891/896，标称参照右转入口为121/128，停止于22.50秒。完成工况均按原七指令单项门槛评分，0.8低摩擦另行验收。[供电、时延、各指令和录像](../nominal-skill-composition/closed-loop-voltage-delay/README.md)

2026-11-21，低摩擦起身步长候选完成滚翻与落地859/896。前进入口122/128、左转119/128，原单项门槛为122/128。0.8工况的独立确认仍使用保留种子2246。[低摩擦对照与逐场景失分](../nominal-skill-composition/low-friction-release-integrator/README.md)

2026-11-21，低摩擦承重转正指标在13对归档样本和本机CUDA计分预检通过。左髋横滚脉冲结束于26个同初态的短回放。当前保留原起身、滚翻与七指令评分作为完整路线测试，下一项核验实际求解器的奖励采样。[局部回放与奖励预检](../nominal-skill-composition/low-friction-early-transfer-authority/README.md)

S2 七指令共同接口已通过独立留出：三个动作参考中的站立、前进、后退、左右转和左右平移均为 32/32。S3 四姿态起身的候选在独立留出中为 2,983/3,072，四种实际姿态逐项过线；实际右侧相对两组参照分别净增 4、5 例，达到原定的各净增至少 1 例的门槛。候选总体分别比两组参照少 4、5 例；此前汇总将总体差值误用作右侧门槛，现依据原轨迹和冻结方案更正。 [逐项核对](../getup-gate-clarification/README.md)

S4 官方前滚翻单动作开发集和留出均为 117/128，滚翻接站立和行走仍是单独的连续门槛。S5 的起身接七指令独立留出为 2,992/3,072，七条指令逐项达到门槛；滚翻→站立→前进路线在 1231、1232 两个种子上无偏航失败，官方父模型合计 243/256，教师约束候选也是 243/256，逐种子差异相反，所以候选不晋级。前进阶段左右速度扰动的三组均达到 116/128 开发门槛。

滚翻阶段扰动的已有开发评估没有通过：无扰动 124/128，左推 115/128 后在第 500 控制步提前停止，右推 118/128 且完整序列 116/128。S13 新种子三组短评估均过原门槛，但 `head_roll` 触发器左右均覆盖 128/128 个环境；S14 的六个单特征和两特征组合筛查也未达到失败覆盖与成功误触发门槛。S14b 对 109 个因果时序特征做留一环境重拟合后仍未准入。S15 又复核了共同场景切换；S16 名义三臂和 S17 官方 seed-1235 路线通过原门槛。S18 分别核对四组参数端点、三臂名义扰动和 S17 路线；后续复核确认端点与 S17 的落地过渡不同，不能合并宣布统一控制器抗扰通过。参数抗扰、随机化训练收益和可迁移触发器继续开放，训练与仿真保持暂停。 [S20 离线复核](../roll-domain-parameter-random-s20/offline-review.md) [S18 汇总](../robustness-evidence-s18.md) · [S17 报告](../official-route-independent-s1235/README.md)

11月21日更新了S5的前进开发证据。0.115m/s内部输入在两组低摩擦与两组标称摩擦的直接前进20秒评估中各通过128/128。低摩擦连续路线种子1247的完整通过数由0/128增至120/128，种子1248滚翻112/128，低于116/128原门槛。下一项核验滚翻开始后0.54–0.92秒的支撑计分缺口，独立留出种子2246待用。[直接前进原片](../forward-input-full-route/README.md) · [连续路线与停止复盘](../roll-forward-input-integration/README.md)

支撑交接对照已复核。脚底摩擦0.8时，种子1247和1248的颈部俯仰候选分别通过114/128和108/128。第30次保存点在相同开发种子分别通过121/128与111/128。末端接触奖励候选在两组分别通过111/128与112/128。连续路线的两组门槛及独立留出继续保留。[颈部俯仰对照](../roll-forward-input-integration/neck-support-pair/README.md) · [滚翻保存点对照](../roll-forward-input-integration/checkpoint-selection/README.md) · [末端接触奖励训练和动作对照](../roll-forward-input-integration/endpoint-progress-training/README.md)

补测4ms载荷和力矩，两组候选按原评分为115/128和109/128。新测24个配对失分场景中，多数180°–270°未计分正向转角发生在头部仍承重的物理子步，载荷在控制边界卸除。后续先核对能否延长实际承重。[分阶段复算、载荷和原视频](../roll-forward-input-integration/endpoint-progress-training/README.md#头部再接触载荷测量)

标称连续路线的滚翻前站立对照已复算。两条原站立参照完成864/896和866/896条路线，第二条通过七指令门槛。官方站立候选滚翻867/896，右转入口121/128。继续验证标称路线的重复通过和独立低摩擦门槛。[同快照结果与原片](../nominal-skill-composition/pre-roll-stand-pair/README.md)

标称种子1256的原站立路线在4ms载荷核验中完整通过866/896。独立种子1257的起身与持稳880/896，滚翻与落地站立848/896，低于七指令总入口852/896；下一阶段针对仰卧恢复后的滚翻支撑交接。标称与低摩擦最终留出种子均待用。[4ms载荷与评分](../nominal-skill-composition/contact-load/README.md) · [独立动作为何停在入口](../nominal-skill-composition/independent-development/README.md)

种子1257补录中860/896个场景达到滚翻与落地站立，站立指令入口121/128。仰卧失分组42/46次卸载后由头部重新承重。第600步共同快照的两次原策略滚翻通过855/896和859/896，颈部目标下调组856/896，三组均在七指令入口结束。下一阶段以实际起身后的启动状态设计滚翻训练课程。[支撑交接、执行力矩和原片](../nominal-skill-composition/support-transfer/README.md) · [同状态对照及逐场景载荷复算](../nominal-skill-composition/neck-descent-pair/README.md)

起身后滚翻课程完成两组各30次更新。种子1257对照中，官方853/896、原始站立重置训练组848/896、课程训练组847/896。三组均在第1125步的七指令入口结束。[等样本训练及连续动作评估](../nominal-skill-composition/post-getup-roll-curriculum/README.md)

随后逐轨迹核对4ms物理承重与20ms评分。仰卧失分组269个漏计区间中268个紧邻一次短暂卸载。对称腿部和头部补偿的固定姿态复算已完成，整机继续使用冻结的原滚翻和七指令判据。[支撑取样与关节复算](../nominal-skill-composition/support-sampling-audit/README.md)

对4ms回放复现原承重顺序的仰卧失分事件，2ms积分使同一时刻卸载由34/34变为26/34，12ms总卸载时间配对差中位数为0ms。闭环对照的原20ms动作判据保留。[步长短回放](../nominal-skill-composition/support-sampling-audit/README.md#4ms与2ms短回放)

种子1257的同状态闭环对照已完成。两条4ms参照与滚翻段2ms候选分别通过855/896、851/896和872/896。候选前进准入120/128，5个仰卧场景未进入滚翻，3个滚翻场景有效支撑转角不足。七指令原入口门槛为每项122/128，标称和低摩擦最终留出继续待用。[闭环时钟、逐项结果和三段原片](../nominal-skill-composition/closed-loop-timestep/README.md)

仰卧起身末段短测中，原起身、颈部归位、归位后调整腿部姿态均为881/896通过滚翻前准入。两种归位均解除12个触发场景的头脚接触，触发场景进入滚翻均为0/12。下一阶段定位早段重心前移与头脚接触发生的顺序。[同态短测和支撑复算](../nominal-skill-composition/supine-terminal-stance/README.md)

仰卧起身早段已核对两个开发种子的1,048个实际仰卧释放。策略接手短测的原策略和接手组分别为885/896与884/896通过滚翻前准入。18个接手场景从原策略10个通过变为接手组8个通过。后续重点是双足确认前的姿态变化及实际干预时刻的同状态配对。[早段支撑曲线与对照](../nominal-skill-composition/supine-early-support/README.md)

双足确认前的时序复算中，种子1256选定的第135步在两份种子1257的前进分组原未准入场景分别覆盖3/4和1/2。第131步同状态实际响应诊断中，两条原策略与迈脚候选均为880/896通过滚翻前原判据，前进分组各122/128，逐场景一致。连续动作的滚翻、落地站立和七项指令仍待整段复评。[迈脚时序、实际响应与录像](../nominal-skill-composition/supine-pre-bilateral-support/README.md)

4ms足底载荷补采在种子1256、1257分别得到886/896与879/896个滚翻前站稳准入。两批实际仰卧未准入组的持续双足承重中位数为696ms与656ms，准入组均为596ms。源于种子1256的提前条件在种子1257覆盖9/14，低于75%门槛。[足底时序与原片](../nominal-skill-composition/early-getup-load/README.md)

S6 已完成离线映射、官方模型选择、导出哈希和回退材料；实物 ID、机械零位、方向、IMU、通信和使能验证仍未开始，真机 IO 保持关闭。

## 逐阶段入口

| 阶段 | 状态 | 关键结果 | 证据 |
| --- | --- | --- | --- |
| S2 | 通过独立留出 | 七指令各 32/32 | [共同接口分析](../common-walk-right-support-yaw-training/yaw-direction-holdout/analysis.json) |
| S3 | 原独立留出门槛通过 | 2,983/3,072；右侧 179/183，相对两组参照净增 4、5 例 | [四姿态分析](../getup-right-robustness/holdout/analysis.json) |
| S4 | 单动作通过；连续门槛开放 | 开发、留出均 117/128 | [滚翻分析](../roll-endpoint-confirmation/analysis.json) |
| S5 | 部分通过；保留开放子门槛 | 起身接七指令 2,992/3,072；滚翻路线两种子 243/256；前进左右推均过开发门槛；滚翻阶段左推 115/128 未过线 | [七指令留出](../continuous-routed-turn-feedback-holdout-zero-recovery/README.md) · [路线对照](../route-aware-teacher-anchor-parent-compare-s1232/README.md) · [横向扰动](../continuous-forward-lateral-push/README.md) · [滚翻扰动复盘](../roll-phase-perturbation-review/README.md) |
| S6 | 离线通过；实物后置 | 物理标定与部署均为 pending | [映射核验](../../hardware/deployment-preparation/mapping-check.json) · [模型选择](../../../artifacts/delivery/simulation-model-selection.json) |

机器可读记录以[核对后的阶段矩阵](../getup-gate-clarification/matrix-corrected.json)为准；[原矩阵](matrix.json)保留，但其 S3 `candidate_promotion` 字段不能作为原右侧门槛结论。吞吐配方见[吞吐与观察选择](../throughput-selection/README.md)：行走 4,096 环境，观察器使用 D3D12 + 4 Hz，起身任务使用独立的 3,072 环境配方。

## 保留的失败证据

- [起身→站立→前进开发失败](../continuous-routed-getup-stand-walk/README.md)：完整序列 2,717/3,072，进入前进后偏航超限 277 例，未进入前进 77 例。
- [七指令开发转向失败](../continuous-routed-seven-command/README.md)：左转 396/439、右转 403/439，固定映射缺少已验收的陀螺仪反馈。
- [参数随机化多技能回归](../parameter-randomized-regression/README.md)：侧躺 19/32、前滚翻 0/32，不能作为统一技能模型。

## 可视化入口

下面的画面来自独立留出批次，保留为报告中的可检查证据；视频不会被标成实时训练画面。

- [起身接七指令 32 秒原片](../../evaluations/duck_switch_routed-turn-gyro_holdout-zero-recovery_s2175_20261110T230520/restored-gyro/clip-step-0.mp4)
- [起身接七指令关键帧](../stage-media/cards/N628.jpg)
- [滚翻→站立→行走路线报告](../route-aware-teacher-anchor-parent-compare-s1232/README.md)
- [吞吐曲线](../../benchmarks/scale_20261101T154057_s42/figures/throughput.png)

下一步只有在矩阵指出具体缺口后，才创建新的有界协议。硬件阶段继续等待用户明确开始和现场条件核对。

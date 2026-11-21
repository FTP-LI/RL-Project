# XGO-Duck关键权重与迭代

| 用途 | 权重与结果 |
| --- | --- |
| 原厂首次上机 | 按已安装的运行时及套件版本使用原厂模型。此处保存的[行走](official/xgoduck_walk.onnx)、[起身](official/xgoduck_getup.onnx)、[前滚翻](official/xgoduck_roulade.onnx)与固定官方来源逐字节一致。 |
| 标称连续仿真 | 保存原厂与自训的行走、分姿态起身策略和对应恢复点。种子2256、摩擦1.0的独立首次回合中，872/896完成整个42.5秒路线。动作切换、控制周期、模型与出处见[标称连续路线模型清单](nominal-continuous-simulation.json)。 |
| 低摩擦训练比较 | 双足承重转正奖励与原奖励组各完成17次学习更新。在种子1257同释放评估中，滚翻前准入原奖励组882/896、候选881/896。该结果与恢复位置见[阶段结果](../../records/handoff/2026-11-22/README.md)。 |

`.onnx`用于推理，`.pt`用于恢复或分析训练。静态映射、原厂来源哈希和合成输入的CPU预检见[模型文件清单](key-models.json)和[离线检查结果](../../records/hardware/deployment-preparation/essential-bundle-check.json)。实机按实际应用构建确认关节顺序与控制链路，再进入候选对照。

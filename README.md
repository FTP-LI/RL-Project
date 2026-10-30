# RL-Project

机器人强化学习、数字孪生与实时控制实验仓库。当前仓库保留两个机器人项目；训练、仿真、硬件接管和验收资料均按项目独立维护。

## 项目索引

| 项目 | 当前状态 | 入口 |
| --- | --- | --- |
| XGO Lite2 | 已完成方案与阶段文档，实物成交/到货验机记录仍待建立 | [XGO-Lite2/README.md](XGO-Lite2/README.md) |
| XGO Duck | 已购套件，官方资料归档与强化学习部署规划；实物验机待完成 | [XGO-Duck/README.md](XGO-Duck/README.md) |

## 仓库边界

- Git保存Markdown文档、机器可读元数据、配置、源代码、轻量模型和可复现测试资料。
- XGO Duck 已归档的官方源码、CAD、网格、模型和 ZIP 快照随 Git 一起保存，支持换电脑恢复；源文件按原始字节跟踪。
- 后续新增的系统镜像、训练数据和长日志按体量另行存储，并在项目资产清单中记录位置与 SHA-256。
- 日期快照必须同时提供ZIP、manifest和SHA-256；快照不等同于硬件、策略或验收已通过。
- 未经实测确认的协议、力矩控制、FOC和执行器能力必须明确标记为待验证。

## 推荐阅读顺序

1. 换电脑后先阅读 [Development handoff](HANDOFF.md)，了解当前成果、恢复步骤和下一步任务。
2. 进入具体项目的`README.md`查看当前阶段和边界。
3. 阅读`project.json`获取机器可读状态。
4. 按`docs/`中的阶段顺序推进。
5. 通过验收文档和归档清单判断结果是否可复现。

## 当前归档状态

- XGO Lite2已有2026-09-11规划快照及校验文件。
- XGO Duck 的 4 份官方源码树、4 个 ZIP 快照、版本校验和部署方案均纳入仓库；实物与策略尚未验收。
- 已保存最新 USB 摄像头／语音候选、OV2735 头部截面图及交接说明；具体采购和音频安装方案尚未定案。
- H7MS-Core-A1已迁移为独立仓库，不再保存在本仓库中；XGO Lite2文档仍将其作为后续实时控制依赖引用。

## 换电脑恢复

```bash
git clone https://github.com/FTP-LI/RL-Project.git
cd RL-Project
```

普通克隆即可获得已归档的官方源码与 ZIP，无需初始化子模块或下载 Git LFS 对象。按 [官方资料恢复与校验说明](XGO-Duck/official_resources/README.md#restore-on-another-computer)核验完整性；训练环境和 UNO Q 板端系统需另行安装。

最后整理：2026-10-30。XGO Duck 的资料研究与源码快照日期仍为 2026-10-25。

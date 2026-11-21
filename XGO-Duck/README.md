# XGO Duck

## 当前入口

训练已暂停。当前目录保存可复现工程与关键结果。完整训练日志、录像和轨迹保存在本机。原厂套机完成标定、走停和飞智黑武士4Pro验收后恢复训练。

- [关键成果与复现入口](records/handoff/2026-11-22/README.md)
- [电脑环境、仿真与导出](docs/runbooks/essential_pc_workflow.md)
- [官方套机首次部署、零点标定与飞智黑武士4Pro遥控](docs/runbooks/xgo_duck_first_deployment_and_vader4pro.md)
- [实机部署验收表](docs/runbooks/hardware_acceptance_checklist.md)

## 11月1日上板记录

用户已完成机械装配并设置舵机 ID。2026-11-01 已完成 UNO Q 软件更新、原版应用部署和 2 Mbps Bridge 配置，实测约 100 Hz 反馈、15/15 舵机在线。**IMU 初始化仍失败，机械零位未校准，站立、行走和训练尚未验证。** 当前进入 IMU 排查与后续校准准备阶段。

## 项目入口

| 文档 | 内容 |
| --- | --- |
| [Development handoff](../HANDOFF.md) | 换电脑恢复、当前成果、未完成事项和下一次开发入口 |
| [Board bring-up and IMU diagnostics](bringup/2026-11-01/README.md) | 本次软件更新、桥接配置、IMU 排查、原版恢复及校准待办；含实际部署 ZIP、日志、诊断草图和校验清单 |
| [Official resources](official_resources/README.md) | 官方硬件、运行和训练仓库，以及音视频开发分支的来源、固定版本、快照和校验 |
| [RL deployment plan](docs/rl_deployment_plan.md) | 官方步态基线 → PC 训练 → ONNX 兼容检查 → UNO Q 部署 |
| [Hardware and accessories](docs/hardware_and_accessories.md) | 缺件核对、供电和结构、USB 摄像头、音频与 ToF 的实际支持状态 |
| [Camera fit analysis](docs/camera_fit_analysis.json) · [section drawing](docs/assets/camera_fit_sections.png) | OV2710 (A) 与原版头部结构的初步尺寸检查；尚未完成装配验证 |
| [Camera and voice selection](docs/camera_voice_selection.json) · [OV2735 section drawing](docs/assets/ov2735_fit_screening.png) | 最新候选：OV2735、ReSpeaker Lite、紧凑 USB 声卡和小喇叭；原壳安装及实机效果待验证 |
| [Policy contract report](docs/policy_contract_report.json) | 5 个官方 ONNX 的接口、元数据与文件哈希；未执行推理 |
| [Project metadata](project.json) | 已确认资产、未知项、当前阶段和文档索引 |

新增目录和文件使用英文名称；官方仓库保持原始内容。本项目研究说明采用中文。文档中的官方源码引用指向归档提交，便于直接在 GitHub 阅读。

## 目录说明

```text
XGO-Duck/
├── README.md                 # 项目入口与阶段边界
├── project.json              # 机器可读的项目状态
├── docs/                     # 部署、配件、策略与相机研究
│   └── assets/               # 文档引用的图示
├── bringup/2026-11-01/        # 实际上板记录、部署版本和诊断证据
└── official_resources/       # 官方来源、版本与恢复说明
    ├── source_manifest.json
    ├── manifests/            # 各源码树的逐文件哈希
    ├── snapshots.sha256
    ├── snapshots/            # 4 个随 Git 保存的原始 ZIP
    └── <source directories>  # 4 份随 Git 保存的官方源码树
```

## 当前状态

| 字段 | 当前值 |
| --- | --- |
| 项目 ID | `xgo-duck` |
| 已购买/持有 | 套件；UNO Q、机器人扩展板和舵机由用户确认 |
| 待确认 | 板卡版本、UNO Q 内存、舵机型号/固件、电源与装配质量；IMU 初始化故障原因 |
| 官方模型 | 训练仓库提供 XGO-Duck URDF/MJCF、网格及 1910 执行器模型 |
| 推进路线 | 先解决 IMU、取得零位图并校准，再验证官方行走；之后训练和视觉 |
| 本机条件 | RTX 5070，约 12 GB 显存；Ubuntu-24.04 WSL2 内 GPU 可见 |
| 实测边界 | 已连接和刷写；15 个舵机反馈正常；8 个模型加载，但 IMU offline、推理 0 Hz；零位和运动尚未验证 |
| 资料研究日期 | 2026-10-25 |
| 文档整理日期 | 2026-11-01 |

## 优先注意事项

- 最新部署为 `b401d85`，与下方 2026-10-25 归档中的 `8837d5a` 不同。请先读[本次上板记录](bringup/2026-11-01/README.md)，尤其是 IMU 阻塞项和新版校准流程。
- 最后实测状态为 `shadow`、扭矩关闭。完整断电后 IMU 仍未初始化，不应直接进入默认站姿或行走测试。
- 上电前核对官方 2026-10-02 旧扩展板 U3/U4/U5 的 GND/SIGNAL 接线更正是否适用。
- 现有行走策略依赖关节状态和 IMU，不需要先买摄像头。成品扩展板已有 IMU 的情况下也不需要先重复购买。
- 新训练模型部署前须统一默认关节姿态：归档训练行走配置为 ±24°，主运行程序为 ±22°，音视频预览分支为 ±23°。观测/动作维度相同不代表可直接互换。
- 官方 `next-audio-video-tof` 分支已有 USB 图传和音频，ToF 尚未实现；先以主分支建立行走基线。
- 用户希望看着自己聊天并听懂动作口令，优先 USB / USB-C。当前媒体分支只有采集、播放和文字转语音，完整对话、人脸跟踪和动作口令尚需开发；语音板是否接受外置小背包仍未决定。

## 归档边界

`official_resources/` 下的 585 个官方源文件与 4 个 ZIP 均由 RL-Project 跟踪，保留原始字节、来源版本和 SHA-256。换电脑后普通 `git clone` 即可取得这些资料，校验和离线解压步骤见[归档说明](official_resources/README.md#restore-on-another-computer)。

源码按普通目录归档，不使用子模块或 Git LFS；上游 Git 提交历史不包含在源码快照中。本机原有 Git 元数据保存在被忽略的 `.upstream_git/`，不参与恢复。归档不包含已安装的训练环境、板端系统或未经记录的实物校准，也不等于实物验收通过。

`bringup/2026-11-01/` 另行保存本次实际部署的 `b401d85` 原始 ZIP 和验证证据；其文件数、来源与 SHA-256 见该目录 manifest。原有 585 个源码文件和 4 个 ZIP 的校验范围未改变。归档时板端未连接，文档中的运行状态均为记录日期的实测快照。

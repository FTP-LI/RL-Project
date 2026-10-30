# XGO Duck

用户已购买套件，并确认拥有 Arduino UNO Q、机器人扩展板和舵机。当前阶段为官方资料归档、硬件清点和强化学习部署规划；尚未进行实物验机或训练。

## 项目入口

| 文档 | 内容 |
| --- | --- |
| [Official resources](official_resources/README.md) | 官方硬件、运行和训练仓库，以及音视频开发分支的来源、固定版本、快照和校验 |
| [RL deployment plan](docs/rl_deployment_plan.md) | 官方步态基线 → PC 训练 → ONNX 兼容检查 → UNO Q 部署 |
| [Hardware and accessories](docs/hardware_and_accessories.md) | 缺件核对、供电和结构、USB 摄像头、音频与 ToF 的实际支持状态 |
| [Camera fit analysis](docs/camera_fit_analysis.json) · [section drawing](docs/assets/camera_fit_sections.png) | OV2710 (A) 与原版头部结构的初步尺寸检查；尚未完成装配验证 |
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
| 待确认 | 板卡版本、UNO Q 内存、舵机型号/数量、电源、结构件和线束完整性 |
| 官方模型 | 训练仓库提供 XGO-Duck URDF/MJCF、网格及 1910 执行器模型 |
| 推进路线 | 先完整装配和官方预训练行走，再复现训练，最后加入视觉 |
| 本机条件 | RTX 5070，约 12 GB 显存；Ubuntu-24.04 WSL2 内 GPU 可见 |
| 实测边界 | 未接机器人、未刷写、未完成训练环境安装、未验证运动策略 |
| 资料研究日期 | 2026-10-25 |
| 文档整理日期 | 2026-10-30 |

## 优先注意事项

- 上电前核对官方 2026-10-02 旧扩展板 U3/U4/U5 的 GND/SIGNAL 接线更正是否适用。
- 现有行走策略依赖关节状态和 IMU，不需要先买摄像头。成品扩展板已有 IMU 的情况下也不需要先重复购买。
- 新训练模型部署前须统一默认关节姿态：归档训练行走配置为 ±24°，主运行程序为 ±22°，音视频预览分支为 ±23°。观测/动作维度相同不代表可直接互换。
- 官方 `next-audio-video-tof` 分支已有 USB 图传和音频，ToF 尚未实现；先以主分支建立行走基线。

## 归档边界

`official_resources/` 下的 585 个官方源文件与 4 个 ZIP 均由 RL-Project 跟踪，保留原始字节、来源版本和 SHA-256。换电脑后普通 `git clone` 即可取得这些资料，校验和离线解压步骤见[归档说明](official_resources/README.md#restore-on-another-computer)。

源码按普通目录归档，不使用子模块或 Git LFS；上游 Git 提交历史不包含在源码快照中。本机原有 Git 元数据保存在被忽略的 `.upstream_git/`，不参与恢复。归档不包含已安装的训练环境、板端系统或未经记录的实物校准，也不等于实物验收通过。

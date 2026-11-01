# XGO Duck 首次上板与 IMU 排查记录

记录日期：2026-11-01（Asia/Shanghai）。机械装配由用户确认完成，舵机此前只设置了 ID。本次完成 UNO Q 软件更新、应用导入、固件编译与刷写、Python 依赖安装和 2 Mbps Bridge 配置，已读到全部 15 个舵机的反馈。**IMU 初始化仍失败，机械零位未校准，站立和行走未验证。**

归档时板端已不在 ADB 设备列表中。以下板端状态来自本次会话最后一次连接和完整断电后的实测，不能视为读取文档时的实时状态。

## 归档内容

| 文件或目录 | 用途 |
| --- | --- |
| [manifest.json](manifest.json) | 来源、部署提交、ZIP 校验及本目录逐文件 SHA-256 |
| [SHA256SUMS](SHA256SUMS) | 本目录文件校验清单；不包含自身和 manifest，避免循环 |
| [xgoduck_runtime_arduino_b401d85.zip](xgoduck_runtime_arduino_b401d85.zip) | 本次导入的原始 GitHub 源码 ZIP，含工具、MCU 草图、网页和 8 个 ONNX 文件 |
| [verification_summary.json](verification_summary.json) | 已验证项、尚未完成项和版本记录 |
| [evidence/runtime_status.json](evidence/runtime_status.json) | 完整断电重启后的状态 API 摘录 |
| [evidence/imu_observations.json](evidence/imu_observations.json) | I²C 识别和复位检查的结构化记录 |
| [evidence/logs/](evidence/logs/) | 编译、刷写、依赖安装及 IMU 诊断的关键日志 |
| [evidence/control_status.png](evidence/control_status.png) | 控制网页：15/15 舵机、IMU offline、torque off |
| [imu_probe/](imu_probe/) | 临时诊断草图及复现说明；最后已从板端运行固件中替换回原程序 |

本目录是 2026-11-01 的部署记录。项目原有 `official_resources/` 中的 2026-10-25 四份固定源码和四个 ZIP 保持原样，其运行库基线是 `8837d5a`，与本次实际部署的 `b401d85` 不同。

命令日志统一为 LF 换行并去除行尾空白；设备身份字段按占位符处理。日志原有 UTC 时间保留，文档日期使用 Asia/Shanghai。诊断结果 JSON 中标明了原始输出文件或会话结果汇总的来源。

## 已完成的工作

| 项目 | 结果与边界 |
| --- | --- |
| 机械装配 | 用户确认完成；没有替代实体装配验收 |
| 应用导入 | 先前出现 HTTP 405；板端 App CLI 更新后，App Lab 已成功导入 XGO Duck |
| 系统更新 | 重启后验证新内核和新版工具；详见下方版本表 |
| 固件编译 | 原版草图编译成功：程序 112352 bytes，静态内存 67388 bytes |
| MCU 刷写 | 更新后曾出现一次 OpenOCD 写入超时；执行板载 `arduino-reset` 后重试成功，后续恢复与冷启动加载也成功 |
| Python 环境 | 安装 numpy 2.2.6、onnxruntime 1.22.1 等依赖；在禁网容器中导入并核对两个版本通过 |
| Bridge | 官方补丁已应用；Linux router 以 2000000 baud 运行，网页反馈约 100 Hz |
| 舵机 | ID 10–14、20–24、30–34 共 15 个在线，反馈掩码 32767，没有缺失 ID |
| ONNX | 8 个模型成功加载；IMU 缺失时推理频率为 0 Hz，尚未验证动作推理或实体运动 |
| 结束状态 | 原版 XGO Duck 程序运行于 `shadow`，`enabled=false`；没有执行零位写入、站立或行走 |

### 本次记录的版本

| 组件 | 版本 |
| --- | --- |
| PC Arduino App Lab | 0.11.0 |
| 板端 arduino-app-cli | 0.14.0（原为 0.7.0） |
| 板端 arduino-cli | 1.5.1 |
| 板端 arduino-router | 0.11.0 |
| arduino-unoq 系统包 | 0.2.1 |
| 重启后的 Linux 内核 | `7.0.0-g122c2c22d838` |
| 全局 Arduino Q Boards | arduino:zephyr 1.1.0 |
| 本项目 sketch profile | **arduino:zephyr 1.0.0**，遵循上游固定版本 |
| RouterBridge / RPClite | 0.4.3 / 0.3.1，加官方 prepare-bridge 补丁 |
| Zephyr 编译工具链 | arm-zephyr-eabi 1.0.1 / GCC 14.3.0 |
| Python 基础镜像 | `ghcr.io/arduino/app-bricks/python-apps-base:0.13.1` |
| 容器 Python | 3.13.15 |

全局开发板包与项目 profile 使用不同版本属于本次保留的上游配置。不要仅因为全局已安装 1.1.0，就直接改写 `sketch/sketch.yaml` 的 1.0.0 固定版本。

## Bridge 与依赖准备

应用导入后的板端目录为 `/home/arduino/ArduinoApps/xgoduck_runtime_arduino-master`。以下是本次使用过的官方准备命令，需在对应板端应用目录中执行：

```bash
cd /home/arduino/ArduinoApps/xgoduck_runtime_arduino-master
python3 tools/prepare-bridge.py
sudo bash tools/router-2m.sh
python3 tools/reuse-toolchain.py
```

`prepare-bridge.py` 为固定版本 Bridge/RPC 库安装更大的线程栈、有界 MsgPack 帧处理、RPC 等待限制和重连处理。原始库备份留在本机工作目录，部分原始文件由脚本以 `.original` 保存；router 脚本的板端备份位于 `/home/arduino/xgoduck-backup`。这些备份没有作为完整系统恢复镜像上传。

router 配置落在 `/etc/systemd/system/arduino-router.service.d/90-xgoduck-baud.conf`。本次确认服务使用 `/dev/ttyHS1`、`2000000` baud 和 `/var/run/arduino-router.sock`，并保留板级 GPIO 就绪钩子。后续调整应运行匹配版本的官方工具，不能用一段通用 service 文件覆盖板级配置。

ADB shell 的 `TMPDIR` 在本机连接中指向不存在的 `/data/local/tmp`，所以命令行编译和启动显式使用 `/tmp`：

```bash
env TMPDIR=/tmp arduino-cli compile --profile default --jobs 2 \
  /home/arduino/ArduinoApps/xgoduck_runtime_arduino-master/sketch

env TMPDIR=/tmp arduino-app-cli app start \
  /home/arduino/ArduinoApps/xgoduck_runtime_arduino-master
```

网页入口为 `http://<BOARD_IP>:9527`。板端密码只在板端终端输入。设备序列号、实际 IP、无线网络名称和个人目录没有写入本次上传记录。

### 网络与离线依赖验证

电脑访问 GitHub/Arduino 下载站需要代理。PC App Lab 的 CLI 配置使用本机代理；板端 Docker 下载不会自动继承该代理。本次通过临时 ADB reverse 将板端 `17890` 端口转发至电脑代理 `7890`，用官方 Python 镜像和应用自己的虚拟环境完成依赖安装；随后移除了该转发。

依赖实际安装在应用 `.cache/.venv` 中；numpy 2.2.6 和 onnxruntime 1.22.1 在 `--network none` 的同版本镜像内导入成功。离线导入不等于 ONNX 动作推理或硬件控制验收通过。首次获取容器镜像、工具链和依赖仍需要联网，本归档不包含它们的二进制缓存。

### MCU 写入超时及恢复

更新后的第一次固件加载报 `timeout waiting for algorithm` 和 `error executing stm32l4 flash write algorithm`。检查了 OpenOCD 配置、GPIO 接口与进程后，使用板载 `arduino-reset` 复位 MCU，再通过原 App CLI 启动命令重试，写入成功。一次恢复成功不足以认定超时的根因已被消除；没有改写运行库 profile 或 OpenOCD 脚本。

## IMU 排查结果与限制

扩展板自带 IMU，用户已确认。程序通过 `Wire` 使用 D20/SDA、D21/SCL，探测地址 0x6A、0x6B，期望 `WHO_AM_I=0x05`。

| 检查 | 实测 |
| --- | --- |
| 100 kHz 读取 0x6A | 写寄存器地址成功、读到 1 byte、芯片标识 0x05 |
| 400 kHz 读取 0x6A | 同样成功 |
| 地址 0x6B | 未应答；设备在 0x6A，无需同时出现在两个地址 |
| Revision ID | 0x7C |
| 上游原始初始化 | 在复位结果检查处返回失败，尚未进入正常传感器配置 |
| 写 0x60=0xB0，等待 20 ms | 0x4D 返回 0x00，程序要求 0x80 |
| 相同复位，等待 500 ms | 仍返回 0x00；诊断阶段编号 3 |
| 失败后的 CTRL1–CTRL7 单字节读取 | `[32, 0, 0, 0, 0, 0, 0]` |
| 完整断电后原程序 | IMU 仍 offline；`imu_seq=0`，推理 0 Hz |

芯片手册将 0x4D=0x80 定义为复位成功。原程序在未通过检查时返回失败，界面因而显示 offline。**基本 I²C 通信通路可用，但没有有效 IMU 样本；目前不能据此断定芯片损坏或已确定驱动兼容性问题。** 供电、实际板卡版本、器件状态与驱动行为仍需进一步区分。

诊断使用独立草图，只访问 IMU 和 Bridge，没有初始化或写入舵机串口。临时测试后已恢复原版 MCU 草图；归档前一次连接中，板端 `sketch/qmi8658.h` 的 SHA-256 与原始副本一致：

```text
7eb3429dd50f77f12c45485af1d9beac204ceb8cf7b9eacaa4e5cd68fbc16dc9
```

用户在 App Lab 终端执行 `sudo poweroff`，关机后断开电池/外接供电及 USB，等待后重新上电；ADB 重新连接并再次启动原程序，IMU 仍未恢复。尚未通过万用表或示波器实测传感器供电与波形，也未完成扩展板实物版本/焊接检查。

## 舵机机械零位尚未校准

设置 ID、编码器读数 2048、机器人站立姿态都不能单独证明机械零位正确。需要与当前结构和舵机版本对应的逐关节图，标注 ID、正/侧视图和相邻零件对齐基准。不能自行假定腿伸直或嘴闭合就是零位。

| 范围 | 关节组 |
| --- | --- |
| 10–14 | 左腿 |
| 20–24 | 右腿 |
| 30–33 | 头颈 |
| 34 | 嘴部 |

用户没有厂家零位图。官方硬件构建指南也说明逐关节机械零位照片仍不完整；装配 PDF 和整机照片可用于辨认安装关系，但未被确认为完整零位标准。

在确认机械零位图、解决 IMU 问题并支撑机身后，按本次部署版本的校准界面操作：

1. 打开 **Calibration**，全部舵机自动卸力。
2. 按厂家图用手摆正一个关节，点击对应 ID 的 **Set center & hold**。
3. 当前关节保持在新设中心；已经设定的关节继续保持，其余关节仍卸力，依次完成。
4. **Unload all** 释放全部关节，不撤销已经写入的中心值。
5. **Exit** 退出校准并卸力；离开页面或将页面隐藏也会退出并卸力，因此全过程需要支撑。

上述步骤尚未在这台机器人上执行。旧硬件指南中关于进入校准即转至 raw 2047、备份 `zero_pos.json` 的说明与本次运行版本不一致，应以归档 ZIP 内 `assets/calibrate.js`、`python/controller.py` 和 `sketch/` 的当前逻辑为准。不要直接复制旧版零位文件或其他机器的偏移值。

## 下一次工作的顺序

1. 根据扩展板实物版本、原理图和焊接情况继续排查 IMU。保留成功标志检查，不能仅删除检查条件就宣布 IMU 恢复。
2. 读取新的 `/api/status`，验证 IMU 持续有新样本、舵机反馈完整、模型推理实际运行，并保持扭矩关闭。
3. 向厂家取得 15 个关节的机械零位图，确认适用的结构和舵机版本后逐个校准。
4. 在支撑条件下验证默认姿态、卸力行为和关节方向，再开展低速行走。当前不具备直接宣告行走可用的依据。
5. 之后继续原项目的强化学习、相机和语音计划；本次没有安装训练环境或验证这些功能。

## 官方资料与版本区别

- [本次运行版本 b401d85](https://github.com/LuwuDynamics/xgoduck_runtime_arduino/tree/b401d85d20bb5bfd0237242a8a5ab6847fc962a2)
- [本次版本校准说明](https://github.com/LuwuDynamics/xgoduck_runtime_arduino/tree/b401d85d20bb5bfd0237242a8a5ab6847fc962a2#calibration)
- [硬件构建指南及零位图缺失说明](https://github.com/LuwuDynamics/xgoduck_hardware/blob/master/docs/BUILD_GUIDE.md#5-calibrate-then-test-in-stages)
- [装配 PDF](https://github.com/LuwuDynamics/xgoduck_hardware/blob/master/Assembly_Guide.pdf)
- [扩展板目录与版本说明](https://github.com/LuwuDynamics/xgoduck_hardware/tree/master/PCBA)
- [扩展板实物照片](https://github.com/LuwuDynamics/xgoduck_hardware/blob/master/media/pcba-v1.1/ArduinoUnoQ-board-photo.jpg)
- [V1.1 丝印位置图](https://github.com/LuwuDynamics/xgoduck_hardware/blob/master/PCBA/SilkscreenLayout/ArduinoUnoQV1.1.pdf)
- [V1.1 Altium 原理图](https://github.com/LuwuDynamics/xgoduck_hardware/blob/master/PCBA/ArduinoUnoQV1.1.SchDoc)
- [QST QMI8658A 数据手册，Waveshare 镜像](https://files.waveshare.com/upload/5/5f/QMI8658A_Datasheet_Rev_A.pdf)

硬件仓库提示照片中的板卡与工程文件存在差异。V1.1 与旧版归档应分别使用；官方照片不能代替对用户实物版本和焊接的检查。上游运行仓库未声明整个项目及 ONNX 模型的统一许可证，原 ZIP 内第三方声明和许可证文件已完整保留。

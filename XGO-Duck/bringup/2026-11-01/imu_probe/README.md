# 临时 IMU 初始化诊断

这份草图用于 2026-11-01 的排查，已编译、上传并读取诊断结果。最后已恢复原版 XGO Duck MCU 程序；该草图不是日常机器人固件。

## 文件与来源

- `imu_probe.ino`：本次诊断入口，使用 Arduino RouterBridge 提供三个 RPC 方法；没有初始化或写入舵机 UART。
- `qmi8658.h`：来自部署 ZIP 中的原版头文件，仅增加阶段编号和可选复位等待时间。未跳过复位或配置检查。
- `sketch.yaml`：沿用原项目的 arduino:zephyr 1.0.0 和库固定版本；需要已执行官方 `prepare-bridge.py`。

`qmi8658.h` 原文件 SHA-256 为 `7eb3429dd50f77f12c45485af1d9beac204ceb8cf7b9eacaa4e5cd68fbc16dc9`。来源为 [LuwuDynamics/xgoduck_runtime_arduino b401d85](https://github.com/LuwuDynamics/xgoduck_runtime_arduino/blob/b401d85d20bb5bfd0237242a8a5ab6847fc962a2/sketch/qmi8658.h)，原仓库未声明统一许可证，第三方声明保留在同级部署 ZIP 中。

## RPC 与结果含义

| RPC | 参数 | 用途 |
| --- | --- | --- |
| `codex_imu_probe` | 100000 或 400000 | 在 0x6A/0x6B 读取 WHO_AM_I，返回 JSON 字符串 |
| `codex_imu_init` | 20、100 或 500 | 调用原初始化流程，仅改变复位后等待毫秒数；返回阶段与寄存器值 |
| `codex_imu_read_reg` | 0–96 的十进制寄存器地址 | 对 0x6A 做一次单寄存器读取，返回整数 |

阶段 1：地址/标识探测；2：发送软复位；3：检查复位结果；4：写配置；5：核验配置；6：初始化通过。本次实测失败在阶段 3，20 ms 和 500 ms 都返回 0x4D=0x00。100 ms 参数已包含在程序中，未单独执行测试。

`write_status=1` 是该 Zephyr Wire 实现返回的通用传输失败，不应据此细分成特定 I²C 错误码。

## 复现边界

编译目录名须与 `.ino` 文件名一致。本归档目录为 `imu_probe`，入口为 `imu_probe.ino`：

```bash
env TMPDIR=/tmp arduino-cli compile --profile default --jobs 2 /path/to/imu_probe
```

部署前需要明确停止原运行容器、保持机身支撑和扭矩关闭；上传会替换 MCU 当前程序。调用 RPC 的进程需具有原 Arduino 应用同等的 router socket 组权限。本次使用官方 Python 容器镜像、用户 1000:1000 和现有 router 组，未更改主机 socket 权限。

优先在具备上述 socket 访问权限的官方 Python 环境中调用：

```python
from arduino.app_utils import Bridge

print(Bridge.call("codex_imu_probe", 100000))
print(Bridge.call("codex_imu_probe", 400000))
print(Bridge.call("codex_imu_init", 20))
print(Bridge.call("codex_imu_init", 500))
print(Bridge.call("codex_imu_read_reg", 1))   # revision
print(Bridge.call("codex_imu_read_reg", 77))  # 0x4D reset result
```

本次板载 `arduino-router-cli` 打印整数响应时出现 `unsupported type: int8`，一次 RPC 错误显示也触发了类似打印异常；这属于命令行工具的输出问题，不能拿它当作 IMU 返回值。改用 Python Bridge 后读到 Revision 124（0x7C）、复位结果 0。

测试结束后，通过原应用的 App CLI 启动流程重新编译/加载原版固件并启动容器，再检查 `/api/status`。本次已这样恢复，并验证 `shadow`、`enabled=false`、15/15 舵机反馈。

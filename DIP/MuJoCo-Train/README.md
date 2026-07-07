# MuJoCo-Train

`MuJoCo-Train` 是 Windows 本机训练目录。

它负责：

```text
1. 编写和修改 MuJoCo 模型
2. 调整 RS05 电机参数
3. 封装 Gymnasium 环境
4. 运行随机 rollout / 传统控制器 / 后续 PPO 训练
5. 把运行时文件同步给 MuJoCo-Lab
```

## 目录结构

```text
MuJoCo-Train/
├── assets/              # MuJoCo XML 模型
├── configs/             # 本机训练/仿真配置
├── docs/                # 建模、规划、电机说明
├── hardware/            # RS05 等硬件参数
├── pendulum_lab/        # Python 运行时代码
├── scripts/             # Windows 一键脚本
├── tests/               # 基础测试
├── runs/                # 训练日志，自动生成
├── artifacts/           # 训练导出结果，自动生成
└── requirements.txt
```

## 初始化

```powershell
cd C:\Users\27348\Desktop\learn\MuJoCo-Train
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

## 检查模型

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_model.ps1
```

## 随机动作仿真

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\random_rollout.ps1
```

## 更新部署验证目录

当你修改了模型、配置、电机参数或 Python 运行时代码后，运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\update_lab_runtime.ps1
```

这个脚本会把这些内容复制到：

```text
../MuJoCo-Lab/
```

然后再用外层 `tools/sync_to_pi.ps1` 同步到树莓派。


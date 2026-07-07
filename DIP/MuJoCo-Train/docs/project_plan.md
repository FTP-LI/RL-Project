# 项目规划：MuJoCo 二阶倒立摆

这个项目的定位不是一开始就做完整机器人，而是先用一个足够小、足够典型的系统，体验强化学习和机器人仿真的完整流程。

项目目标：

```text
在 MuJoCo 中实现一个二阶圆周倒立摆，
完成从建模、仿真、控制、训练到树莓派验证的闭环。
```

## 1. 项目边界

第一阶段只做：

```text
二阶圆周倒立摆
MuJoCo 仿真
Python 控制与训练
Windows 本机训练
树莓派验证模型
```

第一阶段暂时不做：

```text
六自由度机械臂
复杂 ROS 2 系统
真实高速闭环电机控制
复杂视觉感知
多机器人并行训练
```

这样可以先把主流程走通，不会一开始就被工程复杂度淹没。

## 2. 系统理解

二阶圆周倒立摆可以拆成三部分：

```text
主动旋转臂
第一节被动摆杆
第二节被动摆杆
```

在当前模型里：

```text
shoulder_yaw = 主动电机控制的水平旋转关节
pendulum_1   = 第一节被动倒立摆
pendulum_2   = 第二节被动倒立摆
```

控制目标：

```text
通过控制 shoulder_yaw 的力矩，
让 pendulum_1 和 pendulum_2 尽量保持竖直向上。
```

## 3. 推荐目录结构

当前目录结构保持为：

```text
MuJoCo-Train/
├── assets/              # MuJoCo XML 模型
├── configs/             # 实验参数
├── docs/                # 文档和规划
├── hardware/            # 电机、传感器、真实硬件参数
├── pendulum_lab/        # Python 源码
│   ├── controllers/     # 传统控制器
│   ├── envs/            # Gymnasium 风格环境
│   ├── scripts/         # Python 命令行入口
│   └── utils/           # 工具函数
├── scripts/             # PowerShell 一键脚本
├── tests/               # 测试
├── runs/                # 训练日志，自动生成
└── artifacts/           # 导出模型，自动生成
```

通俗理解：

```text
assets    = 机器人长什么样
configs   = 参数怎么设
hardware  = 真实电机和硬件信息
envs      = 强化学习看到的环境
controllers = 不用 AI 的传统控制方法
runs      = 训练过程
artifacts = 最终可部署的模型
```

## 4. 阶段规划

### 阶段 0：环境跑通

目标：

```text
MuJoCo 能安装
模型能加载
环境能 reset 和 step
随机动作能跑完一局
```

对应命令：

```powershell
cd C:\Users\27348\Desktop\learn\MuJoCo-Train
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\check_model.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\random_rollout.ps1
```

完成标准：

```text
check_model 输出 nq/nv/nu
random_rollout 输出 episode reward
```

### 阶段 1：模型调参

目标：

```text
让 MuJoCo 模型更接近真实机构
```

需要补：

```text
连杆长度
连杆质量
质心位置
摩擦
电机力矩上限
RS05 电机参数
```

重点文件：

```text
assets/double_rotary_pendulum.xml
hardware/motors/rs05.json
docs/model_notes.md
docs/motor_selection.md
```

完成标准：

```text
模型尺寸和你计划做的实物基本一致
RS05 的关键参数填入配置
```

### 阶段 2：传统控制基线

目标：

```text
先不用强化学习，写一个简单控制器观察系统行为
```

推荐顺序：

```text
PD 阻尼控制
能量控制
LQR / iLQR
```

为什么要做传统控制：

```text
它能帮助你判断模型是否合理。
如果传统控制都完全不对，RL 训练通常也会很难调。
```

完成标准：

```text
能用一个固定控制器跑 rollout
能画出角度、角速度、力矩曲线
```

### 阶段 3：强化学习环境

目标：

```text
把倒立摆封装成标准 Gymnasium 环境
```

环境要定义清楚：

```text
observation: 角度、角速度、sin/cos 编码
action: 底座电机力矩
reward: 直立奖励、动作惩罚、速度惩罚
done: 倒得太厉害或超时
```

当前已经有初版：

```text
pendulum_lab/envs/double_rotary_pendulum.py
```

完成标准：

```text
reset/step 接口稳定
随机动作能跑
reward 数值范围合理
```

### 阶段 4：本机训练

目标：

```text
在 Windows 本机训练一个能让二阶倒立摆保持直立的策略
```

推荐算法：

```text
PPO 作为第一版
SAC 作为后续版本
```

第一版建议使用：

```text
stable-baselines3 + PPO
```

完成标准：

```text
训练日志能保存
模型能保存
模型能复现验证
```

### 阶段 5：树莓派验证

目标：

```text
把训练好的策略导出到树莓派上运行验证
```

树莓派第一阶段只做：

```text
加载模型
运行推理
记录输入输出
低频验证控制逻辑
```

树莓派暂时不做：

```text
高频电机闭环
大规模训练
复杂仿真
```

完成标准：

```text
树莓派能加载策略
同一组观察输入能输出动作
能和 Windows 推理结果对齐
```

### 阶段 6：真实硬件准备

目标：

```text
从仿真走向真实 RS05 电机实验
```

需要准备：

```text
RS05 CAN 通信
角度零点标定
机械限位
急停
电源保护
日志记录
低力矩安全模式
```

完成标准：

```text
电机能低力矩稳定响应命令
编码器角度可信
控制程序有急停和限位保护
```

## 5. 当前第一优先级

现在不要急着训练 PPO。

第一优先级是：

```text
1. 安装 MuJoCo
2. 跑通 check_model
3. 跑通 random_rollout
4. 打开 viewer 看模型运动
5. 补 RS05 参数
```

等模型能看、能跑、参数大概可信，再进入训练。

## 6. 学习路线

建议按这个顺序理解：

```text
MuJoCo XML
MuJoCo qpos/qvel/ctrl
Gymnasium reset/step
reward 设计
传统控制器
PPO 训练
模型导出
树莓派验证
真实硬件安全控制
```

这条路走完，你就会对 Isaac Lab 那类项目的整体流程有感觉，而不是只看到一堆工程文件。

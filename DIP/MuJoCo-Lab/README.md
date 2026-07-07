# MuJoCo-Lab

`MuJoCo-Lab` 是树莓派部署验证目录。

它负责：

```text
1. 接收 MuJoCo-Train 导出的运行时代码
2. 在树莓派安装最小依赖
3. 加载 MuJoCo 模型
4. 运行验证脚本
5. 后续加载训练好的策略做推理验证
```

它不负责训练。训练和模型调参请在：

```text
../MuJoCo-Train/
```

## 目录结构

```text
MuJoCo-Lab/
├── assets/              # 从训练目录同步来的 MuJoCo XML
├── configs/             # 验证配置
├── hardware/            # RS05 电机参数
├── pendulum_lab/        # 验证所需 Python 代码
├── scripts/             # 树莓派一键脚本
├── artifacts/           # 后续放导出的策略/模型
└── requirements.txt
```

## 树莓派初始化

同步到树莓派后：

```bash
cd ~/MuJoCo-Lab
bash scripts/setup_pi_ubuntu.sh
```

## 树莓派验证

```bash
cd ~/MuJoCo-Lab
bash scripts/validate_mujoco.sh
```

当前验证内容：

```text
1. MuJoCo 模型能加载
2. 环境能 reset
3. 随机动作能 step
4. reward 能输出
```

后续训练出策略后，再把策略导出到 `artifacts/`，在这里做树莓派推理验证。


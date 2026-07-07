# 项目架构

这个项目参考 Isaac Lab 的分层思想，但先保持轻量：

```text
assets   = 机器人/环境模型
configs  = 参数配置
envs     = 强化学习环境接口
scripts  = 可直接运行的命令入口
runs     = 训练过程数据
artifacts = 导出的策略或模型
```

和 Isaac Lab 不同的是，当前项目不引入复杂任务注册系统。先把最小链路跑通：

```text
MuJoCo XML -> Python Env -> rollout -> control baseline -> RL training
```

等任务多起来，再加：

```text
task registry
experiment manager
policy exporter
deployment package
```


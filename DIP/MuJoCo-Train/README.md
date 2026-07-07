# MuJoCo-Train

MuJoCo-Train 是 Windows 本机训练目录。它的目标不是只保存代码，而是形成一条可以复现的学习和验证流程：

CAD → MJCF → MuJoCo → Python 策略 → Raspberry Pi + CAN

也就是说，先用 CAD 或 3D 建模工具把机械结构设计清楚，再整理成 MuJoCo 使用的 MJCF 模型，在 MuJoCo 里验证动力学和控制策略，最后把策略部署到树莓派，通过 CAN 模块控制电机。

URDF 可以作为中间文件，但不要把 URDF 当成最终训练模型。更合理的用法是：

CAD 导出 URDF
用 URDF 检查 link、joint、inertial 是否合理
再手动整理成 MJCF
在 MuJoCo 里补 actuator、contact、damping、frictionloss、ctrlrange 等训练需要的参数

## 目录结构

MuJoCo-Train/
├── assets/              # MuJoCo XML 模型，以及可能用到的 mesh/texture
├── configs/             # 本机训练和仿真配置
├── docs/                # 建模说明、架构说明、阶段计划
├── hardware/            # 电机、执行器等硬件参数
├── pendulum_lab/        # Python 运行时代码
├── scripts/             # Windows 一键脚本和验证脚本
├── tests/               # 基础测试
├── runs/                # 训练日志，自动生成，不提交
├── artifacts/           # 临时验证文件和训练产物，自动生成，不提交
└── requirements.txt

assets、configs、hardware 和 pendulum_lab 是替换自己模型时最常改的地方。runs 和 artifacts 是运行过程中产生的数据，不应该提交到 GitHub。

## 第一部分：基础环境验证

基础环境验证的核心目标是先回答一个问题：在不使用自己模型的情况下，官方 MuJoCo 是否能在当前 Windows 电脑上正常工作。

这样做是为了把问题拆开。如果一开始就加载自己的 XML，报错来源可能是 Python 版本、依赖安装、MuJoCo 包、OpenGL 显示环境，也可能是 XML 写错。先用官方模型验证，可以避免把环境问题误判成模型问题。

### 1. 使用虚拟环境隔离依赖

项目使用 .venv 虚拟环境，而不是把依赖直接装进系统 Python。这样做有两个好处：

1. 这个项目需要的 mujoco、gymnasium、numpy 不会影响电脑上其他 Python 项目。
2. 后续如果强化学习库对 Python 版本有要求，可以直接重建 .venv，不需要动系统环境。

初始化环境时，先进入项目目录：
C:\Users\27348\Desktop\RL-Project\DIP\MuJoCo-Train

然后运行初始化脚本：
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1

这个脚本会创建 .venv，升级 pip，并按照 requirements.txt 安装依赖。

如果默认 PyPI 网络不稳定，可以临时使用镜像安装依赖：
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

这只是网络下载源的区别，不改变项目依赖本身。

### 2. 确认当前 Python 版本

确认 Python 版本是为了判断依赖包是否匹配。MuJoCo 的官方 Python 包会按 Python 版本和 Windows 架构下载对应 wheel，如果版本太旧或架构不对，安装和导入都会失败。

查看系统 Python：
python --version

查看当前项目虚拟环境里的 Python：
.\.venv\Scripts\python.exe --version

项目实际运行时更应该以后者为准，因为所有脚本都使用 .venv\Scripts\python.exe。

### 3. 确认 MuJoCo 包可以导入

这一步只验证 Python 能否找到 mujoco 包，还不涉及任何模型。它可以快速判断依赖安装是否完成。

运行：
.\.venv\Scripts\python.exe -c "import mujoco; print('MuJoCo version:', mujoco.__version__); print('Package:', mujoco.__file__)"

成功时应该能看到版本号和包路径，例如：
MuJoCo version: 3.x.x
Package: ...\site-packages\mujoco\__init__.py

如果这里报 ModuleNotFoundError: No module named 'mujoco'，说明当前 .venv 里没有安装 MuJoCo，需要重新安装 requirements.txt。

### 4. 使用官方模型验证编译和仿真

导入成功并不代表 MuJoCo 完全可用。真正的仿真流程还需要 MJCF 编译器能读取 XML，并能创建 MjModel、MjData，最后通过 mj_step 推进仿真。

本项目提供了脚本，运行：
powershell -ExecutionPolicy Bypass -File .\scripts\verify_official_mujoco.ps1

这个脚本做的事情是：

1. 导入官方 mujoco Python 包并打印版本号。
2. 下载官方 humanoid.xml 到 artifacts/official_models。
3. 用 MjModel.from_xml_path 编译官方模型。
4. 创建 MjData 并执行 100 步 mj_step。

成功输出类似：
MuJoCo version: 3.x.x
Package: ...\site-packages\mujoco\__init__.py
official humanoid ok: nq=28, nv=27, nu=21, time=0.5000

这里的 nq、nv、nu 分别代表广义坐标数量、广义速度数量和 actuator 控制输入数量。能打印这些值，说明模型已经被 MuJoCo 正确编译。

### 5. 使用官方 viewer 验证图形显示

前面的 mj_step 是无窗口仿真，只能证明物理引擎能跑。后续调模型时经常需要肉眼检查关节方向、连杆位置、碰撞体和 actuator 效果，所以还要验证 viewer。

运行：
powershell -ExecutionPolicy Bypass -File .\scripts\verify_official_mujoco.ps1 -OpenViewer

如果能看到官方 humanoid 模型，说明 MuJoCo viewer、OpenGL 和显卡显示环境基本可用。

如果无窗口仿真通过，但 viewer 打不开，通常说明 MuJoCo 本体没有问题，问题更可能在显卡驱动、OpenGL、远程桌面或窗口权限上。此时可以先继续做无窗口仿真验证，后面再单独处理显示问题。

### 6. 基础环境里程碑

[ ] .venv 中的 Python 版本明确
[ ] import mujoco 成功
[ ] 官方 humanoid.xml 能被 MjModel.from_xml_path 编译
[ ] 官方 humanoid.xml 能通过 mj_step 推进仿真
[ ] mujoco.viewer 可以打开官方模型，或已确认只是图形显示问题

## 第二部分：CAD 建结构

这一部分解决的是“真实机械结构怎么先被设计清楚”。这里还不是 MuJoCo 建模，而是先把机械结构、关节关系和电机位置想明白。

### 1. 推荐工具

可以使用 SolidWorks、Fusion 360、FreeCAD、Onshape 这类 CAD 工具。它们主要负责：

画真实结构
确定连杆长度和安装位置
估算质量
查看质心
导出惯量
导出 STL/OBJ mesh
必要时通过插件导出 URDF

注意：CAD 软件通常不能直接导出一个训练可用的 MJCF。即使能通过第三方工具导出，也通常还需要手动整理 actuator、contact、damping、frictionloss、ctrlrange、joint axis 等 MuJoCo 专用参数。

### 2. 建结构时先确定什么

双旋转倒立摆第一版至少要明确：

底座是否固定
主动旋转关节在哪里
第一根摆杆和底座怎么连接
第二根摆杆和第一根摆杆怎么连接
哪个关节由 RS05 电机驱动
哪些关节是被动关节
编码器或电机反馈能测到哪些量
机械限位在哪里
电机线缆和结构是否会干涉

这一步不要一开始追求漂亮外观。先把“能运动、能测量、能控制、不会机械干涉”想清楚。

### 3. 建结构阶段里程碑

[ ] CAD 中有完整机械草图或初版装配体
[ ] 明确主动关节和被动关节
[ ] 明确每个关节的旋转轴方向
[ ] 明确电机安装位置和输出轴方向
[ ] 明确机械限位
[ ] 明确需要导出的连杆、质量、惯量、mesh

### 4. B 站补充学习建议

这一部分建议在 B 站找 CAD 建模和 URDF 导出的教程。不要只看“建得很漂亮”的视频，要找能讲清楚 link、joint、坐标系、惯量的内容。

建议搜索关键词：
SolidWorks 机器人 URDF 导出
Fusion 360 URDF 导出
机器人 link joint inertial 讲解
ROS URDF RViz 模型检查

看视频时重点看：

坐标系怎么定
link 和 joint 怎么分
惯量和质量怎么导出
mesh 路径怎么管理
URDF 在 RViz 里怎么检查

## 第三部分：提取尺寸、质量、惯量、关节轴

这一部分解决的是“CAD 模型里哪些信息要变成仿真参数”。MuJoCo 训练效果好不好，很大程度取决于这些参数是否靠谱。

### 1. 需要记录的参数

每个连杆至少记录：

长度
宽度或半径
质量
质心位置
惯量矩阵或近似惯量
与父 body 的连接位置
是否参与碰撞
是否只作为视觉显示

每个关节至少记录：

关节名称
父 body
子 body
关节类型
旋转轴方向
初始角度
角度范围
阻尼
摩擦
是否被电机驱动

每个电机至少记录：

电机型号
控制模式
峰值力矩
持续力矩
最大速度
CAN ID
CAN 波特率
命令格式
反馈格式

### 2. 参数放在哪里

建模说明写在：
docs/model_notes.md

电机参数写在：
hardware/motors/rs05.json

模型配置写在：
configs/double_rotary_pendulum.json

这样做是为了避免把重要参数散落在 Python 代码里。以后换电机、换模型、换训练参数时，只需要改配置文件。

### 3. 参数提取阶段里程碑

[ ] 每个连杆有尺寸记录
[ ] 每个连杆有质量或估计质量
[ ] 每个连杆有质心或估计质心
[ ] 每个关节有明确 axis
[ ] 每个关节有角度范围
[ ] 电机力矩、速度、控制模式有记录
[ ] 哪些参数是估计值已经写进 docs/model_notes.md

### 4. B 站补充学习建议

建议搜索关键词：
转动惯量 通俗讲解
机器人 URDF inertial 参数
SolidWorks 质量属性 惯量
Fusion 360 质量属性 惯量

看视频时重点看：

质量属性从哪里导出
坐标系不一致时惯量会不会变
质心位置怎么读
URDF inertial 和 CAD 质量属性怎么对应

## 第四部分：整理成 MJCF

这一部分解决的是“把机械结构和参数整理成 MuJoCo 能运行的 XML”。MJCF 是 MuJoCo 原生模型格式，也是这个项目最终使用的模型格式。

### 1. MJCF 和 URDF 的区别

URDF 是 ROS 里最常听到的机器人描述格式。它更偏“机器人结构描述”，常用于 ROS、RViz、MoveIt、Gazebo 这类工具链，重点是 link、joint、visual、collision、inertial。

MJCF 是 MuJoCo 的原生模型格式。它不只是描述机器人长什么样，还会更直接地描述 MuJoCo 仿真需要的内容，例如 actuator、contact、default 参数、solver 设置、tendon、sensor、equality constraint 等。

有一说一，如果你的工作流主要在 ROS 里，URDF 更常见；如果目标是 MuJoCo 训练和控制，MJCF 更直接。URDF 可以作为中间文件，但最终要整理成 MJCF。

简单判断：
接 ROS、机械臂规划、RViz 展示，优先考虑 URDF。
做 MuJoCo 动力学仿真、控制器、强化学习训练，优先整理成 MJCF。

### 2. 从 URDF 到 MJCF 的建议

如果 CAD 能导出 URDF，可以这样用：

CAD 导出 URDF
先用 ROS/RViz 或文本检查 link、joint、inertial
确认结构关系没错
再把结构整理成 MJCF
在 MJCF 里补 actuator、contact、damping、frictionloss、ctrlrange
用 MuJoCo viewer 检查最终模型

不要直接认为“URDF 能打开”就等于“MuJoCo 可以训练”。URDF 里的电机、接触、阻尼、控制输入，往往还不够训练使用。

### 3. 第一版 MJCF 应该简单

第一版 MJCF 只需要做到：

body 层级正确
joint 轴方向正确
geom 简单稳定
mass 和 inertia 合理
actuator 接到正确 joint
模型能被 MuJoCo 编译

不建议第一版就加入复杂 mesh、复杂材质、复杂接触、传感器和高级约束。先让最小动力学链路跑通。

### 4. 文件应该放在哪里

主 MJCF 模型放在：
assets/double_rotary_pendulum.xml

如果是新模型，也可以使用更明确的名字：
assets/my_pendulum.xml

mesh 文件放在：
assets/meshes/

texture 文件放在：
assets/textures/

所有引用都尽量使用相对路径，这样以后同步到 MuJoCo-Lab 或树莓派时不容易丢文件。

### 5. MJCF 阶段里程碑

[ ] XML 能被 MuJoCo 编译
[ ] body 层级符合真实机构
[ ] joint axis 与真实旋转方向一致
[ ] geom 没有明显穿插
[ ] mass 和 inertia 不为 0，数量级合理
[ ] actuator 连接到正确 joint
[ ] model.nq、model.nv、model.nu 与预期一致
[ ] mesh、texture 引用都使用相对路径

### 6. B 站补充学习建议

建议搜索关键词：
MuJoCo MJCF 教程
MuJoCo XML 建模
MuJoCo viewer 使用
URDF 转 MJCF

看视频时重点看：

body 和 joint 怎么写
axis 怎么判断
geom 和 mesh 区别
actuator 怎么接 joint
viewer 报错怎么排查

## 第五部分：MuJoCo viewer 校验模型

这一部分解决的是“模型看起来和动起来是否对”。viewer 是模型进入 Python 环境之前的第一道检查。

### 1. 打开模型

进入 MuJoCo-Train 目录：
C:\Users\27348\Desktop\RL-Project\DIP\MuJoCo-Train

打开当前模型：
.\.venv\Scripts\python.exe -m mujoco.viewer --mjcf=.\assets\double_rotary_pendulum.xml

### 2. 先看静态结构

重点检查：

模型是否出现在视野中央附近
模型大小是否正常
连杆有没有接在正确位置
父 body 和子 body 的层级是否符合真实机构
摆杆初始角度是否正确
几何体有没有互相穿插

如果模型特别远、特别大、特别小，通常是单位或坐标写错。

### 3. 再看关节和运动

重点检查：

关节轴方向是否和预期一致
轻微仿真后摆杆是否绕正确方向运动
重力方向是否正确
模型有没有一开始就抖动、弹飞或穿模

如果模型一开始就炸开，常见原因是质量、惯量、碰撞体、关节限制或初始姿态不合理。

### 4. 最后看 actuator

重点检查：

actuator 是否作用在正确 joint 上
控制输入变化时，运动的是不是你真正想驱动的关节
model.nu 是否等于实际电机数量

后面有控制输入时，如果电机一动，动的不是你想要的关节，就说明 actuator 绑定错了。

### 5. viewer 阶段里程碑

[ ] viewer 能打开模型
[ ] 结构位置正确
[ ] 关节轴方向正确
[ ] 模型仿真不炸开
[ ] actuator 绑定正确
[ ] nq、nv、nu 与预期一致

## 第六部分：MuJoCo 里封装 Gymnasium 环境

这一部分解决的是“让模型能被强化学习代码调用”。MuJoCo 只负责物理仿真，Gymnasium 环境负责把仿真包装成 reset、step、reward、terminated 这套接口。

### 1. 修改 model_path

模型路径写在：
configs/double_rotary_pendulum.json

当前默认模型是：
assets/double_rotary_pendulum.xml

如果你换了模型文件名，例如 assets/my_pendulum.xml，就修改 model_path。

### 2. 检查 nq、nv、nu

运行模型检查：
powershell -ExecutionPolicy Bypass -File .\scripts\check_model.ps1

这一步会打印：
model: ...
nq: ...
nv: ...
nu: ...
obs shape: ...

nq 表示广义坐标数量，nv 表示广义速度数量，nu 表示 actuator 控制输入数量。它们决定了 reset、action_space、observation_space 和 step 的写法。

### 3. 根据 nu 调整动作空间

当前双旋转倒立摆的 nu 是 1，所以当前环境只输出一个控制量，并写入：
self.data.ctrl[0] = torque

如果自己的模型有 2 个电机，nu 就会变成 2。这时只改 XML 不够，还要修改 pendulum_lab/envs/double_rotary_pendulum.py：

action_space 的维度要等于 model.nu
step 中写入 data.ctrl 的数量要等于 model.nu
每个 actuator 的 torque_limit 要和实际电机能力一致

### 4. 根据 nq、nv 调整 reset

如果模型关节数量变了，qpos、qvel 的长度也会变。需要同步修改：

reset 中初始化 qpos 的长度
reset 中初始化 qvel 的长度
reset_noise 的维度
初始姿态是否符合真实机械结构

### 5. 修改观测空间

强化学习环境不会直接把完整 MuJoCo 状态原样交给算法，而是通过 _get_obs 生成观测。

当前环境的观测维度是：
obs shape: (9,)

如果模型状态不同，需要同步修改：

_get_obs 返回的 numpy 数组
observation_space 的 shape
测试或脚本里对 obs shape 的预期

角度通常用 sin(angle) 和 cos(angle) 表示，而不是直接用角度值。原因是角度有周期性，pi 和 -pi 附近直接用角度会产生不连续跳变。

### 6. 修改奖励函数和终止条件

模型替换后，旧 reward 不一定还合理。需要重新确认：

哪个关节代表要平衡的摆杆
哪个角度是竖直方向
速度惩罚是否过强
动作惩罚是否与电机能力匹配
什么状态应该 terminated
episode_length 是否足够

终止条件 _is_terminated 也要跟机械结构一致。

### 7. Gymnasium 阶段里程碑

[ ] check_model.ps1 能通过
[ ] action_space 与 model.nu 一致
[ ] observation_space 与 _get_obs 一致
[ ] reset 中 qpos/qvel 长度与 nq/nv 一致
[ ] reward 使用正确关节索引
[ ] terminated 使用正确关节索引

## 第七部分：训练和验证控制策略

这一部分解决的是“控制策略是否能在仿真中跑起来”。第一版不要急着直接上真实硬件。

### 1. 先跑随机 rollout

运行：
powershell -ExecutionPolicy Bypass -File .\scripts\random_rollout.ps1

随机 rollout 的意义不是看控制效果，而是验证 reset、step、reward、terminated、truncated 能连续运行。

成功时应该看到：
episode=  1 steps= ... reward= ... terminated=...
episode=  2 steps= ... reward= ... terminated=...
episode=  3 steps= ... reward= ... terminated=...

### 2. 再做传统控制基线

在强化学习前，建议先做一个简单基线：

PD 控制
能量摆起
LQR 或线性化控制
手写安全限幅

这样做的原因是：如果传统控制都完全不稳定，通常说明模型、观测、奖励或电机参数还有问题。

### 3. 再训练强化学习策略

等环境稳定后，再考虑 PPO、SAC 等强化学习算法。训练产物应该放在 artifacts，训练日志放在 runs。

不要把训练产物直接提交到 GitHub。它们通常比较大，而且会频繁变化。

### 4. 策略验证阶段里程碑

[ ] random_rollout.ps1 能连续完成 episode
[ ] reward 没有 NaN
[ ] 状态没有 NaN
[ ] 控制输出有明确限幅
[ ] 有一个传统控制基线
[ ] 强化学习训练日志写入 runs
[ ] 策略产物写入 artifacts

### 5. B 站补充学习建议

建议搜索关键词：
MuJoCo Gymnasium 强化学习
倒立摆 PPO 教程
强化学习 PPO SAC 入门
Gymnasium 自定义环境

看视频时重点看：

reset 和 step 怎么写
action_space 和 observation_space 怎么定义
reward 怎么设计
terminated 和 truncated 有什么区别
训练日志怎么看

## 第八部分：树莓派通过 CAN 控制电机

这一部分解决的是“仿真策略怎么走向真实硬件”。推荐路线是：

MuJoCo 训练策略
Python 导出或复现策略
树莓派读取状态
树莓派通过 CAN 发送电机命令
电机内部控制器执行电流环、速度环或位置环

### 1. 推荐工具链

树莓派侧建议使用：

Linux SocketCAN
can-utils
python-can
RS05 电机 CAN 协议封装

SocketCAN 负责把 CAN 模块变成 can0 这种系统接口。

can-utils 负责调试，比如 candump、cansend。它适合先确认 CAN 总线是否能收发。

python-can 负责在 Python 里发送和接收 CAN 帧。

RS05 电机协议封装负责把目标力矩、速度、位置编码成电机能理解的 CAN 帧，也负责把反馈帧解码成角度、速度、电流、温度。

### 2. 树莓派不建议做什么

树莓派不适合直接做高频底层闭环，例如：

几 kHz 电流环
硬实时安全保护
毫秒级绝对稳定闭环

这些更适合交给电机内部控制器，或者后续加 STM32 做实时层。

树莓派更适合做：

策略推理
发送目标力矩、速度或位置
记录数据
状态机
实验流程控制
安全限幅

### 3. 部署阶段建议顺序

第一步：只接 CAN，不接机械负载，确认 can0 可用。

第二步：用 candump 看是否有电机反馈。

第三步：用 cansend 或 python-can 发送低风险命令，例如使能、读取状态、小速度转动。

第四步：写 RS05Motor 类，封装协议。

第五步：加入安全限幅，例如最大力矩、最大速度、最大角度。

第六步：让树莓派以较低频率发送目标命令，例如 100Hz 到 500Hz。

第七步：确认稳定后，再接入仿真训练出来的策略。

### 4. 硬件部署阶段里程碑

[ ] 树莓派能识别 CAN 模块
[ ] can0 能正常启动
[ ] candump 能看到电机反馈
[ ] python-can 能发送和接收 CAN 帧
[ ] RS05Motor 能正确编码命令
[ ] RS05Motor 能正确解码反馈
[ ] 有力矩、速度、位置安全限幅
[ ] 电机空载测试稳定
[ ] 低速带负载测试稳定
[ ] 策略输出接入前有急停方案

### 5. B 站补充学习建议

建议搜索关键词：
树莓派 SocketCAN
树莓派 CAN 通信
python-can 教程
CAN 总线 电机 控制
MIT 模式 电机 CAN

看视频时重点看：

CAN H/CAN L 接线
终端电阻
波特率设置
can0 怎么启动
candump 和 cansend 怎么用
电机 ID 怎么设置
电机反馈帧怎么解析

## 专业词汇解释

这一节专门用来解释 README 里出现的专业词。后面看到不熟的词，可以先回到这里查。

### 1. 模型格式相关

MJCF：
MuJoCo XML Format，MuJoCo 原生模型格式。它用 XML 描述 body、joint、geom、actuator、contact 等仿真元素。这个项目最终训练用的主模型建议是 MJCF。

URDF：
Unified Robot Description Format，ROS 生态常用的机器人描述格式。它擅长描述 link、joint、visual、collision、inertial，适合 RViz、MoveIt、Gazebo 等工具链。URDF 可以作为中间文件，但在 MuJoCo 训练前通常还要整理成 MJCF。

XML：
一种文本格式。MJCF 和 URDF 本质上都是 XML 文件，只是标签规则不同。XML 写错标签、路径、层级，MuJoCo 或 ROS 都会报错。

mesh：
三维网格模型，比如 STL、OBJ。它通常来自 CAD 或 Blender。mesh 更适合做外观显示，不建议第一版直接用复杂 mesh 做碰撞体。

texture：
贴图或材质图片。它只影响视觉显示，通常不影响动力学训练。

### 2. MuJoCo 模型结构相关

body：
刚体。可以理解成模型中的一个物理零件，例如底座、旋转臂、第一根摆杆、第二根摆杆。

parent body / child body：
父刚体和子刚体。MuJoCo 的模型是树形结构，子 body 会跟着父 body 运动。父子关系写错，模型的运动链就会错。

joint：
关节。它定义两个 body 之间怎么相对运动，例如绕某个轴旋转，或者沿某个方向平移。

joint axis：
关节轴方向。它决定关节绕哪个方向转。比如 axis 写成 1 0 0 表示绕 x 轴转，写成 0 1 0 表示绕 y 轴转。很多模型“看起来能加载，但运动方向不对”，问题就出在 joint axis。

geom：
几何体。它可以用于显示，也可以用于碰撞。常见 geom 有 box、sphere、capsule、cylinder。训练初期优先用简单 geom，因为它稳定、好排查。

visual：
视觉外观。通常只负责看起来像不像，不一定参与碰撞。

collision：
碰撞体。它决定模型和模型之间、模型和地面之间怎么接触。碰撞体太复杂或互相穿插，仿真容易抖动或炸开。

inertial：
惯性参数，包括质量、质心、惯量。它决定刚体在受力后怎么运动。惯性参数不合理，模型会很不真实。

mass：
质量，单位通常是 kg。不能随便填 0，也不能所有零件都填一样。

center of mass：
质心。物体质量分布的中心。质心位置不合理，会影响摆杆下落、旋转和控制效果。

inertia：
转动惯量。它描述物体绕不同轴转动时有多“难转”。同样质量的物体，长杆和圆盘的惯量不一样。

### 3. MuJoCo 专用参数

actuator：
执行器。它是 MuJoCo 里“施加控制输入”的东西，可以理解成电机、舵机、力矩源。没有 actuator，模型可以被动运动，但你的程序不能主动给它发控制命令。

contact：
接触参数。它决定两个物体碰到一起时怎么反应，例如接触软硬、摩擦、弹性、是否容易穿透。倒立摆这类项目第一版通常尽量减少复杂接触。

damping：
阻尼。它像“速度相关的阻力”，速度越大，阻力越明显。关节 damping 可以让模型不那么理想化，也能减少高频抖动。

frictionloss：
干摩擦或静摩擦损失。它表示关节运动时需要克服的一部分固定摩擦。太小会过于理想，太大又会让关节很难动。

ctrlrange：
控制输入范围。它限制 actuator 接收的控制命令范围。例如力矩输入不能超过电机能承受的最大值。

forcerange：
输出力或力矩范围。它限制 actuator 实际能输出的力或力矩。ctrlrange 限制“你能发什么命令”，forcerange 更接近“执行器最多能输出多少”。

joint range：
关节角度范围。它限制关节能转到哪里，类似真实机械限位。

solver：
求解器。MuJoCo 用它来计算动力学、约束和接触。一般初期不用深改 solver 参数，除非模型出现接触抖动或约束不稳定。

default：
MJCF 里的默认参数机制。可以把一类 joint、geom、actuator 的通用参数写在 default 里，减少重复配置。

tendon：
肌腱或传动结构。用于描述绳索、皮带、联动机构等。双旋转倒立摆第一版通常用不到。

sensor：
传感器。MuJoCo 里可以定义位置、速度、力、IMU 等传感器输出。训练初期可以先直接读 qpos 和 qvel。

equality constraint：
等式约束。用于把两个物体或关节用强约束绑定起来。第一版模型尽量少用，避免增加排查难度。

### 4. MuJoCo 状态变量

MjModel：
MuJoCo 编译 XML 后得到的模型对象。它包含模型结构，例如有多少关节、多少 body、多少 actuator。

MjData：
MuJoCo 的运行时数据对象。它包含当前状态，例如位置、速度、控制输入、接触力、仿真时间。

mj_step：
MuJoCo 推进一步仿真的函数。调用它之后，模型会根据当前状态和控制输入往前走一个时间步。

qpos：
广义坐标。可以理解成模型当前的位置状态，例如关节角度、自由体位置等。

qvel：
广义速度。可以理解成模型当前的速度状态，例如关节角速度。

nq：
qpos 的长度。也就是模型有多少个广义坐标。

nv：
qvel 的长度。也就是模型有多少个广义速度。

nu：
控制输入数量。通常和 actuator 数量有关。nu 决定 action_space 应该有几维。

obs shape：
观测向量的形状。比如 obs shape 是 9，说明环境每一步给策略 9 个观测量。

### 5. Gymnasium 和强化学习相关

Gymnasium：
强化学习环境接口库。它规定环境一般要有 reset 和 step。算法通过这两个函数和环境交互。

reset：
重置环境。每个 episode 开始时调用，用来把模型放回初始状态。

step：
推进环境一步。输入 action，环境执行仿真，返回 observation、reward、terminated、truncated 等结果。

action：
动作。策略输出的控制量，比如目标力矩、目标速度或目标位置。

action_space：
动作空间。它定义 action 的维度和范围。比如一个电机通常对应一维动作。

observation：
观测。环境提供给策略看的状态信息，例如角度、角速度、电机状态。

observation_space：
观测空间。它定义 observation 的维度和范围。

reward：
奖励。强化学习用它判断动作好不好。倒立摆里常见奖励包括“越接近竖直越好”“速度不要太大”“动作不要太猛”。

terminated：
任务是否失败或自然结束。例如摆杆倒得太厉害，可以 terminated。

truncated：
不是失败，但因为时间上限等外部原因结束。例如 episode 到达最大步数。

episode：
一轮完整实验。从 reset 开始，到 terminated 或 truncated 结束。

rollout：
让策略或随机动作在环境中连续运行一段时间，收集状态、动作、奖励。random rollout 就是用随机动作先检查环境能不能跑通。

policy：
策略。它根据 observation 输出 action。可以是手写控制器，也可以是神经网络。

PPO / SAC：
常见强化学习算法。PPO 比较稳，SAC 适合连续动作控制。第一版不急着上算法，先保证环境正确。

PD 控制：
比例-微分控制。它根据位置误差和速度误差输出控制量，常用于做基础控制器。

LQR：
线性二次调节器。适合在线性化模型附近做稳定控制，可以作为强化学习前的基线。

### 6. CAN 和树莓派相关

CAN：
Controller Area Network，一种常见工业总线。很多电机驱动器通过 CAN 收命令、发反馈。

CAN H / CAN L：
CAN 总线的两根差分信号线。接反、没接好、没有共地或终端电阻不对，通信都可能失败。

终端电阻：
CAN 总线两端通常需要 120 欧姆终端电阻。没有终端电阻或电阻位置不对，容易通信不稳定。

波特率：
CAN 通信速率，例如 500k、1M。树莓派、CAN 模块、电机三者波特率必须一致。

CAN ID：
CAN 帧的标识符。电机通常用不同 ID 区分不同设备。

CAN frame：
CAN 帧。一次 CAN 通信的数据包，里面包含 ID 和最多若干字节数据。

SocketCAN：
Linux 下的 CAN 标准接口。它会把 CAN 模块变成 can0 这样的网络接口。

can0：
Linux 中的 CAN 网络接口名称。树莓派识别 CAN 模块后，通常会出现 can0。

can-utils：
Linux 下调试 CAN 的工具集合。常用命令包括 candump 和 cansend。

candump：
监听 CAN 总线的工具。用它看电机有没有发反馈。

cansend：
发送 CAN 帧的工具。用它可以手动发一帧命令测试总线。

python-can：
Python 里的 CAN 通信库。适合在树莓派上写电机控制脚本。

MIT 模式：
很多无刷电机控制器支持的一种控制模式，通常可以同时发位置、速度、力矩、刚度、阻尼等目标。具体格式要看电机协议。

力矩控制：
直接控制电机输出力矩。适合倒立摆这类控制任务，但风险也更高，需要做好限幅和急停。

速度控制：
控制电机转速。比力矩控制更容易初步测试。

位置控制：
控制电机到指定角度。适合低风险测试，但倒立摆最终不一定只靠位置控制。

急停：
紧急停止方案。真实硬件测试前必须有，比如断电、失能电机、限幅保护、物理开关。

## 常用脚本说明

scripts/setup_windows.ps1

创建 .venv 并安装依赖。适合第一次配置电脑、重建虚拟环境或换 Python 版本后使用。

scripts/verify_official_mujoco.ps1

验证官方 MuJoCo 安装是否可用。它使用官方模型，不依赖本项目 XML，所以适合判断基础环境是否正常。

scripts/check_model.ps1

验证当前配置指向的项目模型是否能被加载，并打印 nq/nv/nu/obs shape。适合每次修改 XML、配置或环境封装后运行。

scripts/random_rollout.ps1

用随机动作跑几个 episode。它不用于评价控制效果，只用于检查环境循环是否能稳定执行。

scripts/update_lab_runtime.ps1

把训练目录中已经验证过的运行时内容同步到 MuJoCo-Lab。

## 总里程碑

[ ] 基础 MuJoCo 环境验证完成
[ ] CAD 结构明确
[ ] 尺寸、质量、惯量、关节轴记录完成
[ ] MJCF 初版完成
[ ] MuJoCo viewer 校验通过
[ ] Gymnasium 环境封装通过
[ ] 随机 rollout 通过
[ ] 传统控制基线完成
[ ] 强化学习策略完成初版
[ ] 树莓派 CAN 通信打通
[ ] 电机空载控制通过
[ ] 低速带负载控制通过
[ ] 策略接入真实硬件前完成安全限幅和急停

## 官方参考

- MuJoCo Python 文档：https://mujoco.readthedocs.io/en/stable/python.html
- MuJoCo 官方模型目录：https://github.com/google-deepmind/mujoco/tree/main/model
- MuJoCo XML Reference：https://mujoco.readthedocs.io/en/stable/XMLreference.html
- python-can SocketCAN 文档：https://python-can.readthedocs.io/en/stable/interfaces/socketcan.html

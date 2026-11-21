# XGO-Duck官方套机首次部署、零点标定与飞智黑武士4Pro遥控

现场起点为整机已按官方步骤装配、Arduino App Lab已安装。依次验收**通信→舵机ID与位置→机械中位→IMU→默认姿势→官方策略行走→手柄→同条件模型对照**。本手册整理于2026-11-21，实物和手柄的检查结果填入现场记录。

仓库的11月1日上板记录显示，当时一台UNO Q已导入运动控制应用、联网、设好2Mbps Router并读到15/15舵机，但IMU初始化失败。若为同一块板，本次从**打开已有控制台、核对当前IMU和反馈**开始，按下面的结果选择后续操作。[11月1日上板与IMU排查原记录](https://github.com/FTP-LI/RL-Project/blob/b9d7f58fdde1b74e25ec82ebd78a75931ad61f91/XGO-Duck/bringup/2026-11-01/README.md)

## 1. 选定与实物匹配的程序

| 检查到的配置 | 操作入口 |
| --- | --- |
| 包装清单包含相机、USB音频和ToF，扩展板及接线与官方整套音视频配置一致 | 使用XGO Duck AI整套安装包。核对控制台构建信息和标定页。官网当前批量标定与内置蓝牙手柄教程对应XGO Duck AI 0.3.6，标定说明核对到上游提交`d66192d`。 |
| 包装清单与官方基础款一致，控制台提供逐关节标定 | 使用官方运动控制分支。使用逐关节机械中位流程，网页摇杆先完成基础验收。 |

套机名称可见于订单，扩展板版本以丝印为准，配件以装箱单和实物为准。为当日建一个记录目录`records/hardware/<日期>/`，先拍清晰的正面、侧面、板卡丝印和电源线照片。准备水平桌面、支撑躯干的标定架或稳定托架、数据用USB-C线、已充电的套机电池以及同一局域网的电脑。台架应留出腿和头部的活动空间。关节活动时手指离开夹缝。[官方整机套件与基础款](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-kit-editions) · [官方整机装配与音视频接线](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-guide-06)

## 2. 联网并启动原厂程序

1. 让Duck处于支撑状态。断开电池后核对供电与扩展板插头、绝缘和线束走向。首次配置UNO Q时，按官方联网指引用数据线接电脑，在Arduino App Lab选择UNO Q并完成Linux凭据和Wi-Fi设置。记下`Network Info`中的完整IPv4地址。常用账户为`arduino`，密码在本机交互输入。电脑和UNO Q应位于能互访的同一局域网。后续从路由器客户端列表或`Network Info`复查地址。[官方开发板联网与获取IP](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-guide-01) · [Arduino UNO Q用户手册](https://docs.arduino.cc/tutorials/uno-q/user-manual/)
2. 若板上已有程序，先打开控制页核对构建和当前模式。请求`/api/status`连续两次，确认反馈随时间更新。能正常使用的同一构建可直接复用。更换构建前记录原App的路径、默认App和安装包，按官方页面备份原程序与`data/`，使用同一台Duck的标定和运行配置。[官方软件更新与设备数据备份](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-ai-overview)

### 2.1 整套音视频配置

1. 从同一发布来源获取`xgoduck-ai-unoq-full.zip`及其`.sha256`。有指定构建时从Release选择相同标签的那一对文件，首次安装用全量包。板端要求官方Debian 13 ARM64、`arduino`用户和UID 1000。全量包约470MB，安装说明要求板上至少2.5GiB可用空间。[官方安装说明](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-guide-04) · [陆吾智能GitHub发布记录](https://github.com/LuwuDynamics/xgoduck_runtime_arduino/releases) · [官方魔搭全量包与哈希入口](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-resources)
2. 在现有可见终端打开下载目录。用下列PowerShell命令校验文件，并把文件传到UNO Q。把示例地址换成第1步记录的地址。

   ~~~powershell
   $duckAddress = '192.168.1.123'
   $duckPackage = 'xgoduck-ai-unoq-full.zip'
   $expectedHash = ((Get-Content "$duckPackage.sha256" -TotalCount 1).Trim() -split '\s+')[0]
   $actualHash = (Get-FileHash $duckPackage -Algorithm SHA256).Hash
   if ($actualHash -ine $expectedHash) { throw 'Package SHA-256 mismatch' }
   scp $duckPackage "$duckPackage.sha256" ('arduino@' + $duckAddress + ':')
   if ($LASTEXITCODE -ne 0) { throw 'Package copy failed' }
   ssh "arduino@$duckAddress"
   ~~~

3. 在UNO Q上复核哈希、空间和安装说明。用新目录展开，进入包含`INSTALL.txt`和`install.sh`的目录阅读并执行。在此之前先支撑整机、预留卸力空间、确保原App已停止。安装脚本会配置并启动官方App，首次过程可能上传官方STM32程序。安装目标目录已存在时，按官方文档核对其安装收据后复用。将执行时间、命令、构建信息、退出码和错误写入当日记录。

   ~~~bash
   cd "$HOME" &&
     sha256sum -c xgoduck-ai-unoq-full.zip.sha256 &&
     df -h "$HOME" /var/lib/arduino-app-cli
   ~~~

   哈希匹配并确认有可用空间后，展开安装包并查找说明。

   ~~~bash
   unzip xgoduck-ai-unoq-full.zip -d xgoduck-first-install &&
     find xgoduck-first-install -name INSTALL.txt -print
   ~~~

   进入包含`INSTALL.txt`的目录，先读完包校验、安装路径和已有版本的处理说明，再执行安装。

   ~~~bash
   cat INSTALL.txt
   bash install.sh
   ~~~

4. 在电脑浏览器打开`http://<板子IP>:9527/`查看控制页，`http://<板子IP>/`查看驾驶舱。可用`http://<板子IP>:9527/api/status`核对反馈。记下版本与构建标识、所选官方模型、当前模式、舵机在线数和IMU状态。进入后续步骤前，整机应稳定显示15/15在线和IMU在线。[官方安装并启动XGO Duck AI](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-guide-04)

### 2.2 基础款运动控制

1. 选定并记下官方运动控制分支的提交。用Arduino App Lab导入或打开含`app.yaml`的整套App工程，在实际UNO Q上先运行以完成板端依赖获取，然后停止App。
2. 在UNO Q的工程目录运行首装准备，回到Arduino App Lab重新启动App。板端Linux与STM32的Router链路为2Mbps，外部舵机总线配置按扩展板与舵机说明分别核对。

   ~~~bash
   python3 tools/prepare-bridge.py
   sudo bash tools/router-2m.sh
   ~~~

3. 打开`http://<板子IP>:9527/`，验收`Inference only`、舵机15/15和IMU在线。如果机器已经安装了同一套依赖，先复用原App。基础款的标定以该App实际显示的菜单、机械参考和所用构建的说明为准。[官方运动控制分支说明](https://github.com/LuwuDynamics/xgoduck_runtime_arduino/blob/master/README.md) · [Arduino App Lab导入与管理应用](https://docs.arduino.cc/software/app-lab/apps/manage-apps)

通信检查可在Windows运行`Test-NetConnection -ComputerName <板子IP> -Port 9527`，将尖括号及其中的文字换成真实地址。浏览器能到达控制页但舵机读数不足时，从电池和扩展板供电、舵机线束、重复ID三项排查。控制台无法访问时，先核对当前IP、同网段访问和App运行日志。

### 2.3 已有控制台显示IMU离线

保持卸力，记录板卡丝印、App构建、当前状态页与IMU错误日志。关电后确认UNO Q与扩展板插接及扩展板供电。历史记录在D20/D21的`0x6A`地址读到芯片标识`0x05`，当时失败点是原程序的复位结果检查。若此机复现同样现象，将现行App、板卡和故障记录交给陆吾核对该硬件修订对应的官方驱动与初始化流程。IMU在线且静止读数持续更新，才进入机械检查后的IMU标定与上力。[历史I²C观测与版本记录](https://github.com/FTP-LI/RL-Project/blob/b9d7f58fdde1b74e25ec82ebd78a75931ad61f91/XGO-Duck/bringup/2026-11-01/README.md#imu-排查结果与限制)

## 3. 确认舵机ID和机械对应

两条腿的左右均按Duck自身视角记录。用`Servo setup`扫描、读取，再逐个对照实物贴纸、接线与安装位。套机已经正确预置ID时，记录核对结果即可。

| 同一轴向 | Duck左腿ID | Duck右腿ID |
| --- | ---: | ---: |
| 髋关节偏航 | 10 | 20 |
| 髋关节横滚 | 11 | 21 |
| 髋关节俯仰 | 12 | 22 |
| 膝关节 | 13 | 23 |
| 踝关节 | 14 | 24 |

| 头颈部位 | ID |
| --- | ---: |
| 颈部俯仰 | 30 |
| 头部俯仰 | 31 |
| 头部偏航 | 32 |
| 头部横滚 | 33 |
| 鸭嘴 | 34 |

发现重复、缺失或位置错配时，先卸力、关电，再只接待修正的一台。上电扫描旧ID，输入目标ID，写入后重新扫描读取；重新接回整机前再次关电。参数从与套件对应的官方说明读取。当前整套音视频教程在15台在线且卸力时核对运行增益`KP=5`、`KD=10`，过流设置同时对照该构建的自检提示、实际供电和舵机读回。把修正前后均记录在[逐舵机记录表](../../configs/hardware/commissioning_record_template.csv)。[官方舵机编号、单舵机检查与保护设置](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-guide-05)

换接线的位置后，用上表验收总线ID与安装位。更换物理舵机时，给新舵机写回对应ID，重新核对方向与中位。若确实更改逻辑ID或策略关节顺序，连同观测的关节位置、速度和上一动作，以及输出、默认角、增益、限幅与伺服方向一起核对。最省工的方式是恢复官方逻辑ID，再由台架逐轴验证。[官方运动控制运行库](https://github.com/LuwuDynamics/xgoduck_runtime_arduino) · [本工程策略接口与部署方案](../rl_deployment_plan.md)

## 4. 写入机械中位，再标定惯性测量单元

### 4.1 整套音视频配置的批量标定

1. 支撑躯干后，打开`Calibration`。确认页面显示全部卸力且15台都有新反馈。点开官方图1至图4，把参考照片放在手边。
2. 按图1和图2将颈部后仰、头部贴向颈部、轻轻合嘴。按图3让两腿同时闭合，按图4将双腿同时转到躯干后侧的参考位置。线束拨开，关节到达参考位置即可，保持躯干和姿态由托架支撑。
3. 手离开关节，点击`标定15个舵机`，等待页面确认全部中位保存。这一流程包括30号颈部和34号鸭嘴。记录结果并拍摄参考姿态和保存提示。批次出现部分写入时，查阅逐台状态，摆回同一参考姿态后统一复核。
4. 保持卸力，移到同一页面的IMU区。对照官方实物图，**主板直立、电池接口朝上、主板下边缘贴水平桌面**。点击`标定IMU`后保持约1.5秒静止，保持当前标签页在前台，等到显示已保存。在站姿检查前恢复正常的支撑方向。[官方四张摆位照片、IMU参考图及验收](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-guide-07)

### 4.2 基础款的逐关节标定

确认该版本的逐关节机械参考后，打开`Calibration`并确认卸力。一次把一个关节放到参考中位，点击该关节的`Set center & hold`，检查它保持在正确位置，之后确认卸力再调整下一个关节。`Servo setup`中的`Reset Mid`会写入选中关节并保持该关节卸力，适合按说明逐个设置。标定结果与各轴参考截图一起保存。若控制台没有IMU标定区，从该构建对应说明确认安装方向和偏置处理，验收静态反馈后再上力。机械中位写入和模型默认姿势分别记录。[官方运动控制分支的逐关节标定](https://github.com/LuwuDynamics/xgoduck_runtime_arduino/blob/master/README.md#calibration)

## 5. 用官方权重完成第一次站立和地面走停

1. **台架默认姿势。** 扶稳机身，在实际版本的`Default pose`入口上力。观察头颈、膝踝、两脚方向、结构限位、异响和电压；从正面和侧面拍照。姿势异常时先托住躯干再卸力，依次复查ID、机械中位、连杆和IMU。`Zero motion`把运动指令清零并保持扭矩。`Torque off`释放扭矩，操作前应托稳。
2. **官方网页低速验证。** 确认通信稳定，移除连接电脑的USB-C线，按套件装好后部保护壳。在平整、开阔、有抓地力的地面启动`Walk / get up`。在保持支撑可及的范围内先站立，再用网页摇杆逐一给出小幅前、后、左移、右移和左右转向。每次输入之后回中，观察是否停止、是否方向一致。确认默认姿势与低速走停后，再进入手柄测试。
3. **留档。** 第一轮按动作分别记录，建议站立10秒，六个方向各3次短输入与回中，每次只测一个方向。用同一地面与初始朝向。记录电池电压的测试前后值、官方模型文件SHA-256、是否换过脚垫与保护壳、控制模式和接触异常。视频从通电或开始动作前的静止段录到完全停止，机身和脚须入镜。起身和前滚翻留到单项台架及地面保护验证，录全过程。[官方第一次行走和控制台按钮](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-guide-07)

## 6. 接入飞智黑武士4Pro

### 6.1 首选板端蓝牙，先验证输入

当前整套音视频应用支持一部分支持蓝牙的Xbox和Switch Pro类手柄。黑武士4Pro说明书列出蓝牙模式可用`Xbox Wireless Controller`名称配对，型号在XGO兼容页中未单独列出。设备名称用于发现，板端`输入就绪`并收到正确的按键、轴和释放事件才是此组合的兼容证据。[官方蓝牙手柄连接与按键](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-ai-gamepad) · [飞智黑武士4Pro原厂说明书](https://fccid.io/m/80e4b76d6df46fe7d104ebbc8b6be18dd07c5cf3c1452f70c55d8a9484f868e9.pdf)

1. 板端保持卸力。从驾驶舱进入`设置→手柄`，或打开`http://<板子IP>:9527/gamepad.html`。让手柄与旧主机暂停回连，将手柄背部开关切到蓝牙档，依飞智手册长按Home约3秒至指示灯快闪。
2. 点`扫描手柄`，按实际扫描名选择并点`配对并连接`。等待`输入就绪`。两个摇杆回中且松开所有按键。按键绑定页可在手柄控制关闭时学习单个键，记录实际识别的位置。摇杆在启用后做输入与运动检查。若蓝牙配对后没有输入，可在支持匹配驱动的UNO Q上验证手柄的`NS`档。驱动需与UNO Q当前内核匹配，设备和输入均需重测。
3. 确认网页停止按钮可及，机身有支撑后才点`启用手柄控制`。上游默认映射如下。先核对**实体按键位置**，尤其是不同模式下外壳A/B标记与发送键值的区别。

   | 默认操控输入 | 运行作用 |
   | --- | --- |
   | 左摇杆纵向／横向 | 前后／侧移 |
   | 右摇杆横向 | 左右转向 |
   | Menu或Start | 进入行走／起身 |
   | 底部面键（Xbox布局的A） | 默认姿势，上力 |
   | 右侧面键（Xbox布局的B） | 舵机卸力，先托稳 |

4. 托稳机身先验证卸力按键和网页`Torque off`，再依第5节小幅验证方向和回中。先在托架上并清空腿部行程做断联试验，记录手柄关闭后的零指令、卸力与控制权状态，再验证重新连接仍需显式启用。应用原生手柄断线、输入过期会解除控制并卸力。浏览器摇杆在关闭手柄控制后恢复。现场以实测记录这两项。首次验收仅保留默认映射，特殊动作单独验收。[官方手柄控制权与断线处理](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-ai-gamepad)

### 6.2 板端没有手柄输入时，用电脑核验和接管

1. 把飞智原装2.4GHz接收器插到**Windows电脑**，手柄切换到PC接收器档并唤醒。也可用USB数据线直连。按`Win+R`运行`joy.cpl`，选择手柄后进入`属性→测试`。记录三条目标轴的中心、正负方向、端点和实体按钮编号。摇杆回中有明显漂移时先在飞智空间站校准，初次只使用一台控制电脑。[飞智黑武士4Pro原厂说明书](https://fccid.io/m/80e4b76d6df46fe7d104ebbc8b6be18dd07c5cf3c1452f70c55d8a9484f868e9.pdf) · [黑武士4Pro连接电脑实拍](https://www.youtube.com/watch?v=lzEeEOFgS5U)
2. 基础款的已知可用入口是官方网页，键盘`W/S`前后、`A/D`侧移、`Q/E`转向、空格卸力。空间站能把按钮映射为键鼠键值。若使用这条路径，先在本地文本窗口验证即按即发、松键结束。在机器人控制网页上再托架验证回中、失焦和手柄断开是否都清零。首次只用独立方向键映射，保留网页卸力按钮。模拟量摇杆和失联保护经专用桥接单独验证后再用于落地。键盘映射只是在Windows与浏览器之间传递按键。[官方控制页面使用说明](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-guide-07) · [官方运动控制分支网页源码](https://github.com/LuwuDynamics/xgoduck_runtime_arduino/blob/master/assets/app.js)
3. 若后续开发电脑桥接，按`前进速度、侧移速度、转向角速度`建立映射。在卸力模式测试摇杆死区、方向、限速、回中和显式使能。验收断手柄、应用退出、Windows与UNO Q网络中断三种情形，三种情形均由**板端超时**清零，重新连接均要求显式使能。先保留原厂网页遥控作为恢复入口，再验证桥接。

## 7. 模型替换与原厂回退

官方站立与遥控验收完成后，先留一套原厂ONNX及其SHA-256、原App构建与标定记录，将候选运行于同一实机的独立App或独立模型槽位。把标定、轨迹与模型版本分别记录，同一A/B试验固定电源、地面、控制周期和指令幅值。

1. 先查候选是哪类策略、训练使用的观测和输出顺序、角度单位、上下限、默认姿势、前一动作和IMU坐标。分别对照**板端运行库实际接口**与**候选导出契约**，包含14个策略关节与独立鸭嘴。对于涉及上层路由、控制反馈、目标插值或多级物理步长的组合，另列板端对应实现和仿真参数。替换单个ONNX只复现这一个策略。
2. 先记录30秒卸力推理的输出形状、有限值、动作边界、全关节方向和推理周期，再进行有支撑的默认站姿试验。同一天对比原厂和候选的站姿、低速前进、后退、转向及侧移。之后单独安排起身与前滚翻。模拟评估的42.5秒连续路线及其权重来源见交付清单。
3. 回退时先扶稳躯干并卸力，停止当前App，切回备份的原厂App和正确的官方模型，验证其构建及哈希，再重复15/15、IMU、默认姿势和单次走停。保持原有机械中位和同一机器的数据，恢复记录写入当日目录。[官方导出与部署说明](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-reinforcement-learning) · [本工程仿真模型与可恢复保存点](../../artifacts/delivery/nominal-continuous-simulation.json)

## 8. 现场记录与参考视频

| 证据 | 建议保存 |
| --- | --- |
| 控制台与实物 | 板卡丝印、套件类别、原厂Git提交或构建、状态页、峰值与测试前后电压 |
| 零点与惯性测量单元 | 逐关节ID和方向、中位写入结果、机械参考姿势照片、IMU安装方向和保存提示 |
| 台架和地面走停 | 正侧视站姿与六个方向、回中停步、测试前后脚位、异常片段与出现时间 |
| 飞智黑武士4Pro | 固件和连接档、扫描名称、三个轴、卸力键、断联和恢复、是否需要手动启用 |
| 模型与恢复 | 官方与候选的SHA-256、运行条件、每项成功次数、原App恢复后走停 |

1. **XGO-Duck装配实拍。** 对照头颈、腿部、线束和舵机位置。零点按钮和IMU摆位以官方标定页的当前照片为操作参照。[陆吾智能装配视频](https://youtu.be/nElGguHO5dY) · [装配视频备选播放页](https://www.reddit.com/r/arduino/comments/1x6z98s/how_to_build_xgoduck/) · [官方中位与IMU标定照片](https://wiki.xgorobot.com/xgo/kb/xgo-duck/duck-guide-07)
2. **飞智黑武士4Pro连接电脑。** 0:08是档位位置，0:18至0:28是蓝牙配对，0:33至0:45是2.4GHz接收器接入。用于定位实体开关和Windows配对流程。[《Flydigi Vader4 Pro – How to Pair with PC》](https://www.youtube.com/watch?v=lzEeEOFgS5U)
3. **飞智黑武士4Pro连接档切换。** 对照PC接收器、蓝牙与`NS`三档，板端接口仍以控制页实测为准。[《Flydigi Vader4 Pro – How to Switch Connection Mode》](https://www.youtube.com/watch?v=DyZZXFQl-X4)
4. **官方实机走姿及起身。** 用于比较机身前倾、脚步节奏和起身时机。真实评估采用第5节的固定条件和完整原片。[陆吾智能产品页的实机影片](https://dev-web.xgorobot.com/en/duck.html) · [XGO-Duck硬件、板端与仿真实拍](https://www.reddit.com/r/arduino/comments/1wsf9ew/we_adapted_microduck_for_arduino_uno_qheres_a/)

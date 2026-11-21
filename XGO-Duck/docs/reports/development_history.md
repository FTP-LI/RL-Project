# XGO-Duck 完整过程记录

本页保留历次开发记录。下文的“当前”“待执行”和暂停/启动描述均对应记录当时；最新操作与状态见[主报告](development_report.md)。

2026-11-21晚，用户暂停训练与仿真。四路首次承重前的足底和质心复算完成，累计重建5,304个状态，双方足底接触与原结果一致5,300个。360ms和420ms有界回放的最大转正分别为0.70°和1.99°。在完整轨迹内重配后，第一遍4ms参照的正向配对13/15，父策略10/13。本轮之后的工作为归档和真机准备，独立确认种子2246保留。[低摩擦复算与逐例结果](../../records/reports/nominal-skill-composition/low-friction-forward-support/pre-support-geometry/README.md) · [阶段移交](../../records/handoff/2026-11-21/README.md)

2026-11-21，4ms采集四组共448,000条转移、2,500次批量求解，恢复346项状态。本轮候选的场景439和779在头部卸载确认后64ms、76ms重新承重。首次离线核验改用原始求解器字段后逐值通过，四段4秒机制原片均已解码并核对逐帧关节姿态。随后用34个初态做6,800个CPU物理步，颈部和髋膝两种调整的最大目标转正增益为0.30°。[采集、运行观察、图及回放](../../records/reports/nominal-skill-composition/low-friction-forward-support/README.md)

2026-11-21，离线复算同释放动作对照三组及此前同释放同权重的两遍4ms参照。两遍参照均有123/128个前进场景准入，场景283和727的逐例判定交换。前进场景439和779在五组均未准入，候选439在两脚接地且头部离地后100ms重新确认头部接地。新增物理步和学习更新均为0。[逐场景时序及来源](../../records/reports/nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/action-comparison/forward-getup-readback.json)

2026-11-21，右脚承重后的转正奖励完成实时采集、配对训练和同释放动作比较。首轮恢复两足0.8摩擦核对，随后按CUDA浮点顺序复算归档奖励。三档短测取3,072并行场景，两组各完成17次更新、1,253,376条场景转移。原奖励组和候选均导出并核对ONNX及可恢复保存点。种子1257动作对照的准入为880、882和881/896；候选相对原奖励组前进净变−1例。两组训练视频、三组动作视频、采样图、关键帧与原始配对结果一并归档。[完整对照记录](../../records/reports/nominal-skill-composition/low-friction-early-transfer-authority/loaded-upright-ppo/README.md)

2026-11-21，400–480ms固定状态核验19种扭矩调整，2,184次求解的104个状态中有103个与原CUDA承重分类一致。通过右脚接近地面的几何检查后，对左髋横滚候选做26个同初态短回放，目标右脚承重最多提前4ms，原准入保护通过11/13。右脚连续承重后80ms的双足承重转正量通过13对归档的指标筛查，12对为正，配对中位差4.76°。该指标在本机CUDA上核对896个场景的246,400条归档样本。[冻结方案、逐场景与GPU读回](../../records/reports/nominal-skill-composition/low-friction-early-transfer-authority/README.md)

2026-11-21，从4ms载荷归档复算四个低摩擦持续失分的左转场景。释放后400–480ms的四个配对均为原准入场景转正量较大，中位差10.00°；左足减右足的法向冲量份额中位差为−13.27个百分点。结果文件登记错误通过两次原数组复算恢复，新增物理步为0。[较早承重交接和恢复记录](../../records/reports/nominal-skill-composition/low-friction-right-support-trial/support-timing/README.md#较早的承重交接)

2026-11-21，复算低摩擦第136–156步的右脚前移。三个持续失分场景在两遍原策略都存在几何可行位姿，所需连续关节执行读数为5，位姿当时只剩1或2个端点。四个左转失分场景均未达到原动作幅度所需的时长，后续核验较早的承重交接。新增物理步和学习更新均为0。[几何、时长与逐场景结果](../../records/reports/nominal-skill-composition/low-friction-right-support-trial/support-timing/README.md#支撑窗口与关节限位)

2026-11-21，种子2258冻结三组供电及时延对照。首次CUDA零步核验保留了原始数组，随后按供电比例修正初力矩预期。标称、7.4V和8ms三组完成4,816,000条场景转移。标称在第1125步的右转入口以121/128结束；两个单因素组跑满第2125步，完整通过881/896和891/896。三个工况实际4ms或8ms舵机时延、带载电压、状态文件和三段原片已逐项核对。[方案、复算、VS Code过程和原片](../../records/reports/nominal-skill-composition/closed-loop-voltage-delay/README.md)

2026-11-21，按20ms接触、姿态和脚部位置复算低摩擦右髋膝对照。新增准入场景的三组均于第139步确认双足接地，候选到第156步确认转正、双足接地及头部离地。到第150步，候选新增准入场景右脚前移32.39mm，8个持续失分场景右脚前移中位数为2.65mm。其余8个场景沿用原开发数据继续评估右脚动作窗口。[双足支撑和位移量测](../../records/reports/nominal-skill-composition/low-friction-right-support-trial/support-timing/README.md)

2026-11-21，按4ms延迟回放实际右髋膝目标，0.02rad通过四条归档与两端供电的最小幅度门槛。随后在共同第136步快照开展三组12秒闭环短测，共1,369,088条场景转移。21个触发场景中20个达到双关节扭矩方向和幅度要求。滚翻前准入884、884、885/896，新增站立分组场景218，左转分组各124/128。[固件复算](../../records/reports/nominal-skill-composition/low-friction-right-support-position/README.md) · [三组闭环动作、舵机读数及原片](../../records/reports/nominal-skill-composition/low-friction-right-support-trial/README.md)

2026-11-21，种子2257在第608步共用快照恢复三组标称连续路线。核对346项状态，左右组向885个相同场景施加0.08m/s横向速度增量。无扰动、左向和右向完整路线各通过874、867和873/896。三组共完成4,622,464条场景转移，保留各42.5秒原片及真实进度曲线。相邻的低摩擦工作核验了第136步姿态观测，保留右髋和右膝位置目标的后续检查。[滚翻横向扰动评分](../../records/reports/nominal-skill-composition/closed-loop-roll-lateral-disturbance/README.md) · [早段姿态观测](../../records/reports/nominal-skill-composition/low-friction-right-support-signal/README.md)

2026-11-21，复算0.8足底滑动摩擦下13对实际仰卧场景的髋膝扭矩响应。采用52个保存状态，在每个状态计算两次原扭矩和一次候选扭矩。13个失分场景均在至少一个时点达到转正加速度门槛，成功场景两个时点均满足保护要求的为8/13。新增物理步及学习更新均为0。[髋膝方向、接触响应与可复算日志](../../records/reports/nominal-skill-composition/low-friction-hip-knee-authority/README.md)

2026-11-21，用4ms实际扭矩与载荷复算13对仰卧轨迹。在7对反复失分轨迹中，首次持续双足承重前160–80ms的转正量配对中位差为−14.24°。同期颈头俯仰实际扭矩积分增大，双髋及双膝减小，作为后续关节方向和接触响应预检的起点。[分组、量测与可复现命令](../../records/reports/nominal-skill-composition/low-friction-first-support-effort/README.md)

2026-11-21，复算七批同种子低摩擦数据，85次实际仰卧在滚翻开始前失分。随后从同一第100步释放快照运行两遍4ms和一遍起身首1.2秒2ms，共2,844,800条固定权重仿真转移。滚翻完成且落地持稳855、859和859/896，候选前进入口122/128、左转119/128。保存三段至22.50秒入口边界的原片、关键帧和VS Code观察文件。评分按实际32位浮点精度核对积分时钟。[归档复算](../../records/reports/nominal-skill-composition/low-friction-failure-recurrence/README.md) · [动作、时钟和原片](../../records/reports/nominal-skill-composition/low-friction-release-integrator/README.md)

2026-11-21，将承重交接奖励接入训练采样器并核对十次实际4ms求解。3,072场景短测完成150步，3,328场景在第11步达到显存上限。两组各完成17次更新与1,253,376条场景转移，整卡峰值同为13,185MiB。随后从同一释放快照比较父策略、原奖励及候选，滚翻前准入依次为885、878和880/896。候选留作实验权重。[训练、容量、对照、原片与关键帧](../../records/reports/nominal-skill-composition/low-friction-support-effort/support-credit-ppo/README.md)

2026-11-21，从低摩擦CUDA初态复采896个场景至第600步，共537,600条转移，1,008,000条场景子步含4ms载荷及舵机读数。记录13个实际仰卧未准入场景在第7–8秒的头足分载，并核验固件和求解器限幅。修正首轮输入限幅检查及轮次复算中的四元数表示，保留两次日志和重新核验的哈希。承重交接奖励在三批归档达到预检门槛，GPU读数随后在初态、算例和五组带载接触记录上通过核验。[运行记录、12秒原片和逐对预检](../../records/reports/nominal-skill-composition/low-friction-support-effort/README.md)

2026-11-21，复用同一腿部模板与触发条件，把交还原策略由第550步提前到第450步。两遍原策略及候选从新的共同第250步快照开始，各评价至第600步，总计1,164,800条场景转移。两遍参照和候选滚翻前准入分别为878、878和879/896，左转分组均为123/128。阶段原片及实际关节跟踪误差已独立核验。[提前交还记录、原片及关键帧](../../records/reports/nominal-skill-composition/low-friction-upright-early-return/README.md)

2026-11-21，重建两条低摩擦轨迹的29个前倾姿态，验证其中25个存在只调腿部的12°几何解。随后从第250步共同CUDA快照运行两遍原策略与一遍观测触发候选，合计1,164,800条场景转移。15个候选调整场景均未达到原第600步准入，三组总体各881/896。3个候选在第550步交还后抬头，提供提前交还的试验依据。[几何模型、闭环评分与原片](../../records/reports/nominal-skill-composition/low-friction-upright-recovery/README.md)

2026-11-21，低摩擦滚翻早段步长对照恢复了同一第600步快照，核对346项状态。两遍2ms和首1.20秒的1ms策略分别通过853、862、870/896个滚翻及落地场景。三组共完成1,948,800条场景转移、18,225次批量求解。各组22.50秒原片及采样指标已存。读回候选左转失分，6例来自共同起身前倾接地，2例来自滚翻有效支撑转角。[原始评分、运行与复盘](../../records/reports/nominal-skill-composition/closed-loop-low-friction-roll-window/README.md)

2026-11-21，重新核验仰卧起身站稳带奖励配对的两组训练及标称动作评分。在0.8足底滑动摩擦下，三分支从同一第100步快照恢复，共采集1,433,600条场景转移。仰卧父策略、原奖励组、候选的滚翻前准入为880、882和881/896，前进分组均125/128，左转分组为123、124、122/128。保存原命令、采样曲线及三段12秒原片。[低摩擦逐项对照与训练回看](../../records/reports/nominal-skill-composition/early-getup-load/support-progress/low-friction-action-comparison/README.md)

2026-11-21，低摩擦滚翻时钟短测使用16个归档场景和5份共用电机输入，分别以2ms与1ms求解，完成30次批量求解。首次冻结时读取函数签名不一致，发生在模拟器创建之前。修正后冻结配置和两组实测时钟均完成。[时钟核验与全过程日志](../../records/reports/nominal-skill-composition/low-friction-roll-clock/README.md)

2026-11-21，完成2ms连续滚翻同释放快照重复评估。两遍共3,718,400条场景转移。恢复时核对346项状态，首步输入、目标和延迟电机输入逐位相同。两遍滚翻并落地分别873/896和874/896，最终完整动作均873/896，七项逐项过线。30个场景的最终通过状态互换。两段42.5秒原片各1,064帧。[重复评估评分、进度与录像](../../records/reports/nominal-skill-composition/closed-loop-timestep-repeat/README.md)

2026-11-21，保留种子2256完整运行42.5秒，四种实际释放姿态按时起身并保持共888/896，滚翻落地及后续完整路线872/896，七项指令逐项满足原门槛。权重保持与两遍开发评估相同，实测1,904,000条转移。已导出标称工况的七策略与五保存点清单。种子2246保留。[独立确认及原片](../../records/reports/nominal-skill-composition/closed-loop-timestep-confirmation/README.md)

2026-11-21，种子1257的低摩擦评估仅更改左右足滑动摩擦为0.8，采集1,008,000条场景转移。前进120/128、左转121/128，按原入口条件在22.50秒停止。前进5个有效支撑转角失分，左转6个滚翻前失分，均已用原轨迹复算。第一次低摩擦零步的随机状态检查及隔离加载修复均已留档。[初态日志](../../records/reports/nominal-skill-composition/closed-loop-timestep-low-friction-preflight/README.md) · [低摩擦评分和原片](../../records/reports/nominal-skill-composition/closed-loop-timestep-low-friction-development/README.md)

2026-11-21，完成仰卧起身站稳带奖励配对，各3,072个并行环境、17次更新及1,253,376条训练转移。候选只在释放后480–820ms添加0.35倍的有界姿态变化量。首次对照组因观察快照的元组格式在24步后停止，保留73,728条转移及部分保存点。首次候选启动在授权检查处结束，记录的物理步及更新均为零。完整原奖励组与候选的初始、第8次和最终保存点、ONNX及8秒原片均复核。两组实际训练释放不逐位相同，后续使用共同物理快照计分。[训练过程、资源测量与原片](../../records/reports/nominal-skill-composition/early-getup-load/support-progress/README.md)

2026-11-21，完成第100步释放后的三分支动作对照，共1,433,600条评估转移。种子1257的父策略、原奖励组、站稳带奖励候选组滚翻前准入为877/896、882/896、884/896，前进分组为122/128、124/128、125/128。候选相对原奖励前进分组净增1个，晋级要求为2个。首次复算因归档读取出错结束，改用单独脚本从原轨迹复算。三份12秒原片和标注关键帧完成核验。[同释放对照与评分](../../records/reports/nominal-skill-composition/early-getup-load/support-progress/README.md#同释放快照对照)

同日从保存轨迹核验361个共用策略场景的首步输入、目标与第一个延迟力矩。场景9在第101步已出现物理状态差异，随后观察和动作也有差异。第600步候选与原奖励组的旋转距离中位数为0.85°，43个场景超过5°。复算新增物理步和学习更新均为零。[轨迹复算](../../records/reports/nominal-skill-composition/early-getup-load/support-progress/action-comparison/unchanged-route-review.json)

同日对账旧2ms滚翻评估的5个仰卧起身失分场景。初值和实际释放类别一致，本轮仰卧父策略、原奖励组与候选分别有1、1、2个通过滚翻前准入。[跨批次对账](../../records/reports/nominal-skill-composition/early-getup-load/support-progress/action-comparison/timestep-forward-readiness.json)

2026-11-21，完成种子1256与1257的仰卧起身早段4ms载荷补采，共1,075,200条仿真转移、645,120条场景子步载荷记录。两批滚翻前站稳886/896与879/896，按实际仰卧释放分组，两批未准入组在持续双足承重确认时的机身倾角中位数为26.18°与25.06°。校正前的两次入口检查都结束于零步。最终保存逐场景时序、采样曲线和两段12秒原片。[方案、数据与录像](../../records/reports/nominal-skill-composition/early-getup-load/README.md)

同日按释放姿态与转正速度匹配6对和14对归档仰卧场景。未准入组从首次持续承重前80ms到后80ms的倾角改善量分别比同种子准入参照少11.56°和9.83°，均为配对中位数。[配对定义与读回](../../records/reports/nominal-skill-composition/early-getup-load/rotation-aligned-to-load.json)

2026-11-21，核对三份原策略轨迹从第100步释放开始的足地接触。实际仰卧场景数513、535和533，至第108步全部至少读到一次足部接触。第108–120步的预定姿态读数因此未进入首次接触前样本。[姿态筛查与接触时序凭据](../../records/reports/nominal-skill-composition/supine-pre-bilateral-support/README.md#释放后首次足部接触)

2026-11-21，完成仰卧起身足部实际响应诊断，共1,378,048条仿真转移。第131步恢复同一CUDA状态的两条原策略和前移候选各880/896通过第600步滚翻前站稳判据。57个场景实际改变目标，40个场景有共同离地读数，1个场景前移比两条原策略的重复差多出至少1mm。三段12秒实录和场景439的同步关键帧已核对。[执行方案、逐场景复算与VS Code观察](../../records/reports/nominal-skill-composition/supine-pre-bilateral-support/README.md)

2026-11-21，按原轨迹核对双足确认前的倾角、足底接触和迈脚目标。第130步倾角在两份种子1257的提前覆盖均为66.7%。第122步的未准入场景均超出姿态范围。逐步检查第122–160步后，三份轨迹中分别有7/7、11/12和9/9的原未准入场景曾出现几何机会。种子1256选出的第135步在种子1257两份轨迹的前进分组覆盖3/4和1/2。复算使用已有轨迹，新增仿真与学习均为零。[三份轨迹与复算命令](../../records/reports/nominal-skill-composition/supine-pre-bilateral-support/README.md)

2026-11-21，两批1,048个实际仰卧释放的双足接地对齐复算完成。后续同快照策略接手短测共967,680条仿真转移，原起身和接手组滚翻准入分别为885/896和884/896，实际接手18场景，唯一新增准入者接手计数为0。两段12秒原片与逐场景动作、关节目标、足地支撑已核验。[早段支撑与策略接手](../../records/reports/nominal-skill-composition/supine-early-support/README.md)

2026-11-21，完成仰卧起身末段同状态对照，共896,000条仿真转移。原起身、颈部归位、颈部归位后调整腿部三组均为881/896通过原滚翻前准入。联合组10/12个触发场景在调整腿部前失去双足接触。3段12秒原片、场景439的对照关键帧和足下接触几何复算已核验。[仰卧起身末段姿态对照](../../records/reports/nominal-skill-composition/supine-terminal-stance/README.md)

2026-11-21，以种子1257的896个匹配场景完成闭环积分步长对照。共1,948,800条仿真转移及13,125次批量求解。三组共用权重和第600步快照，2ms候选只调整滚翻段积分。两条4ms参照与候选滚翻并落地分别为855/896、851/896和872/896，候选前进准入120/128。5个未进入滚翻的仰卧场景在共同前段末25步均因倾角失分。三段22.50秒原片、关键帧、时钟、配对失分和算力开销已核验。[闭环对照报告](../../records/reports/nominal-skill-composition/closed-loop-timestep/README.md)

2026-11-21，完成标称共同场景首轮连续评估。首次CUDA零步因方向预测器加载消耗CPU随机数退出，隔离初始化后通过实际CUDA复核。896场景执行1,125步、1,008,000条转移，起身与末段持稳886/896、滚翻865/896。后退和左转入口分别为118/128和121/128，采样止于22.50秒指令下发前。复算确认10个场景未进入滚翻，21个有效支撑转角不足，原片564帧均与同一场景的控制时刻核对。[运行、原片和复算](../../records/reports/nominal-skill-composition/README.md)

2026-11-21，按足部连续3个4ms子步承重的定义复算1,536条已有轨迹。两种改动各两个种子，在全部场景的头足接棒间隔上配对中位变化均为0ms。配对失分场景的中位变化介于−2和+4ms。保留低摩擦原门槛，下一项准备标称共同场景的连续技能验收。[冻结口径、结果与日志](../../records/reports/roll-forward-input-integration/endpoint-progress-training/README.md#头部与足部支撑接棒)

2026-11-21，完成同快照的观测相位颈部对照，共358,912条仿真转移。种子1247的两次原策略参照及候选为113、117、108/128，种子1248为114、115、115/128。候选物理颈部目标变化通过独立读回。33个候选失分场景均为有效支撑转角不足，4ms头部承重时长全场景配对中位数没有延长。六段完整10秒原片和标注关键帧已保存。[逐场景评分](../../records/reports/roll-forward-input-integration/endpoint-progress-training/observed-neck-support/current-evaluation.md) · [真实载荷与录像复算](../../records/reports/roll-forward-input-integration/endpoint-progress-training/README.md#按观测相位调节颈部目标)

2026-11-21，使用两种提前量对比头部和颈部俯仰目标。1,045条原力矩在延迟、限速和电压降重放中复核，最大绝对误差1.89×10⁻⁷N·m。候选配对失分组中，颈部俯仰上调0.02rad同时缩小静态头部间隙并产生向下无约束法向加速度的记录为37/45和40/48。触发改用当前投影重力和俯仰角速度。[关节目标复算与后续对照](../../records/reports/roll-forward-input-integration/endpoint-progress-training/README.md#关节目标预检)

2026-11-21，逐条复算768条滚翻轨迹，定位24个配对失分场景在180°–270°的端点离地扣分。随后在相同站立快照补测4ms接触与力矩。初次因计数回调错误在49步后退出，保留6,272条真实转移。修复后测完358,912条仿真转移、142,080条场景子步数据。新测11和13个配对失分场景内，未计分正向角度中的头部承重比例为85.57%和87.54%。两个种子各有1个配对失分场景与上次记录重合。[分阶段复算、载荷与录像](../../records/reports/roll-forward-input-integration/endpoint-progress-training/README.md)

2026-11-21，滚翻末端接触奖励候选完成50次更新、5,529,600条训练转移。候选和原奖励对照共用4,608并行场景、父保存点和低摩擦抽样。两个开发种子共358,912条动作评价转移。种子1247为119、116、111/128，种子1248为110、114、112/128，依次为两次原奖励自训参照和末端接触奖励候选。候选未达到开发门槛，权重、保存点、两段7秒训练原片及六段10秒动作原片已归档。[修改和正式训练](../../records/reports/roll-forward-input-integration/endpoint-progress-training/README.md) · [配对支撑转角复算](../../records/reports/roll-forward-input-integration/endpoint-progress-training/action-evaluation/paired-roll-support-analysis.json)

2026-11-21，低摩擦滚翻保存点对照完成358,912条仿真转移。两次第50次更新保存点参照及第30次更新保存点候选，在种子1247分别为116、113、121/128，在种子1248分别为117、111、111/128。第30次保存点通过首组门槛，独立留出继续保留。随后用种子1248的另一批固定轨迹复算加权进度奖励，改用控制步末端接触后17个未通过场景的奖励均下降，配对变化中位数为−0.382。[逐场景复算和六段原片](../../records/reports/roll-forward-input-integration/checkpoint-selection/README.md) · [奖励复算](../../records/reports/roll-forward-input-integration/support-window-force/endpoint-progress-payment-replay.json)

2026-11-21，两个低摩擦种子的完整周期筛选完成256,000条环境转移，0.115m/s输入按配对降幅入选。完整20秒同快照对照完成947,200条转移，候选四组各128/128。原脚本评分错误按保存轨迹复算，恢复过程新增物理步为0。保存十段8秒原片和八段21秒原片。[完整周期筛选](../../records/reports/forward-stride-yaw-response/README.md) · [完整20秒对照](../../records/reports/forward-input-full-route/README.md)

2026-11-21，右脚支撑持续修正完成四组20秒同快照闭环对照。仅首段脉冲／持续修正，低摩擦两组均为0/128，标称两组为128/128／128/128和127/128／126/128。完整采样共998,400条转移。首次标称组观察中断保留173,184条部分转移和9.04秒候选原片。新观察协议完成两个标称组，全部正式录像独立解码。[逐组结果、录像与运行记录](../../records/reports/forward-right-support-continuation/README.md)

2026-11-16 后续预检：以实际 CPU 配置和编译模型核对标称、低摩擦和高电压。高电压同步改变电机力限值；复核历史 128 环境零步数组及注入故障，并对状态指纹工具做 CPU 检查。进一步发现已有更完整的精确守卫和标称 GPU 零步，后续沿用原守卫；本次简化适配仅用于 CPU 对照。本阶段零物理步、零学习。[参数预检](../../records/reports/roll-matched-factor-preflight-20261116/README.md)

2026-11-16 恢复后：核验旧归档、123 份来源、两组零步及本机资源，在原 VS Code 窗口打开候选状态并留存运行中截图。候选完成 17 次更新、1,253,376 条转移，与此前同步组同预算；总耗时 172.19→133.13 秒、吞吐提高 29.3%。候选监控无故障，保存点、ONNX、201 帧原片及 17 条指标通过独立核验。两组间隔约三小时、只有一次对照，未做最终动作评价；两份模型保留为诊断材料。已从两段真实原片生成 0–8 秒关键帧图。[训练与对照](../../records/reports/getup-monitored-authorization-20261116/README.md)

2026-11-16：按用户要求暂停并归档。最近同步检查组已在 10:02 完成 17 次更新、1,253,376 条转移；监控候选未启动。暂停后的工作仅包括进程与锁核验、保存点和 ONNX 的 CPU 读取、完整视频解码、来源哈希核对和文档整理，没有新增物理步或学习更新。123 份冻结来源、20 份既有交付文件及上次归档中的 57 份外部资产保持一致。无法访问本机网页时，改从 VS Code 打开本地状态、曲线和原片。[本次暂停归档](../../records/reports/pause-archive-20261116-monitored/README.md) · [本轮结果](../../records/reports/getup-monitored-authorization-20261116/README.md)

2026-11-14：吞吐与观察器短测完成独立复核，冻结行走 4,096 环境和 D3D12 + 4 Hz 观察配方；随后建立 S2–S6 阶段验收矩阵。矩阵以源 JSON 的计数和 SHA-256 为准，保留起身→站立→前进开发失败、七指令固定映射转向失败和参数随机化多技能失败；当前只列出滚翻连续切换、滚翻阶段抗扰和实物标定等开放门槛，不追加训练。 [吞吐与观察选择](../../records/reports/throughput-selection/README.md) · [阶段验收矩阵](../../records/reports/stage-acceptance-matrix/README.md)

同日：S12 只读取 seed-1173 的滚翻阶段横向扰动记录。左推 115/128 后按原规则停止，右推 118/128 且完整序列 116/128；保存数据把新增失败定位到短暂头部支撑中断，冻结 `head_roll` 0.02 rad 的零步准入草案，未启动动态候选或 PPO。 [滚翻扰动复盘](../../records/reports/roll-phase-perturbation-review/README.md)

同日：S13 新种子 1233 的三组短评估全部完成原 1,550 控制步。参考 124/128，左推候选 120/128，右推候选 122/128；三组原门槛和安全项通过。候选触发器左右均 128/128 全环境触发，覆盖过宽，候选不晋级，进入触发信号收窄阶段。 [S13 评分](../../records/reports/roll-phase-perturbation-review/s13-score.json)

同日：S14 对保存的左右失败窗口做单特征和两特征组合筛查，失败覆盖 75%、成功误触发 5% 的门槛均未达到，关闭固定阈值路线。下一步需要更长的接触时序和机身朝向归一化观察协议。 [S14 筛查](../../records/reports/roll-phase-perturbation-review/s14-trigger-screen.json)

同日：S14b 在不增加物理步、学习更新或硬件 IO 的条件下，对同一批左右扰动轨迹提取 109 个因果特征并做留一环境重拟合。左推覆盖 9/13 个失败、误触发 7/115 个成功，右推覆盖 6/10 个失败、误触发 4/118 个成功，均未达准入线；未启动零步、动力学或 PPO，下一步转向已有共同场景的切换时序复核。 [S14b 因果审计](../../records/reports/roll-phase-perturbation-review/s14b-causal-feature-audit.json)

同日：S15 读取 seed-1218 的完整滚翻→站立→前进轨迹。123/128 个世界在控制步 210–229 进入落地站立、260–279 进入前进，未进入前进的 5 个世界可由切换缺失事件识别；已进入前进的 12 个偏航失败在前 50 步没有形成留一环境稳定的选择性规则。保持原门槛，没有启动零步、动力学或 PPO。 [S15 审计](../../records/reports/continuous-roll-full-sequence-s1218/route-transition-causal-s15.json)

同日：S16 在新 seed-1234 上采集三臂完整滚翻扰动轨迹。参考 123/128、左推 119/128、右推 127/128，均通过原动作门槛；独立因果审计在左推只覆盖 1/9 个失败、右推 0/1，未准入新的零步或训练。首次左推采集的 guard 边界失败保留，随后以独立进程按原协议完成重跑。 [S16 评分](../../records/reports/roll-phase-perturbation-review/s16-score.json) · [S16 审计](../../records/reports/roll-phase-perturbation-review/s16-temporal-audit.json)

同日：S17 在 seed-1235 上复用官方滚翻、官方行走和固定 0.15 rad/s 前进反馈，zero-step 后完成 1,550 控制步完整路线。滚翻落地 120/128、前进入口 120/128、完整路线 119/128，原路线门槛通过；8 个滚翻支持旋转失败和 1 个偏航失败保留，未修改模型或启动 PPO。关键帧生成器的相对路径错误也已保留并修复。 [S17 报告](../../records/reports/official-route-independent-s1235/README.md) · [S17 评分](../../records/reports/official-route-independent-s1235/analysis.json)

同日：S18 只读取既有固定策略记录，独立核对四组参数端点、S16 三臂名义扰动和 S17 官方路线。固定策略鲁棒性证据通过；参数随机化候选统一回归仍未准入，S16 可迁移触发器仍未准入。没有新增物理步、学习更新或硬件 IO。 [S18 汇总](../../records/reports/robustness-evidence-s18.md) · [S18 机器记录](../../records/reports/robustness-evidence-s18.json)

# XGO-Duck 开发与部署报告

更新日期：2026-11-01。本报告是环境部署、官方复现、训练修改、评估和真机迁移的统一阅读入口。原始日志、参数和模型清单保存在对应目录，后续结果继续补入本文。

## 1. 当前结论

S0 的离线准备已核验，真实运行闭环仍待验证。优化后的目标已采用，当前项目阶段为 S1a（待执行）：先核对、导出已有模型，再进行行走评估与监控适配；本次应用目标管理器查询没有活动目标，未启动自动训练。九类动作范围保留，真机部署后置。目标审查发现停止规则、预算和阶段安排仍有待改进之处，见[目标审查](#goal-review)。完整顺序见[动作训练计划](#motion-curriculum)，入口与离线核验见[训练工作流](#training-workflow)。历史训练复盘已完成。12 次主要训练、14 次性能短测和 67 条已保存评估轨迹已重新核对；64 段训练录像与 67 段评估录像均已解码，选取 262 张真实帧并标注。当前最主要的问题是实际髋内收与脚部交叉长期存在、请求目标惩罚未修复姿态、部分头部目标不可达，以及中断保存和导出入口不完整。完整分析与后续顺序见[训练复盘](#training-review)，可直接打开[单张总览图](training-gait-overview.jpg)或[逐段图册](training-gait-gallery.html)。

用户于 22:09 要求暂停当前训练。22:10 核对确认本项目没有训练或评估进程在运行；任务 34 两组此前均已正常完成各 250 次更新，最终模型与恢复副本保留。自动收尾在导出入口因实验名称未被接受而停止，尚未完成模型导出及固定评估，纳入 S1 优先事项。最新暂停状态以[暂停记录](../../records/experiments/head_command_repair_pilot_20261101T213026_s42/pause-check.json)为准；下文的运行描述是对应时间的过程记录。用户随后已要求按阶段训练；按新的阶段门槛推进，不直接重启旧的失败长训练。

电脑环境、训练、恢复、导出与评估流程已经跑通，稳定行走仍未验收。正式基线已完成累计 5000 次更新及 ONNX 一致性检查；后续长训练因持续并脚停止，最近完整恢复点保留累计 5751 次。两组从相同 5000 次模型出发、各追加 250 次的修复试验也已完成：目标越界惩罚降低了请求超额，但两组都未通过运动与姿态标准。任务 31 的六项连续显示性能短测已结束并独立核验，诊断权重不进入正式步态模型选择。任务 32 的头部指令配置检查也已通过；任务 33 因实时查看器关闭中断，原样组 172 次更新已核验，最近完整保存点包含其中 151 次。任务 34 两组从该保存点开始，均关闭原生实时窗口，并在暂停前完成各 250 次更新，最终累计 5401 次。本次复盘另在 CPU 上读取两组最终模型，核对张量有限性、优化器步数、归一化计数和保存计数；两组均通过这些检查。修改组原导出流程专用的最终检查记录仍未补写，自动导出与固定评估没有重试，因此头部改善尚无结论。原长训练仍停止。

头部问题已单独定位。用户看到的训练片段包含随机头部指令，不能全部解释为平衡动作；中立头部指令的固定评估仍有摆动。配置存在颈部期望姿态上界超出模型关节限位约 23.03° 的不一致，新增真实遥测也在六项短测中的五项发现了越界请求。可达指令修正已通过完整配置差异和真实 CPU 采样检查，只收窄后两段课程的颈部幅度；已接入独立的成对试验入口，训练效果尚未评估，不能宣称摆动已经改善。诊断见第 23.13 节，新增曲线、修改方法及复现入口见第 26 节。

当前官方行走 ONNX 已独立保存、检查接口，并完成同条件评估和录像。它前进时平均速度约 0.102 m/s，脚间距中位数约 54 mm，32 个前进 episode 未出现足部位点交叉，颈部摆动小于自训原样组；但五条指令仍仅停止达到原数值标准。原归档参考保留，官方模型的训练历史未知，不能将两者差异归为某个奖励修改。第 24.5 节提供指标、曲线、真实回放和复查入口。

效率方面，既有六项规模筛查支持本机使用 4096 环境；完整静态模型的 EGL、GLFW 渲染可使用 RTX 3080 Laptop，默认路径则是 CPU 软件渲染。连续显示短测中，GPU 4 Hz 相对默认 4 Hz 的吞吐比两遍均大于 1，几何平均为 1.5297；提高到 15 Hz 上限后实际仍只有每秒 4.26～4.34 个新状态，流畅显示尚未达标。按用户后续要求，新的训练入口关闭原生实时窗口，采用正常速度回放的真实片段。当前先验证可达的头部目标，行走姿态约束另设试验。官方公开路线、云训练入口及原生 Ubuntu 备选见第 24 节，实测与选择规则见第 25 节。

用户已完成机器人装配并自行安装 Arduino App Lab；当前授权范围先处理训练，真机连接、标定及部署验证尚未执行。后文的历史暂停、启动和阶段预算按对应时间理解，最新状态以本节、目标清单和 `project.json` 为准。

| 项目 | 实际状态 | 证据或后续入口 |
| --- | --- | --- |
| 电脑环境部署 | 已完成并核验 | 第 3、4 节；部署日志保留 |
| 短训练与修改流程 | 两组各 5 次，导出及接口通过 | 第 5、6 节；不代表修改收益 |
| 基线 5000 次 | 已完成，五指令仅停止达标 | 第 18、19 节 |
| 追加至 7500 次 | 因持续并脚停止，保留累计 5751 次 | 第 21、22 节 |
| 并脚诊断 | 固定权重单因素诊断完成 | 第 22 节；动作限制分开双脚，跟踪仍失败 |
| 修复短训练 | 两组各 250 次完成，均未通过运动及姿态 | 第 23.12 节 |
| 头部诊断 | 真实请求越界已确认，可达指令检查通过，训练效果待验 | 第 23.13、24.5、26 节 |
| 官方模型参照 | 原归档及当前模型均完成，未通过全部指令 | 第 15、24.5 节 |
| 训练效率 | 规模筛查通过，连续显示条件下 GPU 吞吐约提升至 1.53 倍 | 第 20、21、25 节；流畅显示仍未达标 |
| 原生 Ubuntu | 入口已准备，未切换或实测 | 第 13 节 |
| 稳定行走与正式修改收益 | 未验收 | 第 7 节；多 seed 比较尚未执行 |
| 真机标定与策略部署 | 未执行 | 第 8、9 节 |

## 2. 目录和来源

Windows 项目根目录：`C:\Users\atlas-win\Desktop\RL-Project\XGO-Duck`。Linux 训练副本：`/home/atlas/XGO-Duck/workspace/xgoduck_rl`。文档、辅助脚本和实验定义在 Windows 项目维护；训练虚拟环境、checkpoint 和 TensorBoard 原始事件放 Linux 文件系统。

```text
XGO-Duck/
├── README.md                         阅读入口
├── project.json                      实际状态
├── official_resources/               官方源码、原始快照与哈希，不直接修改
├── docs/reports/development_report.md 本报告
├── docs/runbooks/                     分步操作参考
├── scripts/setup/                    可见安装、镜像下载、环境核验
├── scripts/training/                 训练启动与记录
├── scripts/validation/               导出、接口与数值检查
├── configs/experiments/               修改补丁、训练预算和评估定义
├── records/deployment/<date>/         部署日志及问题证据
├── records/experiments/<run-id>/      配置、结果、截图、视频与媒体清单
├── records/evaluations/<run-id>/      固定指令轨迹、指标、视频与复核
├── records/hardware/                 映射检查与后续真机记录
├── workspace/                        下载和本地工作文件
└── artifacts/<run-id>/               ONNX、接口契约与来源清单
```

| 来源 | 本轮固定提交 | 用途 |
| --- | --- | --- |
| `xgoduck_rl` | `fa621fc77ad9a05d69da13270e86118bdd3603fe` | PC 仿真、PPO 训练、导出 |
| `xgoduck_runtime_arduino` | `8837d5a3bc7cf3746f6da2b55c7b9191026b9838` | UNO Q 官方运行程序与预训练策略 |
| `xgoduck_hardware` | `b5e9ff16a98f7bbe5c146aad29ad38f0dde30a89` | 装配图、结构和电路资料 |

官方归档保持不动。上游运行程序已出现桥接、增益和控制参数变化；首次上板前应固定实际采用的提交，再核对本文操作与该份代码。不能把新网页参数与旧运行代码混合使用。

## 3. 电脑环境部署

### 3.1 实测环境

| 项目 | 本机实测 |
| --- | --- |
| Windows | Windows 11 Pro for Workstations，10.0.26200 |
| CPU / 内存 | Ryzen 9 5900HX / 约 64 GB |
| GPU | RTX 3080 Laptop / 16 GB，驱动 592.54 |
| WSL / Linux | WSL 3.0.3.0 / Ubuntu 24.04.5，用户 atlas |
| Python / uv | 3.12.3 / 0.13.6 |
| PyTorch | 2.9.1，CUDA 12.8 构建 |
| mjlab / Warp | 1.3.0 / 1.12.0 |
| MuJoCo / mujoco-warp | 3.10.0 / 3.8.1 |
| rsl-rl-lib | 5.0.1 |
| ONNX Runtime | Linux 1.24.4；独立 Windows 兼容检查 1.22.1 |

Ubuntu 24.04 和 WSL2 是本机选择。归档官方训练说明要求 CUDA GPU、Python 3.12 和 uv，没有指定 Ubuntu 24.04。本机是否适用依据实际 CUDA、训练、回放和导出结果判断。

### 3.2 安装 WSL 和 Ubuntu

以下记录完整安装路线；当前电脑已经安装完成，无需重装。所有安装在单个可见终端顺序执行，保留命令、开始结束时间、输出和退出码。

```powershell
Set-Location -LiteralPath 'C:\Users\atlas-win\Desktop\RL-Project\XGO-Duck'
.\scripts\setup\install_wsl.ps1
```

初次调用微软 `wsl.exe --install --no-distribution --web-download` 无有效进度，随后采用 `scripts/setup/install_wsl_msi.ps1` 下载并核验微软 MSI 的签名和 SHA-256。WSL 安装完成后，VirtualMachinePlatform 返回 3010，表示需要重启。实际从开始菜单执行“电源 → 重启”，不能把重启应用当作系统重启。

```powershell
wsl --version
wsl --status
wsl --list --verbose
.\scripts\setup\install_ubuntu.ps1
```

在线 Ubuntu 安装出现 `WININET_E_INVALID_SERVER_RESPONSE`，保留失败记录后改用官方本地镜像：

```powershell
.\scripts\setup\install_ubuntu.ps1 -UseLocalPackage
.\scripts\setup\initialize_ubuntu.ps1
```

本地包 `ubuntu-24.04.5-wsl-amd64.wsl` 大小为 388975696 字节，SHA-256 为 `bb415d824822c4b878125729af451a5d18fb13d1cf5cbed9a7393ad64ac6039e`。脚本检查包后调用 `wsl --install --from-file`。用户本人完成用户名 atlas 和密码设置；密码不进入记录。

在 Ubuntu 内检查：

```bash
cat /etc/os-release
uname -r
nvidia-smi
python3 --version
df -h ~
free -h
```

### 3.3 网络与镜像源

本机 FlClash 自动切换节点，Windows 的本机代理可用，但 WSL NAT 下下载超时。实际创建的 `C:\Users\atlas-win\.wslconfig` 为：

```ini
[wsl2]
networkingMode=mirrored
autoProxy=true
dnsTunneling=true
```

结束当时的项目同步进程后，只重启 WSL，再检查代理连通性。配置与执行记录见 [网络恢复记录](../../records/deployment/2026-10-31/network-recovery-224217-result.json)。现有文件不得盲目覆盖；未来网络失败先检查自动切换的节点、原进程和缓存。

包下载优先使用 `https://pypi.tuna.tsinghua.edu.cn/simple`。本机镜像下载仍需经现有代理通路。辅助脚本 `sync_locked_mirror.py` 只替换下载地址，保留官方包的精确依赖、哈希和平台条件，不重新求解依赖：

```powershell
.\scripts\setup\run_linux_bootstrap.ps1 -UseMirror
```

该步骤依次核验源码、建立 Linux 副本、安装 uv、同步锁定依赖、验证 CUDA 张量运算、列出任务和保存 CLI 帮助。镜像同步使用 `uv sync --frozen --default-index https://pypi.tuna.tsinghua.edu.cn/simple`，结束后恢复官方 `uv.lock` 并执行 `uv sync --locked --offline`。官方锁文件 SHA-256：`0ef769135d86e63c03663b2df570c0217217b541ab234460a879dff8fb260fa9`。

工作副本曾做奖励修改，2026-11-01 正式训练前已撤回补丁并核对官方源码哈希；存在未撤回修改时，首次部署脚本会拒绝覆盖。后续在既有副本中核验依赖即可，不要删除实验来重装。每次进入 Linux 训练终端先加载项目环境：

```bash
source /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/setup/project_environment.sh
cd /home/atlas/XGO-Duck/workspace/xgoduck_rl
uv sync --locked --offline
```

uv、CUDA、Torch、Triton 等缓存放在 `~/XGO-Duck/cache/`。已有缓存迁移记录，不重复下载。自动执行集中到 `scripts/setup/visible_console.ps1` 的一个可见窗口；浏览器用于曲线、回放和报告，不另外堆积终端。

### 3.4 实际问题与处理

| 现象 | 排查所得 | 实际处理与验证 |
| --- | --- | --- |
| WSL 安装无进度 | 在线安装没有形成可用安装结果 | 改官方 MSI，校验签名、哈希及退出码 |
| 安装后仍提示重启 | 虚拟机平台返回 3010，启动时间未变 | 真正重启 Windows 后核验 WSL |
| Ubuntu 在线下载失败 | 下载服务器响应异常 | 官方本地镜像断点续传、校验、注册 |
| PyPI 超时 | Windows 代理可用，WSL NAT 通路不通 | 镜像网络与代理继承后重测 |
| 原下载路径很慢 | 同一文件的小范围测速存在明显差别 | 保留锁定依赖，改清华传输地址 |
| 任务列表返回非零 | 官方入口返回任务数量，被解释为退出码 | 直接调用列表函数，并断言目标任务存在 |
| 修改后首次部署脚本拒绝覆盖 | 副本已不是官方原始内容 | 保留副本与实验，仅按记录应用或撤回补丁 |

完整时间线与原始输出见 [部署记录](../../records/deployment/2026-10-31/README.md)。没有保存到的历史安装画面不伪造，使用真实文本日志作为证据。

## 4. 官方训练与回放

任务是 `Mjlab-Velocity-Flat-XgoDuck`。官方短训练使用 256 个并行环境、每环境每次采样 24 步、5 次迭代、seed 42；合计 30720 个环境转移。官方默认最大训练迭代数为 50000；本轮 5 次仅用于检查流程。

2026-11-01 已执行以下反向补丁并核验，当前工作副本恢复为官方权重 3.0。以下命令只在副本确实处于修改组状态时使用，不重复反向应用；已有模型保留：

```bash
git apply -R --check /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/configs/experiments/velocity_tracking.patch
git apply -R /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/configs/experiments/velocity_tracking.patch
```

在单个可见 PowerShell 中调用记录脚本：

```powershell
.\scripts\training\run_linux_smoke.ps1 -Mode baseline
```

其核心训练命令如下；实际运行名称由脚本自动生成，避免覆盖：

```bash
uv run --locked --offline train Mjlab-Velocity-Flat-XgoDuck \
  --env.scene.num-envs 256 --agent.max-iterations 5 \
  --agent.seed 42 --agent.run-name official_smoke \
  --agent.logger tensorboard
```

训练结束检查日志退出码、`params/env.yaml`、`params/agent.yaml` 和实际 checkpoint，不能只看命令启动。已验证基线 checkpoint 为：

```text
/home/atlas/XGO-Duck/workspace/xgoduck_rl/logs/rsl_rl/xgoduck_velocity/2026-10-31_23-01-39_baseline_smoke_20261031T230120/model_4.pt
```

回放命令在上述 Linux 副本执行：

```bash
uv run --locked --offline play Mjlab-Velocity-Flat-XgoDuck \
  --checkpoint-file logs/rsl_rl/xgoduck_velocity/2026-10-31_23-01-39_baseline_smoke_20261031T230120/model_4.pt \
  --num-envs 1 --viewer viser
```

打开终端实际打印的浏览器地址。画面可用于观察站姿、摆腿、滑脚和跌倒，正式结论还需固定指令与数值评估。训练中的随机采样画面与确定性评估视频须分别标注。

## 5. 修改位置与作用

本轮只改 Linux 副本 `src/mjlab_microduck/tasks/__init__.py` 中 `_xgoduck_velocity_cfg` 的一行：

```python
cfg.rewards["track_linear_velocity"].weight = 3.3  # 官方原值 3.0
```

| 修改 | 预期作用 | 可能代价 | 当前实测结论 |
| --- | --- | --- | --- |
| 线速度跟踪权重提高 10% | 优化时更重视速度指令跟踪 | 可能牺牲平滑度、站稳或其他奖励目标 | 5 次短训练不足以判断收益 |

这不是直接把机器人速度提高 10%，也不是舵机增益修改。奖励总分的量纲随权重变化，不能单凭修改组总奖励较高就认定更好。

补丁保存在 [velocity_tracking.patch](../../configs/experiments/velocity_tracking.patch)。在官方副本状态下执行：

```bash
git apply --check /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/configs/experiments/velocity_tracking.patch
git apply /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/configs/experiments/velocity_tracking.patch
```

随后在可见 PowerShell 执行 `.\scripts\training\run_linux_smoke.ps1 -Mode candidate`。脚本校验源码哈希；两组保存配置已分别核对为 3.0 和 3.3。后续每项修改都按“文件与参数 → 假设 → 同预算实验 → 数值与视频 → 保留或撤回”记录，不把多个参数同时改动归因到某一项。

## 6. 已执行结果与媒体

### 6.1 数值检查

| 检查 | 官方基线 | 修改组 |
| --- | --- | --- |
| 实验标识 | `baseline_smoke_20261031T230120` | `candidate_smoke_20261031T230835` |
| 环境 / 迭代 / seed | 256 / 5 / 42 | 256 / 5 / 42 |
| 退出码 | 0 | 0 |
| checkpoint 张量 | 76 个，全部有限 | 76 个，全部有限 |
| ONNX 驱动仿真 | 1000 帧 | 1000 帧 |
| 超容差帧数 | 0 | 0 |
| 最大动作绝对误差 | 1.7881 × 10⁻⁷ | 1.4901 × 10⁻⁷ |
| 重置，包含超时 | 1 | 6 |
| Runtime 1.22.1 检查 | 通过 | 通过 |

误差容差为 `atol=1e-5, rtol=1e-4`，同一真实仿真观测同时输入 PyTorch actor 和 ONNX，ONNX 输出负责驱动物理仿真。重置数包含 episode 超时，不能作为跌倒次数；修改组重置较多也不足以单独判断修改收益。

### 6.2 仿真截图

以下是实际 Viser 回放截图，暂停后调整观察视角。两图分别来自各自的 5 次短训练模型，仅证明加载与可视化，不代表连续行走成功。

![官方基线](../../records/experiments/baseline_smoke_20261031T230120/screenshots/01_simulation.jpg)

![修改组](../../records/experiments/candidate_smoke_20261031T230835/screenshots/01_simulation.jpg)

### 6.3 视频与训练曲线

已完成一次 64 环境、20 迭代、seed 42 的视频预检，退出码 0。奖励和 episode 长度各保存 20 个真实指标点；首段视频已经完整解码核验，为 960 × 720、50 fps、200 帧、4 秒。它记录初期随机策略的训练采样，策略会在片段中更新，不能对应成单个固定 checkpoint，也不作为稳定行走演示。

<video controls preload="metadata" src="../../records/experiments/baseline_video_preflight_20261101T000737_s42/videos/rl-video-step-0.mp4"></video>

[训练初期视频](../../records/experiments/baseline_video_preflight_20261101T000737_s42/videos/rl-video-step-0.mp4) · [媒体来源与解码检查](../../records/experiments/baseline_video_preflight_20261101T000737_s42/media-manifest.json) · [训练记录](../../records/experiments/baseline_video_preflight_20261101T000737_s42/training-result.json)

后续视频继续放在 `records/experiments/<run-id>/videos/`。固定策略的评估视频另记录 checkpoint 哈希、指令、seed、仿真时长和重置；保留失败片段。真机视频仍待首次运行与标定后采集。

训练观看包含可见终端的迭代日志、浏览器的实时曲线，以及按训练阶段保存的仿真片段。训练采样视频展示策略学习过程；正式评估视频采用固定指令和固定条件。真机拍摄需看到完整机器人和脚底接触，另存控制网页及反馈状态截图。

正式基线启动后的实时页面截图如下，图中曲线来自该次训练的 TensorBoard 事件，视频为该次训练采样。截图只反映捕获时进度，继续训练后的结果以日志为准。

![实时训练](../../records/experiments/baseline_formal_20261101T001206_s42/screenshots/01_live_training.jpg)

### 6.4 导出与核验操作

以下 `<run>`、`<checkpoint>` 替换为已经核实存在的实验路径：

```bash
uv run --locked --offline python scripts/export.py Mjlab-Velocity-Flat-XgoDuck \
  --checkpoint-file 'logs/rsl_rl/xgoduck_velocity/<run>/<checkpoint>' \
  --num-envs 1 --onnx-file 'logs/rsl_rl/xgoduck_velocity/<run>/policy.onnx'
uv run --locked --offline python \
  /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/validation/validate_export.py \
  --checkpoint 'logs/rsl_rl/xgoduck_velocity/<run>/<checkpoint>' \
  --onnx 'logs/rsl_rl/xgoduck_velocity/<run>/policy.onnx' \
  --output-dir 'logs/rsl_rl/xgoduck_velocity/<run>/export_validation' \
  --steps 1000 --seed 42
```

输出目录必须新建，避免覆盖已有证据。本轮两份 ONNX 都为 float32 输入 `[1,61]`、输出 `[1,14]`，IR 8、opset 18。模型包位于 [基线产物](../../artifacts/baseline_smoke_20261031T230120/README.md) 和 [修改组产物](../../artifacts/candidate_smoke_20261031T230835/README.md)，均未通过实机验收。

## 7. 充分训练与评估方案

首轮正式记录为 `baseline_formal_20261101T001206_s42`，曾按用户要求暂停；随后已在同一可见 VS Code 终端恢复，当前续训证据见第 18 节。终端每次迭代打印进度，每 250 次迭代附近记录一段 4 秒训练视频；关闭可见训练控制台会中断工作。本机页面地址为 `http://localhost:8765/`，当前浏览器访问限制与替代的本地曲线、录像入口也见第 18 节。首轮命令和冻结协议见 [运行请求](../../records/experiments/baseline_formal_20261101T001206_s42/training-request.json) 与 [评估协议](../../configs/experiments/formal_velocity_tracking.json)。

| 正式设置 | 已固定内容 |
| --- | --- |
| 首轮训练 | 1024 环境 × 每迭代 24 步 × 5000 迭代，约 1.2288 亿环境转移 |
| 保存间隔 | 250 迭代，保留最终 checkpoint |
| 每个评估条件 | 32 个 episode，每个 20 秒；开发 seed 1001，留出 seed 2001 |
| 指令 | 停止；前进 0.15 m/s；后退 0.10 m/s；左右转向 0.4 rad/s |
| 逐条件验收 | 完整存活率 ≥95%；平面速度 MAE ≤0.08 m/s；yaw MAE ≤0.20 rad/s；停止漂移中位数 ≤0.15 m；观测与动作有限 |
| 规则来源 | 本项目预先设定的验收条件，不是官方认证标准 |

5000 次是本项目首阶段预算。[归档 README](../../official_resources/xgoduck_rl/README.md)中的平地训练示例显式使用 4096 个环境；XGO-Duck 注册配置继承步态配置的 `max_iterations=50000`。已只读构造实际任务配置，并核对归档与运行副本的源码、锁文件，见[预算核对](../../records/reports/training-budget-check.json)。本次检查没有创建模拟器或初始化 CUDA。

| 计算设置 | 本轮首阶段 | 官方示例与默认上限 |
| --- | ---: | ---: |
| 并行环境数 | 1024 | 4096 |
| 每环境每次采样步数 | 24 | 24 |
| 最大更新数 | 5000 | 50000 |
| 到上限的计划环境转移数 | 122,880,000 | 4,915,200,000 |

本轮计划采样量为右列的 1/40，因此它是采用官方任务、奖励和 PPO 配方的阶段训练，计算规模并未完整复现官方示例。右列表示 README 命令运行到默认上限的计划量，不是官方预训练 ONNX 的实际训练记录；该模型的训练预算仍未知。直接读取未加命令行参数的配置会看到 `num_envs=1`，README 的 `--env.scene.num-envs 4096` 和本项目的 1024 都是显式覆盖值。

训练到 5000 次后，先导出并执行固定指令开发评估。若未达标，保留本阶段结果，核对导出一致性、失败条件及动作表现，再确定是否追加训练或调整诊断方案；追加预算和恢复方式另行记录，不改写本阶段协议或用留出 seed 挑选方案。后续奖励对照须统一两组的最终预算、主机和起点条件。是否训练充分以步态验收为准，不能仅凭达到 5000 或 50000 次判定。

后续先完成官方配置基线，确认它在仿真中稳定行走，再开展 3.3 权重组。用户已明确恢复 WSL 基线，当前执行情况见第 18 节；继续旧 checkpoint 与从头启动新实验的区别见第 12.3 节。正式训练预算与评估条件在执行前写入 `configs/experiments/`，训练开始后不依据某组成绩偷偷更换条件。训练 seed 采用 42、43、44，同一比较中的两组使用相同环境数、每迭代采样步数和总更新数。

对照还必须匹配运行环境和恢复安排。当前 seed 42 基线分为已保存的 251 次更新和续训 4749 次，恢复时没有保留随机状态、episode 与延迟历史；现有 3.3 入口默认从头训练，候选恢复入口尚未准备。因此不能直接把两者的结果差异解释成单独修改奖励的效果，见 [起点核对](../../records/reports/comparison-restart-audit.json)。进入正式修改对照前，先固定两组均从头训练，或在相同更新数保存并按一致规则恢复的成对方案；候选组的恢复点须由它自己按 3.3 权重训练得到，不能用 3.0 基线 checkpoint 充当候选前段。若选原生 Ubuntu，两组一起采用该环境。当前续训继续用于稳定性验收，是否纳入主对照要等起点条件核验，不因已经花费训练时间而省略这一条件。

修改组入口现在同时检查基线步态和对照起点。若 `project.json` 中的 `comparison_start_conditions_verified` 尚未通过核验，程序在创建实验目录或训练进程前退出；这个标记记录人工核验结论，本身不是两组可比的证据。临时项目元数据下的九种路径检查通过，覆盖未匹配、缺失或非布尔标记、用户暂停，以及基线与续训路径；没有启动训练或模拟器，见 [入口检查](../../records/reports/comparison-entry-check/check.json)。通过两个标记的测试也在读取缺失的临时协议时停止，只证明检查顺序，不代表真实候选训练成功。当前运行副本未重载。

以下命令可在当前任务结束后、项目根目录的同一终端复查入口。它使用临时元数据，不改实际项目标记或训练参数：

```bash
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/check_comparison_entry.py
```

官方站立课程到约第 2000 次迭代仍在变化，因此 5 次短训练远远不够。首轮正式预算应覆盖该课程并留出收敛时间；是否需要继续由学习曲线及固定评估判断，不能承诺某个迭代数必然稳定。若扩展预算，两组按相同最终预算比较，并记录中断与恢复。

固定平地评估覆盖停止、前进、后退、左右转向；将正常行走与扰动测试分开。正式协议要指定每条件 episode 数、时长、指令范围、独立评估 seed、跌倒定义和接受阈值。指标包括线速度 MAE、角速度 MAE、完整 episode 成功率、存活时间、站立漂移、动作变化量和接触时脚底滑移。失败 episode 不从统计中剔除，超时与跌倒分开统计。

checkpoint 选择使用开发评估，最终模型再用预留条件复核。展示各 seed 的结果和汇总离散程度，不将同一策略的许多并行 episode 当成许多独立训练种子。报告比较“跟踪改善了多少、稳定性是否降低、动作是否更激烈”，不以奖励总分判优。

### 7.1 训练效率

暂停后重新分析完整终端日志与原始 TensorBoard 事件，核对 0–259 共 260 次完整迭代，每次 1024 × 24 = 24576 个环境转移，共 6389760 个。控制台四舍五入后的采样/更新耗时、吞吐与事件中的精确数值一致。剔除前 20 次作为热身，并按实际录像触发实现区分区间；没有删除非录像时段的慢迭代。

录像每 6000 个环境调用步触发一次，连续记录 200 帧；本次对应迭代 0–8、250–258。两段保存视频均完整解码为 200 帧、4 秒。表中时间与吞吐为各区间的中位数。

| 区间 | 迭代数 | 采样（秒） | 更新（秒） | 事件间隔（秒） | 日志吞吐（转移/秒） |
| --- | --- | --- | --- | --- | --- |
| 热身后、无录像 | 231 | 2.0048 | 0.1496 | 2.2031 | 11384 |
| 热身后、含录像 | 9 | 3.8749 | 0.1995 | 4.0638 | 6116 |
| 录像前局部区间 230–249 | 20 | 2.1484 | 0.1498 | 2.3832 | 10564 |

“采样”包含策略前向、环境步进、重置、数据检查和录像，不等同于纯物理仿真耗时；“更新”是 PPO 学习计时。日志吞吐由这两项相加计算，不包含全部日志写入与 checkpoint 保存。相邻 TensorBoard 事件的时间差更接近连续迭代的实际节奏，但受日志与保存位置影响，也不是严格的逐段 profiler。

在热身后无录像区间，采样总时间占采样加更新总时间的 **93.06%**；总环境转移除以事件间隔总和为 **10263.85 转移/秒**。这与“每次吞吐的中位数 11384”是不同统计量，不能混用来估算总训练时间。训练进程从启动到中断共 680.53 秒，完整迭代的两项内部计时合计 643.98 秒，差值 36.55 秒混合了启动、记录、保存、中断和未完成工作，不能全部解释为可消除的开销。

下图不作平滑，阴影是实际录像区间，竖点线是 20 次热身边界。录像区间较慢，但它发生在不同训练状态，不能据此直接宣称关闭视频能提速多少。迭代 87、242 等非录像区间也出现采样尖峰，需要后续受控分析；保留真实可视化仍是正式训练要求。

![训练耗时](../../records/experiments/baseline_formal_20261101T001206_s42/efficiency-analysis/training-efficiency.png)

早期的 [短时遥测](../../records/experiments/baseline_formal_20261101T001206_s42/efficiency-20261101T002055.json) 保留为原始证据：12 次 GPU 采样中，利用率中位数 36%，`utilization.memory` 为 100%，显存占用约 2854 MiB、功耗约 61.27 W、温度约 67°C。`utilization.memory` 表示采样周期内显存读写活跃的时间比例，不能解释成容量用满或带宽达到峰值；WSL 的 NVIDIA-SMI / NVML 查询也有功能限制。该遥测没有同步记录 CPU 利用率，尚不足以确定具体硬件瓶颈。[NVIDIA 指标定义](https://docs.nvidia.com/deploy/nvidia-smi/index.html) · [NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

在 VS Code 运行 **09 暂停实验效率分析（不训练）** 可重算统计与图像。输入只来自已保存文件，不运行采样或更新权重；两边锁定依赖准备好后可使用同一入口。结果见 [分析记录](../../records/experiments/baseline_formal_20261101T001206_s42/efficiency-analysis/analysis.json)、[逐迭代 CSV](../../records/experiments/baseline_formal_20261101T001206_s42/efficiency-analysis/iterations.csv) 和 [绘图检查](../../records/experiments/baseline_formal_20261101T001206_s42/efficiency-analysis/plot-check.json)。原始 TensorBoard 事件已复制到项目记录，复制件与 Linux 原件哈希一致，见 [事件来源](../../records/experiments/baseline_formal_20261101T001206_s42/tensorboard/source.json)。

![效率分析](../../records/experiments/baseline_formal_20261101T001206_s42/efficiency-analysis/vscode-analysis.jpg)

下一步先测量采样内部各段和录像成本，再做第 13 节的同机环境对照；改变并行环境数时单独记录样本预算。当前完成的是诊断数据整理，尚无受控提速结果。

### 7.2 官方预训练模型参照

新增目标是研究官方训练好的模型，在相同停止、前进、后退、左右转向条件下评估并录制视频，作为自训模型的表现参照。首先采用归档运行程序的 `python/xgoduck_walk.onnx`，其 SHA-256 为 `57cdebf36e7fa23116270dec96f013a6a64511f37311d322443f5297bdc3d6db`，输入 61、输出 14，元数据的腿部 home 约 ±24°，动作比例 1.0。该元数据与普通板端程序的 ±22° 默认姿态不同，必须明确使用哪种适配条件。

固定条件仿真评估、媒体复核与后续离线接口检查已完成，详细结果见第 15 节。五个条件只有停止达标，尚未在本机通过稳定行走验收。报告分别列出官方预训练参考、官方配置从头训练和奖励修改组；来源不同、训练预算未知时，不把参考模型当成同预算实验。

## 8. 真机首次运行与标定

### 8.1 从哪里开始

你已完成机械装配、尚未运行程序。先确认 UNO Q 能在 Arduino App Lab 中被识别，记录板卡系统、扩展板批次、舵机型号、电源、官方运行源码提交和模型哈希。电脑训练环境不需要复制到 UNO Q；UNO Q 运行官方应用和 ONNX 推理。

用户已确认自行完成 Arduino App Lab 安装；桌面清单同时出现 Arduino App Lab 0.11.0 窗口，见 [窗口记录](../../records/deployment/2026-11-01/app-lab-window-observation.json)。安装来源、签名、应用内容和首次连接尚未由本项目核验。用户要求当前先只处理训练，因此不再继续安装或操作 App Lab。此前未检出安装及下载访问受阻的历史保留在 [准备记录](../../records/deployment/2026-11-01/app-lab-preparation.json)。自主安装时应从 [Arduino 官方软件页](https://www.arduino.cc/en/software) 选择 **Arduino App Lab** 的 Windows 安装程序，并保存实际安装记录。

若舵机装配前没有设置 ID，先处理 ID，不能假设整条总线上已有 15 个唯一地址。改变 ID 时按官方说明只连接一颗舵机；断电后调整线束，再接通对应单舵机。已经装好的机构不必无理由全部拆开，但必须能隔离需要设置的舵机。

先保持机器人有稳定支撑、关节有活动空间。上电前按实物批次核对供电与接线，特别检查官方旧扩展板 U3/U4/U5 的 GND/SIGNAL 更正是否适用。这些操作与电机卸载行为直接相关，不用推测电源电压或机械角度。

### 8.2 启动官方程序

使用官方运行仓库的独立部署副本，在 Arduino App Lab 打开。首次 Run 下载板端库后停止应用，在 **UNO Q 内该应用项目目录** 执行以下命令，不能在 Windows 或 WSL 训练目录执行：

```bash
python3 tools/prepare-bridge.py
sudo bash tools/router-2m.sh
```

保存输出后再次 Run，打开 App Lab 显示的网址或 `http://<board-ip>:9527`。官方接线为舵机总线 D0/D1，QMI8658 IMU D20/D21。应用启动为 **Inference only**，电机卸载。检查 **Servos online 15/15**、**IMU online**，并观察反馈是否持续更新。若 ID 尚未配置，先完成下一小节，再回来验收 15/15。

首次部署的精确 App Lab 导入操作需在实际电脑与板卡被识别后核验。本报告尚未声称已经连接、刷写或获得板端反馈。板端终端和 App Lab 日志同样保持可见，使用既有一个终端顺序操作。

### 8.3 舵机 ID 与机械零位

| ID | 关节 |
| --- | --- |
| 10 / 11 / 12 / 13 / 14 | 左髋偏航 / 左髋侧摆 / 左髋俯仰 / 左膝 / 左踝 |
| 20 / 21 / 22 / 23 / 24 | 右髋偏航 / 右髋侧摆 / 右髋俯仰 / 右膝 / 右踝 |
| 30 / 31 / 32 / 33 | 颈俯仰 / 头俯仰 / 头偏航 / 头侧摆 |
| 34 | 嘴部，独立于 14 维行走策略 |

单舵机连接时进入 **Servo setup → Scan all IDs**，选择扫描结果，通过 **Change ID to → Change ID** 设置，再重新扫描并读取设置。给实物贴 ID 标签，与装配图对应。网页 KP/KD 输入框的初值不是对所有舵机都适用的校准结果，不要因看到默认数字就批量改写增益。

**机械零位**是关节按装配几何定义的参考位置；**Default pose** 是行走策略的屈腿站姿。不能把机器人随意摆成站姿就点击设零。归档硬件指南明确指出缺少逐关节机械零位照片，装配图也不能自动替代每个轴的零位定义；对不确定的关节，需结合官方装配图和厂家零位参考确认后再写中位。本报告不编造“所有关节拉直即为零位”的规则。

确认某关节的机械中位后，在 **Calibration** 中按下面顺序操作：

1. 进入页面后电机卸载，手动把目标关节放到确认的中位。
2. 点击该关节 **Set center & hold**。该关节写入中位并保持；已经标定保持的关节可能仍保持，不要强行扳动。
3. 逐关节重复；需要重新调整时点击 **Unload all**。
4. 点击 **Exit** 返回控制页，退出会卸载。支撑机器人，避免卸载时倒下。

**Reset Mid** 位于 Servo setup，用于写中位后保持卸载；**Set center & hold** 还会使能该关节并保持位置。归档源码中 `servoMid()` 发出舵机中位指令，角度换算中心固定为 2047，保持目标为 2048。二者相差一个原始计数不是需要修改的左右偏差。

硬件指南中“进入校准就统一移动到 2047”“备份 data/zero_pos.json”的文字与本轮运行源码不一致：该源码进入校准先卸载，没有用该 JSON 实现零位。操作以实际运行代码为准，不能新建一个同名文件就认为完成标定。保存每个 ID 的中位照片、读回数据和操作时间；在支撑状态断电重启后复核机械中位是否保留。未验证前不要宣称零位持久化已通过。

逐关节记录可从 [标定表模板](../../configs/hardware/commissioning_record_template.csv) 复制到本次 `records/hardware/` 的实际操作目录。模板按 MCU 顺序列出 15 个 ID，所有结果均为 `pending`；依次填写实物标签、舵机型号、单颗扫描 ID、机械中位依据与照片、写中位后的读数、重启读回、方向观察、供电和时间。没有确认的项目保持空白，不用数字 0 代替未知。

读原始位置还要核对 `raw_mask`：运行源码启动时就把 `raw_pos` 初始化为 15 个 `2047`，此时 `raw_mask=0`，这些值不是舵机读回。`cal_done` 表示当前校准页面中的保持状态，退出页面会清空，也不能代替非易失中位记录。模板中的重启复核必须结合当时有效反馈与实物照片完成。

### 8.4 IMU 与关节方向

先在 **Inference only** 观察静止反馈，再手动缓慢改变身体俯仰、侧倾，确认网页姿态和角速度方向一致。运行代码已处理传感器坐标映射，不能在另一层再重复变换。身体参考姿态水平且静止时，策略坐标中的重力应接近 `[0,0,-1]`，角速度应接近零。

归档 `python/rl_core.py` 的 `IMU_BIAS_DEG` 是安装姿态修正，顺序为 roll、pitch、yaw。仅在身体参考姿态已确定、静止采样稳定后，才按代码注释从平均重力估计 roll/pitch，保存修改前后值并重新核验；不要把身体真实倾斜抵消掉，也不能仅凭重力估计 yaw。此常量不是自动陀螺仪零偏标定工具。

关节卸载时依次手动小幅移动，检查实际 ID、回传关节和正负方向。归档固件 `MOTOR_SIGN` 全为 -1，不能因左右对称就自行把右侧取反。发现映射或方向不一致时先排查装配与代码契约，不用改奖励函数补救。

### 8.5 官方策略验收

| 顺序 | 操作 | 通过条件 | 保存内容 |
| --- | --- | --- | --- |
| 1 | Inference only，观察静止与手动姿态变化 | 无电机驱动，舵机与 IMU 在线且反馈持续更新 | 网页截图、状态数据、应用日志 |
| 2 | 支撑下检查 Torque off | 操作后卸载，实际状态与网页一致 | 短视频 |
| 3 | 支撑下 Default pose | 关节向预期站姿移动，无反向、卡死或异常目标 | 正面与侧面视频、目标与反馈 |
| 4 | 官方 Walk / get up，先零速度再短时低速 | 支撑撤除后能站稳，停止和卸载可控 | 完整试走和失败片段 |
| 5 | 重复停止、前后移动和转向 | 多次表现可复现，记录跌倒与恢复 | 指令、时长、成功次数、温度及供电 |

这里的低速与试验时长将在实际网页和场地核验后固定。**Walk / get up** 包含自动跌倒恢复，启用前要知道其可能带来额外动作；首次不同时测试 Pick、Roulade 或其他行为。**Zero motion** 只是清零运动指令，**Torque off** 才卸载。离开 Control 页面也不等于关闭电机。

出现反向、卡死、失联、异常发热或错误目标时，在支撑条件下卸载并保留日志，先找对应原因。修复后从失败的前一级重新验证，不反复盲目点击 Walk。

### 8.6 保存板端反馈

板端记录工具已完成软件检查，尚未连接真实 UNO Q，也没有取得现场反馈。工具 [capture_board_status.py](../../scripts/validation/capture_board_status.py) 只访问 `GET /api/status`，不发送模式切换、卸载、ID、中位、增益或运动命令。它用于首次运行时记录通信和推理活动；机械中位和步态仍按前面的小节分别验证。

实际使用顺序如下：

1. 在 App Lab 中确认板卡与程序已经运行，按网页选择 **Inference only** 并确认卸载；复制 App Lab 显示的板卡地址。保持机器人有稳定支撑。
2. 在 VS Code 运行 **17 板端状态采集（空地址仅自检）**。输入真实地址，例如 `http://板卡IP:9527`；尚未接板时留空，只运行软件自检。工具不会搜索或猜测板卡地址。
3. 有地址时采集约 15 秒，终端逐次显示模式、使能标志、在线 ID 数、反馈年龄及接收/推理计数。它直接访问该局域网地址，忽略本进程的代理环境变量，不改系统代理设置。
4. 查看打印出的记录目录。采集结束后再进入 Calibration 或 Servo setup；一旦观察到模式离开 shadow 或使能标志为真，工具会结束采集并留下记录，但不会替你切换模式或卸载。

也可在同一个 Linux / WSL 集成终端执行以下命令，按提示粘贴 App Lab 地址；不要把示例中的“板卡IP”原样输入：

```bash
read -r -p 'App Lab board URL: ' XGO_BOARD_URL
bash scripts/training/vscode_task.sh board_status 42 "$XGO_BOARD_URL"
```

| 记录 | 内容 |
| --- | --- |
| `records/hardware/status_<时间>/samples.jsonl` | 每次 HTTP 响应原始字节的 Base64、解码状态、主机单调时钟和错误 |
| 同目录 `summary.json` | 缺失 ID、过期反馈、异常模式、计数变化、重启迹象与判定 |
| 同目录 `capture_board_status.py` | 本次采集脚本副本；参数及源码、数据哈希写入摘要 |
| `records/hardware/status-tool/console-<时间>.log` | VS Code 入口的完整输出与退出码 |

`telemetry_ready=true` 只表示这段采样满足工具的初步反馈检查：15 个预期 ID 被报告、IMU 和相关就绪标志有效、反馈年龄小于归档程序使用的 150 ms 界限、模式为 shadow 且未报告使能、数值有效、接收和推理计数推进，以及未发现重启或新增坏帧/序列丢失。退出码 2 表示需要检查记录或采集被中断。它没有验证真实力矩、机械零位、舵机总线上不存在重复 ID、50 Hz 控制性能或稳定行走。

尤其不要只看 `control_ready`：归档实现只检查 IMU、反馈年龄与页面模式，不要求所有舵机在线；缺少 ID 12 的离线样例中它仍为真，而 `full_robot_ready` 为假。新工具同时核对完整 ID 列表、缺失列表与时间，避免沿用这个较弱条件。上位机约 5 Hz 的 HTTP 采样率也不等于 MCU 反馈率；摘要中的接收/推理频率按板端计数差与板端 uptime 差计算。推理延迟只汇总接口报告的滚动统计，不把多个窗口的 p99 平均成新的整体 p99。状态中的 `gyro` 单位是 degree/s，策略使用的滤波后角速度另为 rad/s。

本轮使用真实归档 `Controller.status()` 产生字段结构，反馈、时钟与计数均为明确标记的合成样例。18 种状态场景及 5 种无效响应检查通过，包括缺 ID、过期、错误模式、非有限值、计数停滞、重启和正常 uint32 回绕；假 HTTP 请求检查确认只有 GET 且不携带请求体，重定向被拒绝。测试未启动控制循环、网络请求或 Bridge 通信。证据见 [软件检查](../../records/hardware/status-tool/software-check.json) 和 [可见终端日志](../../records/hardware/status-tool/console-20261101T022403.log)。实际 HTTP 链路与板端表现仍待实物接入验证。

![状态工具](../../records/hardware/status-tool/vscode-status-check.jpg)

## 9. 自训模型迁移到真机

### 9.1 部署前必须对齐的接口

| 项目 | 当前事实 | 处理方式 |
| --- | --- | --- |
| 策略顺序 | 左腿 10–14、头颈 30–33、右腿 20–24 | 与 MCU 总线顺序分开核对 |
| 维度 | 61 维观测、14 维动作；嘴部单独控制 | 维度正确后仍需逐项核对语义 |
| 默认姿态 | 新训练约 ±24°；归档普通行走约 ±22° | 观测减去的 home 与动作加回的 home 同时匹配模型 |
| 模型元数据 | 导出包含关节名、home 和动作比例 | 保存到交付契约，不只复制 ONNX |
| Runtime | PC 1.22.1 兼容检查已通过 | UNO Q 上重新检查 ARM 推理、延迟及抖动 |
| 时序 | 策略目标约 50 Hz，反馈目标约 100 Hz | 板端实测频率、状态年龄和超时次数 |
| 滤波与增益 | 运行代码存在动作平滑和舵机增益逻辑 | 记录实际值，再做含延迟/滤波的仿真回归 |

归档 `Policy.load(..., use_model_home=False)` 普通行走默认不读取模型 home；不能仅替换 `xgoduck_walk.onnx` 就认为 ±24° 已自动生效。应在独立运行副本中为候选模型明确选择与模型一致的 home，并验证观测与动作映射。归档动作滤波切片还把策略索引 9 的右髋偏航纳入头部滤波段，这个实现差异也要记录和评估，不能无记录地修正后仍称同一部署基线。

### 9.2 分步迁移

1. 先保存通过验收的官方运行副本、模型、配置、哈希与标定记录，作为可回退的整套基线。
2. 选取通过仿真验收的 checkpoint，导出 ONNX，检查数值一致性和 ONNX 驱动闭环；短训练模型不进入此阶段。
3. 建立独立候选部署副本，修改模型路径与对应契约。保存完整差异，不覆盖 `official_resources/`。
4. 在 UNO Q 保持 Inference only，核验模型加载、真实观测下的有限输出、推理时延和反馈状态。频率不足先定位推理或通信瓶颈。
5. 检查默认姿态与每个关节目标，再完成支撑、低速和平地重复验收。条件与官方实机基线一致，才能比较改动效果。
6. 失败时停止应用，恢复已验证的模型和配套配置，再从 Inference only 重新检查。若改过舵机中位或增益，单纯恢复 ONNX 不能恢复硬件设置。

每次实机实验记录模型哈希、源码提交、零位标识、电池状态、地面材质、指令、反馈频率、延迟、试验时长、跌倒与温度。根据真实偏差再决定是否增加质量、摩擦、执行器延迟和传感器误差的仿真随机化；先定位偏差，不靠扩大随机化掩盖接线或标定错误。

## 10. 证据索引与更新规则

- [安装时间线](../../records/deployment/2026-10-31/README.md)：失败、修复、镜像同步和环境检查。
- [基线训练检查](../../records/experiments/baseline_smoke_20261031T230120/training-summary.json) 与 [导出检查](../../records/experiments/baseline_smoke_20261031T230120/export-validation.json)。
- [修改组训练检查](../../records/experiments/candidate_smoke_20261031T230835/training-summary.json) 与 [导出检查](../../records/experiments/candidate_smoke_20261031T230835/export-validation.json)。
- [工作副本完整性](../../records/experiments/workspace-integrity.json)：归档文件不变，仅实验指定源码变化。
- [官方运行说明](../../official_resources/xgoduck_runtime_arduino/README.md)、[标定实现](../../official_resources/xgoduck_runtime_arduino/sketch/sketch.ino)、[角度常量](../../official_resources/xgoduck_runtime_arduino/sketch/duck_config.h)、[策略适配](../../official_resources/xgoduck_runtime_arduino/python/policy.py)。
- [官方装配指南](../../official_resources/xgoduck_hardware/docs/BUILD_GUIDE.md) 与 [装配图](../../official_resources/xgoduck_hardware/Assembly_Guide.pdf)：机械中位资料不足之处按第 8 节处理。
- [微软 WSL 安装](https://learn.microsoft.com/en-us/windows/wsl/install)、[镜像网络](https://learn.microsoft.com/en-us/windows/wsl/networking#mirrored-mode-networking)、[清华 PyPI 帮助](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)。
- [上游运行程序](https://github.com/LuwuDynamics/xgoduck_runtime_arduino)：2026-11-01 核对，已与本地归档存在差异，部署前另行固定实际提交。

本文持续记录已执行事实。每轮训练或真机操作完成后，更新状态、对应结果表、截图/视频和异常原因；保留原始实验，不把旧失败覆盖为成功。阶段操作说明作为参考附录，本报告保持唯一的全流程阅读入口。


## 11. 阶段交付

| 阶段 | 工作与交付 | 当前状态 |
| --- | --- | --- |
| P0 | 环境、短训练、导出与可见视频流程 | 已通过流程验证 |
| P1 | 训练效率诊断与优化、官方预训练模型参考评估 | 参考评估、适配诊断和本机规模筛查已交付；4096 环境按冻结规则入选，原生系统比较尚未进行 |
| P2 | 官方配置充分训练与稳定性验收 | 5000 次基线完成训练和导出，固定评估仅停止通过；当前追加至 7500 次，稳定步态待验收 |
| P3 | 同预算、多 seed 修改组比较 | 待基线稳定性验收 |
| P4 | 真机首次运行、ID/零位/IMU核验、舵机顺序变更分析 | 源码映射分析与离线检查通过；真机尚未运行 |
| P5 | 官方实机策略逐级验证 | 未执行 |
| P6 | 自训策略 sim2real、接口对齐与回退验证 | 未执行 |
| P7 | 全流程报告与复现材料核验 | 主报告持续更新，最终验收未完成 |

每个阶段交付结果和证据后单独核验，总目标仅在全部对应要求实际满足后完成。最新目标与阶段条件见 [目标清单](../current_goal.md)。舵机顺序研究与离线映射检查见第 14 节；实际 ID、方向、零位仍待逐关节验证，当前未写入舵机或改变物理接线。

## 12. 用 VS Code 自己操作

VS Code 入口最初在暂停期间准备，当前按第 21 节的追加训练状态使用；第 18 节记录已完成的首段续训。Windows 项目已在 VS Code 打开，任务配置位于 `.vscode/tasks.json`；它调用 WSL 的 Linux 脚本，不依赖 Windows PowerShell 执行策略。原生 Ubuntu 下同一任务配置会改用本地 Bash，但该系统环境尚未核验。

### 12.1 先看状态

1. VS Code 中选择“文件 → 打开文件夹”，打开 `C:\Users\atlas-win\Desktop\RL-Project\XGO-Duck`，不要只打开一个 Markdown 文件。
2. 打开 `docs/current_goal.md` 查看当前阶段和执行约束；打开本报告 Markdown 后可用 `Ctrl+Shift+V` 预览。需要播放嵌入视频时，用浏览器打开同目录 `development_report.html`。
3. 训练占用终端时直接查看现有输出，暂不运行其他任务。任务结束后，可选择“Terminal → Run Task”，搜索 `XGO-Duck`，运行 **01 查看状态（不训练）**；应看到与 `project.json` 相符的训练状态、恢复 checkpoint 路径和未启动新训练的提示。
4. **02 更新图文报告** 仅重新生成 HTML 并检查引用；不会训练。

状态入口已经通过实际命令核验。任务配置的 Windows 参数转换和 Linux 入口均需保持在项目目录；脚本遇到训练暂停状态会在创建训练进程前退出。暂停期间曾执行基线入口命令，确认它按当时要求拒绝启动，见 [暂停检查记录](../../records/reports/vscode-pause-guard-check.json)。第 15 节的推理评估及下面的 09、10、11 三个任务也在暂停期间通过 VS Code 界面执行，顺序复用同一个项目任务终端；实际续训另见第 18 节。

| 任务 | 用途 | 应看到的结果 |
| --- | --- | --- |
| 09 暂停实验效率分析（不训练） | 读取现有日志、事件和媒体，生成统计与图 | `status: passed`，260 次完整迭代及图像路径 |
| 10 官方模型接口复核（不训练） | CPU 重放已保存输入，核对归档 runtime 接口 | `status: passed`，640 个样本、输入/输出误差及滤波结果 |
| 11 Linux 环境核查（不安装） | 记录发行版、GPU、项目依赖与文件系统 | `host_kind`、环境记录路径、`training_started: false` |
| 12 适配诊断预检（不训练） | 检查动作滤波诊断的 2 环境、2 秒流程 | 5 个条件完成、指标/视频复核通过，再做 runtime 数学核对 |
| 13 适配单因素对照（不训练） | 顺序执行参考、home、滤波和增益四组 | 每组独立记录，最后输出配置差异、runtime 复核与对照 CSV |
| 14 续训恢复检查（不训练） | CPU 核对原加载行为，再验证项目续训修正及暂停保护 | 两份检查通过，编号 251、剩余 4749 次，`RESUME_AUDIT_EXIT_CODE=0` |
| 15 复核已有适配诊断（不仿真） | 读取已完成的四组轨迹，复核配置、动作与指标汇总 | `VERIFICATION_EXIT_CODE=0`；不重新创建模拟器 |
| 16 继续已保存基线（受暂停限制） | 从固定恢复点补足累计预算，创建独立续训记录 | 本次已获授权并运行；不重复启动，暂停时拒绝执行 |
| 17 板端状态采集（空地址仅自检） | 地址留空做软件检查；填写 App Lab 地址后只读采集反馈 | 区分 `board_contacted: false` 的自检与实际 HTTP 记录，不自动控制机器人 |
| 18 指令幅度诊断（不训练） | 同一模型比较五档前进与左转指令，保存轨迹和视频后复核 | `AMPLITUDE_DIAGNOSTIC_EXIT_CODE=0` 表示流程通过，不替代原低速步态验收 |
| 19 导出已完成的续训基线 | 核验最后 checkpoint 和总预算，导出 ONNX，比较 1000 帧输出 | 训练完成后才能执行；入口预检通过，实际导出尚未执行 |
| 20 评估已导出的续训基线 | 使用冻结的开发条件，录制五段固定指令视频并独立复算 | 任务 19 成功后执行；数值通过与流程完成分别记录 |
| 21 已选基线留出评估 | 固定已选模型后使用 seed 2001，保存一次尝试并独立复算 | 任务 20 的全部开发条件通过后执行；同一训练来源仅一次 |

09、10 需要已经装好的项目 Python 环境；11 只需系统 Python，适合切换原生 Ubuntu 后先运行。缺少 GPU 工具或项目环境时，核查会如实记录缺失，不自动安装。任务 09 重新绘图后，自动检查会重建记录，人工看图一项须针对新文件再确认。

### 12.2 阅读实验注释

| 实验入口 | 代码注释说明 |
| --- | --- |
| `scripts/training/experiments/official_baseline.sh` | 保持官方奖励、物理和 PPO；为什么需要基线、控制哪些参数、怎样验收 |
| `scripts/training/experiments/tracking_reward.sh` | 线速度奖励 3.0 → 3.3 的动机、可能代价、控制变量和指标；要求先通过基线验收 |
| `scripts/training/experiments/resume_baseline.sh` | 保留原基线配置，说明保存编号、剩余预算、学习率恢复与重新开始的环境状态 |

每次从这些入口启动时，辅助脚本会把带注释的入口保存为实验记录中的 `experiment-entry.sh`，并保存实际训练启动脚本、协议和配置。续训还保存适配代码及来源哈希。这样以后修改注释或代码，也不会改变旧实验记录。官方原始归档不添加个人注释，说明放在独立实验入口和补丁旁。

### 12.3 恢复和从头训练的区别

当前完整恢复点已经另存到 `artifacts/recovery/baseline_formal_20261101T001206_s42/model_250.pt`，与 Linux 原文件 SHA-256 一致，包含 actor、critic、optimizer 和迭代信息。暂停记录见 [pause-record.json](../../records/experiments/baseline_formal_20261101T001206_s42/pause-record.json)。

**04 官方基线从头训练（非续训）** 会创建新的实验，不会自动读取这个 checkpoint。**16 继续已保存基线（受暂停限制）** 则读取固定恢复点，创建单独的 continuation 实验。续训入口先通过 CPU 状态与配置检查，随后在本次用户授权下通过真实环境 reset 和初始 rollout 核验，见第 18 节。项目被用户暂停时，两项入口均拒绝启动；当前续训已经运行，不应重复执行任务 16。

已使用当前安装的实际 runner，在 CPU 上加载 actor、critic、观测归一化和 Adam 状态，并逐项与 checkpoint 比较；环境部分仅用状态容器接收计数，没有创建模拟器、计算梯度或执行更新。结果见 [恢复检查](../../records/experiments/baseline_formal_20261101T001206_s42/resume-analysis/resume-check.json)，源码副本保存在同目录 `sources/`。这项检查发现：

| 项目 | 实际结果 | 续训时的处理 |
| --- | --- | --- |
| 保存编号 | `iter=250`，在该次更新完成后保存；实际包含 251 次更新 | 下一次日志编号应为 251；达到累计 5000 次还需 4749 次 |
| 采样计数 | `common_step_counter=6024`，等于 251 × 24；归一化计数为 6,168,576 | 保留此全局计数，并核验首次 reset 后的课程参数 |
| Adam 状态 | 每个参数的更新计数为 5020，等于 251 × 5 epoch × 4 minibatch | 动量与方差已在 CPU 精确载入；初始 rollout 与学习日志通过，后续 checkpoint 状态另验 |
| 学习率 | Adam 载入值为 0.000170859375；PPO 自适应学习率变量仍为配置初值 0.001 | 首次自适应更新前须把 PPO 变量同步到已恢复的 Adam 学习率 |
| 官方迭代参数 | `max_iterations` 被传给“额外执行次数”，加载后仍从已保存编号开始循环 | 不能直接用 `--resume` 加 `--agent.max-iterations 5000` 表示补足总计 5000 次；须同时处理编号与剩余预算 |
| 未保存状态 | checkpoint 没有完整随机数、物理状态、每环境指令、噪声/延迟和录像状态 | 新建续训记录并链接恢复点，明确环境重新开始，不能称为逐步完全相同的断点恢复 |

暂停前日志最后为 259，编号 251–259 的 9 次更新未保存，不能从奖励日志重建。上述 CPU 检查只核对加载行为；后续真实续训证据另存于第 18 节，不回写暂停记录或原 checkpoint。

需要自行复查暂停准备时，可在项目确实处于 `paused-by-user` 状态时选择 **14 续训恢复检查（不训练）**，或在同一 VS Code 终端运行以下命令。该检查要求暂停状态，当前续训过程中应使用第 18 节的只读进度检查，不能为运行检查而擅自改写状态：

```bash
bash scripts/training/vscode_task.sh resume_audit
```

任务 14 先保留官方 loader 的原始检查，所以其中的 `learning_rate_variable_matches_optimizer: false` 仍表示未加适配时的行为。随后执行项目的 [续训适配检查](../../scripts/validation/verify_resume_preparation.py)，使用实际 CPU loader 与同一 checkpoint 核对修正。两份记录分开保存，避免把官方原始行为改写成已修正行为。

续训代码位于 [状态修正](../../scripts/training/resume_state.py) 和 [训练入口](../../scripts/training/resume_official.py)。它只调整下一次编号、剩余预算与 PPO 自适应学习率变量；actor、critic、归一化和 Adam 动量保持已载入值。除实验名称和剩余次数外，构造的环境与 agent 配置必须与保存的基线配置一致，才允许进入模拟器。当前 CPU 结果为编号 `251`、补足 `4749` 次、最后编号 `4999`、学习率 `0.000170859375`。完整证据见 [适配结果](../../records/experiments/baseline_formal_20261101T001206_s42/resume-analysis/preparation/resume-preparation-check.json)。

环境也有一处恢复顺序需要处理：官方 `RslRlVecEnvWrapper` 在加载 checkpoint 前已 reset，此时课程计数仍从零开始。项目入口加载计数 `6024` 后再 reset；源码确认课程计算先于指令重采样。本次真实 reset 已执行，计数和课程参数保存于第 18 节的 `resume-ready.json`。随机状态、动作延迟历史和 episode 均重新开始，不能把续训描述为精确重放中断前的后续轨迹。

CPU 配置核对最初发现两项序列化差异：保存配置中的 terrain 环境数已经由场景构造传播为 `1024`，而刚构造的配置仍是 `1`；相机俯角还存在整数 `-20` 与浮点 `-20.0` 的表示差异。入口现按官方派生规则同步 terrain 数量，并保持相机字段类型；没有忽略整个 scene 或放宽物理参数检查。

更新后的任务 14 已在同一个 VS Code 项目终端执行，日志以 `RESUME_AUDIT_EXIT_CODE=0` 结束，见 [完整输出](../../records/experiments/baseline_formal_20261101T001206_s42/resume-analysis/console-20261101T021711.log)。四种不一致计划和重复应用均被拒绝；Bash 任务、训练监督脚本和实际续训脚本的三层暂停检查均返回暂停提示，没有创建新训练目录，也没有初始化 CUDA 或执行优化器更新。

![续训准备](../../records/experiments/baseline_formal_20261101T001206_s42/resume-analysis/preparation/vscode-preparation.jpg)

本次是在用户明确恢复、项目状态同步之后选择任务 **16** 启动。等效 Linux 入口如下，供复现恢复流程时参考；本次 5000 次训练已完成，不要重复启动同一实验：

```bash
bash scripts/training/vscode_task.sh resume_baseline 42
```

应先出现 `RESUME_READY` 及上述编号、计数和学习率，然后开始后续迭代。`training-request.json` 保存父实验和恢复点，`resume-plan.json` 保存预算与配置比较，`resume-ready.json` 保存首次 reset 信息。原 checkpoint、暂停记录和旧实验目录保留；新实验继续保存真实日志、曲线和视频。初始更新已验收，最终总预算和步态仍须分别核验。正式奖励消融须让两组采用一致的启动/恢复规则，并将中断损失的计算单列，不能计成算法提速或改进。

该入口目前针对这次已登记的 `model_250.pt`。如果未来再次中断，先登记并核验新的 checkpoint、参数和计数，再建立下一段恢复记录，不能反复从旧恢复点启动并把它当作接续最新训练。

恢复检查命令入口已实际运行，任务 14 随后也在诊断结束后通过同一 VS Code 任务终端执行。底层 `visible_training.py` 增加了暂停检查，直接调用时在创建实验目录或子进程前退出；实测返回暂停提示，未创建新训练记录。见 [入口核验](../../records/reports/resume-entry-check.json)。

![恢复检查](../../records/experiments/baseline_formal_20261101T001206_s42/resume-analysis/vscode-resume-check.jpg)

恢复须先记录用户明确指令，再更新项目状态并运行对应任务，本次已按此执行。**05 奖励修改组从头训练** 还要求稳定基线已验收、两组运行环境和启动/恢复安排已核验，以及源码匹配候选补丁。共享终端配置和训练锁用于避免多任务并发；不设置打开文件夹后自动训练。

### 12.4 怎样看过程与修改

先读每个实验的注释与冻结协议，再改一个因素，保存补丁和假设。终端查看迭代、采样耗时、更新耗时和异常；浏览器查看真实 TensorBoard 曲线与阶段采样视频；正式评估再看固定指令下的完整视频和数值。每轮记录“预期是什么、实际发生了什么、与基线差在哪里”，不能只保留看起来好的片段。

官方预训练参照的操作、终端截图、曲线和视频见第 15 节；实际续训见第 18 节。受控效率实测和多 seed 对比尚未完成，全部自主复现验收仍未完成。

## 13. 原生 Ubuntu 选择

用户确认本机是 Windows / Ubuntu 双系统。当前原生 Ubuntu 的发行版、GPU 驱动和训练依赖尚未检查，本轮没有切换系统或重启。原生系统是否更快，以同机受控测试为准，不从 WSL 的单次利用率推断。

### 13.1 迁移材料

切换前保存项目中的官方源码和锁文件、实验入口与补丁、配置、checkpoint、日志、媒体及哈希。当前 checkpoint 已复制到 `artifacts/recovery/baseline_formal_20261101T001206_s42/` 并核验；原始 TensorBoard 事件也已复制到该实验的 `records/experiments/` 目录，避免只带走曲线截图。保留原 Windows/WSL 文件作为回退，不删除旧环境。

原生 Ubuntu 中将项目资料复制到可写的 Linux 文件系统，VS Code 打开复制后的项目根目录。训练运行副本仍由脚本建立在 `$HOME/XGO-Duck/workspace/xgoduck_rl`，缓存与 Python 工具位于同一专属根目录。不要复制 WSL 的 `.venv` 或显卡驱动；它们需要在目标系统按锁定依赖重新建立。旧记录中的绝对路径保留为历史来源，新运行另存实际路径。

复制完成后，在**复制后的项目根目录**运行以下命令。它只检查已有文件，不安装或启动训练。两个文件都应输出 `OK`；任意一项不匹配时，重新检查复制来源，不用改校验值使检查通过。

```bash
sha256sum --check <<'XGO_CHECKSUMS'
0ef769135d86e63c03663b2df570c0217217b541ab234460a879dff8fb260fa9  official_resources/xgoduck_rl/uv.lock
6f521e883f0c192f96fea2f7d0dfe3a600abf7e2dbd39bd3a64fd5997a854d6d  artifacts/recovery/baseline_formal_20261101T001206_s42/model_250.pt
XGO_CHECKSUMS
```

这是锁文件与恢复点的检查，不代表全部媒体与源码都已复制完整；环境入口还会依据官方文件清单核验训练源码。当前这两项已在 Windows 项目副本重新计算并相符，上述命令在 WSL 对同一副本执行也均为 OK，原生系统复制后的检查尚未执行。不要只复制已提交的 Git 内容：本轮新增的脚本、记录和报告也属于迁移材料。

旧评估记录里的 `model` 字段含有当时的 WSL 绝对路径，应保留为来源，不批量改写历史记录。接口复核和保存输入的指令敏感性脚本现默认读取**当前项目**的归档行走模型，再校验其 SHA-256 必须与原记录相同；检查其他模型时通过 `--onnx` 显式指定，哈希不匹配就停止。这修复了旧入口在换盘符或切换系统后找不到模型的问题，VS Code 任务 10 使用的也是该修正入口。

路径迁移已经使用真实轨迹做过 CPU 检查：把必要模型、代码和五条轨迹复制到 `workspace/` 下的临时项目，将副本中的历史模型路径设为不可用，再执行实际脚本。640 个接口样本、1000 步滤波重放和指令探针数值与原记录一致；两项错误模型检查均被拒绝。原始文件的前后哈希相同，临时副本已清理，见 [迁移读取检查](../../records/reports/reference-relocation/check.json)。这不是原生 Ubuntu、GPU 或新环境安装的验收。

在项目根目录可以复查该流程；使用已有 Python 环境，不安装软件或运行仿真：

```bash
"$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" -B \
  scripts/validation/verify_reference_relocation.py
```

### 13.2 先核查环境

在 VS Code 选择 **11 Linux 环境核查（不安装）**。同一任务在 Windows 调用 WSL，在原生 Ubuntu 调用本地 Bash；也可从原生项目根目录执行：

```bash
bash scripts/training/vscode_task.sh host_profile
bash scripts/setup/bootstrap_linux.sh --plan-only
```

第一条生成 `records/deployment/<日期>/host-profile-<系统>-<时间>.json`，记录发行版、内核、CPU、内存、GPU/驱动、磁盘文件系统及现有项目依赖；第二条只显示安装计划，不创建环境。先查看 GPU 是否识别、目标目录是否为 Linux 文件系统、空间是否足够，再安排实际安装。

该核查已在当前 WSL 的 VS Code 集成终端执行，结果为 Ubuntu 24.04.5、16 个逻辑 CPU、WSL 可用总内存约 31.27 GiB、RTX 3080 Laptop 16 GB、CUDA 可用；Windows 项目目录为 9p 挂载，实际训练副本在 ext4。主机物理内存约 64 GB 与 WSL 当时分配到的内存不是同一量。见 [环境快照](../../records/deployment/2026-11-01/host-profile-wsl-011242.json)。这是 WSL 结果，不能代替原生系统核验。

![环境核查](../../records/deployment/2026-11-01/vscode-host-profile.jpg)

### 13.3 建立锁定环境

原生系统已确认 GPU 和必需工具可用后，在同一个可见 VS Code 终端执行：

```bash
bash scripts/setup/bootstrap_linux.sh --mirror
```

入口先核验归档，再建立独立训练副本；缺少 uv 时从官方地址安装固定的 uv 0.13.6，镜像同步使用 Python 3.12。清华镜像仅替换锁文件的下载地址，保留依赖及包哈希，完成或异常退出后恢复官方锁文件。它会做 CUDA 与任务入口检查，不启动训练；命令、输出和退出码写入 `records/deployment/<日期>/linux-bootstrap-*.log`。已有 uv 时先记录实际值，受控对比前确认与 WSL 一致。

本轮检查修正了旧入口在 uv 尚不可用时就执行缓存迁移的问题。新入口只建立项目专属缓存，不移动用户共享缓存；历史迁移脚本和旧安装记录保留。Bash 语法与 `--plan-only` 实际通过，见 [入口检查](../../records/reports/bootstrap-entry-check.json)。本轮没有执行新的安装，原生 Ubuntu 的全新环境安装仍待实测。若 GPU 核查失败，先依据目标系统与驱动的实际输出处理，不在两个系统间照搬驱动命令。

### 13.4 怎样公平比较

当前已按用户明确指令在 WSL 续训官方基线，系统性能对照安排在这次基线结束之后，系统切换仍由用户安排。协议已保存为 [同机性能比较](../../configs/experiments/host_performance_comparison.json)，目前没有运行这些试验，也尚未提供已实测的性能试验启动任务。

| 条件 | 对比方法 |
| --- | --- |
| 固定计算量 | 同一官方源码、锁文件和 PPO 配置，1024 环境、每迭代 24 步；seed 42、43、44 分别配对 |
| 固定计时区间 | 每次从头运行 300 次迭代，前 20 次仅作热身；主计时覆盖编号 20–299 的完整 280 次迭代，共 6,881,280 次环境转移；另报启动、首次编译和进程总耗时 |
| 保持可见 | 两边都使用一个可见 VS Code 终端、相同日志和曲线展示方式；正式视频配置均为每 6000 步录 200 帧 |
| 分离录像成本 | 每个系统另做一次 seed 42 的关闭录像试验，与相同 seed 的主试验配对；主比较保留录像与保存开销，不剔除慢点 |
| 记录系统差异 | GPU、驱动、依赖、磁盘、供电、温度/功耗、GPU SM 时钟、CPU/GPU 利用率和显存；驱动不能完全一致时注明混杂因素 |
| 分开改并行数 | 先固定 1024 环境比较系统；环境数扫描另做试验，同时报告总转移数与更新次数 |
| 判定收益 | 列出三个 seed 的“原生吞吐 / WSL 吞吐”及几何均值；均值至少 1.15、三个比值均大于 1 且全部运行正常时，优先选择原生系统；否则继续依据瓶颈决定 |

上述 15% 是本项目在试验前设定的实际使用标准，不是官方要求。每次完整短试验为 7,372,800 次环境转移，不能将 300 次迭代与 5000 次正式训练混记。采样与更新计时用于解释来源，主吞吐必须按整个固定计时区间计算；录像、保存和慢点都计入。两套系统的驱动、内核或供电策略若不同，结果代表两套实际环境的综合差异，不能单独归因于虚拟化。三个 seed 的结果逐个公开，不据此声称统计上普遍成立。

协议中的源码哈希、锁文件哈希和预算算术已经核对，见 [协议检查](../../records/reports/host-protocol-check.json)。第 18.3 节观测到不同时段的 GPU 时钟有差异，因此在系统对照开始前明确增加 GPU SM 时钟这一必录字段；其他条件和判定标准保持不变。[原协议](../../records/reports/host-protocol-initial.json)及其[检查记录](../../records/reports/host-protocol-initial-check.json)保留用于追溯。这项准备没有运行主机性能试验，也没有核验原生系统。

当前最有依据的排查顺序是采样内部耗时、周期性尖峰和录像成本，再比较 WSL / 原生 Ubuntu。若原生系统提速明显且运行稳定，再选择它完成充分训练；若没有明显收益，保留已验证环境继续排查。性能试验与正式 5000 次训练分别记账，不能把短基准结果当成完整训练结果。

## 14. 更换舵机顺序

只改变同一总线上的接线先后、保留每个物理关节的 ID，通常不需要修改策略或重新训练：归档驱动用 ID 寻址，并按回包 ID 找回数组位置，而不是按电缆上的先后来判断关节。这是从 [scs_bus.h](../../official_resources/xgoduck_runtime_arduino/sketch/scs_bus.h) 的 `motorIndexById`、同步写入和反馈处理推导出的软件结论，供电、接头与通信质量仍须实测。若“换顺序”指改变 ID、软件数组或机械结构，影响如下。

### 14.1 改动范围

| 变更 | 难度 | 修改与验证 | 标定或训练 |
| --- | --- | --- | --- |
| 同一总线重新整理接线，关节与 ID 不变 | 低 | 一般不改代码；重接后核验 15 个 ID、反馈连续性和逐关节对应 | 不因接线先后重训；未拆动安装位置时复核原标定 |
| 将装错 ID 的舵机恢复为官方关节 ID | 低至中 | 使用官方 Servo setup 单舵机改 ID，重新扫描；源代码保持官方映射 | 改 ID 本身不替代机械零位；复核方向与中位，不因改名重训 |
| 保留自定义 ID 分配 | 中 | 同步 Python、MCU、校准页面、嘴部特例及相关脚本；观测和目标必须使用同一映射 | 只改变地址且物理行为等价时可复用策略；逐关节验证必需 |
| 重排策略的 14 个输出/观测位置 | 中至高 | 在策略边界做显式置换，连同位置、速度、上一动作、home、尺度和滤波核对；旧 ONNX 的语义顺序不能随便改 | 完整且等价的适配不要求重训；改模型接口或训练定义则须重新导出并评估 |
| 翻转安装方向或改变关节轴 | 中至高 | 先区分坐标正负变化与真实机构变化；检查角度、速度、目标方向、机械限位和 home | 纯坐标变换可适配；轴线、连杆或运动范围变化需改仿真并重新评估，可能重训 |
| 换不同型号、减速比或性能的舵机 | 高 | 核对协议、编码器单位、速度单位、扭矩/速度/延迟/摩擦/间隙及增益，更新执行器模型 | 重新标定并做响应测试；根据仿真与实机偏差决定重训或微调 |

“难度”是按当前源码耦合范围的工程判断，不是已完成改造的工时统计。当前最省改动的路径是保留官方关节 ID 和策略语义，先完成官方实机基线。下表用于识别关节，不能替代机械中位照片。

### 14.2 关节映射

策略中的左右以机器人自身为准；索引从 0 开始。“报文槽位”也是 MCU 与校准命令使用的关节索引。关节名称来自行走 ONNX 元数据，并与归档仿真 XML 的执行器顺序核对。

| 策略索引 | 关节 | ID | 报文槽位 |
| --- | --- | --- | --- |
| 0 | 左髋偏航 left_hip_yaw | 10 | 0 |
| 1 | 左髋侧摆 left_hip_roll | 11 | 1 |
| 2 | 左髋俯仰 left_hip_pitch | 12 | 2 |
| 3 | 左膝 left_knee | 13 | 3 |
| 4 | 左踝 left_ankle | 14 | 4 |
| 5 | 颈部俯仰 neck_pitch | 30 | 10 |
| 6 | 头部俯仰 head_pitch | 31 | 11 |
| 7 | 头部偏航 head_yaw | 32 | 12 |
| 8 | 头部侧倾 head_roll | 33 | 13 |
| 9 | 右髋偏航 right_hip_yaw | 20 | 5 |
| 10 | 右髋侧摆 right_hip_roll | 21 | 6 |
| 11 | 右髋俯仰 right_hip_pitch | 22 | 7 |
| 12 | 右膝 right_knee | 23 | 8 |
| 13 | 右踝 right_ankle | 24 | 9 |
| 无 | 嘴部，独立控制 | 34 | 14 |

完整数值见 [关节映射 CSV](../../records/hardware/servo-mapping/joint-mapping.csv)，包含方向、两套 home 和滤波系数。归档固件的 15 个 `MOTOR_SIGN` 都是 -1。其目标换算为 `2047 + trunc(angle_deg / 360 × 4095 × sign)`，之后限制到编码器范围；反馈角度为 `(raw - 2047) × 360 / 4095 × sign`。同一方向约定同时用于输出、位置反馈和速度反馈，不能只给输出乘负号。

例如 ID 20 的反馈来自报文槽位 5，进入策略位置/速度索引 9；策略动作索引 9 经 home 与角度转换后，又写回报文槽位 5。若误以为两个数组顺序相同，就可能将右髋值当成颈部值。校准 ID 20 使用索引 5，不能传策略索引 9。

### 14.3 对应源码

| 文件 | 需要保持一致的内容 |
| --- | --- |
| [python/rl_core.py](../../official_resources/xgoduck_runtime_arduino/python/rl_core.py) | `MOTOR_IDS`、`POLICY_IDS`、`POLICY_INDEX`、`DEFAULT_DEG`、`MOUTH_INDEX`；观测抽取与动作写回 |
| [sketch/duck_config.h](../../official_resources/xgoduck_runtime_arduino/sketch/duck_config.h) | MCU `MOTOR_IDS`、`MOTOR_SIGN`、`MOUTH_MOTOR_ID`、编码器和零位常量 |
| [sketch/scs_bus.h](../../official_resources/xgoduck_runtime_arduino/sketch/scs_bus.h) | 按 ID 解包、目标角度换算、速度符号与单位、嘴部增益特例 |
| [assets/calibrate.js](../../official_resources/xgoduck_runtime_arduino/assets/calibrate.js) | 页面硬编码的 `MOTORS` 与按索引提交校准请求 |
| [python/wire.py](../../official_resources/xgoduck_runtime_arduino/python/wire.py) 与 [sketch/protocol.h](../../official_resources/xgoduck_runtime_arduino/sketch/protocol.h) | 15 槽位报文与反馈掩码约定；换排列不等于可以只改一端 |
| [python/policy.py](../../official_resources/xgoduck_runtime_arduino/python/policy.py) | home 选择、上一动作历史、按索引的滤波切片 |
| [xgoduck_constants.py](../../official_resources/xgoduck_rl/src/mjlab_microduck/robot/xgoduck_constants.py) 与 [robot_walk.xml](../../official_resources/xgoduck_rl/src/mjlab_microduck/robot/xgoduck/robot_walk.xml) | 更换执行器或机构时检查模型配置、关节轴/范围、姿态及动力学 |

特别注意：当前 `[5:10]` 采用头部滤波系数 0.2，包含策略索引 9 的右髋偏航；其余两段采用 0.45。此处已通过实际 `Policy.infer` 的受控输出检查，未擅自修改。普通行走 runtime 的髋/踝 home 约为 ±22°，官方行走 ONNX 元数据约为 ±24°；软件顺序检查分别使用两套 home，但这不证明两者步态等价。

真正实施自定义映射时，先从官方归档复制独立运行副本，写出“关节名 → 旧 ID → 新 ID → 报文槽位”的对照，再修改上述位置并搜索残留数字或索引假设。ID 是地址，不是关节的物理方向；单纯重新编号不应同时改变奖励、home 或滤波，否则无法判断问题来源。

### 14.4 自己核验

软件检查只读取文件和计算数组，不连接总线或改变设备。在 VS Code 的“Terminal → Run Task”选择 **06 舵机映射检查（不连接硬件）**。也可在 Windows 的 VS Code PowerShell 终端切到项目目录后执行：

```powershell
wsl.exe -d Ubuntu-24.04 --exec /home/atlas/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/validation/inspect_servo_mapping.py
```

原生 Ubuntu 环境准备好后，在项目根目录执行：

```bash
"$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" scripts/validation/inspect_servo_mapping.py
```

脚本输出 `status: passed` 表示本节范围的软件检查通过。实际记录为两套 home、14 个关节、正负两个方向，共 56 次探针；动作与观测往返最大误差为 `7.45e-9 rad`。另检查嘴部独立控制、15 槽位报文、MCU/Python/校准页面 ID 一致性、ONNX/XML 关节顺序和 14 个滤波位置。结果与源码哈希见 [检查记录](../../records/hardware/servo-mapping/mapping-check.json)。检查没有编译或运行 MCU 固件，也没有运行真实舵机或证明步态稳定。

真实设备的验证顺序如下，当前均未执行：

1. 按第 8 节启动官方程序，在卸载且机身受支撑时扫描，保存 ID 列表。扫描到 ID 不能证明它装在正确关节上。
2. 逐个手动小幅移动卸载关节，记录实际关节、变化的 ID/报文槽位和正负方向；同一次只移动一个关节，拍下对应位置。重名、缺失或错位先解决，再继续。
3. 若确实要改 ID，按官方 [Servo setup 页面](../../official_resources/xgoduck_runtime_arduino/assets/servo.html) 要求仅连接目标舵机，在断电状态整理接线；上电扫描、选中旧 ID、设置唯一的新 ID，再扫描确认旧 ID 消失且新 ID 出现。`controller.py` 会检查目标占用和新旧地址响应，但不能区分同一总线上重复地址的两个舵机。
4. 恢复整机接线后再次确认 15 个唯一地址和逐关节对应。按第 8 节建立机械中位、记录照片与读回值，断电重启复核；改 ID 成功不代表中位正确。
5. 先核验 Inference only 的目标和映射，再按支撑站姿、低速、停止的顺序试验。若某关节反向或到达机械限位，停止并核查该关节的映射/方向/中位，不用继续走路来判断。

最终记录每个关节的物理位置照片、原/新 ID、方向、机械中位证据、断电复核和对应模型/源码哈希。软件 CSV 的 `physical_verification` 当前全部为 `pending`，现场核验记录另存，不能把这份软件表直接改写成真机验收证据。

## 15. 官方策略固定评估

这一评估只运行 ONNX 推理与仿真，不更新权重，不读取优化器，也不连接机器人。正式训练仍保持暂停。参考对象为归档 `python/xgoduck_walk.onnx`，模型、运行脚本、协议和主要仿真源文件的哈希随结果保存；官方模型的训练预算未知，因此不作为同预算奖励消融组。

### 15.1 自己运行

在 VS Code 打开项目根目录，选择“Terminal → Run Task”：先执行 **07 官方策略评估预检（不训练）**，检查每种指令 2 个环境、2 秒的完整流程；预检成功后执行 **08 官方策略固定评估（不训练）**。两个任务复用共享终端，并与训练共用计算锁，避免同时占用 GPU。终端每 100 步输出进度、首个 episode 仍在进行的数量和耗时，每条指令结束后输出存活与跟踪指标。

固定协议为停止、前进 0.15 m/s、后退 0.10 m/s、左右转向 0.4 rad/s，每条指令 32 个环境、20 秒，开发 seed 1001。按 `Ctrl+C` 可中断当前评估，已完成的条件与日志保留；重新运行建立新记录，不覆盖旧结果。预检不用于稳定行走验收，留出 seed 2001 尚未使用。

每次输出位于 `records/evaluations/<run-id>/`：`evaluation.json` 为总状态，`env.yaml` 为初始评估配置，`protocol.json` 为本次协议副本，五个条件 JSON 保存逐环境结果，`*-trace.npz` 保存逐步观测、动作、速度、接触与终止标记，`videos/` 保存录像与起止帧。早期预检只保留了精简轨迹；完整评估新增完整观测和初始位置，以便独立复算有限值与漂移。控制台日志位于 `records/evaluations/console-<时间>.log`。

完成后可在 Windows 的 VS Code 终端重算结果。将最后的 `<run-id>` 替换为终端实际输出的目录名：

```powershell
wsl.exe -d Ubuntu-24.04 --exec /home/atlas/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/validation/verify_fixed_evaluation.py /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/records/evaluations/<run-id>
```

该检查从保存的轨迹重新计算指标，并完整解码五段视频；不会运行训练或再次仿真。成功后生成 `verification.json`。任务运行完成仅表示评估流程完成，是否达到标准必须看逐条件指标。

### 15.2 评估条件

以官方 play 配置为基础，保留其传感器噪声、观测延迟、编码器偏差、IMU 安装扰动和动力学随机化；实际参数以 `env.yaml` 为准。关闭外部推搡与课程调度，把速度指令固定、头部与身体指令置零。启动时的随机参数在同一次评估中保留，每种指令重置时使用相同 seed；这不等于测过多个训练 seed 或多个独立场地。

模型采用自身元数据中的 home 和动作比例。位置观测从仿真 home 转换到模型 home，动作从模型坐标转换回仿真目标，上一动作按模型原始输出保存。归档实机的动作滤波和 IMU 滤波没有加入本轮，结果属于训练仿真中的策略参考，不是整套 UNO Q 控制程序的等价复现。±22° 实机 home 与约 ±24° 模型 home 的差异仍按第 9 节处理。

### 15.3 指标定义

| 指标 | 计算方式 |
| --- | --- |
| 完整存活率 | 首个 episode 达到 20 秒且没有失败的环境比例；32 个环境至少 31 个完整存活才达到 95% |
| 跌倒数 | 官方 `fell_over` 终止项计数，其当前倾斜角阈值为 70°；正常到时不算跌倒，其他终止原因另列 |
| 平面速度误差 | 每步计算 `sqrt((vx-cx)^2+(vy-cy)^2)`，先在各首个 episode 内平均，再对环境等权平均；阈值 0.08 m/s |
| 转向误差 | 身体坐标 yaw 角速度与指令之差的绝对值，先按时间、再按环境平均；阈值 0.20 rad/s |
| 停止漂移 | 首个 episode 结束时相对初始位置的平面距离，取所有环境中位数；停止条件阈值 0.15 m |
| 动作变化 | 原始策略动作相邻步差的 RMS，包含从零历史到首次动作的变化；按环境计算后平均 |
| 脚底滑移 | 接触期间脚底 site 平面速度，按有效接触样本平均，再对有接触的环境平均；无接触环境数另记 |

失败环境保留失败前指标和时长，不丢弃短 episode。需要重置时，后续新 episode 不再进入统计；在最后一步跌倒仍判失败。出界即使被环境标为 timeout，也不能当作完整存活。独立的 [计数检查](../../records/evaluations/metric-accounting-check.json) 覆盖提前跌倒、重置后排除、正常到时、最后一步跌倒和出界。

在首轮完整评估前，修正了初版预检的平面误差定义：若对 x/y 两个分量的绝对误差取平均，机器人在 0.15 m/s 前进指令下完全静止也会得到 0.075 m/s，低于原阈值。这不能证明在行走。因此正式评估固定为平面误差向量长度的均值，数值阈值不变；旧预检指标保留，仅用于流程诊断，不能与新定义直接比较。

每段视频固定显示环境 0 的首个 episode，带指令、seed、仿真时间和终止结果；它不是 32 个环境的平均表现。初始帧加每两步一帧，共用 25 fps 编码；完整 20 秒轨迹有 501 帧，播放器时长约 20.04 秒。若环境 0 失败，录像在该处结束，不接上重置后的站立画面。

### 15.4 预检故障

| 现象 | 原因与处理 | 记录 |
| --- | --- | --- |
| 初始化提示 `ranges.heading is set but heading_command=False` | 关闭 heading 控制后需同时把 heading 范围设为 `None`；修正独立评估配置 | `official_walk_preflight_20261101T004603_s1001` |
| 停止指令完成，切换下一条件时出现 inference tensor 原地修改异常 | 循环创建的延迟缓存跨出了 `inference_mode` 上下文；改用 `no_grad`，保留跨条件重置流程 | `official_walk_preflight_20261101T004705_s1001` |
| 五条件预检完成 | 2 个环境、2 秒的流程通过，5 段视频各 51 帧；这不证明长期步态 | [预检复核](../../records/evaluations/official_walk_preflight_20261101T004815_s1001/verification.json) |

正式评估在同一个 VS Code 任务终端完成。以下为执行中与结束时的真实截图。

![VS Code 评估](../../records/evaluations/official_walk_reference_20261101T005022_s1001/screenshots/01_vscode_evaluation.jpg)

![评估完成](../../records/evaluations/official_walk_reference_20261101T005022_s1001/screenshots/02_vscode_completed.jpg)

### 15.5 输入输出复核

对完整评估已保存的前进轨迹，抽取环境 0 的第 0、100、500 步输入重新运行同一 ONNX，输出与对应保存动作的最大绝对差均为 0；三次输入的速度指令均为 `[0.15, 0, 0]`。在保持同一状态其他观测不变时，把停止指令改为前进 0.15 m/s，动作最大变化约为 0.045–0.058。模型对指令有响应，但单次动作差不能证明会产生步态，也不能直接解释闭环跟踪不足。

结果见 [指令响应检查](../../records/evaluations/official_walk_reference_20261101T005022_s1001/command-sensitivity.json)，可用 `scripts/validation/inspect_reference_response.py` 加本次结果目录重算。这项检查只排除所抽样输入中的指令漏传与输出记录不一致；模型训练时的任务/动力学、原始指令范围、实机 home 和滤波仍需要进一步核对。

### 15.6 正式结果

本次记录为 `official_walk_reference_20261101T005022_s1001`，五种指令共 160 个首个 episode 全部完成 20 秒，观测与动作均为有限值。只有停止条件达到全部阈值；前后行走未达到平面速度误差要求，左右转向未达到 yaw 误差要求。不能把“没有跌倒”解释成已经稳定行走。

| 条件 | 完整存活 | 跌倒数 | 平面误差（m/s） | yaw 误差（rad/s） | 判定 |
| --- | --- | --- | --- | --- | --- |
| 停止 | 32/32 | 0 | 0.0012 | 0.0028 | 通过 |
| 前进 0.15 m/s | 32/32 | 0 | 0.1488 | 0.0089 | 未通过 |
| 后退 0.10 m/s | 32/32 | 0 | 0.1000 | 0.0026 | 未通过 |
| 左转 0.4 rad/s | 32/32 | 0 | 0.0014 | 0.3982 | 未通过 |
| 右转 0.4 rad/s | 32/32 | 0 | 0.0118 | 0.3748 | 未通过 |

停止漂移中位数为 0.001416 m，约 1.42 mm，低于 0.15 m 阈值。补充指标如下；动作变化和脚底滑移目前用于描述与比较，没有另设通过阈值。脚底滑移低也可能是因为没有实际迈步。

| 条件 | 动作变化 RMS | 接触脚底速度（m/s） | 位移中位数（m） |
| --- | --- | --- | --- |
| 停止 | 0.01423 | 0.000218 | 0.001416 |
| 前进 | 0.01759 | 0.000687 | 0.003582 |
| 后退 | 0.01421 | 0.000208 | 0.001326 |
| 左转 | 0.01422 | 0.000259 | 0.001753 |
| 右转 | 0.02102 | 0.000935 | 0.004430 |

下图使用保存的原始速度，不作平滑。上图为前后指令下的身体前向速度，下图为左右转向的 yaw 角速度；细线是各环境轨迹，粗线是 32 个环境均值，虚线是指令。大部分轨迹接近静止，少数出现前向运动或右转，整体跟踪仍不足。每条轨迹均保留，避免只挑选能动的一例作为效果。

![速度响应](../../records/evaluations/official_walk_reference_20261101T005022_s1001/figures/command-response.png)

完整数据见 [评估结果](../../records/evaluations/official_walk_reference_20261101T005022_s1001/evaluation.json)。[独立复核](../../records/evaluations/official_walk_reference_20261101T005022_s1001/verification.json) 从 NPZ 重算逐条件指标、有限值、漂移和判定，并完整解码五段视频，每段均为 501 帧、960 × 720、25 fps。图像经过文字边界检查和逐图查看，记录见 [绘图检查](../../records/evaluations/official_walk_reference_20261101T005022_s1001/figures/plot-check.json)。

在第 15.1 节复核命令中将脚本改为 `plot_fixed_evaluation.py`，保留同一个结果目录参数，即可重新生成速度曲线。该命令只读取保存的轨迹，不再次仿真。随后运行 VS Code 的 **02 更新图文报告**，将图像和视频嵌入 HTML。

### 15.7 仿真视频

五段视频均为各条件环境 0 的完整首个 episode，未拼接成功片段。它们用于查看姿态和动作；统计结论来自上面的全部 32 个环境。HTML 嵌入视频与播放控件，Markdown 阅读器不支持时打开对应 MP4。视频文件已完整解码；自动浏览器受本地文件协议限制，本轮未完成 HTML 内的界面播放核验。

**停止**

<video controls preload="metadata" src="../../records/evaluations/official_walk_reference_20261101T005022_s1001/videos/stop_env0.mp4"></video>

[停止视频](../../records/evaluations/official_walk_reference_20261101T005022_s1001/videos/stop_env0.mp4)

**前进**

<video controls preload="metadata" src="../../records/evaluations/official_walk_reference_20261101T005022_s1001/videos/forward_env0.mp4"></video>

[前进视频](../../records/evaluations/official_walk_reference_20261101T005022_s1001/videos/forward_env0.mp4)

**后退**

<video controls preload="metadata" src="../../records/evaluations/official_walk_reference_20261101T005022_s1001/videos/backward_env0.mp4"></video>

[后退视频](../../records/evaluations/official_walk_reference_20261101T005022_s1001/videos/backward_env0.mp4)

**左转**

<video controls preload="metadata" src="../../records/evaluations/official_walk_reference_20261101T005022_s1001/videos/turn_left_env0.mp4"></video>

[左转视频](../../records/evaluations/official_walk_reference_20261101T005022_s1001/videos/turn_left_env0.mp4)

**右转**

<video controls preload="metadata" src="../../records/evaluations/official_walk_reference_20261101T005022_s1001/videos/turn_right_env0.mp4"></video>

[右转视频](../../records/evaluations/official_walk_reference_20261101T005022_s1001/videos/turn_right_env0.mp4)

### 15.8 模型与运行程序核对

固定评估后，使用已保存观测进行 CPU 离线复核，没有重新仿真、训练或连接真机。VS Code 的 **10 官方模型接口复核（不训练）** 调用 [检查脚本](../../scripts/validation/audit_reference_interface.py)，结果见 [接口记录](../../records/evaluations/official_walk_reference_20261101T005022_s1001/interface-audit.json)。检查使用归档 `Policy.build_obs` 和 `Policy.infer` 的实际实现，未重写一套替代运行逻辑。

从五种指令各抽取第 0、100、500、999 步的 32 个环境，共 640 个输入。按模型 home 把保存的相对关节角转回角度反馈，速度转成 degree/s，再放入 MCU 报文顺序；角速度、重力、指令和上一原始动作采用保存值。由归档运行程序重新构造的 61 维观测，最大绝对误差为 `4.77e-7`；同一 ONNX 原始动作与保存结果最大差为 `1.34e-7`。这验证了这些样本中的单位、顺序和 home 换算；构造的反馈并非真实传感器，也没有复现 IMU 融合与通信。

ONNX 图中已经包含逐项均值相减和尺度相除，61 项除数均有限且为正，因此本轮不再额外归一化。轨迹中上一动作也与前一步原始策略输出一致，初始历史为零。图中的归一化统计不能反推出完整训练指令范围；模型元数据只给出 `run_path: local`，训练提交、seed 和预算仍未知。

| 项目 | 本次核对 | 还需要验证的内容 |
| --- | --- | --- |
| home | 模型髋/踝约 ±24°，归档普通行走默认 ±22°；还有小幅膝关节差异 | 两套姿态对闭环步态与机械余量的影响 |
| 动作滤波 | 实际切片 `[5:10]` 使用 0.2，包含右髋偏航；其余为 0.45 | 完整运行滤波与仿真执行器组合的闭环效果 |
| 仿真时序 | 本次 `env.yaml` 为物理步长 0.005 秒、decimation 4，即策略 50 Hz | 板端实测频率、状态年龄和抖动 |
| 执行器 | 仿真 BAM 的 `kp_fw=5.0`，命令延迟 3–6 个物理步，电压随机范围 7.4–8.0 | 真实舵机响应、供电及延迟 |
| 固件增益 | 归档启动逻辑设置运行 P=6、D=20，嘴部另有处理 | 实际固件采用内容与舵机读回；源码常量不等于现场状态 |

保持保存的物理状态与其他输入相同，仅改为 runtime 默认 home 后，原始动作差的 RMS 为 0.01060、最大值 0.03392。这是条件输入变化下的模型输出差，没有包括 home 本身加到物理目标上的偏移，也不是步态改善量。对前进条件环境 0 的 1000 步做动作滤波重放，结果与独立 EMA 公式最大差 `9.69e-8`，滤波输出相对原始动作的 RMS 差为 0.00495；它同样不代表闭环运动结果。

仿真实际参数以本次保存的 `env.yaml` 为准，不能用未被该行走任务调用的通用配置覆盖它。ONNX 中的 `joint_stiffness` / `joint_damping` 元数据也不是可以直接写入舵机寄存器的 P/D。以上差异给出后续诊断对象，目前没有证据把某一项认定为“不迈步”的根因。

![接口复核](../../records/evaluations/official_walk_reference_20261101T005022_s1001/screenshots/03_vscode_interface_audit.jpg)

后续仿真诊断需先写出单项假设，例如只切换 home、只加入归档动作滤波，或只改变已经有依据的执行器参数；每项另存配置、轨迹、完整视频与原协议指标。不得同时改多个因素、改写本轮结果或放宽阈值。官方配置训练仍暂停；原生 Ubuntu 的效率对比按第 13 节在实际切换与环境核验后进行。

## 16. 单因素适配诊断

### 16.1 诊断设计

为区分第 15.8 节发现的差异，先冻结 [诊断协议](../../configs/experiments/reference_adaptation_diagnostics.json)，再启动新的固定权重仿真。四组均使用同一归档行走 ONNX、五条原固定指令、32 个环境、20 秒、开发 seed 1001；每组重新创建仿真环境，保留原噪声、延迟与动力学随机化。正式验收阈值不变，不调用训练或优化器。

| 组别 | 唯一改动 | 验证的问题 |
| --- | --- | --- |
| 参考重跑 `reference_repeat` | 不改模型 home、动作和增益 | 当前入口是否重现原有行为，作为同批对照 |
| Runtime home `runtime_home` | 观测和目标同时使用归档 runtime home | 默认姿态换算是否影响迈步；物理初始姿态仍与参考相同 |
| Runtime filter `runtime_filter` | 只加入归档动作 EMA | 动作平滑对闭环运动的影响；上一动作观测仍为原始网络输出 |
| 执行器增益 `actuator_kp6` | 仿真 `kp_fw` 从 5 改为 6 | 对已知 P 差异的敏感性；不能等同于真实固件 P/D 行为 |

诊断使用原来的指标脚本，新增原始动作、实际施加的策略动作、请求关节目标与步进后的关节位置/速度。请求目标与实际位置的误差只用于观察执行器响应，它不是传感器标定结果，也不能与原始策略动作变化混为一项。

### 16.2 自己运行

在 VS Code 选择 **12 适配诊断预检（不训练）**，成功后再选择 **13 适配单因素对照（不训练）**。它们复用同一个任务终端，并持有共享计算锁；四组顺序执行，不同时启动多个模拟器。与正式训练的暂停状态互相独立，运行它们不会恢复训练。

预检已完成：每条指令 2 个环境、2 秒，五条条件共 10 个首个 episode；五段视频各 51 帧，指标复算与完整解码通过。对每条条件的环境 0 全程重放归档 `Policy.infer`，共 500 步，施加动作最大绝对差为 `8.94e-8`。见 [预检批次](../../records/evaluations/adaptation_preflight_20261101T012322_s1001/batch.json) 与 [适配复核](../../records/evaluations/adaptation_preflight_20261101T012322_s1001/comparison.json)。这只验收入口与数学对应关系，不验收长期步态。

完整批次 `adaptation_comparison_20261101T012535_s1001` 已在可见 VS Code 终端完成四组仿真，后续复核也已通过。以下截图为第一组执行期间的真实状态；完整结果见下节和 [批次记录](../../records/evaluations/adaptation_comparison_20261101T012535_s1001/batch.json)。

![适配诊断](../../records/evaluations/adaptation_comparison_20261101T012535_s1001/screenshots/01_vscode_diagnostics.jpg)

每组分别保存 `evaluation.json`、`env.yaml`、协议、脚本副本、轨迹、指标和视频，批次目录保存运行命令与退出码。先独立复算每组指标并完整解码，再检查配置差异是否仅限协议允许的内容，重放归档 runtime 并生成 `comparison.json` / `comparison.csv`。两套 home 的变更同时影响相对观测与目标重建，必须按完整链路核对。

若仿真阶段中断，保留该批记录；再次运行任务 13 会创建新批次。若四组已经完成，仅最后的 CPU 复核中断，应检查中断记录后使用任务 15 复核已有数据，避免重复仿真。本轮确实处理过这种情况，过程见第 16.5 节。诊断中的数值达标仍须与正式基线、留出 seed 和实际部署验收区分。

### 16.3 对照结果

四组各完成 160 个首个 episode，共 640 个；全部完整存活，未记录跌倒。各组只有停止条件达到全部数值标准。下表列出运动条件的等权平均误差，前后行走比较平面速度误差，左右转向比较 yaw 误差；完整 20 条结果、动作变化和脚底滑移见 [对照 CSV](../../records/evaluations/adaptation_comparison_20261101T012535_s1001/comparison.csv)。

| 条件 | 前进平面误差（m/s） | 后退平面误差（m/s） | 左转 yaw 误差（rad/s） | 右转 yaw 误差（rad/s） |
| --- | --- | --- | --- | --- |
| 参考重跑 | 0.14882 | 0.10000 | 0.39823 | 0.37448 |
| 仅 runtime home | 0.14732 | 0.10018 | 0.39846 | 0.37631 |
| 仅 runtime 滤波 | 0.14989 | 0.10003 | 0.39845 | 0.37886 |
| 仅 `kp_fw=6` | 0.14808 | 0.09998 | 0.39798 | 0.36135 |
| 预设上限 | 0.08000 | 0.08000 | 0.20000 | 0.20000 |

停止条件的漂移中位数在四组中为 0.00129–0.00389 m，均低于 0.15 m。home 组前进误差略低于参考组；增益组右转 yaw 误差有所降低，但同时平面误差从 0.01198 增至 0.01736 m/s，且 yaw 仍未达标。这些是本轮开发 seed 的实际均值，不能解释为已获得可迁移的训练改进。

![适配对照](../../records/evaluations/adaptation_comparison_20261101T012535_s1001/figures/adaptation-comparison.png)

小点表示每个条件的全部 32 个首个 episode，大符号表示环境等权均值，虚线为预先固定的阈值。图中没有筛除慢速、漂移较大或跟踪较差的轨迹；这些点来自同一模型与同一开发 seed，不是 32 次独立训练。

前进条件还保存了请求关节目标与步进后的实际关节状态，便于区分“网络给了动作”和“身体完成了移动”。

| 条件 | 目标偏差 RMS（rad） | 关节速度 RMS（rad/s） | 最终位移中位数（m） |
| --- | --- | --- | --- |
| 参考重跑 | 0.04839 | 0.30382 | 0.00356 |
| 仅 runtime home | 0.05557 | 0.38455 | 0.00650 |
| 仅 runtime 滤波 | 0.04926 | 0.32303 | 0.00386 |
| 仅 `kp_fw=6` | 0.04975 | 0.40172 | 0.00270 |

目标偏差在所有有效首轮样本和全部关节上计算，包含动态延迟与接触约束，并非机械零位误差。增益组关节速度 RMS 更高，却没有达到前进跟踪要求；不能只看关节“动得更多”就判断步态更好。当前证据支持的结论是：这三项单独变化均不足以解决本轮跟踪问题。原模型训练任务和物理配置的完整来源仍未知，组合因素、留出 seed 与真机效果也尚未验证。

### 16.4 前进录像

以下均为预先固定的环境 0、前进 0.15 m/s、开发 seed 1001，完整时长 20.04 秒（501 帧、25 fps）。它们用于直观看四种设置，不代替上表全部环境的结果。其他四条指令的视频保留在每组记录的 `videos/` 目录。

**参考重跑**

<video controls preload="metadata" src="../../records/evaluations/official_walk_diagnostic_reference_repeat_20261101T012547_s1001/videos/forward_env0.mp4"></video>

**Runtime home**

<video controls preload="metadata" src="../../records/evaluations/official_walk_diagnostic_runtime_home_20261101T013412_s1001/videos/forward_env0.mp4"></video>

**Runtime 滤波**

<video controls preload="metadata" src="../../records/evaluations/official_walk_diagnostic_runtime_filter_20261101T014232_s1001/videos/forward_env0.mp4"></video>

**执行器增益**

<video controls preload="metadata" src="../../records/evaluations/official_walk_diagnostic_actuator_kp6_20261101T015047_s1001/videos/forward_env0.mp4"></video>

### 16.5 复核与故障记录

逐组复核重新计算全部保存轨迹的指标，并完整解码 20 段视频。批次复核确认 home/滤波组的环境配置与参考组逐字节一致；增益组仅有 `kp_fw: 5.0 → 6.0` 一处环境配置变化，home/滤波的变化分别位于策略适配层。四组每个条件的初始平面位置和关节位置逐值一致；不据此宣称整个随机轨迹逐步相同。

每组将环境 0 的全部 5000 步保存输入交给归档 `Policy.infer` 重放，共 20000 步；最大重构观测误差为 `4.77e-7`，最大施加动作误差为 `1.95e-7`。同时逐步检查全部环境的原始上一动作、EMA 和请求目标公式。见 [复核结果](../../records/evaluations/adaptation_comparison_20261101T012535_s1001/comparison.json)。这些检查验证软件对应关系，不能恢复缺失的原训练配置或证明硬件传感器正确。

最后 CPU 复核最初在每个时间步反复读取压缩 NPZ 成员，造成重复解压。四组仿真完成后，从同一 VS Code 终端中断该复核，保存 [中断前批次](../../records/evaluations/adaptation_comparison_20261101T012535_s1001/batch.before-reverification.json)，将读取改为每个文件一次加载。第一次重新复核又遇到 WSL 重启后的 PID 复用，旧判断把存在的数字 PID 误当作原监督进程；现已改为核对命令身份。两个过程的输出和脚本均保留在批次目录。

随后通过 **15 复核已有适配诊断（不仿真）** 在共享终端完成全部检查，退出码为 0；仿真没有重跑，原始轨迹与视频哈希保持一致。[首次重新复核日志](../../records/evaluations/adaptation_comparison_20261101T012535_s1001/reverification-console-20261101T020321.log) 保留了 PID 判断失败，[完成日志](../../records/evaluations/adaptation_comparison_20261101T012535_s1001/reverification-console-20261101T020427.log) 保存修正后的结果。此次修复改善的是离线文件读取方式，未测量或宣称训练提速。

![复核完成](../../records/evaluations/adaptation_comparison_20261101T012535_s1001/screenshots/02_vscode_verified.jpg)

## 17. 指令幅度诊断

### 17.1 为什么增加这项检查

归档运行界面的普通行走键盘默认前进为 0.2 m/s、转向为 0.5 rad/s，前进上限为 0.4 m/s，见 [界面源码](../../official_resources/xgoduck_runtime_arduino/assets/app.js)。原低速评估使用 0.15 m/s 和 0.4 rad/s，因此单独检查指令增大后模型是否开始有效跟踪。网页中 `deadzone` 的 0.15 是无量纲摇杆输入阈值；本次仿真直接注入速度指令，不经过网页摇杆处理，不能把该数值当作 0.15 m/s 的仿真死区。

[诊断协议](../../configs/experiments/reference_command_amplitude.json)在运行前冻结：前进 0.15、0.2、0.4 m/s，以及左转 0.4、0.5 rad/s；每档 32 个首个 episode、20 秒、seed 1001。采用同一归档行走 ONNX、模型 home、原始动作和动力学，不调用训练或优化器。原正式协议及数值阈值不变，较大指令即使表现改善，也不替代原低速条件验收。

### 17.2 自己运行

在 XGO-Duck 的 VS Code 工作区选择 **18 指令幅度诊断（不训练）**。终端显示 `--command-amplitude`，顺序运行五档指令，然后自动复算轨迹指标、完整解码视频并绘图。任务使用已有共享终端和计算锁；正式训练仍受暂停状态限制。每次执行创建新目录，已有完整结果应直接查看或复算，避免重复仿真。

执行入口为 [可见运行脚本](../../scripts/validation/run_command_amplitude.sh)，[分析脚本](../../scripts/validation/analyze_command_amplitude.py)解释修改动机、均值定义和结论边界。分析先对每个首轮 episode 时间平均，再对环境等权平均；图中的粗线是该时刻仍有效的首轮 episode 均值，需同时查看存活情况。

![指令诊断](../../records/evaluations/official_walk_amplitude_diagnostic_20261101T023405_s1001/screenshots/01_vscode_running.jpg)

### 17.3 实测结果

五档指令共 160 个首个 episode 全部完成 20 秒，未记录跌倒，但没有一档达到全部数值标准。下表来自 [结果汇总](../../records/evaluations/official_walk_amplitude_diagnostic_20261101T023405_s1001/amplitude-summary.json)；[CSV](../../records/evaluations/official_walk_amplitude_diagnostic_20261101T023405_s1001/amplitude-summary.csv)还保留各档首轮均值的最小值和最大值。

| 指令 | 实际平均响应 | 平面速度误差（m/s） | yaw 误差（rad/s） | 数值标准 |
| --- | --- | --- | --- | --- |
| 前进 0.15 m/s | 0.00509 m/s | 0.14881 | 0.00901 | 未通过 |
| 前进 0.20 m/s | 0.12476 m/s | 0.14141 | 0.07760 | 未通过 |
| 前进 0.40 m/s | 0.22795 m/s | 0.21596 | 0.15464 | 未通过 |
| 左转 0.40 rad/s | 0.00174 rad/s | 0.00140 | 0.39826 | 未通过 |
| 左转 0.50 rad/s | 0.06101 rad/s | 0.01797 | 0.44463 | 未通过 |

前进指令增大后，响应明显增强；因此不能把本机模型概括为“任何指令下都不会移动”。0.2 m/s 时各首轮 episode 的平均前进速度仍有明显差异，范围约为 0.00007–0.15161 m/s；0.4 m/s 时为 0.20751–0.25263 m/s，仍低于请求速度。平面误差是在每个时间步计算二维速度向量误差后求均值，不能用“指令减实际平均前进速度”代替，也不能由最终移动距离单独判定通过。

![指令响应](../../records/evaluations/official_walk_amplitude_diagnostic_20261101T023405_s1001/figures/command-amplitude.png)

曲线保留全部 32 个首轮 episode，没有平滑或挑选轨迹。较大转向指令的少数轨迹开始转动，但整体仍跟踪不足。此结果支持“指令幅度影响该模型在当前仿真中的响应”，尚不能确定低速不足来自原训练分布、奖励还是其他模型与仿真差异；不据此修改真机指令上限或取消低速验收。

### 17.4 视频与复核

下面三段固定展示环境 0，不能代替上表对全部环境的统计。五段完整视频均保存在本次记录的 `videos/`，每段 501 帧、25 fps、20.04 秒；最后一帧对应 20 秒状态。

**前进 0.2 m/s**

<video controls preload="metadata" src="../../records/evaluations/official_walk_amplitude_diagnostic_20261101T023405_s1001/videos/forward_keyboard_env0.mp4"></video>

**前进 0.4 m/s**

<video controls preload="metadata" src="../../records/evaluations/official_walk_amplitude_diagnostic_20261101T023405_s1001/videos/forward_max_env0.mp4"></video>

**左转 0.5 rad/s**

<video controls preload="metadata" src="../../records/evaluations/official_walk_amplitude_diagnostic_20261101T023405_s1001/videos/turn_keyboard_env0.mp4"></video>

[独立复核](../../records/evaluations/official_walk_amplitude_diagnostic_20261101T023405_s1001/verification.json)已重算五档轨迹指标并完整解码全部视频；环境 YAML 与上一轮原始动作参考组逐字节相同，五档初始关节位置和平面位置逐值相同。源码、模型、协议与原正式协议哈希均核对；执行这项诊断时，正式训练仍暂停。图像检查修正了裁剪坐标，随后只重算保存数据和重绘，未重新仿真或修改轨迹，见 [绘图记录](../../records/evaluations/official_walk_amplitude_diagnostic_20261101T023405_s1001/analysis-rerender.log)与 [视觉检查](../../records/evaluations/official_walk_amplitude_diagnostic_20261101T023405_s1001/figures/visual-review.json)。

![诊断完成](../../records/evaluations/official_walk_amplitude_diagnostic_20261101T023405_s1001/screenshots/02_vscode_completed.jpg)

[终端日志](../../records/evaluations/amplitude-console-20261101T023352.log)中的 `AMPLITUDE_DIAGNOSTIC_EXIT_CODE=0` 表示仿真、数据复核和初次绘图流程完成。该状态不表示步态验收通过；原低速协议、正式训练及真机阶段仍按原门槛推进。

## 18. 官方基线续训

### 18.1 恢复结果

用户在 2026-11-01 明确恢复 WSL 官方基线后，先更新目标和项目状态，再通过现有 VS Code 任务终端执行 `resume_baseline 42`。新实验为 `baseline_continuation_20261101T101809_s42`，于北京时间 10:18:09 创建记录。仍使用 1024 个环境、seed 42、官方奖励和物理参数；没有重新安装依赖或改变冻结协议。本阶段训练于 14:04:33 正常结束，退出码为 0。

恢复点保存于第 251 次更新之后，因此从日志编号 251 开始，再做 4749 次更新。修改只处理恢复编号、剩余预算、学习率变量和 reset 顺序，不是新的奖励实验。实际配置与保存基线一致，见 [恢复计划](../../records/experiments/baseline_continuation_20261101T101809_s42/resume-plan.json)。

| 检查项 | 已核验结果 |
| --- | --- |
| checkpoint | SHA-256 与归档恢复点一致，原文件未改写 |
| 环境 reset | 加载后再次 reset，课程计数保持 6024 |
| 初始学习率 | PPO 自适应变量同步至 Adam 的 0.000170859375 |
| 新迭代编号 | 251–4999，连续 4749 次，已核对完整日志 |
| 日志与 TensorBoard | 采样耗时、学习耗时、吞吐、奖励、episode 长度逐项相符 |
| 累计预算 | 5000 次更新、122880000 条环境转移；本次续训新增 116711424 条 |
| 新恢复点 | `model_4999.pt` 已核验累计 5000 次，包含续训新增的 4749 次更新 |
| 进程 | 训练与监督进程已退出，成功退出记录已保存 |
| 步态 | 尚未进行自训策略的固定指令验收 |

实际 reset 参数见 [加载记录](../../records/experiments/baseline_continuation_20261101T101809_s42/resume-ready.json)。本段编号 251–4999 的完整日志见 [本阶段核验](../../records/experiments/baseline_continuation_20261101T101809_s42/resume-verification/20261101T060522460407Z/check.json)，历次快照保存在 [进度归档](../../records/experiments/baseline_continuation_20261101T101809_s42/resume-verification)。每次核验均保存各自的日志前缀、TensorBoard 文件与哈希，后续训练增长不会改变这些快照。

checkpoint 的 CPU 核验按实际保存计数进行：采样计数应为累计更新数乘 24，归一化计数再乘 1024，Adam 更新数为累计更新数乘 5 epoch、4 minibatch。19 个新恢复点各有 76 个保存张量，均为有限值；actor 与 critic 各 8 个网络张量已相对原恢复点发生变化。

| checkpoint | 累计更新 | 采样计数 | 归一化计数 | Adam 计数 | 证据 |
| --- | --- | --- | --- | --- | --- |
| model_500.pt | 501 | 12024 | 12312576 | 10020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_500.json) |
| model_750.pt | 751 | 18024 | 18456576 | 15020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_750.json) |
| model_1000.pt | 1001 | 24024 | 24600576 | 20020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_1000.json) |
| model_1250.pt | 1251 | 30024 | 30744576 | 25020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_1250.json) |
| model_1500.pt | 1501 | 36024 | 36888576 | 30020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_1500.json) |
| model_1750.pt | 1751 | 42024 | 43032576 | 35020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_1750.json) |
| model_2000.pt | 2001 | 48024 | 49176576 | 40020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_2000.json) |
| model_2250.pt | 2251 | 54024 | 55320576 | 45020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_2250.json) |
| model_2500.pt | 2501 | 60024 | 61464576 | 50020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_2500.json) |
| model_2750.pt | 2751 | 66024 | 67608576 | 55020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_2750.json) |
| model_3000.pt | 3001 | 72024 | 73752576 | 60020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_3000.json) |
| model_3250.pt | 3251 | 78024 | 79896576 | 65020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_3250.json) |
| model_3500.pt | 3501 | 84024 | 86040576 | 70020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_3500.json) |
| model_3750.pt | 3751 | 90024 | 92184576 | 75020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_3750.json) |
| model_4000.pt | 4001 | 96024 | 98328576 | 80020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_4000.json) |
| model_4250.pt | 4251 | 102024 | 104472576 | 85020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_4250.json) |
| model_4500.pt | 4501 | 108024 | 110616576 | 90020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_4500.json) |
| model_4750.pt | 4751 | 114024 | 116760576 | 95020 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_4750.json) |
| model_4999.pt | 5000 | 120000 | 122880000 | 100000 | [检查](../../records/experiments/baseline_continuation_20261101T101809_s42/checkpoint-checks/model_4999.json) |

恢复副本保存在 `artifacts/recovery/baseline_continuation_20261101T101809_s42/`。`model_4999.pt` 的 SHA-256 为 `350a1a7300213e94ba6f9e98a3662ffa40bf2b8995153e51f2f60460ce80ceb7`，与 Linux 原件一致。最终日志与 checkpoint 均达到累计 5000 次更新，完成预算核验。

### 18.2 曲线与录像

训练任务正常结束后的实际 VS Code 界面如下，终端显示退出码 0。截图保留当前窗口布局；它记录任务结束，稳定步态须通过后续固定条件评估。

![训练结束](../../records/experiments/baseline_continuation_20261101T101809_s42/vscode-training-finished.jpg)

以下曲线来自已核验的 4749 次新更新，无平滑。横轴是累计完成更新数，所以第一点为 252，最后一点为 5000；终端中的迭代标签从零开始，比累计完成数少一。终端 `Total steps` 则只统计这次启动新增的环境转移，不能直接当作包含父实验的总样本数。数据表与绘图检查保存在 [曲线目录](../../records/experiments/baseline_continuation_20261101T101809_s42/resume-verification/20261101T060522460407Z/figures)。训练奖励与 episode 长度总体上升，期间也有回落，这不是固定指令下的泛化或稳定性结果；吞吐保留全部波动，含热身和录像区间，不据此声称优化提速。

![续训曲线](../../records/experiments/baseline_continuation_20261101T101809_s42/resume-verification/20261101T060522460407Z/figures/continuation-progress.png)

官方训练条件会随采样计数变化。对编号 251–4999 的冻结快照，已逐项核对 7 个课程标量，共 33243 个值：终端与 TensorBoard 相符，数值落在对应迭代起止计数允许的配置阶段内，最后一轮也与配置一致。核验保留了环境参数、课程函数、reset 顺序源码和逐轮 CSV，见 [课程记录](../../records/experiments/baseline_continuation_20261101T101809_s42/resume-verification/20261101T060522460407Z/curriculum/check.json)。下表列出实际课程变化，并保留本次边界处的过渡值；首行是本次续训的起点。

| 日志编号 | 动作变化惩罚权重 | 站立指令采样比例 | 头部范围日志值 | 两处质心扰动范围 | 头部偏置权重 |
| --- | --- | --- | --- | --- | --- |
| 251 | −0.1 | 2% | 0.07 | ±3 mm | 0 |
| 500 | −0.2 | 5% | 0.21 | ±5 mm | 0 |
| 600 | −0.2 | 5% | 0.21 | ±5 mm | 1 |
| 750 | −0.4 | 10% | 0.21 | ±5 mm | 1 |
| 1000 | −0.6 | 15% | 0.49 | ±10 mm | 2 |
| 1250 | −0.8 | 15% | 0.49 | ±10 mm | 2 |
| 1500 | −1.0 | 20% | 0.91 | ±15 / ±10 mm | 3 |
| 1999 | −1.0 | 20% | 约 0.93042（过渡） | ±15 / ±10 mm | 3 |
| 2000 | −1.0 | 约 24.79%（过渡） | 1.4 | ±15 / ±10 mm | 3 |
| 2001 | −1.0 | 25% | 1.4 | ±15 / ±10 mm | 3 |

动作变化项计算相邻两步原始策略输出之差的平方和；权重更负时，同样的动作变化产生更大的负奖励。站立比例是指令采样设置，不是实测存活率。头部范围日志值是各指令分量范围的最大绝对值，实际各分量和单位须看归档参数，不能把它当成测得的头部角度；身体范围日志值在本段保持 0.05。两处质心扰动分别对应 `randomize_com` 与 `randomize_head_com`；范围不同时，单元格按这个顺序分别列出。

这些都是本轮官方配置已有的课程，线速度跟踪奖励权重仍为 3.0。它们说明总奖励的尺度和采样任务会变化，因此不同阶段的奖励不能直接当作同条件性能比较。课程切换在环境 reset 时生效，函数对边界使用 `>` 或 `>=`，日志又会汇总采样步；不能只用阈值除以 24 推断精确生效时刻。编号 1999 的头部范围日志和编号 2000 的站立比例位于切换前后值之间，按逐轮计数边界核验通过；后续分别达到 1.4 和 25%。本段的奖励回落与课程边界接近，但尚未做隔离试验，不能认定回落一定由课程引起。当前仅解释已观测条件，不改课程、奖励或验收标准。

首段录像为实际训练环境 0 的随机策略采样，完整解码为 200 帧、50 fps、4 秒、960×720。副本与 Linux 原始 MP4 的 SHA-256 相同，见 [媒体记录](../../records/experiments/baseline_continuation_20261101T101809_s42/media-manifest.json)。文件名中的 `step-0` 是本次录像计数起点，不表示从零训练。视频内会发生随机动作、指令变化和重置，不能拿来替代固定策略的行走评估。

<video controls preload="metadata" src="../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-0.mp4"></video>

![训练画面](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-0.jpg)

已归档本次采样步数 0、6000、12000、18000、24000、30000、36000、42000、48000、54000、60000、66000、72000、78000、84000、90000、96000、102000、108000 的 19 段录像，每段均完整解码为 200 帧、4 秒，副本与原文件哈希一致。下方展示步数 108000 的录像，步数 [6000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-6000.mp4)、[12000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-12000.mp4)、[18000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-18000.mp4)、[24000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-24000.mp4)、[30000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-30000.mp4)、[36000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-36000.mp4)、[42000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-42000.mp4)、[48000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-48000.mp4)、[54000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-54000.mp4)、[60000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-60000.mp4)、[66000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-66000.mp4)、[72000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-72000.mp4)、[78000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-78000.mp4)、[84000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-84000.mp4)、[90000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-90000.mp4)、[96000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-96000.mp4)、[102000](../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-102000.mp4) 的中间录像仍保留。各段采用真实随机训练条件，不能把画面变化直接算成同条件评估收益。

<video controls preload="metadata" src="../../records/experiments/baseline_continuation_20261101T101809_s42/videos/rl-video-step-108000.mp4"></video>

媒体清单已按数字步数排序，修正文件名排序会把 12000 放在 6000 前的问题，见 [排序检查](../../records/experiments/baseline_continuation_20261101T101809_s42/media-order-check.json)。下一次启动所用的训练监督脚本也已修正；当前进程仍使用启动时保存的脚本副本，没有为刷新页面代码而重启训练。本次网页尚未获浏览器设置放行，若后续手动打开原页面，应按视频文件中的步数选择片段。

![终端片段](../../records/experiments/baseline_continuation_20261101T101809_s42/vscode-training.jpg)

### 18.3 自己查看和复查

训练已经启动时，直接查看现有 VS Code 终端，不重复运行任务 16，不修改当前实验的任务配置或训练参数。日志保存在本实验目录的 `training.log`；checkpoint 和原始 TensorBoard 事件保存在核验记录所列 Linux `log_dir`。实际终端显示迭代、采样与更新耗时、奖励、episode 长度和预计剩余时间。预计时间随热身、录像和系统负载变化，不能把开头的估计当作完成承诺。

本机训练页地址为 `http://localhost:8765/`。虽然用户已在聊天中允许访问，但本次浏览器仍被已保存的权限设置拒绝，尚未核验本次网页显示；这一限制保存在 [访问记录](../../records/experiments/baseline_continuation_20261101T101809_s42/visualization-access.json)。训练和文件记录不依赖浏览器成功打开。保持用户现有软件窗口大小和布局，不为截图调整窗口。

下面的工具只读取并保存进度快照，不调用训练、模拟器或优化器。它核对完整日志块与 TensorBoard 的共同前缀，避免文件增长过程中把半条日志记成完成。命令在项目根目录、现有 Linux 环境中执行；复查时须等待正在占用共享终端的任务结束，再在该终端执行。后续代理监测使用同一只读工具，不另开训练终端。

```bash
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/verify_resumed_progress.py baseline_continuation_20261101T101809_s42
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/plot_continuation_progress.py baseline_continuation_20261101T101809_s42
```

检查应返回 `status: passed`，并列出 `first_iteration`、`last_verified_iteration`、`cumulative_completed_updates` 与 `target_updates`。图表显示的是对应快照，不会自行刷新；每次新核验均另存输入证据。视频检查可用 `inspect_training_video.py <实验目录> --live`，只处理已复制完成的录像，不代表训练已经结束。

训练进行中的报告进度可通过 [刷新脚本](../../scripts/validation/refresh_continuation_report.py)统一更新。它从已核验的日志快照、checkpoint、课程数据和媒体清单填写本节、目标清单及项目状态，保留历史记录；曲线和预览图还须已有实际目视检查记录。缺少前置证据时会停止，不代替核验，也不自动判定稳定行走。完成阶段核验后，在项目根目录运行：

```bash
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/refresh_continuation_report.py
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/render_development_report.py
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/verify_report_artifacts.py
```

这组命令用于正在训练的阶段报告；最终训练退出、导出和固定指令评估结果仍按后文分别验收。

课程核验同样只读保存数据。下面命令复查上表所用快照，输出配置来源、实际阶段首次出现的日志编号和逐轮 CSV；不会启动训练或修改当前参数：

```bash
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/inspect_curriculum_progress.py records/experiments/baseline_continuation_20261101T101809_s42/resume-verification/20261101T060522460407Z
```

本次早期只读硬件采样记录了 GPU 利用率、显存、时钟、功耗和当前电源方案，见 [硬件快照](../../records/experiments/baseline_continuation_20261101T101809_s42/host-performance-sample.json)。它只是一个时间点，不能用来确定 WSL 是瓶颈或宣称原生 Ubuntu 会提速。本轮保留官方基线设置，性能优化与双系统比较按固定条件另行执行。

11:06 的 [效率观测](../../records/experiments/baseline_continuation_20261101T101809_s42/efficiency-20261101T110610.json)补充了连续 12 次、间隔约 1 秒的 GPU 读数，以及截至编号 1161 的最近 100 次普通更新。后者的采样耗时中位数为 2.669 秒，学习更新为 0.210 秒，总迭代时间为 2.895 秒，终端吞吐中位数为 8486 转移/秒；各项中位数分别计算，不能相加后当作总耗时。GPU 利用率中位数为 53%，显存占用约 2712 MiB。采样仍是主要耗时部分，但这些是在本次训练运行期间获得的观测，尚不能证明具体瓶颈或提速方案。后续 WSL / 原生 Ubuntu 比较仍须保持相同环境数、录像设置与更新预算；本次未改电源方案、时钟或训练参数。

12:08 的 [后续观测](../../records/experiments/baseline_continuation_20261101T101809_s42/efficiency-20261101T120848.json)保留了编号 2271–2370 的 100 次普通更新和 12 次 GPU 读数。采样、学习、总迭代耗时中位数分别为 2.7025、0.213、2.920 秒，终端吞吐中位数为 8413.5 转移/秒；GPU 利用率中位数 55.5%，显存 2705 MiB，功耗 37.12 W，温度 59°C。两次观测来自不同训练阶段，不能据其差值判断配置提速或减速。

12:10:21–12:10:33 又记录了准确匹配本次实验命令的 PID 415，见 [CPU 原始计数](../../records/experiments/baseline_continuation_20261101T101809_s42/cpu-observation-20261101T121033.json)。13 个快照组成 12 个时间区间，期间日志编号从 2402 到 2406。训练进程 CPU 占用中位数为 99.43%，这里 100% 表示一个逻辑核心；WSL 的 16 个逻辑核心合计忙碌比例中位数为 6.34%。进程常驻内存约 2657 MiB，线程数为 69。线程数量不等于同时参与计算的核心数量，整体占用较低也不能说明训练进程处于闲置。

CPU 计算使用进程 `utime + stime` 的增量除以时钟频率及实际间隔；客体忙碌比例按 `/proc/stat` 前八项总增量扣除 idle、iowait 后计算，字段含义见 [Linux 内核文档](https://docs.kernel.org/filesystems/proc.html)。这项读数属于 WSL 客体，不代表 Windows 全机 CPU；也没有用 iowait 推断磁盘瓶颈。CPU 与 GPU 来自先后两个窗口，不能当作同步采样。现有证据支持继续调查采样阶段和 CPU/GPU 等待关系，尚不能确定单一瓶颈或证明切换原生 Ubuntu 的收益。

已从实际日志重新核对 120 条保存记录，复算 100 次更新的中位数、12 次 GPU 统计及全部 CPU 区间，见 [独立复核](../../records/experiments/baseline_continuation_20261101T101809_s42/efficiency-20261101T120848-check.json)。观测脚本、原始计数、对应日志摘录和哈希均保留。下列命令只复查已有数据，不需要 PID 仍然存在；在项目根目录、现有 Linux 环境运行，共享任务终端空闲后再自行尝试：

```bash
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/verify_training_efficiency.py records/experiments/baseline_continuation_20261101T101809_s42/efficiency-20261101T120848.json records/experiments/baseline_continuation_20261101T101809_s42/cpu-observation-20261101T121033.json
```

13:06 的[最新观测](../../records/experiments/baseline_continuation_20261101T101809_s42/efficiency-20261101T130648.json)覆盖编号 3572–3671 的 100 次普通更新，GPU 另在 13:06:37–13:06:48 取得 12 次读数。下表各项均为中位数；更新计时覆盖最近 100 次更新，硬件读数只覆盖约 12 秒，不能逐轮对应。代理在两次观测间没有修改训练配置、电源方案或 GPU 时钟设置。

| 指标 | 12:08 观测 | 13:06 观测 |
| --- | ---: | ---: |
| 采样耗时（秒） | 2.7025 | 2.1945 |
| 学习耗时（秒） | 0.213 | 0.1915 |
| 总迭代耗时（秒） | 2.920 | 2.410 |
| 吞吐（转移/秒） | 8413.5 | 10214.5 |
| GPU 利用率 | 55.5% | 35% |
| GPU SM 时钟（MHz） | 997.5 | 1800 |
| GPU 功耗（W） | 37.12 | 62.545 |

后段观测的吞吐更高，但 GPU 利用率更低、时钟与功耗更高。运行状态存在变化，因此不能单凭利用率高低判断训练效率，也不能把这段差值算作代码优化收益。GPU 读数覆盖整张显卡，未隔离其他进程；当时各计算阶段的耗时仍待测量，后续采样分析见本节后续记录，时钟变化原因没有单独验证。

13:06:49–13:07:01 的[CPU 观测](../../records/experiments/baseline_continuation_20261101T101809_s42/cpu-observation-20261101T130701.json)仍匹配 PID 415，进程占用中位数为一个逻辑核心的 99.96%，WSL 合计忙碌比例为 6.37%。两组硬件读数仍为先后窗口。[最新复核](../../records/experiments/baseline_continuation_20261101T101809_s42/efficiency-20261101T130648-check.json)从日志和原始计数重新计算了 120 条日志记录、12 次 GPU 读数和 13 个 CPU 快照，结果一致。复查命令如下：

```bash
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/verify_training_efficiency.py records/experiments/baseline_continuation_20261101T101809_s42/efficiency-20261101T130648.json records/experiments/baseline_continuation_20261101T101809_s42/cpu-observation-20261101T130701.json
```

已完成一项只读的 [采样路径核查](../../records/experiments/baseline_continuation_20261101T101809_s42/sampling-code-audit/check.json)，归档了实际安装的相关源码、函数位置和本次环境参数。当前每次 PPO 更新包含 24 个环境步，每个环境步执行 4 个物理子步，即 96 次面向 1024 个并行环境的批量物理步进；此外还有 forward、传感、奖励、重置和观测计算。这解释了计时范围，尚未给出各段耗时占比。

| 路径 | 当前代码行为 | 后续测量重点 |
| --- | --- | --- |
| 环境步进 | 执行动作和物理子步，再计算终止、奖励、重置、传感与观测 | 分开记录物理计算和管理逻辑的 CPU/CUDA 时间 |
| 检查与统计 | runner 默认检查 NaN，环境和日志统计使用 `nonzero`；日志还把结束 episode 的统计复制到 CPU | 查看同步、数据复制和 Python 调度的耗时 |
| 录像 | 渲染路径包含可视化回调和图像输出 | 与已有录像区间对应，使用预定的有/无录像对照 |
| CUDA Graph | 已安装的模拟器具备物理步进和传感图执行分支 | 核实运行时是否启用及实际覆盖范围，不假设整个训练循环已捕获 |

CUDA 输入上的 `torch.nonzero` 会触发主机与设备同步，见 [PyTorch 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nonzero.html)。这使上述位置值得测量，但源码里有同步调用并不能证明它是主要瓶颈。另一个包含 `.item()` 的 `raw_accelerometer` 函数未被本次 actor 或 critic 观测配置选用，不能把它当作当前耗时来源。

本阶段训练已完成，后续可用同一可见终端单独采集短时 CPU/CUDA 跟踪，区分策略前向、环境、检查统计和录像，并记录分析工具自身的开销。主机吞吐比较仍使用第 13.4 节固定的普通运行，不能把带性能分析开销的结果混入比较。本次没有删除有限值检查、改变物理子步、奖励、环境数或正在执行的代码。

复查上述源码和配置记录的命令如下，仅读取文件并保存来源副本：

```bash
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/inspect_sampling_code.py baseline_continuation_20261101T101809_s42
```

性能分析入口已准备为 VS Code 任务 **22 采样性能分析（独立短运行）**，协议见 [分析配置](../../configs/experiments/runtime_profile.json)，代码说明见 [分析入口](../../scripts/training/profile_components.py)。任务先运行普通参照，再运行带分析器的一项；每项从本轮最终模型加载到新的环境，恢复采样计数、下一迭代编号和学习率，然后 reset。两项各做 27 次更新，奖励、物理、1024 环境和录像设置相同，新增的权重只保存在独立日志目录，不参与步态选型，也不增加正式模型的 5000 次预算。

| 区间 | 更新数 | 用途 |
| --- | ---: | --- |
| 5000–5019 | 20 | 启动后等待，避开初始录像 |
| 5020–5021 | 2 | 分析器预热 |
| 5022–5026 | 5 | 采集策略、环境、物理、检查、统计和 PPO 事件 |

事件采集使用已安装的 `torch.profiler`，安排等待、预热和采集区间，并输出时间线；方法对应 [PyTorch 文档](https://docs.pytorch.org/docs/2.9/profiler.html)。入口不增加逐步 CUDA 同步，CPU 与设备时间分别保存；嵌套区域的耗时不能直接相加。普通参照使用相同的 5 次更新区间记录单调时钟边界，用于报告一次有序运行对的观测开销。前后 GPU 时钟、功耗等读数随记录保存；这不是受控提速结论，也不替代第 13.4 节的主机比较。

任务仍复用共享终端和计算锁，导出、评估或其他计算运行时会拒绝并发。输出放在 `records/profiling/`，包含普通与分析两项的终端日志、逐次耗时、来源副本、初始状态和结果；分析项还生成 `trace.json`、`operators.txt`。最后从原始时间线重新数调用次数，检查真实 CUDA 事件、配置一致性和新增 checkpoint 计数。未取得 CUDA 事件时保留记录并报告未完成，不把缺失事件解释为 GPU 没有工作。

等效 Linux 命令如下，在项目根目录、现有终端中执行：

```bash
# 只读预检；不创建模拟器或初始化 CUDA
bash scripts/validation/profile_current_baseline.sh --check-only
# 导出及固定条件评估处理完后，再顺序运行两项性能诊断
bash scripts/validation/profile_current_baseline.sh
```

启动该阶段前先完成了 [CPU 软件检查](../../records/reports/profile-entry-check/check.json)：实际保存配置核对通过；小型 CPU 样例验证了 5 个采集更新内的 120 次环境调用、480 次物理调用标记，并检查返回值、正常退出和异常后的方法恢复。样例没有创建机器人模拟器或执行优化器更新，其耗时不能代表机器人训练性能。共享终端配置与脚本语法也已检查。当时实际两项性能运行、CUDA 跟踪和具体瓶颈尚未测得，后续执行与恢复结果保留在下文。

### 18.4 导出与数值核验

[完成记录](../../records/experiments/baseline_continuation_20261101T101809_s42/training-completion-check.json)将正常退出、完整日志、最终 checkpoint、课程与媒体来源绑定在一起。[导出预检](../../records/experiments/baseline_continuation_20261101T101809_s42/post-training-export-readiness.json)已通过，只读取文件，没有导出模型或运行仿真。

最终 checkpoint 的累计采样计数、Adam 更新数、76 个有限张量与恢复副本均已核验，训练预算达到 5000 次。任务 19 已实际导出 ONNX 并通过数值一致性检查；任务 20 已按第 7 节冻结的五条指令、32 环境和 20 秒时长完成评估，结果见第 19 节。先判断存活、跌倒、速度误差和停止漂移，再决定是否进入奖励修改组。训练奖励和退出码不能代替步态验收。

VS Code 任务 **19 导出已完成的续训基线** 于 14:43:47 正常结束，终端记录 `EXPORT_CHECK_EXIT_CODE=0`。任务从 `project.json` 读取本次续训实验，先调用 [checkpoint 核验](../../scripts/validation/verify_continuation_checkpoint.py) 检查最终预算，再执行官方导出及 1000 帧数值检查。**20 评估已导出的续训基线** 随后于 14:44:22 在同一任务终端启动，使用开发 seed 1001，现已完成独立轨迹复算与视频完整解码。

为保持现有窗口尺寸和布局，通过当前 VS Code 的 **Terminal → Run Task…** 菜单选择任务，继续使用同一个终端。桌面工具的自动激活曾意外还原窗口，现已恢复并停止该输入路径，详见[窗口记录](../../records/reports/window-geometry-check.json)。本次由用户完成菜单启动，后续从文件核验进度。

[导出记录](../../records/experiments/baseline_continuation_20261101T101809_s42/export-validation.json)包含真实 ONNX 驱动仿真的 1000 帧结果。输入为 61 个 float32，输出为 14 个 float32；PyTorch 与 ONNX 在相同输入上的最大绝对动作差为 5.245208740234375e-6，平均差为 3.446604353030125e-7，超出既定容差的帧数为 0。此处动作是网络原始输出，不是舵机角度。使用的容差仍为绝对 1e-5、相对 1e-4。期间一次环境重置包含超时可能，不能据此填写固定评估的跌倒数。

[CPU 独立复核](../../records/experiments/baseline_continuation_20261101T101809_s42/export-verification/check.json)从最终 checkpoint 和保存的网络配置重新构建官方 actor，在原始 1000 帧输入上分别运行 PyTorch 与交付 ONNX，两者各自都与保存动作完全相同。复核还检查 ONNX 图、接口、有限值、原件与副本哈希，以及 actor 状态未改变；没有启动新模拟器、CUDA 或优化器。保存输入、动作、配置和核验源码集中在同一 `export-verification/` 目录，可按下列命令复查：

```bash
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/verify_export_trace.py baseline_continuation_20261101T101809_s42
```

交付模型为 [policy.onnx](../../artifacts/baseline_continuation_20261101T101809_s42/policy.onnx)，SHA-256 为 `611e8d9b67912f6eebff008c71d80649f4de8afbff871ee57361e8fcc6937e38`。本项验证使用 ONNX Runtime 1.24.4；板端所用环境的兼容性、运行时延和实机运动仍须另验。固定条件评估记录为 `records/evaluations/custom_onnx_reference_20261101T144436_s1001/`，其中 `custom_onnx` 表示本机自训导出模型，不能与第 15 节归档的官方预训练权重混淆。

等效的 Linux 命令如下，在项目根目录和同一可见终端中依次执行：

```bash
bash scripts/validation/export_current_continuation.sh
bash scripts/validation/evaluate_current_continuation.sh
```

两个入口均支持 `--check-only`，只读取输入记录，不初始化 CUDA 或模拟器。在训练尚未完成时，入口检查曾实际返回 `EXPORT_NOT_STARTED` 和 `EVALUATION_NOT_STARTED`，没有创建评估目录。原有短训练的导出输入检查仍通过；Bash 语法与六段内嵌 Python 语法也已检查，见 [入口准备记录](../../records/experiments/baseline_continuation_20261101T101809_s42/post-training-entry-check.json)。这证明入口检查生效，导出实际结果见上文，步态以固定条件评估为准。

任务 20 的 `fixed-evaluation-link.json` 分别记录数值标准是否全部通过、独立复算是否通过，以及保留 seed 是否已评估。流程成功不自动把 `stable_gait_verified` 改为真，也不开放修改组或真机动作。保留 seed 2001 尚未使用。

### 18.5 留出评估

留出评估使用开发期间保留的 seed 2001。先完成任务 19、20，确认开发评估的五条指令均达到数值标准且独立复核通过，再执行共享终端中的 **21 已选基线留出评估（一次）**。如果开发条件未通过，继续分析开发数据，保留留出 seed；不使用它选择训练长度、奖励或 checkpoint。

任务 21 会在续训记录中写入 `heldout-selection.json`，固定最终模型、checkpoint、协议、开发结果、环境配置、源码及依赖来源。采样前在父训练记录中创建 `heldout-attempt.json`，同一训练来源只能登记一次；切换续训记录不代表获得新的留出机会。五条指令、32 个环境、20 秒时长及判定阈值保持一致，评估器只改种子，并在创建模拟器前比较两份环境配置。

等效命令如下，在项目根目录的同一可见终端执行：

```bash
# 只读检查；不创建选择、尝试或仿真记录
bash scripts/validation/evaluate_heldout_continuation.sh --check-only
# 开发条件通过后，实际执行一次留出评估
bash scripts/validation/evaluate_heldout_continuation.sh
```

终端中的 `HELDOUT_EVALUATION_EXIT_CODE=0` 表示仿真和复核流程完成。结果保存在 `heldout-evaluation-link.json`，数值通过与否单独记录；还须检查固定指令录像，才能讨论稳定步态。留出失败要如实保留，不反复采样挑选通过结果。后续若依据该结果调整方案，应明确这次留出已经使用。

若日志已显示 `EVALUATION completed:`，但随后的 CPU 复核中断，不重新运行任务 21。使用该行给出的实际评估目录，仅复核保存数据：

```bash
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/verify_fixed_evaluation.py 'records/evaluations/<实际留出记录>'
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/heldout_evaluation.py finish 'records/evaluations/<实际留出记录>'
```

本轮完成了 18 项 [离线入口检查](../../records/reports/heldout-entry-check/check.json)：临时元数据覆盖模型、源码和证据变更、重复尝试、错误种子、环境差异及复核缺失；实际评估器的创建环境前代码确认开发/预检仍使用 seed 1001，留出分支使用 2001，并拒绝用留出做预检。临时数据仅用于测试检查逻辑，没有模型推理或仿真。

另外，在临时副本中重新复算已有官方参考评估的 160 条真实轨迹并完整解码五段视频，原始记录未改，见 [兼容检查](../../records/reports/heldout-entry-check/legacy-data-check.json)。Bash、两段内嵌 Python 和任务 21 的共享终端配置检查通过。准备阶段的真实入口曾因训练未结束而拒绝启动，未创建留出选择或尝试记录。本轮开发评估未通过，结果见第 19 节；留出 seed 2001 继续保留，当前不执行任务 21。

## 19. 自训基线固定评估

### 19.1 结果

本轮最终模型于 14:52:44 完成五条固定指令仿真，14:52:51 完成独立复核。每条指令 32 个环境、20 秒，开发 seed 1001，共 160 个首个 episode；全部完整存活、未发生跌倒，观测和动作均为有限值。但只有停止条件达到全部数值标准，前后行走与左右转向均未通过，因此当前不能验收稳定行走，也不进入奖励修改组。完整来源见[评估记录](../../records/evaluations/custom_onnx_reference_20261101T144436_s1001/evaluation.json)、[独立复核](../../records/evaluations/custom_onnx_reference_20261101T144436_s1001/verification.json)和[任务结果](../../records/experiments/baseline_continuation_20261101T101809_s42/fixed-evaluation-link.json)。

| 指令 | 平面速度误差（m/s） | 转向速度误差（rad/s） | 存活率 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 停止 | 0.00211 | 0.02024 | 100% | 通过 |
| 前进 0.15 m/s | 0.10558 | 0.15414 | 100% | 未通过 |
| 后退 0.10 m/s | 0.10018 | 0.03032 | 100% | 未通过 |
| 左转 0.4 rad/s | 0.00194 | 0.40771 | 100% | 未通过 |
| 右转 0.4 rad/s | 0.00200 | 0.41296 | 100% | 未通过 |

原定阈值保持为平面速度误差不超过 0.08 m/s、转向误差不超过 0.20 rad/s、完整存活率至少 95%。停止位移中位数为 0.00998 m，低于 0.15 m 阈值。平面速度误差先逐时刻计算二维速度差的长度，再对时间和 episode 等权平均；它包含横向偏移与波动，不能用平均前进速度与指令之差代替。

前进时实际平均纵向速度为 0.08833 m/s，后退时为 −0.00360 m/s；左右转向的实际平均 yaw 速度分别为 0.01808 和 0.01286 rad/s。曲线显示前进响应已有持续运动，但未跟上目标；后退和转向在初始过渡后大多接近静止。图中细线为全部 32 个首个 episode，粗线为均值，虚线为指令，未平滑，也未删除启动过渡。两幅面板和文本边界均已检查。

![指令响应](../../records/evaluations/custom_onnx_reference_20261101T144436_s1001/figures/command-response.png)

同一次评估的动作变化 RMS 和接触期间脚底滑移均保存在逐 episode 数据中。前进条件两项均值分别为 0.07515 和 0.01284 m/s；它们是附加观测，本次没有事后设置阈值来改变验收结果。终端在第 1000 步显示首个 episode 活动数为 0，是 20 秒时限到达后的正常结束；跌倒与超时由轨迹中的不同字段分别判定。

### 19.2 录像与界面

以下五段均为实际评估的环境 0，保留从初始化到首个 episode 结束的完整过程。每段 501 帧、25 fps、960×720，完整解码通过；因保留了 t=0 和 t=20 秒两个端点，视频时长为 20.04 秒，物理评估时长仍为 20 秒。单段录像不代表全部 32 个环境的表现，群体结果由完整轨迹计算。

**停止**

<video controls preload="metadata" src="../../records/evaluations/custom_onnx_reference_20261101T144436_s1001/videos/stop_env0.mp4"></video>

**前进**

<video controls preload="metadata" src="../../records/evaluations/custom_onnx_reference_20261101T144436_s1001/videos/forward_env0.mp4"></video>

**后退**

<video controls preload="metadata" src="../../records/evaluations/custom_onnx_reference_20261101T144436_s1001/videos/backward_env0.mp4"></video>

**左转**

<video controls preload="metadata" src="../../records/evaluations/custom_onnx_reference_20261101T144436_s1001/videos/turn_left_env0.mp4"></video>

**右转**

<video controls preload="metadata" src="../../records/evaluations/custom_onnx_reference_20261101T144436_s1001/videos/turn_right_env0.mp4"></video>

下图为评估进行到后退条件时的真实 VS Code 界面，沿用原终端、窗口尺寸与位置；截图来源和哈希见[界面记录](../../records/evaluations/custom_onnx_reference_20261101T144436_s1001/vscode-capture.json)。

![评估终端](../../records/evaluations/custom_onnx_reference_20261101T144436_s1001/vscode-evaluation-running.jpg)

### 19.3 官方模型参照

两项评估的保存环境文件、固定指令、开发种子、时长和指标实现一致；评估器源码快照不同，双方原始轨迹已用当前复算脚本核验。较早参考记录的复核在临时副本进行，原始文件保持不变。[比较记录](../../records/evaluations/custom_onnx_reference_20261101T144436_s1001/reference-comparison/comparison.json)绑定来源和哈希，[数据表](../../records/evaluations/custom_onnx_reference_20261101T144436_s1001/reference-comparison/metrics.csv)保留完整指标。

| 条件 | 目标 | 官方预训练模型实际均值 | 本机自训模型实际均值 |
| --- | ---: | ---: | ---: |
| 前进速度（m/s） | 0.15 | 0.00510 | 0.08833 |
| 后退速度（m/s） | −0.10 | 0.00001 | −0.00360 |
| 左转速度（rad/s） | 0.40 | 0.00177 | 0.01808 |
| 右转速度（rad/s） | −0.40 | −0.02727 | 0.01286 |

本机模型前进响应更明显，但两个模型在这套条件下都只有停止条件通过；本机模型的停止漂移、转向误差也没有整体优势。官方预训练权重的训练历史和预算未知，本机模型则经历保存、恢复和累计 5000 次更新，因此这里仅作效果参考，不能把差异解释为奖励修改的收益。线速度奖励 3.3 的正式对照尚未启动。

### 19.4 复查与后续训练

以下命令在项目根目录使用现有 Python，只读取已完成评估并重算、绘图，不启动仿真：

```bash
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/verify_fixed_evaluation.py records/evaluations/custom_onnx_reference_20261101T144436_s1001
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/compare_fixed_evaluations.py \
  records/evaluations/official_walk_reference_20261101T005022_s1001 \
  records/evaluations/custom_onnx_reference_20261101T144436_s1001 \
  --output "records/reports/reference-comparison-$(date +%Y%m%dT%H%M%S)"
~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/plot_fixed_evaluation.py records/evaluations/custom_onnx_reference_20261101T144436_s1001
```

本轮失败表现为指令跟踪不足，而非 20 秒内跌倒。训练预算不足、低速指令响应和策略学习程度仍需区分，当前不能认定单一原因。后续任务 22 已完成两项独立短运行，采样路径及分析器开销结果见第 20 节；后续并行规模筛查已完成，WSL 设置与有界追加训练记录见第 21 节，原生 Ubuntu 对照尚未执行。追加官方基线训练前，记录新的累计更新预算、恢复点和课程计数；继续保持官方奖励、既定评估阈值和独立修改组门槛，保留当前模型作为阶段记录。留出 seed 2001 和真机动作尚未执行。

## 20. 采样性能诊断

### 20.1 执行与记录

本轮由助手在已有 VS Code 窗口自动选择“XGO-Duck: 22 采样性能分析（独立短运行）”，复用任务 19、20 的同一个可见终端。启动前、任务选择器和启动后的截图均为 1707×1019，未观察到窗口或面板布局变化。自动操作通过 VS Code 自身入口唤起现有项目，再使用命令面板选择任务；此前会意外还原窗口的操作路径继续停用。[截图记录](../../records/profiling/profile_20261101T151652_s42/visibility/capture.json)保留文件哈希。

![任务选择](../../records/profiling/profile_20261101T151652_s42/visibility/vscode-picker.jpg)

![诊断终端](../../records/profiling/profile_20261101T151652_s42/visibility/vscode-running.jpg)

[冻结协议](../../configs/experiments/runtime_profile.json)规定先普通运行、再带分析器运行，各 27 次更新、1024 环境、seed 42。两组均从累计 5000 次的同一 checkpoint 恢复，恢复 Adam、归一化、课程计数和自适应学习率后重置环境；奖励、物理与录像设置一致。前 20 次等待、随后 2 次热身，只分析最后 5 次，即编号 5022–5026。每组产生的 27 次诊断更新独立保存，不加入正式基线预算，也不用于步态选型。

普通组正常退出；分析器组完成更新、保存最终 checkpoint 和原始跟踪后，在汇总检查中退出码为 1。原因是计数代码只用事件名称作为字典键，同名 GPU 注释覆盖了 CPU 调用计数。原始记录中 `user_annotation` 的环境步为 120 次，`gpu_user_annotation` 同名记录为 240 条，两者不应混计。该问题影响诊断汇总，不是正式模型训练失败。

已修正后续采集入口，使其保留设备类型且只用 CPU 区间核验调用次数；[四项检查](../../records/profiling/profile_20261101T151652_s42/recovery/profile-count-fix-check.json)验证真实记录、条目顺序变化、仅 GPU 条目和设备标识保留。原退出码、错误日志、原摘要和约 756 MB 的跟踪文件均保留，修复没有重跑训练。[恢复说明](../../records/profiling/profile_20261101T151652_s42/recovery/capture-recovery.json)与[独立复核](../../records/profiling/profile_20261101T151652_s42/verification.json)验证两组配置、来源、checkpoint 有限值与预算，实际采集到 120 次环境步、480 次物理步、5 次 PPO 更新，以及 417986 条 GPU 内核/复制/清零事件。分析器组缺失的结束摘要由已保存数据重建，单独标识为恢复记录；原进程并未改记为成功退出。

### 20.2 耗时与解释

下表全部来自事先指定的最后 5 次更新，每组均为 122880 个环境转移。计时截止到每次更新的日志结束，原始跟踪导出和事件整理的收尾时间不在表内；收尾曾持续数分钟，不能把它当成正式训练每次更新的开销。

| 指标 | 普通运行 | 带分析器运行 |
| --- | ---: | ---: |
| 五次更新总时间（秒） | 12.12187 | 17.42711 |
| 单次更新时间中位数（秒） | 2.39678 | 3.35513 |
| 吞吐（转移/秒） | 10137.05 | 7051.08 |
| 采样内部计时合计（秒） | 10.70963 | 15.07460 |
| 学习内部计时合计（秒） | 1.09159 | 1.70832 |

这一个顺序运行的诊断对中，带分析器的采集区间耗时为普通组的 1.43766 倍。不能将它当成受控提速试验，也不能把 43.77% 的差异全部归因于某一项分析器操作。图中保留了全部五个更新时间，竖线为中位数；下图显示带分析器运行的非嵌套入口区间，CPU 时间包含等待。

![采样耗时](../../records/profiling/profile_20261101T151652_s42/analysis/runtime-profile.png)

| CPU 区间 | 调用次数 | 五次更新累计（秒） |
| --- | ---: | ---: |
| 环境交互 | 120 | 14.30645 |
| PPO 更新 | 5 | 1.66442 |
| 更新日志 | 5 | 0.64391 |
| 策略动作 | 120 | 0.31878 |
| 有限值检查 | 120 | 0.07911 |

环境交互区间包含重置、奖励、观测、传感器和物理子步等工作。其中重置、奖励与观测的 CPU 区间分别为 3.72732、1.99011 和 1.15588 秒；这些是嵌套时间，不能与环境交互再相加。物理子步调用的 CPU 区间仅为 0.26162 秒，也不能据此认为 GPU 上的物理计算只用了这些时间，因为 GPU 调用是异步的。

按 CPU 自身时间排序，`cudaLaunchKernel` 为 4.25538 秒，`cudaStreamSynchronize` 为 2.37649 秒；说明本次带分析器运行中，内核派发和同步值得优先检查。普通组记录的 CUDA Graph 开关和四种图均已启用，跟踪组实际观察到 804 次 `cudaGraphLaunch`，因此不能简单归因于“没有打开 CUDA Graph”。从原始内核/复制/清零事件计算的 GPU 时间区间并集为 1.98148 秒；这不是 `nvidia-smi` 利用率，也不应与 CPU 等待时间相加。

这些证据支持先检查环境交互的主机侧开销及批量规模，尚不能证明 WSL 是唯一原因。当前遥测只有运行前后两次，分析器组的结束读数位于长时间 CPU 收尾之后；其 540 MHz 不代表训练期间一直低频。后续规模比较需记录运行中的时钟、功耗、温度和显存序列。[完整分析](../../records/profiling/profile_20261101T151652_s42/analysis/analysis.json)包含原始输入哈希、全部区间和边界遥测。

### 20.3 自行复查

在 WSL 的项目根目录执行以下命令，仅分析已经保存的数据，不创建环境或更新模型。现有记录已经恢复完成，不需要再次运行任务 22 来复查数值。

```bash
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
PAIR=records/profiling/profile_20261101T151652_s42
$PY -B scripts/validation/verify_profile_pair.py "$PAIR"
$PY -B scripts/validation/analyze_runtime_profile.py "$PAIR"
$PY -B scripts/validation/plot_runtime_profile.py "$PAIR"
```

第一个命令应输出 `status: passed`，同时如实保留 `original_exit_codes` 中的普通组 0、分析器组 1，以及 `recovered_modes` 中的 `traced`。第二个命令重算表中数据，第三个生成图和面板检查图；重新绘图后应检查标签和裁切。若独立复核内容被重新写入，后续两个命令也按上述顺序重算，以更新对应的文件哈希。

### 20.4 下一阶段

已固定[并行规模筛查协议](../../configs/experiments/parallel_env_benchmark.json)，入口预检通过后已由助手自动启动任务 23，实际结果继续记入第 21 节。在同一个可见终端顺序测试 1024、2048、4096 个环境，正序、逆序各一遍，每次 40 次更新，前 20 次热身；保持同一恢复模型、奖励、物理和依赖，关闭性能分析器，保留普通录像设置。每次独立记录样本数与更新数，所有诊断权重都排除在步态选型之外。

项目预定选择规则为：两遍吞吐比均大于 1、几何平均至少提升 15%，配置和有限值检查通过，观测显存占用不超过总量的 85%；同时检查时钟、功耗和执行顺序影响。该阈值是本项目的实用决策规则，不是官方要求。筛查后再固定正式基线的下一段预算；改变并行环境数也会改变优化 batch，不能将其当作等价步态训练。原生 Ubuntu 的同机对照仍保留，当前不自动重启或切换系统。

## 21. 并行规模与续训

### 21.1 测试方法

助手于 15:40:57 自动在原 VS Code 任务终端启动任务 23。按 `1024 → 2048 → 4096 → 4096 → 2048 → 1024` 顺序启动六个独立前台训练进程，每项都从正式 `model_4999.pt` 恢复，不接着前一项的诊断权重训练。seed 均为 42，恢复 Adam、自适应学习率、归一化统计和课程计数 120000，再重置环境。配置核对覆盖完整保存配置，只允许并行环境数、输出名称和本次更新预算变化。脚本注释说明改动、原因、预期和验证方法，入口为 [benchmark_parallel_envs.py](../../scripts/training/benchmark_parallel_envs.py)。

每项完成 40 次更新，只统计预先规定的最后 20 次，全部尖峰保留。吞吐定义为 `20 × 24 × 环境数 ÷ 完整更新区间的总秒数`，包括采样、学习与每次日志收尾；同时单列整个进程的耗时。初始录像落在热身区间，因此测量结果不包含长训练的周期录像开销。六项共采样 13762560 次环境转移，其中 6881280 次用于计时统计；这些样本不计入正式基线预算。

每 5 秒采集 GPU 利用率、显存、SM 时钟、功耗、温度、进程 CPU 时间和可用内存。显存读数包含 GPU 上其他活动，5 秒采样也可能漏掉瞬时峰值；表中“观测最大值”不等同于精确分配峰值。CPU 百分比以单个逻辑核为 100%，多线程进程可能超过 100%。每种规模只有两个顺序运行，同一个 seed，不能据此估计跨机器或跨 seed 的统计收益。

三种配置、异常拒绝路径与共享终端预检见[入口检查](../../records/reports/parallel-entry-check/check.json)。自动操作的真实画面如下，窗口及编辑器/终端布局保持原样；[截图清单](../../records/benchmarks/scale_20261101T154057_s42/visibility/capture.json)保存来源与哈希。

![任务选择](../../records/benchmarks/scale_20261101T154057_s42/visibility/vscode-picker.jpg)

![测试终端](../../records/benchmarks/scale_20261101T154057_s42/visibility/vscode-running.jpg)

### 21.2 测试结果

六项均完成 40 次更新并以 0 退出，15:57:20 结束采集；15:58:28 从保存数据独立复核通过。配置、恢复状态、40 个连续编号、有限 checkpoint、120 个预定计时点与六段各 200 帧录像均通过。[完整核验](../../records/benchmarks/scale_20261101T154057_s42/verification.json)绑定原始文件哈希。
| 环境数 | 正序吞吐（转移/秒） | 逆序吞吐（转移/秒） | 相对 1024 的几何平均倍数 | 整进程耗时（秒，正/逆） |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 10047.24 | 9858.06 | 1.000 | 134.4 / 135.1 |
| 2048 | 16084.19 | 16304.60 | 1.627 | 156.4 / 157.5 |
| 4096 | 24581.99 | 23976.73 | 2.439 | 198.9 / 200.5 |

4096 环境满足提前固定的选择规则，两遍提升比分别为 2.44664 和 2.43220，几何平均约 2.44 倍，选作下一段训练规模。1024 的单次更新中位数约 2.49 秒，4096 为 4.01、4.10 秒；每次更新虽然更长，每次处理的样本是原来的四倍。该结果只证明本机这次短测的采样吞吐改善，不代表更快达到步态标准，也没有验证原生 Ubuntu 的收益。

| 环境数 | GPU 利用率中位数（%，正/逆） | SM 时钟中位数（MHz） | 功耗中位数（W，正/逆） | 温度中位数（℃，正/逆） |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 41.0 / 45.0 | 1800 | 65.54 / 69.35 | 68.0 / 71.5 |
| 2048 | 48.5 / 51.5 | 1800 | 79.20 / 80.68 | 75.0 / 75.0 |
| 4096 | 68.0 / 64.5 | 1785 | 98.59 / 98.46 | 82.0 / 82.0 |

4096 环境的全程观测显存最高 4644 MiB，占 16384 MiB 的 28.35%；六项可用系统内存观测下限约 27.9 GiB。测量区间的进程 CPU 使用约为一个逻辑核的 100%。正、逆序结果方向一致，4096 组时钟略低、温度与功耗更高；保留这些条件差异，不能据六次短测声称长期性能或操作系统因果关系。

![并行规模](../../records/benchmarks/scale_20261101T154057_s42/figures/parallel-throughput.png)

图中每个吞吐点是一项运行；下图的点是全部 20 次预定更新，横线为中位数，不把更新点当成独立训练重复。两个面板已完成碰撞检查和逐图查看。完整批次原始状态保留在[运行记录](../../records/benchmarks/scale_20261101T154057_s42/batch.json)。

自动收尾最初调用系统 ffprobe，因未安装而退出 1。改用锁定环境中已有的 FFmpeg 后，完整解码并核验全部视频；保存数据复核退出 0。没有安装新包或重训，原终端截图、错误日志、修复前脚本及[修复核验](../../records/benchmarks/scale_20261101T154057_s42/verification-recovery/check.json)均保留。因此原批次的 completed-awaiting-verification 是采集结束时状态，应结合后续 verification.json 阅读。

下面固定展示首个 4096 环境运行的环境 0 训练片段。它包含随机采样和重置，其他五段在各自 trial 目录；这段录像不作为稳定行走验收。

<video controls preload="metadata" src="../../records/benchmarks/scale_20261101T154057_s42/trial-03-n4096/training-env0.mp4"></video>

### 21.3 自行复查

在 VS Code 打开本项目目录，使用 `Terminal → Run Task…` 选择“XGO-Duck: 23 并行环境规模测试（独立短运行）”可启动一批新的规模测试；任务复用共享终端，持有计算锁，另一项计算任务未结束时会退出。已经完成的批次只需运行下面的保存数据复核，无须重训。

```bash
cd /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
BATCH=records/benchmarks/scale_20261101T154057_s42
$PY -B scripts/validation/verify_parallel_benchmark.py "$BATCH"
$PY -B scripts/validation/plot_parallel_benchmark.py "$BATCH"
```

复核应逐项输出六次 `VERIFIED`，最后为 `status: passed`；绘图保留 120 个实际更新耗时。若最终复核失败，应保留已完成训练和错误日志，检查保存数据与脚本，不立即重复整批训练。

### 21.4 追加训练

规模测试通过后，已冻结[追加训练协议](../../configs/experiments/baseline_extension.json)。从正式 5000 次模型追加 2500 次至累计 7500 次，seed 42、4096 环境，每次每环境 24 步。父段采样 122880000 次，本段计划新增 245760000 次，合计 368640000 次，约为官方 README 示例按默认上限计划样本量的 7.5%。这是分段学习与评估安排，尚不能称为完整执行官方长训练预算。

实际 CPU loader 预检核对了网络、Adam 和归一化状态的精确恢复，拒绝错误编号、课程计数、学习率和预算，检查相同/不同并行环境数的真实旧 checkpoint 记账，未创建仿真或执行优化器步。结果见[入口核验](../../records/reports/extension-entry-check/check.json)。[extend_official.py](../../scripts/training/extend_official.py) 注释说明只改并行数和本段预算的原因、预期与验证方法；[可见任务入口](../../scripts/training/experiments/extend_baseline.sh) 完成锁检查后启动同一个训练监督脚本。官方奖励、物理和冻结评估标准保持原样。

助手已自动启动“XGO-Duck: 24 官方基线追加至 7500 次”，运行记录为 [baseline_extension_20261101T160203_s42](../../records/experiments/baseline_extension_20261101T160203_s42/training-request.json)。16:02:34 实际加载并 reset，下一更新编号为 5000，课程计数为 120000，自适应学习率恢复到 0.0000759375。16:04:48 保存的日志与 TensorBoard 前缀核验覆盖编号 5000–5028，共新增 29 次、累计 5029 次；该数字是当时的已核验快照，后续实际终端继续前进。[前缀证据](../../records/experiments/baseline_extension_20261101T160203_s42/resume-verification/20261101T080448266775Z/check.json)与[恢复记录](../../records/experiments/baseline_extension_20261101T160203_s42/resume-ready.json)可独立查看。

16:19:03 的[第二次前缀核验](../../records/experiments/baseline_extension_20261101T160203_s42/resume-verification/20261101T081903845858Z/check.json)覆盖编号 5000–5254，共新增 255 次、累计 5255 次。两次完整恢复点已在 CPU 加载检查：各自 76 个张量均为有限值，actor 与 critic 的网络参数均实际更新，原件与恢复副本哈希一致。这里的归一化计数对 actor 和 critic 相同。

| 恢复点 | 累计更新 | 课程计数 | 归一化样本数 | Adam 计数 |
| --- | ---: | ---: | ---: | ---: |
| [model_5000.pt](../../records/experiments/baseline_extension_20261101T160203_s42/checkpoint-checks/model_5000.json) | 5001 | 120024 | 122978304 | 100020 |
| [model_5250.pt](../../records/experiments/baseline_extension_20261101T160203_s42/checkpoint-checks/model_5250.json) | 5251 | 126024 | 147554304 | 105020 |

模型副本保存在 `artifacts/recovery/baseline_extension_20261101T160203_s42/`。日志已完成的更新可能多于最近完整 checkpoint；中断恢复以模型实际保存的次数为准，不能把日志中未保存的更新当作可恢复状态。本段最终 7500 次预算仍未完成。

环境数改变后，归一化样本数按“父模型原有样本数 + 本段更新数 × 24 × 本段环境数”计算；不能用新环境数乘此前的全部课程步数。课程计数与 Adam 计数仍按实际累计更新计算。新入口会恢复这些状态并重置环境，不能恢复未保存在 checkpoint 中的完整随机和 episode 历史。固定指令评估通过后，才进入留出评估与成对修改训练。

![追加训练](../../records/experiments/baseline_extension_20261101T160203_s42/resume-verification/20261101T081903845858Z/figures/continuation-progress.png)

曲线使用 16:19 快照中的前 255 次更新原始 TensorBoard 值，不平滑，含重新开始的 episode、录像热身与全部耗时波动；数据与三幅分图检查保存在[曲线目录](../../records/experiments/baseline_extension_20261101T160203_s42/resume-verification/20261101T081903845858Z/figures)。底部吞吐是训练器内部采样与学习计时的 FPS，与第 21.2 节包含日志收尾的完整更新区间定义不同。奖励或 episode 长度上升均不代替固定指令步态验收。

![可见续训](../../records/experiments/baseline_extension_20261101T160203_s42/visibility/vscode-progress-5250.jpg)

同一个 VS Code 窗口上方已切换到本段训练曲线，下方继续原任务；这是保存时刻的曲线快照，终端输出持续更新。切换前后截图均为 1707×1019，窗口与面板布局保持不变，[本次截图记录](../../records/experiments/baseline_extension_20261101T160203_s42/visibility/progress-5250-capture.json)保存哈希，[启动截图](../../records/experiments/baseline_extension_20261101T160203_s42/visibility/capture.json)另行保留。

本段已归档采样步数 0、6000 的两段真实训练录像，各自完整解码为 200 帧、50 fps、4 秒，原件与副本哈希一致，见[媒体核验](../../records/experiments/baseline_extension_20261101T160203_s42/live-media-check.json)。下方展示 6000 步片段，[初始片段](../../records/experiments/baseline_extension_20261101T160203_s42/videos/rl-video-step-0.mp4)保留供回看。步数是本次启动的环境控制步，不是优化器更新次数；两段均来自环境 0，采用随机训练条件，不是同指令下的 checkpoint 效果比较。

<video controls preload="metadata" src="../../records/experiments/baseline_extension_20261101T160203_s42/videos/rl-video-step-6000.mp4"></video>

### 21.5 运行与恢复

任务 24 曾用于启动本段，随后在 model_5750.pt 转入带实时观察器的任务 25；发现持续并脚异常后，任务 25 也已停止，当前执行位置见第 22 节，不要重新启动旧训练入口。原设计是在训练正常结束后，于同一个可见终端核验最终 model_7499.pt、导出 ONNX、复算 1000 帧动作一致性，再执行五条固定开发指令及独立轨迹/视频复核。本段尚未走到这些步骤，不能把已准备的收尾脚本写成已执行结果。

评估结束后的[比较脚本](../../scripts/validation/compare_fixed_evaluations.py)使用实际评估记录标识作为 CSV 标签，避免把后续模型固定写成“5000 次”。已用官方预训练参考与已有 5000 次基线的保存轨迹重新计算，十行条件指标与原结果逐项一致，原始比较记录保留；[核验记录](../../records/reports/comparison-label-check/check.json)绑定源码与输出。第 19.4 节的命令使用新的输出目录，该目录若已存在会拒绝覆盖。本段 7500 次模型的评估尚未产生，不能把这次已有数据复算当作新模型结果。

旧任务 16、19、20 分别绑定前一段续训、导出和评估，不用于这段追加模型。需要更新训练曲线时，可在 WSL 的项目根目录执行以下仅分析保存记录的命令；输出是带时间戳的快照，不会再启动训练：

```bash
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
RUN=baseline_extension_20261101T160203_s42
$PY -B scripts/validation/verify_resumed_progress.py "$RUN"
$PY -B scripts/validation/plot_continuation_progress.py "$RUN"
```

新快照路径由 `resume-live-check.json` 的 `evidence_directory` 给出，图在其 `figures/continuation-progress.png`。在 VS Code 用 `Ctrl+P` 输入图片路径可直接查看，不需调整窗口或启动浏览器。图表中的训练 FPS 只用于过程观察，比较规模应使用第 21.2 节冻结的统计范围。

若需要暂停，在当前任务终端按 `Ctrl+C`，等待退出，再核对已经完整写出的 checkpoint；不要直接重复任务 24。入口会拒绝再次执行同一冻结方案，避免无意从 5000 次重跑。恢复前须登记新父模型、其实际累计更新/样本量和剩余预算；未保存的更新不能恢复。若只在导出、视频或指标复核阶段失败，优先使用保存数据修复对应步骤。本段最终步态和 7500 次预算目前仍待完成后核验。

<a id="gait-diagnosis"></a>

## 22. 并脚与显示诊断

### 22.1 异常证据

用户连续指出双腿并拢。实际数据确认这是持续内收的退化姿态：实时观察环境的双髋从 home 附近约 ±5° 移到内侧限位附近，双脚在身体横向轴上的顺序甚至发生交换。开启显示并没有修改策略，因此画面中的异常不会随显示方式自动消失。

16:57:19–17:00:19 冻结的 406 条实际训练采样中，左右髋侧摆中位数分别为 −22.47°、+22.49°；双脚有符号横向间距最小 −12.61 mm，其中 166 条为负。间距定义为“左脚位置减右脚位置，在躯干左向轴上的投影”，负值表示脚部参考点横向交叉，不等于已计算出网格穿透深度。这些是一个训练环境的相关采样，期间策略、指令和 episode 可改变，不能当成 406 次独立试验。原始快照、CSV、配置和复算见[诊断记录](../../records/experiments/baseline_extension_20261101T165647_s42/gait-diagnostics/20261101T090019523741Z/analysis.json)。

![腿部状态](../../records/experiments/baseline_extension_20261101T165647_s42/gait-diagnostics/20261101T090019523741Z/leg-adduction.png)

上图不平滑、不删点。上幅虚线为零横向间距，下幅虚线为模型髋侧摆 ±22° 限位；观测值可略超出限位，图中保留实际值。短暂回到初始姿态不能代表策略已修复。

回查以前保存的固定评估也发现同一问题，说明它早于实时查看器接入。每种指令各有 32 个环境、20 秒的首个 episode；包含全部有效帧，未挑选时间段。两套模型来源和训练预算不同，此处只比较姿态现象。

| 前进条件 | 左髋侧摆中位数 | 右髋侧摆中位数 | 双髋向内超过 20° 的帧占比 |
| --- | ---: | ---: | ---: |
| 官方预训练参考 | −4.86° | +4.97° | 0% |
| 自训 5000 次模型 | −22.53° | +22.51° | 99.53% |

自训模型在五种指令下的上述帧占比均超过 99%。前进时，它请求的左右髋侧摆目标角中位数为 −310.31°、+147.69°；这是策略发出的目标，实际关节没有转过这些角度，而是长期停在约 ±22° 附近。目标重建已与保存的动作、policy home 和动作尺度逐项核对。[官方参考复算](../../records/reports/saved-joint-posture/official_walk_diagnostic_reference_repeat_20261101T012547_s1001/analysis.json)和[自训模型复算](../../records/reports/saved-joint-posture/custom_onnx_reference_20261101T144436_s1001/analysis.json)保留各条件的范围、哈希与模型身份。原有“停止达到速度/漂移数值标准”的结论保持其原范围，不能延伸为姿态或真机验收通过。

### 22.2 碰撞规则

当前训练编译后的模型中，地面为 `contype=1, conaffinity=1`，两只脚均为 `contype=1, conaffinity=0`，没有显式脚间接触对。依照 [MuJoCo 碰撞选择规则](https://mujoco.readthedocs.io/en/stable/computation/index.html#selection)，脚与地面通过掩码检查，两只脚之间则不通过。归档的 [XGO-Duck 配置](../../official_resources/xgoduck_rl/src/mjlab_microduck/robot/xgoduck_constants.py)也明确采用脚碰地、脚间不碰撞的行走设置；本轮没有改过该官方归档。

406 条采样中的 GPU 自碰撞信号全为零。由于脚间接触本来就被过滤，这个零值不能证明双脚没有相交。已确认的直接机制是“策略持续请求过大的髋侧摆目标，实际关节向内顶住限位；脚间接触未阻止双脚靠近”。哪项奖励及学习过程使策略偏向这种动作，仍须受控诊断，不能仅凭此认定某个奖励权重就是根因。

### 22.3 画面速度

任务 25 使用[只读观察器](../../scripts/training/live_observer.py)：把实际训练环境 0 的状态复制到独立 CPU 模型，进行运动学与接触重建后显示；不推进训练仿真、不调用策略、不把鼠标扰动传回训练。网络、优化器、归一化和物理参数保持原样。初始化及退出连接测试、CPU 加载检查均通过，见[查看器检查](../../records/reports/live-view-entry-check/gui-check.json)与[恢复检查](../../records/reports/live-view-entry-check/resume-check.json)。原始连接测试退出异常及清理线程后的修复记录均保留。

该入口限制最高 4 Hz，上述快照实测为 2.26 帧/秒。4096 个环境同步采样时，409 个控制步代表 8.18 秒仿真时间，却花了 179.53 秒墙钟时间；直接观察训练状态因此呈现慢速逐帧变化，提高窗口刷新上限不能生成更多真实仿真状态。观察器同步复制、重建及日志累计耗时 5.32 秒，占该区间约 2.96%，但此计时不含异步绘制、GPU 争用等开销，不能据此宣称实时显示只使训练慢了 2.96%，也不能据此判断原生 Ubuntu 的提速幅度。

已保存录像按仿真时间播放，能够流畅查看动作。下面是同一训练段的 4 秒片段，完整解码为 200 帧、50 fps、960×720；它是已经录制的训练过程，不是连续实时画面，也不是固定 checkpoint 的评估。文件副本哈希核对及完整解码见[媒体记录](../../records/experiments/baseline_extension_20261101T165647_s42/media-manifest.json)。

<video controls preload="metadata" src="../../records/experiments/baseline_extension_20261101T165647_s42/videos/rl-video-step-0.mp4"></video>

实时窗口已由训练进程启动，日志与帧计数确认状态持续更新；原生窗口截图未核验。用户明确允许截图后，桌面工具仍返回未批准访问 `msrdc`，没有换工具绕过。VS Code 任务启动截图与训练录像可查看，不能用它们冒充原生实时窗口截图。后续显示必须区分当前训练状态、保存训练录像和固定模型回放；若要流畅连续观察模型动作，需要单独安排回放计算及其预算。

### 22.4 保存与停止

16:54 按用户选择，在新完整 model_5750.pt 核验后停止任务 24，保存累计 5751 次更新。模型 SHA-256 为 `a4faf3aec1b5a4b4d264eb488860c1da6ad28314d330752f7a4ad77fa1810019`，归一化样本数 196706304、课程计数 138024、Adam 计数 115020。停止时日志比模型多完成 1 次，该更新未保留；原结果与恢复副本各自留档。

随后自动在同一个 VS Code 终端 3 启动任务 25，运行标识为 `baseline_extension_20261101T165647_s42`，计划从 5751 继续剩余 1749 次。实际加载、reset 与新采样通过了[前缀核验](../../records/experiments/baseline_extension_20261101T165647_s42/resume-live-check.json)。新 episode 和随机历史从恢复边界重新开始，因此不是逐轨迹无缝续训。

![可见诊断入口](../../records/reports/live-view-entry-check/vscode-task25.jpg)

用户再次指出并脚异常后，17:09 停止这段续训并转入定位。该段新增 78 次完整更新，最后日志编号 5828，尚未到下一个保存间隔，因而没有新 checkpoint；78 次更新及可能进行中的部分采样未保留。可恢复模型仍为累计 5751 次的 model_5750.pt，退出码 130、进程结束与未保存数量见[停止核验](../../records/experiments/baseline_extension_20261101T165647_s42/gait-diagnostic-stop-result.json)。没有把日志中的 5829 次当作可恢复模型，也没有生成 7500 次完成报告。

### 22.5 固定权重诊断

长训练现已停止，训练入口会在创建实验前拒绝继续学习，直到并脚异常完成定位与修复验证；P0–P7 项目目标仍在推进。为区分动作请求和碰撞规则的作用，先冻结[短诊断协议](../../configs/experiments/gait_mechanism_diagnostic.json)，再自动运行 VS Code 任务 **26 并脚原因诊断（固定模型，不训练）**。它使用已导出并验证的 5000 次模型，seed 1001；每组仅停止、前进两条固定指令，每条 8 个环境、8 秒。没有使用留出种子或更新权重。

| 组别 | 唯一改动 | 要回答的问题 |
| --- | --- | --- |
| reference | 无 | 短诊断是否重现内收 |
| foot_contact | 两脚增加共同碰撞位；仍保留地面接触 | 脚间接触能否阻止当前异常，是否影响存活和跟踪 |
| hip_roll_guard | 仅把髋侧摆请求目标限制为 ±10° | 过大请求是否直接维持内收，限制后是否仍能保持平衡 |

±10° 是本次探索性的姿态限制，参考官方模型髋侧摆接近 ±5° 的观察设置；它不是硬件标定结果，也不是已经证明有效的控制范围。修改只存在于这组评估进程，不改官方归档、不覆盖原模型。各组保存原始策略目标、实际发送目标、关节位置、脚部位置、躯干姿态、接触、首个 episode 轨迹及 25 fps 录像。批次 [gait_mechanism_20261101T171428_s1001](../../records/evaluations/gait_mechanism_20261101T171428_s1001/batch.json)于 17:25:33 完成；三组速度/存活指标已独立复算，六段视频已完整解码。[姿态复核](../../records/evaluations/gait_mechanism_20261101T171428_s1001/posture-analysis.json)另行核对单因素配置、原始目标、实际目标和躯干坐标变换，输出[对照表](../../records/evaluations/gait_mechanism_20261101T171428_s1001/posture-comparison.csv)。

| 条件 | 组别 | 脚间距中位数（mm） | 双髋内收超过 20° 的帧占比 | 完整存活 | 平面速度误差（m/s） |
| --- | --- | ---: | ---: | ---: | ---: |
| 停止 | 原样 | −1.38 | 98.47% | 8/8 | 0.0044 |
| 停止 | 脚间碰撞 | 9.90 | 40.41% | 6/8 | 0.0640 |
| 停止 | 髋侧摆限制 | 38.82 | 0% | 8/8 | 0.0303 |
| 前进 | 原样 | −1.48 | 98.81% | 8/8 | 0.1063 |
| 前进 | 脚间碰撞 | 9.08 | 64.85% | 1/8 | 0.2595 |
| 前进 | 髋侧摆限制 | 36.95 | 0% | 8/8 | 0.1547 |

原样组前进时，左右髋角中位数为 −22.55°、+22.53°；限制组变为 −9.35°、+9.48°，全部有效帧的有符号脚间距至少为 25.46 mm，8 个首个 episode 都没有参考点交叉。该干预支持“过大的髋侧摆请求直接维持内收”的判断。限制组请求前进 0.15 m/s，实测平均前向速度只有 0.0309 m/s，yaw 误差为 0.2627 rad/s；平面速度和 yaw 均超过原定 0.08 m/s、0.20 rad/s 阈值。原样组平均前向速度为 0.0873 m/s，也未通过平面速度标准。

脚间碰撞组的停止条件跌倒 2 次、前进条件跌倒 7 次，表明把该约束直接加到已学会并脚的固定策略上会明显破坏平衡。这里不能据此断言“训练时不应启用碰撞”，也不能只以脚间距变大判定修改有效。各行姿态统计包含该组全部首个 episode 的有效帧；跌倒组采样时间较短，帧占比不是独立成功概率。速度指标先按各 episode 的实际有效时长求均值，再等权汇总；短时跌倒产生的速度不能当作有效行走。

下面依次为原样、仅启用脚间碰撞、仅限制髋侧摆目标的前进录像，均固定显示环境 0。第一、第三段各 201 帧、8.04 秒；第二段在仿真 0.92 秒跌倒，24 帧、0.96 秒，按真实终止时间结束，没有循环补足。录像按仿真时间以 25 fps 播放，不代表计算达到实时速度；全体 8 个环境的结果以上表为准。

<video controls preload="metadata" src="../../records/evaluations/custom_onnx_mechanism_reference_20261101T171443_s1001/videos/forward_env0.mp4"></video>

<video controls preload="metadata" src="../../records/evaluations/custom_onnx_mechanism_foot_contact_20261101T171627_s1001/videos/forward_env0.mp4"></video>

<video controls preload="metadata" src="../../records/evaluations/custom_onnx_mechanism_hip_roll_guard_20261101T172402_s1001/videos/forward_env0.mp4"></video>

第二组首次复核曾失败：记录代码保存了动作数组的引用，episode 结束后的复位把其中 9 行动作清零，而在线统计已经在复位前完成。由于本诊断滤波系数严格为零，独立复制保存的 `applied_actions` 与原始动作精确相等，可用于恢复这 9 行。修复前的两份 NPZ 留在[原始记录目录](../../records/evaluations/custom_onnx_mechanism_foot_contact_20261101T171627_s1001/trace-repair)；[修复清单](../../records/evaluations/custom_onnx_mechanism_foot_contact_20261101T171627_s1001/trace-repair/repair.json)保存索引、哈希与限制条件，全部其他数组、在线指标、模型及视频保持不变。修复后[独立核验通过](../../records/evaluations/custom_onnx_mechanism_foot_contact_20261101T171627_s1001/verification.json)，未重跑仿真。

随后任务 27 只补充第三组，并将动作记录改为独立复制。源码语法树比较确认执行代码只改变了这一记录操作，未改变物理或推理；第一、第二组只复核保存数据。[原失败批次](../../records/evaluations/gait_mechanism_20261101T171428_s1001/batch-failure.json)和控制台输出保留，避免把修复过程隐藏为一次无故障执行。

![可见执行](../../records/evaluations/gait_mechanism_20261101T171428_s1001/visibility/vscode-task27.jpg)

截图下方是复用终端 3 的任务 27，上方仍是此前保存的训练片段，不是该诊断的实时画面。诊断完成后已在同一 VS Code 编辑区打开并播放限制组的前进录像。

![诊断回放](../../records/evaluations/gait_mechanism_20261101T171428_s1001/visibility/vscode-guard-playback.jpg)

播放器已推进至 8.04 秒，下方任务显示完成；[播放核验](../../records/evaluations/gait_mechanism_20261101T171428_s1001/visibility/playback.json)记录所播放的文件及截图哈希，窗口和面板未执行调整操作。

当前顺序是先解释并脚及控制约束，再验证修复后的站姿、接触与固定指令表现，之后重新登记可学习的配置和预算。原有 3.0/3.3 奖励比较仍后置；不能把诊断性限制后的策略当成未经修改的官方基线，也不能直接部署到真机。

### 22.6 自行复查

任务 26、27 已完成，不需要再次启动仿真。任务 26 会拒绝对同一冻结协议创建重复批次；任务 27 检测到本批次全部完成后直接返回。只复算已有证据时，在 WSL 的 Windows 项目根目录使用下面的命令；它们不进行仿真或训练：

```bash
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
cd /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck
RUN=baseline_extension_20261101T165647_s42
SNAP=records/experiments/$RUN/gait-diagnostics/20261101T090019523741Z
$PY -B scripts/validation/analyze_live_gait.py "$RUN" --snapshot "$SNAP"
$PY -B scripts/validation/summarize_saved_joint_posture.py custom_onnx_reference_20261101T144436_s1001
$PY -B scripts/validation/summarize_saved_joint_posture.py official_walk_diagnostic_reference_repeat_20261101T012547_s1001
BATCH=records/evaluations/gait_mechanism_20261101T171428_s1001
$PY -B scripts/validation/verify_gait_diagnostic.py "$BATCH"
$PY -B scripts/validation/audit_gait_constraints.py --output "$BATCH/action-constraint-audit.json"
```

读取 analysis.json 中的样本数、模型哈希、关节角与碰撞掩码，再打开同目录曲线；录像通过 VS Code 的文件预览或报告中的播放器播放。首次绘图曾因坐标刻度超出图框被检查拦截，限制刻度后重绘通过，保存数据没有变化。后续若仅在分析或媒体处理处失败，应重算该步骤，保留原始轨迹，不重新执行训练来修复报告。

### 22.7 修复路线

本阶段交付的是原因诊断，尚未交付稳定行走模型。下一阶段先检查训练中的目标约束，再设计有界的修复试验：保持现有模型作回退，冻结修改、起点、采样预算与判定条件；短训练达到姿态要求后，再按完整五指令协议验收。不得为了让限制组通过而放宽已有跟踪标准。

官方归档已经定义 [action_over_limit_penalty](../../official_resources/xgoduck_rl/src/mjlab_microduck/tasks/mdp.py)，按“请求目标超出关节硬限位加余量的部分”处罚。该函数当前用于另一项 rollers 任务，行走任务的已保存配置未启用它；现有 `dof_pos_limits` 针对实际关节位置，不能与请求目标约束混为一谈。不能直接照搬另一任务的权重，也不能声称启用它必然恢复步态。

本轮已完成该函数的[离线核验](../../records/evaluations/gait_mechanism_20261101T171428_s1001/action-constraint-audit.json)。检查使用未改动的官方函数、实际安装类的动作属性，以及五条件共 160000 帧保存评估数据；14 个关节名称和弧度限位与 XML、训练观察器逐项一致。函数输出与独立 NumPy 重建一致。采用该函数默认的 0.3 rad 余量时，前进条件的越界量跨关节累加后平均为 16.10 rad，其中双髋侧摆占 41.56%；这个数是多个关节的超额之和，不是单个关节转角。说明后续还需检查其他关节的请求，不能只让双脚看起来分开。检查未构建仿真环境、未更新权重；它验证接口与计算量级，不等于奖励权重已选定或训练改善已验证。首次分析曾把按环境保存的 home 当成单个向量，广播检查阻止了输出；改为按有效帧保留环境对应关系后通过。

若训练或部署采用目标限制，仿真、导出回放与真机执行必须使用同一约定，记录限制前后的目标；±10° 诊断值不能直接作为真机限位。碰撞修改也应让策略在相应物理条件下重新学习并独立验收。后续先验证正常站姿、脚间接触和动作范围，再验证行走速度、转向及停止；真机仍按第 8、9 节先完成 ID、零位和 IMU 标定。原有 3.0/3.3 成对比较保留到基线通过之后。

<a id="gait-repair-pilot"></a>
## 23. 并脚修复短训练

### 23.1 目的与预算

本试验检验策略能否通过学习缩小过大的关节请求，并改善姿态和运动跟踪。固定策略的限幅诊断已经能把脚分开，却没有恢复行走，因此本次把约束作为连续惩罚加入训练，让策略重新调整平衡。执行动作本身不裁剪，碰撞、其他奖励和评估阈值保持原样。

[冻结协议](../../configs/experiments/gait_repair_pilot.json)在启动前写定，SHA-256 为 `f9bd268a6999506670a3e56ef7489c476b1d8499d540b9b4e82fd14268d95724`。两组从已完成导出和固定评估的同一份 model_4999.pt 开始，其哈希为 `350a1a7300213e94ba6f9e98a3662ffa40bf2b8995153e51f2f60460ce80ceb7`；累计 5751 次的模型继续保留作恢复依据，不混入本次比较。

| 项目 | 两组共同设置 |
| --- | --- |
| 组别 | control 原样组；target_envelope 目标约束组 |
| 起点与随机种子 | 累计 5000 次模型，seed 42 |
| 新增预算 | 各 250 次更新，每次 4096 环境 × 24 步，即各 24576000 次转移 |
| 最终计数 | 累计 5250 次，最后标签 5249；不是 7500 次长训练的完成记录 |
| 恢复状态 | 同一份网络、Adam、归一化、学习率与课程计数；重新创建相同种子的环境和 episode |
| 保存与画面 | 每 50 次保存模型；每 1200 个控制步录制 200 帧；实际状态观察器上限 4 Hz |
| 最终检查 | 只导出最后模型，1000 帧一致性，再做原五指令各 32 环境、20 秒、seed 1001 的评估 |

250 次是用于判断修复方向的有界探索预算。该试验只有一个训练种子，不能据此推断跨种子的稳定收益；后续 3.0/3.3 比较仍保留原计划。新增目标惩罚也使两组总奖励的含义不同，不能仅比较奖励总值来选择模型。

### 23.2 修改位置

实际修改保存在项目自己的 [gait_rewards.py](../../scripts/training/gait_rewards.py)，官方归档与工作副本的官方任务定义均未改写。函数先重建策略请求目标 `q_target = raw_action × scale + offset`，再按每个关节计算超额：`max(q_target − upper, 0) + max(lower − q_target, 0)`，最后对 14 个受控关节求和。

双髋侧摆的探索范围为 ±10°，其余关节使用硬限位外加 0.3 rad 余量；惩罚权重固定为 −0.1。这一范围来自上一节的姿态敏感性诊断，尚不是实际舵机标定值。采用软惩罚后，策略仍可能请求范围外的目标，因而必须检查实际轨迹，不能只看配置中写了限位。

新增项由 [gait_repair.py](../../scripts/training/gait_repair.py)只注入目标约束组。入口逐项比较配置，要求组间唯一学习差异为这一项奖励；两组共同提高保存和录像频率。代码开头及奖励函数注释说明修改、原因、预期效果与限制。每组记录目录保存实际参数、协议、源码副本和哈希。原长训练的异常检查继续生效，这个入口只允许已冻结的两组各 250 次预算。

### 23.3 启动前核验

任务启动时重新执行了 [CPU 预检](../../records/experiments/gait_repair_pilot_20261101T174853_s42/preflight.json)。它调用实际 runner 加载器，核对网络、归一化和 Adam 恢复前后相等，下一次迭代为 5000，课程计数为 120000；没有在预检中执行优化器更新或创建 CUDA 环境。

奖励检查使用五种指令共 160000 帧的保存轨迹及实际动作类的属性。独立 NumPy 计算与实际奖励函数相符，且原始动作与物理关节限位没有被改写。前进轨迹的目标超额均值为 17.1189 rad，乘 −0.1 后为 −1.7119，这是多个关节累计的惩罚量级，尚未乘奖励管理器的时间步长；不是单个关节角，也不是新训练的收益。检查同时验证用户暂停会阻止启动、扩大为 2500 次会被拒绝、VS Code 使用共享可见终端。

### 23.4 执行与复现

本机已在现有 VS Code 的 `Terminal → Run Task…` 中启动 **XGO-Duck: 28 并脚修复短训练（成对 250 次）**，复用终端 3。批次为 [gait_repair_pilot_20261101T174853_s42](../../records/experiments/gait_repair_pilot_20261101T174853_s42/batch.json)。任务先执行原样组，再执行目标约束组；各组正常完成后，在同一终端核验模型、导出 ONNX、运行五指令评估与姿态复算。当前处于执行中，学习效果尚待评估。

训练时原生 MuJoCo 窗口显示当前训练环境 0，更新速度受实际计算限制；保存的 MP4 按仿真时间播放。两者均来自真实状态，但不能把保存片段当作实时画面。每组记录目录的 `progress.json` 给出新增更新数，`training.log` 保留命令和输出，`iteration-timing.jsonl` 保留逐次采样、学习与墙钟计时。完整片段每 10 次更新检查并复制到记录目录，短暂没有新文件不表示训练停止。

原样组首个保存点 model_5000.pt 已通过[独立 CPU 核验](../../records/experiments/gait_repair_control_20261101T174853_s42/checkpoint-checks/model_5000.json)：累计 5001 次更新、归一化样本数 122978304、课程计数 120024、Adam 计数 100020，actor 与 critic 各有 8 个网络张量发生变化，所有检查张量有限。它证明实际新学习及保存已经发生，不是最终预算或步态验收。

下面是本次原样组的第一个实际训练片段。它已与 Linux 原文件核对哈希，并完整解码为 200 帧、50 fps、960×720、4 秒，见[媒体检查](../../records/experiments/gait_repair_control_20261101T174853_s42/media-check/check.json)。VS Code 播放进度已到 4 秒，见[播放记录](../../records/experiments/gait_repair_pilot_20261101T174853_s42/visibility/playback.json)。这一组尚未增加目标惩罚，录像不能解释为修改后的效果。

<video controls preload="metadata" src="../../records/experiments/gait_repair_control_20261101T174853_s42/videos/rl-video-step-0.mp4"></video>

![训练回放](../../records/experiments/gait_repair_pilot_20261101T174853_s42/visibility/vscode-training-playback.jpg)

本次桌面工具只捕获了 VS Code 局部区域；该图用于证明打开的训练片段，不能证明整个窗口布局保持不变。操作未发出调整窗口或面板大小的指令，后续停止额外桌面激活操作，训练继续由原任务终端执行。原生窗口没有单独的访问核验，不用遮挡在其他应用上方的图像代替该核验。

要自行检查准备工作，可在可见 WSL 终端执行以下只读入口；它不会开始学习：

```bash
cd /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck
bash scripts/training/experiments/gait_repair.sh --check-only
```

只核验已保存训练视频时，使用 `~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B scripts/validation/audit_repair_training_media.py gait_repair_control_20261101T174853_s42`。脚本逐帧解码并登记完整或不足 200 帧的片段，不创建仿真；结果写回该运行的 `media-check/check.json`。

本批次正在运行或已经存在时，不要再次点击任务 28；入口会拒绝对同一协议重复训练。如果导出或分析失败，保留已完成的权重和仿真记录，定位该步骤后在同一可见终端运行 `finish_gait_repair.py` 继续后处理，不通过重训修复报告。需要停止时，在正在运行的任务终端按 Ctrl+C；只有完整写出的 checkpoint 可以恢复，尚未保存的更新须登记。

### 23.5 判定方法

最终逐条件报告实际髋角、双脚在身体横向轴上的有符号间距、目标超额、速度误差、存活和跌倒。姿态检查要求首个 episode 中没有脚部参考点交叉，也没有双髋同时向内超过 20° 的帧；完整运动标准继续使用原协议。只有姿态与五指令运动标准同时通过，才能称修复达到本轮仿真标准。

如果目标超额下降而姿态或跟踪失败，结论只能是该短预算下修复不足。训练结束、奖励提高、模型导出成功和视频看起来好看均不能替代该判定。本试验不使用留出种子，也不构成真机部署许可或标定证据。

### 23.6 原样组早期记录

17:59 已独立核验并备份 model_5050.pt，其中包含本段 51 次更新、累计 5051 次更新，归一化样本数为 127893504、课程计数为 121224、Adam 计数为 101020。该保存点的哈希为 `471ecb69a25a503472a732723410e311b61b7385141735d66723f626a234e502`，见[模型核验](../../records/experiments/gait_repair_control_20261101T174853_s42/checkpoint-checks/model_5050.json)。它是中间恢复点，最终 250 次预算尚未验收。

18:08 又核验并备份 model_5100.pt，包含本段 101 次更新、累计 5101 次更新，归一化样本数为 132808704、课程计数为 122424、Adam 计数为 102020；哈希为 `9d93f7623e7927744f799d3a8914ce36f07444e8262e222f4b9c0fdc33d2b576`，见[后续保存点](../../records/experiments/gait_repair_control_20261101T174853_s42/checkpoint-checks/model_5100.json)。当时原样组仍在训练，目标约束组及最终五指令评估尚未开始。

17:49:29–17:59:33 保存的 1290 条实际观察记录中，左髋侧摆中位数为 −22.48°、右髋为 +22.50°，双脚横向间距有 643 条为负。原样组在这一阶段仍持续内收。下面展示全部样本；指令、权重和 episode 在训练中变化，不能把这些相关样本当成 1290 次独立试验，也不能用此图替代固定指令评估。数据、时间范围与绘图检查见[冻结快照](../../records/experiments/gait_repair_control_20261101T174853_s42/gait-diagnostics/20261101T095933874250Z/analysis.json)。

![早期姿态](../../records/experiments/gait_repair_control_20261101T174853_s42/gait-diagnostics/20261101T095933874250Z/leg-adduction.png)

横轴是从恢复点开始的控制步，上幅为躯干横向轴上的脚间距，下幅为实际髋侧摆角。虚线分别为零间距和仿真关节限位。图中突变包含训练复位，不将其解释为步态改善；原始值和超出限位的瞬时值均保留。两幅子图已分别检查文字、图例和坐标，没有发生裁切或标签重叠。

下一段录像从本次运行的相对控制步 1200 开始，完整解码为 200 帧、4 秒，原文件与复制件哈希一致。它仍属于原样组，当前没有加入目标惩罚。

<video controls preload="metadata" src="../../records/experiments/gait_repair_control_20261101T174853_s42/videos/rl-video-step-1200.mp4"></video>

### 23.7 渲染开销

本轮新增的图形检查发现，本机独立 EGL 查询返回 `Mesa / llvmpipe (LLVM 20.1.2, 256 bits)`。根据 [Mesa 官方说明](https://docs.mesa3d.org/drivers/llvmpipe.html)，LLVMpipe 是利用多个 CPU 核心的软件光栅化器。CUDA 训练可用和 OpenGL 图形加速可用需要分别验证；本次 EGL 结果不能证明原生窗口的具体渲染器已被直接查询。

随后从正在训练的同一个工作进程读取线程计数，两次短采样均发现 `llvmpipe-*` 线程占据大量 CPU 时间。可复查的第二次采样覆盖本段第 72–73 次更新：10.04 秒墙钟时间内，共测得 110.81 CPU 秒，其中 llvmpipe 线程为 97.31 CPU 秒，占 87.82%，平均相当于 9.70 个逻辑核持续工作。CPU 秒会跨线程累计，所以可以超过墙钟秒数；87.82% 也不是墙钟时间中的渲染比例，更不是已经测得的提速幅度。

[原始线程计数及图形查询](../../records/experiments/gait_repair_pilot_20261101T174853_s42/render-diagnostics/repeat-query.json)保存两次采样、进程启动标识、每秒时钟滴答数、逐线程差值和训练进度，可重新计算。该区间已结束控制步 1200 的录像，尚未进入下一次录像触发；持续运行的软件渲染线程成为后续效率优化的明确检查对象。原生窗口的渲染器字符串与关闭显示后的训练速度仍未直接验证，不能把线程占用全部等同于可消除的耗时。

此前观察器约 3% 的计时只覆盖状态复制、同步和记录，遗漏异步渲染线程，不能用来估计完整显示成本。这里的检查没有访问或操作原生窗口，没有加载机器人模型、推进仿真或修改现有工作进程。当前两组短试验继续使用冻结设置，避免中途改变比较条件。

复查时先通过 `ps` 确认仍在运行的工作进程，再执行以下命令，将 PID 和输出文件名替换成当次值。PID 可能随重启被重复使用，不能只照抄旧数字；脚本还会核对命令和进程启动标识。

```bash
cd /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
ps -eo pid,ppid,etime,args | grep 'gait_repair.py --worker'
$PY -B scripts/validation/probe_training_renderer.py \
  --worker-pid 469 --seconds 10 \
  --output records/experiments/gait_repair_pilot_20261101T174853_s42/render-diagnostics/manual-query.json
```

下一项效率工作先分开核验原生显示和录像所用的图形路径，再在固定训练规模下比较开销。[微软 WSLg 文档](https://github.com/microsoft/wslg#opengl-accelerated-rendering-in-wslg)说明，其 OpenGL 加速依赖 Mesa 的 D3D12 驱动与兼容的主机驱动；本轮没有更改驱动、全局环境变量、依赖或系统。原生 Ubuntu 比较路线继续保留，在实际测得两边的图形与训练性能之前不声称迁移会提速多少。

18:07 的后续隔离查询已找到本机可用的硬件路径。驱动清单确认 `d3d12_dri.so`、WSL 的 D3D12 库和 `/dev/dxg` 存在；当前训练进程没有设置强制软件渲染的变量。仅为新的查询进程设置 `GALLIUM_DRIVER=d3d12` 与 `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` 后，返回 `Microsoft Corporation / D3D12 (NVIDIA GeForce RTX 3080 Laptop GPU)`，OpenGL 为 4.6，退出码为 0。见[硬件路径查询](../../records/experiments/gait_repair_pilot_20261101T174853_s42/render-diagnostics/d3d12-query.json)、[进程范围记录](../../records/experiments/gait_repair_pilot_20261101T174853_s42/render-diagnostics/d3d12-query-process.json)和[驱动清单](../../records/experiments/gait_repair_pilot_20261101T174853_s42/render-diagnostics/driver-inventory.json)。

该结果只验证独立 EGL 上下文可以选择 RTX 3080。查询期间采样的 CPU 线程仍属于原来的训练工作进程，不能把同一个 JSON 中的约 90% llvmpipe CPU 占比解释为新 D3D12 上下文的性能。当前训练、原生窗口和两组协议都没有切换图形驱动；实际机器人渲染、录像、原生查看器退出，以及固定规模下的训练速度仍需在后续独立任务中验证。

要复查硬件路径，将前一条查询命令改为下面的形式，仍先确认实际工作进程 PID。变量只传给这条查询命令，不写入 `.bashrc` 或系统设置；输出使用单独文件，保留默认软件路径的结果。

```bash
GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
$PY -B scripts/validation/probe_training_renderer.py \
  --worker-pid 469 --seconds 10 \
  --output records/experiments/gait_repair_pilot_20261101T174853_s42/render-diagnostics/manual-d3d12-query.json
```

后续完整渲染验证入口已准备为 [validate_renderer.py](../../scripts/validation/validate_renderer.py) 与[可见终端启动脚本](../../scripts/validation/run_renderer_validation.sh)，条件固定在[图形验证配置](../../configs/experiments/renderer_validation.json)。它将依次检查默认 EGL、默认 GLFW、D3D12 EGL 和 D3D12 GLFW；每项使用官方完整网格和静态 STAND 姿态，旋转相机并输出 120 帧、30 fps、960×720 视频及三个原始渲染帧。画面标注为静态渲染验证，不是策略行走。GLFW 每次只创建一个新的诊断窗口，结束后关闭，不调整已有窗口。

18:41 仅完成源码、Shell、资产文件和实际执行条件检查，见[入口检查](../../records/reports/renderer-validation-entry-check.json)与[资产预检](../../records/reports/renderer-validation-preflight.json)。当前成对试验尚未完成时，入口已实际拒绝开始渲染，未导入 MuJoCo、未创建图形上下文或新实验目录。完整渲染与录像还没有运行，不能称为图形加速或训练提速已完成。

在 WSL 项目根目录可执行下面的只读预检。当前试验的训练、导出和固定评估全部结束后，再在同一个可见 VS Code 终端执行第二条命令；入口会取得项目计算锁，保持四项顺序执行。图形变量只作用于新子进程，不修改系统设置、已有工作进程或官方源码。

```bash
bash scripts/validation/run_renderer_validation.sh --check-only
# 当前成对试验及其固定评估完成后，在可见终端执行：
bash scripts/validation/run_renderer_validation.sh
```

这项验证检查完整机器人是否能通过两种图形接口正确出图、视频是否完整、资源是否正常释放。新 GLFW 诊断窗口与训练所用的被动查看器不是同一个实现，因此实际被动查看器和完整训练吞吐仍需随后独立测量；不能用静态渲染耗时推算训练提速。

### 23.8 学习记录

原样组于 18:31:16 正常结束，退出码为 0。结束后的冻结快照已交叉核对编号 5000–5249 的全部 250 次更新：终端日志、TensorBoard 与逐次计时记录一致，新转移数为 24576000，载入状态和保存源码哈希也一致，原工作进程已退出。证据见[完整日志核验](../../records/experiments/gait_repair_control_20261101T174853_s42/resume-verification/20261101T103146835987Z/check.json)。

![学习记录](../../records/experiments/gait_repair_control_20261101T174853_s42/resume-verification/20261101T103146835987Z/figures/continuation-progress.png)

横轴为累计完成更新数 5001–5250。三幅图依次展示训练回报、episode 长度和按采样加学习内部计时计算的吞吐；不额外平滑或删去录像期间的下降。图中文字与边界检查通过，并逐幅检查了实际输出。[绘图检查](../../records/experiments/gait_repair_control_20261101T174853_s42/resume-verification/20261101T103146835987Z/figures/plot-check.json)保存 CSV 和图片哈希。

需要注意回报曲线的起段含义：实际 [logger 源码](../../records/experiments/gait_repair_control_20261101T174853_s42/sources/logger.py.txt)对最近 100 个已结束 episode 的累计回报和长度求均值；新的运行会重新创建这些统计缓冲。较长 episode 尚未结束时不会进入这个均值，因此开头的上升包含记录窗口建立和 episode 完成时间的影响。图中回报由 0.51 到 88.89 不能单独证明新学习带来步态改善；本段起点已经是训练 5000 次的模型，姿态和固定评估仍要分别验收。

最终 `model_5249.pt` 的 SHA-256 为 `0b1e32b6abc425d899421ed0efa71bd77721af028153626a8957139ef3a42975`。模型实际包含本段 250 次更新、累计 5250 次，归一化样本数 147456000、课程计数 126000、Adam 计数 105000；76 个张量均为有限值，恢复副本哈希一致，见[最终模型核验](../../records/experiments/gait_repair_control_20261101T174853_s42/checkpoint-checks/model_5249.json)。五段训练录像分别从相对控制步 0、1200、2400、3600、4800 开始，均完整解码为 200 帧。模型完整性与媒体完整性均不代替最终步态评估。

要自行生成新的已核验曲线，在 WSL 项目根目录执行以下命令。第一步冻结当前完整记录，第二步读取该快照画图；两步都不训练或仿真。输出目录见第一步的 `evidence_directory`，图片在其 `figures/continuation-progress.png`。原快照保留，可回查本报告对应的数值。

```bash
cd /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
RUN=gait_repair_control_20261101T174853_s42
$PY -B scripts/validation/verify_repair_progress.py "$RUN" --require-finished
$PY -B scripts/validation/plot_continuation_progress.py "$RUN"
```

对正在运行的修改组检查完整前缀时，改用它的运行名称并去掉 `--require-finished`；该选项要求真实退出码为 0、全部 250 次记录完整且工作进程已退出。修改组增加了惩罚项，两组回报总值的含义不同。最终比较以同预算的五指令跟踪、存活和姿态为准；训练曲线用于了解学习过程，不用总回报较高替代修复判断。

### 23.9 并脚复查

用户再次观察到脚并拢后，先更新目标，再检查实际工作进程、运行指针、协议和保存遥测。18:18 运行中的进程参数为 `--arm control`，当前组没有 `action_target_envelope` 项；原样组按冻结预算提供同起点、同更新数的对照，随后才运行目标约束组。本次没有重启训练或改变两组条件。

冻结到 18:18:17 的 3949 条观察记录中，左髋侧摆中位数为 −22.47°，右髋为 +22.50°；足部参考点横向间距中位数为 −0.032 mm，1990 条记录出现左右横向次序反转。两脚碰撞几何的接触掩码仍不允许相互碰撞，也没有显式脚间接触对。因此，并脚是实际仿真状态中的异常，不能解释成单纯的镜头视角。这里的间距指模型足部参考点沿躯干横向轴的投影，不能当作两只脚外轮廓之间的净空；记录来自一个训练环境，不能当作独立试验次数。数值、完整输入与图片哈希保存在[本次姿态核查](../../records/experiments/gait_repair_control_20261101T174853_s42/gait-diagnostics/20261101T101817081017Z/analysis.json)。

![并脚姿态](../../records/experiments/gait_repair_control_20261101T174853_s42/gait-diagnostics/20261101T101817081017Z/leg-adduction.png)

图中髋角长期接近内侧限位，脚距反复接近或越过零。复位时短暂恢复开脚姿态不代表学会了正常步态。两幅子图已分别检查，没有文字重叠或裁切。

下面是原样组相对控制步 3600 开始的已保存录像，完整解码为 200 帧、50 fps、4 秒；原文件与复制件哈希一致，见[媒体核验](../../records/experiments/gait_repair_control_20261101T174853_s42/media-check/check.json)。它是保存片段，不是当前实时画面。

<video controls preload="metadata" src="../../records/experiments/gait_repair_control_20261101T174853_s42/videos/rl-video-step-3600.mp4"></video>

拟检验的修复让网络为过度的关节目标付出奖励代价，不直接把执行动作裁成一段好看的姿态。因此，修改组刚从同一模型开始时也可能并脚；必须实际检查目标越界是否减少、双脚是否保持合理次序，以及停止、前后行走和转向是否通过原标准。18:31 修改组已经开始，效果结论仍待最终评估，原长训练继续保持停止。

18:18 同时核验并备份了 `model_5150.pt`，包含本段 151 次更新、累计 5151 次更新，归一化计数 137723904、课程计数 123624、Adam 计数 103020。它只证明恢复点完整，不能作为并脚修复证据，详见[保存点核验](../../records/experiments/gait_repair_control_20261101T174853_s42/checkpoint-checks/model_5150.json)。

### 23.10 修改组起点

原样组结束后，任务 28 自动在同一终端启动 `gait_repair_target_envelope_20261101T174853_s42`。新工作进程已实际载入同一个 5000 次模型并开始更新。18:32 的[成对起点检查](../../records/experiments/gait_repair_pilot_20261101T174853_s42/matched-start-check.json)确认：恢复计划、载入后的计数与学习率修正一致，actor/critic 初始观测和首个随机动作的哈希一致；移除新增的 `action_target_envelope` 项及运行名称后，两组完整配置一致，函数名称和冻结源码也一致。原始输入和核验脚本副本保存在该批次的 `matched-start-inputs/`。

修改组首个 `model_5000.pt` 已独立加载核验：包含 1 次新更新，归一化计数 122978304、课程计数 120024、Adam 计数 100020，actor 和 critic 的参数均已发生更新。模型哈希为 `682655328ede346ee16fb4585f343f0da70c8798df73b968cb50cad95bcf772c`，见[修改组保存点](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/checkpoint-checks/model_5000.json)。

18:40 又核验并备份包含 51 次新更新的 `model_5050.pt`：归一化计数 127893504、课程计数 121224、Adam 计数 101020，哈希为 `cbf25b03138e7a307b83e295732841d1aeab38c3bd9eb78b30398525f656fa29`，见[修改组中间保存点](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/checkpoint-checks/model_5050.json)。

修改组起始至 18:40:43 的 1305 条训练观察记录仍显示持续内收：左、右髋侧摆中位数分别为 −22.49°、+22.46°，足部参考点横向间距中位数为 −0.755 mm，749 条记录出现横向次序反转。这一早期阶段尚未消除并脚。记录来自一个环境且权重、指令与复位状态不断变化，不能把这些样本当作独立 episode，也不能直接与原样组的随机训练片段比较修复优劣。原始输入、数值和图像哈希见[修改组早期姿态](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/gait-diagnostics/20261101T104043385067Z/analysis.json)。

![修改组姿态](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/gait-diagnostics/20261101T104043385067Z/leg-adduction.png)

前四次更新的终端日志、TensorBoard 和计时记录已交叉核验；新增项 `Episode_Reward/action_target_envelope` 的记录分别约为 −0.0104、−0.0453、−0.0815、−0.1228，两种日志来源一致，见[惩罚启用检查](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/reward-activation.json)。这说明惩罚参与了实际运行；这些值经过 episode 统计，起段还受复位和统计缓冲建立影响，不能据此判断目标越界变好或变坏，更不能据此宣布并脚已修复。

下面是修改组从相对控制步 0 开始的首段真实训练录像，已核对原文件与复制件一致，并完整解码为 200 帧、50 fps、4 秒，见[修改组媒体核验](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/media-check/check.json)。片段对应刚从父模型开始学习的阶段，不是最终模型回放。

<video controls preload="metadata" src="../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/videos/rl-video-step-0.mp4"></video>

下一段从相对控制步 1200 开始，同样完整解码为 200 帧、50 fps、4 秒。它覆盖修改组约第 50 次更新之后的训练片段，原文件和复制件的哈希一致；与上方姿态曲线一起保留，最终效果仍以固定模型评估为准。

<video controls preload="metadata" src="../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/videos/rl-video-step-1200.mp4"></video>

自行复查两组起点可执行以下只读命令。脚本读取实际记录，不启动训练，也不改变两组模型；两组都产生载入、首个动作和更新记录后才能执行。

```bash
cd /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
$PY -B scripts/validation/verify_repair_progress.py \
  --compare-batch-starts records/experiments/gait_repair_pilot_20261101T174853_s42
```

接下来保持修改组 250 次的原定预算。两组最终 checkpoint 将由同一任务顺序导出并执行五指令评估，最后才比较目标越界、脚间距、内收限位、存活和运动跟踪；当前不增加新的奖励、碰撞改动或训练预算。

18:52 已在 CPU 上独立加载并备份修改组 `model_5150.pt`，包含本段 151 次、累计 5151 次更新。归一化计数 137723904、课程计数 123624、Adam 计数 103020，76 个张量均有限，actor 和 critic 参数均已更新，见[保存点检查](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/checkpoint-checks/model_5150.json)。本次检查没有初始化 CUDA、运行仿真或更新权重，也不构成步态通过证据。

### 23.11 修改组训练完成

修改组于 18:59:54 正常退出，最终 `model_5249.pt` 包含本段 250 次更新、累计 5250 次更新；归一化计数 147456000、课程计数 126000、Adam 计数 105000。76 个张量均有限，模型已独立加载并保存恢复副本，SHA-256 为 `b2c4681cf507573e339bf1b4475436dff804e5a05aa6ccc4e339437bf77efbf3`。[最终保存点](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/checkpoint-checks/model_5249.json)与[完整训练记录](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/resume-verification/20261101T110009772104Z/check.json)分别核验了模型状态及 250 条终端、TensorBoard、计时记录。工作进程已退出，没有补跑学习。

![修改组训练曲线](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/resume-verification/20261101T110009772104Z/figures/continuation-progress.png)

曲线保留原始记录，没有平滑。末次日志回报为 74.67，episode 长度为 840.46 步；开头的上升还受新建 episode 统计缓冲影响。修改组新增惩罚，总回报不能与原样组直接等同解释，episode 长度也不能代替运动跟踪验收。三个子图已分别检查，没有文字重叠或裁切。

曲线中约累计 5094 次更新后的吞吐跃升，需要结合显示状态解释：训练日志第 5447 行记录 `LIVE_VIEW_CLOSED`，其后打印迭代 5093。原样组没有对应关闭事件，实时观察记录为 5928 条；修改组只有 2233 条，最后一条在 18:47:00、恢复后第 2245 个控制步。关闭原因没有记录，不能推断是谁关闭了窗口。训练完成时统一的 `observer-finished` 状态不能掩盖这次提前关闭，详见[显示覆盖核查](../../records/experiments/gait_repair_pilot_20261101T174853_s42/observer-coverage.json)。因此两组总耗时不构成显示条件相同的性能对照，本次没有把修改组较快结束解释成奖励修改提速。

下面的姿态图只覆盖查看器关闭前的记录，不能写成全部 250 次更新的姿态表现。该段髋侧摆中位数为左 −22.50°、右 +22.46°，双脚参考点横向间距中位数为 −1.142 mm，1353 条记录出现横向次序反转。它说明早期仍有并脚，最终模型是否改善必须检查完整固定评估。[姿态快照](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/gait-diagnostics/20261101T110019873269Z/analysis.json)保留输入、时间范围和相关限制。

![修改组观察记录](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/gait-diagnostics/20261101T110019873269Z/leg-adduction.png)

定期录像独立于原生查看器继续工作。五段录像均完整解码为 200 帧、50 fps、4 秒，且与 Linux 原文件哈希一致。[录像核验](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/media-check/check.json)列出全部文件；第 23.10 节已有前两段，下列链接提供中段，下面嵌入的是控制步 4800 开始的后段保存片段。

- [控制步 2400](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/videos/rl-video-step-2400.mp4)
- [控制步 3600](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/videos/rl-video-step-3600.mp4)

<video controls preload="metadata" src="../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/videos/rl-video-step-4800.mp4"></video>

这段视频已检查首末帧，但仍是训练片段，不是最终固定策略评估。当前任务随后自动导出两组最终模型，先做 1000 帧 ONNX 一致性验证，再以原五指令各 32 个首个 episode、20 秒进行评估。原长训练的异常门限保持关闭。

### 23.12 固定评估结果

两组各 250 次学习、最终导出和五指令固定评估均已完成。两组最终模型均通过 1000 帧 PyTorch/ONNX 一致性检查，最大动作差分别为 6.20×10⁻⁶ 和 1.91×10⁻⁶，CPU 独立重放也通过。来源分别见[原样组导出](../../records/experiments/gait_repair_control_20261101T174853_s42/export-verification/check.json)与[修改组导出](../../records/experiments/gait_repair_target_envelope_20261101T174853_s42/export-verification/check.json)。

修复试验没有通过：每组五条指令各 32 个首个 episode、20 秒，160 个 episode 均存活，但两组都只有停止满足原数值标准，全部指令均未通过姿态标准。前进条件的动作目标超额量均值从 19.77 rad 降至 2.18 rad，说明新增惩罚确实改变了请求动作；双髋仍长期顶向内侧，脚部位点仍交叉，前进平面误差还从 0.1032 增至 0.1172 m/s。因此，降低请求越界量没有在本次 250 次更新内转化为合格步态，不据此继续追加长训练。

| 指令 | 原样组平面误差（m/s） | 修改组平面误差（m/s） | 原样组转向误差（rad/s） | 修改组转向误差（rad/s） |
| --- | ---: | ---: | ---: | ---: |
| 停止 | 0.0024 | 0.0022 | 0.0195 | 0.0220 |
| 前进 | 0.1032 | 0.1172 | 0.1986 | 0.1333 |
| 后退 | 0.0967 | 0.0996 | 0.0578 | 0.0423 |
| 左转 | 0.0022 | 0.0023 | 0.4087 | 0.4067 |
| 右转 | 0.0045 | 0.0031 | 0.4115 | 0.4158 |

平面误差为每帧二维速度误差的 L2 范数，先在 episode 内平均，再跨 episode 平均；不能只用下图的前向速度偏差代替。两组均使用相同的原五指令标准，没有为修改组放宽阈值。

| 指令 | 原样组脚间距中位数（mm） | 修改组脚间距中位数（mm） | 出现脚部位点交叉的 episode（每组） |
| --- | ---: | ---: | ---: |
| 停止 | -0.36 | -5.36 | 32/32、32/32 |
| 前进 | 2.10 | -2.99 | 32/32、32/32 |
| 后退 | -1.71 | -5.45 | 32/32、32/32 |
| 左转 | -2.38 | -7.71 | 32/32、32/32 |
| 右转 | 0.86 | -2.94 | 32/32、32/32 |

脚间距为左右足部位点差在机体左向轴上的投影，负值表示位点左右次序交叉，并不等于测得接触力。每个 episode 只要有一帧为负即计入交叉数；本次各条件两组均为 32/32。双髋同时内收超过 20° 的帧占比，原样组约 99.3%–99.6%，修改组约 99.2%，姿态问题仍在。

![原样组指令响应](../../records/evaluations/custom_onnx_reference_20261101T190142_s1001/figures/command-response.png)

![修改组指令响应](../../records/evaluations/custom_onnx_reference_20261101T191205_s1001/figures/command-response.png)

细线为各 episode，粗线为均值，虚线为指令；保留启动瞬态，未平滑。两组全部图形面板已逐一查看，十段视频完整解码通过；独立证据见[成对核验](../../records/experiments/gait_repair_pilot_20261101T174853_s42/verification.json)、[原样组评估](../../records/evaluations/custom_onnx_reference_20261101T190142_s1001/verification.json)与[修改组评估](../../records/evaluations/custom_onnx_reference_20261101T191205_s1001/verification.json)。这里的 `status=passed` 表示记录和计算通过核验，步态验收结果是两组均为 `false`。

以下两段是最终固定策略的前进评估，头部和身体姿态指令均为零。它们是保存录像，不是正在学习的实时画面；可与第 23.13 节的随机头部指令训练片段区分。

<video controls preload="metadata" src="../../records/evaluations/custom_onnx_reference_20261101T190142_s1001/videos/forward_env0.mp4"></video>

<video controls preload="metadata" src="../../records/evaluations/custom_onnx_reference_20261101T191205_s1001/videos/forward_env0.mp4"></video>

### 23.13 头部动作诊断

用户反馈“头变成平衡杆了”时，VS Code 显示的是原样组 `rl-video-step-4800.mp4`，不是修改组或正在进行的评估。已从窗口路径核对，并从同一控制步区间提取 201 条实际关节记录：颈部俯仰约 −76.34°～11.42°，头部俯仰约 34.11°～76.88°，头部偏航约 24.33°～92.12°。这里使用关节绝对角，不是相对 home 的误差；遥测与录像属于同一控制步区间，尚未逐帧对齐。

![训练头部姿态](../../records/experiments/gait_repair_pilot_20261101T174853_s42/head-motion/training-head-motion.png)

官方训练同时学习四维头部姿态指令，约每 2–5 秒重新采样；从第 2000 次更新对应的课程阶段起，颈部和头部俯仰的相对 home 指令范围为 ±63.03°，偏航为 ±80.21°，侧摆为 ±17.76°。本组日志已经记录该课程达到最终范围。因此，训练视频中的转头混合了主动姿态指令和策略响应，不能直接全部归因于平衡补偿。同期旧遥测仅记录了行走指令，没有保存四维头部目标，无法补造逐帧头部跟踪误差。源定义见[官方头部任务配置](../../official_resources/xgoduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)。

范围核对还发现一个实际不一致：颈部 home 约为 20.00°，加上最终姿态指令后，目标上界约为 83.03°；编译模型中颈关节上限为 60.00°，相差约 23.03°。这一比较针对执行器关节及配置端点，不能替代实际连杆、被动间隙和控制动态分析，但它表明不能仅按配置注释假定全部头部指令均可达。录像区间颈部角度还低于配置目标下界，随机指令本身不足以解释全部偏离，必须继续检查跟踪与动力学。

为排除主动转头要求，另用两组已完成固定评估与归档官方参考比较。三者环境配置哈希相同，每个条件的头部和身体指令均检查为零；以下为前进条件去掉第一个完整秒后，各 episode 头部角度围绕自身均值波动的 RMS，再对 32 个 episode 取中位数。完整 0–20 秒和其余四条指令也保存在分析文件中，没有用去掉启动段替代完整结果。

| 关节 | 归档官方模型（°） | 原样组（°） | 修改组（°） |
| --- | ---: | ---: | ---: |
| 颈部俯仰 | 0.37 | 4.25 | 3.87 |
| 头部俯仰 | 0.34 | 1.03 | 2.23 |
| 头部偏航 | 0.36 | 3.75 | 2.22 |
| 头部侧摆 | 0.34 | 2.83 | 1.03 |

表中统一使用策略输入中的角度观测，以便与早期只保存观测的官方记录比较；它包括原配置的观测噪声。原样组另有无观测噪声的实测关节速度：前进去掉首秒后，颈部绝对角速度的 95 分位为 127.95°/秒，修改组为 119.02°/秒；此条件下两组头部关节均未贴近硬限位，也没有请求超出硬限位的头部目标。说明中立头部指令下仍存在明显动态摆动，但这些量没有证明摆动是维持平衡所必需。官方参考本身也未通过行走跟踪标准，不能把其较小摆动直接当作成功步态的最优阈值。

![评估录像](../../records/experiments/gait_repair_pilot_20261101T174853_s42/head-motion/vscode-fixed-head-evaluation.png)

已在原 VS Code 窗口实际打开并播放原样组固定前进评估，截图显示录像推进至约 9.32 秒；窗口仍为 1707×1019，原任务终端保留。该画面标明 ONNX 推理与仿真时间，不称作实时训练。诊断数据与输入哈希见[头部分析](../../records/experiments/gait_repair_pilot_20261101T174853_s42/head-motion/analysis.json)，每个 episode 的统计见[数据表](../../records/experiments/gait_repair_pilot_20261101T174853_s42/head-motion/episodes.csv)，界面记录见[显示核对](../../records/experiments/gait_repair_pilot_20261101T174853_s42/head-motion/display-observation.json)。

后续修复先对齐头部指令与实际关节范围。中立头部目标下的行走检查应独立保留；头部动作平滑和姿态质量与速度跟踪一起验收，不能只看不倒。官方当前已经有头部姿态奖励、慢变化偏置惩罚和全动作平滑惩罚；是否修改指令课程或增加约束，需要作为新的独立试验记录，不能在已结束的成对试验中补改条件。当前没有实施头部控制修复，也没有把头部关节锁死后宣称问题解决。

已为后续运行补充观察器记录：按颈部俯仰、头部俯仰、偏航、侧摆次序，保存姿态指令、home 加指令后的目标、执行器关节实际角度、角速度与跟踪误差；画面叠加目标和实际角度。误差针对执行器关节，不包含被动间隙连杆的独立误差。查看器关闭时间和对应控制步现在会保留到收尾，遥测仍在窗口关闭后停止，不能据此声称覆盖全程。

[观察器代码](../../scripts/training/live_observer.py)已通过[CPU 隔离检查](../../records/reports/live-view-entry-check/check.json)：使用真实编译模型及明确标记的模拟传感器值，核对目标换算、关节次序、显示限频、关闭事件保存，并确认源状态、源模型和随机数状态未改变。该检查没有启动 GUI、CUDA、物理步或学习；随后第 25 节的六项真实训练短测核验了新增记录、完整采样覆盖和查看器存活状态。完成试验的原始观察器副本保留，后续源码变化单列在[同起点重查](../../records/experiments/gait_repair_pilot_20261101T174853_s42/matched-start-recheck.json)，未覆盖原核验记录。

在项目根目录 WSL 终端可复算本节；脚本只读取保存轨迹，不启动仿真或训练：

```bash
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
$PY -B scripts/validation/audit_head_motion.py records/experiments/gait_repair_pilot_20261101T174853_s42
```

## 24. 官方训练路线

### 24.1 可用方式

2026-11-01 核对官方训练仓库、运行仓库、发布列表及本机锁定源码后，没有发现另一套已提供给 XGO-Duck 的低计算量训练算法。官方主线仍为 CUDA 上的 MuJoCo Warp + PPO；mjlab 本身采用较少依赖的设计。官方仓库另外提供云 GPU 提交、训练规模调整、模型回放等入口，各自节省的资源不同。[XGO-Duck 训练说明](https://github.com/LuwuDynamics/xgoduck_rl)、[mjlab 说明](https://github.com/mujocolab/mjlab)。

| 方式 | 官方能力 | 对当前项目的意义 |
| --- | --- | --- |
| 本机 GPU 训练 | 官方示例使用 4096 环境；本机锁定入口默认不连续渲染，录像可选 | 优先修复额外显示开销，保留现有训练及评估协议 |
| 少环境短运行 | 可调整并行环境数和更新数 | 适合检查接口、奖励和报错；少占显存不等于同样样本更快完成 |
| checkpoint 续训 | 框架可加载 `.pt` 继续 PPO | 可使用我们的完整保存点；尚未找到公开的官方训练 checkpoint |
| 官方 ONNX 回放 | 运行仓库提供部署模型 | 适合低成本检查官方效果；不能作为包含 critic、Adam 等状态的 PPO 恢复文件直接加载 |
| HF Jobs 云训练 | 训练命令增加 `--hf-jobs`，可选择云 GPU，并流式显示日志、保存模型 | 减轻本机负担，算法与总计算需求仍在；需要账号、额度及代码上传 |
| 原生 Ubuntu | 上游将 Linux + NVIDIA GPU 作为主要训练平台 | 是本机双系统的可比较路线，当前尚无同条件提速数据 |

HF Jobs 入口确实包含在固定官方源码中，不是本项目新增的云平台方案。其文档示例使用 MicroDuck 任务名，选择 XGO-Duck 时应指定本项目任务；提交程序把任务和训练参数转交同一训练入口。云端作业与本机进程生命周期不同，终端退出不必然取消作业。当前没有登录云账号、上传项目或提交付费任务。[官方 HF Jobs 说明](https://github.com/LuwuDynamics/xgoduck_rl/blob/fa621fc77ad9a05d69da13270e86118bdd3603fe/scripts/hf/README.md)、[平台作业说明](https://huggingface.co/docs/huggingface_hub/guides/jobs)。

上游当前文档包含 CPU 执行选项，但 XGO-Duck 官方安装要求 CUDA GPU，没有提供“CPU 训练更快或更省总资源”的证据。当前在线 mjlab 文档与本机锁定依赖的参数形式可能不同，实际命令以锁定源码和本机 `--help` 为准，不为使用新参数直接升级依赖。[训练参数](https://mujocolab.github.io/mjlab/main/source/training/rsl_rl.html)、[平台与显示说明](https://mujocolab.github.io/mjlab/v1.6.0/source/faq.html)。

### 24.2 本机选择

图形加速现已有实测依据。此前独立 EGL 查询返回 llvmpipe；同一训练进程的软件渲染线程在约 10 秒采样中占其 CPU 时间约 88%。这个占比不是总训练耗时占比，不能推导出“可提速 88%”。完整机器人静态渲染及录像已通过 GPU 验证，第 25.4 节另外测得连续显示条件下约 1.53 倍的训练吞吐。用户要求持续看到动作，因此保留实时显示要求；周期录像与独立 checkpoint 回放作为补充，不能冒充当前训练环境的实时状态。

少环境更适合快速排错。本次重新从既有规模测量的样本数和耗时计算：4096 相对 1024 环境的吞吐比分别为 2.44664、2.43220，几何平均 2.43941；最高观测显存占比约 28.35%。这支持当前本机正式训练继续使用 4096 环境，而不是为“轻量”简单降到 1024。该结论仅针对本机当时配置，测量没有启用当前连续原生观察器；不代表更大批量必然更易收敛，也不代表原生 Ubuntu 的结果。[规模测量](../../records/benchmarks/scale_20261101T154057_s42/verification.json)、[本次重算与来源](../../records/reports/official-training-options-evidence.json)。

还需区分默认预算与阶段验收：官方平地行走配置继承 `max_iterations=50_000`，此前完成的 5000 次是本项目的阶段预算。5000 次未通过步态标准不能证明官方完整训练必定失败；反过来，默认预算也不能保证继续到 50000 次就能消除并脚。原样组与目标约束组各 250 次的试验现已完成，结果见第 23.12 节；仅增加目标越界惩罚没有通过步态验收，下一段训练需重新明确修复因素。[官方任务注册](https://github.com/LuwuDynamics/xgoduck_rl/blob/fa621fc77ad9a05d69da13270e86118bdd3603fe/src/mjlab_microduck/tasks/__init__.py)、[官方行走配置](https://github.com/LuwuDynamics/xgoduck_rl/blob/fa621fc77ad9a05d69da13270e86118bdd3603fe/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)。

### 24.3 预训练资源

本次 GitHub API 查询完整列出训练仓库和运行仓库的默认分支文件，两棵树均未截断，发布列表均为空；公开 Hugging Face 名称检索 `xgoduck` 返回零项。在此范围内确认了部署用 ONNX，没有找到 `.pt`、`.pth` 或 `.ckpt`。这不排除私有模型或其他名称下的资源。若后续取得官方训练 checkpoint，还需配套环境、观测、动作、归一化和训练状态；仅把 ONNX 改成 `.pt` 后缀不能续训。来源、查询时间、全部文件清单和边界保存在[官方资源核对](../../records/reports/official-training-options-evidence.json)。

训练仓库当前提交与归档一致；运行仓库当前提交为 `b401d85d20bb5bfd0237242a8a5ab6847fc962a2`，行走 ONNX 的 Git blob 为 `b3a5ca66ec5d1c62aaba2b080562f6c6a3ee8154`，与原归档文件的 `79d89116d4deb0ce49d82a7644e857663db93198` 不同。官方提交也同时调整了运行参数。初次检索只核对元数据，随后已按第 24.4 节保存独立副本并完成 CPU 接口检查，固定评估结果见第 24.5 节；归档没有覆盖。[官方变更](https://github.com/LuwuDynamics/xgoduck_runtime_arduino/commit/b401d85d20bb5bfd0237242a8a5ab6847fc962a2)。

各项进展如下，已完成成对试验的条件和原始记录保留：

1. 两组最终模型导出与原五指令评估已完成；目标越界惩罚降低了请求超额，未通过姿态及运动跟踪验收。
2. 完整静态渲染验证和实际连续显示耗时比较均已在同一可见 VS Code 终端完成；GPU 提升了吞吐，流畅显示仍待改善。
3. 当前官方行走模型的来源、接口及固定评估均已完成并保留录像；原参考保留。
4. 如果 WSL 图形加速验证仍不能满足速度和画面需求，再在原生 Ubuntu 做同预算比较。云 GPU 路线保留为需要外部算力时的选择。

本节完成的是来源核对与路线选择依据；没有声称已经提速、获得官方可续训权重或解决并脚。

### 24.4 当前官方模型接口

已把上述固定提交的行走 ONNX、数值适配程序和相关运行说明共 9 个文件保存到 `workspace/official-reference/b401d85d20bb5bfd0237242a8a5ab6847fc962a2/`。每个文件的 Git blob 均与官方 API 清单一致，同时保存 SHA-256；没有覆盖 `official_resources/`，也没有连接机器人。模型 SHA-256 为 `5eaae52c92df20139a779b4656e49ab824bb501fdd219e1241bcda91386fd75a`，详见[来源清单](../../records/reports/official-reference-refresh/source-manifest.json)。

| 项目 | 原归档 | 当前官方副本 |
| --- | --- | --- |
| 策略接口 | 61 个输入、14 个动作 | 尺寸和关节次序相同 |
| 髋俯仰、踝关节模型 home | 约 ±24.007° | 约 ±22.002°，运行默认值为 ±22° |
| 右髋偏航 ID 20 的滤波 | 使用头部系数 0.45 | 使用腿部系数 0.35 |
| 网络输入归一化 | 已包含在 ONNX 中 | 仍包含在 ONNX 中 |
| 固件启动及使能的临时增益 | 旧代码设置运行增益 | 当前代码仅为嘴部设置固定临时增益，保留腿及头部已有值 |

表中增益结论来自固件源码检查，尚未读取真机寄存器；模型 metadata 中的刚度字段也不能当作舵机实际增益。当前模型的 home 与 runtime 默认值最大差异约 0.286°，对应膝关节的小偏置，不能把“更接近”写成所有关节完全一致。

CPU 检查读取已完成基线评估中五条指令的 640 个保存观测，分别用 runtime 默认 home 和模型 home 检查，共 1280 次条件回放。输入重建最大误差为 4.77×10⁻⁷，未滤波动作历史误差为 7.15×10⁻⁷，滤波输出误差为 5.96×10⁻⁷，舵机目标换算误差为 3.81×10⁻⁵°。ONNX 结构检查、源文件哈希、按物理 ID 分配的滤波系数及嘴部排除均通过，推理只使用 `CPUExecutionProvider`。完整数据见[接口核验](../../records/reports/official-reference-refresh/interface-audit.json)。

这些输入是保存仿真观测重建的反馈，包含原来的噪声和延迟；回放强制使用原动作历史，用于隔离检查软件转换。它没有重新推进仿真，不能证明速度跟踪、站稳或真实传感器正确。模型 metadata 中虽出现训练文件路径，但缺少完整训练配置、seed 与恢复记录，不能仅按文件名推断训练预算或奖励参数。

自行复核时，在项目根目录的 WSL 终端执行以下 CPU 检查即可，不会启动训练或仿真：

```bash
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
$PY -B scripts/validation/audit_current_official_reference.py
```

首次检查选用的旧参考记录没有保存报告哈希，程序因此停止。现使用同时绑定报告和轨迹哈希的已完成基线记录，没有放宽检查或重跑仿真；过程保留在[检查记录](../../records/reports/official-reference-refresh/interface-audit-attempts.jsonl)。随后按同一五指令标准完成独立仿真与录像，结果如下。

### 24.5 当前官方模型效果

任务 30 已在原可见终端 3 完成，运行目录为 `custom_onnx_reference_20261101T195044_s1001`。目录前缀来自通用评估入口；这里实际评估的是第 24.4 节固定来源的官方模型，不是我们的自训模型。仍使用 seed 1001、五条指令各 32 个首个 episode、20 秒，模型自身 home 和未滤波动作；没有模拟完整 UNO Q 控制环。

160 个 episode 全部存活、没有跌倒，但仍只有停止满足原数值标准。当前模型的平均前进速度为 0.1019 m/s，旧参考为 0.0051 m/s，说明前进响应已有明显变化；当前后退平均速度约 −0.0305 m/s，左右转平均角速度约 +0.0152、−0.0157 rad/s，仍低于各自指令。数值判定使用原协议的逐时刻平面速度误差与转向误差，没有因为换用官方模型而调整阈值。

| 指令 | 指令值 | 当前平均响应 | 旧参考误差 | 当前误差 | 当前判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| 停止 | 0 m/s | 0.0002 m/s | 0.0012 m/s | 0.0018 m/s | 通过 |
| 前进 | 0.15 m/s | 0.1019 m/s | 0.1488 m/s | 0.1269 m/s | 未通过 |
| 后退 | −0.10 m/s | −0.0305 m/s | 0.1000 m/s | 0.1045 m/s | 未通过 |
| 左转 | 0.40 rad/s | 0.0152 rad/s | 0.3982 rad/s | 0.3868 rad/s | 未通过 |
| 右转 | −0.40 rad/s | −0.0157 rad/s | 0.3748 rad/s | 0.3858 rad/s | 未通过 |

前三行“响应”为前向速度均值，“误差”为二维逐帧 L2 误差；后两行为偏航角速度均值及其绝对误差。前进虽然平均速度更接近指令，横向速度和时间波动仍进入平面误差，所以不能用 `0.15−0.1019` 代替 0.1269 m/s。环境 YAML、指令、seed、统计定义与原参考一致，评估器后来增加的姿态记录和诊断入口另有源码哈希；模型训练历史未知，这不是受控的奖励消融。详见[同条件比较](../../records/evaluations/custom_onnx_reference_20261101T195044_s1001/reference-comparison/comparison.json)。

![官方指令响应](../../records/evaluations/custom_onnx_reference_20261101T195044_s1001/figures/command-response.png)

腿部姿态与自训模型有区别：当前官方模型前进时脚间距中位数为 54.12 mm、最小 40.77 mm，32/32 个 episode 没有脚部位点交叉，也没有双髋同时内收超过 20° 的帧。其余四条指令同样没有这些现象。这为并脚修复提供了有效参照，但没有消除上面的速度跟踪失败。[姿态数据](../../records/reports/saved-joint-posture/custom_onnx_reference_20261101T195044_s1001/analysis.json)。

中立头部指令下，当前官方模型前进时去掉首秒后的颈部摆动 RMS 为 0.81°，自训原样组为 4.25°；颈部实际绝对角速度的 95 分位分别为 24.16°/秒和 127.95°/秒。头部其他轴并非都更平稳：当前头部侧摆 RMS 为 3.46°，原样组为 2.83°。这些 RMS 均按第 23.13 节的观测口径统计，实际速度单独使用关节记录。完整与去首秒两种时间窗、四个头部关节、五条指令均保存在[扩展头部比较](../../records/evaluations/custom_onnx_reference_20261101T195044_s1001/head-motion/analysis.json)；不能把较小的颈部摆动推广成所有姿态问题已经解决。

<video controls preload="metadata" src="../../records/evaluations/custom_onnx_reference_20261101T195044_s1001/videos/forward_env0.mp4"></video>

![官方录像回放](../../records/evaluations/custom_onnx_reference_20261101T195044_s1001/vscode-official-forward.png)

已在原 VS Code 窗口播放至录像末尾，截图与路径记录见[显示核验](../../records/evaluations/custom_onnx_reference_20261101T195044_s1001/display-observation.json)。这是已保存的固定策略仿真，当前没有学习进程。窗口仍为 1707×1019，共享终端 3 保留。五段录像均完整解码为 501 帧、25 fps；评估只计算 1000 个控制步，录像额外包含初始画面，所以媒体时长为 20.04 秒。两个曲线面板已逐一查看，全部统计已从保存轨迹独立复算，见[评估核验](../../records/evaluations/custom_onnx_reference_20261101T195044_s1001/verification.json)。

自行复现时，在现有 VS Code 中运行 `XGO-Duck: 30 当前官方模型固定评估（不训练）`，等待任务结束，再对新运行目录执行 `scripts/validation/verify_fixed_evaluation.py`。每次会新建评估目录，不覆盖原结果。只复查本次数据时，在项目根目录的 WSL 终端执行：

```bash
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
RUN=records/evaluations/custom_onnx_reference_20261101T195044_s1001
$PY -B scripts/validation/verify_fixed_evaluation.py "$RUN"
$PY -B scripts/validation/summarize_saved_joint_posture.py "${RUN##*/}"
```

当前成果是补齐了官方效果参照、并脚和头部量化证据。自训策略仍需修复，尚未恢复长训练；下一步先测连续观察器的真实开销，再分别验证可达的头部目标和行走姿态约束，保留原模型及回退记录。

## 25. 图形加速验证

### 25.1 完整模型渲染

任务 29 已在原 VS Code 可见终端 3 顺序完成四项检查。EGL 和 GLFW 默认均使用 llvmpipe；只在对应子进程中设置 D3D12/NVIDIA 后，两条路径均使用 RTX 3080 Laptop。每项使用包含 15 个网格的完整机器人、960×720 分辨率，保存 120 帧、30 fps 的四秒录像及首、中、末三张图。这里固定机器人站姿并移动相机，没有推进物理或策略，也没有训练。

| 路径 | 实际渲染设备 | 录像帧数 | 结果 |
| --- | --- | ---: | --- |
| EGL 默认 | llvmpipe，CPU 软件渲染 | 120 | 通过 |
| GLFW 默认 | llvmpipe，CPU 软件渲染 | 120 | 通过 |
| EGL 指定 D3D12 | RTX 3080 Laptop | 120 | 通过 |
| GLFW 指定 D3D12 | RTX 3080 Laptop | 120 | 通过 |

四段视频均完整解码，12 张图的尺寸、非空像素和哈希均检查通过；四项中间帧逐一查看，机器人网格、头脚和地面可见，标注可读。以下是 GLFW 的 GPU 渲染结果，图和视频由此次新诊断进程直接输出，不是对已有 MuJoCo 桌面窗口的截图。

![GPU 渲染](../../records/benchmarks/renderer_20261101T194141/glfw_d3d12/frame-060.png)

<video controls preload="metadata" src="../../records/benchmarks/renderer_20261101T194141/glfw_d3d12/static-render-validation.mp4"></video>

证据见[执行记录](../../records/benchmarks/renderer_20261101T194141/batch.json)与[独立核验](../../records/benchmarks/renderer_20261101T194141/verification.json)。渲染器报告 Mesa 25.2.8，MuJoCo 为锁定环境的 3.10.0；没有安装或升级依赖，没有设置全局图形环境变量。

### 25.2 性能结论与复现

这一步证明本机 WSL 能用 GPU 完整渲染机器人。四项总时长约 9.56～10.29 秒，包含初始化、渲染、编码和解码；短静态测试中这些开销混合，不能拿它们推导 PPO 或实时查看器的加速比例。后续按相同训练规模、初始状态、更新数和录像完成了连续显示比较，结果见第 25.4 节。当前没有把默认显示路径改为全局 D3D12，也没有重新启动原长训练。

在现有 VS Code 选择 `Terminal → Run Task… → XGO-Duck: 29 完整 GPU 渲染验证（不训练）` 可复现。它复用共享任务终端，顺序创建并关闭各自的诊断图形上下文，记录实际设备名称、退出码、源代码和输出哈希。不要同时运行训练或其他仿真。协议为[渲染检查配置](../../configs/experiments/renderer_validation.json)，入口为[执行脚本](../../scripts/validation/run_renderer_validation.sh)；工作进程仅设置自身的图形环境，退出后不影响后续任务。

### 25.3 连续显示短测

为同时解决训练慢和动作画面卡顿，新增三种显示条件：默认驱动、4 Hz 上限；D3D12、4 Hz 上限；D3D12、15 Hz 上限。先按此顺序执行，再反向执行，共六项。第一对只比较驱动，GPU 下两项只比较显示上限，避免把两个变化混成一个提速因素。15 Hz 是允许的最高状态采样频率，不保证训练能产生这么多新状态，也不代表显示器刷新率。

每项从同一个累计 5000 次 checkpoint、Adam、归一化及课程计数恢复，seed 42、4096 环境、每次更新 24 个控制步，各运行 40 次更新。前 20 次热身，后 20 次全部计时，保留尖峰。奖励、物理、精度和录像配置不变；初始 200 帧录像位于热身区间，正式计时没有周期录像，因此结果适用于这一段连续观察负载，不能覆盖所有录像开销。每项生成的模型仅用于诊断和核验，不用于步态交付，也没有解除原长训练的异常限制。

入口为 [benchmark_observer.py](../../scripts/training/benchmark_observer.py)，协议为[连续显示配置](../../configs/experiments/observer_performance.json)。默认执行仅做 CPU 预检，已检查真实保存点、恢复计数、完整配置差异、源文件及任务接线；用户暂停和超预算两种门限也已检查会拒绝启动。实际训练复用原 VS Code 终端 3，任务为 `XGO-Duck: 31 实时显示性能短测（六项各 40 次）`，批次为 `observer_20261101T201418_s42`。六项现已正常结束，结果和复查命令如下。

观察器只从真实环境复制状态，头部目标与实际角度同步记录，鼠标操作不会写回训练。每次更新记录采样、学习、总耗时、观察器耗时和实际状态样本数，同时记录 CPU 与 GPU 状态；如果窗口关闭，当前短测停止，保留失败记录，不能把少做显示后的时间算作提速。设备名称查询针对同一进程的 EGL 上下文，原生 GLFW 完整渲染能力来自第 25.1 节；没有直接查询被拒绝访问的桌面窗口。

选择规则在执行前固定：GPU 4 Hz 相对默认 4 Hz 两遍都更快且吞吐比几何平均至少 1.15，才认为此驱动值得采用；GPU 15 Hz 需两遍均保留 GPU 4 Hz 至少 90% 吞吐、实际状态更新率均达到每秒 8 次，才选较高上限。这些是本机工程决策条件，不能替代画面观感或步态验收。本轮按该规则选出 GPU 4 Hz，高上限未达到流畅条件。

### 25.4 连续显示结果

六项各完成 40 次更新，全部正常退出；每项最后 20 次的原始计时均纳入统计。完整配置、共同 checkpoint、实际初始观测和首个随机动作一致；最后模型的 Adam、归一化与课程计数正确，参数均为有限值。查看器没有提前关闭，六段录像均完整解码为 200 帧、50 fps。证据见[独立核验](../../records/benchmarks/observer_20261101T201418_s42/verification.json)及[图像检查](../../records/benchmarks/observer_20261101T201418_s42/visual-review.json)。

| 条件 | 第一遍吞吐（样本/秒） | 第二遍吞吐（样本/秒） | 第一遍实际状态/秒 | 第二遍实际状态/秒 |
| --- | ---: | ---: | ---: | ---: |
| 默认驱动，4 Hz 上限 | 12,230 | 11,783 | 2.868 | 2.823 |
| GPU，4 Hz 上限 | 18,380 | 18,347 | 2.290 | 2.249 |
| GPU，15 Hz 上限 | 17,773 | 17,442 | 4.339 | 4.258 |

这里的样本为一个环境的一次控制步。GPU 4 Hz 相对默认条件的吞吐比分别为 1.5028、1.5570，几何平均 1.5297；更新耗时中位数从约 7.99～8.35 秒降至 5.35～5.37 秒。此结论来自同一台电脑的两种运行顺序，适用于本轮连续观察负载，不是跨机器保证，也没有覆盖长训练中的全部周期录像开销。

![显示性能](../../records/benchmarks/observer_20261101T201418_s42/figures/observer-performance.png)

提高显示上限保留了 GPU 4 Hz 条件约 95.1%～96.7% 的吞吐，但仍只有每秒约 4.3 个新状态，没有达到预设的 8 Hz。对应每秒墙钟时间只推进约 0.085～0.087 秒仿真时间，所以仍会看到缓慢、跳动的动作。GPU 4 Hz 的实际状态更新率反而低于默认条件：其控制步更快，却常在 0.25 秒限频窗口内跳过下一次显示；每次 24 步更新通常只记录约 12 个状态。提高训练吞吐和提高画面流畅度需要分别判断。

进程累计 CPU 时间从每个计时区间约 1768～1815 秒降至约 177 秒。CPU 时间会叠加各线程，不能当作墙钟耗时；观察器回调本身也未包含全部原生渲染线程开销。按冻结规则采用 GPU 驱动的证据已足够。用户随后因画面卡顿选择后续关闭原生窗口，显示方式变更见第 26.7 节。后续若比较原生 Ubuntu，应沿用相同预算和真实状态频率统计，不从这轮结果推定其提速比例。

下面的截图记录原 VS Code 任务终端和当时叠放的观察器；窗口边界仍为 1707×1019，没有执行窗口调整。它是一次界面记录，不替代连续刷新率或完整机器人取景的验证。[界面记录](../../records/benchmarks/observer_20261101T201418_s42/vscode-display-observation.json)。

![可见训练任务](../../records/benchmarks/observer_20261101T201418_s42/vscode-visible-task.png)

任务 31 对已有相同协议批次有重复运行限制。只复查本次数据时，在项目根目录的 WSL 终端执行下列命令；它们读取保存的模型、轨迹和录像，不重新训练：

```bash
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
BATCH=records/benchmarks/observer_20261101T201418_s42
$PY -B scripts/validation/verify_observer_benchmark.py "$BATCH"
$PY -B scripts/validation/plot_observer_benchmark.py "$BATCH"
$PY -B scripts/validation/audit_observer_head_commands.py "$BATCH"
```

## 26. 头部指令修正

### 26.1 实际请求与运动

新增头部遥测已在上面六项真实训练中核验：期望姿态等于 home 加姿态指令，实际角度与完整关节数组一致，记录误差可独立复算。这些期望姿态是头部任务给策略的目标，与策略输出的舵机动作目标是两个量。六项中的五项在环境 0 记录到颈部期望姿态超过模型的 60° 限位，所见最高请求约 81.96°；第 23.13 节此前查出的范围不一致确实进入了实际采样。

以下固定展示第一遍 GPU 4 Hz 条件的全部记录，没有按动作好坏挑选片段。横轴为累计仿真时间，橙线为期望姿态，蓝线为实际关节角，虚线为模型关节限位。颈部存在不可满足的越界请求，头部侧摆也有明显跟踪误差。训练期间策略持续更新并包含随机指令及探索噪声，这张图用于定位问题；最终姿态质量仍要回到固定策略、中立头部指令的评估。

![头部目标与实测](../../records/benchmarks/observer_20261101T201418_s42/head-commands/head-command-tracking.png)

六项全部统计及源文件哈希见[头部请求分析](../../records/benchmarks/observer_20261101T201418_s42/head-commands/analysis.json)。各条件的保存频率不同，统计中的样本占比不能直接当作时间占比，不能据此比较“哪种显示驱动改善了头部”。这些记录也没有证明头部摆动是维持平衡所必需。

### 26.2 修改范围

[head_command_bounds.py](../../scripts/training/head_command_bounds.py)提供独立的配置准备函数：编译当前 XGO-Duck 模型，从真实 home、关节限位和原来的软限位系数 0.9 计算允许的对称指令幅度，返回配置副本。代码注释写明问题、修改原因和验证范围，原始归档及现有训练模型保留。

颈部 home 约为 20.002°，模型限位为 −135°～60°，对应软限位为 −125.25°～50.25°。保持指令围绕 home 对称时，幅度由较近的一侧决定，得到 ±0.527928 rad，约 ±30.248°；期望姿态范围因此为约 −10.246°～50.25°。这也同时收窄了原来可达到的负向范围，是保留对称分布的取舍。

| 课程控制步 | 原颈部指令幅度 | 修正后幅度 |
| --- | ---: | ---: |
| 36,000 | ±0.72 rad（41.25°） | ±0.527928 rad（30.248°） |
| 48,000 | ±1.10 rad（63.03°） | ±0.527928 rad（30.248°） |

完整配置比较确认只有这两个区间改变。前三段课程、其他头部轴、奖励、重采样周期、动作执行及物理设置均保持原值；头部关节仍可运动。预期效果是让采样的期望姿态落在模型范围内。摆动是否减小、行走是否更稳，仍需独立训练和固定评估，目前没有修改收益结论。

### 26.3 检查与继续方法

任务 `XGO-Duck: 32 检查头部可达指令（不训练）` 已在原 VS Code 终端 3 执行，退出码为 0。它检查了完整配置差异，并调用原有课程函数和真实指令采样器，在 10 个检查点各采样 4096 组指令；包括各课程边界前后和当前恢复计数 120000。40,960 组采样全部位于对应软限位内，原配置及检查进程的随机数状态保留。没有创建仿真环境、推进物理或更新权重。

结果见[配置与采样核验](../../records/reports/head-command-bounds/check.json)，命令和输出见[终端记录](../../records/reports/head-command-bounds/console.log)。在 VS Code 选择同名任务即可自行复查；也可在项目根目录的可见 WSL 终端执行：

```bash
bash scripts/validation/run_head_command_check.sh
```

![头部检查](../../records/reports/head-command-bounds/vscode-head-check.png)

上图为同一 VS Code 窗口中的头部曲线与任务退出结果，原终端 3 复用，窗口大小与面板布局未调整。[显示记录](../../records/reports/head-command-bounds/display-observation.json)。

检查脚本为 [check_head_command_bounds.py](../../scripts/validation/check_head_command_bounds.py)。新的 [head_command_repair.py](../../scripts/training/head_command_repair.py) 已显式调用 `prepare_reachable_head_commands(cfg.env)` 接收配置副本，并保存完整差异。任务 33 及当前任务 34 都只在 `reachable_head` 组应用该修改；任务 32 本身仍只检查配置。

以下单因素试验已冻结并从原可见终端启动，采用已验证的 GPU 显示路径，记录实际头部请求与运动。完成导出后沿用原五条指令评估，并单独比较中立头部目标下的偏置、摆动和角速度；并脚与速度跟踪仍需同时通过。实际髋姿态或横向脚间距约束另作下一项试验，不能把多种修改叠加后归因于颈部范围修正。整个项目继续执行，原长训练的异常门限和真机后置范围保持有效。


### 26.4 成对训练协议

本节记录原任务 33 的冻结条件。批次 `head_command_repair_pilot_20261101T210228_s42` 于 21:02 启动，21:19 因查看器关闭退出，仅原样组运行，不能当作成对结果。原始协议、源码快照、日志和模型全部保留；当前执行任务 34，恢复方式见第 26.7 节。

| 项目 | 两组共同设置 |
| --- | --- |
| 恢复起点 | 累计 5000 次的 `model_4999.pt`，保留权重、Adam、学习率和观测归一化 |
| 训练预算 | seed 42，4096 环境，每次采样 24 步，各追加 250 次更新、24,576,000 次转移 |
| 唯一组间差异 | `reachable_head` 组收窄末两段颈部指令；`control` 组保持原指令 |
| 显示与记录 | 进程内 D3D12、4 Hz 观察上限；每 50 次保存，每 1200 控制步录像 200 帧 |
| 退出条件 | 查看器关闭则停止；不在失去可见画面后继续学习，不自动重启失败组 |
| 模型选择 | 只取最后一个保存点，不挑最好片段或中间模型 |
| 固定评估 | 原五指令、32 个首个 episode、20 秒、seed 1001、原阈值；头部指令为零 |

修改只涉及头部任务的要求，不对动作输出做裁剪，也不添加头部奖励或并脚惩罚。从旧 checkpoint 恢复仍使用全新环境 reset，不能称为逐步精确续接旧轨迹。原样组和修改组需核验物理初态、CPU/CUDA 随机状态及其他观测一致；颈部观测按 `0.5279279491271505 / 1.1` 缩放。策略输入因此不同，首个动作无需完全相同；另核验动作分布采样所用的标准正态噪声一致。真实数据由 [head_command_start.py](../../scripts/training/head_command_start.py) 保存，修改组在第一步物理推进前进行比较。

头部指标按每个 episode 计算相对 home 的平均偏角、去均值角度 RMS 和绝对角速度的 95 分位数；同时报告整段及排除首秒的结果。逐条指令、逐个关节保留统计，比较同编号 episode 的差异，不把大量连续帧当作独立试验。单 seed 的 250 次试验用于判断方向，不能单独证明普遍改善；原有运动和并脚标准保持有效，结果不会自动解除长训练限制。

原任务入口为 `XGO-Duck: 33 头部指令修复短训练（成对 250 次）`，现不再运行。其[批次记录](../../records/experiments/head_command_repair_pilot_20261101T210228_s42/batch.json)、[冻结协议](../../records/experiments/head_command_repair_pilot_20261101T210228_s42/protocol.json)和[预检快照](../../records/experiments/head_command_repair_pilot_20261101T210228_s42/preflight.json)可用于核查。任务 33 没有完成修改组，因此不能直接执行其最终成对收尾，也不能把随后新试验的结果补写进这个失败批次。

### 26.5 可见过程

原样组首段完整录像已逐帧解码：200 帧、50 fps、960×720，共 4 秒，来源及哈希见[媒体核验](../../records/experiments/head_command_repair_control_20261101T210228_s42/media-check/check.json)。这是本次训练环境 0 的已保存片段，含探索噪声和随机指令，不是实时画面或最终策略效果。

<video controls preload="metadata" src="../../records/experiments/head_command_repair_control_20261101T210228_s42/videos/rl-video-step-0.mp4"></video>

![训练录像与终端](../../records/experiments/head_command_repair_pilot_20261101T210228_s42/vscode-training-video.png)

当时原终端 3 正常输出更新，VS Code 中已打开本次录像并能看到全身。截图停在 0 秒，尚未核验界面播放。当时实时观察器在采集真实状态，但它的取景过近，只能看到部分头部，[启动截图](../../records/experiments/head_command_repair_pilot_20261101T210228_s42/vscode-start.png)和[显示记录](../../records/experiments/head_command_repair_pilot_20261101T210228_s42/display-observation.json)保留此问题。实时全身取景和流畅度均未验收；用户随后选择关闭此类窗口，后续采用保存片段。

一次视频标签聚焦引发桌面工具改变窗口状态；已恢复并重新核对为原来的 1707×1019、位置 (0,0)，面板仍为同一任务终端。此后停止该聚焦方式，详见[恢复记录](../../records/experiments/head_command_repair_pilot_20261101T210228_s42/window-restoration.json)。


### 26.6 首个阶段检查

截至 21:08，原样组的 `model_5050.pt` 已通过[独立 checkpoint 检查](../../records/experiments/head_command_repair_control_20261101T210228_s42/checkpoint-checks/model_5050.json)：累计完成 5051 次，其中本段 51 次；环境计数 121224，actor/critic 观测计数均为 127893504，Adam 每个参数的步数为 101020。76 个待查张量均为有限值，actor 和 critic 各 8 个 MLP 张量相对父模型发生变化，恢复副本已保存。

同一时刻，[冻结的过程检查](../../records/experiments/head_command_repair_control_20261101T210228_s42/resume-verification/20261101T130828953797Z/check.json)核对了前 55 次完整更新的终端日志、TensorBoard 与独立计时，三者一致。以下曲线保留全部原始点，没有平滑；横轴为累计完成更新次数。启动后的 episode 历史会逐渐累积，奖励与平均 episode 长度的早期上升不能单独当作模型改善。

![原样组训练过程](../../records/experiments/head_command_repair_control_20261101T210228_s42/resume-verification/20261101T130828953797Z/figures/continuation-progress.png)

这只是当时原样组的阶段证据。原任务随后中断，没有完成成对预算或最终评估；下一节保留中断账目及新的共同起点。

### 26.7 关闭实时窗口后的恢复

实时查看器于 21:19 关闭，具体发起原因未知。观察器通用日志写着“training continues”，但本试验外层立即抛出 `Live display closed during update`，最终退出码为 1；实际没有在关窗后继续完成这段学习。[中断核验](../../records/experiments/head_command_repair_control_20261101T210228_s42/interruption-recovery.json)登记了 172 次完整新更新，其中 `model_5150.pt` 保留 151 次，累计 5151 次；另外 21 次完整更新和下一段的 20 个控制步没有写入模型。修改组未启动，四段已保存视频均完整解码。

[保存点检查](../../records/experiments/head_command_repair_control_20261101T210228_s42/checkpoint-checks/model_5150.json)确认 actor/critic 归一化计数均为 137723904，Adam 步数为 103020，课程计数为 123624，76 个张量均为有限值。恢复文件 SHA-256 为 `f0580233e45029328bd1a5539b1599d857b4e5036a94325df8c78262cf5a2d32`。选择它是因为它是退出前最新的完整保存点，没有按步态或头部效果挑选。

新试验的两组均从这份原样模型开始，各训练 250 次，最后累计 5401 次，预期最终文件为 `model_5400.pt`。保留原 151 次学习作为共同起点，使两组预算和模型状态一致；重新 reset，不能称为逐步重现中断轨迹。唯一组间差异仍是末两段颈部指令范围，奖励和物理不变。协议见[录像模式成对训练](../../configs/experiments/head_command_repair_recorded.json)，真实配置、保存点及任务接线已通过[入口检查](../../records/reports/head-command-recorded-entry-check.json)。

显示方式由[训练显示配置](../../configs/environments/training_display.json)统一记录：`native_live_window=false`，新入口实际跳过 `LiveObserver` 的创建、显示和逐步回调。训练依然运行在单个可见 VS Code 任务终端；每 1200 个控制步录制 200 帧，视频按 50 fps 播放，相当于一段 4 秒的真实仿真动作。录像含探索和随机指令，不是直播，也不是最终模型验收。编码帧率不等于已测得的屏幕刷新率；本次还没有完成去掉查看器前后的受控速度比较。

目前任务 34 已在原终端启动，批次为 `head_command_repair_pilot_20261101T213026_s42`。原样组随后运行、修改组顺序接续，结束后才导出和执行固定评估。查看[批次状态](../../records/experiments/head_command_repair_pilot_20261101T213026_s42/batch.json)，不要重复启动同一协议。

自行操作时，在原 VS Code 选择 `Terminal → Run Task… → XGO-Duck: 34 头部修复成对训练（录像显示）`。实际入口为 [head_command_repair_recorded.sh](../../scripts/training/experiments/head_command_repair_recorded.sh)，核心修改及原因写在 [head_command_repair.py](../../scripts/training/head_command_repair.py) 和 [display_preferences.py](../../scripts/training/display_preferences.py) 的注释中。只检查配置可在项目根目录的可见 WSL 终端执行：

```bash
bash scripts/training/experiments/head_command_repair_recorded.sh --check-only
```

每组的 `training.log`、`progress.json`、`iteration-timing.jsonl` 和 `videos/` 分别保存终端输出、最新更新、逐次计时和片段；`display-state.json` 明确记录不创建原生窗口，`training-model-metadata.json` 保存实际模型关节名称与限位。若两组学习都正常完成、但导出或评估失败，保留权重，在同一可见终端只执行收尾：

```bash
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
$PY -B scripts/validation/finish_head_command_repair.py \
  records/experiments/head_command_repair_pilot_20261101T213026_s42
```

头部结果会逐条指令、逐个关节报告整段和首秒后的偏置、摆动 RMS 与角速度。独立分析入口 [compare_head_command_outcomes.py](../../scripts/validation/compare_head_command_outcomes.py) 还会从原始轨迹重算并保存同编号 episode 的差值；已对当前官方参考的保存轨迹通过 [CPU 核验](../../records/reports/head-comparison-entry-check.json)，本次两组的结果仍待训练和评估完成。并脚、运动跟踪和头部稳定分别验收，关闭窗口本身不代表这些问题已修复。

### 26.8 录像模式的实际检查

任务 34 的原样组于 21:49 正常结束，退出码为 0。全部 250 次更新的终端日志、TensorBoard 和独立计时已[交叉核验](../../records/experiments/head_command_repair_control_20261101T213026_s42/resume-verification/20261101T135014603755Z/check.json)，累计完成 5401 次。实际课程计数和学习率恢复正确，完整计时记录中原生查看器均关闭，观察器采样和耗时均为零。

[最终保存点检查](../../records/experiments/head_command_repair_control_20261101T213026_s42/checkpoint-checks/model_5400.json)确认 `model_5400.pt` 包含本段全部 250 次更新：actor/critic 归一化计数均为 162299904，Adam 步数为 108020，课程计数为 129624。76 个待查张量均为有限值，actor 和 critic 各 8 个 MLP 张量相对共同起点发生变化；恢复副本已经保存。该检查证明学习预算和保存完整，步态仍须由固定评估验收。

![原样组训练曲线](../../records/experiments/head_command_repair_control_20261101T213026_s42/resume-verification/20261101T135014603755Z/figures/continuation-progress.png)

图中为本段 250 次更新的全部原始点，不做平滑；单 seed、4096 环境。重新 reset 后 episode 历史逐步累积，因此早期奖励和 episode 长度上升不能证明步态改善。吞吐曲线也不能代替去掉查看器前后的受控速度比较。

五段录像均含 200 帧、50 fps、960×720，各长 4 秒，已完整解码并核对运行副本与报告引用文件一致，见[媒体检查](../../records/experiments/head_command_repair_control_20261101T213026_s42/media-check/check.json)。它对应新原样组的相对控制步 0 开始片段，而非旧训练画面。

<video controls preload="metadata" src="../../records/experiments/head_command_repair_control_20261101T213026_s42/videos/rl-video-step-0.mp4"></video>

![录像与可见终端](../../records/experiments/head_command_repair_pilot_20261101T213026_s42/vscode-recorded-training.png)

本次已在 VS Code 从 0 秒播放至 4 秒结尾，机器人全身可见，原终端 3 同时继续训练。操作前后窗口均为 1707×1019、位置 (0,0)，没有调整窗口或面板；[界面记录](../../records/experiments/head_command_repair_pilot_20261101T213026_s42/display-observation.json)保存核验。截图只是播放到结尾的界面证据，不代表测得屏幕帧率。此片段中的头部动作仍含随机指令，头部改善和并脚问题须等固定评估后判断。


### 26.9 修改组的实际起点

修改组已在同一终端顺序开始学习，没有另开训练。21:53 的[独立复查](../../records/experiments/head_command_repair_pilot_20261101T213026_s42/matched-start-recheck.json)重新读取两组实际保存的初始张量、完整 YAML 和源码快照：物理状态、CPU/CUDA 随机状态、episode 计数、home、其他观测及动作采样噪声一致。两组恢复计划及恢复后的学习率、课程计数一致；每组 38 份源码快照均与启动清单和当前文件哈希一致。

唯一配置差异仍为末两段颈部范围：±0.72、±1.1 rad 均收窄至 ±0.5279279491 rad。首个颈部指令观测按 0.4799344992 缩放，其他观测列完全一致。由于输入已改变，不要求两组首个动作相同；核验的是标准正态采样噪声一致。这能确认对比起点和修改范围，不能提前证明训练有效。

修改组也实际跳过了原生查看器，只保留同一可见终端及录制片段。两组各 250 次学习正常结束后，原任务将顺序导出并执行原五指令评估；头部、脚间姿态和速度跟踪分别检查，不自动追加长训练。

截至 21:54，修改组包含 50 次新更新、累计 5201 次的 [保存点](../../records/experiments/head_command_repair_reachable_head_20261101T213026_s42/checkpoint-checks/model_5200.json)已核验，前 53 次完整更新的[日志、TensorBoard 和计时](../../records/experiments/head_command_repair_reachable_head_20261101T213026_s42/resume-verification/20261101T135401887811Z/check.json)一致。以下是该组相对控制步 0 开始的真实训练片段，已逐帧解码，并在原 VS Code 中从 0 秒播放至 4 秒结尾。

<video controls preload="metadata" src="../../records/experiments/head_command_repair_reachable_head_20261101T213026_s42/videos/rl-video-step-0.mp4"></video>

![修改组录像](../../records/experiments/head_command_repair_pilot_20261101T213026_s42/vscode-candidate-recorded-training.png)

[播放记录](../../records/experiments/head_command_repair_pilot_20261101T213026_s42/candidate-display-observation.json)确认窗口大小和位置不变，原任务终端 3 继续运行。该片段为随机训练条件下的保存画面，不是最终模型的固定指令评估。

## 27. 实际髋姿态约束

### 27.1 保存轨迹中的区别

当前头部成对训练继续按冻结协议运行；本节只复算既有数据，没有改动学习入口。此前目标越界惩罚减少了过大的舵机请求，但实际髋关节仍停在内侧限位。因此下一项候选直接使用实际关节角，而不是再用请求目标代替姿态。

分析覆盖已完成的原样组、目标惩罚组和当前官方参考，各五条指令、32 个首个 episode、20 秒。分别统计全段及首秒后，保存 960 行逐 episode 数据。两支自训模型均为累计 5250 次；官方模型训练历史未知，不能把它当作同预算改进组。以下为前进指令、排除首秒后的逐 episode 平均，再在 32 个 episode 上取平均。

| 模型 | 双髋持续内收占比 | 姿态奖励 | 近限位代价（rad） | 平均脚间距（mm） |
| --- | ---: | ---: | ---: | ---: |
| 原样组 | 100.0% | 0.000086 | 0.3180 | 1.71 |
| 目标惩罚组 | 100.0% | 0.000116 | 0.3129 | -3.88 |
| 官方参考 | 0.0% | 0.450443 | 0.0000 | 56.38 |

双髋内收指左髋低于 −20° 且右髋高于 20°。脚间距是足部位点差在机身左方向上的带符号投影，负值表示位点交叉，并不等于测得了碰撞力。本表使用排除首秒后的均值，与前文全段中位数定义不同。完整五指令数据见[分析记录](../../records/reports/hip-posture-candidates/analysis.json)，逐 episode 数据见[原始统计表](../../records/reports/hip-posture-candidates/episodes.csv)。

### 27.2 原姿态奖励为何难以区分

安装环境中的 `variable_posture` 对十个腿部关节计算 `exp(-mean((q-home)^2/std^2))`，髋侧摆的 `std` 为 0.05 rad，训练权重为 1.3。这里的表格列出未乘权重的函数值，用保存的实际关节角和仿真 home 复算，并非历史训练日志中的逐项奖励。

旧两组在首秒后的全部五指令中，双髋同时超过内收 20° 的有效帧占比均为 100%，姿态奖励低于 0.01 的占比也均为 100%。在前进片段中，仅把输入奖励公式的两个髋角替换成 home、其他角度保持原值，原样组的函数均值从 0.0000858 变为 0.1510，目标惩罚组从 0.0001162 变为 0.1804。这说明髋角对该公式接近零的结果有很大贡献；此操作只重算奖励，不是可实现的物理姿态、策略修复或因果训练试验。仍不能据此保证加大原姿态权重就能摆脱当前行为。

### 27.3 候选函数与下一步

官方归档已经提供 `joint_pos_limit_proximity` 和 `joint_deviation_l1`。前者读取实际关节角：在硬限位内预留 `margin` 后，按进入边缘区域的弧度超额求和；后者对实际角度相对 home 的绝对偏差求和。本次都调用原函数，并用独立 NumPy 公式复核，最大绝对误差约为 1.2×10⁻⁷，未创建仿真环境或初始化 CUDA。

近限位函数沿用源码默认 `margin=0.15 rad`，没有在这些轨迹上搜索阈值。对当前髋侧摆硬限位，这给出约 ±13.41° 的零代价区间。首秒后的五条指令中，旧原样组代价约 0.317～0.318 rad，目标惩罚组约 0.311～0.313 rad，官方参考均为零；相对 home 的绝对偏差项则也会惩罚官方参考的正常髋动作。因此，下一项优先考虑仅给左右髋侧摆增加近限位代价，保留区间内的运动空间。训练权重、共同起点与预算须在当前头部试验完成后固定，不把这项诊断写成已启动或已有效的修复。

复算入口为 [analyze_hip_posture_candidates.py](../../scripts/validation/analyze_hip_posture_candidates.py)，代码注释明确了数据来源、修改范围、预期与验证边界。在项目根目录的 WSL 终端运行：

```bash
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
$PY -B scripts/validation/analyze_hip_posture_candidates.py
```

后续仍需同起点、同预算的真实训练验证，再沿用原五指令运动、跌倒、脚间距和髋角标准验收。不会因离线代价区分清楚就解除长训练限制。

### 27.4 惩罚量级

在选择训练权重前，另对已核验轨迹比较 −1、−2、−4 三个候选值。以下仍是旧原样模型的前进指令、首秒后、32 个首个 episode；只计算“代价 × 权重”，没有用这些权重重新训练或推进仿真。

| 权重 | 加权惩罚 | 占姿态奖励上限 | 每控制步惩罚 |
| --- | ---: | ---: | ---: |
| −1 | −0.3180 | 24.46% | −0.00636 |
| −2 | −0.6360 | 48.92% | −0.01272 |
| −4 | −1.2720 | 97.85% | −0.02544 |

姿态奖励的加权上限为 1.3。当前配置启用按时间缩放，控制周期为 0.02 秒，因此最后一列还乘以 0.02。加权惩罚不等于 episode 总回报，也不是当前训练的实际日志。若两个关节恰好都在硬限位上，未加权代价为 `2 × 0.15 = 0.30 rad`，三个候选对应 0.3、0.6、1.2 的惩罚幅度；既有原样组轨迹的实测均值为 0.318，与恰好处于硬限位时的假设数值 0.30 分开报告。

这给后续选择权重提供量级参照：−4 在现有并脚轨迹上接近原姿态奖励上限，−1 约为其四分之一。官方参考的这项代价在五指令首秒后均为零。当前尚未选择权重，不能据此断言越大越好或已经改善步态；先完成头部试验，再固定下一项单因素协议。

完整结果见[量级检查](../../records/reports/hip-posture-candidates/weight-scale.json)和[五指令统计表](../../records/reports/hip-posture-candidates/weight-scale.csv)。复算入口为 [analyze_hip_reward_scale.py](../../scripts/validation/analyze_hip_reward_scale.py)：

```bash
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
$PY -B scripts/validation/analyze_hip_reward_scale.py
```

<a id="training-review"></a>

## 28. 训练复盘

### 28.1 范围与图像

本次只读取已有训练文件、模型、轨迹和视频，没有训练、模型推理、仿真推进或导出重试。原始录像和官方归档保持不动。复盘覆盖 12 次主要训练和 14 次性能诊断，固定评估按各自协议分别核对，不能把不同指令、时长、模型的结果混成一次试验。

共找到 197 个视频文件：64 段训练在 Linux 与项目目录共有 126 份文件，另有 67 段固定评估和 4 段静态渲染。训练副本按 SHA-256 关联，同一内容属于不同试验时仍保留不同来源。64 段训练及 67 段评估全部完整解码，未发现不可读片段；两次各 5 次更新的流程预检没有保存训练视频，不补造画面。4 段静态渲染没有步态，列入清单而未放进步态图。

先对每段提取四个时间候选帧，再逐段查看，从中选择两帧，合计 262 帧。选择覆盖倾倒、腿脚靠拢、正侧视变化和头颈变姿态，不把抽样帧标成没有测量依据的“触地”“离地”事件。图中橙框标出腿脚或倾倒区域，黄框标出头颈，注明片段编号、来源阶段、控制步、帧号及片段内播放时间。训练片段可能包含随机指令、探索、扰动和自动复位，前后两帧不一定属于同一 episode。评估视频仅显示环境 0，旁注的存活和交叉数来自全部有效首个 episode。

![关键步态](training-gait-keyframes.jpg)

上图是关键问题的阅读入口。包含全部片段的[单张高清总览图](training-gait-overview.jpg)为 3760 × 17414 像素，需放大阅读；[逐段图册](training-gait-gallery.html)可按编号、试验或问题搜索，打开未裁剪原始帧和原视频。不同片段的截图只等比例缩放和裁剪背景，没有改变机器人姿态。每帧的精确来源、SHA-256 和裁剪区域均在[图像清单](../../records/reports/training-review/overview-manifest.json)中。

### 28.2 实际训练量

下表的“日志更新”是本次运行完整输出的更新数，“保留更新”是本次运行已写入最新模型的新增部分。“累计保存”包含父模型，不能把分叉试验各行相加当作一条策略的训练量。每个主要训练均为 seed 42。

| 阶段 | 环境数 | 日志更新 | 保留更新 | 累计保存 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 基线流程预检 | 256 | 5 | 5 | 5 | 流程通过，未验收步态 |
| 线速度奖励 3.3 预检 | 256 | 5 | 5 | 5 | 只验证修改入口，无收益结论 |
| 录像预检 | 64 | 20 | 20 | 20 | T001，姿态失稳 |
| 早期基线 | 1024 | 260 | 251 | 251 | 中断，9 次完整更新未保存 |
| 基线续训 | 1024 | 4749 | 4749 | 5000 | 只有停止通过原数值标准 |
| 基线追加 | 4096 | 752 | 751 | 5751 | 中断，1 次完整更新未保存 |
| 实时续训 | 4096 | 78 | 0 | 5751 | 中断，78 次完整更新未保存 |
| 请求目标惩罚对照 | 4096 | 250 | 250 | 5250 | 并脚与运动标准未通过 |
| 请求目标惩罚修改 | 4096 | 250 | 250 | 5250 | 请求缩小，姿态仍失败 |
| 头部原样试验中断段 | 4096 | 172 | 151 | 5151 | 查看器关闭后退出，21 次未保存 |
| 头部原样录像组 | 4096 | 250 | 250 | 5401 | 模型保留，固定评估未执行 |
| 可达颈部指令组 | 4096 | 250 | 250 | 5401 | 模型保留，固定评估未执行 |

两次剖析各 27 次、六次规模筛查各 40 次、六次显示筛查各 40 次，共 14 次独立诊断、534 次新增更新。这些用来测性能，不算正式步态改进。四次中断共留下 109 次未保存的完整更新，另有不完整控制段；不能从日志终点直接续训。退出码 130 对应当时的中断，不能称为数值发散。完整运行名、原始文件哈希和模型计数见[训练台账](../../records/reports/training-review/training-ledger.csv)与[离线审计](../../records/reports/training-review/history-audit.json)。

本次读取全部主要训练的最新保留模型，核对张量均为有限值、Adam 步数与累计更新数对应、归一化样本数按每段实际环境数累计；已有课程计数也与保存更新数对应。最新两组最终模型各保留 250 次新增更新，累计 5401 次，优化器步数均为 108020，归一化样本计数均为 162299904。这只证明文件与训练计数完整，不证明步态合格。

### 28.3 问题与证据

**第一，策略学会了持续内收的站姿，没有学到合格的行走。** 5000 次基线的前进轨迹中，左右髋侧摆中位数为 −22.53°、+22.51°，双髋同时内收超过 20° 的有效样本占 99.53%。请求目标中位数甚至为 −310.31°、+147.69°；这是执行器请求，实际关节没有转到这些角度。T010、T032、T058、T064 的正面画面呈现腿脚向内聚拢，E043 和后续机制轨迹确认这不只是镜头重叠。

**第二，请求目标惩罚改变了动作数值，却没有修复实际姿态。** 原样组与修改组从同一累计 5000 次模型各追加 250 次；前进评估中，双髋请求中位数从约 −254.58°、+254.34° 缩小至 −57.75°、+58.40°，实际角度仍约为 ±22.4°。两组五条指令均各有 32/32 个 episode 出现负脚间距，只有停止通过原运动指标。因此，不能因请求超额下降或训练奖励增加而继续无门槛追加预算。

下表是本次直接从 NPZ 复算的前进条件，指令均为 0.15 m/s，头部与身体指令为零。平面误差先对每个时刻取二维速度误差的 L2 范数，再按时间及 episode 求均值。脚间距为左右足部测点差在机身左方向的带符号投影；负值表示测点左右次序交叉，不是脚外轮廓净空或碰撞力。

| 条件 | 回合与时长 | 存活 | 平面误差（m/s） | 脚间距中位（mm） | 曾交叉 |
| --- | --- | ---: | ---: | ---: | ---: |
| 5000 次基线 E043 | 32 × 20 s | 32/32 | 0.1056 | 未保存此通道 | 未保存此通道 |
| 机制原样 E047 | 8 × 8 s | 8/8 | 0.1063 | −1.48 | 8/8 |
| 开启脚间碰撞 E049 | 8 × 8 s | 1/8 | 0.2595 | 9.08 | 7/8 |
| 髋目标限制 ±10° E051 | 8 × 8 s | 8/8 | 0.1547 | 36.95 | 0/8 |
| 5250 次原样 E054 | 32 × 20 s | 32/32 | 0.1032 | 2.10 | 32/32 |
| 5250 次目标惩罚 E059 | 32 × 20 s | 32/32 | 0.1172 | −2.99 | 32/32 |
| 当前官方参考 E064 | 32 × 20 s | 32/32 | 0.1269 | 54.12 | 0/32 |

这七条前进条件都没有通过原数值标准。官方模型姿态更接近可接受站姿，前进平均速度为 0.1019 m/s，但不能称为在本协议下已全面走稳；其训练历史未知。机制诊断没有重新训练：直接开启脚间碰撞会让旧策略在 8 次前进中跌倒 7 次；直接限制髋目标能分开脚，却把平均前进速度降到 0.0309 m/s。两项都不能直接当作部署修复。

**第三，训练目标和物理约束仍有不一致。** 既有模型检查发现脚间碰撞掩码不允许两脚互撞，因此“自碰撞次数为零”不能排除脚部交叉。官方姿态函数又是十个腿部关节误差经指数压缩，髋侧摆标准差仅 0.05 rad。本次用已存角度重算，前进首秒后的姿态奖励均值在原样组为 0.0000858、目标惩罚组为 0.0001162，当前官方参考为 0.45044。持续内收时该奖励已很接近零，说明它在当前姿态附近的数值区分有限。这是候选解释，不是“某一个奖励必然导致并脚”的因果证明；候选实际髋角代价及量级见第 27 节，尚未启动训练。

**第四，头部训练混入了不可达目标，不能把所有摆动都理解成平衡动作。** T039 显示颈部大幅倾斜，训练同时包含随机头部姿态命令；既有配置与 CPU 采样核查确认最终颈部期望上界约 83.03°，超过模型 60° 上限约 23.03°。中立指令的固定评估也仍有头部摆动，故随机命令并不能解释全部误差。两支最新模型只改变后两段颈部指令范围，但没有最终固定评估。T064 仍可见并脚，不能据此量化头部收益，也不能宣称“头被用作平衡杆”已经得到验证。

**第五，计算吞吐改善没有转化为合格步态，实时显示也拖慢了观察。** 最新关闭原生窗口的两组分别以约 22175、21579 个并行环境转移/秒完成更新计时；其中采样分别占 90.58%、90.36%，策略学习仅约 7.2%。单个环境的物理时间每墙钟秒仅推进 0.108、0.105 秒，所以高并行吞吐不等于单只机器人画面按真实速度运动。计时不含全部启动、编译和收尾，不能拿它冒充整个任务耗时。旧观察器上限调至 15 Hz 后也只实际产生约 4.3 个新状态/秒，卡顿不能仅靠调整播放窗口解决。原生 Ubuntu 尚无同条件测量，不能预先承诺提速比例。

**第六，流程对中断和新实验接入的处理不足。** 中断时保存间隔过长，导致上述 109 次完整更新未保留；任务 33 的查看器生命周期还与学习进程绑定。最新成对试验虽已各完成 250 次，`prepare_policy_export.py` 的名称白名单没有接受头部试验，自动收尾报 `Unsupported recorded experiment name`，在导出开始前退出。尚不能从没有导出的模型生成新的固定评估结论。本次只记录这个失败，没有重试。

**第七，证据范围还不足以支持正式修改收益。** 主要训练全为 seed 42，多 seed 与留出指令尚未完成；前期长预算没有阻止并脚策略继续固化。奖励提高、核验脚本写入 `passed`、文件导出完成，分别只代表相应计算或流程通过。首次 E001 预检还使用 x/y 分量绝对误差均值，并在首条条件后报错；后续使用平面 L2 误差。本次按各自保存的计算代码核对原值，同时保留统一 L2 重算值，E001 明确标为失败预检，不与正式结果混比。

### 28.4 继续之前的顺序

当前不再追加训练。若用户以后明确恢复，应先处理已有成果和流程，再决定是否值得学习：

1. 修复头部试验的导出准入适配，检查真实配置和恢复状态，导出现有两组累计 5401 次模型，完成同协议固定评估；不重复已完成的 250 次更新。
2. 把“速度与转向跟踪、存活、髋角、带符号脚间距、头部中立跟踪”同时作为准入条件。先验证官方参考与自训模型的接口和动力学差异，再判断奖励、动作尺度及碰撞约束的修改范围。
3. 只选一个有离线证据支持的修复因素做短试验。实际髋角近限位代价是已有候选；是否加入脚间碰撞、头部约束应分别设计，避免把多处修改的效果混在一起。短试验未改善姿态和跟踪时停止，不把增加更新数当作默认补救。
4. 把受控停止时保存完整状态、查看器退出不终止训练、导出入口在长任务前验证作为流程修复。学习恢复后继续使用一个可见 VS Code 终端和已保存录像，不恢复卡顿的原生窗口。
5. 只有短试验通过预设条件后才扩大预算、增加 seed 43/44 和留出评估。原生 Ubuntu 先做与 WSL 同模型、同预算、同渲染条件的速度对照；切换系统本身不会修复奖励或姿态问题。

以上是恢复后的顺序，不代表当前已启动这些工作。当前模型均未通过稳定行走验收，也没有执行真机标定或部署。

### 28.5 离线复现

在项目根目录的 WSL 终端执行以下命令，只处理保存文件，不会创建训练环境。完整扫描使用现有 Linux 专属日志目录；机器迁移时需一并保留原视频，或使用清单中登记的项目副本。

```bash
cd /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck
PY=~/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
$PY -B scripts/validation/build_gait_video_overview.py inventory
$PY -B scripts/validation/build_gait_video_overview.py extract
$PY -B scripts/validation/build_gait_video_overview.py extract-evaluations
$PY -B scripts/validation/audit_training_history.py
$PY -B scripts/validation/build_gait_video_overview.py compose
$PY -B scripts/validation/verify_training_review.py
```

`build_gait_video_overview.py` 中的 `TRAIN_CHOICES` 保存人工选择的候选帧，`TRAIN_NOTES` 保存逐段观察，`annotated_frame` 保存裁剪和标注规则。修改注释或选帧后重跑 `compose` 即可，不必再次解码。`audit_training_history.py` 独立读取 checkpoint 和 NPZ，不加载策略推理。审计、图像、原始候选帧和媒体清单集中在 `records/reports/training-review/`，最终图片及阅读入口集中在 `docs/reports/`。


<a id="training-workflow"></a>
## 29. 训练工作流

### 29.1 为什么调整

旧流程把“奖励增长、存活、完整跑完预算”看得过重。复盘发现实际髋角已长期贴近内收限位，脚位交叉却未触发停止；头部随机目标也包含不可达范围。持续原生显示只得到约每秒四个新状态，关窗又曾中断学习。训练内频繁自动导出 ONNX、过多编号任务、不同入口的名称白名单，使观察和收尾难以维护。

本次把学习、观察、判断和记录分开：奖励与物理不在工作流整理中悄悄改变；观察降低频率，异常判断持续进行；只在完整更新后保存模型并停止。新入口仍沿用已安装的官方 runner，不另装训练框架。改动原因和验证边界写在脚本开头。没有测得新入口的实际训练吞吐，不能把减少渲染操作直接写成提速比例。

| 原问题 | 已做的调整 | 当前验证范围 |
| --- | --- | --- |
| 名称难辨认、菜单过长 | 26 条历史记录增加展示名；日常任务由 34 项收敛为 5 项 | 名称索引、原路径和文件存在性检查 |
| 持续窗口卡顿 | 默认每 180 秒从真实训练环境保存一张 640×480 画面；不创建原生查看器 | 页面历史帧预览；实际训练渲染待 S1/S2 小预算验证 |
| 只看奖励，异常仍继续 | 32 个分散环境的姿态抽样、跌倒事件、数值检查；持续异常退出 | 15 条已有固定评估轨迹回放 |
| 关窗、中断丢进度 | Ctrl+C 请求完整更新后保存；每 25 次更新或 120 秒保存，以先到者为准 | CPU 检查保存顺序、计数和失败保留逻辑 |
| 每次保存自动导出 | 保存训练状态，ONNX 在独立评估阶段导出 | 实际安装的保存函数在 CPU 上验证，不产生 ONNX |
| 收尾失败仍可能误判成功 | 异常退出写原因，停止后不接下一组、不自动重试 | 软件契约检查；真实 GPU 运行仍待验证 |

### 29.2 命名与位置

新运行标识采用 `duck_<动作>_<改动>_<组别>_s<种子>_<本地时间>`。动作使用 `walk`、`stand`、`standup`、`roll`、`switch`，组别使用 `reference`、`control`、`candidate`。例如 `duck_walk_hip-margin_candidate_s42_20261102T010000` 表示行走、髋限位余量候选、修改组；这只是命名示例，尚无对应训练结果。改动词必须说明内容，不能只写“更好”或随意编号。

历史目录没有改动，避免模型、截图和报告链接断开。[名称索引](../../configs/experiments/run_catalog.json)列出展示名、原运行标识、保存模型与哈希。典型名称为“行走｜官方配置｜续至 5000 更新”“并脚诊断｜目标越界惩罚｜修改”“头部诊断｜可达颈部指令｜修改”。[历史任务菜单](../../configs/environments/vscode_tasks_history.json)保留原始 34 项定义，供复现查阅。

| 位置 | 用途 |
| --- | --- |
| `configs/experiments/motion_curriculum.json` | 动作阶段、评估条件与门槛 |
| `configs/experiments/next_training.json` | 下一次单组训练协议，当前为未冻结草案 |
| `configs/experiments/training_watch.json` | 行走修复的观察与停止阈值 |
| `scripts/training/guarded_training.py` | 有边界的新行走入口，记录本次源码与配置 |
| `scripts/training/training_watch.py` | 命名、姿态判定、完整更新保存、画面采集 |
| `records/experiments/<运行标识>/` | 配置、源码、终端日志、画面、停止原因与索引 |
| `~/XGO-Duck/workspace/xgoduck_rl/logs/rsl_rl/` | Linux 文件系统中的实际模型与 TensorBoard |
| `records/reports/training-workflow/` | 本阶段离线核验和官方模型结构检查 |

### 29.3 画面与停止规则

新入口启动时拍摄真实环境 0，此后每 180 秒拍摄一次，异常停止时再拍一张。画面从训练中的状态复制到独立渲染模型，只做运动学重建，不额外执行策略或推进训练物理。网页标明运行名称、采样时刻、环境编号和控制步；超过 240 秒未取得新图或 30 秒未取得进度心跳会显示陈旧提示。训练中的页面为 `http://localhost:8766/`，服务器随前台训练进程退出；旧的 8765 页面不作为新入口的数据来源。结束后可直接打开该运行目录的 `index.html` 查看保存结果。

[观察页预览](training-observer-preview.html)使用 E054 的真实历史评估帧，明确标为历史画面，没有伪装成当前训练。以下截图来自独立无界面的浏览器渲染，未操作用户现有窗口。

![观察页面](../../records/reports/training-workflow/observer-preview.png)

当前自动门槛专用于已有保存点上的行走修复。每 8 个控制步抽取 32 个均匀分布环境，跳过 reset 后前 1 秒的姿态。每约 2 秒仿真时间汇总一次：双髋同时向内超过 20°的有效样本比例大于 50%，或横向脚位距离小于 −1 mm 的比例大于 25%，或本窗口超过一半采样环境发生非超时终止，均记为异常窗口。连续两个同类异常窗口后，在下一次完整更新边界保存并停止。样本持续不足也停止检查。跌倒事件每控制步累计，不能因为抽样间隔而漏掉重置。

数值检查覆盖动作、观测、抽样姿态及待写入模型张量；发现非有限数时保留最近完整保存点，不把半段 rollout 冒充完整更新。Ctrl+C 第一次请求在更新边界停止，再次按下强制中断，后一种情况只保证已有完整保存点。渲染失败同样要求退出分析。网页关闭不会触发学习进程崩溃。

这些是项目的诊断门槛，不是官方标准，也不能取代运动质量验收。脚位距离是左右脚站点在躯干横轴上的投影，不等于网格接触。只有 32 个环境被监控，仍可能漏掉未抽中的异常。真实训练中的误停率、渲染耗时和极端卡死尚未实测；进程完全卡死时无法保证在内部写出新保存点，网页会显示心跳陈旧，需从终端中断。起身与滚翻需要另外的分阶段判据，不能沿用“摔倒即异常”的行走规则。

[离线核验](../../records/reports/training-workflow/verification.json)重新读取两组并脚模型与当前官方行走参照的五指令轨迹，共 15 条。两组故障模型的十条轨迹在 4.16 秒仿真时间触发停止，官方参照五条未触发姿态门槛。这只是回放推断的触发时刻，不是新训练的实际停止时间，也不意味着官方参照已通过速度验收。短暂异常、reset 过渡、持续跌倒和非有限数的边界案例通过；实际安装的保存方法保留优化器、归一化和课程计数，并拒绝用损坏状态覆盖已有文件。

### 29.4 如何操作

在现有 VS Code 打开 XGO-Duck 文件夹，使用 Terminal → Run Task。五项日常入口共用任务终端，不改变窗口或面板布局：

1. **状态与当前阶段**：阅读当前阶段与下一次协议。
2. **离线核验观察与停止**：重放已有数据并检查保存契约，不训练。
3. **检查待训练协议**：核对父模型、来源哈希、配置一致性和阶段门槛；当前草案会明确拒绝启动。
4. **训练当前冻结协议**：只有阶段与协议都允许时才运行单组有界训练；结束后返回，不自动开展下一组。
5. **更新全流程报告**：重新生成 HTML 并核对本地媒体。

也可在现有 WSL 终端中操作，使用已经安装的解释器，不重新安装依赖：

```bash
ROOT=/mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck
bash "$ROOT/scripts/training/training_workflow.sh" status
bash "$ROOT/scripts/training/training_workflow.sh" verify
bash "$ROOT/scripts/training/training_workflow.sh" preflight
```

`preflight` 当前提示协议未冻结是预期结果。S1 先补齐既有两组模型的导出与固定评估，确定父模型和下一项单因素改动，再填 `parent_checkpoint`、SHA-256、原配置和阶段证据。不能只把 `status` 改为 `frozen` 就绕过这些检查。当前配置适配器支持官方配置和可达颈部指令；新的髋角奖励、起身和滚翻适配器需在对应阶段实现与检查。所有预检通过后才执行 `run`，在浏览器打开观察页；停止后先查看 `stop.json`，结合模型、画面和固定评估决定下一步。

### 29.5 空间整理

本次扫描了 1996 个项目文本文件检查模型引用。现存中间模型均被已有记录或报告引用，因此未按文件名批量删除。最大的性能追踪 `trace.json` 为 755989112 字节；将其无损压缩为 49913741 字节的 `trace.json.gz`，重新解压流并核对 SHA-256 与总字节数一致后，已删除原始未压缩文件。实际减少文件占用 706075371 字节，约 673.37 MiB；没有压缩 WSL 虚拟磁盘，也没有删除原始视频、评估轨迹、模型或官方资源。

[清理清单](../../records/reports/storage-cleanup/plan.json)与[执行结果](../../records/reports/storage-cleanup/result.json)记录绝对路径、哈希、保留位置和执行时间。原性能分析摘要、逐次计时和算子表继续保留；其中旧摘要标为 `incomplete-profile` 的状态没有被改成成功。需要重新打开完整时间线时，可在临时位置恢复，不改变归档：

```bash
gzip -dc "$ROOT/records/profiling/profile_20261101T151652_s42/traced/trace.json.gz" > /tmp/xgoduck-profile-trace.json
# 用时间线查看器打开；分析结束后删除上述临时文件。
```

页面截图使用的临时浏览器目录 `records/reports/training-workflow/browser-preview-temp/` 的删除被自动审批拦截，返回 `blocked by policy`；该目录暂留并已排除 Git 收录，未计入上述回收量。此项清理尚未完成，见[页面检查记录](../../records/reports/training-workflow/page-check.json)。

后续新训练保留首个完整点、最近两个恢复点、阶段最终模型、所有被评估模型及停止诊断点。其余中间模型先写摘要与清单，再检查引用后清理；不在学习过程中删除正在写入的文件。保存频率与保留数量分别管理，不能为了节省空间降低可恢复性。

<a id="motion-curriculum"></a>
## 30. 动作训练计划

### 30.1 当前范围

本阶段优先完善训练及其可复现记录，真机部署暂不执行。九类动作全部纳入目标：摔倒后站起、站立、前进、后退、左转、右转、左平移、右平移、前滚翻。旧 P0–P7 的未完成项继续保留；下述 S0–S6 是当前训练工作的执行顺序，真机工作后置。各阶段达到门槛后再进入下一阶段，阶段通过不代表整个机器人已经开发完成。

官方源码已提供平地 `Velocity`、`StandUp`、`Roulade` 和 `Roulade-Angle` 任务。[官方说明](https://github.com/LuwuDynamics/xgoduck_rl)与本地固定来源一致：前滚翻单次任务为 7 秒，Angle 方案在原填零观测中加入累计矢状面转角，部署时也要计算这个状态。行走配置已有横向速度指令，不需要凭空另造“左移”动作，但已有训练是否真正学会横移尚未评估。

[本地官方模型结构检查](../../records/reports/training-workflow/official-skill-models.json)确认归档中存在 `xgoduck_walk.onnx`、`xgoduck_getup.onnx`、`xgoduck_roulade.onnx`，三个模型均通过 ONNX 结构检查，输入为 61 维、输出为 14 维。本次没有运行它们的新推理或仿真。相同维度不保证关节零位、动作缩放、观测含义和任务环境相同；必须分别对齐。官方 ONNX 可作效果参照或另行设计的模仿教师，不能作为包含优化器与 critic 的 PPO 保存点直接续训。

### 30.2 分阶段执行

| 阶段 | 实际工作 | 进入下一阶段的证据 |
| --- | --- | --- |
| S0 工作流 | 历史复盘、命名、轻量画面、异常保存停止、空间整理和文档 | 离线检查、页面预览、保留证据完整；真实运行检查列入下一阶段 |
| S1 参照与任务核对 | 先评估已完成的头部对照组；核对官方三种模型的接口；行走扩展至七指令；核对碰撞和执行器 | 可复现的模型来源、命令、轨迹和视频；确定合理父模型与首项改动 |
| S2 站立与六方向运动 | 修复任务定义和不可达目标；单因素、小预算对照；再扩大训练与种子 | 七指令分别通过，姿态与速度同时达标；跨种子确认 |
| S3 摔倒起身 | 官方 StandUp 起点，按俯卧、仰卧、左右侧卧分别评估，再加入随机跌落 | 每种起始姿态均达标，站起后稳定保持，能切回站立 |
| S4 前滚翻 | 官方 Roulade 参照；单独比较有无累计角观测；核对全身接触 | 正向完整一圈、正确接触顺序、稳定落地；不能用侧滚或重复转动冒充 |
| S5 技能切换 | 起身→站立→行走、滚翻→站立→行走，加入有边界的扰动与随机化 | 连续序列通过，切换无大幅目标跳变；分项回归不退化 |
| S6 部署文档 | 整理 ID/关节顺序、机械零位、IMU、控制周期、导出与回退流程 | 文档和接口核对齐全；当前不连接、不刷写、不使能机器人 |

### 30.3 训练策略

先用已有结果消除不确定性，不因新目标增加就重跑已完成的 250 次头部训练。S1 应先修复导出入口的名称识别、补齐最终模型核验，随后对现有两组做同条件评估。官方效果参照也要保持来源固定，不能在比较途中静默换新权重。

S2 先判断物理模型是否允许不合理的脚部穿插，再设计实际关节姿态约束。此前“惩罚请求目标越界”的候选已经失败，不作为默认优化方向。优先研究实际髋角接近限位的惩罚与明确的脚位约束，逐项加入；权重须根据已保存轨迹中的数值量级选择，并在训练前冻结。不可达颈部指令单独评估，不能与髋奖励同时改动后把收益归给其中一项。

每个候选先用 seed 42、相同父保存点、相同环境数和总采样量开展最多 250 次更新的诊断。中途触发异常立即停止并分析；对照组与候选组预算不齐时只比较共同已完成区间并报告失败，不再补跑失败组凑数。通过诊断后，下一段最多 1000 次更新，结束必须固定评估，不自动加预算。最终效果确认使用 seed 42、43、44 和独立评估种子，保留每个种子的结果与离散程度。单次短跑只能筛选，不能宣称已经稳定掌握动作。

起身和滚翻的接触计算、重置方式与行走不同，不能照搬行走的并行规模和停止规则。先测内存、编译和采样成本，再冻结适合该技能的有界训练段。对起身按“离地恢复、接近直立、稳定支撑”检查进展；对滚翻按“启动、正向旋转、接触、落地、站稳”检查。进展持续停滞、接触模型失真、反复错误旋转或数值异常时退出；这些技能的专用监测器尚待对应阶段实现。

当前 WSL 路线继续使用已安装环境。优先减少持续渲染、冗余自动导出和不必要的全量遥测，不删掉执行器模型或碰撞来制造吞吐提升。若新入口的小预算测试仍显示采样瓶颈，再在原生 Ubuntu 用同一来源、环境数、种子、父模型和无连续窗口条件比较。跨系统测试先排除编译热身，报告采样吞吐和观察开销；只有实际测得收益再迁移训练。双系统切换及重启不在本阶段自动执行。

### 30.4 分项验收

以下为项目提出的验收条件，机器可读定义见[动作计划](../../configs/experiments/motion_curriculum.json)。它们将在新试验前冻结；此前五指令结果继续按原协议保留，不回填成新门槛下的结果。

| 动作 | 固定条件 | 主要通过条件 |
| --- | --- | --- |
| 站立 | 指令全零，32 次×20 秒 | 存活至少 95%；平面速度误差 ≤0.03 m/s；位移中位数 ≤0.15 m |
| 前进 | +0.15 m/s | 平面误差 ≤0.08 m/s；指令方向平均速度 ≥0.10 m/s |
| 后退 | −0.10 m/s | 平面误差 ≤0.05 m/s；后退方向平均速度 ≥0.06 m/s |
| 左转、右转 | 偏航 ±0.4 rad/s | 偏航误差 ≤0.20 rad/s，指定转向平均速度 ≥0.28 rad/s；平移误差 ≤0.05 m/s |
| 左移、右移 | 横向 ±0.08 m/s | 平面误差 ≤0.03 m/s；指定横移方向平均速度 ≥0.05 m/s |
| 摔倒起身 | 四类起始姿态各 32 次，10 秒窗口 | 每类成功 ≥90%、总体 ≥95%；到达直立支撑并连续保持 3 秒 |
| 前滚翻 | 32 次，7 秒动作期限，落地后另观察 2 秒 | 成功 ≥90%；正向累计旋转约 2π，误差 ≤30°，接触和落地姿态正确 |

运动指令还需各自达到至少 95% 存活，不能把停止条件的好成绩平均进去掩盖某个方向失败。排除前 1 秒过渡后，双髋同时内收超过 20°与脚位投影小于 −2 mm 的样本比例分别低于 1%；所有动作/状态有限。再检查接触、足底滑移、动作跳变和真实录像。小幅速度指令使用更严格的误差与方向速度门槛，避免原地不动也被判通过。

起身的稳定站立暂定义为躯干高度距声明的站立高度不超过 2 cm、倾斜小于 15°、角速度小于 1 rad/s、双脚支撑且关节/执行器不越界。滚翻不能仅看欧拉角回到零：必须累计有符号旋转并核对接触过程，排除反向翻滚、侧滚、穿模和额外一圈。不同任务使用各自编译模型的限位，不将行走模型的关节范围直接套给滚翻。

### 30.5 如何说明优化来自哪里

每次试验在学习前写出：问题证据、唯一主要改动、预期作用、对照组、父保存点、预算、评价条件及失败判据；脚本注释和记录中的 `changes.json` 保持一致。结束后记录实际变化，失败也保留代表片段和原因。任务配置有多处耦合变化时拆成独立对照，或明确不能归因。

| 改动 | 预测作用 | 当前证据或下一步 |
| --- | --- | --- |
| 速度奖励 3.0→3.3 | 加强速度跟踪 | 只有流程短测，未证明改善；优先级低于任务与姿态问题 |
| 请求目标越界惩罚 | 降低不合理目标幅度 | 已降低请求超额，仍并脚；不能称步态优化 |
| 可达颈部指令 | 消除不可达头部目标 | 配置检查通过，两组学习已完成；S1 评估效果 |
| 实际髋角近限位惩罚 | 在实际姿态上给出更明确的代价 | 已做离线量级分析，尚未选择训练权重或学习 |
| 每 180 秒画面、减少自动导出 | 降低观察和收尾开销 | 软件与历史数据检查通过，实际吞吐改善未测量 |
| 滚翻累计角观测 | 为动作进展提供明确状态 | 官方已有配置；需独立对照，并记录部署观测变化 |

固定评估记录速度、存活/跌倒、姿态、足部滑移、头部误差和动作变化；每个方向、姿态分组、训练种子分别报告。奖励曲线用来解释学习过程，动作成功率和真实视频决定是否进入下一阶段。没有对照证据时，结论写“观察到变化，原因待验证”，不把猜测写成已证实优化。


<a id="goal-review"></a>
## 31. 目标审查

### 31.1 结论

当前总方向保留：完成九类动作的仿真训练，形成能独立复现的全流程报告，真机部署后置。需要优化的是阶段范围、训练准入与验收方式。原方案中的“自动停止”“50/250/1000 次预算”“32 次成功率”和“S0 已完成”不能脱离适用条件解释。本节及[当前目标清单](../current_goal.md)更新第 29、30 节中尚未冻结的后续计划；已经完成的试验、原五指令协议和结果保持原样。

本轮只做源码检查、已有轨迹回放、数值计算及文档调整，没有执行新策略推理、仿真、学习或真机操作。应用目标管理器当前没有活动目标，文件中的 S1a 是执行安排，不代表有进程在自动推进。

### 31.2 优先调整

| 优先级 | 当前问题 | 调整方向 | 落实状态 |
| --- | --- | --- | --- |
| 高 | 当前停止器不能区分父模型已有缺陷与新恶化 | 为修复诊断、合格基线回归、起身和滚翻分别定义判据 | 已写入准入条件，运行代码尚未修改 |
| 高 | S1 包含行走、起身、滚翻及监控，交付过大 | 拆为来源/导出、行走评估、运行闭环；完整起身/滚翻评估后移 | 目标与阶段计划已整理 |
| 高 | 官方原配置未通过时可能反复追加预算 | 保留原始参照，允许明确记录的正确性修复，建立工程基线再做改进对照 | 已区分基线角色，尚未选择新父模型 |
| 高 | 从更新次数看不出实际计算量 | 同时记更新、环境数、总转移、墙钟时间和观察开销 | 草案改为最多 50 次诊断；仍未冻结、不可启动 |
| 高 | S0 的离线验证被误解为整个入口已经通过 | 增加真实端到端观察、保存停止和恢复验收 | 已列入 S1c，未实测 |
| 中 | 静态图无法判断周期、滑移和头部摆动 | 每约 3 分钟一段短片，加状态图、趋势和异常案例 | 方案已记录，周期短片尚未实现 |
| 中 | 32 次试验容易被写成可靠性结论 | 分开发筛查与最终确认，报告成功数、条件和置信区间 | 评价层次已定义，未重算成新动作成绩 |
| 中 | 当前目标混有大量旧启动/暂停指令 | 当前清单只保留有效状态、近期交付和长期范围；历史单独存档 | 当前清单整理为 72 行，原内容保留快照 |
| 中 | 清理按全局文件名保护，过于保守 | 改为完整路径、哈希、实际用途与依赖关系判断 | 目标已记录，本轮不删除产物 |

### 31.3 停止规则与修复窗口

本轮重新读取一条已有并脚对照组的前进轨迹，运行当前 `GaitWatch`。在第 208 个控制步，即 4.16 秒仿真时间，出现持续内收与交叉告警。每次更新采样 24 步，因此若沿用这条轨迹，首次可保存退出的完整边界约为第 9 次更新，对应 4.32 秒。数据和脚本来源绑定见[审查计算](../../records/reports/goal-review/analysis.json)。这不是一次新训练，也不能断言未来策略必在第 9 次退出；它说明当前规则没有给“已知坏姿态的修复尝试”单独定义过程。

建议区分三类情况。数值损坏、接口错误和无界数值发散立即退出并保留有效保存点。父模型原本存在的缺陷先登记严重程度，在预先冻结的短诊断窗口内观察变化，达到预算仍无可解释进展就停下分析。一个已合格策略出现新的明显恶化则尽早停止。有限诊断窗口不是关闭保护，更不是允许无限训练；其长度、恶化指标和退出条件必须在看结果前确定。

若父模型已陷入严重错误姿态，应比较更早的有效保存点、从头建立修复后基线，或单独评估教师模仿的必要性；不能默认最新、更新次数最多的模型就是最好起点。当前 50 次草案仅是工程诊断上限，不能证明某个候选足够训练，也不能凭一段未改善就认定方法永远无效。

起身和滚翻过程中会有预期的倒地或大角度旋转，必须用动作进展、接触和落地阶段判断。当前行走停止器尚不支持这些技能，不能仅修改任务名后复用。

### 31.4 阶段与计算预算

近期只要求先交付一个完整的行走研究流程：已有模型能导出、固定评价能复现、问题能被定位、具体修复可以公平比较。S1a 完成来源和接口，S1b 完成七指令对比与父模型选择，S1c 完成真实运行闭环。起身和滚翻的资料清点可以先做，完整训练和评价按其阶段开展。

S2 内再按站立与前后走、左右转、左右移推进；每次都保留七指令回归记录，防止新动作进步而旧动作退化。先获得最小可用能力，再补齐全动作，最终仍须逐项验收。计划采用行走、起身、滚翻三类策略与显式技能切换，站立和六方向运动先共用带速度指令的行走策略；是否进一步合并应由实际结果决定。

“250 次更新”在 4096 环境、24 步 rollout 下等于 24576000 次环境转移，是原 1024 环境、5000 次训练样本量的五分之一。重新累加此前两组原始逐次计时，250 次分别为 1108.27 与 1138.87 秒，约 18.5–19.0 分钟；这不是完整启动到退出的总时长，编译、检查和收尾另计。因此短诊断不能只用一个看起来小的更新次数命名。

[下一次协议草案](../../configs/experiments/next_training.json)现在以最多 50 次为诊断上限，对应 4915200 次环境转移，仍因父模型、评价证据与适用监控缺失而拒绝启动。实际代码每次仍只允许最多 250 次，原方案提到的 1000 次改为经过复核的累计扩展上限；必须拆段、逐段评价，不把它描述成当前可直接启动的单次任务。没有进展不自动延长，有进展也要记录继续的证据。

对照同时固定父模型、学习起点、环境数、样本量、更新数和评价条件。共用父模型微调只回答“从这一点开始，这项改动的效果”，不能冒充从头训练的全程对照。不同环境数的吞吐测试与学习质量比较分开记账。多个训练种子留给有希望的最终候选，不让所有初步想法都承担完整确认成本。

### 31.5 评价与时间定义

训练中的探索、随机化和奖励曲线不能替代固定条件下的策略评价；评价环境的 wrapper、episode 长度和观测处理也要核对。采用独立测试环境与多次运行是相关实现文档建议的基本做法；这里沿用现有框架，不因此新增 SB3 依赖。[SB3 评价说明](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

32 次/条件继续用于开发筛查，最终候选拟用更大的独立场景集确认，并分别报告训练种子、动作条件、成功次数和区间。使用本轮脚本的双侧 95% Wilson 计算，31/32 次成功的区间约为 84.3%–99.4%；即使 32/32 次成功，下限也只有约 89.3%。因此“样本内通过 95% 门槛”不能写成已经证明真实成功率至少 95%。方法来源见[NIST 比例置信区间](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)。计算假定同条件下试次可按独立二项样本解释；它不包含仿真模型误差，也不能把连续视频帧当成独立试次。

最终确认拟以 128 次/条件为起点，在预算与场景核对后冻结；仍要报告区间，不能认为换成 128 就自动证明可靠性。开发集用于选方案，留出场景用于选定模型后的最终评价，不反复用同一留出成绩调整奖励或选择保存点。

当前姿态提案也需要按动作核对。左右脚位在躯干横轴上的投影只是几何代理指标，不能独立证明碰撞或穿模；转向和横移的合法脚位关系应结合实际接触与模型行程判断。原“所有动作均用 20°、1%”只是尚未冻结的提案，本轮改为要求先做逐指令校准，运行中的行走停止阈值未悄悄放宽。

另有时间定义问题：固定来源的官方起身任务包含 2 秒冻结与 6 秒起身，训练 episode 为 8 秒；滚翻训练 episode 为 7 秒。如果评价器直接沿用训练 timeout，就可能在观察站稳之前重置。拟保持原动作完成期限：起身在释放后的 6 秒内完成，再最多观察 3 秒持稳，评价最长 11 秒；滚翻在 7 秒内完成，再观察最多 2 秒，评价最长 9 秒。只扩展评价观察窗口并明确 reset 处理，不擅自改变训练任务时长。相应评价器仍待实施。

### 31.6 观察、实现与存储边界

周期画面应同时回答“现在在哪个阶段”“动作是否在改善”“异常是什么”。单张图只能展示姿态，不能辨别稳定周期、短暂滑移、反复蹦跳或头部摆动。拟采用每约 180 秒采集 2–3 秒仿真时长的真实短片，保留即时图、变化趋势与被触发的异常案例。不得为录制额外推进训练环境，也不能把单独的确定性重放标成实时训练。拟定观察开销目标为不超过训练墙钟时间的 10%，这是待测工程目标，不是本次测得的收益。

当前新入口还有明确的功能边界：只允许 `official` 和 `reachable-neck`，`reward_edits` 非空时拒绝启动；它在应用候选改动之前，把父环境配置与官方配置比较。因此不能认为它已经支持新的髋奖励或继续训练一个带继承修改的父配置。后续只补当前阶段所需的适配与验证，优先复用已有同起点核验、录像和导出代码。本轮未改写 runner 或监控实现，避免把规划修改当成代码已交付。

空间管理也应按用途判断。当前工具把所有文本中出现的 `model_编号.pt` 文件名加入同一个保护集合，某次试验出现 `model_0.pt` 就可能保护其他运行中的同名文件；历史清单记录也被视作永久引用。因此“本轮没有可删模型”是保守扫描的结果，不等于所有中间模型都必须永久保存。后续应按完整路径和哈希核对真正的依赖，保留最终/已评估模型、恢复父模型、代表性失败点与审计输入，对只有清单记载的可替代中间件保存摘要和删除记录后再清理。本轮没有进行新的删除，先前被自动审批拒绝的临时目录也未重试。

### 31.7 本轮落地

当前目标清单由原先 306 行整理为 72 行，近期顺序、未实现项和长期范围分别明确；原清单与规划 JSON 的内容和 SHA-256 存于[规划输入快照](../../records/reports/goal-review/planning-inputs.json)。[动作计划](../../configs/experiments/motion_curriculum.json)补充了分段目标、策略分工、评价层次、时间定义、监控适用条件与实现缺口。运行状态继续保持未启动学习。

离线计算可用现有环境复现：

```bash
ROOT=/mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck
PY=$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
"$PY" "$ROOT/scripts/validation/audit_goal_plan.py"
bash "$ROOT/scripts/training/training_workflow.sh" preflight
```

第一项只读取已有数据并重算；第二项目前应返回 `PREFLIGHT_STOP`，因为草案未冻结。近期的有效目标是完成 S1a 的已有模型来源与导出核对，再按证据推进 S1b、S1c；本次目标审查没有自动创建或启动持续训练任务。


<a id="adopted-goals"></a>
## 32. 当前目标

### 32.1 总目标

分阶段完成 XGO-Duck 九类动作的可靠仿真训练，并交付可独立复现的全流程报告，覆盖环境部署、官方模型参考、训练方法、修改原因、对照效果、失败分析和后续真机迁移。九类动作包括摔倒后站起、站立、前进、后退、左转、右转、左平移、右平移和前滚翻。

本轮将前节审查结论纳入正式项目目标，更新了当前目标清单、动作阶段计划与项目状态；没有启动训练、推理、仿真或真机操作。训练策略仍须在对应阶段完成实现、核验与具体协议冻结。完整执行清单见[当前目标](../current_goal.md)。

### 32.2 近期交付

当前优先完成 S1，建立有证据的行走训练流程：

| 顺序 | 交付 | 验收依据 |
| --- | --- | --- |
| S1a 来源与导出 | 已完成头部对照组、修改组及一个固定官方行走参照的模型清单；补齐导出与接口核对 | 来源、配置、哈希可追溯，导出数值一致；观测、关节、零位、缩放、时序及物理差异有记录 |
| S1b 行走对比 | 站立、前后走、左右转、左右移的逐条件指标与视频；有依据的父模型和单因素改动 | 有效旧结果继续使用，缺失条件补齐；每项失败和未知原因明确 |
| S1c 实际运行 | 适用于修复阶段的异常监控、周期短片、保存停止与恢复 | 在单个可见终端完成小预算端到端验证，观察开销有实测，未将离线检查当成实际运行成功 |

稳定行走尚未验收。S0 已通过的离线准备保留，真实运行验收仍待 S1c。已有两组头部训练不重复学习；起身、滚翻可以先清点来源和接口，其完整评估与训练按后续阶段开展。

### 32.3 动作推进

先训练并验收站立及前后走，再扩展左右转向、左右平移，每一步都对七种指令做回归。随后处理四种倒地姿态的起身、前滚翻和稳定落地，最后验证起身/滚翻到站立、行走的技能切换与扰动适应。

优先采用行走、起身、滚翻三类策略及显式技能切换；站立与六方向运动先共用带速度指令的行走策略。真机接口约束提前研究，实际连接、标定、刷写和使能继续后置。

### 32.4 共同完成标准

- 每项改动都能回答改了什么、为什么、预期如何、实际如何；保留源码说明、固定配置、对照指标和真实画面。没有实测收益就不写成已优化成功。
- 首次工程诊断草案最多 50 次更新；通过相应准入后再实施。每次运行不超过 250 次；每轮经复核的扩展累计最多 1000 次后重新复盘。1000 是本轮预算边界，不是全部技能的训练总上限；训练是否充分由动作验收决定。
- 数值损坏、已知父模型缺陷、新出现的行为恶化分别处理；起身和滚翻使用各自的阶段判据。异常或预算结束后保存有效证据并停止，不自动重试或盲目加预算。
- 目标观察方式为每约三分钟一段 2–3 秒真实短片，加状态图和指标趋势；新增短片与监控仍待实现和测量。训练使用单个可见 VS Code 终端，不调整用户窗口布局。
- 开发筛查与最终确认分开；逐动作、逐训练种子报告结果，最终确认使用独立场景并给出成功数与不确定性。固定评价通过才能进入下一阶段。
- 主报告同时保留部署环境的过程和后续真机步骤；当前只做仿真相关工作。无用产物先总结、核对用途与引用，再按清单清理，保留关键模型和证据。


<a id="official-initialization"></a>

## 33. 官方策略初始化与中断恢复

以下保留初始化、精度检查和中断恢复时的原始过程说明，实时结论以主报告为准。


2026-11-02 继续执行后，先启动官方策略初始化的工程试验。官方行走 ONNX 的网络结构与当前训练网络相同，权重和观测归一化可直接读取；18,432 组输入上的最大导入误差约为 3.55×10⁻⁶。该文件没有价值网络、优化器或归一化历史样本数，因此只恢复策略，归一化暂时固定，其余状态新建。首轮预算 50 次：10 次价值网络预热和 40 次小步长 PPO 更新，保留严格异常停止和周期短片。初始标准差为 0.05 rad，学习率为 0.0001。它与原来的失败模型续训不同，效果需另行评价，不据此声称某一奖励带来改善。见[协议](../../configs/experiments/official_actor_bootstrap.json)、[CPU 核验](../../records/reports/official-actor-bootstrap/actor-import-check.json)。首次配置预检在创建仿真前发现不可直接替换冻结配置字段，已改为 `dataclasses.replace`，未发生学习，见[修复记录](../../records/reports/official-actor-bootstrap/preflight-repair.json)。

顺序为 **S1 来源/评估/运行闭环 → S2 站立与行走 → S3 跌倒起身 → S4 前滚翻 → S5 技能切换 → S6 真机迁移资料**。采用行走、起身、滚翻三套策略，加显式切换；站立与六方向运动先由同一速度策略承担。

1. 先评估已有模型。根据姿态、跟踪、失败模式和来源选择父模型，不按文件日期选择。
2. 新修复先做最多 50 次更新的工程诊断，用于判断实现和趋势，不代表足以收敛。
3. 每次运行最多 250 次更新；同轮经复核扩展累计最多 1000 次。每段结束评估后再决定继续、修改或停止，不自动追加。
4. 先站立与前后走，再转弯、平移；每次检查七条指令，避免旧动作退化。
5. 有希望的方案再扩展训练种子 42、43、44，并在独立场景确认。留出条件不用于反复调参。
6. 起身先分俯卧、仰卧、左右侧卧，再加入随机跌落；滚翻检查正向完整旋转、接触顺序和稳定落地。

4096 环境、每次 24 步时，50 次约 491.52 万次环境转移，250 次约 2457.6 万次。更新数、样本量、墙钟时间和观察耗时一起记录。

起身训练为 2 秒冻结加 6 秒恢复，评估拟再观察 3 秒持稳；滚翻训练期限 7 秒，评估拟再观察 2 秒。需防止训练任务 timeout 在持稳检查前重置，两个评估器均待实现。完整条件见[动作计划](../../configs/experiments/motion_curriculum.json)。


本轮先完成 50 次初始化短训，其中 10 次只更新价值网络、40 次更新策略；共 4,915,200 个环境转移，学习段用时 226.62 秒。前 10 次策略权重逐项保持一致，策略观测归一化始终未变。检查点计数、Adam 的两类更新次数和两段真实录像均已核验；导出模型在 8,960 组已有观测上与检查点一致，最大误差约 2.98×10⁻⁷。这些结果证明初始化与导出链路可用，步态是否达标要由独立固定指令评估决定。

启动时曾因框架默认 TF32 精度超过严格推理容差，在零仿真步、零学习更新处退出。随后保持同一容差，对同一批实际观测比较 TF32 和完整 FP32：最大误差分别为 6.63×10⁻⁵ 与 2.38×10⁻⁷，后者通过；本轮学习也使用完整 FP32。失败运行、修复说明和新运行均分别保留，见[启动审查](../../records/reports/official-actor-bootstrap/startup-review.json)、[训练核验](../../records/reports/official-actor-bootstrap/training-verification.json)。

![本轮训练后半段](../../records/experiments/duck_walk_official-actor-bootstrap_candidate_s42_20261102T180647/frame-step-985.jpg)

[训练开始短片](../../records/experiments/duck_walk_official-actor-bootstrap_candidate_s42_20261102T180647/clip-step-0.mp4) · [训练后半段短片](../../records/experiments/duck_walk_official-actor-bootstrap_candidate_s42_20261102T180647/clip-step-985.mp4)。均为环境 0 的真实训练过程，包含随机动作与可能的 reset，不能替代固定指令评估。

### 官方起点的评估

| 指令 | 官方速度误差 | 初始化后速度误差 | 初始化后指令方向速度 | 存活 | 首秒后双髋持续内收 |
| --- | --- | --- | --- | --- | --- |
| 站立 | 0.00183 m/s | 0.00169 m/s | — | 32/32 | 0 |
| 前进 | 0.12690 m/s | 0.12515 m/s | 0.10917 m/s | 32/32 | 0 |
| 后退 | 0.10449 m/s | 0.10283 m/s | 0.02908 m/s | 32/32 | 0 |

前进与后退均未通过既定运动标准，变化幅度不足以证明明确优化；这轮建立的是保留正常姿态的可训练起点。下图使用相同前进指令、环境 0 和 2/8/16 秒时间点；中行是既有失败模型，下行是官方初始化后的模型。姿态变化不能归因于单个奖励，因为初始化、HOME 和学习设置均已声明改变。

![初始化姿态对照](assets/official-bootstrap-keyframes.jpg)

<video controls preload="none" src="../../records/evaluations/custom_onnx_reference_20261102T182248_s1001/videos/forward_env0.mp4"></video>

后续 150 次适应更新，在第 99 次更新后的 `state.json` 原子发布处出现短暂权限错误。最后完整检查点保留本段 76 次；23 次未保存计算仍计入预算，因此只恢复剩余 51 次。恢复已完成，累计实际计算 200 次，其中 50 次为初始化；最终模型包含 177 次完整更新的状态。记录写入增加最多 1 秒的占用重试，持续故障仍退出。故障注入、最终检查点和导出一致性均通过；详见[恢复审查](../../records/reports/official-actor-bootstrap/io-recovery-review.json)、[最终核验](../../records/reports/official-adaptation/training-verification.json)。

本段结束后的继续条件已提前固定：保持姿态与存活、站立通过，前进或后退至少一项速度误差降低 0.005 m/s，另一项恶化不超过 0.01 m/s，偏航误差恶化不超过 0.05 rad/s。条件不满足就结束直接续训路线，分析任务与奖励，不继续追加预算。



## 2026-11-02 至 11-03：速度统计、指令响应与分组仿真

速度时间统计学习从同一完整父 checkpoint 出发，对照和候选各 50 次更新、4,915,200 个转移。两组都维护世界速度向量的 EMA，候选只将 XY 奖励改为时间常数 0.2 秒的统计，Z 项保持瞬时。候选未改善前后走，后退几乎停止，关闭本轮；没有追加训练。见 `records/reports/planar-filter/analysis.json`。

随后固定当前官方模型 SHA-256 `5eaae52c92df20139a779b4656e49ab824bb501fdd219e1241bcda91386fd75a`，完成 19 档、304 个首次 episode 的动态指令响应。小输入响应弱，增大输入后出现运动，但不等于所有动作通过。原观测噪声、延迟及物理随机化保留。记录全部轨迹、七段真实短片和主机分段耗时，独立推理重放最大误差为零。协议 `configs/experiments/command_response_probe.json`。

输入映射把前后、平移、转向的策略输入分别乘 1.4、3、1.5，外部目标不变，零学习更新。跨进程逐点站立一致性门槛未通过，按条件边界停止，已完成五项、80 个场景；左右移未执行。初始 qpos/qvel、观测和首个动作相同，微小分歧从第一物理步开始。CPU 同输入零映射推理 240 次完全一致；尚未唯一查明物理分歧原因。原失败记录保留，不按部分正向结果晋级。

新的独立前后走确认在运行前固定功能筛查，并同时保留旧瞬时指标。seed 1005，三项各 32 个首次 episode、20 秒，同批 96 环境共享 1000 个控制步。三项各 32/32 通过新功能筛查，实际前进 0.1518、后退 0.1221 m/s，未出现髋内收或实际关节超容差；旧瞬时前后误差仍未通过。采集与保存共 86.9 秒。真实前进录像为全局环境 67，三秒 76 帧，已在原 VS Code 编辑区播放，未额外推进物理。证据 `records/reports/straight-parallel/analysis.json`。

为复查左转误差，先从已有轨迹统计偏置、波动和策略 gyro，再冻结一次转向反馈/平移筛查。反馈仅使用原策略带噪声和延迟的 gyro；参数及继续条件见 `configs/experiments/turn_feedback_probe.json`。当前在同一可见 VS Code 终端执行，完成后按 `records/reports/turn-feedback/analysis.json` 判定；本段不追加学习。

复算及准备均使用现有 Python，不安装依赖。运行学习或 GPU 仿真只通过原 VS Code 的 Terminal → Run Task…，所有任务 `panel=shared`、`instanceLimit=1`；不调整窗口大小和布局。独立 CPU 命令示例：

```bash
PY=/home/atlas/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python
ROOT=/mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck
"$PY" -B "$ROOT/scripts/validation/review_velocity_timescales.py"
"$PY" -B "$ROOT/scripts/validation/review_turn_tracking.py"
```

`--freeze` 仅用于从未执行的新协议，已经冻结的文件禁止覆盖，已完成任务也禁止重新启动。换机复查先恢复同哈希源码、模型和依赖，已有轨迹可独立核验；需要新的物理重复时另建协议、命名及证据目录。

### 七指令确认与可复现练习

2026-11-03，固定控制器完成开发 seed 1007、1008 及一次留出 seed 2001。每个种子七项各 32 个首次 episode、20 秒，全部 672/672 功能通过，新增身体姿态和脚中心异常比例为零。原瞬时平面误差的运动失败仍保留。这是官方原权重加输入映射与转向 gyro 反馈，零学习更新，不能写成三个新训练模型。完整协议与轨迹哈希见 `records/reports/seven-command/confirmation-summary.json`。

已建立 `artifacts/simulation/duck_walk_command-gyro_reference_20261103/`，保留官方 ONNX 原字节、三个控制器模块及契约。用户练习通过 VS Code 的“新建七指令练习（不训练）”，使用新名称和开发种子；CPU 预检通过，未额外运行演示 GPU 批次。不覆盖已完成尝试、不再次消费留出、不改变工程目标。

全部已保存录像覆盖到 258 段、516 张真实帧，其中训练 87 段、评价 171 段。新增 42 段均完整解码，反馈/确认标签核查后重做 27 张卡片；两段部分训练片段保留实际时长。抽查三张新全尺寸卡片及四个总览/重点图裁剪，不声称手工查看全部卡片。各组真实全局环境编号已经标注，完整总览、图册与抽查哈希见 `records/reports/stage-media/`。

### 滚翻输出记忆对照的关闭

首次封装少导入 MODEL 常量，在创建环境前退出，零仿真帧，原协议、日志和失败记录保留。独立恢复适配只导入原模型常量，没有覆盖已冻结源码；配置和全局名称检查通过后在同一可见终端运行剩余32+32次。对照28/32、候选24/32，配对4次变成功、8次变失败，候选关闭，不追加。原输出失败为2次头顶接触缺失、2次支撑转角不足；候选为3次和5次；均无实际关节越界或头部深度超限。真实环境0对照图保留几何/支撑转角，数值按全部32场景复算。

随后审查源码：已有支撑旋转门槛与头顶接触 latch，头部非顶面接触仍获得30%的支点奖励。新短训仅比较这一基础奖励是否保留，两组共同保留22°重置、按模型24° HOME适配坐标。CPU 1536条导入比较、冻结归一化、带偏置/不带偏置坐标及奖励夹具检查通过。新的1024环境、每组50次更新（14critic、36actor）协议已冻结；固定学习率2e−5、显式新建std0.05、每次更新检查固定真实观测上的动作偏离，动作后段另查接触/持稳。它不是官方PPO续训。独立评价为seed1004、原模型及两组各32次，不消费留出；具体执行以 `records/reports/roll-top-contact/` 为准。

## 2026-11-03 滚翻头顶奖励短训

原奖励与仅头顶奖励两组各完成 50 次更新，14 次拟合 critic、36 次更新 actor，均保存完整点并导出。独立 CPU 从原始轨迹复算：同场景官方 27/32、原奖励 30/32、仅头顶 28/32。修改组按事前门槛否决，不追加。本轮保留所有模型、标量、轨迹和录像；原奖励 30/32 仅作为独立开发筛查线索。新筛查 seed 1005、两模型各 32 次，零学习，CPU 协议检查通过。

## 2026-11-03 滚翻支撑、切换与角度输入

头顶奖励试验原奖励30/32、修改28/32、官方27/32；独立物理种子1005原奖励27/32、官方25/32，均按各自门槛关闭。160次轨迹分解的23次失败中，21次已有头顶接触，缺失有效旋转来自离地，侧倾折扣为零。完成奖励门槛新对照官方28/32、对照27/32、候选28/32，仍未通过，未追加。

完成门槛候选先在零步精确参数比较失败，两个逆惯性派生字段分别相差3、6ULP；源码与数据检查后，另立一次仅比较派生量的启动恢复，其余捕获参数仍精确，物理值不改、剩余50次候选预算不增。旧失败源码与记录保留。

零学习站立准备对照中，32/32准备通过，直接30/32、准备后29/32，未改善。两项各自的原始动作、完整影片和连续物理边界通过独立核验；父汇总错误比较了直接触发前与准备后摩擦预算，原停止保留，[CPU结案](../../records/reports/roll-stand-preparation/postprocessing-review.json)解释BAM每步更新dof_frictionloss，不把不同时点字段相同当作配对要求。没有重跑该物理试验。

官方角度输入审计首次因两个地面工厂lambda对象地址不同停止；[首个源码快照](../../records/reports/roll-angle-input/initial-inspection.py)和界面保留。核对同一函数代码、闭包和实际XML后再进行语义比较，物理及奖励一致。2560条真实输入的保持策略初始化最大误差8.34e-7，原第55项标准差为零、分母0.01，新角度列须先置零并明确归一化。已有160次轨迹的重力角重建通过，仅为离线数学与合成噪声验证；新对照仍为草案，没有学习。

图册增量解码14段新影片，当前全部274段、548帧；新的连续六帧图和完整影片均来自预选环境0，注明29/32整体结果。关键图、全部新增卡片和总览首尾已目视检查，源码与影片来源哈希保留。


## 2026-11-03 角度输入对照

官方角度任务的 CPU 适配审计通过后，新增同任务、同 critic 的 actor 角度可用性对照。两组各 1024 环境、50 次更新，其中 14 次仅训练 critic，原模型与固定判据不变。前置预检确认梯度屏蔽跨越冻结/解冻仍有效、其他列初始梯度一致，逐步重力角度与离线结果一致。操作在原 VS Code 的共享任务终端完成，窗口尺寸与布局保持不变。

首次冻结任务完成源模型 seed 1008 的 28/32 评估后，在训练零步检查处停止。该检查误将官方速度指令第 48–50 项也要求为零；官方配置保留 20% 前进指令采样，首批 32 个输入中有 5 个为 0.3。失败为 0 步、0 更新，没有生成训练短片或启动候选，原协议和源码均保留。检查失败见[原父记录](../../records/reports/roll-angle-feature/progress.json)。

随后另建[启动修复协议](../../configs/experiments/roll_angle_feature_startup_recovery.json)：只把零填充检查起点从 48 改为 51，原实际输入不变，其他零步核验、学习预算和成功标准保持原样。CPU 预检核对整段函数，除该检查和阶段标识外一致；头部和其余填充的错误反例仍被拒绝。源模型结果直接复用，不重评、不追加样本。两组仍各最多 50 次更新，运行名包含 angle-feature-recovery、组别、种子和时间。

对照组完成 50 次更新，14 次预热 actor 完全未变，第 55 列权重最终仍严格为零。288 条真实非零角度输入上的导出最大误差为 7.15e−7 rad。第 700–850 个物理控制步的三秒真实训练录像已经在 VS Code 播放并截图，开始时完成 15 次 actor 更新，不额外推进仿真。固定评估为 29/32；候选随后完成 50 次及导出检查，含真实非零角度的 288 条输入最大导出误差 5.96e−7 rad，512 个角度列权重非零。固定评估为 27/32，配对新增 3 次成功、丢失 5 次成功；原模型 28/32、对照 29/32，候选未通过继续条件，关闭、不追加。全部七段新增录像已解码、逐段查看标注，合并总览现为 281 段、562 张真实帧。


## 2026-11-03 共同物理环境离线准备

关闭角度输入学习后，没有追加训练。新增共同物理配置和 CPU 编译核验，仍在同一可见 VS Code 终端执行。统一滚翻全身碰撞机器人、4 ms 物理步长、20 ms 控制周期、7.7 V、电压压降增益 0.1、4 ms 延迟及每环境 200 接触容量；物理随机化与推扰暂关闭，加入共同传感器并集，各技能观测与动作接口仍单独检查。

初次纯配置构建因滚翻模板与当前对象同址，筛选事件后缺失 foot_friction 而停止。改为先深复制模板，旧源码与截图保留。下一次比较的 27 组数值字段一致，但原始 XML 的十五个可选 STL 类型声明不同，严格文本检查停止；保存原文件后补查网格顶点、面、法线、纹理坐标及其余 XML。最终 33 组数值字段逐项相同，其余 XML 完全相同，保留原始文本不一致的事实。[核验结果](../../records/reports/common-plant/source-contract.json)明确为 0 物理步、0 更新、未初始化 Torch CUDA。

下一步为[共同环境零学习筛查草案](../../configs/experiments/common_plant_screen_draft.json)，尚未执行。必须先核对完整场景、名义初态、各模型坐标与原始动作历史、控制周期和实际接触容量，再冻结协议；可靠滚翻、连续切换、扰动和真机准备仍保留在长期目标中。


## 2026-11-03 共同场景与起身碰撞诊断

共同场景按事前协议完成 384 个 episode、308800 个控制转移。行走七指令通过功能门槛（223/224），起身四姿态未通过，滚翻 30/32。全部三技能先通过实际 GPU 零步接口、完整场景字段及名义参数检查，原始判据和失败均保留。固定回归没有学习更新、没有消费留出；原命令、时间、输出和退出码见 [命令账本](../../records/reports/common-plant/screen-commands.jsonl)。

CPU 复盘发现：后退环境 184 几乎未运动；侧倒组 33 次在等待期变成仰卧；数次真正的仰卧恢复失败中，颈部约 −135°、躯干约 32°。CPU 接触几何只更新已保存的姿态，不积分，显示共同规则会计算头脚接触，而原起身掩码忽略这些接触。该阶段根部水平坐标未保存，所以几何复盘明确标作平地原点重建，不能当作 GPU 实测力或新成功率。

### 问题与处理

| 现象 | 原因与处理 | 结果 |
| --- | --- | --- |
| 新碰撞对照在零步拒绝继续 | 启动检查默认俯卧，正式条件为仰卧。保留原协议和错误，单列启动修复，让预检选择声明姿态 | 原任务 0 个控制转移；新任务仍总计 35200 个控制转移 |
| 原始 `qpos` 与报告四元数差 1 ULP | 实体缓存归一化了四元数；直接逐分量的 1e−7 检查过严。改为核对归一化旋转，并保留原差异 | 最大方向差 7.15×10⁻⁸ rad，动作判据未改，不重跑物理 |
| 下一轮 CPU 关节夹具缺属性 | 当前库读取 `joint_pos_biased`，夹具只提供 `encoder_bias`；补齐与真实实体相同的属性 | 适配代码未改；真实输入导入与冻结边界复核通过 |

碰撞对照只改两个掩码字段，三组 CAD 排除、完整机器人、物理周期、执行器、观测配方、IMU 和初态一致。种子 1010 仰卧组为完整碰撞 30/32、原掩码 32/32，配对改善 2、退化 0。两次失败在最后 100 个控制步均有头脚、头髋约束接触。放宽碰撞只用于定位问题，不作为最终模型配置。[独立接触核对](../../records/reports/getup-contact-release-recovery/independent-contact-check.json)逐步验证完整 `qpos` 与高度、姿态及关节一致。GPU 对照仍使用原起身判据：2 秒冻结，第 8 秒前开始 3 秒持稳，总长 11 秒。

本轮先后在同一个 VS Code 共享终端选择“起身碰撞规则对照（不训练）”“起身碰撞对照启动修复（不训练）”和“重放已保存的起身失败（不训练）”，没有调整窗口位置、大小或面板。诊断的 [命令账本](../../records/reports/common-plant/getup-release-recovery-commands.jsonl)与[重放账本](../../records/reports/getup-contact-release-recovery/replay-commands.jsonl)记录真实命令、TTY、时间和退出码。历史失败入口归档到任务历史，不通过重跑它们继续训练。

六帧重放图保留了完整原始水平坐标和关节位置，只更新几何、没有积分；它另列为保存状态重放。两段原始环境 0 录像收录为 N163、N164。全部原始录像总览为 295 段、590 张标注帧；六帧重放图不混入视频数。[主报告](development_report.md)给出可读结论，[图册](stage-video-gallery.html)提供原录像。

下一轮尚未启动学习。新坐标适配保持模型约 20° 的观测与释放后目标，同时保留约 22° 物理重置及前两秒冻结，原始动作历史为零。CPU 在 1440 条真实输入上导入误差最大 1.19×10⁻⁶，释放边界和带偏差关节转换通过。价值拟合、专用接触传感器、精确预算及实际 GPU 零步一致性仍须先完成；[草案](../../configs/experiments/getup_collision_adaptation_draft.json)不可执行。


## 起身价值预检准备

CPU 坐标重查通过：1440 条实际观测最大 ONNX 导入误差 1.19e−6；冻结边界保留调用者采样值。有限回报、剩余时间、960/64 环境划分、训练专用 RMS 及四个原生头腿传感器配对通过。两例来自既有完整物理轨迹，仅调用 mj_forward，无新 episode。

首次冻结的 JSON 配置序列化失败，保留在 `records/reports/getup-adaptation/cpu-freeze-format-failure/`；无 GPU、物理或优化器步。配置记录改用工程既有 YAML BaseLoader 表示后，冻结协议 SHA-256 为 `2555fa6d1e625c83bd521aa024224fc2022d5d4dfb751fa5355022aa7c1d072f`。

CPU 复查命令：

```powershell
wsl -d Ubuntu-24.04 -u atlas --exec /home/atlas/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/validation/check_getup_value_bootstrap.py
```

正式运行只从 VS Code 的“起身价值预检（策略固定）”任务启动；不从工具另开隐藏学习。

实际首次可见启动 `duck_getup_value-bootstrap_reference_s42_20261103T052758` 在 `torch.testing.assert_close(Tensor,TorchArray)` 处失败，runtime 记录 0 控制步、0 物理步、0 更新。源文件和失败记录保持原字节。独立恢复包装只在启动比较期间将确切的 TorchArray 类型复制成同精度 Tensor；CPU 正例和故意不等的反例均验证，原容差不变。恢复协议 SHA-256：`d861488b7a46a0f0d03520e4d56e42fd77b5d62015f5ea7128df958c93bb2401`，仍为原 409600 转移与 50 次 critic 梯度更新。

价值预检恢复实际运行 `duck_getup_value-bootstrap_reference_s42_20261103T053111` 已按预算结束：409600 控制转移、50 次 critic 小批量梯度更新、actor 更新 0。采集 43.5631 秒，全流程 55.7193 秒。独立 CPU 复算验证解释方差 0.828415961、均方误差 0.024214788；四请求方向解释方差均为正，预定门槛通过。16 次同瞬时原生传感器/物理接触审计一致。 actor 初末逐位一致，输入归一化和 RMS 缩放只用 960 个训练环境；64 个验证环境不参与拟合。两个真实片段、N165/N166 关键帧和曲线均已检查。

复算命令：

```powershell
wsl -d Ubuntu-24.04 -u atlas --exec /home/atlas/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/validation/verify_getup_value_startup_recovery.py
```


## 动作录像时长调整

用户指出起身只看 3 秒无法判断完整过程，先更新目标和工程状态，再新增 `action_recording.json` 与 `ActionVideo`。起身/滚翻按实际 episode 从起点到终点拍摄，自动重置前保存末帧；行走/站立4秒，切换录完整声明序列。CPU 手动生命周期检查覆盖实际8/11秒起身、7/9秒滚翻、途中等待和重置前捕获。旧冻结的录像器源码和3秒原片保持不变，新学习入口必须接入新录像器。

在原可见 VS Code 共享终端运行“完整起身回放（8秒，不训练）”，从价值预检完整保存的 qpos/qvel 生成环境0和环境20的0–8秒视频与十帧过程图。环境20按前32个保存环境中末段合格相位头腿接触最多选择；属于诊断个案，不代表新固定评估。两段各201帧、25fps、8.04秒；CPU完整解码和逐帧源姿态哈希核验通过。无插值、无新增物理或学习步，原始录像297段另计。

复算：

```powershell
wsl -d Ubuntu-24.04 -u atlas --exec /home/atlas/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/validation/check_action_video.py
wsl -d Ubuntu-24.04 -u atlas --exec /home/atlas/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python -B /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/validation/verify_complete_getup_video.py
```


## 起身成对策略协议冻结

真实 MaskedPPO 优化器 CPU 检查通过：32个保存环境，先完全冻结、再主动更新、再带已有 Adam 动量冻结，策略及动量在冻结期保持原值。固定归一化/探索、有限 GAE、独立策略/价值 Adam 为两组共同适配。配置仅有 `late_head_leg_contact.params.enabled` 一个差异。协议 SHA-256 为 `01ae3e0e097a4e1ee67664ea327add5afd6f446daf063aeb43410d37aae38b02`，每组1228800转移，独立评价105600转移，新开发种子1011。无最终留出、无实物操作。


## 颈部归位顺序

在原VS Code共享任务终端顺序完成共同接口、只归位、归位后站立三组固定模型评估。种子1015，三组各320次完整11秒，528000个控制转移、零学习；成绩291/320、288/320、287/320，均未晋级。后两组末段头腿接触清零，失败仍趴低，故关闭本轮，不追加。

过程依次为CPU配置与边界核验、冻结66份来源、每组实际GPU零步及恢复检查、550步首次episode采样、原评分与最后8–11秒持稳独立重算、全轨迹诊断、12段原片解码和可视检查。采样录像各约109–111秒。真实终端和末帧见[阶段索引](../../records/reports/getup-neck-home-order/README.md)，实现与复算命令见主报告第7.7.9节。

录像统一截取3秒会遗漏冻结后的起身和持稳。本阶段采用完整动作时长，保留25fps、0–11秒两端状态；旧短片保留原来源，保存状态回放单列。CPU检查确认拍摄等待动作起点、自动重置前保存末帧、中止片段明确未完成。


## 全身目标和共同接口价值起点

逐关节保存轨迹审计发现：共同接口失败样本末段颈部越界幅度中位数约1.22rad，而早期成功动作也有越界请求。新训练方向改为5秒后全身目标可达性，原奖励和价值目标相同。

为匹配共同噪声、末段目标处理和0.10rad探索，新建固定行为的价值采样。首次在1步后因日志字段读错停止，1024个转移、零梯度保留。只读别名经过CPU张量身份、反例和配置等价检查后，单独协议采集400步×1024，50次价值更新、行为0。两次开销合计410624个转移；独立验证EV0.82861、归一化MSE0.02473，价值门槛通过。完整8秒片段、真实终端截图和逐项目标重建见主报告第7.7.10节。原协议与失败记录不覆盖。

新辅助目标的解析梯度、零系数与原优化器逐位一致、正系数仅改变行为网络、冻结期Adam保持不变均通过CPU检查。训练协调器尚未冻结，尚无新行为学习效果。全部阶段仍按原目标继续。


## 起身末段可达目标训练与完整动作录像

2026-11-03，按既有完整目标在原VS Code同一个共享可见任务终端执行，未改变窗口或面板尺寸。冻结协议为`configs/experiments/getup_late_feasibility_pair.json`，SHA-256为`d06c575d9e9044cc13cf5c1202bac09b09474c53ff7961749178d84f331ff645`，76份来源一致；首次执行从CPU和实际GPU零步检查开始。

| 步骤 | 实际结果 | 记录 |
| --- | --- | --- |
| 原奖励训练对照 | 50次rollout、1228800转移、800次actor小批量梯度 | [证据](../../records/experiments/duck_getup_late-feasibility_control_s42_20261103T095147/command.json) |
| 对照独立评价 | 283/320，功能未通过 | [证据](../../records/reports/getup-late-feasibility/control-analysis.json) |
| 官方同条件参照 | 295/320，功能未通过 | [证据](../../records/reports/getup-late-feasibility/original-analysis.json) |
| 可达目标候选训练 | 50次rollout、1228800转移、800次actor小批量梯度 | [证据](../../records/experiments/duck_getup_late-feasibility_candidate_s42_20261103T095147/command.json) |
| 候选独立评价 | 283/320，功能及相对门槛未通过 | [证据](../../records/reports/getup-late-feasibility/candidate-analysis.json) |

协调器顺序调用`--train`、`--verify-train`、`--evaluate`、`--verify-eval`，实际argv、TTY、时间、PID及输出保存于每项记录。候选仅在覆盖充分且晋级次数上限可达时运行；最终本协议关闭，无自动重试、无延长、无留出消费。学习合计2457600转移，评价528000转移；真机IO为0。

本轮4段训练原片均为0–8秒、201帧、25fps；12段评价为0–11秒、276帧、25fps，自动重置前保留真实终态，没有为拍摄增加物理步。原VS Code已分别播放8秒训练及11秒评价至终态，截图在[证据目录](../../records/reports/getup-late-feasibility/README.md)。原3秒历史片段不改写，保存状态回放另列。图册新增时间范围、帧数和帧率，总览现为358段、716帧；16段新录像全量解码，其余211张当前图卡按来源和图像哈希复用。

全样本保存数据复盘确认末段请求幅度降低但成功数未增加，见[阶段分析](../../records/reports/getup-late-feasibility/mechanism-review.json)和主报告第7.7.11节。图像检查修正了自动刻度、图例位置及审阅裁剪的DPI换算，没有修改任何轨迹、模型或冻结采样源。下一步先查释放后早期支撑，旧协议不继续。

## 起身早期解困与事件录像

2026-11-03，先更新完整目标，再从960次已保存轨迹定位支撑形成时间，不新增物理步。多数失败在3秒前开始持续头腿接触，而旧末段辅助损失在5秒才启用；成功过程也存在短暂接触，因此新诊断要求姿态、颈角、速度和原目标共同满足并持续0.30秒。保存输入筛查与真实处理器检查分别记录，不能把检测重叠当成恢复成功。

单一规则对照冻结为83份来源，协议`configs/experiments/getup_early_guard_study.json`，SHA-256为`059c7b64dbe267001e67ae1dba12bc724be6dbcafa64dfb63fbcf64cf087f5fb`。在原VS Code共享可见终端先完成实际GPU零步，再顺序运行官方原样对照和候选，各320个完整11秒首次episode，合计352000个评价转移、零学习。候选只启用早期颈部有界修正；官方权重、共同物理、噪声、原始动作历史、冻结和五秒后全身目标处理相同。

对照291/320、候选296/320，原及时持稳与末段持稳门槛均未通过，无入选方案，不追加、不使用留出。全样本复盘确认候选29次实际触发、7次最终持稳；候选24次失败均无末段头腿接触，倾角中位31.34°。冻结期首步已数值分歧，不能把同编号的10次转成功、5次转失败视作逐场景因果证明。命令、实际argv、CPU/GPU检查和独立复核见[阶段索引](../../records/reports/getup-early-guard/README.md)。

本轮8段原始录像均完整覆盖0–11秒、276帧、25fps。四个预选相机环境均未触发规则，故按“候选5秒前触发的最小环境编号”选择环境2，另作两段完整11秒保存状态回放；不按成功筛选、无插值、无新增物理步。按实际触发2.88秒和修正峰值2.96秒提取事件截图，避免固定3秒、4秒截图漏掉短暂变化；全片仍保留正常时间。回放已在原VS Code播放至11秒终态，截图记录1214×818、位置49/28，未调整窗口或面板。

CPU保存数据复核入口如下；它们只读已有物理轨迹，不启动新训练：

```bash
cd /home/atlas/XGO-Duck/workspace/xgoduck_rl
./.venv/bin/python -B /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/validation/review_getup_early_guard.py
```

图册增加8段原片，现共366段、732张标注帧；6段保存状态回放与1个启动单帧文件单列。8张新图卡、两个图册首尾和事件截图已目视检查，旧原片按哈希复用。已完成的两个临时任务归入任务历史，日常菜单保留5项，后续不重复运行已关闭试验。

S4独立确认只完成[CPU批量推理准备](../../records/reports/roll-common-confirmation/cpu-import-check.json)：512组旧滚翻真实观测与官方ONNX最大误差5.96×10⁻⁷，行为和归一化不变。新动态协调器、完整冻结及实际GPU零步尚未完成；不据此报告新成功率或速度提升。S3、S5、S6与共同噪声下的行走回归仍未完成，真机后置。

## 官方滚翻独立确认与随机状态修复

2026-11-03，在完整目标内独立推进S4，不重启任何已关闭学习试验。CPU先把旧32次轨迹按两种几何算法复算为30次成功，512组真实输入的批量导入最大误差5.96×10⁻⁷，完整场景物理字段与原共同环境一致。原协议固定87份来源，最多384个首次episode、172800个评价转移，零学习。

启动器首次因Windows写入CRLF导致Bash的pipefail选项读取失败，在Python前以2退出，没有协调器或物理进程。保留原字节和截图后只统一LF，并完成Bash静态检查。随后原协议实际GPU零步触发CPU/CUDA随机状态断言；运行契约确认control、transition和physics_substeps均为0，原轮次关闭。CPU spy复现导入器内部torch.manual_seed调用CUDA全设备设种子一次，CPU虽恢复，CUDA未被原CPU-only fork保护。

独立修复仅在导入外层加CPU/CUDA随机状态保护，88份来源冻结为`configs/experiments/roll_common_rng_recovery.json`，SHA为`2c8011cf4bcb13d0d955de86f581eb0b6ce8634103e560213e294d737d4c6d6b`。父87份源码保持原字节，动作数值、物理、原始历史、评分和预算不变；相同种子复用是因为父协议从未开始动态episode。新GPU零步通过，最大误差7.15×10⁻⁷，随机状态和初态均未改变。

原VS Code共享可见终端运行“滚翻随机状态修复确认（不训练）”，顺序为startup、evaluate/verify development-a、development-b，两个开发条件均通过后才evaluate/verify heldout。实际结果117/128、116/128、112/128，留出未通过，已关闭；384个首次episode、172800个评价转移、零学习，无物理重置。seed2020已消费，不用于后续挑参数。真实argv、TTY、PID和日志在[索引](../../records/reports/roll-common-confirmation/rng-recovery/README.md)。

三段原始相机录像均0–9秒、226帧、25fps，完整解码及逐帧姿态哈希通过。预定环境0全部成功，事件图册明确说明个案不能代表总体。各自真实头顶支撑、有效旋转完成和持稳结束时刻对应到已有25fps帧，无插值、无新增物理步。原窗口已播放开发A至9秒，位置49/28、尺寸1214×818不变；已完成任务移到历史，日常保留5项。每组采样、录像和保存约48秒，整轮含初始化和CPU复核约5分7秒，不作PPO提速证明。

开发23次失败中，7次缺少规定头顶接触顺序、15次有效支撑旋转不足，另1次未满足其余条件。后续假设仅参考开发失败，不利用留出选择参数。录像图册新增3段原片，总数369段、738张标注帧，6段状态回放和1个启动单帧文件仍单列；235张旧图卡按哈希复用。

## 起身累计漂移与成功动作保持准备

旧两组50次更新的局部KL约0.0038/0.0040且没有触发KL限制，仍不能说明保留了官方行为。新保存数据审计冻结全部320个官方episode、11个输入时刻，三份模型在相同3520组输入上CPU推理；原模型输出与保存动作最大误差2.15×10⁻⁶。7.98秒原成功输入上，两组相对官方输出RMS为0.0801/0.0762rad。静态偏离不作退化因果证明，见[结果](../../records/reports/getup-policy-drift/analysis.json)。首次审计误用评价ID读取训练日志，CPU阶段在写出结果前停止；随后按导出产物保留的真实训练ID读取，输入和模型未改，错误单列。

新训练草案保留官方起点、共同噪声/碰撞、原奖励、原始历史和post5s目标处理，仅检验训练时的成功输入均值保持。CPU提取2–7.98秒2360组原成功输入，四种起始姿态×八个时刻构成32个单元，独立采样器各取8组组成256输入。解析损失7、均值梯度5、教师无梯度、完全匹配时零梯度、零系数原张量保持及CPU随机状态检查均通过，没有优化器或物理步。辅助函数检查不等于完整PPO等价，价值起点、检查点日程、全来源冻结和实际GPU零步仍须完成，新学习尚未开始。全部九类动作、连续切换、扰动与部署准备目标保持有效。


## 起身价值精确尺度与动作保持准入

新种子43固定价值检查首次在0control、0transition、0gradient时停止。检查点标量为15.416328204600083，报告独立CPU复算为15.416328204600084，差1.7763568394002505×10⁻¹⁵。原严格相等断言正确暴露了协议取值来源不一致；修复直接读取原检查点值，不改容差或重新计算尺度。原82份源码、日志、初态与运行计数均保留。

CPU修复准备首次误用带CUDA随机状态保护的通用网络工厂，该工厂即便指定CPU也会初始化CUDA；未运行物理或优化器，未生成冻结协议。改用独立CPU价值网络构造后，CPU随机数恢复、检查点反序列化、精确标量和完整场景检查通过。随后83份来源冻结为`configs/experiments/getup_anchor_scale_recovery.json`，SHA为`ae880202d6f647d79258f32b2934ed6738f9e26bbbee80e0b24984e81b247153`。

原VS Code共享可见终端运行“起身精确尺度检查（不训练）”：实际GPU零步通过，固定actor/critic采集400步×1024；409600转移、零网络更新。CPU用直接折扣求和重新计算25600验证输入，EV0.8078015、归一化MSE0.029093，四姿态EV均为正；两个网络及归一化逐位未变。初次失败0转移，合计仍409600。[复算](../../records/reports/getup-anchor-value-check/scale-recovery/analysis.json)和完整8秒201帧录像已落盘，全部录像qpos哈希一致。

随后冻结`configs/experiments/getup_teacher_anchor_pair.json`，SHA为`987445179953f33729b9e5947637c294282de9d1bbdd3935df50085413a25bf5`，87份来源。原共享任务终端启动“起身成功动作保持对照”；每组最多50次更新、24步×1024，系数0与0.01，逐次累计偏离触线停止，候选预算由对照实际完成量确定。每组320个11秒首次episode评价使用新开发seed1030；没有使用最终留出、硬件或自动延长。结果须等实际运行及独立复算完成后填写。


起身保持约束成对运行已关闭。对照第41次RMS0.05154rad触线停止；候选按相同41次结束，RMS0.02422rad，降低53%，未触线。actor小批量梯度步分别618/620，两组各820次新critic小批量更新；已有critic的50次拟合只作继承来源，未重新计入新学习。两组实际初态、展开物理字段、网络、Adam、归一化及随机状态一致，固定回放批次哈希相同。

独立seed1030评价为官方285/320、对照287/320、候选303/320，960个首次episode、528000转移；学习合计2015232转移。候选未通过总体、指定仰卧/左侧及实际仰卧门槛，本轮关闭，无入选方案、不追加、不消费最终留出。17个候选失败均有末段头腿接触与倾角超限，中位约33°；16个实际仰卧。冻结期数值已分歧，不按同编号作单场景因果结论。

本轮新增17段原片：零更新固定价值检查1段8秒、两组训练各1段8秒及1段3.68秒中断段、三组各4段11秒固定评价。全部新录像完整解码、逐帧物理状态哈希通过。对照首段在原VS Code完整播放至8秒，截图尺寸1214×818、位置49/28。图像核验修正自动轴刻度和浅色文字对比度，未改数据、原片或冻结来源；事件图册按实际释放、站高、躯干竖直和持稳时间选帧，缺失事件保留。

录像总览更新到386段、772张标注帧，6段状态回放和1个启动单帧文件继续单列。新持续接触奖励仅建不可执行草案，先离线核验成功借力与失败持续接触的差异；共同噪声/目标过渡下七指令回归仍需独立完成。


本轮两个未采用的25更新中间检查点已按[清理清单](../../records/reports/getup-teacher-anchor-pair/intermediate-cleanup.json)删除，回收9.26MiB。初始与最终模型、准入critic、全部物理轨迹、源码、原片和验证证据均保留并核对哈希。


## 共同接口七指令与持续接触复盘

2026-11-03，GPU采样、渲染均从原VS Code同一共享任务终端执行，窗口1214×818、屏幕位置(49,28)保持原样。CPU数据分析、文档和图册更新独立执行，不启动隐藏训练。

原46份来源的共同接口七指令协议：1040完成223/224，1041在第79步因后退环境98终止而触发mjlab强制重置要求。原采集器遗漏已结束环境的重置，实际17696转移保留；原轮累计241696。旧协议、源码、日志和部分原片不覆盖，不重跑。

修复先写目标与草案，再在CPU检查只重置已结束且永久退出首次episode评分的环境；保留重置前后状态、仅清除对应历史/控制器，检查仍在评分环境的qpos/qvel和动作历史逐位不变。新协议冻结50份来源，开发1042/1043、条件留出2040，最多672000转移、零学习。实际执行命令：

```bash
cd /home/atlas/XGO-Duck/workspace/xgoduck_rl
.venv/bin/python -B /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/validation/recover_common_walk_resets.py --freeze
# 原VS Code：Terminal → Run Task… → XGO-Duck · 七指令首次采样修复（不训练）
# 对应可见入口：
bash /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/training/training_workflow.sh common-walk-resets
```

原协议SHA为`a1be791e28c82a4f5eea203543e164a54dfc06fc57236fa4e482237d424ff1f3`，独立修复SHA为`ee201b1dd019711d44e3183d47c4d28fc428a378d294e8bc219a2cb07441bc12`。Shell入口以LF保存并通过`bash -n`。每个种子先核验CPU配置、GPU实际输入与CPU ONNX、目标处理、模型和随机状态；源码和模型哈希在运行前检查。启动与采样输出保存在各自`execution.log`，退出由协调器顺序处理。

新开发1042、1043均221/224且七项通过，随后唯一一次使用2040。留出221/224，但前进30/32低于31/32要求，2026-11-03 15:31:45按规则关闭；没有追加样本。前进两次仅偏航误差失败，约0.2041/0.2052rad/s；后退一例速度接近零。三个新种子均未提前结束，因此实际重置0次，CPU助手通过不能冒充GPU重置分支已实测。新增672000，连同原轮合计913696评价转移、零学习。2040此后不用于参数选择。

```bash
# 下列命令只复算保存数据，不能重启关闭的评价。
.venv/bin/python -B /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/validation/review_common_walk_closure.py
.venv/bin/python -B /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/validation/audit_getup_contact_duration.py
.venv/bin/python -B /mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck/scripts/validation/check_action_video.py
```

持续接触分析复算上一轮三组共960条起身轨迹，按原标准核对全部成功/失败后，再测量2–8秒内连续头腿接触时长。成功动作最长0.76/0.86/0.84秒，保持约束候选17次失败均超过5秒；这支持研究持续卡住信号，不证明惩罚能改善动作。第一次CPU写入因报告子目录未创建而结束，创建该分析目录后完整复算，未增加物理步。相应草案仍不可执行，接触计时、价值输入、共同价值起点、新种子与来源冻结均需先通过。

本轮原片35段全部解码、与真实qpos逐帧哈希核对；28段行走展示覆盖4秒，7段中断覆盖0–1.56秒，控制状态另保留至1.58秒。新增70张标注帧，全图册现421段/842帧；6段保存状态回放和1个启动单帧文件仍单列。35张新增全尺寸卡片、两幅总览首尾和接触时长图已逐项查看。前进原片在原VS Code播放到4秒终态，截图见[评估与播放](assets/common-walk-vscode-complete.png)。

资料入口为[阶段索引](../../records/reports/common-walk-confirmation/README.md)、[独立复算](../../records/reports/common-walk-confirmation/closure-review.json)、[接触时长](../../records/reports/getup-teacher-anchor-pair/contact-duration/analysis.json)。关闭的一次性任务已归档，日常菜单保留5项。全部九类动作目标仍有效，可靠起身、滚翻、共同接口行走、连续切换和扰动尚未全部完成，真机运行后置。

## 持续接触计时与价值准入

[组件检查](../../records/reports/getup-persistent-contact/component-preflight.json)复算960条保存轨迹，核验计时释放边界、重复调用、无接触清零和局部重置。192组输入与官方ONNX最大差1.07×10⁻⁶。共同价值任务在原可见VS Code终端运行，实际采样409600条，50次critic小批量更新，actor0。GPU选择性重置只清除了环境2、4。完整8秒原片及每帧状态哈希保留在[运行目录](../../records/experiments/duck_getup_contact-duration-value_reference_s44_20261103T160850)。

[原CPU检查失败](../../records/reports/getup-persistent-contact/value/original-verification-failure.json)来自归一化计时舍入差异。409600项中4785项与CPU直接除法结果不同，最大5.96×10⁻⁸；独立[倒数乘法复核](../../records/reports/getup-persistent-contact/value/cpu-reciprocal-review/analysis.json)与实际GPU值逐位一致，其余核验和门槛保持。没有重复仿真或梯度更新。共同价值通过后，在CPU临时克隆中验证了PPO/Adam接线，保存[优化器预检](../../records/reports/getup-persistent-contact/ppo-preflight.json)。

按暂停指令保留续接位置，恢复后核验图册新增N291和两幅总览。起身成对对照的CPU冻结和来源校验已通过，协议固定96份工程来源。执行位置仍是一个共享VS Code任务终端。实际步数、训练停止与独立评估随后以执行记录为准。

## 滚翻支撑损失时点

在起身对照运行期间，单独用CPU复算两个开发种子的256条已保存滚翻轨迹。7次失败没有规定的头顶接触顺序；其余16次失败均在头顶接触后损失较多离地旋转。保持原判据，开发分析没有新增仿真或学习。[完整复算](../../records/reports/roll-common-confirmation/rng-recovery/development-support-timing.json)

## 起身持续接触对照

任务在原VS Code窗口运行，工作区切到XGO-Duck。置前工具一度恢复窗口到非最大化状态，随后恢复到操作前的1707×1019。[训练截图](assets/getup-persistent-ppo-terminal.png)

96份工程来源、初始策略、共同critic与Adam逐位核对。两组各50次完整更新，共2457600条转移。候选实际触发惩罚45709次，两组各三个重置边界均检查。三组同一开发种子各320次评估，官方296、原奖励294、候选289，均未达到功能门槛。结果与配对释放差异见[独立复核](../../records/reports/getup-persistent-contact/pair/saved-data-review.json)。

新增四段8秒训练和十二段11秒评估原片，逐帧状态哈希及解码通过。可见任务共916.1秒，训练循环合计266.1秒。曲线首次校验发现自动生成的60次刻度超出图面，改用0–50的实际预算刻度重新生成，数据和评分保持。两份未评估的第25次更新中间点按[清单](../../records/reports/getup-persistent-contact/pair/unused-checkpoint-cleanup-plan.json)删除，共9.27 MiB，保留并复核初末权重、导出、轨迹和原片。[清理核验](../../records/reports/getup-persistent-contact/pair/unused-checkpoint-cleanup-result.json)

## 共同释放状态准备

CPU逐项复查旧三组前100步状态与目标。第一个控制步已有差异，释放时最大躯干姿态差13.25°。首次共享释放协议通过CPU缓冲检查，但GPU在固定地形实体登记断言处停止，实际积分0。保留原协议、CPU预检、实际GPU接口启动及停止日志。 [核验索引](../../records/reports/getup-shared-release/README.md)

独立固定地形协议使用开发 seed1052，冻结 102 份工程来源。原 VS Code 共享任务终端的窗口边界为 (0, 0, 1707, 1019)，执行前后相同。实际运行通过 GPU 零步恢复，两次固定策略共 32,000 条转移、5,000 个物理子步、零学习。协调器记录采集约 90.7 秒，随后进行独立 CPU 检查。两段 11 秒原片及逐帧状态哈希保留。 [终端截图](assets/getup-shared-release-terminal.png) · [采集记录](../../records/reports/getup-shared-release/fixed-terrain/coordinator.json)

```bash
cd ~/XGO-Duck/workspace/xgoduck_rl
ROOT=/mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck
# 本次采集使用原 VS Code 的共享任务终端，命令仅供查阅。
bash "$ROOT/scripts/training/training_workflow.sh" getup-shared-release-terrain
# 下列命令复算保存数据。
.venv/bin/python -B "$ROOT/scripts/validation/review_getup_shared_release_motion.py"
.venv/bin/python -B "$ROOT/scripts/validation/review_getup_supine_support.py"
```

视觉检查发现两张新增卡片把分支名写成了落下方向，且关键帧把失败环境在第 3 秒的画面标为正在起身。已修正这两处注释，并把卡片的后帧改为第 11 秒终态。两段原片和冻结来源保持原字节。姿态拆分显示最大末帧差来自水平朝向，逐样本评分一致。仰卧分析复算 537 条保存轨迹，并将双脚接触与躯干进入直立区间分开统计。

两张新增卡片及两幅总览首尾已核验，累计 440 段原片和 880 张标注帧。一次性 VS Code 任务归入历史，日常菜单保留五项。两次冻结协议的 100 份及 102 份工程源哈希复核通过。 [视觉核验](../../records/reports/stage-media/visual-review.json)

## 仰卧支撑伸膝准入

加入 seed1016 的归档后，共复算 1,920 条首次轨迹，其中实际仰卧 1,039 条。离线筛查通过预先设定的失败覆盖率及成功样本触发率条件。实际 CPU 通路检查选用 seed1050 官方轨迹中首个满足条件的环境编号 29，并使用第 155 步后的保存动作。首次读回在比较 `torch.Tensor` 与 mjlab 的 `TorchArray` 时停止，修正为对两侧都复制张量后通过。此检查两次均未积分。CPU 读回确认原始动作历史、物理位置、速度和时间不变，伸膝差异经过延迟缓冲到达执行器。 [准入记录](../../records/reports/getup-knee-support/README.md)

## 伸膝同状态筛查

seed1053 协议固定 106 份工程来源，分支前共用首次跌落和触发状态，最多 92,160 条转移。任务从原 VS Code 共享终端启动。CPU 模型、旧轨迹的目标复算和实际 GPU 的 335 项快照恢复通过。随后合成目标检查在五次执行器写入的全程电机隔离断言处停止，退出码为 1，实际物理步、控制转移、学习均为 0。任务已归档。 [原协议](../../configs/experiments/getup_knee_support_screen.json) · [停止记录](../../records/reports/getup-knee-support/screen/zero-step-stop.json)

BAM 在每次写入前用上一子步同环境所有电机的绝对扭矩和计算压降。这一路径可能在膝电机先响应后，将供电变化传给同环境其他电机。后续零步检查将记录延迟后目标、电压与实际控制，并加入相同电压下的对照，以定位影响来源。

CPU 对照的四条路径各写入五次，物理步为 0。两次官方写入逐位相同。常规供电下候选第二次只改变两膝，第三至第五次同环境出现供电及非膝电机差。相同电压对照消除了非膝差。采用独立 seed1054，冻结 108 份工程来源，保留原姿态判据、脉冲和采样上限。命令记录如下。

```bash
cd ~/XGO-Duck/workspace/xgoduck_rl
XGO_PROJECT=/mnt/c/Users/atlas-win/Desktop/RL-Project/XGO-Duck
source "$XGO_PROJECT/scripts/setup/project_environment.sh"
.venv/bin/python -B "$XGO_PROJECT/scripts/validation/getup_knee_support_supply_recovery.py" --diagnose-cpu
.venv/bin/python -B "$XGO_PROJECT/scripts/validation/getup_knee_support_supply_recovery.py" --freeze
# 以下采集入口从原 VS Code 共享任务终端运行。该协议现已关闭。
bash "$XGO_PROJECT/scripts/training/training_workflow.sh" getup-knee-support-supply-recovery
# 复算保存轨迹与在同一终端生成关键帧。
.venv/bin/python -B "$XGO_PROJECT/scripts/validation/review_getup_knee_screen_motion.py"
bash "$XGO_PROJECT/scripts/training/training_workflow.sh" getup-knee-trigger-frames
```

GPU 零步复核通过。实际 141 个共享控制步、三个 409 步的分支，合计 1,368 个批量控制步、6,840 个物理子步和 87,552 条转移。采集加独立核验约 151.5 秒，无学习更新。三组持稳 56、57、54/64，候选未达继续门槛，关闭并归档。三段原片均为 276 帧、25 fps、物理时间 0–11 秒。另用环境 31 的 12 个保存状态渲染对照图，逐帧核对源轨迹，物理步为 0。 [结果](../../records/reports/getup-knee-support/supply-recovery/analysis.json) · [图册核验](../../records/reports/stage-media/visual-review.json)

首次任务的快捷键使 VS Code 从最大化状态短暂恢复成 1202×802 小窗口。启动前通过窗口按钮恢复到原 (0, 0, 1707, 1019) 边界。复核采集与保存状态渲染均通过原 Terminal 菜单启动，复用同一任务终端。 [窗口核验](../../records/reports/getup-knee-support/supply-recovery/terminal-capture.json)

## 仰卧接触轨迹复算

按暂停指令保留脚本和轨迹。用户恢复目标后，在已有 WSL 环境读回三个开发种子的 395 次实际仰卧释放，原及时持稳、末段持稳和静态场景字段逐项复算。第 2.30 秒配成 54 对，51 对双方的首次接触都涉及右脚。对齐后，失败组在约 0.24 秒失去继续转正的俯仰趋势，0.50 秒时 50/51 有头腿接触。图的坐标、中文标签、图例及色带已查看。无新增物理步、学习和留出使用。 [数据、插图及命令](../../records/reports/getup-contact-entry/README.md)

接触后支撑几何复算确认，第 0.10 秒三批配对的右髋、右膝需求差方向一致，第 0.24 秒反转。共同接口前进的三批开发轨迹用原偏航指标核对，另拆分均值与波动。两项均使用原保存状态，未重新评估留出。 [支撑几何](../../records/reports/getup-contact-entry/support-geometry.json) · [前进偏航](../../records/reports/common-walk-confirmation/development-yaw-review.json)

## 髋膝协同对照

固定 109 份工程来源。CPU 和实际 GPU 的零步检查通过，确认右髋和右膝的修正经过 BAM 延迟到达电机。实际采集从第 122 个控制步的共同状态分为三路。合计 1,406 个批量控制步、7,030 个物理子步、89,984 条转移。独立检查复算评分、触发和目标，并逐帧核对三段完整 11 秒原片。继续门槛未通过，关闭并归档一次性任务。 [过程与命令](../../records/reports/getup-contact-shift/README.md)

从原 Terminal 菜单复用同一任务终端。置前工具曾把窗口恢复到 1202×802，启动前已恢复原 1707×1019。运行截图记录候选的实际进度，编辑区显示的是先前滚翻录像。预设相机环境 0 的髋膝对照原片另行保存。 [终端截图](assets/getup-contact-shift-terminal.jpg)

同一阶段复查滚翻的两批开发轨迹。16 次已触及头顶的失败中，15 次在最大离地段开始前由头部支撑。该复算只用了保存数据。 [滚翻复算](../../records/reports/roll-common-confirmation/rng-recovery/development-contact-handoff.json)

起身新增原片 3 段，对应图册 N313–N315，累计 446 段、892 帧。三张新增卡片、两幅总览首尾和 15 张对齐关键帧逐一检查，保留原片时间和来源哈希。 [媒体核验](../../records/reports/stage-media/visual-review.json)

## 前进偏航相位

开发种子 1040 用于选取编码器信号和时差，开发种子 1042、1043 原样复核。双髋俯仰速度之和领先偏航约三个控制步。用保存的 4,608 组观测复查 0.06 rad/s 的内部偏航输入，只核验官方动作到既有关节投影的通路。没有重新评估原留出。 [复现入口](../../records/reports/common-walk-confirmation/README.md)

## 前进相位对照

首次 CPU 准入在类奖励守卫处停止，物理步和控制转移均为零。复核固定 `upright` 的调用与重置行为，快照保留实际奖励管理器缓冲并核对类配置哈希。修正后原开发种子 1042 的七指令全部复算一致，CPU 和实际 GPU 零步均通过。 [首次停止](../../records/reports/common-walk-confirmation/forward-phase-cpu-snapshot-first-stop.json) · [准入记录](../../records/reports/common-walk-forward-phase/preflight.json)

从原 VS Code 的 Terminal 菜单启动，复用原任务终端。启动及运行截图均为 (0, 0, 1707, 1019)，编辑区仍显示此前的滚翻录像。开发种子 1044 共完成 2,900 个批量控制步、14,500 个物理子步和 649,600 条转移，含独立复算约 236 秒。候选的相位偏航在前进环境实际生效 30,400 次，三路全数跑到原期限，无额外局部重置。保留 3 段完整 4 秒原片及逐帧来源。 [过程与结果](../../records/reports/common-walk-forward-phase/README.md)

配对平均收益低于事前 0.01 rad/s 门槛，关闭并归档任务。核对相位图、三张原片卡片及两幅总览的首尾。总览增至 449 段录像、898 帧。 [图册核验](../../records/reports/stage-media/visual-review.json)

## 滚翻颈部支撑准备

离线电气复算最初使用通用驱动对象，未初始化 STS3215 内部目标，计算在读取该状态时停止。改用实际场景的驱动配置后，按固定 4 ms 延迟及固件限速重放历史请求。两个开发批次各 450 个控制点与实际缓冲和固件算法一致，再在 7.0 V 和 7.7 V 下复算头部与颈部俯仰的电机侧输出。 [驱动复算](../../records/reports/roll-common-confirmation/rng-recovery/development-head-motor-envelope.json)

颈部触发的保存数据检查覆盖两批 1,144 个控制点和全部 15 个相关失败。它只读取原轨迹，计算增加颈部请求的固定状态效应。规划使用新的开发种子、两个固定官方重复和一个颈部候选。草案当前未冻结，尚无本方案的动态采样或学习结果。 [试验参数](../../configs/experiments/roll_neck_support_draft.json) · [准入复算](../../records/reports/roll-common-confirmation/rng-recovery/neck-support-timing-preflight.json)

## 前滚翻颈部支撑

首次 CPU 快照守卫要求起身策略使用的观测延迟，滚翻场景没有该字段。改用滚翻状态清单后，重放两批保存数据的触发计数 578 和 566，实际 CPU 电机写入验证到第二次写入生效。随后种子 1022 的可见任务在建环境前因日志重定向错误停止，零物理步。种子 1023 的独立协议先实际验证标准输出和错误输出，再通过 CPU、GPU 零步准入。 [停止与复现](../../records/reports/roll-neck-support/README.md)

使用原 VS Code 的 Terminal 菜单，复用一个任务终端。置前操作短暂使窗口变为 1202×802，两个任务启动前已恢复原 1707×1019 及位置。候选组截图保存于第 150 步。采样实际完成 1,330 个批量控制步、6,650 个物理子步，原终端任务含首次校验约 135 秒。 [窗口记录](../../records/reports/roll-neck-support/log-recovery/terminal-layout.json)

原独立校验将第一条策略请求误作首个延迟值。mjlab 的重置会先写入一次零目标，CPU 零步读取实际缓冲复现。只读复算三组保存轨迹与完整录像，原停止状态和日志保留，未增加物理步。结果为 116/128、114/128、111/128，候选关闭。图册新增 3 段 9 秒原片及 6 张标注帧。 [独立复核](../../records/reports/roll-neck-support/log-recovery/readback/verification.json) · [图册核验](../../records/reports/stage-media/visual-review.json)

## 起身释放配对

已保存的种子 1030 三组各 320 次起身按原及时和末段持稳重新计分。匹配要求实际姿态一致，身体位置、姿态、关节角度及速度都在预定容差内。与官方接近的仰卧 166 次中，训练候选 150 次通过、官方 135 次通过。两个对应比较均无完全相同的释放状态。保留完整四姿态门槛，下一项先验证共同释放，再决定是否针对仰卧继续训练。 [复算](../../records/reports/getup-teacher-anchor-pair/release-match-review.json) · [草案](../../configs/experiments/getup_anchor_same_release_draft.json)

## 起身同一释放对照

种子 1056 固定了 116 份工程来源。CPU 和实际 GPU 的零步回读均核对 335 项。采集从原 VS Code 的 Terminal 菜单启动并复用已有任务终端，前后窗口均为 (0, 0, 1707, 1019)。第 2 秒三路回读均核对 345 项状态，合计 1,450 个批量控制步、7,250 个物理子步、464,000 条转移，物理采样约 180 秒。学习更新为 0。 [过程与截图](../../records/reports/getup-anchor-same-release/README.md)

独立检查逐场景重算及时持稳和末段持稳，三路分别 301/320、294/320、305/320。候选未通过实际仰卧和指定仰卧的门槛，关闭并归档一次性任务。六段原片均核对到第 11 秒，图册增加 N322–N327，累计 458 段和 916 张标注帧。下一项先核对检查点和仰卧释放首秒的策略权重。 [媒体核验](../../records/reports/stage-media/visual-review.json) · [训练草案](../../configs/experiments/getup_supine_weighted_learning_draft.json)

## 仰卧训练准备

CPU 恢复成功动作保持模型的 actor、critic、固定归一化及两套 Adam。320 个保存释放观测与 ONNX 动作对齐，仰卧分类与原评分逐项相同。临时优化器副本在无加权时复现原更新，混合权重下只改变 actor 更新。两组 512 环境配置编译后与父实验的 33 项静态场景字段一致。种子 44 已用于持续接触训练，当前草案选用训练种子 45 和独立评价种子 1057。无新物理步，临时 CPU 更新未保存。 [命令与检查](../../records/reports/getup-supine-weighted-learning/README.md)

## 仰卧释放加权训练

冻结 128 份工程来源。新核验器对旧 984 个控制步复算动作历史、关节目标和仰卧窗口，并重算种子 1056 的三个原成绩。种子 1057 在 CPU 零步核对 335 项快照字段。随后在原 VS Code 的 Terminal 菜单启动可见任务。首次点击使窗口短暂恢复成 1202×802，启动前恢复原位置和 1707×1019。 [冻结检查](../../records/reports/getup-supine-weighted-learning/preflight.json) · [窗口记录](../../records/reports/getup-supine-weighted-learning/terminal-layout.json)

两组实际 GPU 起点、随机状态、父模型和优化器逐项一致。两组各完成 50 次更新、614,400 条训练转移，独立核验权重样本、实际电机目标、Adam 计数、ONNX 导出和两段完整 8 秒原片。首次释放的仰卧数为 253 和 254，后两次两组均为 264 和 256。训练完成后进入种子 1057 的独立评估。 [过程、命令与原片](../../records/reports/getup-supine-weighted-learning/README.md)

种子 1057 的三分支共用 2 秒跌落和 345 项释放状态，完成 464,000 条评估转移。父模型、等权、加权分别为 303/320、302/320、310/320。候选的指定仰卧 30/32、实际右侧 37/41 仍低于门槛，已关闭一次性任务并归档。新增两段完整训练原片、六段完整评估原片，图册增至 466 段、932 张标注帧。 [结果](../../records/reports/getup-supine-weighted-learning/analysis.json) · [图册核验](../../records/reports/stage-media/visual-review.json)

两组训练含保存与导出分别约 101 秒和 106 秒，三模型评估采样约 190 秒。复查种子 1056 和 1057 的两类父模型成功轨迹，逐场景检查 900 条输入和动作，将回放存为 24 个组。CPU 验证等额抽样与非零策略梯度，物理和优化器更新均为零。 [样本复盘](../../records/reports/getup-supine-weighted-learning/recovery-retention-audit.json) · [回放检查](../../records/reports/getup-recovery-retention/replay-preflight.json)

## 起身动作保持对照

使用两个保存批次复算完整 CPU PPO 更新，零系数时网络、Adam、指标及抽样和旧更新一致。正系数只影响 actor。16 个小批次用于预先声明的梯度比例检查，确定系数 0.01。训练冻结 135 份工程来源，沿用种子 46、每组 512 个环境及至多 50 次更新。原 VS Code 共享终端启动前后位置和大小一致，两组实际 GPU 零步一致。 [优化器检查](../../records/reports/getup-recovery-retention/optimizer-preflight.json) · [窗口记录](../../records/reports/getup-recovery-retention/terminal-layout.json)

对照组第 32 次更新越过成功样本偏移停止线。实际执行 768 个批量控制步、3,840 个物理子步、393,216 条训练转移。CPU 重算动作历史、目标、已保存的完整 Adam 计数、抽样及所有 201 帧，确认重置前终态。保留第 25 次与停止点，归档一次性任务。候选和开发评估从未启动。 [过程、停止复算与完整原片](../../records/reports/getup-recovery-retention/README.md)

## 起身更新步幅

在两个原偏移上限下检查三档学习率。近阈值的单次 CPU 检查中，两组以半步幅通过，与直接采用半学习率更新逐项一致。穷尽步幅和注入 Adam 步后异常均触发完整回滚。另从原加权父模型顺序检查保存的 32 个批次，两组在单关节偏移守卫分别于第 14、16 次停止。临时权重已丢弃，物理步和保存更新为零。 [单次核验](../../records/reports/getup-success-step-control/preflight.json) · [连续核验](../../records/reports/getup-success-step-control/sequence-preflight.json)

## 滚翻脚掌触地

重算开发种子 1020、1021 的 256 条轨迹与原评分。逐场景读取最大离地段、首次恢复地面接触及首次脚掌着地，使用共同场景中原脚底碰撞网格复算最近点和速度。只在每批内匹配先恢复头部接触的轨迹，得到 6 对和 8 对。首次脚掌触地延迟的配对中位差均为 0。失败组在首次脚掌触地时的有效支撑旋转中位数为 252.0°、259.4°。本轮复算新增物理步和学习均为零。 [数据、哈希与分组](../../records/reports/roll-common-confirmation/rng-recovery/development-foot-transfer.json)

## 滚翻接触计时

从 14 对原开发样本分解支持损失后，冻结 102 份工程来源。原 VS Code 共享终端实际 GPU 零步与采样起点逐位一致。采样种子 1024，450 个批量控制步、2,250 个物理子步、57,600 条控制转移，采样约 38.5 秒。动作、目标、原评分和 226 帧原片独立核验，原门槛未通过。窗口尺寸和位置与启动前相同，原片在原编辑器播放至第 9 秒。 [过程与原片](../../records/reports/roll-contact-clock/README.md)

保存 300 个原有物理子步的前后姿态、接触点数和求解器法向力。失败组 45 个粗粒度离地区间在末子步积分前为零力，下一次同姿态求解均有载荷。实际传感器类复现底层更新与缓存失效分离，复算 7,552 个场景边界。一次性任务归档，下一项准备比较同轨迹的缓存与实时接触字段。 [边界](../../records/reports/roll-contact-clock/boundary-review.json) · [缓存](../../records/reports/roll-contact-clock/sensor-cache-review.json)

## 滚翻同轨迹接触对照

开发种子 1025 先做实际 GPU 零步，再用原 VS Code 共享终端完成 450 个控制步和 2,250 个物理子步。地面、头部和任一脚掌的对齐读数各有 57,728 个时点与原始约束接触点完全一致。缓存和对齐读法各通过 116/128，7 个场景转为通过，另有 7 个反向变化。保留 226 帧原片、日志与窗口截图，归档一次性任务。 [对照结果与命令](../../records/reports/roll-endpoint-readback/README.md)

## 暂停与产物整理

用户要求暂停时，滚翻独立确认仅通过 CPU 预检。共享终端的新任务定义移至任务档案，开发种子 1026 和留出种子 2026 均未启动。盘点 325 个权重和模型文件后，删除 36 份旧中间保存点并回收 166.3 MiB。保留的 289 个文件均通过哈希复核。 [保留索引](../../records/reports/training-curation/README.md) · [执行记录](../../records/reports/training-curation/result.json)

## 吞吐复算与档案归拢

按两遍规模试验的原始计时重新计算行走吞吐。4,096 环境相对 1,024 环境为 2.44 倍，最高观测显存占比约 28%。起身原配方为 512 环境，恢复后按任务短测规模，再以等转移预算比较动作质量。 [提速方案](development_report.md)

训练、评估、基准与剖析共 10,150 个文件按原批次归入训练档案，逐文件 SHA-256 核验通过。Windows 和 WSL 均验证兼容入口，冻结滚翻协议所需输入哈希一致。交付目录的 10 个权重文件逐一与原件比对。 [训练归档](../../records/training-archive/README.md) · [权重与迭代记录](../../artifacts/delivery/iteration_record.md)

## 仰卧左髋四路对照

开发种子 1156 从相同的第 115 步物理状态分叉。四路完成 1,424,640 条转移和 9,275 个批量物理子步，执行退出码为 0。独立核对首个电机写入、3,996,996 条接触记录和四段原片。两个脉冲方向均未通过。额外复算已保存的关节响应，确认两个方向确实改变左髋位移及两脚受力。 [冻结、日志、评分与录像](../../records/reports/getup-supine-left-hip-pulse-trial/README.md)

## 仰卧承重与二次起身准备

三路原策略的右脚载荷比例均呈同向差异。更严格的时序复算没有找到失败场景中的 20 毫秒提前量，关闭按此信号提前干预。改用四个开发种子的实际策略重力观测，复算后期持续卡住的检测和原截止下的时间余量。完整 HOME 的限速返回在 24 例均超出预算，主要受颈部越界请求影响。 [时序与二次起身记录](../../records/reports/getup-supine-left-hip-pulse-trial/README.md)

## 短程回撤同初态对照

四批触发与动作处理复演后完成 CUDA 原生零步核验。三路使用开发种子 1157，从第 160 步共同状态开始对比，共 1,021,440 条转移、6,650 个批量物理子步。评分核验通过，短程回撤的有效二次起身为零。场景 86 的三路完整回放从保存姿态渲染，采样原片另列。 [控制链](../../records/reports/getup-second-attempt-native/README.md) · [结果与回放](../../records/reports/getup-second-attempt-trial/README.md)

## 困难初态和归档权重

种子 1159 的 3,072 个 CUDA 环境中，原生零步读回核对了 2,048 个复位替换和 1,024 个原样复位，关节坐标、随机抽样流程及根部静止条件通过。正式采样执行 2,750 个批量物理子步和 1,689,600 条转移，退出码为 0。采样 116.92 秒，显存峰值 12,188 MiB，GPU 平均利用率 36.1%。独立评分通过完整性检查，初态筛查未达到难度门槛。 [初态和录像](../../records/reports/getup-hard-start-screen/README.md)

归档的奖励续训与侧卧到仰卧回放两份权重已复核 ONNX、恢复检查点和接口，旧配对实际仰卧分别为 171→174/177 和 159→162/166。本轮沿用原模型和检查点，采样和接口复核均无学习更新。 [模型复核](../../records/reports/getup-archived-policy-shortlist/analysis.json)

归档策略对照种子 1160 完成 CPU 冻结、Windows 起始页检查和 CUDA 零步恢复。CUDA 快照实测 291,123,232 字节，零步显存峰值 5,418 MiB。正式入口发现跨进程零步整场哈希不同而停止，实际与尝试物理步均为零。原协议、日志和退出状态保留。 [零步](../../records/reports/getup-archived-policy-comparison/s1160/native-zero.json) · [运行记录](../../records/evaluations/duck_getup_archived-policies_development_s1160_20261110T170800/README.md)

两次独立 CUDA 零步读回把整场哈希差异定位到 25 个求解和传感器派生字段。根部与关节状态、控制、观察和随机状态等 14 项关键输入签名相同。后续限定跨进程差异范围，四路以同一进程内第 100 步的完整物理与控制快照为共同释放点。 [字段清单](../../records/reports/getup-archived-policy-comparison/s1160/reset-field-diagnosis/analysis.json)

同进程启动通过后，种子 1160 的四路完成 1,945,600 条转移，Windows 观察记录 39 次运行刷新。首次独立评分在后续电机子步的逐字节要求处停止。随后保留首个子步和全部控制器输入的严格检查，复算后续电机差异及逐场景持稳。四路分别通过 971、974、993、994 个场景，两份归档自训候选进入开发复验。显存峰值 5,494 MiB，最后一步后约 57 秒完成压缩和写盘。 [结果、资源和录像](../../records/reports/getup-archived-policy-comparison/README.md) · [原评分停止日志](../../records/reports/getup-archived-policy-comparison/s1160-qualified/verify.log)

种子 1161 按相同四路和门槛再采样 1,945,600 条转移，独立评分及四段 11 秒原片通过核验。两份候选总体 980/1,024、983/1,024，指定仰卧 632/672、634/672，未达 639/672。后续对两个开发种子的保存轨迹做离线互补上界和接触复算。复验实际学习更新为零，最终留出未使用。 [复验与观察](../../records/reports/getup-archived-policy-comparison/README.md) · [残余失败复算](../../records/reports/getup-archived-policy-comparison/residual-review.json)

离线释放分流仅使用首个动作前 34 项实际输入。两个整种子交叉方向分别替换 415、426 个场景的策略，全部动作净少通过 4、3 例，关闭。后续只准备检查互补成功动作及其与接触的时序。 [冻结筛查](../../configs/experiments/getup_archived_policy_release_routing.json) · [复算](../../records/reports/getup-archived-policy-comparison/release-routing.json)

两批的首四个起身控制周期均未复现有用的单关节差异。按当前重力方向最多混合一半教师动作，30 个仅教师成功的场景均未达到 0.008 rad 变化门槛，新增目标越界超线。随后冻结 2.20–3.10 秒有界腿部回放，按完整种子分开两批。末层候选与对照各求解一次，原候选和 70% 固定缩放未通过限位。按共同开发包络导出 62.6% 残差后，两批的离线限位和保留门槛通过。 [完整命令入口和数值](../../records/experiments/duck_getup_teacher_head_offline_s1160_20261110T175320/README.md)

末层候选的原生检查点与导入策略在 1,110 条输入上的输出逐值一致。新种子 1162 的 CUDA 三路完成 1,484,800 条转移，独立复算三段原片和原动作门槛。候选的实际仰卧净增 3、2 例，实际右侧少 1 例，未晋级。三批释放观测的仰卧锁存离线筛查通过，需独立物理验证。 [起身末层评估](../../records/reports/getup-teacher-head-evaluation/s1162/README.md)

## 仰卧锁存与纠正回放

冻结新场景后，CPU 校验、独立 CUDA 零步以及 Windows 观察先后通过。三路共享第 100 步的释放快照，实测状态读回和锁存分支后，共采样 1,484,800 条转移和 7,250 个批量物理子步，退出码为 0。三段原片各 11 秒。两路参照和候选的指定仰卧成绩为 633/672、634/672、636/672，原门槛为 639/672。保留本轮运行和评分，最终留出保持封存。 [过程、资源与三段原片](../../records/reports/getup-teacher-head-evaluation/s1163-latch/README.md)

从两批保存轨迹筛选 32 个候选和父策略结果不同的场景，把 1,472 条保存输入按时间和用途压缩为 24 个单元、900 条回放。两组使用相同输入、抽样索引和辅助损失权重。首次跨设备的单精度逐元素检查退出码为 1，随后按双精度参考复核，候选梯度和对照梯度均符合各自门槛，且只进入末层腿部行。复核退出码为 0，演员权重逐位保留。没有执行优化器步或物理步。 [回放与输入来源](../../records/reports/getup-teacher-head-evaluation/corrective-replay/analysis.json) · [首次日志](../../records/reports/getup-teacher-head-evaluation/corrective-replay/native-loss-first-attempt.log) · [复核](../../records/reports/getup-teacher-head-evaluation/corrective-replay/native-loss-preflight.json)

从归档保存点读回演员、76 项价值网络和价值 Adam。CPU 与 CUDA 分别用原训练的 0.10 rad 标准差对临时演员执行两次 Adam。仅末层腿部行与对应动量发生变化，价值参数和原检查点哈希一致。2,048 条既有价值输入的两端预测最大差为 4.77×10⁻⁷。退出码为 0，物理步和持久学习更新均为零。 [执行日志](../../records/reports/getup-teacher-head-evaluation/corrective-replay/checkpoint-adam-preflight.log) · [核验结果](../../records/reports/getup-teacher-head-evaluation/corrective-replay/checkpoint-adam-preflight.json)

## 四姿态保持与在线纠正

从不与纠正回放重叠的 128 个成功场景提取 1,024 条观测。每姿态 32 个场景、8 个动作时点。保持标签来自当前训练起点的 ONNX，和真实轨迹动作最大差为 1.67×10⁻⁶ rad。两组各 3,072 环境的 CUDA 零步启动签名一致，显存峰值均为 11,986 MiB。 [回放与初态](../../records/reports/getup-latched-head-corrective-ppo/README.md)

首次正式入口在价值优化器签名比较处退出，实际与尝试物理步及学习更新均为零。独立 CUDA 启动确认，唯一差异是优化器参数编号从整数键序列化为 JSON 字符串键。按保存格式比较后，所有签名字段一致。单独冻结恢复协议，保留原预算和训练门槛。 [原停止日志](../../records/reports/getup-latched-head-corrective-ppo/control-execution.log) · [诊断](../../records/reports/getup-latched-head-corrective-ppo/signature-recovery/diagnosis.json) · [新运行观察](../../records/getup-latched-head-corrective-ppo-signature-live.html)

修复后的两组先后退出，退出码均为 0。共完成 2,506,752 条训练转移，独立重载了检查点和导出 ONNX，按原始物理状态及帧哈希核对了两段 8 秒训练原片。每组实际 17 次更新，显存和端到端速度均通过冻结门槛。 [训练结果与资源](../../records/reports/getup-latched-head-corrective-ppo/README.md) · [独立核验](../../records/reports/getup-latched-head-corrective-ppo/signature-recovery/training-verification.json)

新种子评估完成 CPU 场景及 CUDA 335 个状态字段核对，四路策略的仰卧分支逐位重放。额外的 GPU 实测发现，单独导入回退演员会改变 CUDA 随机状态。用独立 RNG 作用域导入后，CPU 和 CUDA 状态保持原字节，导入权重逐位一致。原四路评估仍是零物理步。 [原 CUDA 零步](../../records/reports/getup-latched-head-corrective-ppo/evaluation/independent-zero/proof.json) · [导入随机状态实测](../../records/reports/getup-latched-head-corrective-ppo/evaluation/import-rng-proof.json)

保护随机状态的正式四路评估退出码为 0。四路共享 345 项释放状态，共记录 1,945,600 条转移和四段 11 秒原片。独立评分和 Windows 观察均通过核验，四路均未通过逐姿态功能门槛。纠正标签关闭，保留训练对照作为仰卧研究参照，后续核对三个开发种子的共同右侧回退。 [评估与原片](../../records/reports/getup-latched-head-corrective-ppo/evaluation/rng-guarded/README.md)

## 右侧失败与专用策略

复算九路保存分支的 156 个实际右侧场景。三个种子各有 1、1、3 个共同失败，合计 17 条分支末段均持续双脚着地、头腿接触和倾角超限。135/144 个共同成功也早于或达到 2.38 秒出现接触。按最小场景编号重放共同成功、共同失败和一例同权分歧，20 帧的原始状态哈希已保存。 [复盘与关键帧](../../records/reports/getup-right-fallback-cross-seed/README.md)

双姿态分流首次固定评估通过原门槛。三路从 345 项逐位一致的释放状态开始，采样和观察均以退出码 0 完成，独立核验重新推理了三路真实输入。候选仅在右侧增加专用策略，实际右侧挽回同三个场景，进入独立复验。 [结果、日志及原片](../../records/reports/getup-dual-pose-route/README.md)

独立复验同样完成 1,484,800 条转移，采样、观察和独立评分均以退出码 0 完成。候选实际右侧 58/59，维持全部原逐姿态门槛。预设镜头实际释放为仰卧，保留原片及真实姿态标签。 [复验与原片](../../records/reports/getup-dual-pose-route/replication/README.md)

最终留出在 CUDA 和 Windows 观察通过后执行一次。三路共 1,484,800 条评估转移，候选总体 993/1,024，实际仰卧 765/792，实际右侧 69/73，距离原右侧门槛差 1 例。候选未晋级，留出封存。录像总览更新到 741 段、1,482 张标注帧。下一项准备短测并行规模和写盘开销，扩大新开发样本的实际右侧覆盖。 [最终评分和原片](../../records/reports/getup-dual-pose-route/holdout/README.md) · [新开发草案](../../configs/experiments/getup_right_robustness_development_draft.json)

规模短测共完成 1,433,600 条转移和六段 3.2 秒录像。3,072 环境初次在快照上限停止，物理步为零。依据实测快照大小单独复核，保留整卡 85% 显存上限。两档含写盘为 9,189 和 11,152 条转移/秒，选用 3,072 环境，实际显存峰值为 13,253 MiB。 Windows 观察及六段录像哈希核对通过。 [短测与选择记录](../../records/reports/getup-right-robustness/scale/README.md)

扩大右侧覆盖时，CPU 入口首次因两个协议对同一个姿态样本门槛采用不同字段名而停止，物理步为零。统一字段后通过 CPU 场景、独立 CUDA 零步和 Windows 观察检查。实际右侧 960 例。三路完成 4,454,400 条转移，独立评分为参照 A 910/960、参照 B 903/960、候选 936/960。候选所有原门槛通过，转入新独立留出准备。 [冻结、运行与评分](../../records/reports/getup-right-robustness/development/README.md)

恢复原四组请求比例，在 3,072 环境完成一次独立留出。实际释放右侧 183 例，满足至少 128 例的门槛。候选总体 2,983/3,072，实际右侧 179/183，相对参照净增 4、5 例，原门槛全部通过。采样与写盘 358.16 秒，实际 CUDA 显存峰值 13,334 MiB。预检首次指向不存在的官方缓存路径，改用项目原始归档后与固定上游哈希和交付文件一致。三份自训 ONNX、三个保存点与官方参照均通过核验。 [独立留出](../../records/reports/getup-right-robustness/holdout/README.md) · [交付清单](../../artifacts/delivery/getup-simulation.json)

起身接站立的首步预查使用开发轨迹。Windows 推理环境缺少 YAML 依赖，改用现有 WSL 训练环境完成，未安装新依赖。四个实际姿态各取 32 个场景核对关节零位转换和首步限速，最大未限速请求差为 0.214 rad，限速后 0.040 rad。后续冻结 32 秒连续回合。 [边界核验](../../records/reports/getup-stand-handoff/routed-development-boundary/README.md) · [连续方案草案](../../configs/experiments/continuous_routed_getup_stand_walk_draft.json)

完成 3,072 环境的连续回合。前 11 秒沿用起身控制链，第 11、12 秒分别检查站立和前进接入，使用独立动作历史和首段限速。总计 4,915,200 条转移、8,000 个批量物理子步，只有初始重置。原起身门槛通过 2,994 例，站立 2,995 例，前进 2,718 例，完整序列 2,717 例。完整序列低于 95%，前进的 277 例失败均含偏航超限。 [冻结、运行与评分](../../records/reports/continuous-routed-getup-stand-walk/README.md)

复盘从前进开始分成一秒窗口。77 例仅偏航失分，200 例还伴有其他失分，其中 198 例出现持续半秒的严重姿态异常，开始时间中位数为 0.66 秒。保留原 32 秒原片和六张阶段关键帧。下一项以共同站立终态做配对诊断，仅比较前 25 个前进控制步的指令渐升，随后按原判据决定是否进行完整复验。 [逐秒结果](../../records/reports/continuous-routed-getup-stand-walk/forward-failure-review.json) · [起步对照草案](../../configs/experiments/continuous_routed_forward_entry_ramp_draft.json)

前进起步完成同站立终态的配对对照。3,072 场景共用前 600 步，逐位核对 345 项分支状态，再分别采样 100 步。采样以退出码 0 完成，累计 2,457,600 条转移。独立评分首次因入口未包含训练脚本目录退出，补充 Python 路径后按原冻结代码复算。两组各有 3,009 例进入前进，持续严重姿态异常为 185、0，平均绝对偏航为 0.409、0.129 rad/s。候选通过短测门槛，完整 32 秒复验另用新开发种子。 [日志、评分与两段原片](../../records/reports/continuous-routed-forward-entry-ramp/README.md) · [完整复验草案](../../configs/experiments/continuous_routed_getup_ramped_forward_draft.json)

渐升前进的完整开发复验共 4,915,200 条转移、8,000 个批量物理子步。起身通过 2,983 例，完整序列通过 2,976/3,072，超过原 95%。复算核验 25 步渐升、636 条 ONNX 输入和 801 帧原片。CPU 冻结初次因导入顺序退出，修复前物理步为零。交付目录当前 14 份权重及保存点与各自来源逐字节相符。滚翻阶段扰动另用历史轨迹核对介入时点和 CPU 写入，尚待实际 CUDA 和仿真评估。 [完整复验](../../records/reports/continuous-routed-getup-ramped-forward/README.md) · [滚翻扰动准备](../../records/reports/continuous-roll-early-lateral-push/README.md)

渐升前进的单次独立留出共 4,915,200 条转移和 8,000 个批量物理子步。起身和站立接入通过 2,992 例，完整序列通过 2,990/3,072，整段通过 97.33%。保留实际使用的五个 ONNX、四个自训保存点、控制来源和完整原片的核验索引。滚翻阶段扰动另完成 128 个保存场景、294 项状态的 CUDA 零步写入与恢复。 [独立留出](../../records/reports/continuous-routed-getup-ramped-forward/holdout/README.md) · [仿真交付](../../artifacts/delivery/ramped-getup-standing-forward-simulation.json) · [滚翻扰动](../../records/reports/continuous-roll-early-lateral-push/README.md)

滚翻阶段三组共 445,952 条实际转移。无扰动和右向的完整序列为 124/128、116/128。左向在 115/128 个滚翻完成后触发原停线，115 个前进窗口未评完。末尾索引的协议路径变量未定义，采样以退出码 1 结束。保留原观察状态和执行日志，离线恢复索引并独立复算。左、右扰动各新增 13、10 例滚翻失败，其中 2 个场景重合。新增失败均越过 330° 几何角，损失主要来自离地旋转。 [索引、诊断和三段原片](../../records/reports/continuous-roll-early-lateral-push/README.md)

起身接七指令在 CPU 核对三份行走 ONNX。40 次对照的最大输出差为 2.24×10⁻⁷ rad，首步目标变化不超过 0.04 rad。跨模型清空原始动作历史，同模型保持历史，前进保留 25 步渐升。新开发计划为 3,072 场景，每条指令 438 或 439 个，待实际 CUDA 接入检查。 [方案与 CPU 核验](../../records/reports/continuous-routed-seven-command/README.md)

本机 CUDA 初始化 3,072 场景，七路目标写入并恢复，实际物理步为零。24 次 ONNX 对照最大差 2.09×10⁻⁷ rad，显存峰值 11,799 MiB。下一步冻结七路完整开发采样、分组评分和观察入口。 [CUDA 核验](../../records/reports/continuous-routed-seven-command/cuda-zero.json)

冻结七路连续评估后，独立 CUDA 复核通过 335 项初态字段。七路共接入 2,986 个场景，采样以退出码 0 完成 4,915,200 条转移。Windows 观察器通过 32 个含真实画面的运行状态，保存一段 32 秒原片。显存峰值 12,906 MiB，常规窗口中位吞吐 19,684 条/秒，跨分块写入的窗口为 5,480 条/秒。首次独立评分漏传已保存的终止标记，保留失败日志并通过原轨迹补读复算。 [七路执行记录](../../records/reports/continuous-routed-seven-command/README.md) · [观察与原片](../../records/continuous-routed-seven-command-live.html) · [性能记录](../../records/reports/continuous-routed-seven-command/sampling-performance.json)

滚翻新增失败的首个明显离地步，无扰动分支在左、右组分别有 13/13、9/10 个配对仍由头顶支撑。扰动组全部在第 4 步恢复头部接触，脚部还未参与首次失支撑。第 0.36 秒 IMU 读数与成功场景重叠，未选定共用单轴触发。 [支撑中断](../../records/reports/continuous-roll-early-lateral-push/head-contact-review.json) · [观测回读](../../records/reports/continuous-roll-early-lateral-push/support-signal-review.json)

七指令独立评分补读终止标记后完成。总分 2,932/3,072，达到总体门槛。左转 396/439、右转 403/439，未达到逐项门槛。两组已接入的 53 个失败都包含偏航跟踪误差超限，首秒之后仍有 50 例超限。转向复盘同时对照原 20 秒评分和后 19 秒诊断，完整七指令评估留档。 [评分、补读与原片](../../records/reports/continuous-routed-seven-command/README.md) · [逐场景转向复盘](../../records/reports/continuous-routed-seven-command/turn-failure-review.json)

核对第 601–800 步实际轨迹的写入开销。压缩等级 6 的两次平均耗时 17.22 秒，等级 1 为 14.69 秒，文件增加约 2.83 MB。两种方案的 21 个数组均逐字节往返一致，临时文件已清除。 [压缩基准](../../records/reports/continuous-routed-seven-command/trace-compression-benchmark.json)

核对通过固定七指令留出的源代码，左右转原本经过 0.2 秒低通和 0.75 的转速误差反馈，而连续入口仅用了固定映射。恢复入口按保存观测检查 200 步，两条转向逐项匹配原行走控制器。其余五路和未接入场景的输入逐项一致，下一步从同一站立终态做物理对照。 [反馈接入核验](../../records/reports/continuous-routed-seven-command/turn-feedback-cpu.json)

冻结同站立状态的转向反馈对照，3,072 个新开发场景共用前 12 秒，两组各执行完整 20 秒指令。实际 CUDA 零步通过，覆盖 335 项初态。第一次启动在读取旧入口的显存配置键时退出，物理步为零。兼容入口继续沿用冻结的 85% 上限和原观察页。 [配对协议与日志](../../records/reports/continuous-routed-turn-feedback-pair/README.md)

滚翻碰撞几何在首次明显离地时复现左、右配对头部间隙中位差 +0.120、+0.108 mm。头部俯仰小幅制动在所查窗口受模型输出限幅，两个供电端点下均无力矩增量。更早的 120° 窗口改查头部横滚，两侧 23 个样本同时有几何与模型驱动余量。两控制步的有界候选已核验保存观测的提前量，实际物理对照待准入。 [滚翻抗扰记录](../../records/reports/continuous-roll-early-lateral-push/README.md)

转向同状态开发对照完成 2,600 个批量控制步、7,987,200 条实际转移，仅执行一次初始重置。恢复组完整通过 2,977/3,072，固定组为 2,923/3,072。恢复组左右转分别净增 35、19 例，其他五组完整通过数一致。两个原片各 801 帧，Windows 核验了 52 次含原始帧的运行状态。预留独立种子 2175。 [采样、评分与原片](../../records/reports/continuous-routed-turn-feedback-pair/README.md)

七指令独立留出的首次启动在重置派生量 `warp.cdof_dot` 的哈希检查停止。控制步和物理步均为零，预留种子尚未使用。复核协议保留原场景、控制器和门槛，逐元素比较两次 CUDA 重置的原生数组，只允许零的符号位不同。 [首次评估记录](../../records/evaluations/duck_switch_routed-turn-gyro_holdout_s2175_20261110T225946/evaluation.json) · [复核方案](../../configs/experiments/continuous_routed_turn_feedback_holdout_zero_recovery.json) · [运行观察](../../records/continuous-routed-turn-feedback-holdout-zero-recovery-live.html)

滚翻目标层已在 CPU 复现两步脉冲。一个样本的头部横滚目标每步下调 0.02 rad，靠近下限的样本每步下调约 0.005 rad，非滚翻样本保持原目标。原始策略动作历史保持一致，触发状态可从配对分支还原。 [目标层检查](../../records/reports/continuous-roll-early-lateral-push/head-roll-overlay-cpu.json)

留出复核确认 `warp.cdof_dot` 的 368,640 个数值完全一致，只有一个零值的符号位不同。原分组累计 4,915,200 条转移，总体通过 2,992/3,072。Windows 观察核验了 32 次含原始画面的更新。六份实际使用的 ONNX 与五份自训保存点均逐字节核对。 [留出全流程与原片](../../records/reports/continuous-routed-turn-feedback-holdout-zero-recovery/README.md) · [交付清单](../../artifacts/delivery/routed-seven-command-simulation.json)

滚翻的首个 CUDA 预检未能用构造观测证明头部横滚余量。改用保存的真实观测后，复核了 28,160 组模型输入，256 个头部横滚目标按预期改变。随后快照适配器因从精简场景对象读取环境数而退出，记录显示两次预检均为零物理步。适配器改用冻结方案中的 128 个环境。 [首次预检](../../records/reports/continuous-roll-head-support-pulse-zero-startup/first-preflight.log) · [实际目标证据](../../records/reports/continuous-roll-head-support-pulse-zero-snapshot/pulse-cuda.json) · [快照停止](../../records/reports/continuous-roll-head-support-pulse-zero-snapshot/second-preflight.log)

快照适配器保留了原接口中的随机数生成器清单，CUDA 零步还原核验通过 299 项字段。原 128 个场景的横向速度写入、首帧接触和头部目标均通过，随后冻结六组同初态对照。每组保持原 31 秒上限和 116/128 门槛。 [快照接口退出](../../records/reports/continuous-roll-head-support-pulse-zero-inventory/third-preflight.log) · [CUDA 核验](../../records/reports/continuous-roll-head-support-pulse/cuda-zero.json) · [冻结方案](../../configs/experiments/continuous_roll_head_support_pulse.json) · [运行观察](../../records/continuous-roll-head-support-pulse-live.html)

三组历史对照耗时 399 秒、执行 3,484 个控制步。按六组最多 9,010 步外推并留 15% 余量，总墙钟上限冻结为 1,200 秒。首次入口在切换协议前记录为零物理步。实际动作时长、评分、扰动幅度和单组提前停止规则保持原定。 [零步记录](../../records/evaluations/duck_switch_roll-head-support-pulse_development_s1176_20261110T232008/runtime-contract.json) · [墙钟预算](../../records/reports/continuous-roll-head-support-pulse/wall-budget/preflight.json) · [六组观察](../../records/continuous-roll-head-support-pulse-wall-budget-live.html)

按用户要求在左推原控制暂停。进程关闭时累计执行 2,225 步，最后一拍在物理计算后触发停止，留存 2,224 步轨迹。无扰动原控制完整 31 秒，CPU 复算滚翻 120/128、完整序列 119/128。左推保存 14.64 秒，其余四组各保留共同起始段。 [暂停证据与原片](../../records/reports/continuous-roll-head-support-pulse/wall-budget/README.md)

用户恢复目标后，使用新运行编号完成滚翻六组同初态对照。原控制完整序列为 122/128、117/128、126/128，头部脉冲为 119/128、119/128、124/128。候选标称分支退化，关闭。运行完成 1,153,280 条转移，显存峰值 2,530 MiB，独立核验确认 305 项恢复字段和六组首步一致性。 [六组报告](../../records/reports/continuous-roll-head-support-pulse/resume-20261113/README.md) · [独立分析](../../records/reports/continuous-roll-head-support-pulse/resume-20261113/analysis.json)

参数端点筛查的多次启动和收尾问题已归档。最后一次使用 seed 1176 的低摩擦 arm 完成了 1,550 个物理控制步，但调用了旧的 `PolicyControl`，滚翻只有 3/128。该结果不用于参数结论。下一次端点验证必须直接使用六组对照通过的 `ProjectedRollPolicy`。 [失败记录](../../records/reports/roll-domain-parameter-screen/preflight-failures/README.md)
### 2026-11-13 · 物理端点筛查完成

在同一冻结协议、seed 1176、128 个 CUDA 环境和 1,550 个控制步下，使用已通过的 `ProjectedRollPolicy` 分别改变脚底摩擦 0.7/1.3 和供电 7.4/8.0 V。四组均完成 198,400 次控制迁移和 7,750 个物理子步，独立验证了来源哈希、首步接口、接触回读、动作历史、完整原片和原始门槛。结果为低摩擦 106/128、高摩擦 120/128、低电压 123/128、高电压滚翻 118/128 但连续前进 0/128。低摩擦在第 500 步记录了理论不可达点，但仍完整采集；高电压未提前停止，连续前进失败来自偏航门槛。第一次重跑暴露验证器停止字段、父协议哈希和跨 arm 环境类泄漏，均已修复并将失败产物归档；最终四组结果见[端点报告](../../records/reports/roll-domain-parameter-screen/README.md)。下一步只组合通过端点并更换 seed 复验，再决定训练随机化范围。

### 2026-11-13 · 端点交叉复验完成

换用 seed 1190 对标称、高摩擦、低电压和高摩擦低电压四组组合做完整固定策略复验。四组均通过原 116/128 滚翻与连续前进门槛，结果分别为 123/128、125/128、124/128、125/128；连续前进分别为 122/128、125/128、124/128、125/128。交叉组合仍保持 198,400 次控制迁移、7,750 个物理子步、独立来源核验和完整视频。候选随机化区间暂定为脚底摩擦 1.0–1.3、供电 7.4–7.7 V，下一阶段先做零学习随机化对照，再进行有界短预算学习。[交叉报告](../../records/reports/roll-domain-parameter-cross-check/README.md)

### 2026-11-13 · 参数随机化训练前置核验

为进入训练，先把候选区间接入起身训练配置的 CPU 副本，再做两个 GPU 零步。标称 arm 固定脚底摩擦 1.0、供电 7.7 V；candidate 使用摩擦 1.0–1.3、供电 7.4–7.7 V。两个 arm 均保持 1024 环境、actor 61 维、critic 75 维、动作 14 维、4 ms 物理步和 20 ms 控制步，reset 后显存分别为 4,522 和 4,862 MiB。首轮零步因 512 环境姿态分组复用而在 reset 前停止，改为显式 1024 环境四姿态分组后通过；没有物理步或学习更新。下一步冻结 8 次更新的同父点、同预算对照，并在首个 PPO 更新后重新核对显存和真实录像。[前置报告](../../records/reports/parameter-randomized-training/README.md)

冻结协议随后完成 nominal/candidate 各 8 次更新，每组 196,608 条转移、960 个物理子步。candidate 末次稳定率 93.75%、control 91.89%，两组 drift RMS 均约 0.0345 rad，最大 drift 0.1366/0.1432 rad，显存峰值 30.73%/34.34%。PPO 曲线、末次 checkpoint、ONNX 与真实 3.84 秒片段均保留。短预算结果暂不宣称动作收益，下一步按原四姿态起身、官方滚翻和七指令连续门槛复归。[训练报告](../../records/reports/parameter-randomized-training/README.md)


### 2026-11-13 · 参数随机化模型回归完成

冻结 seed 1202、同一四姿态起身、官方前滚翻和七指令连续屏幕，先后运行 candidate 与 control 两个 8 次更新末次 ONNX。起身两组 prone/supine 均为 32/32，left-side/right-side 均为 19/32；失败来自冻结结束时实际释放姿态转为 supine，与策略无关的共同物理现象。滚翻两组均为 0/32，保存轨迹没有形成 feet→head-top 接触和前向旋转。七指令两组 224 个环境均完整运行 20 秒，但线速度和角速度误差显示前后进退、转向和平移均未达到原门槛，只有 stand 的保存指标屏幕通过。完整运行、视频、轨迹和失败目录见[参数随机化回归](../../records/reports/parameter-randomized-regression/README.md)。本阶段不延长训练，转入独立 getup、roll、walk 技能的前置计划。[多技能前置计划](../../records/reports/multi-skill-training/README.md)

### 2026-11-14 · 多技能前置核验通过

在 WSL Ubuntu-24.04 中完成多技能 CPU 模型接口和 GPU reset-only 零步。官方 getup、roll、walk 以及 candidate/control getup 的 ONNX 输入均为 61 维、输出均为 14 维，14 个关节名与 `action_scale=1.0` 一致。getup 1024、roll 128、walk 224 个环境均只 reset 不 step，物理步和学习更新均为零，reset 后最大显存占用约 27.52%。首次 CPU 读取字段和首次 GPU 导入路径错误均在零步阶段停止并归档，修正后证据通过。[多技能前置报告](../../records/reports/multi-skill-training/README.md)

### 2026-11-14 · 滚翻 airborne-support 训练协议冻结

支持诊断复算了 21 个 head-top 失败，全部以空中前向旋转损失为主，未发现横向损失主导的样本。基于这一结果，候选只在累计前向旋转 30–240°、脚/头/机身均未接触且仍有前向角速度时增加 `roulade_airborne_support_penalty`，权重为 -0.35；控制组不改奖励。CPU 差异核验通过，GPU reset-only 零步在两组各 1024 环境通过，物理步和学习更新均为零，最大 reset 显存约 15.58%。[支持诊断](../../records/reports/roll-support-diagnosis/analysis.json) · [协议与前置记录](../../records/reports/roll-airborne-support/README.md)

协议固定为同一官方父模型、seed 1210、每组 50 次更新（critic-only 14 次、actor 36 次）、每组 1,228,800 条转移。训练使用周期完整片段和逐更新指标，遇到漂移、非有限状态、显存超限或配对退化立即停止，不自动重试或延长。训练后以 seed 1211 独立评估 128 回合，候选需达到 116/128 且比控制组多 4 个成功，同时不能新增头部穿透或关节限位失败。当前学习更新尚未开始，下一条记录将在控制组和候选组实际运行后追加。

### 2026-11-14 · 滚翻 airborne-support 配对训练与独立评估关闭

控制组与候选组在同一官方父模型和 seed 1210 下各完成 50 次更新，每组 1,228,800 条转移。控制组实际学习耗时约 105.95 秒，候选组约 106.76 秒；两组都通过 live import parity、初始 actor/critic 配对和 14 次 critic-only 检查，均保留最后完整保存点、ONNX、逐更新日志和 3 秒实际训练片段。候选唯一新增项是 30–240° 中段无接触前向旋转惩罚，训练时奖励项在候选日志中可见，控制组没有该项。[控制组运行](../../records/experiments/duck_roll_airborne-support_control_s1210_20261114T004321/state.json) · [候选组运行](../../records/experiments/duck_roll_airborne-support_candidate_s1210_20261114T004321/state.json)

独立 seed 1211 使用同一原始滚翻物理和 128 个环境，每个模型完整运行 9 秒并保留 env0 视频和全量轨迹。官方 108/128，控制组 112/128，候选 111/128。候选没有新增头部穿透或关节限位失败，但没有达到 116/128，且比控制组少 1 个成功；逐回合配对为控制组胜 14、候选胜 13、共同成功 98、共同失败 3。失败主要仍是 supported forward revolution 和及时两秒着地保持，不是新的安全回归。[评估总表](../../records/reports/roll-airborne-support/evaluation.json) · [失败分析](../../records/reports/roll-airborne-support/analysis.json) · [三组原片](../../records/reports/roll-airborne-support/README.md)

按冻结协议停止延长训练和 router 连续切换。官方滚翻、控制组和候选组均保留；候选标为失败研究证据，起身和行走继续使用已有交付。后续只有在新的轨迹分析形成不同机制假设后，才重新冻结下一轮滚翻方案。

### 2026-11-14 · 官方组件 router seed-1212 提前停止

滚翻候选失败复盘没有形成新的单一奖励机制，随后冻结官方滚翻与已验收行走组件的 `stand → roll → stand → forward` 连续开发协议。seed 1212 未在历史评估目录中出现；CPU 场景编译和 GPU reset-only 零步均通过，128 环境启动时物理步和学习更新为零。

实际 dynamics 运行 746 控制步、95,488 条转移和 3,730 个物理子步。完整滚翻窗口后 0/128 世界进入站立，0/128 进入前进，按早停规则停止。状态、执行日志、接触计数、700 步关键帧和部分原片保留；本轮不重试 seed 1212。 [router 记录](../../records/reports/continuous-roll-stand-walk-s1212/README.md) · [停止复盘](../../records/reports/continuous-roll-stand-walk-s1212/stop-review.json)

### 2026-11-14 · 起始站立与滚翻边界复盘

seed 1213 的 50 控制步准备度筛查显示第 50 步 128/128 个环境达到 25 步连续稳定门槛。seed 1214 的 50+450 控制步边界筛查显示 128/128 个环境进入 `ROLL`，但只有 1 个 latch 就绪；首次头部穿透集中在滚翻局部第 22 步的 bridge 窗口。 [准备度](../../records/reports/continuous-roll-stand-initial-screen-s1213/analysis.json) · [边界](../../records/reports/continuous-roll-stand-roll-boundary-s1214/mechanism-analysis.json)

### 2026-11-14 · bridge 配对消融完成

seed 1215 只改变 `cross_model_bridge_controls`。25 步 bridge 的头部穿透超限为 127/128，关闭 bridge 为 0/128；末尾完整状态分别为 1/128 和 128/128。该差异支持 bridge 是当前 router 的主要故障源，未启动训练或模型晋级。 [配对分析](../../records/reports/continuous-roll-bridge-ablation-s1215/analysis.json)

### 2026-11-14 · bridge 关闭后的完整滚翻边界完成

seed 1216 关闭 bridge 后运行 50+450 控制步，121/128 个环境达到滚翻→落地 latch，超过 116/128 开发门槛；最大头部穿透 0.0032455 m，关节限位和头部穿透超限均为 0。7 个未完成环境全部属于 supported rotation 不足（316.7°–325.9°），已生成失败清单，尚未进入站立接力。 [独立分析](../../records/reports/continuous-roll-bridge-disabled-boundary-s1216/analysis.json) · [失败复盘](../../records/reports/continuous-roll-bridge-disabled-boundary-s1216/failure-review.json) · [完整原片](../../records/evaluations/duck_switch_roll-stand-walk_development_s1216_20261114T021238/clip-step-0.mp4)

### 2026-11-14 · bridge 关闭滚翻接力完成

seed 1217 延长落地站立并记录前进相位入口。128/128 个环境进入 `ROLL`，116/128 个环境完成滚翻→站立，116/128 个环境进入前进相位；零学习、无重置，冻结完整路线前置接口。 [接力分析](../../records/reports/continuous-roll-stand-handoff-s1217/analysis.json) · [接力原片](../../records/evaluations/duck_router-roll-stand-handoff_development_s1217_20261114T024805/clip-step-0.mp4)

### 2026-11-14 · 完整无重置路由完成

seed 1218 执行 50 步站立、450 步滚翻、50 步落地站立和 1,000 步前进，恢复原有 20 秒前进评分。滚翻入口 128/128、落地 123/128、前进入口 123/128；完整前进 111/128，低于 116/128。5 个世界未进入前进，12 个世界仅偏航超限，暂不晋级。 [完整分析](../../records/reports/continuous-roll-full-sequence-s1218/analysis.json) · [完整原片](../../records/evaluations/duck_switch_roll-stand-walk_development_s1218_20261114T025305/clip-step-0.mp4)

### 2026-11-14 · 前进反馈上限固定控制探针完成

seed 1220 只把前进反馈上限从 0.12 提到 0.15 rad/s，其他组件和原评分保持不变。完整路线为滚翻入口 128/128、落地 127/128、前进入口 127/128、前进成功 127/128；前进偏航失败为 0。该结果通过开发门槛但不是模型晋级，下一步使用 seed 1221 独立确认。 [探针分析](../../records/reports/router-forward-feedback-probe-s1220/analysis.json) · [探针记录](../../records/reports/router-forward-feedback-probe-s1220/README.md)

### 2026-11-14 · 前进反馈独立确认完成

seed 1221 只更换随机流并复用 0.15 rad/s 固定反馈上限。滚翻入口 128/128、落地 121/128、前进入口 121/128、前进成功 121/128，超过 116/128；7 个滚翻 supported rotation 失败，前进偏航失败为 0。与 seed 1220 一起通过开发门槛，进入留出确认和轻量 PPO 设计。 [确认分析](../../records/reports/router-forward-feedback-confirmation-s1221/analysis.json) · [确认记录](../../records/reports/router-forward-feedback-confirmation-s1221/README.md)

### 2026-11-14 · 前进反馈一次性留出完成

seed 1222 复用 0.15 rad/s 固定反馈上限且不调参。滚翻入口 128/128、落地 125/128、前进入口 125/128、前进成功 124/128，超过 116/128；3 个滚翻 supported rotation 失败、1 个偏航失败。开发、独立确认和留出三组均通过完整路线门槛，固定参数进入候选路由冻结和轻量 PPO 设计。 [留出分析](../../records/reports/router-forward-feedback-holdout-s1222/analysis.json) · [留出记录](../../records/reports/router-forward-feedback-holdout-s1222/README.md)

### 2026-11-14 · 官方与自训练行走权重连续路由对比完成

seed 1223 同 seed 比较官方 walk ONNX 与已有自训练 `walk-common-base.onnx`。官方完整前进 119/128，自训练 111/128；自训练速度中位数 0.1452 m/s，官方 0.1627 m/s，自训练偏航失败 7 个。自训练权重未通过连续路由门槛，下一步冻结滚翻后前进状态的轻量 PPO 方案。 [对比记录](../../records/reports/router-self-trained-walk-compare-s1223/README.md)

### 2026-11-14 · 滚翻后前进状态课程提取完成

从 seed 1220、1221、1222 的真实连续轨迹提取 373 个前进入口状态及随后 50 个控制步，有限值、形状和 phase 回读通过，新增物理步和学习更新均为 0。以官方 actor 为父点的 5 次更新试运行协议已写入草案，尚未启动 PPO。 [课程报告](../../records/reports/route-aware-walk-training/README.md) · [课程清单](../../records/reports/route-aware-walk-training/forward-entry-curriculum.json)

官方 ONNX 对 373 个保存入口逐样本回读通过，最大动作误差 2.24×10⁻⁷；真实物理状态恢复和 GPU 零步仍是下一门槛。 [接口核验](../../records/reports/route-aware-walk-training/curriculum-check.json)

实际 CUDA walking 环境恢复 32 个入口 qpos/qvel 后，最大回读误差为 0，两个物理步计数保持 0；训练 pilot 的 CPU/GPU 零步协议进入冻结准备。 [CUDA 回读](../../records/reports/route-aware-walk-training/cuda-restore-readback.json)

5 次更新 route-aware pilot 已冻结，协议哈希为 `6867bddb001acf80c143c29766d1da1b8c0f8474ede8996404dd229e5f19214e`；学习更新仍为 0，下一步只执行冻结协议的零步和短 pilot。 [冻结协议](../../configs/experiments/route_aware_walk_training_s1224.json)

首轮 pilot 入口在构造 CUDA 环境和保存 `model_init.pt` 后因共享导入器缺少扁平兼容字段停止，物理步和学习更新均为 0。失败记录已保留，训练入口增加兼容适配后重新预检。 [入口失败记录](../../records/reports/route-aware-walk-training/entry-failures/initialization-missing-flat-fields.json)

兼容适配后的入口预检重新通过，run id 为 `duck_walk_route-aware-entry_candidate_s1224_20261114T045600`，协议哈希为 `8011a20194d2802b397c741d849c67ae35b57a3e5ff2d6ce05fd40b0da470876`；4096 环境、373 个课程入口、1 次 critic warmup 和 5 次 pilot 已核对，CUDA、物理步和学习更新仍为 0。 [入口预检](../../records/reports/route-aware-walk-training/pilot-config-preflight.json)

随后 pilot 按冻结预算完成 5 次更新，共 491,520 个采样步。末次采样约 23,232 steps/s，均值奖励由 1.63 增至 18.60；官方 actor 导入最大误差为 2.98×10⁻⁷，课程状态 qpos/qvel 回读误差为 0，检查点和 ONNX 已导出，等待 seed 1225 的原始完整路线评分。 [训练状态](../../records/experiments/duck_walk_route-aware-entry_candidate_s1224_20261114T045600/state.json) · [导出清单](../../artifacts/duck_walk_route-aware-entry_candidate_s1224_20261114T045600/manifest.json)

seed 1225 的 pilot 原始路线为 28/128，官方父模型同 seed 为 125/128；pilot 偏航失败 95 个、父模型为 0，速度中位数相近而偏航中位数相差 0.01316 rad/s。候选拒绝晋级，下一步只做动作输出、历史和归一化统计的离线诊断。 [对照汇总](../../records/reports/route-aware-pilot-parent-compare-s1225/README.md)

动作漂移离线回放显示，两模型在同一前进输入上的确定性动作平均绝对差 0.00272，最大差 0.01144；两条实际路线的前进观测平均绝对差 0.42335。小幅 actor 差异在无重置路线中被放大，未启动新的学习。 [漂移分析](../../records/reports/route-aware-pilot-parent-compare-s1225/actor-drift.json)

动作历史与归一化审计完成。两条路线普通非终止步的历史回读异常均为 0，重置行历史最大值为 0；异常只出现在滚翻结束后的 `FAILED` 世界。pilot 初始和终止 checkpoint 的归一化统计逐位不变，变化集中在 actor 权重和探索标准差。 [审计](../../records/reports/route-aware-pilot-parent-compare-s1225/history-normalization-audit.json)

低学习率 actor-drift 恢复探针已冻结：seed 1226、4096 环境、1 次 critic warmup、1 次 actor 更新，学习率为原 pilot 的四分之一；每次更新比较 64 个课程入口的确定性输出，超过门槛立即停止。CUDA 零步通过，物理步和学习更新均为 0。 [冻结协议](../../configs/experiments/route_aware_actor_drift_probe_s1226.json) · [零步](../../records/reports/route-aware-actor-drift-probe-s1226/probe-zero.json)

首轮探针完成第 1 次更新后因输出门禁读取课程输入维度错误停止，未进入第 2 次更新。已保留更新检查点和失败记录，修正后重跑同一冻结预算。 [失败记录](../../records/reports/route-aware-actor-drift-probe-s1226/probe-entry-failure.json)

修正后的探针完成两次更新，确定性输出门禁分别为平均 4.15×10⁻⁸/最大 2.09×10⁻⁷ 和平均 6.46×10⁻⁴/最大 1.78×10⁻³，均低于阈值；导出候选等待 seed 1228 原路线评分。 [门禁记录](../../records/experiments/duck_walk_route-aware-drift-probe_candidate_s1226_20261114T061500/output-drift-update-2.json)

### 2026-11-14 · route-aware 独立 seed-1229 对照完成

seed 1229 在同一 bridge-disabled `roll → stand → forward` 路线、0.15 rad/s 反馈上限和原 116/128 门槛下完成候选与官方父模型对照，双方均为零学习、128 个环境和 1,550 个控制步。候选滚翻落地、前进入口、完整前进均为 119/128，官方父模型均为 121/128；两组偏航失败、头部穿透超限和关节限位失败均为 0。候选通过独立路线门槛，但没有优于父模型，保留权重、轨迹、原片和分析作为复现证据，关闭该学习线，不追加训练。 [对照汇总](../../records/reports/route-aware-probe-parent-compare-s1229/README.md) · [候选原片](../../records/training-archive/evaluations/duck_switch_roll-stand-walk_route-aware-probe_candidate_s1229_20261114T063125/clip-step-0.mp4) · [父模型原片](../../records/training-archive/evaluations/duck_switch_roll-stand-walk_route-aware-probe_parent_s1229_20261114T063932/clip-step-0.mp4)

### 2026-11-14 · route-aware holdout seed-1227 完成

冻结探针协议声明的 seed 1227 留出评估完成。候选滚翻落地、前进入口、完整前进均为 124/128，官方父模型均为 121/128；两组偏航失败、头部穿透超限和关节限位失败均为 0。将 seed 1227、1228、1229 合并后，候选为 367/384，官方为 365/384，但逐种子差异为 +3、+1、−2，收益不稳定。因此候选只作为开发证据保留，当前 route-aware 学习线关闭；新训练必须先提出可测机制并重新完成零步、短窗口和独立路线门禁。 [留出对照汇总](../../records/reports/route-aware-probe-parent-compare-s1227/README.md) · [候选原片](../../records/training-archive/evaluations/duck_switch_roll-stand-walk_route-aware-probe_candidate_s1227_20261114T065157/clip-step-0.mp4) · [父模型原片](../../records/training-archive/evaluations/duck_switch_roll-stand-walk_route-aware-probe_parent_s1227_20261114T065653/clip-step-0.mp4)

### 2026-11-14 · route-aware 教师约束短训与留出完成

新的教师约束草案先在 373 个真实前进入口上通过 CPU admission：官方 ONNX 与归档动作最大差 2.24×10⁻⁷，教师锚定损失 8.35×10⁻¹²，系数为 0 时保持原 policy loss，正系数产生额外 actor 梯度。固定协议随后通过 4096 环境 GPU 零步和课程 qpos/qvel 回读，完成 3 次更新（1 次 critic warmup、2 次教师锚点 actor 步骤）。实际采样约 16,692–24,007 steps/s，输出漂移检查均通过；最后一次均值漂移 9.28×10⁻⁴、最大 2.49×10⁻³，导出 ONNX 与 checkpoint 的 373 行 CPU 回放最大差 2.53×10⁻⁷。入口配置遗漏任务名和共享 bootstrap 兼容字段的两次失败均在物理步 0 前保留并修正。 [训练协议](../../configs/experiments/route_aware_teacher_anchor_s1230.json) · [训练记录](../../records/experiments/duck_walk_route-aware-teacher-anchor_candidate_s1230_20261114T071414/state.json) · [导出复核](../../records/reports/route-aware-teacher-anchor-s1230/export-verification.json)

seed 1231 开发路线候选 121/128、官方 122/128；seed 1232 留出候选 122/128、官方 121/128。两组均无偏航失败、头部穿透超限和关节限位失败，合计均为 243/256，逐种子差异为 −1、+1。教师约束候选通过路线门槛但没有稳定相对优势，不晋级、不追加训练；官方模型作为后续硬件准备的仿真基线，教师约束 ONNX 和 checkpoint 作为可复现实验产物保留。 [开发对照](../../records/reports/route-aware-teacher-anchor-parent-compare-s1231/README.md) · [留出对照](../../records/reports/route-aware-teacher-anchor-parent-compare-s1232/README.md) · [模型选择清单](../../artifacts/delivery/simulation-model-selection.json)

### 2026-11-14 · route-aware 训练入口预检通过

统一冻结协议中的 critic warmup 字段为 1 次，并修正训练入口的运行名和课程清单路径。Ubuntu-24.04 CUDA 入口预检成功构造 4096 环境和 373 个课程入口，`cuda_initialized=false`，学习更新和物理步均为 0；协议哈希为 `b124c9e3ea0602fd79bfa6d932220917888b93044d5c9b388b940c4c64d32edd`。下一步才允许启动严格 5 次更新。 [入口预检](../../records/reports/route-aware-walk-training/pilot-config-preflight.json) · [冻结协议](../../configs/experiments/route_aware_walk_training_s1224.json)

### 2026-11-14 · 离线真机准备审计完成

S19 只读核验了后续上机所需的官方模型哈希、ONNX 接口、策略关节顺序、软件映射、15 行标定模板和回退材料。官方行走参照加载为 `[1,61] → [1,14]`，软件准备通过；实体舵机 ID、机械中心、方向、IMU、通信和供电仍保持待现场测量，未发生硬件 IO。 [审计记录](../../records/hardware/deployment-preparation/s19-readiness.md) · [机器摘要](../../records/hardware/deployment-preparation/s19-readiness.json)

### 2026-11-14 · 参数随机化筛查暂停

S20 固定官方滚翻/行走控制器，预留八个摩擦/电压组合，每臂为固定参数，零学习、零硬件 IO。首次启动因继承协议保存了过期的验证脚本哈希，在零物理步处停止；刷新当前脚本哈希后继续，失败记录保留。六个参数臂完整完成：低摩擦三个电压点的滚翻完成为 110/128、108/128、112/128；中摩擦/低电压 125/128、高摩擦/低电压 124/128；中摩擦/高电压滚翻 117/128 但连续路线 25/128。用户要求暂停时，高摩擦/名义电压臂已完成 1400/1550 控制步，保留 29 张部分帧、状态、日志和停止记录；部分视频缺少索引，未写出完整轨迹或可恢复检查点，最后一个参数臂未开始。 [暂停汇总](../../records/reports/roll-domain-parameter-random-s20/README.md) · [媒体检查](../../records/reports/roll-domain-parameter-random-s20/partial-media-audit.json) · [机器摘要](../../records/reports/roll-domain-parameter-random-s20/pause-summary.json) · [停止记录](../../records/reports/roll-domain-parameter-random-s20/stop-record.json) · [首次预检失败](../../records/reports/roll-domain-parameter-random-s20/failure-first-run.json)

### 2026-11-14 · 暂停期间离线复核

六个 S20 参数臂从已保存的原始轨迹在 CPU 重新评分，结果与原记录一致。低摩擦三臂进入前进后偏航超限分别为 37、108、112 例，中摩擦/高电压为 92 例。S20 的滚翻入口没有目标过渡，但落地站立阶段仍保留 25 步过渡；S17 保存指令为 0 步，两者不能直接作为只改变物理参数的对照。三臂 `early_stop` 字段为事后诊断，原运行均采集到 1,550 步，外层循环没有按评分安全结果阻止下一臂。未发现六臂的头部穿透或关节包络超限。

独立停止判定在六份原轨迹和构造的非有限值、关节越限、头部穿透及臂间错误上通过 CPU 核验，尚未接入实际运行入口。暂停标记已在项目状态的多个活动字段中对齐，旧值保存在状态复核记录。训练和仿真继续暂停。 [离线复核](../../records/reports/roll-domain-parameter-random-s20/offline-review.md) · [停止判定核验](../../records/reports/roll-domain-parameter-random-s20/stop-guard-offline-verification.json) · [状态复核](../../records/reports/goal-review/s20-status-reconciliation.json)

### 2026-11-14 · 有界入口离线接线

在保持训练和仿真暂停的前提下，新增独立名义入口、收集器及停止循环。六份保存轨迹的回放中，三个低摩擦臂在第 500 步停止；假环境检查确认保存 501 个状态与 500 个控制，没有调用下一步。构造的头部穿透、无效四元数、暂停、运行故障和观察器故障均停止并留档；实际暂停状态下入口在导入训练环境前拒绝运行。该入口只接受名义参数，参数注入、多臂调用、真实观察入口和 CUDA 零步仍未验证，未启动物理或学习。 [入口记录](../../records/reports/roll-domain-parameter-random-s20/guarded-entry.md) · [收集器核验](../../records/reports/roll-domain-parameter-random-s20/guarded-collector-offline-verification.json) · [准入核验](../../records/reports/roll-domain-parameter-random-s20/guarded-nominal-entry-offline-verification.json)

### 2026-11-14 · 暂停进度归档

按用户要求继续暂停训练与仿真，将环境、已验收和未通过的动作、模型、最近筛查中断位置及后续条件整理为一个入口。文本与代码建立快照，原模型、轨迹和视频保留在原工程目录，并以 SHA-256 清单关联；不将快照称为脱离原工程的完整备份。 [进度与归档](../../records/reports/progress-archive-20261114/README.md)

### 2026-11-14 · 归档完整性保护

原归档核验要求工程中的源文件始终与快照相同，后续正常更新会导致误报；原创建入口还可能覆盖已有快照。已增加独立检查，将归档成员和保留大文件的完整性与当前源文件变化分开报告，并拒绝覆盖已有归档。临时文件中的损坏、缺失及路径检查和真实覆盖拒绝均通过；原快照、清单和首次核验记录保持不变。 [归档检查](../../records/reports/archive-integrity-followup/README.md)

### 2026-11-14 · 滚翻参数配置离线核验

在训练和仿真暂停期间，新增单因素配置适配器，拒绝双因素误标及非法值。标称、低摩擦和高电压三个真实配置的标准化字段差异分别为 0、2、2，两处变化对应同一范围的上下限；S17 冻结的 52 个来源文件哈希一致。首次检查因枚举序列化失败，保留退出记录并修复后通过。没有构建场景或环境，也没有执行物理步；运行时注入、配对初态、多臂停止和 CUDA 零步仍未验证。 [配置检查](../../records/reports/roll-factor-configuration/README.md)

### 2026-11-14 · 参数回调绑定修复

离线复现发现，参数回调创建后若调用方修改原字典，低摩擦回调会变成高电压配置。回调现保存独立的已验证参数、标称值和范围；新增夹具先修改调用方字典，再确认实际仍应用原低摩擦配置。修复后 19 项夹具检查和真实配置对象核验通过，S17 的 52 个冻结来源仍一致。原问题与修复前结果保留，未启动场景、物理步或训练。 [问题与核验](../../records/reports/roll-factor-configuration/README.md)

### 2026-11-14 · 参数筛查初态复核

暂停期间只读取 S20 七组零步和采样起点：保存的 128 环境姿态、速度、观测及 32 环境策略接口输出一致；六个完整组采样首条接触记录与各自零步一致。独立核对了原比较器跳过的摩擦数组，记录了几何索引差异。高摩擦／名义电压组的采样起点存在，但完整接触轨迹缺失；未保存的跨组随机状态、执行器内部状态及 S17 控制器等价性仍不能推断。没有新增物理步或训练。 [初态复核](../../records/reports/roll-s20-saved-startup-review/README.md)

### 2026-11-14 · 摩擦字段精确校验

针对旧比较器跳过全部摩擦字段的问题，新增静态字段和重置后字段的精确比较。使用七组零步及采样起点共 14 份配置和构造异常进行 CPU 检查；首轮发现双方同时缺字段时会误通过，保留问题及源码后，固定 33 个静态字段和 3 个重置字段，32 项异常和绑定检查通过。夹具中的几何索引与名称对应尚未在实际模型验证，代码也未接入运行入口；没有新增物理步或训练。 [摩擦字段校验](../../records/reports/roll-selective-friction-guard/README.md)

### 2026-11-14 · 参数与场景守卫适配

在训练和仿真暂停期间，新增参数配置与精确摩擦检查的适配层。先在假场景完成 21 项绑定、异常与恢复检查，再直接调用原场景守卫的安装函数，以假环境执行 8 项流程检查；真实策略接口使用替身。校验被放到重置返回之前，同一上下文不自动重试，参考文件需反复核对哈希。S17 的 52 个固定源码和依赖文件经只读检查一致。实际训练入口、真实模型映射、观察器和 CUDA 零步仍未验证。 [适配与验证](../../records/reports/roll-factor-guard-adapter/README.md)


## 2026-11-16 观察与恢复准备记录

以下保留各项准备工作当时的记录；当前结果与下一步以主报告首页为准。

# XGO-Duck 开发与部署报告

更新日期 2026-11-16。

| 需要查看的内容 | 入口 |
| --- | --- |
| 当前阶段、暂停状态和剩余目标 | [进度与归档](../../records/reports/pause-archive-20261115/README.md) · [当前目标](../current_goal.md) |
| 环境部署、镜像源和恢复条件 | [电脑环境与训练](../runbooks/pc_training.md) |
| 官方模型、自训模型和修改结果 | [权重与迭代记录](../../artifacts/delivery/iteration_record.md) |
| 训练曲线、真实录像与评估 | [历史训练对照](../../records/reports/historical-getup-training-comparison/README.md) · [最新官方路线录像](../../records/reports/duck_roll-native_nominal_s1241_20261115T230228/evaluation-view.mp4) |
| 后续舵机标定与真机准备 | [真机标定手册](../runbooks/hardware_commissioning.md) |

起身留出的汇总现已按冻结判据和保存轨迹核对：候选实际右侧相对两组参照净增 4、5 例，满足各净增至少 1 例的原门槛；总体分别少 4、5 例。此前部分汇总把总体差值误当作右侧门槛，已更正。该结果只证明本次四姿态留出通过，不代表完整技能组合或真机部署通过。 [判据与轨迹复算](../../records/reports/getup-gate-clarification/README.md)

本报告的本地链接、图片和视频引用已经过文件检查；内置浏览器拒绝直接打开 `file:` 地址，本次没有核验该报告的浏览器排版和媒体播放。 [媒体检查记录](../../records/reports/report-media-check.json)

用户已恢复全阶段目标。本机训练网页无法访问，现改用 VS Code 直接查看本地状态、曲线和录像，窗口位置与大小保持不变。第一轮 1,536 与 3,072 环境的实际 CUDA 零步均通过，初始保存点逐字节一致。[本轮方案和证据](../../records/reports/getup-observed-throughput-20261116/README.md)

首次实际短训完成 24 控制步、73,728 条转移后因观察数据格式校验停止；只有价值网络执行了 120 次小批量更新，策略梯度更新为 0，对照组未启动。已用首条真实指标复现 Python 元组写入 JSON 后变成列表的问题，并通过独立适配入口的 5 项离线检查。原失败保留，新协议先核验匹配的 CUDA 零步，不把离线修复当作实际训练完成。[修复核验](../../records/reports/getup-observed-throughput-20261116/publication-verification.json) · [原执行日志](../../records/reports/getup-observed-throughput-20261116/training/throughput/execution.log)

此前统一入口的 31 项合成进程检查仍保留；这次实际运行说明合成数据未覆盖真实指标中的元组类型。训练结束后的曲线由最终保存记录导出，避免短任务结束时遗漏最后一条指标。新入口仍核对继承来源、WSL 依赖和本次查看确认。[入口与原检查](../../records/reports/native-ppo-observation-preflight/observed-entry-check/README.md) · [本地查看方法](../runbooks/pc_training.md#恢复前先检查什么)

最新的官方标称连续路线评估已完成：123/128 达到滚翻、站立和前进的门槛，录像与保存轨迹已独立复核。VS Code 显示了本次实际画面和速度曲线；训练观察及多参数配对尚未完成。环境 0 的倒立关键帧中头部接触地面、双脚离地，该记录没有测量接触力。 [结果与录像](../../records/reports/duck_roll-native_nominal_s1241_20261115T230228/README.md) · [接触复核](../../records/reports/duck_roll-native_nominal_s1241_20261115T230228/roll-contact-review.json)

5 个未通过环境都进入了滚翻，机身累计旋转接近 359°，但原判据折算的有效支撑旋转在窗口结束时仅约 303°–325°，没有达到 330° 的完成条件。这是保存轨迹的描述，不能直接当作训练收益或真机表现。 [失败复盘](../../records/reports/duck_roll-native_nominal_s1241_20261115T230228/failure-review.json)

控制步边界未计入支撑的前向旋转中位数，失败组为 35.21°，已进入站立组为 7.47°；两组样本量和结束时点不同。本批使用前向更新后的边界接触读数，并与同一时点的约束接触核对，但没有记录边界之间各物理子步的受力。另一批历史高频记录显示，控制边界没有接触不代表该区间所有物理子步都没有接触；先前空中旋转惩罚的训练候选未晋级。恢复后先检查接触采样时序，再决定是否修改训练。 [本批分解](../../records/reports/duck_roll-native_nominal_s1241_20261115T230228/support-gap-analysis.json) · [历史高频记录](../../records/reports/roll-contact-clock/analysis.json) · [旧训练对照](../../records/reports/roll-airborne-support/README.md)

![官方标称路线关键帧](../../records/reports/duck_roll-native_nominal_s1241_20261115T230228/keyframes.jpg)

<video controls preload="none" src="../../records/reports/duck_roll-native_nominal_s1241_20261115T230228/evaluation-view.mp4"></video>

最近进度及本机观察记录见[进度归档](../../records/reports/progress-archive-20261115/README.md)；模型、训练轨迹和录像仍保留在工程原路径，归档快照只包含所列文件。

本机观察检查中，VS Code 的状态文件从第 1 条真实预检样本自动刷新到第 2 条；第 3 条由 WSL 写入，也被 Windows 的 VS Code 自动显示。历史仿真录像播放到了末尾。该测试未运行训练或仿真，也未证明新训练的曲线和画面可用；新运行仍须单独验收。 [检查步骤与截图](../../records/reports/native-observer-preflight/README.md)

已从一份归档训练的 17 次更新读取回报、价值损失与采样速度，生成可在 VS Code 打开的历史观察示例，并保留原始数值及录像链接。最后一次价值损失较此前明显升高，是需要结合具体训练过程复查的线索，不足以单独说明动作效果或修改收益。 [历史观察示例](../../records/reports/native-observer-preflight/historical-example.md)

暂停期间整理了同父点、等样本的两组历史起身训练：1,536 与 3,072 环境的全程吞吐分别为 7,273 和 7,564 转移/秒，整卡显存峰值分别为 7,357 和 12,637 MiB；两份训练模型均未通过该轮全部姿态门槛。逐次曲线、GPU 采样、完整录像和评估结果可在 VS Code 本地查看。 [训练过程对照](../../records/reports/historical-getup-training-comparison/README.md) · [曲线](../../records/reports/historical-getup-training-comparison/training-curves.png)

PPO 本机观察桥与起身工作器的发布、停止回调已用测试子进程和合成状态检查；真实训练接入及中断保存点恢复尚未验收。历史中断样本的部分点标记 42 次完整更新，但优化器计数已进入第 43 次，不能自动当作完整更新恢复；同一文件也没有保存完整物理现场。 [接线与保存点检查](../../records/reports/native-ppo-observation-preflight/README.md)

对 9 份交付或研究保存点完成 CPU 内容核对；五个当前仿真组件的策略、价值网络和优化器状态可按各自结构加载，保存观测上的策略输出与对应 ONNX 的最大差不超过 1.08×10⁻⁶。起身组件的价值网络输入有 75 和 76 维两种，行走为 77 维。检查没有执行恢复后的训练，完整物理状态续跑仍未验证。 [保存点检查](../../records/reports/native-ppo-observation-preflight/README.md#保存点边界)

外层训练监督器已在隔离测试中检查暂停、冻结命令、训练锁、进度停滞、运行时间及退出信号，并与实际观察子进程完成一次合成接线。强制结束合成监督进程后，测试工作进程退出并释放锁；22 项检查通过。它还没有接入真实训练入口，异常退出后的真实保存点不可视为已验证。 [监督范围与测试](../../records/reports/native-ppo-observation-preflight/README.md) · [检查结果](../../records/reports/native-ppo-observation-preflight/parent-exit-verification.json)

进一步的合成进程测试中，旧观察接线在监督器强制退出后约 32.93 秒才标记来源过期。加入工作进程身份检查后，三次模拟分别在 1.86、1.97、1.86 秒后标记异常并退出；正常观察和监督停止的隔离回归通过。实际延迟受刷新和渲染影响，真实训练中的观察与异常停止仍待验收。 [新检查](../../records/reports/native-ppo-observation-preflight/observer-owner-guard-verification.json) · [原检查](../../records/reports/native-ppo-observation-preflight/observer-parent-loss-verification.json)

早期本机观察适配器已通过归档数据与故障样本的离线核验，能显示实际指标、原始帧及来源时间；错误编号、过期记录、无效数值和不安全的媒体路径会产生错误。当时的适配器未连接训练控制器，不能自行停训；后来准备了观察桥、回调和监督器，但新运行仍须单独验证进程、心跳、动作守卫及记录。 [查看方法与核验](../../records/reports/native-observer-preflight/README.md) · [后续接线](../../records/reports/native-ppo-observation-preflight/README.md)

观察器进程与心跳的独立判定已用测试子进程核验，进程退出或来源过期时会拒绝继续。该结果尚未接入训练循环，因此不表示真实训练已具有自动停训能力。 [进程与心跳核验](../../records/reports/native-observer-preflight/observer-guard-verification.json)

已有滚翻停止循环与本机观察器的接线已在保存轨迹上完成 6 项离线检查：观察器退出或用户暂停后不会推进下一步，头部碰撞超限的异常状态得到保留。此结果尚不代表真实仿真或训练已经接入。 [离线接线核验](../../records/reports/native-observer-preflight/guarded-loop-verification.json)

零步方案涉及的工程文件、WSL 训练与依赖源码、本机观察证据已重新核对哈希。原浏览器准入仍未通过，来源核验不代表本机协调器或真实 CUDA 零步已经通过。 [来源核验](../../records/reports/native-observer-preflight/native-zero-provenance.json)

本机观察的零步检查实际启动后，在执行器字段类型检查处停止；运行契约记录控制步和物理子步均为 0，接口检查尚未完成。同版 MuJoCo 的 CPU 小模型显示限力标志为 `bool`，可复现原检查器只接受整数的报错；独立兼容候选已通过 CPU 检查，真实设备模型的标志和力限值仍待读回。本次完整控制台日志未保存，后续需补齐。 [失败、诊断与候选](../../records/reports/native-observer-preflight/zero-attempt-20261115.md)

随后使用单独冻结的零步方案读取真实模型，保留原失败和冻结源码。14 个限力标志均为 `bool` 且已启用；按 7.7 V 和冻结的电机参数计算，编译限力为 ±0.97816 N·m。32 个接口样本的最大输出差为 5.96×10⁻⁷；128 环境的两脚摩擦为 1.0，控制步和物理子步均为 0。VS Code 显示了该次运行状态，完整输出和退出码已保存。原失败没有保存实际标志，故其原因仍属推断；此项不代表动态抗扰或训练通过。 [独立复核](../../records/reports/duck_roll-factor_nominal_s1241_20261115T224902/independent-verification.json) · [完整日志](../../records/reports/duck_roll-factor_nominal_s1241_20261115T224902/native-zero-execution.log) · [本机截图](../../records/reports/duck_roll-factor_nominal_s1241_20261115T224902/native-zero-run/ui-check.jpg)

![本机状态检查](../../records/reports/native-observer-preflight/status-check.jpg)

![历史录像播放](../../records/reports/native-observer-preflight/video-check.jpg)

已有滚翻快照曾在同一物理模型内恢复六个分支，305 项状态字段包括随机数、执行器和延迟缓冲。这份历史快照发生在第 58 步，不能直接证明新摩擦或电压组具有相同的零步初态；后续需在新方案中固定源码，并核验实际参数与未解释的状态差异。 [快照复用核对](../../records/reports/resume-20261115/paired-snapshot-audit.json)

归档检查保留原始快照与首次核验记录，区分快照损坏、工程文件后续变化和未打包的大文件缺失；重复创建不会覆盖原归档。 [归档完整性检查](../../records/reports/archive-integrity-followup/README.md)

训练资料盘点显示，训练归档约 65.92 GiB，主要是用于复算和分析的轨迹。部分相同内容同时承担实验快照和模型恢复用途，本次没有删除文件。最新目标、报告、盘点和离线检查已补充归档；完整轨迹、模型和视频仍在工程原位置。 [存储盘点](../../records/reports/storage-review-20261114/README.md) · [当前进度补充归档](../../records/reports/progress-archive-20261114/current-state/README.md)

Windows 训练归档的 21,135 个文件现已逐项记录路径、大小与 SHA-256；全部路径和 21 个样本复核通过。与首次目录归档相比，10,149 项一致，另一个最近评估状态文件发生变化。该清单不覆盖 WSL 文件系统，也不能替代异地备份。 [训练归档文件清单](../../records/reports/progress-archive-20261114/training-files/README.md)

WSL 早期训练日志的 398 个文件也已逐项核验；其中 202 个未在所查项目目录找到同字节副本，已集中复制并核对归档，原件保留。现有 TensorBoard 已离线读出一份历史滚翻训练的奖励、损失和吞吐标量；网页服务未在本次启动。 [WSL 日志、归档与曲线查看](../../records/reports/progress-archive-20261114/wsl-training-logs/README.md)

共同接口的站立与六方向运动、四姿态起身和官方前滚翻均已通过独立仿真验收。恢复转向反馈后，起身接站立与七指令的完整留出通过 2,992/3,072（97.40%），七项均达到各自门槛。真机处于离线准备阶段。 [独立留出与原片](../../records/reports/continuous-routed-turn-feedback-holdout-zero-recovery/README.md) · [仿真交付](../../artifacts/delivery/routed-seven-command-simulation.json)

行走方案按指令选用策略，并在前进时按高置信度的偏航方向小幅修正关节目标。两批 20 秒开发和最终留出分别为 223/224、224/224、224/224，均通过原七指令门槛。前进偏航相对两次原反馈在三批均有下降。仿真交付保存实际使用的官方参照、两组自训 ONNX 和对应保存点。 [最终留出](../../records/reports/common-walk-right-support-yaw-training/yaw-direction-holdout/analysis.json) · [权重与配置清单](../../artifacts/delivery/walking-simulation.json)

本报告说明怎样搭建环境、运行仿真、修改训练、判断效果和准备真机迁移。操作入口见第 4 节，判据见第 5 节，修改及效果见第 6–7 节。逐次命令、故障和历史媒体保留在[完整过程记录](development_history.md)。现有模型按动作归入[训练产物保留索引](../../records/reports/training-curation/README.md)。

最新整理先完成吞吐与观察器配方冻结，再完成 S2–S6 阶段验收矩阵。S12 的滚翻扰动复盘和实际 CUDA 零步已通过。S13 新种子三组短评估中，参考 124/128，左推候选 120/128，右推候选 122/128，均过原门槛；但候选触发器左右两组均覆盖 128/128 个环境。S14 固定阈值筛查未形成选择性规则，S14b 的 109 个因果特征留一环境审计也未达到准入线。S15 对 seed-1218 全路线做了保存数据审计，S16 新 seed-1234 的三臂原门槛通过但留一审计未准入。S17 官方 seed-1235 固定路线通过，S18 分别核对四组参数端点和三臂名义扰动；参数端点与 S17 的落地过渡不同，统一控制器的参数抗扰尚未通过；随机化训练收益和可迁移触发器仍未准入。S19 完成了离线真机准备审计：官方模型哈希、ONNX 接口、策略关节顺序、软件映射、15 行标定模板和回退材料均通过，实体设备的连接、标定和使能继续延期。S20 固定策略参数随机化筛查已完成六个臂，低摩擦三个电压点分别为 110/128、108/128、112/128，中摩擦/低电压 125/128，高摩擦/低电压 124/128；中摩擦/高电压完整评分为滚翻 117/128、连续路线 25/128。高摩擦/名义电压臂在 1400/1550 步时暂停。阶段未完成，不能宣称参数区间通过。 [阶段验收矩阵](../../records/reports/stage-acceptance-matrix/README.md) · [S18 汇总](../../records/reports/robustness-evidence-s18.md) · [S19 准备审计](../../records/hardware/deployment-preparation/s19-readiness.md) · [S20 暂停汇总](../../records/reports/roll-domain-parameter-random-s20/README.md) · [S17 报告](../../records/reports/official-route-independent-s1235/README.md)

[TOC]

## 1. 当前进展

### 电机力限值检查

零步入口现会在重置前读取 14 个受控电机的传动映射、力限值开关和上下限，并与冻结的电机参数及声明电压比较；实际运行时保存原始读数和哈希。同时补充 BAM、执行器和延迟缓冲等源码的冻结要求。43 项入口假数据、8 项原守卫假环境及 3 项原入口来源检查通过；这些结果没有验证真实模型或 CUDA 零步，也不能证明步态改善。 [修改与验证](../../records/reports/force-limit-guard/README.md)

### 参数对照的初态与源码

七组筛查均以 1176 构造环境、以 1240 重置；官方路线两处均为 1235。历史起点保存了姿态、速度和观测等字段，但没有完整随机数或执行器内部状态。当前 BAM 源码表明电压范围上限参与计算编译模型的电机力限值，而已有 33 项静态字段不包含该限值：按当前源码与电机参数推算，7.7 V 和 8.0 V 分别约为 0.9782、1.0163 N·m，历史实际模型值未保存。官方路线的训练与依赖哈希清单未列出本次核查的七项相关源码，不能把当前源码快照当作历史运行的直接证明。 [来源、边界与后续采集](../../records/reports/matched-start-source-audit/README.md)

### 单因素零步入口

参数配置与摩擦守卫已接入独立零步入口。此前 34 项假数据检查和 6 项经过原场景守卫的假环境检查及其源码保留，覆盖暂停、因素、来源、输出隔离和禁止物理步；后续补充的电机力限值检查见上文。当前动态采样不受支持，真实策略接口、模型映射、观察器及 CUDA 零步尚未核验。 [此前接线记录](../../records/reports/roll-factor-zero-entry/README.md)

### 参数与场景守卫适配

参数配置和精确摩擦检查已通过假环境串联：静态、重置主字段与全环境扩展字段在重置返回前检查；异常后恢复原有绑定，同一适配上下文不重试环境创建。21 项假场景检查和 8 项原守卫假流程检查通过；真实策略接口使用替身，实际执行入口、模型映射、观察器及 CUDA 零步尚未验证。 [适配与边界](../../records/reports/roll-factor-guard-adapter/README.md)

### 摩擦字段校验

针对旧参数筛查跳过全部摩擦字段的问题，新增独立校验：静态模型的 33 个字段须齐全且一致，重置后 3 个物理字段须齐全，仅允许指定几何的滑动摩擦变化。14 份保存配置和 32 项异常、绑定检查通过。首轮双方同时缺字段仍可通过的问题已记录并修复。当前几何索引只用于夹具，真实模型名称映射和运行入口尚未验证。 [摩擦校验](../../records/reports/roll-selective-friction-guard/README.md)

### 参数筛查的初态复核

S20 七组保存的零步检查点与采样起点，在已记录的 128 个环境的姿态、速度、观测和默认姿态上逐项一致；接口中保存的策略输出覆盖前 32 个环境。六个完整采样组的首条接触记录与各自零步一致，中断组没有完整接触轨迹。原比较器跳过了摩擦字段，但回执仍将其列为已比较；现已独立核对保存的摩擦数组。完整随机状态、执行器内部状态和 S17 控制器等价性仍未核验。 [初态复核](../../records/reports/roll-s20-saved-startup-review/README.md)

### 滚翻参数配置检查

在暂停期间，按 S17 场景配置准备标称、低摩擦和高电压三组；真实配置对象的完整标准化字段比较分别发现 0、2、2 处变化，后两组的两处分别是同一范围的上下限。S17 冻结的 52 个工程、训练源码和依赖文件哈希一致；19 项 CPU 夹具检查通过。枚举序列化和回调受外部字典影响的问题均已记录并修复。参数已接入后续零步入口，配对初态、真实观察入口、CUDA 零步和抗扰效果尚未验证。 [配置核验](../../records/reports/roll-factor-configuration/README.md)

### 离线真机准备

S19 已将后续上机所需的材料固定为可审查入口。官方行走参照加载后为 `[1,61] → [1,14]`，元数据保留策略关节顺序、默认姿态、观测布局和动作尺度；软件关节映射的 14 个策略输出与 15 个舵机记录表均通过离线检查。当前不把这些结果解释为实体标定通过：舵机 ID、机械中心、方向、IMU 安装变换、通信和供电仍需要现场逐项记录。 [审计结果](../../records/hardware/deployment-preparation/s19-readiness.md) · [机器记录](../../records/hardware/deployment-preparation/s19-readiness.json) · [标定手册](../runbooks/hardware_commissioning.md)

### 参数随机化筛查暂停

S20 只使用官方固定策略，保持该批次的控制器、动作投影、20 ms 控制周期和评分器不变，八个固定摩擦/电压组合中六个已经完整评分；这不是逐环境随机采样。低摩擦条件在 0.8 摩擦、7.4/7.7/8.0 电压下均未达到滚翻门槛；1.05 摩擦/7.4 电压和 1.3 摩擦/7.4 电压分别通过 125/128 和 124/128。1.05 摩擦/8.0 电压虽完成滚翻 117/128，但连续路线只有 25/128。用户要求暂停时，1.3 摩擦/7.7 电压已运行到 1400/1550 控制步；保存了 29 张部分帧、状态、日志和停止原因，部分视频未封装完成，没有完整轨迹或可恢复物理检查点，训练更新数为零。 [暂停汇总](../../records/reports/roll-domain-parameter-random-s20/README.md) · [媒体检查](../../records/reports/roll-domain-parameter-random-s20/partial-media-audit.json) · [机器摘要](../../records/reports/roll-domain-parameter-random-s20/pause-summary.json) · [停止记录](../../records/reports/roll-domain-parameter-random-s20/stop-record.json)

六组已从原始轨迹重新评分。低摩擦组中，进入前进后的偏航超限分别为 37、108、112 例；中摩擦/高电压为 92 例。S20 在落地站立阶段使用 25 步目标过渡，S17 已验收路线为 0 步，不能将二者直接作为只改变物理参数的对照。低摩擦组的 `early_stop` 是事后诊断，实际均采集完整 1,550 步；历史外层循环也未接入评分后的安全停止。新的名义入口与收集器已接入停止判定，在保存轨迹和假环境中验证暂停、异常与臂间停止；当前暂停状态能在导入训练环境前阻止运行。该入口不支持参数扰动，真实观察入口、CUDA 零步和 GPU 停止行为均未验证。 [离线复核](../../records/reports/roll-domain-parameter-random-s20/offline-review.md) · [入口验证](../../records/reports/roll-domain-parameter-random-s20/guarded-entry.md)

滚翻六组同初态对照已完成。无扰动、左推、右推原控制完整序列为 122/128、117/128、126/128。头部脉冲为 119/128、119/128、124/128。候选的无扰动和右推标称成绩下降，关闭头部脉冲方案。采样完成 1,153,280 条转移，显存峰值 2,530 MiB，独立核验确认 305 项恢复字段和六组首步一致性。 [六组报告与原片](../../records/reports/continuous-roll-head-support-pulse/resume-20261113/README.md) · [独立分析](../../records/reports/continuous-roll-head-support-pulse/resume-20261113/analysis.json)

参数端点验证早期曾因运行入口使用旧的 `PolicyControl` 而得到低摩擦 3/128，该结果被隔离为失败记录。后续改用 `ProjectedRollPolicy` 完成端点筛查和交叉复验；S20 离线复核进一步发现该控制器在落地阶段仍与 S17 无过渡路线不同，不能把不同控制器的成绩直接合并。 [失败记录](../../records/reports/roll-domain-parameter-screen/preflight-failures/README.md) · [控制器复核](../../records/reports/roll-domain-parameter-random-s20/offline-review.md)

原行走环境的一秒平均速度、瞬时偏航和姿态标准已通过。共同接口的最终留出中，站立、前后走、左右转和左右移各通过 32/32。早期自训模型曾持续髋内收、双脚靠拢或交叉，因此评分单独检查姿态和足间距。共同物理环境中的连续切换另行验收。

提高实际仰卧释放后首秒的 PPO 策略损失权重，实际仰卧相对等权续训净增 7 次。两组续训的实际右侧成绩均低于父模型。随后的动作保持对照在第 32 次更新触发累计偏移停止线，候选训练未启动。保存检查点复算显示，第 25 次更新至停止时约 69% 的新增平方偏差来自右膝和右髋俯仰。

当前起身对照改用覆盖四种实际落地姿态的同一回放池。两批开发轨迹提供 150 个成功场景和 900 条输入，父模型复算动作的最大误差为 1.91×10⁻⁶ rad。修正价值归一化读回和探针上下文后，两组各 3,072 环境的 CUDA 初态通过，物理、策略、价值、优化器与回放状态逐项一致，每组峰值 11,922 MiB。两次初始化停止和通过的检查均没有产生物理步或更新。独立评价沿用原逐姿态门槛，右侧候选须比对照净多通过至少 4 个场景，其余三姿态对两路参照均不得退步。 [初态核验与复现命令](../../records/reports/getup-right-upright-ppo-zero/README.md)

正式对照的两组各完成 17 次更新、1,253,376 条转移。候选仅增加右侧受阻后的直立进展奖励。对照和候选全程分别用时 108.83 秒和 104.08 秒，显存峰值分别为 12,272 和 12,265 MiB。两组检查点、导出策略、回放抽样和完整 8 秒原片均已独立复核，Windows 端各记录了 16 个实际运行中曲线状态。 [运行观察](../../records/getup-right-upright-ppo-live.html) · [两组原片和耗时](../../records/reports/getup-right-upright-ppo/README.md) · [独立核验](../../records/reports/getup-right-upright-ppo/training-verification.json)

新场景同次释放后，父模型通过 311/320、对照 316/320、右侧奖励候选 308/320。候选比对照少通过 1 个实际右侧和 6 个实际仰卧场景，关闭该方案。三组指定仰卧分别为 29/32、30/32、28/32，仍低于 31/32。对照在实际右侧已达 49/50，本批原定净增 4 次超过剩余可挽回场景数。后续提前检查增益门槛的可达性，并从两例共同仰卧失败定位支撑环节。 [完整评分和 12 段原片](../../records/reports/getup-right-upright-ppo/evaluation/README.md)

两例共同失败在释放后约 0.58 秒达到 30°以内。对照组成功样本到达 60°的中位时间为 0.44 秒，两例为 0.46 秒。最后 3 秒，两例均保持双脚触地并持续头腿接触，最小机身倾角仍为 33.1°、33.9°。下一步跨开发批次检查转正后的支撑与倾角回升时序。 [保存轨迹复算](../../records/reports/getup-right-upright-ppo/evaluation/requested-supine-review.json) · [失败动作关键帧](../../records/reports/getup-right-upright-ppo/evaluation/requested-supine-keyframes.jpg)

三批固定父模型的 523 个实际仰卧场景中，18 个失败均未能通过回倾或脚离地规则获得 60 毫秒提前量。脚离地还会触发 218/505 个原成功场景。将检查提前到第 2.44 秒后，最近 40 毫秒的倾角下降幅度成为入选信号。规则只在 1084、1087 选择，覆盖 4/5、6/7 个失败；在 1102 原样应用后覆盖 5/6，触发 6/171 个原成功场景。实际观测读回显示，1102 的原奖励对照用真实姿态只覆盖 1/3 个失败，用策略收到的重力观测覆盖 2/3，误触发 19/174。两路均未过筛查门槛，关闭本规则。 [跨批时序](../../records/reports/getup-supine-support-timing/README.md) · [提前量筛查](../../records/reports/getup-supine-early-geometry/README.md) · [实际观测读回](../../records/reports/getup-supine-early-signal-readback/README.md)

延长到决策前 120 毫秒的七帧真实 IMU 输入，并加入陀螺仪俯仰速度特征。两批整种子交叉验证只提前覆盖 3/5、0/7 个失败，误触发 5/175、2/159 个成功场景。覆盖低于 75% 门槛，关闭本方案。下一步用保存的接触与动作轨迹定位可干预的支撑环节。 [离线结果与复现](../../records/reports/getup-supine-causal-imu-screen/README.md)

从三批共 523 个实际仰卧场景分别读回原请求、下发目标和实测角度。第 2.60–3.00 秒，失败组左髋俯仰的跟踪差距中位数为 9.30°–17.80°，成功组为 3.59°–4.19°。四个髋膝关节的下发目标均未因投影改变超过 0.001 rad。失败组的头腿接触占 73.0%–76.2% 的有效帧，成功组为 1.9%–2.9%。后续采集实际执行器力矩和接触受力，以区分执行器跟踪与接触约束。 [逐批回读](../../records/reports/getup-supine-support-target-readback/analysis.json) · [采集草案](../../configs/experiments/getup_supine_left_hip_load_capture_draft.json)

固定策略在 3,072 个环境完成 1,689,600 条转移，记录 2.30–3.10 秒的 200 个物理子步，独立复算 3,645,053 条接触记录。双脚承重且头腿接触时，失败组左髋驱动力矩为 0.236 N·m，高于成功组的 0.162 N·m，但朝目标的关节速度由成功组的 0.201 降至 0.062 rad/s。两组的反向约束合量分别为 0.153、0.256 N·m。严格的同一场景接触前后对齐只覆盖成功 6 例、失败 2 例，低于每组 18 例门槛。此批固定策略的实际仰卧为 2,590/2,766，起身未达到原 95% 门槛。 [受力结果和完整原片](../../records/reports/getup-supine-left-hip-load-capture/README.md) · [严格时序复算](../../records/reports/getup-supine-left-hip-load-capture/contact-timing/analysis.json)

首次双脚承重时仍有持续头腿承力，失败组为 39/171，成功组为 24/2,584。按真实 20 毫秒边界复算，左脚单支撑时的候选触发覆盖 91 个失败和 123 个成功场景。双向目标在 1,070 条保存输入上通过原限位，并在实际 CUDA 零步核对了电机写入和原动作历史。自然 BAM 供电在同一场景变化约 1.6 mV，后续物理对照将保留此耦合。运行控制器重放了 214 次同样的触发，每个方向 856 个目标变化。下一步核对哪一方向能在同初态下改善转正与完整持稳。 [承重顺序](../../records/reports/getup-supine-left-hip-load-capture/load-order/analysis.json) · [脉冲预检](../../records/reports/getup-supine-left-hip-load-capture/pulse-component/README.md)

<video controls preload="none" src="../../records/evaluations/duck_getup_left-hip-load_reference_s1154_20261110T151744/evaluation-fixed-weighted-supine/clip-step-0.mp4"></video>

左髋双向脉冲已关闭。两次原策略在实际仰卧中持稳 647/687、645/687，沿跟踪误差方向与反向为 644/687、646/687，均低于 95%。两个方向在触发后 100 毫秒造成约 1.39° 的左髋实际位移差，持稳未达到晋级门槛。三路保存的原策略中，触发后 160 毫秒的右脚冲量占比在成功组中位数均为 0%，在失败组为 19.0%–20.8%。时序复算只有 2 个失败同时出现右脚持续承重和躯干停滞，两例均在停滞后才确认承重。右脚承重未获得干预所需的提前量。 [配对评分与四段原片](../../records/reports/getup-supine-left-hip-pulse-trial/README.md) · [逐场景时序](../../records/reports/getup-supine-left-hip-pulse-trial/floor-transfer-order.json)

此轮用 768 个环境保存完整场景状态，四路合计 1,424,640 条转移，约 6,238 条/秒。显存峰值 4,620 MiB，GPU 利用率采样均值 31%。正式 PPO 的并行规模由先前两轮吞吐短测选为 3,072。观察页保留实时刷新记录，录像均为 11 秒，拍摄的环境 0 四路均未触发脉冲。 [运行观察](../../records/getup-supine-left-hip-pulse-live.html) · [四路关键帧](stage-video-gallery.html)

同回合二次起身的触发筛查在四个开发种子中识别 24/27 个仰卧失败。675 个原成功仰卧场景与其他姿态均未触发。原颈部请求越界使完整 HOME 返回在 0.8 秒预算内为 0/24。将颈部目标收回到原下限内侧 0.15 rad，并让其余关节趋近原起始姿态，24/24 条目标轨迹在相同限速下到达。实际 CUDA 零步核验复现了原延迟、目标写入和状态恢复。回撤末步的颈部电机力矩最大变化约 1.086 N·m。实际退离、安静仰卧与二次持稳分别评分。 [触发与目标预检](../../records/reports/getup-supine-left-hip-pulse-trial/README.md) · [原生控制链](../../records/reports/getup-second-attempt-native/README.md)

三路同初态仰卧持稳分别为 578/617、578/617、577/617。短程回撤在 38 个共同失败场景均未完成二次起身，候选关闭。目标开始明显向内拉动的中位时点为回撤后 0.62 秒。回撤结束后的 0.4 秒内，38 个场景的颈部位移曾达到 0.08 rad，只有 1 例同时出现持续头腿分离，直到原 8 秒截止仍无安静仰卧。 [配对结果、复算和三段原片](../../records/reports/getup-second-attempt-trial/README.md)

![同初态起身关键帧](../../records/reports/getup-second-attempt-trial/matched-keyframes.jpg)

<video controls preload="none" src="../../records/reports/getup-second-attempt-trial/replay-neck-retreat.mp4"></video>

从五批原轨迹选出 45 对失败初态与匹配成功初态。在同一个新种子中，困难组、匹配组和原始跌落组各有 1,024 个实际仰卧场景，完整持稳分别为 959、945、943 次。失败初态在新环境中未提高难度，停止用这批高度与朝向替换复位。三组在 GPU 上共采样 1,689,600 条转移，用时 116.92 秒。 [初态库](../../records/reports/getup-hard-start-bank/README.md) · [评分、资源和原片](../../records/reports/getup-hard-start-screen/README.md)

两份归档自训起身权重已核对 ONNX、恢复保存点与 61 项输入到 14 项输出的接口。首批两路参照为 971/1,024、974/1,024，原奖励续训为 993/1,024，侧卧到仰卧回放为 994/1,024，均通过四姿态门槛。独立复验的两份候选为 980/1,024、983/1,024。实际仰卧仍分别净增 9、12 例，指定仰卧分别少 7、5 例才过线，两份候选均未晋级。两批重复参照分别有 33、35 个场景的成败不同。 [两批逐姿态评分、过程和八段原片](../../records/reports/getup-archived-policy-comparison/README.md)

![同场景仰卧起身](../../records/reports/getup-archived-policy-comparison/s1160-qualified/paired-recovery-keyframes.jpg)

第二批 26 个指定仰卧场景仅一份候选通过，26 个两份均失败。按已知结果选择策略的离线上界为 646/672。两批共同失败的 37 个场景在末端均双脚触地并有头腿接触。用释放时的实际观测分流，整种子交叉测试分别少通过 4、3 例，关闭。 [离线复盘](../../records/reports/getup-archived-policy-comparison/residual-review.json) · [分流筛查](../../records/reports/getup-archived-policy-comparison/release-routing.json)

释放后前四步的单关节差异未通过时序准入，平滑混合也未达到动作幅度和限位门槛。随后取两批 30 个仅教师成功的实际仰卧场景，使用 2.20–3.10 秒有界腿部标签，另用 1,024 条四姿态成功回放保留父策略动作。两路等数据末层拟合各做一次求解。原残差和固定 70% 收缩均有新增越界量超线，解析收缩至 62.6% 后两批均过离线限位，第二批教师动作误差仍下降 66.8%。训练改动集中在末层腿部输出。实际起身使用新种子复验。 [训练方案、过程和导出核验](../../records/experiments/duck_getup_teacher_head_offline_s1160_20261110T175320/README.md)

新种子三路完整持稳为 987/1,024、988/1,024 和 989/1,024。候选指定仰卧 641/672 已过线，实际仰卧 788/820 达到配对增益，但实际右侧 46/49 低于 47/49。后续只在仰卧释放时启用末层候选。三批保存观测的仰卧识别覆盖 2,450/2,450，其他姿态 0/622 误选，实际效果待新场景验证。 [三路原片与评分](../../records/reports/getup-teacher-head-evaluation/s1162/README.md) · [仰卧限定复算](../../records/reports/getup-teacher-head-evaluation/supine-latch-preflight.json)

新场景的 813 个实际仰卧均正确选择末层候选，其他 211 个姿态无误选。三路实际仰卧为 772/813、772/813、775/813，指定仰卧为 633/672、634/672、636/672，候选未晋级。三路固定策略合计 1,484,800 条转移，含写盘约 7,258 条/秒。运行中画面和三段 11 秒原片均已核验。 [三路评分与原片](../../records/reports/getup-teacher-head-evaluation/s1163-latch/README.md) · [运行观察](../../records/getup-teacher-head-latch-s1163-live.html)

两批保存轨迹共有 28 个指定且实际仰卧场景三路都失败。其中 27 个在末两秒持续双脚着地、头腿接触且倾角超线。另从两路父策略均成功而候选失败、以及仅候选成功的场景各选 16 个，整理为 900 条同输入回放。只替换 450 条的腿部标签。原生 CUDA 的两批候选加权梯度范数为 0.3550、0.3429，对照小于 4.1×10⁻⁶，只有末层十个腿部输出收到梯度。 [接触与时序复算](../../records/reports/getup-teacher-head-evaluation/paired-failure-review.json) · [梯度复核](../../records/reports/getup-teacher-head-evaluation/corrective-replay/native-loss-preflight.json)

沿用归档训练的 76 项价值输入和 0.10 rad 探索标准差。CPU 与 CUDA 各对临时演员执行两步 Adam，只有末层腿部行及对应动量变化。2,048 条保存的价值输入在两端的最大输出差为 4.77×10⁻⁷。从四姿态各 32 个成功场景提取 1,024 条独立保持观测。两组 3,072 环境的 CUDA 初态签名相同，峰值显存均为 11,986 MiB。首次正式启动在签名读回处零步停止。独立 CUDA 证明统一整数键的 JSON 表示后全部字段相同，另以原预算执行修复后的训练。 [检查点核验](../../records/reports/getup-teacher-head-evaluation/corrective-replay/checkpoint-adam-preflight.json) · [初态、停止与修复记录](../../records/reports/getup-latched-head-corrective-ppo/README.md) · [训练观察](../../records/getup-latched-head-corrective-ppo-signature-live.html)

纠正回放对照的两组各完成 17 次更新和 1,253,376 条转移，端到端速度为 12,024、12,737 条转移/秒，显存峰值为 12,432、12,278 MiB。独立核验确认两组抽样相同，仅演员末层腿部参数更新。两组各有一段完整 8 秒原片，Windows 分别读取 16 次实际运行曲线。下一步在新的共同释放场景比较两组和两次固定起点，保持原 95% 逐姿态验收。 [训练过程、权重和原片](../../records/reports/getup-latched-head-corrective-ppo/README.md) · [独立训练核验](../../records/reports/getup-latched-head-corrective-ppo/signature-recovery/training-verification.json)

四路评估的首次零步核验通过后，额外测试发现回退模型导入会改写 CUDA 随机状态。保护随机状态的导入方式保持权重和 CPU、CUDA 随机状态逐位一致，已单独冻结。旧协议保持零物理步。 [随机状态实测](../../records/reports/getup-latched-head-corrective-ppo/evaluation/import-rng-proof.json) · [四路评估方案](../../configs/experiments/getup_latched_head_corrective_evaluation_s1164_rng.json)

新场景四路评估共执行 1,945,600 条转移。两路固定参照实际仰卧分别通过 775/812 和 776/812，训练对照为 793/812，纠正候选为 779/812。候选相对训练对照多通过 12 例、少通过 26 例。两组实际右侧为 49/53 和 50/53，低于 51/53。关闭纠正标签方案，保留训练对照作为仰卧研究参照，三个开发种子的右侧回退轨迹将单独复盘。 [四路评分、运行记录和原片](../../records/reports/getup-latched-head-corrective-ppo/evaluation/rng-guarded/README.md) · [右侧复盘范围](../../configs/experiments/getup_right_fallback_cross_seed_review.json)

三个开发种子中，156 个实际右侧场景按同权分支分为 144 个共同成功、7 个分歧和 5 个共同失败。共同失败的 17 条分支均持续双脚着地、头腿接触且末段倾斜，释放后从未达到 15°。2.38 秒前出现接触的共同成功有 135/144。下一项固定策略对照保留仰卧模型，只在首帧识别为右侧时改用已有独立开发证据的右侧专用模型。 [逐场景结果](../../records/reports/getup-right-fallback-cross-seed/README.md) · [新方案](../../configs/experiments/getup_dual_pose_route_s1166.json)

双姿态分流在新开发种子的实际右侧由两路参照的 54/57 提升到 57/57，挽回同三个场景。候选实际仰卧 781/806，指定仰卧 649/672，总体 999/1,024，达到原逐姿态及总体门槛。三路在本机 CUDA 共采样 1,484,800 条转移，完整右侧原片各 11 秒。下一批只更换开发种子。 [评分、运行记录和原片](../../records/reports/getup-dual-pose-route/README.md)

第二个开发种子沿用相同权重及分流，总体为 1,000/1,024，实际右侧 58/59，较两路参照净增 3、2 例。实际仰卧为 783/806，全部原门槛再次通过。 [独立复验与原片](../../records/reports/getup-dual-pose-route/replication/README.md)

最终留出使用一次，候选总体 993/1,024，实际仰卧 765/792。实际右侧 69/73，低于 70/73，候选未晋级。两次开发和留出合计 4,454,400 条评估转移，零学习更新。留出封存。 [最终评分、运行记录和原片](../../records/reports/getup-dual-pose-route/holdout/README.md)

本机起身短测选用 3,072 环境。含写盘吞吐 11,152 条转移/秒，较 2,048 环境高 21.4%，显存峰值 13,253 MiB，占整卡 80.9%。短测的六段 3.2 秒录像单列。 [资源、进度与原片](../../records/reports/getup-right-robustness/scale/README.md)

扩大实际右侧覆盖后，候选在新开发场景中总体达到 3,010/3,072，实际右侧为 936/960，两路参照为 910/960 和 903/960。候选通过原逐姿态与总体门槛，且相对两路参照净增 26、33 例。三路共 4,454,400 条转移，含录像和写盘约 12,357 条转移/秒。候选 24 例实际右侧失败的末段均持续双脚着地、头腿接触且倾角超限。 [开发评分、失败复盘与录像](../../records/reports/getup-right-robustness/development/README.md)

按原四组请求比例运行一次新的独立留出。候选总体 2,983/3,072，实际右侧 179/183，相对两路参照各净增 4、5 例。候选总体略低于两路参照，原逐姿态及总体门槛全部通过。三套训练 ONNX 及对应保存点已归档。下一项在同一物理回合内验证起身接站立再接前进，开发观测上的首步目标变化已通过 0.040 rad 限速检查。 [独立留出与录像](../../records/reports/getup-right-robustness/holdout/README.md) · [起身交付清单](../../artifacts/delivery/getup-simulation.json) · [站立边界核验](../../records/reports/getup-stand-handoff/routed-development-boundary/README.md)

首次连续回合从跌倒开始，依次运行起身、站立和前进 32 秒。两个切换点的实际应用目标变化均不超过 0.040 rad。完整序列通过 2,717/3,072 例，低于 95%。198 例出现持续严重姿态异常，开始时间中位数为进入前进后 0.66 秒。起步对照从同一站立终态比较原指令与半秒渐升指令。 [独立评分和完整录像](../../records/reports/continuous-routed-getup-stand-walk/README.md) · [起步对照与运行观察](../../records/reports/continuous-routed-forward-entry-ramp/README.md)

半秒渐升指令通过起步两秒门槛。两组各有 3,009 个场景进入前进，持续严重姿态异常从 185 降到 0，平均绝对偏航从 0.409 降到 0.129 rad/s，第二秒平均前向速度为 0.151、0.162 m/s。两组只改变前半秒输入，模型权重相同。 [对照评分、两段原片与关键帧](../../records/reports/continuous-routed-forward-entry-ramp/README.md)

完整 32 秒的开发与独立留出分别为 2,976/3,072（96.88%）和 2,990/3,072（97.33%），均通过原 95% 门槛。留出有 2,992 例通过起身和站立接入，2,990 例通过完整前进。留出在第一个物理步前完成一次性签收，评分、原片、策略和对应保存点可从交付清单核对。 [开发原片](../../records/reports/continuous-routed-getup-ramped-forward/README.md) · [独立留出](../../records/reports/continuous-routed-getup-ramped-forward/holdout/README.md) · [仿真交付](../../artifacts/delivery/ramped-getup-standing-forward-simulation.json) · [滚翻扰动](../../records/reports/continuous-roll-early-lateral-push/README.md)

头部两组既有训练没有重跑。两组配置只在已声明的颈部指令范围上不同，真实物理初态和随机状态一致。每组完成 1000 帧 ONNX 驱动仿真，均无超容差帧，最大动作误差分别为 6.68×10⁻⁶、5.72×10⁻⁶。

| 阶段 | 当前结论 | 复查入口 |
| --- | --- | --- |
| 环境与来源 | 电脑环境已部署；依赖、镜像下载、源码与模型哈希有记录 | 第 2–3 节、[部署日志](../../records/deployment/2026-10-31/README.md) |
| S1 训练和评估流程 | 导出、真实观察、停止、保存及恢复已核验；流程正确不等于步态合格 | 第 5、8 节 |
| S2 站立与六方向 | 两批开发 223/224、224/224，最终留出七项各 32/32。留出前进偏航较两路参照下降 0.00603、0.00524 rad/s | 第 7.15 节及[最终留出](../../records/reports/common-walk-right-support-yaw-training/yaw-direction-holdout/analysis.json) |
| S3 跌倒起身 | 新独立留出通过原逐姿态及总体门槛，候选总体 2,983/3,072，实际右侧 179/183 | [独立评分与录像](../../records/reports/getup-right-robustness/holdout/README.md) · [交付清单](../../artifacts/delivery/getup-simulation.json) |
| S4 前滚翻 | 对齐接触读数后，开发 1026、独立留出 2026 各 117/128，单动作通过 | 第 7.7.13 节 |
| S5 连续动作 | 滚翻接前进开发复验通过 124/128。首次起身、站立与前进的完整序列为 2,717/3,072，未达 95%，主要失分在前进偏航 | [32 秒连续评估](../../records/reports/continuous-routed-getup-stand-walk/README.md) · [前进抗扰](../../records/reports/continuous-forward-lateral-push/README.md) |
| S6 部署准备 | 映射、标定、IMU、时序及回退说明已整理；实物操作后置 | 第 9 节、[标定手册](../runbooks/hardware_commissioning.md) |
| 真实媒体 | 总览收录 759 段录像和 1,518 张标注帧，含七指令留出、起步失稳和滚翻扰动对照 | [单张总览](assets/all-video-overview.jpg) · [逐段图册](stage-video-gallery.html) · [留出关键帧](../../records/reports/stage-media/cards/N628.jpg) · [转向两段原片](../../records/reports/continuous-routed-turn-feedback-pair/README.md) |

历史短训、失败候选和核验修复保留在第 6–7 节及[完整过程记录](development_history.md)；[原阶段明细](../../records/reports/common-plant-screen/report-stage-ledger-before-simplify.md)保留整理前的逐项索引。

滚翻阶段抗扰从同一状态分支。无扰动和右向组的完整序列分别通过 124/128、116/128。左向组仅 115/128 个场景完成滚翻，按原规则停在第 500 步，前进未评完。与参照配对后，左、右分别新增 13、10 例失败，均完成至少 330° 的几何转动，两侧离地转角的配对增量中位数分别约 30.5°、29.8°。原始采样在写索引时退出，轨迹、录像和运行状态均已保留，索引和评分已离线复算。 [独立评分与三段原片](../../records/reports/continuous-roll-early-lateral-push/README.md)

首次起身接七指令的 32 秒开发评估完成 4,915,200 条实际转移。总体通过 2,932/3,072（95.44%），站立、前后走与左右平移分别达标。左转 396/439、右转 403/439，低于各自的 418 例门槛。已接入的 53 个转向失败均有偏航超限，其中 50 例在起步首秒之后仍超限。复查发现连续入口漏接了原行走验收的陀螺仪转向反馈，随后恢复原输入映射。 [七指令评分、原片和复算](../../records/reports/continuous-routed-seven-command/README.md) · [反馈接入核验](../../records/reports/continuous-routed-seven-command/turn-feedback-cpu.json)

同站立状态对照共用前 600 步和 345 项状态。恢复陀螺仪转向反馈后，完整通过由 2,923 增到 2,977/3,072，七项达到原门槛。左转由 391 增到 426/439，右转由 404 增到 423/439，平均偏航误差分别下降 0.0453、0.0500 rad/s。新方案进入单次独立留出。两组共 7,987,200 条转移，显存峰值 13,157 MiB。保留原评分所需逐步字段并使用压缩等级 1 后，跨分块写入窗口的吞吐中位数为 12,198 条/秒，整轮为 16,587 条/秒。首次入口在零物理步时因配置键名退出，兼容后完整采样。 [评分和原片](../../records/reports/continuous-routed-turn-feedback-pair/README.md) · [留出草案](../../configs/experiments/continuous_routed_turn_feedback_holdout_draft.json)

七指令独立留出总体通过 2,992/3,072（97.40%）。左转为 422/439，右转为 430/439，均高于 418 例门槛。其余五项同样达标。首次启动在 CUDA 初态哈希检查停止，物理步为零。复核确认两份原生派生量只有一个零值的符号位不同，按原分组完成独立采样。3,072 个并行环境完成 4,915,200 条转移，耗时 290.9 秒，显存峰值 13,122 MiB。保留五个自训模型及对应保存点、一个官方参照与控制实现的可核验索引。 [逐项评分与 32 秒原片](../../records/reports/continuous-routed-turn-feedback-holdout-zero-recovery/README.md) · [仿真交付](../../artifacts/delivery/routed-seven-command-simulation.json)

滚翻失支撑的复盘把检查窗口提前到策略重力观测首次跨过 120°。头部横滚目标下调 0.02 rad，在 23 个历史扰动样本上同时有几何和模型驱动余量。CUDA 零步检查复算 28,160 组保存观测，核对 256 个头部目标改变与 299 项状态还原。暂停时三组脉冲仅有共同起始段，实际抗扰效果待验证。 [滚翻复盘](../../records/reports/continuous-roll-early-lateral-push/README.md) · [冻结方案](../../configs/experiments/continuous_roll_head_support_pulse_wall_budget.json) · [暂停记录](../../records/reports/continuous-roll-head-support-pulse/wall-budget/README.md)

整轮显存峰值 12,906 MiB，常规采样窗口中位数 19,684 条转移/秒，跨压缩写入的窗口降到 5,480 条/秒。同一轨迹分块使用低压缩等级后，平均写入耗时从 17.22 秒降到 14.69 秒，大小增加 0.53%，21 个数组逐字节往返一致。 [采样性能](../../records/reports/continuous-routed-seven-command/sampling-performance.json) · [压缩基准](../../records/reports/continuous-routed-seven-command/trace-compression-benchmark.json)

32 秒原片来自预先指定的 0 号场景，该场景分到左转组，实际未能从起身接入指令。下图用保存的物理状态展示每条指令已接入的最小场景编号，另行标记其结果。

![七指令关键帧](../../records/reports/continuous-routed-seven-command/command-keyframes.jpg)

正式训练须先验收运行中可查看的进度、曲线与真实短片。VS Code 任务终端可选。每次运行的改动、过程和结果以同一编号串联。 [右侧起身观察页](../../records/getup-right-upright-ppo-live.html) · [训练对照与操作记录](../../records/reports/getup-right-upright-ppo/README.md) · [全阶段路线与观察验收](../development_roadmap.md)



## 2026-11-16 训练手册历史导言

以下是整理前的手册导言，保留当时的状态与操作条件。当前结果以主报告和目标页为准。

# 电脑环境与训练

适用范围：Windows 11 + NVIDIA GPU，经 WSL2 Ubuntu 24.04 运行本项目归档的官方训练代码。首次编写于 2026-10-31。本机已完成两组短训练、回放、导出与离线检查，结果见 [验证报告](../reports/pc_pipeline_validation.md)，过程见 [部署记录](../../records/deployment/2026-10-31/README.md)。

环境选型说明：Ubuntu 24.04 与 WSL2 是本机实施选择，不能表述为官方指定或官方已验证的训练环境。[XGO-Duck 固定提交的安装说明](https://github.com/LuwuDynamics/xgoduck_rl/blob/fa621fc77ad9a05d69da13270e86118bdd3603fe/README.md#install)只明确要求 CUDA GPU、Python 3.12 和 uv；[mjlab 1.3.0 安装要求](https://github.com/mujocolab/mjlab/blob/v1.3.0/docs/source/installation.rst)要求训练使用 Linux 与 NVIDIA GPU，未固定 Ubuntu 发行版。归档中 `vendor/bam/.readthedocs.yaml` 的 Ubuntu 24.04 是依赖库文档构建配置，不能证明 XGO-Duck 的官方训练主机使用该系统。本机兼容性以 CUDA、实际训练、回放和导出检查为准。

## 历史训练记录与观察

用户已恢复目标，目前尚未启动新的训练和仿真；以下段落保留早期阶段的运行与诊断记录，最新状态和运行条件以[当前目标](../current_goal.md)为准。WSL 历史事件日志及曲线查看方式见[WSL 训练日志归档](../../records/reports/progress-archive-20261114/wsl-training-logs/README.md)。

使用保存点前，先核对其所属训练配置：目前行走组件的价值网络输入为 77 维，起身组件分别为 75 或 76 维。五个主要组件已做 CPU 网络、优化器和 ONNX 输出读回，但没有验证原物理现场续跑。保存点的 `iter` 字段只是该训练器的编号，不能一律按完成次数解释；[保存点检查](../../records/reports/native-ppo-observation-preflight/README.md#保存点边界)列出具体结果与边界。

最近完成官方滚翻独立确认：开发117/128、116/128，留出112/128，未通过，本轮关闭不追加；首次零步随机状态错误和单独修复保留原记录。完整滚翻原片9秒，起身训练8秒、评价11秒，事件截图保留真实时刻。[滚翻索引](../../records/reports/roll-common-confirmation/rng-recovery/README.md)提供只读复核命令。当前准备S3训练时的成功动作保持约束，2360组固定回放与解析梯度检查已完成；完整PPO集成、价值起点、来源冻结和实际GPU零步仍须先通过，尚未开始新学习。

本机环境已部署，不需要重新安装。S1 来源、导出、评估及观察/停止/恢复流程已验收。固定官方行走模型加入指令映射和转向反馈，在两个开发种子及一次留出完成七指令确认，共 672/672；该结果没有更新模型权重，原瞬时平面误差失败仍保留。官方起身完成四姿态、侧卧覆盖和随机跌落开发筛查。

既有滚翻学习候选均已关闭。共同场景固定回归完成：行走 223/224，滚翻 30/32，起身四姿态未通过。仰卧碰撞规则对照为完整碰撞 30/32、原掩码 32/32；后者只用于诊断，不是最终配置。完整碰撞下的起身适配已通过 CPU 和实际 GPU 零步检查；固定策略价值预检完成，独立解释方差 0.828、均方误差 0.024。价值预检的actor未更新；后续50次更新对照已结束，官方/对照评价均俯卧32/32、仰卧31/32。接触候选因继续门槛超过32次上限停止，部分点保留，没有最终评价。独立姿态覆盖已结束：官方299/320、对照298/320，640个首个episode、零学习、四类覆盖充分但功能未通过；判据和完整11秒录像见主报告第7.7.5节。已关闭任务归入历史，不重跑；日常先选择“状态与当前阶段”。可用“XGO-Duck · 重放已保存的起身失败（不训练）”查看保存状态的对照图。当前进度与结果以[目标清单](../current_goal.md)、[主报告](../reports/development_report.md)和工程记录为准。

“起身末段颈部解困对照”已完成：对照291/320、候选288/320，零学习、完整11秒录像。头腿接触清除但仍前倾，候选关闭，不追加。完整保存状态回放见主报告第7.7.6节。站立接力已完成CPU与真实GPU零步检查，并执行完整动态对照，结果见下段；新方案不得直接复用已关闭协议。已完成的冻结任务有防重复检查，不能靠重新运行任务继续训练。恢复动态评估后，若想自己观察固定行走模型，可选择“XGO-Duck · 新建七指令练习（不训练）”，输入新的练习名称和开发种子；它不会覆盖已有记录，也不使用最终留出，但会运行新的仿真。

新的“起身接力站立对照”已在原共享终端完成：共同噪声/目标过渡，两组各320次完整11秒、零学习，真实GPU零步通过后才积分。对照292/320、候选296/320，均未达到功能门槛，已关闭、不追加。不重启既有颈部解困协议；接力成绩单独记录，不能套用旧较小噪声的结果。

最新颈部归位顺序评估已完成三组各320次完整11秒、零学习。共同接口291/320、只归位288/320、归位后站立287/320，均未晋级，任务已归档。12段新原片可在主报告第7.7.9节和逐段图册N199–N210正常速度播放；不要重跑已关闭任务。时长和末帧保存的CPU检查已通过。

共同接口的新价值网络已完成50次拟合并通过独立复核，价值拟合阶段的行为网络保持官方权重；读取接口的一步失败和单独修复均保留。随后末段全身可达目标已完成有界训练及独立评价，结果未晋级，当前关闭。完整8秒原片、拟合曲线、训练结果和复核命令见主报告第7.7.10–7.7.11节。

后续仍需完成共同环境起身适配、可靠滚翻确认、连续技能切换、扰动验证和最终部署准备。真机运行暂不执行。下面的环境步骤用于换机复现；旧训练编号与逐次故障移至[完整过程记录](../reports/development_history.md)，最新操作入口见[主报告第 4 节](../reports/development_report.md)。

观察共同接口的历史试验时，可在 Windows 打开保存的本地运行观察 HTML。页内显示实际控制步、吞吐和已录完的仿真原片。用 VS Code 打开同一运行目录的 `protocol.json`、`state.json` 和 `execution.log`，即可把画面对回命令和采样记录。已完成的冻结批次保留原输出；新建七指令练习会启动动态仿真，暂停期间不执行。 [前进闭环观察页](../../records/walk-actuator-feedback-heldout-live.html) · [同批评分与原片](../../records/reports/common-walk-actuator-yaw-feedback-heldout/README.md)

起身与滚翻按完整动作观察：新入口使用 `ActionVideo` 读取实际 episode 时长，不再统一截取 3 秒。起身训练为 8 秒、固定评估为 11 秒；站立/连续行走短片为 4 秒。约每 180 秒等待下一次动作起点拍摄，自动重置前保存终态。实际规则见[动作录像配置](../../configs/environments/action_recording.json)。完整保存状态回放与原始录像单列，不作为新评估。

## 恢复前先检查什么

当前尚未启动新训练或动态评估。下面的历史短训脚本与裸 `uv run train` 命令会启动训练，但没有接入本项目现在准备的观察桥、外层监督器和暂停标记检查；它们用于理解早期复现过程，不能作为当前阶段的受监督入口。七指令练习虽然不更新权重，仍会新建动态仿真，应先核对当前阶段。[入口核对](../../records/reports/native-ppo-observation-preflight/entry-audit.json)

在 Windows 的项目根目录，可先只读查看当前标记：

```powershell
$projectState = Get-Content -LiteralPath 'project.json' -Raw | ConvertFrom-Json
$projectState.staged_training_request | Select-Object current_stage, pause_requested, learning_running, simulation_running, next_gate
```

用户恢复目标后，依次准备新运行：

1. 为每组使用新的运行编号，记录问题、唯一修改、相同父保存点、种子、样本预算和原动作门槛；旧的 `local_getup_ppo.json` 与 `next_training.json` 已有运行或已关闭，不能作为新运行直接重启。
2. 固定训练源码、配置、依赖、模型和观察器的哈希；按本次方案分别准备 CPU 检查及与新运行对应的 CUDA 零步证据，并核对初态、随机数和执行器状态。
3. 在新运行启动前确认用户能打开其真实状态、曲线和完整动作录像，并接入监督器、观察进程和任务专用异常停止条件。目前这些部件已做隔离检查，尚未组成经过真实训练验收的完整入口。
4. 在固定预算内采样，按同父点和等样本比较；异常或门槛不可达时停止。保留命令、逐轮指标、资源、完整原片、退出原因、源码与模型哈希，再用独立场景评估效果。
5. 中断后区分完整更新保存点和部分优化点。先检查策略、价值网络、优化器与随机数，再决定恢复方式；未保存完整物理现场时不能从原轨迹精确续跑。

本机浏览器训练页不可访问时，可在 VS Code 中直接查看[历史训练曲线与录像](../../records/reports/historical-getup-training-comparison/README.md)和[最新评估原片](../../records/reports/duck_roll-native_nominal_s1241_20261115T230228/evaluation-view.mp4)。在已打开本项目的 VS Code 按 **Ctrl+P**，输入 `records/reports/historical-getup-training-comparison/README.md` 并回车，再按 **Ctrl+Shift+V** 预览；播放录像可用同样方法打开 `records/reports/duck_roll-native_nominal_s1241_20261115T230228/evaluation-view.mp4`。这些是已保存的记录。

新的统一起身入口已完成合成进程接线检查：新运行的查看确认在工作器导入前检查，运行中由外层监督器管理停止，结束后另存最终曲线。具体文件、失败修复记录和隔离复核方式见[训练入口接线](../../records/reports/native-ppo-observation-preflight/observed-entry-check/README.md)。目前仍需具体的新训练协议、对应 CUDA 零步和实际查看验收，不能直接把旧方案路径传给新入口启动训练。



## 2026-11-16 起身训练观察与吞吐结案

本机网页无法访问，实际训练在原 VS Code 窗口核验状态、曲线和录像。首次运行因 Python 元组与 JSON 列表直接比较而停止，采样 73,728 条转移、策略更新 0、价值网络更新 120 次；部分保存点 CPU 读回相符。另建序列化适配入口并重新冻结来源后完成等样本两组训练。

1,536 环境组 34 次更新、229.33 秒；3,072 环境组 17 次更新、209.13 秒，每组 1,253,376 条转移。后者总过程吞吐提高 9.7%，显存峰值增加 5.16 GiB；GPU 利用率中位数分别为 31% 和 33%。完整 8 秒原片、保存点、ONNX 和实际指标均通过独立核验。没有用本轮训练数据替代动作独立评估，也没有更新交付模型。

本阶段加上首次失败累计 2,580,480 条训练转移。两组工作器已退出，转入保存数据的性能分析和归档，不追加本轮预算。完整命令、源码、错误与复算见[阶段报告](../../records/reports/getup-observed-throughput-20261116/README.md)，本地查看入口见[训练记录](../../records/training-current.md)。


## 2026-11-16 训练计时范围纠正

对保存指标和源码的只读复核发现，上一阶段 209.13、229.33 秒包含环境初始化、导出与收尾，先前称为纯循环耗时不准确。原始记录、归档和 9.7% 数值保留，但该数值改称总过程吞吐增益。按采样开始至末次指标发布计算，两组为 199.05、221.50 秒，窗口吞吐增益为 11.3%。窗口仍含观察、检查与录像开销。完整依据见[计时复核](../../records/reports/getup-observed-throughput-20261116/performance-review/README.md)。

只读测量显示固定路径重复解析占用时间，已另建同父点、同种子、同规模的两组对照；只有路径绑定方式不同，每次暂停值、协议内容和源码哈希均重新检查。CPU 检查通过后进入实际 CUDA 零步，旧协议与旧权重不修改。

## 2026-11-16 固定路径对照与暂停归档

同父点、种子 3265、3,072 环境的两组训练均完成 17 次更新，每组 1,253,376 条转移。原检查组总耗时 209.16 秒，固定路径组 153.18 秒，对应吞吐提高 36.5%；每组各 2,075 次检查回调，累计耗时为 80.20 和 32.50 秒。固定路径仅绑定一次，每次暂停标记、完整协议和源码哈希仍重新读取，118 份冻结来源未变。

两组保存初态一致，第一控制步的位置与速度却已有约 10⁻⁷ 的差异，最终模型也不同；原因未定。GPU 利用率中位数为 33% 和 33.5%，不能称算力已用满。新模型尚未进行独立动作评价，保留为诊断材料，不替换交付权重。完整对照、原始命令、退出结果、曲线与两段 8 秒录像见[阶段报告](../../records/reports/getup-canonical-path-20261116/README.md)。

按用户暂停和归档要求，目标状态改为暂停，旧协议关闭启动准入。进程检查未发现本项目工作器，训练锁空闲。后续仅复核保存文件、解码录像和打包；没有新增物理步、学习或硬件操作。首次归档视频检查尝试使用未安装的 OpenCV，随后改用环境已有的 `imageio_ffmpeg`，未增加依赖。两段原片均完整解码为 201 帧、25 fps。保存数据、118 份来源及两模型哈希通过核对。[暂停进度与恢复位置](../../records/reports/pause-archive-20261116/README.md)

## 2026-11-16 监控开销对照与参数预检

恢复目标后，对照组和监控候选组从相同父点和种子出发，各用 3,072 个环境完成 17 次更新、1,253,376 条转移。候选将重复的授权检查移至独立线程，同时要求检查结果保持新鲜；实际训练总耗时 172.19→133.13 秒，单次顺序对照吞吐提高 29.3%。两组间隔约三小时，尚需在相近负载下复测；训练内持稳并非独立动作成绩。保存点、导出模型、曲线及各 8 秒原片已复核，训练过程曾在原 VS Code 窗口显示。[训练方案与结果](../../records/reports/getup-monitored-authorization-20261116/README.md)

后续滚翻单因素准备中，首次标称零步在组合检查停止，另一次诊断在观察确认前停止；两次均无物理步。第三项独立诊断保存了实际执行器状态，在场景实体断言停止。源码显示地形被加入场景实体，原断言仅允许机器人；前一扭矩存在非零值。具体记录及限制见[参数预检](../../records/reports/roll-matched-factor-preflight-20261116/README.md)。低摩擦、高电压和动态参数对照尚未开始。

地形适配后的检查又在传感器类型断言停止，随后确认场景包含 13 个旧清单未允许的内置传感器。修正清单后，两环境 CPU 初始化检查和 128 环境标称 CUDA 零步均完成，310 项字段及源码、执行器、截图通过独立读回，物理步为零；低摩擦组随后单独核验。[零步核验](../../records/reports/roll-matched-factor-preflight-20261116/scene-nominal-zero-verification.json)

低摩擦零步完成后，初始位置、速度、观测、首动作、接触摘要、执行器和随机状态与标称组一致，左右脚摩擦变化符合方案。其余指纹差异中，支撑缓存键因对象地址不同而变化，忽略键后两组内容一致；其他 25 项 GPU 内部状态仍需原始数值分析，暂不进入高电压或动态试验。[差异复核](../../records/reports/roll-matched-factor-preflight-20261116/friction-cache-identity-review.json)

同条件标称重复出现相同的 25 项内部摘要变化。随后保存并核验标称与低摩擦各 34 项原始数组：128 个环境在采集时没有检测到接触；每环境 14 项活动约束按唯一编号对齐后，编号、雅可比、摩擦损失、状态及 D 项一致，约束力最大差 4.28×10⁻⁸。该结果只解释零步记录中的排序与数值差，动态摩擦效果仍待评估。[原始数值对照](../../records/reports/roll-matched-factor-preflight-20261116/friction-raw-zero-comparison.json)

## 2026-11-21 滚翻接站立与前进

两组正式滚翻训练各使用4,608环境、50次更新和5,529,600条转移，低摩擦加权候选用时290.02秒，净吞吐约19,066条/秒。VS Code保存了运行截图、逐轮指标、训练曲线和7秒原片。候选ONNX与对应含优化器的保存点通过哈希和固定输入核验。[训练及恢复材料](../../records/reports/roll-friction-paired-training/README.md)

在本轮连续对照中，脚底摩擦0.8、种子1247的两条路线共用低摩擦自训滚翻权重，候选在各场景开始前进4秒后调整官方行走模型的内部速度输入。独立复算滚翻均为120/128，完整路线通过数为0/128与120/128。收尾检查报告力矩差异从第二个物理子步开始，候选完整力矩数组未保存，原始控制与接触轨迹及两段31秒视频均保留。随后把其余组的力矩逐位核对限定在恢复后的首个物理子步，原动作评分继续沿用冻结标准。[首组复算](../../records/reports/roll-forward-input-integration/s1247-friction-low-low-friction/trajectory-readback.json) · [力矩预检](../../records/reports/roll-forward-input-integration/remaining/first-motor-preflight.json)

种子1248在第500步以112/128触发原滚翻门槛。两轮累计保存422,400条仿真转移，两个标称续测及留出种子2246待用。独立复算显示16个失分场景均完成几何转身而有效支撑转角不足。两个种子三份记录把主要支撑计分缺口定位于滚翻开始后0.54–0.92秒。后续以该窗核验4毫秒求解载荷。[对照、关键帧与原片](../../records/reports/roll-forward-input-integration/README.md) · [支撑窗口与来源](../../records/reports/roll-forward-input-integration/remaining/s1248-friction-low-low-friction/roll-prefix-readback.json)

另行开展的同种子500步载荷核验保存64,000条仿真转移。种子1248原计分111/128。失分场景共83个评分点无接触的正向转动区间，每个区间至少4个子步承重，90.86%的汇总正向子步转角发生于承重子步。保存姿态及两组目标序列的CPU预检完成，后续对照采用0.02rad颈部俯仰偏移，比较原滚翻门槛。留出种子继续保留。[载荷与时序复算](../../records/reports/roll-forward-input-integration/support-window-force/contact-sample-timing.json) · [目标预检](../../records/reports/roll-forward-input-integration/support-window-force/neck-support-target-preflight.json)

## 2026-11-21 颈部俯仰与滚翻保存点核验

同一CUDA快照上，对两个低摩擦种子比较两次原策略与0.02rad颈部俯仰候选。第500步滚翻通过数分别为118、118、114/128和110、110、108/128。实际348,160条仿真转移及六段10秒原片全部完成独立复算。候选按原门槛关闭。[完整对照和关键帧](../../records/reports/roll-forward-input-integration/neck-support-pair/README.md)

随后从同次配对训练核验第30次保存点，导出保留原61维输入和14路输出的ONNX。与第50次保存点在288条固定观测上的输出均方根差为0.0190rad。下阶段以同一站立状态可比较两个保存点的滚翻效果。[早期保存点证据](../../records/reports/roll-friction-paired-training/low-friction/checkpoint-comparison/assessment.json)

## 2026-11-21 滚翻前站立策略对照

同一次跌倒至第575步只采集一遍，三条分支恢复346项物理和控制器状态。两条原站立参照完成864/896和866/896条路线。候选将滚翻前最后25步换成官方行走策略的零指令输出，滚翻为867/896，右转组以121/128止于指令入口。固定权重对照共3,785,600条转移。计数收尾读取了错误字段，独立评分从原始账本恢复，三段录像及16个标注帧完成复核。[对照细节、曲线与录像](../../records/reports/nominal-skill-composition/pre-roll-stand-pair/README.md)

## 2026-11-21 标称滚翻接触与载荷核验

使用种子1256及原站立、官方滚翻策略，完成1,904,000条仿真转移。保存滚翻后0.40–1.10秒的156,800条场景子步载荷记录，并与20ms边界的原始位姿逐项对齐。原动作评分完整通过866/896。支撑转角失分的20个场景共有91个相关区间，其中89.06%的正向转角发生在承重子步。42.50秒原片与标注帧通过核验。[本轮方案、载荷曲线和录像](../../records/reports/nominal-skill-composition/contact-load/README.md)

独立种子1257使用同一原站立和官方滚翻策略，起身与持稳880/896，滚翻与落地站立848/896，低于七指令总入口852/896。运行在第1125步结束，记录1,008,000条仿真转移与22.50秒原片。复算把两组有效支撑转角失分的差异集中到实际仰卧组。[独立分组、支撑窗口与原片](../../records/reports/nominal-skill-composition/independent-development/README.md)

种子1257随后补录4ms载荷、14路关节目标和力矩，共1,008,000条仿真转移。滚翻与落地站立860/896，站立入口121/128。两次仿真初态一致，第1控制步后已出现轨迹数值差异。45/49与42/46个仰卧失分卸载事件恢复为头部承重。首次入口在创建CUDA前遇到类型错误，恢复运行沿用原协议。力矩复算和颈部目标下调候选的归档预检已完成。[比较、原片及命令](../../records/reports/nominal-skill-composition/support-transfer/README.md)

## 2026-11-21 滚翻颈部目标下调对照

初次采样89,600条场景转移，第100步出现1个未达0.75对齐度的释放姿态，三分支未开始。复测沿用种子1257及原覆盖门槛，在第600步核对快照恢复、首个关节目标和首个4ms电机输入。两次原策略和候选滚翻通过855/896、859/896和856/896，均止于第1125步入口。复测1,948,800条场景转移，保留全部三段22.50秒原片和逐场景4ms载荷。随后提取开发种子1256滚翻开始前的886个物理状态并核对来源。[两次采样、结果及训练样本](../../records/reports/nominal-skill-composition/neck-descent-pair/README.md)

## 2026-11-21 起身后滚翻启动状态配对训练

种子1258配对两组各用4,608个环境训练30次更新及3,317,760条环境转移。候选用种子1256的真实滚翻启动状态替换1,152个场景的默认重置。两个ONNX、完整保存点和训练录像都经过独立核验。种子1257的896场景从共同第600步恢复，官方、原始站立重置训练组和候选滚翻通过853/896、848/896和847/896，均止于第1125步入口。采样结束时遇到旧计数字段，已从全部保存轨迹、物理计数及视频独立复算。该过程共1,948,800条仿真转移。[训练过程、对照和原片](../../records/reports/nominal-skill-composition/post-getup-roll-curriculum/README.md)

## 2026-11-21 仰卧滚翻支撑取样与关节复算

逐轨迹对齐五条归档的4ms力学读回与20ms评分。仰卧有效支撑转角失分组的269个漏计区间中，268个在边界前最后一个物理步承重、后一个物理步卸载。种子1257同一批496个头部再接触事件用于左右腿与头部俯仰补偿的固定状态复算，原电机力矩及电压通过读回核验。对称髋俯仰下压估算中位数为0.00128mm。头部补偿把静态转角降至5.7×10⁻⁶°，下压估算中位数为0.00076mm。[分析和图](../../records/reports/nominal-skill-composition/support-sampling-audit/README.md)

同批496个事件完成12ms短回放。4ms CPU复现原载荷顺序的事件为失分组34/42、通过组393/454。仅统计复现事件，2ms的中间时刻卸载数分别由34降至26、由393降至290，12ms总卸载时间的配对差中位数均为0ms。两种步长共执行4,464个CPU积分步。[短回放和配对复算](../../records/reports/nominal-skill-composition/support-sampling-audit/README.md#4ms与2ms短回放)

16个CUDA场景核对了同一份4ms电机输入下的一次4ms积分和两次2ms积分，两个分支的三个外层时刻均为4ms、8ms和12ms。第一次张量读回在一次批量积分后停止。修正读回的运行完成9次批量积分，两次共160个场景积分步。[CUDA时钟与两次日志](../../records/reports/nominal-skill-composition/support-sampling-audit/README.md#cuda时钟核验)

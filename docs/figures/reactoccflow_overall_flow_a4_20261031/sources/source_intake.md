# ReactOccFlow 总体流程图：输入材料

> 2026-10-31 架构同步：以下方法事实按当前标准 DiT 实现更新。目录内已有 SVG/预览/导出文件仍属于旧 U-Net 图件，本次只修改 Markdown，尚待按新规格重生成。旧图件来源摘要保持其历史归属；新架构依据见 [DiT 迁移记录](../../../implementation/2026-10-31-standard-dit.md)。

## 用户要求

- 交付一张 SVG 总体流程图。
- 参考对话中 DOME 流程图：白底、彩色横向分区、编号标题、明确箭头、局部模块展开。
- 参考对话中 Transformer 图：展示 Norm、Attention、FFN、残差、Q/K/V 和层堆叠。
- 内容足够全面、详细；方法事实来自本项目实施计划与实际源码。
- 两张对话图片只作风格参考，不能把 DOME 的 diffusion/noise prediction 当作本项目方法。

## 材料优先级

1. `docs/superpowers/plans/2026-10-27-reactoccflow-implementation-plan.md`：权威总体方案。
2. `reactoccflow/models/react_flow.py`、`joint_transition.py`、`standard_dit.py`、`modules/base/dit.py`：当前联合流与标准 DiT 网络顺序。`modules/transition_models/FlowMatchingV2.py` 已从运行时代码删除，仅保留来源快照。
3. `reactoccflow/models/codec.py`、Swin encoder/decoder：实际 occupancy 表示。
4. `reactoccflow/losses/consistency.py`、`reactoccflow/training/joint.py`：训练与连续端点约束。
5. `reactoccflow/planning/planflow.py`、`configs/risk.yaml`、perception/bridge 文档：规划和接口边界。
6. `README.md`、`docs/implementation/progress.md`：实现状态，不能把计划当作已验证结论。

## 项目事实

### 输入、空间与时间

- 当前及过去观测 h；任务导航/地图上下文 c 是计划允许的条件。未来 GT 单独存在训练 target 中。
- E0 固定当前 ego 坐标：x 前、y 左、z 上；网格 200×200×16，分辨率 0.4 m，范围 [-40,40]×[-40,40]×[-1,5.4] m。
- occupancy 历史为 6 槽，默认最近 4 个有效，显式 history_mask；单帧 latent 为 16×50×50。
- actor 为当前观测最近 8 个，按距离与稳定身份排序，保留 4 个历史时刻。
- actor 动态状态 6 维：[x,y,sin(yaw),cos(yaw),vx,vy]；类别、尺寸、观测/速度 mask 构成条件。
- 未来 6 个时刻，标称 0.5–3 s。真实数据的监督与运动学计算采用实测时间。
- 生成流时间 t 与物理未来槽位 s 分开。

### Occupancy VAE

- 类别 embedding 18→8，加 16 高度层各自 4 维位置编码；高度折叠得到 192 通道，Conv2D 投影为 128 通道 BEV。
- patch_size=1；Swin encoder depth=[2,6]，window=5，heads=8；两次 stride-2 卷积使空间 200→100→50，通道 128→256→512。
- 中间 Conv reduce、Res–Attention–Res、Norm、激活、Conv 输出 32 通道 posterior moments。
- μ/logvar 各 16 通道。研究缓存确定性地取 μ；posterior sampling 是可选功能。
- latent 为 16×50×50；联合流入口放大 ×10，输出解码前还原。
- decoder：Conv、Res–Attention–Res、Swin depth=[6,2]、两次上采样 50→100→200，得到 128 通道。
- reshape 为每高度层 8 维，与类别 embedding 权重内积得到 18 类 logits，再 argmax 得 occupancy。
- 研究阶段 VAE 冻结；连续 logits 保留至一致性损失，不对 argmax 反传。
- 只有一套 occupancy VAE。缓存阶段用同一 encoder 分别处理历史观测与未来监督标签；训练 step 读取 latent 缓存，只对预测端点调用 decoder。DiT patch 线性投影不是第二次 VAE 编码，actor 不经过 occupancy VAE。

### 候选动作

- 目标 PlanFlow：p_phi(tau|h,c)。首版 6×2 waypoint，K=8，另加 1 条显式刹停候选，总数为 9。
- 当前实现是 deterministic local candidate generator 和离线规划接口；learned trajectory head 尚未实现，不可画成已有训练网络。
- 速度、加速度、曲率和航向变化率做运动学检查；路线是导航条件，不是专家未来轨迹。

### ReactFlow 联合网络

- 目标 p_theta(O_future,A_future|h,c,tau_k)。当前核心模型实际接收历史 latent、actor 观测和候选 tau；地图/路线接入属于总体规划条件，不能虚构当前网络地图 encoder。
- occupancy 与 actor 独立高斯噪声、共享流时间；occupancy future 6×16×50×50，actor future 8×6×6。
- history latent 与当前 noisy future latent 沿时间拼接，总计 12 槽；历史保持固定，仅积分 future。
- occupancy patchify：每帧 `16×50×50` latent 切为 100 个 `5×5` patch，每个 patch 400 个值，Linear(400→256)。12 槽共得到 `B×1200×256` token。
- 条件：timestep sinusoidal embedding→MLP；6×2 candidate 的 8-frequency Fourier embedding→MLP；相加得到 256 维条件，用于各 DiT Block 和输出层的 adaLN。空间 patch 与时间槽分别添加固定位置编码。
- 标准 DiT：8 个固定宽度 Block，hidden=256、heads=8、MLP ratio=4；各层直接对全部 1,200 个 token 做全局时空 self-attention，没有 U-Net stem/down/middle/up/skip，也不再拆分空间/时间两次注意力。
- 交互接入：第 4 个 DiT Block 后 `B×1200×256→Linear→B×1200×128`；占据序列与 actor 序列保持独立。
- actor history：[4×(6+2 masks)] + size(3) + class embed(16) = 51 维→MLP→128。
- actor noisy state 6→128，加历史条件、未来槽位 embedding、共享时间 embedding、候选 Fourier/MLP embedding；8×6=48 个 tokens。
- 层间 interaction ×2，hidden=128，heads=4：actor LayerNorm→masked self-attention→residual；随后 actor/occupancy 归一化，从同一轮状态同时计算两个 cross-attention；最后 actor LN→FFN(128→512→128,GELU)→residual。当前交互类为 `TokenInteraction`，不再保留旧瓶颈接口。
- actor←occ：Q=actor，K/V=occupancy；occ←actor：Q=occupancy，K/V=actor，使用当前有效 actor mask。
- 双向 cross delta 分别乘 learnable gate 并 residual add。当前初始化 gate=0.05。
- occupancy 只把两层交互累计的 exchange delta 投影回 256 维并加回原 DiT token，继续第 5–8 个 DiT Block。两种模态不拼接成共同 self-attention 序列。
- 输出：adaLN 最终归一化→Linear(256→400)→unpatchify，恢复 `B×12×16×50×50` occupancy velocity，只积分未来 6 槽。unpatchify 不属于 VAE 解码；ODE 后的未来 latent 由原 occupancy VAE decoder 按需解码。
- actor LN→Linear(6) 输出运动 velocity；独立 LN→Linear(1) 输出 future existence logits。
- joint 开双向交互；parallel 保留 actor self-attention/FFN 而关 exchange；occupancy_only 只用 occupancy 主干。

### Transformer 必须体现的内部顺序

- DiT 的条件 MLP 输出 6 组 shift/scale/gate，分别调制 MSA 和 MLP。
- LN→AdaLN shift/scale→QKV Multihead Self-Attention→gate→残差；LN→AdaLN shift/scale→FFN(ratio4,GELU)→gate→残差。
- 全局时空 self-attention：`B×1200×256` token 经 Q/K/V 投影后分为 8 头，注意力覆盖所有物理时间槽与空间 patch；合并 heads 后保留相同 token 数。
- actor masked self-attention 为 `Q=K=V=actor`；跨模态为 `Q=actor,K/V=occupancy` 和 `Q=occupancy,K/V=actor` 两路。候选轨迹通过条件调制注入，不是第三组 cross-attention token。
- 所有未来状态同时生成，不能画成 autoregressive masked decoder。adaLN 门控与 occupancy 输出层零初始化；结构连接的验证与初始化零输出要区别。
- Swin 的 W-MSA / SW-MSA、Norm、MLP、residual 可以作为 VAE 结构展开；它与核心联合 interaction 的普通 multihead cross-attention区分。

### 训练

- t=sigmoid(N(0,1))；x_t=(1-t)epsilon+t*x_1；目标 velocity=x_1-epsilon，两个模态分别回归。
- future-only occupancy 有效元素 MSE、actor 有效状态 MSE、future existence BCE，存在性 GT mask 仅进入监督。
- 连续预测端点 x_hat_1=x_t+(1-t)*v_theta，用冻结 VAE 解码后施加局部 occ–actor 几何一致性、静态时序一致性和类别运动学损失。
- 总目标 L=L_occ_fm+L_actor_fm+0.1L_exist+0.1L_occ_actor+0.05L_temporal+0.01L_kinematics。
- 一致性只约束已选有效 actor 的局部支持区域；occupancy 仍须覆盖新进入/未显式选择目标。不能把全部 occupancy 等同 8 个 actor 的并集。
- 不把危险未来强行变成无碰撞；世界模型保留危险分布，安全筛选在规划侧。
- VAE 冻结；标准 DiT 与 actor/interaction 从随机初始化共同训练，初始学习率均为1e-4。旧 U-Net 的运行路径和专用配置已删除；历史权重不能载入标准 DiT。

### 采样、风险与规划

- 对每候选采样 M=8 个世界（概率研究增加32）；默认 shifted Euler ODE NFE=8，对照4/8/16。
- 生成流从 t=0 噪声积分到 t=1；未来 occupancy/actor 同步更新，history固定；最终 actor heading 做归一化，existence由独立head输出。
- 解码 semantic occupancy；actor 输出位置/航向/速度/存在性。按候选-世界流式处理、及时汇总风险。
- 事件：3 s 内 ego footprint 与障碍 occupancy 或 actor footprint 相交；使用面积、尺寸、朝向、高度和实测时间；越界单独标记，禁止 clamp。
- 原始风险 r=(collision_count+0.5)/(M+1)。独立事件标签拟合 isotonic；logistic/uncalibrated 对照。
- 阈值在 tune 从0.05/0.10/0.20选择、评测时冻结；筛选运动学有效且风险低于阈值的候选，结合风险与路线/舒适性代价；无可行候选显式刹停并记录原因。
- bridge/controller：带episode、时间、序号，拒绝陈旧或真值payload，按实测时间插值，bounded throttle/steer/brake与故障减速。
- 目标规划2Hz、控制20Hz；尚无实时/闭环性能结论。

### 数据与验证（总体方案配套）

- nuScenes oracle 研究：train/tune/calibration/evaluation=560/70/70/150 scenes，train-only normalization，E0 warp后重新编码，history与target分离。
- CARLA 分支目标：固定公共历史和初始环境，改变ego动作并记录有响应NPC；按origin分组；真实分支采集尚待完成。
- 指标：occupancy mIoU；actor 运动/存在性；局部/时序一致性；动作响应；Brier/NLL/ECE、可靠性和risk–coverage；闭环碰撞/完成度/延迟与消融。
- 旧 CFM 与仍复用的 VAE 权重涉及预训练暴露；新 DiT 随机初始化也不自动消除 VAE 表示层的暴露，新划 tune/calibration 的严格独立性仍需单独控制。

## 状态表述边界

- 已有独立 VAE/CFM、E0缓存、联合flow与一致性、候选/风险/感知/桥接接口。
- learned PlanFlow、训练后真实LiDAR occupancy感知、反事实CARLA采集、独立校准效果、真实闭环及研究质量结论仍需后续工作。
- 图呈现研究总体架构；将后续组件与已实现结构作清楚区分，不将接口测试写成实验效果。

## 保护规则与环境

- OccFM/IR-WM仅只读检索，无脚本执行、导入、写入或资源运行时依赖。
- 本次仅同步项目自有 Markdown，不修改模型函数、训练配置或已有 SVG/导出图件。
- 用户AGENTS.md要求使用ReactOccFlow专用环境；优先于技能中PPTcreater环境建议，不更改参考项目环境。

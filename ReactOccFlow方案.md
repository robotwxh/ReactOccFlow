# ReactOccFlow：面向闭环驾驶的反应式占据流世界模型

## 1. 研究动机

Flow Matching（FM）已经被用于驾驶视频、LiDAR、occupancy 以及 world-action 模型，但现有工作对“自车动作如何改变周围交通参与者反应”的显式建模仍然不足。

已有工作分别覆盖了不同能力：

- [OccFM](https://arxiv.org/abs/2506.23434)：latent flow matching 用于 LiDAR/occupancy 世界模型。
- [OccSim](https://arxiv.org/abs/2603.28887)：长时域 occupancy 世界模型驱动的仿真。
- [AutoWorld](https://arxiv.org/abs/2603.28963)：多智能体交通仿真。
- [OccWAM](https://arxiv.org/abs/2610.17688)：联合生成 occupancy 和自车轨迹。
- [PhysWAM](https://arxiv.org/abs/2609.37970)：通过深度–运动几何约束提高物理一致性。
- [DriveWAM](https://arxiv.org/abs/2605.28544)、[WA-JEPA](https://arxiv.org/abs/2608.20974)：视频/未来 latent 与动作联合建模。

本方案的目标是构建一个 **occupancy-native 的反应式世界模型**：给定当前场景和一条候选自车轨迹，生成对应的未来 occupancy、周围交通参与者运动，并估计该轨迹的闭环风险。

暂定论文标题：

> **ReactOccFlow: Risk-Calibrated Reactive Occupancy Flow Matching for Closed-Loop Autonomous Driving**

## 2. 核心问题

给定历史传感器观测、地图和候选自车轨迹，模型需要学习：

1. 未来场景 occupancy 如何演化；
2. 周围车辆和行人如何响应自车动作；
3. 不同候选轨迹对应的碰撞风险和预测不确定性。

目标分布可以写为：

$$
p(O_{1:H}, A_{1:H} \mid S_0, E_{1:H})
$$

其中：

- $S_0$：当前场景表示；
- $E_{1:H}$：候选自车轨迹；
- $O_{1:H}$：未来语义 occupancy；
- $A_{1:H}$：周围车辆和行人的未来运动。

## 3. 模型结构

### 3.1 场景编码器

输入包括：

- 历史多相机和 LiDAR 特征；
- 当前 BEV/occupancy 特征；
- HD map 或矢量地图；
- 导航路线；
- 当前自车状态。

编码器输出场景 latent $S_0$，并保留 scene-centric 坐标系和 ego-centric 坐标系之间的 SE(2) 变换。

### 3.2 PlanFlow：自车候选轨迹生成

PlanFlow 使用条件 Flow Matching 生成多模态自车候选轨迹：

$$
E^{(k)}_{1:H}=\{x_t,y_t,\theta_t,v_t\}_{t=1}^{H}
$$

候选轨迹应覆盖跟车、停车、减速、绕行和变道等行为模式。与单一回归轨迹相比，FM 可以保留多峰动作分布。

### 3.3 ReactFlow：动作条件的未来世界生成

对每条候选自车轨迹，ReactFlow 联合生成：

$$
Z_{1:H}=[O_{1:H},A_{1:H},D_{1:H}]
$$

其中：

- $O$：未来语义 occupancy latent；
- $A$：周围车辆和行人的轨迹 token；
- $D$：动态目标的速度、加速度或其他运动状态。

Flow Matching 目标为：

$$
\mathcal{L}_{FM}=\left\|v_\theta(z_t,t,c,E)-(z_1-z_0)\right\|_2^2
$$

模型可以采用 scene-centric 和 ego-centric 两阶段生成：先在全局场景坐标下预测未来世界，再按照候选自车轨迹重索引到未来 ego 坐标，完成动作对齐的 refinement。

## 4. 关键创新

### 4.1 显式反应式世界建模

邻居轨迹显式条件于自车轨迹，使同一场景下的刹车、加速和变道能够产生不同的周围交通响应，而不是只预测与自车动作弱耦合的未来 occupancy。

### 4.2 Occupancy-native 交互一致性约束

在 FM 目标之外加入三类约束：

$$
\mathcal{L}=\mathcal{L}_{FM}+\lambda_1\mathcal{L}_{warp}+\lambda_2\mathcal{L}_{int}+\lambda_3\mathcal{L}_{kin}
$$

- $\mathcal{L}_{warp}$：occupancy 与目标轨迹之间的空间变换一致性；
- $\mathcal{L}_{int}$：基于 TTC、occupancy overlap 和安全距离的交互约束；
- $\mathcal{L}_{kin}$：自车和邻居车辆的速度、加速度、转向角约束。

相比 PhysWAM 的深度–运动约束，本方案直接在 occupancy 和轨迹空间中建立约束，不依赖额外的 RGB 或 metric depth 生成分支。

### 4.3 风险校准的多样本轨迹选择

对每条候选自车轨迹采样多个未来世界：

$$
\{(O^{(k,m)},A^{(k,m)})\}_{m=1}^{M}
$$

根据多次采样计算：

- 碰撞概率；
- TTC；
- 越界概率；
- occupancy 预测方差；
- 路线进度和舒适性。

在验证集上使用温度校准或 isotonic calibration，使模型输出的风险概率具有实际含义。最终选择：

$$
E^*=\arg\min_E \operatorname{Cost}(E)
\quad\text{s.t.}\quad
P_{\mathrm{collision}}(E)<\tau
$$

## 5. 实验设置

### 5.1 数据集

第一阶段建议使用：

- nuScenes：训练、开放环预测和 occupancy 评估；
- Bench2Drive：闭环驾驶评估；
- 可选 CARLA 或 OccSim 生成同一初始场景下的多种自车动作，用于验证反应式建模。

建议先限制规模：

- 预测时长 3 秒；
- occupancy 分辨率 0.5 m；
- 最近 8 个交通参与者；
- 每个场景 8 条候选自车轨迹；
- 4 或 8 步 ODE 采样。

### 5.2 对比方法

建议至少比较：

- OccFM；
- OccSim 的 occupancy world model；
- AutoWorld；
- OccWAM；
- PhysWAM；
- 一个普通 occupancy diffusion 或 autoregressive baseline。

### 5.3 评价指标

世界模型指标：

- future occupancy mIoU；
- dynamic occupancy IoU；
- agent ADE/FDE；
- energy score；
- occupancy calibration error。

规划与闭环指标：

- collision rate；
- collision per kilometer；
- route completion；
- off-road rate；
- TTC violation；
- comfort；
- inference latency；
- Flow Matching 的 NFE。

最关键的实验问题是：

> 在相同 occupancy 预测精度和相同采样步数下，ReactOccFlow 能否降低闭环碰撞率，并让风险概率更可靠？

## 6. 消融实验

至少包括：

1. 只生成 occupancy，不生成邻居轨迹；
2. occupancy 与邻居轨迹并行生成；
3. occupancy、邻居轨迹、自车轨迹联合生成；
4. 去掉 $\mathcal{L}_{warp}$；
5. 去掉交互损失 $\mathcal{L}_{int}$；
6. 去掉风险校准；
7. 比较 4、8、16 个 Flow Matching 采样步；
8. 比较不同候选轨迹数量；
9. 比较 scene-centric、ego-centric 和 scene-to-ego 两阶段坐标建模。

一个有说服力的结果是：去掉显式邻居反应分支后，开放环指标变化有限，但闭环碰撞率明显上升。

## 7. 预期贡献

1. 提出一个 occupancy-native 的反应式世界模型，联合生成自车轨迹、邻居运动和未来 occupancy。
2. 提出交互一致性 Flow Matching 目标，使自车动作能够改变周围交通参与者的预测分布。
3. 提出面向闭环规划的多样本风险校准和轨迹选择方法。
4. 在 nuScenes 和 Bench2Drive 上统一验证预测质量、风险校准和闭环安全性。

## 8. 实施路线

### 阶段一：复现基线

复现 occupancy latent 编码器、单条件 Flow Matching 预测器和基础轨迹规划器，先完成 3 秒开放环预测。

### 阶段二：加入联合生成

将邻居轨迹 token 接入 occupancy flow，并比较并行生成和联合生成两种结构。

### 阶段三：加入交互约束

实现 occupancy warp、TTC/安全距离和运动学约束，检查其对开放环和闭环指标的影响。

### 阶段四：风险校准与闭环评估

对候选轨迹进行多样本 rollout，完成风险校准、轨迹筛选和 Bench2Drive 闭环评估。

### 阶段五：效率优化

测试少步 ODE、蒸馏或 MeanFlow 风格的一步/少步推理，报告性能与速度的权衡。

## 9. 主要风险与应对

- **数据缺少同一初始场景下的多种自车动作。** 使用 CARLA、OccSim 或反事实轨迹重采样构造动作条件训练样本，并把真实日志用于主要评估。
- **occupancy 标签稀疏或噪声较大。** 先在 BEV occupancy 和动态目标框上验证，再逐步加入更高分辨率 occupancy。
- **联合生成导致训练不稳定。** 采用 scene-centric 预训练，再加入 ego-centric refinement；先固定 PlanFlow，再训练 ReactFlow。
- **闭环平台复现成本高。** 先用离线反事实风险评估，再迁移到 Bench2Drive 闭环。
- **创新点与 OccWAM 重叠。** 将重点放在“邻居反应建模 + 交互约束 + 风险概率校准”，而不是简单重复 occupancy/action 联合 FM。


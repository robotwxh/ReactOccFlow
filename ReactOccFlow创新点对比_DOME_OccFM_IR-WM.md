# ReactOccFlow 相对 DOME、OccFM、IR-WM 的创新点

核对日期：2026-10-31。依据三篇原论文的方法章节、本项目权威实施计划及实际源码。

架构同步：研究主干现为标准 DiT，采用 occupancy/actor 模态内 self-attention 与模态间双向 cross-attention。新 DiT 从随机初始化训练；旧 U-Net 的检查点和训练记录单独保留。新架构的工程验收及证据边界见 [迁移说明](docs/implementation/2026-10-31-standard-dit.md)。

**明确结论：本项目最有分量的新增设计，是“候选自车动作条件下的 occupancy–邻居 actor 双向耦合联合 Flow Matching”，以及“作用于连续生成端点的局部跨表示一致性”。动作干预验证、事件级风险校准和候选筛选是进一步的研究贡献目标，当前不能写成已经验证的效果。**

与这三篇方法存在设计差异，可以通过论文和源码核对；是否构成全领域首次、是否优于已有方法，需要更广泛查新和公平实验。本文件不使用“首次”“显著提升”“已解决反应式因果建模”等尚无证据的表述。

## 1. 先说明三篇工作已经做了什么

下表比较的是原文所定义的方法，不把仓库中存在的通用检测头、注意力层或辅助工具自动视为论文核心机制。

| 比较维度 | DOME | OccFM | IR-WM | ReactOccFlow 的新增设计／目标 |
| --- | --- | --- | --- | --- |
| 核心表示与生成机制 | 连续 occupancy latent，时空扩散 Transformer | 高压缩 latent，条件 Flow Matching 与 U-Net 式时空主干 | 视觉 BEV、流式记忆、自回归残差演化与特征对齐；默认半耦合规划 | 标准 DiT occupancy 主干与显式邻居运动流，模态内 self-attention、模态间双向 cross-attention |
| 自车动作条件 | 已有轨迹条件及轨迹重采样 | 已有历史／未来自车轨迹条件设置 | 已有 ego 动作条件和预测—规划耦合 | 将每个候选动作作为联合世界的条件；“加入动作条件”本身不算创新 |
| 本文件关注的联合对象 | occupancy 序列 | occupancy／LiDAR 世界状态 | 场景 BEV、occupancy 与自车规划 | **occupancy 场景 + 周围 actor 位置、航向、速度；存在性独立预测** |
| 新机制应体现在哪里 | — | — | — | 生成中间态的双向交换、同一世界样本的几何／运动对应、每候选多世界事件风险 |

原文依据：[DOME §3.1–3.4](https://arxiv.org/html/2410.10429v1)、[OccFM §3、§4.2、附录G](https://arxiv.org/html/2506.23434v2)、[IR-WM §III-B、§III-C](https://arxiv.org/html/2510.16729v3)。表中最后一列依据本项目[实施计划 §3.4–3.6](docs/superpowers/plans/2026-10-27-reactoccflow-implementation-plan.md)。

有三个需要特别区分的术语：

- 本项目的 **actor** 指周围车辆、骑行者、行人等交通参与者；它们的未来状态 A 与自车候选轨迹 τ 是不同变量。因此，“occupancy–actor 联合生成”不能写成“occupancy–ego action 联合生成”。当前学习式 PlanFlow 尚未实现。
- 本项目的 **Flow Matching 速度场** 指生成流时间上的向量场；它与交通场景中的 occupancy flow／物理速度不是同一概念。
- IR-WM 的 **feature alignment** 校正语义和动态特征失配；本项目提出的 **risk calibration** 针对碰撞事件概率的统计可靠性，二者不是同一种校准。

## 2. 创新点一：动作条件的 occupancy–actor 双向耦合联合流

**可明确主张的新增内容：把稠密场景 latent 与显式邻居运动纳入同一个、在生成过程中相互依赖的条件向量场。**

令 Z 为未来 occupancy latent，A 为当前选中邻居的未来动态状态，h 为当前及过去观测，τₖ 为第 k 条候选自车轨迹。目标是：

$$
p_\theta(O_{1:H},A_{1:H}\mid h,\tau_k),\qquad O=\operatorname{Decode}(Z).
$$

实际连续流状态为 Z、A；actor 存在性另设预测头。两种状态使用独立初始高斯噪声、共享生成时间 t：

$$
\frac{\mathrm d}{\mathrm dt}
\begin{bmatrix}Z_t\\A_t\end{bmatrix}
=
\begin{bmatrix}
v^Z_\theta(Z_t,A_t,h,\tau_k,t)\\
v^A_\theta(Z_t,A_t,h,\tau_k,t)
\end{bmatrix}.
$$

这里体现差异的是两个速度分量在每次向量场求值时能够交换信息。仅在 occupancy 网络后接一个 actor 回归 head，或者给两路独立采样器使用相同 t，都不足以体现这项设计。

具体实现如下：

1. occupancy 主干为 8 层、256 维、8 头标准 DiT。12 个 latent 槽按 `5×5` patch 形成 1,200 个 token，进行全局时空 self-attention；第 4 个 DiT Block 后将场景 token 投影到 128 维用于交互。
2. actor tokens 编入历史、当前观测 mask、类别、尺寸、带噪运动状态、未来槽位、共享 t 和候选 τₖ。
3. 两层交互模块先对 48 个 actor token 做 masked self-attention，再从同一轮状态同步计算两路 cross-attention：actor 更新为 `Q=actor, K/V=occupancy`；occupancy 更新为 `Q=occupancy, K/V=actor`。两路结果分别经过可学习门控和残差连接，actor 随后经过 FFN。
4. occupancy 交互增量回投到 256 维并加回原 DiT token，继续后 4 层 DiT，再经输出投影及 unpatchify 得到 occupancy 速度场；actor 分支输出运动速度场与独立存在性 logits。两组模态 token 保持独立，没有拼接后共同做 self-attention。

默认 8 个 actor、6 个未来时刻、128 维、4 个交互注意力头和2层交互，是实现配置；**这些数值本身不是创新**。显式 actor 集合只覆盖当前观测选中的目标，未选中／新进入目标仍需由 occupancy 表示覆盖。

源码依据：[StandardDiT](reactoccflow/models/standard_dit.py)、[TokenInteraction 与 JointTransition](reactoccflow/models/joint_transition.py)，尤其是 `TokenInteraction.forward`、`_actor_tokens`、`_dit_forward`；[ReactFlow.training_loss / sample](reactoccflow/models/react_flow.py)。VAE 编解码与 DiT patch 投影分别承担表示压缩和 token 化，未新增第二套 VAE。

**相对三篇工作的差异落点：**核对上表所列方法章节后，新增的是“场景占据与显式邻居运动的耦合向量场”。不能把它缩写成新提出 VAE、时空 Transformer、FM、轨迹条件或通用 cross-attention。

**当前证据与所需验证：**联合结构、训练和采样接口已经实现；已有工程与小样本验收不能证明分布建模收益。至少应控制主干、预训练、数据、参数量和 NFE，对比 occupancy-only、并行 actor 分支、单向交换、条件分解生成与双向联合流。需验证跨表示矛盾是否减少，同时报告 occupancy、actor 和联合样本质量，排除“只是增加参数或辅助监督”的解释。

## 3. 创新点二：连续端点上的局部 occupancy–actor 一致性

**可明确主张的新增内容：让同一未来样本的场景占据与主体运动在物理位置上对应，并把该约束放到可微的生成端点上。**

一个具体矛盾是：actor 分支预测某车已经变道，而该样本的车辆 occupancy 仍停留在原车道。两个分支各自具有不错的平均误差，也可能出现这种样本内不一致。

本项目利用 FM 的预测端点近似：

$$
\widehat X_1=X_t+(1-t)v_\theta(X_t,h,\tau_k,t).
$$

端点 occupancy latent 经冻结 VAE 解码，保留连续 logits；预测 actor 的位置、航向与尺寸形成连续软车体覆盖。再组合三类约束：

| 约束 | 具体对应关系 | 实现边界 |
| --- | --- | --- |
| 局部 occupancy–actor 几何一致性 | actor 软覆盖与对应语义 occupancy 概率的局部 Dice 对应 | 在有监督的 actor 局部 ROI、已知体素和有效 mask 内计算；ROI 定位可用训练 target |
| 静态时序一致性 | 固定 E0 坐标中静态区域的预测随时间保持一致 | 当前实现利用监督确定有效静态区域，不声称动态目标全部静止 |
| 类别相关运动学一致性 | 位置差分与速度对应；适用类别的速度与航向对应 | 使用实测时间与有效性；对不同类别、低速状态作区别处理 |

关键细节是：冻结 decoder 参数仍允许梯度回传到 latent；避免对 argmax occupancy 求梯度；不把全部动态 occupancy 强制等同于8个 actor 的并集。约束的目标是减少几何和运动矛盾，保留真实危险未来。规划侧再判断是否接受候选。

源码依据：[decode_endpoints、ConsistencyLosses](reactoccflow/losses/consistency.py) 及 [JointTrainingAdapter.training_loss](reactoccflow/training/joint.py)。

**贡献强度：**相对这三篇的已核对机制，这是针对“占据—邻居状态联合流”的具体一致性设计。Dice、运动学正则、端点估计或“一致性损失”这些通用工具不能单独声称原创。尤其不能把 IR-WM 的规划碰撞损失误读成对预测世界强制无碰撞，以此制造差异。

**当前证据与所需验证：**连续约束与反向传播已有实现及小样本验收。需要逐项移除三类约束，比较局部几何一致性、actor 误差、动态 IoU 与危险事件召回；若一致性改善仅来自删除障碍或消除多模态未来，就不能支持该贡献。

## 4. 研究贡献目标三：用实际动作干预验证“正确反应”

**拟新增的研究证据：在相同初始场景和公共历史下实际执行不同 ego 动作，检验邻居响应的方向、幅度和分布是否预测正确。**

仅观察“更换 τ 后模型输出变化”，只能说明条件敏感，不能证明预测了真实交互响应。对于同一辆前车，ego 加速、减速和让行应分别对应什么响应，需要有执行这些动作后的数据。

项目计划固定初始状态、环境、NPC策略和随机种子，重新建立各动作分支；记录参考与实际执行轨迹、邻居状态、occupancy、碰撞和跟踪误差。按原始场景 origin 分组划分，防止同一初始场景的动作分支跨训练／测试集合。

这与 DOME 的轨迹重采样有明确边界：后者提供几何／轨迹条件增强；从已记录场景重新提取 occupancy，不能据此获得邻车对新 ego 控制的实际响应。这个边界来自数据生成过程的差别，**不意味着 DOME 的增强方法无效**。依据：[DOME §3.3](https://arxiv.org/html/2410.10429v1)；本项目[实施计划 T8–T9](docs/superpowers/plans/2026-10-27-reactoccflow-implementation-plan.md)。

实验比较应包括“预测动作差异”与“真实动作差异”的对应，例如让行概率、纵向速度变化、横向位移、TTC及碰撞事件；跨候选共享初始噪声可减少采样波动的干扰。证据只能支持所测仿真器、控制器和NPC策略下的反应性，不能直接推广为真实交通的一般因果识别。

**当前状态：**[分支采集协议](reactoccflow/simulation/branch_collection.py) 已实现 origin／动作／seed 组织及后端契约；根据已核对[实施记录](docs/implementation/progress.md)，真实 CARLA 分支采集和响应效果尚待完成。因此这是一项实验贡献目标，不能写成已完成的数据集或已验证能力。

## 5. 研究贡献目标四：候选条件多世界的事件风险校准与筛选

**拟新增的决策环节：把每个动作对应的联合世界样本转成明确的碰撞事件概率，再用独立执行标签校准，用于可解释的候选接受与拒绝。**

对候选 τₖ 生成 M 个联合世界，在每个世界中定义3秒内事件：ego 车体与障碍 occupancy 或 actor 车体相交。先形成平滑采样频率：

$$
r_k=\frac{\sum_{m=1}^{M}\mathbf1[\text{collision}(\tau_k,O^{(m)},A^{(m)})]+0.5}{M+1}.
$$

在独立校准数据上拟合 isotonic 映射得到碰撞概率估计，再与冻结阈值、运动学有效性、进度和舒适性代价共同用于筛选；没有可行候选时执行显式减速方案并记录原因。

与 IR-WM 的真正区别不在“会不会用 occupancy 筛轨迹”：其紧耦合变体已经包含这一环节。差异目标是本项目对**每个候选自己的多世界分布**计算事件频率，并检验概率可靠性及选择后的决策结果。原文边界见 [IR-WM §III-B.4、§III-C](https://arxiv.org/html/2510.16729v3)。

需要同时说明：

- isotonic／logistic、采样频率、风险阈值都是已有统计工具；这里的贡献目标是它们与候选条件联合世界及执行标签的连接，而非新校准算法。
- 校准标签应来自实际执行对应动作后的事件；日志几何相交是另一种评价，不能自动替代反应式标签。模型生成样本的事件频率也不是天然可靠的真实概率。
- isotonic 是单调映射，通常保持风险次序，但可能产生并列值。不能仅凭阈值下停车更多，就声称规划更好；需比较相同完成度／覆盖率下的事件率，并报告 Brier、NLL、可靠性和 ECE。
- 候选筛选会改变被执行样本分布，闭环还会造成分布变化；应分别核验所有候选与最终被选候选的概率可靠性。校准不提供任意场景的安全保证。
- M=8 对稀有事件支持有限。采样预算、事件数与不确定区间应一并报告，概率研究增加 M 后不能与低预算方法直接比较效率。

源码依据：[evaluate_risk / fit_calibrator / choose_threshold](reactoccflow/evaluation/risk.py)、[候选与 Planner 接口](reactoccflow/planning/planflow.py) 和[风险配置](configs/risk.yaml)。

**当前状态：**数值风险、校准、阈值及离线候选选择接口已实现。learned PlanFlow、真实反事实事件标签、实际校准收益和闭环驾驶收益没有由本文件建立。**这项目前更适合作为系统／实证贡献目标，不能单独作为强算法创新。**

## 6. 对每篇基线，应该如何准确描述差异

| 对比对象 | 推荐强调的新增点 | 应避免的表述 |
| --- | --- | --- |
| DOME | 显式邻居状态与occupancy在生成中耦合；局部跨表示约束；通过执行分支检验动作响应 | “首次实现轨迹控制”“首次使用连续latent／时空Transformer”；将扩散换成FM后直接宣称新颖或更快 |
| OccFM | 在其代码与表示基础上新增邻居状态流、标准 DiT 层间双向交换和端点一致性 | “首次将FM用于occupancy”“首次加入未来ego轨迹”；把新增actor head直接等同于反应式联合分布 |
| IR-WM | 显式occupancy–邻居状态的随机联合生成；候选特定多世界事件风险与独立概率校准目标 | “首次让预测服务规划”“首次用occupancy筛候选”“首次做动作条件／残差／对齐” |

对 OccFM 还应公平指出：其附录G已将前景连续性和周围主体多模态预测列为局限与后续方向。本项目可以研究这些问题，但不能把“提出这个问题”本身作为新贡献。依据：[OccFM 附录G](https://arxiv.org/html/2506.23434v2)。

## 7. 哪些属于继承、实现细节或工程工作

| 项目设计 | 应如何归类 |
| --- | --- |
| occupancy VAE、语义／高度编码、Swin、16×50×50 latent | 表示基础与复用组件 |
| 标准DiT、AdaLN、patch投影、Fourier轨迹编码；已移除运行代码的历史U-Net对照 | 基座设计与已有技术组合；替换主干本身不能直接列作原创 |
| 线性CFM路径、速度回归、Euler ODE、少步采样 | 通用生成方法；实际质量／效率仍需测量 |
| 128维、4头、2层、8个actor、6未来帧 | 配置与预算，不构成贡献 |
| PlanFlow模块命名、确定性候选、刹停候选 | 规划接口／工程设计；学习式head当前未实现 |
| E0统一、future target隔离、按当前观测选actor | 正确实验与部署的必要条件，支撑可信性 |
| 参考仓库隔离、来源记录、DDP、缓存、断点恢复、可视化 | 可复现工程贡献，不应代替方法创新 |
| 从LiDAR替代视觉输入、接入CARLA | 系统适配工作；需要新机制和实证才可能成为独立研究贡献 |

特别是，**工程上“新写了模块”不等于学术上“新提出了方法”**。双向注意力和一致性也已有广泛应用；本项目应证明它们在这一联合随机状态与动作条件任务中的必要性。

## 8. 当前最适合使用的贡献表述

以下表述适合开题、方法介绍及尚无正式效果结论的论文初稿：

> 1. 我们提出候选自车动作条件下的 occupancy–actor 联合 Flow Matching，在共享生成时间的场景与邻居运动状态之间引入 DiT 层间双向交互，建模同一候选动作下相互关联的未来。
> 2. 我们设计作用于连续生成端点的局部占据—主体、静态时序与类别运动学约束，使场景几何与显式运动保持对应，并保留未显式建模目标和危险未来。
> 3. 我们设计受控动作分支验证与事件级风险评价流程，拟通过独立执行标签检验响应真实性、概率可靠性，以及多世界预测对候选筛选的实际价值。

第三条使用“设计／拟检验”，而非“实验表明”；前两条使用“提出／设计”，而非“显著改善”。正式实验后，再按证据将响应验证和风险决策拆成独立贡献，或收窄主张。

一句话版本：**ReactOccFlow 的核心方法差异，是把候选自车动作下的稠密占据与显式邻居运动放进同一个可交互、可约束的生成流；后续要证明这种联合世界能给出正确的交通响应和对决策有用的事件风险。**

## 9. 最小证据闭环与核对范围

| 主张 | 最小必要对照 | 支持它的证据 |
| --- | --- | --- |
| 双向联合流有价值 | 原始动作条件OccFM、occupancy-only、并行分支、单向／条件分解、双向联合流 | 同预算下联合一致性与预测分布质量改善，且不以某一分支明显退化换取 |
| 端点约束有价值 | 无约束、各单项约束、完整约束 | 几何／时序矛盾减少，同时危险召回和多样性得到检查 |
| 学会了正确反应 | 无动作条件、全局动作embedding、显式actor／双向结构；同origin执行分支 | 预测差异与实际响应差异对应，而不仅是输出变化 |
| 校准改善决策 | 无校准／logistic／isotonic；固定模型与候选，冻结阈值 | 独立事件标签上的概率评分，以及完成度／覆盖率匹配后的闭环比较 |

比较 DOME、OccFM、IR-WM 时需另行对齐输入模态、oracle／感知条件、预训练、网格、时域、可见性mask和采样预算。当前oracle研究结果不能直接与IR-WM的相机端到端结果作优劣排名。预训练OccFM权重曾接触官方700个训练场景，因此项目后来划出的70个tune／70个calibration场景不能自动称为该预训练模型的未见数据；严格独立校准需要处理这一暴露问题。依据：[README](README.md)、[实施记录](docs/implementation/progress.md)。

核对使用 DOME v1、OccFM v2、IR-WM v3 的官方原文与本地PDF；重点检查 DOME §3、OccFM §3／§4.2／附录G、IR-WM §III及实验协议。架构同步依据当前源码、README和2026-10-31 DiT 迁移记录，旧实验事实仍按原始记录归属。此次仅同步文档，未新增训练、仿真或三篇论文复现实验，也不据此判断训练进程的实时状态。

对原文“未见该机制”的判断仅限上述方法与目标定义；不外推为作者所有版本、所有仓库分支或全领域都不存在。三个原始研究是方法事实的一手来源，不能独立证明本项目优越性。更广泛相关工作见既有[科学问题与解决机制](ReactOccFlow科学问题与解决机制.md)与[参考文献清单](ReactOccFlow参考文献.md)，本次未重新核验其中其余文献。

### 原论文

1. Gu, S., et al. **DOME: Taming Diffusion Model into High-Fidelity Controllable Occupancy World Model**. arXiv:2410.10429v1, 2024。[原文](https://arxiv.org/html/2410.10429v1)；[本地PDF](docs/references/05_DOME_2410.10429.pdf)。
2. Liu, T., Zhao, S., & Rhinehart, N. **Towards foundational LiDAR world models with efficient latent flow matching**. arXiv:2506.23434v2, 2025。[原文](https://arxiv.org/html/2506.23434v2)；[本地PDF](docs/references/01_OccFM_2506.23434.pdf)。
3. Mei, J., et al. **Vision-Centric 4D Occupancy Forecasting and Planning via Implicit Residual World Models**. arXiv:2510.16729v3, 2026修订，2025首发。[原文](https://arxiv.org/html/2510.16729v3)；[本地PDF](docs/references/07_IR-WM_2510.16729.pdf)。

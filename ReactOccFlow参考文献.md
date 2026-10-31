# ReactOccFlow 方案参考文献（30 篇）

核验日期：**2026-10-27**。对应方案：[ReactOccFlow方案.md](./ReactOccFlow方案.md)。

本清单按方案所需的证据链筛选：**occupancy 世界模型 → Flow Matching → 自车条件交互预测 → 风险校准 → 闭环评测**。除生成模型论文外，也纳入必要的交互规划和统计评价文献。

**书目信息口径**：27 篇 arXiv 条目的题名、作者和首发日期已从官方摘要页在线核对；其余 3 篇从 PMLR 论文页或 DOI/Crossref 元数据核对。arXiv 条目统一标注“预印本首发年份/日期”，不把它与正式会议发表年份混用。部分论文已有正式出版版本，投稿前可按目标期刊格式替换为对应出版条目。下述“建议引用位置”和“比较边界”是针对本方案的选文建议；本清单不表示已逐篇完整复现。

## 优先精读顺序

建议先读 **[1] OccFM、[2] OccWAM、[9] AutoWorld、[21] DTPP、[18] M2I、[16] GoalFlow、[13] Flow Matching、[23] 概率校准、[25] 风险控制、[29] Bench2Drive**。这 10 篇分别对应代码基座、最相似方法、邻居行为、动作条件、交互分解、候选轨迹、生成目标、校准、阈值选择和闭环评测。

## 分类与方案章节映射

| 文献编号 | 主题 | 在方案中的主要作用 |
|---|---|---|
| [1]–[12] | 世界模型及 world-action | 研究动机、方法边界、主要对比 |
| [13]–[16] | FM 与候选轨迹 | PlanFlow、ReactFlow、训练路径与采样 |
| [17]–[22] | 邻居交互与预测–规划 | 反应式建模、动作条件、交互消融 |
| [23]–[26] | 概率校准与风险评价 | 风险校准、阈值、评分规则 |
| [27]–[30] | 数据与闭环平台 | 数据来源、标注、仿真、实验协议 |

## 一、Occupancy 世界模型与 World-Action 模型（12 篇）

### [1] OccFM — 核心基线、代码基础

Tianran Liu, Shengwen Zhao, Nicholas Rhinehart. **[Towards foundational LiDAR world models with efficient latent flow matching](https://arxiv.org/abs/2506.23434)**. arXiv:2506.23434；预印本首发 2025-06-30。

- **与方案的关系**：提供 occupancy 压缩表示、latent conditional flow matching，以及未来自车轨迹条件下的语义 occupancy 预测，是本方案最直接的实现起点。
- **建议引用位置**：§1 研究动机；§3.3 ReactFlow；§5.2 对比方法。
- **比较或使用边界**：原论文已包含未来轨迹条件；仅增加 trajectory conditioning 不能作为本方案的新贡献。

### [2] OccWAM — 最直接竞争方法

Qihang Huang, Wenhui Huang, Collister Chua, Shiyu Fang, Zhan Chen, Songyan Zhang, Zhidong Wang, Daiying Tian, Chen Lv. **[OccWAM: Occupancy World-Action Models with Joint Scene-to-Ego Flow Matching](https://arxiv.org/abs/2610.17688)**. arXiv:2610.17688；预印本首发 2026-10-13。

- **与方案的关系**：联合生成未来 occupancy 和自车轨迹，并采用 scene-centric 到 ego-centric 的两阶段流模型，用于界定本方案与已有 world-action FM 的重叠。
- **建议引用位置**：§1 研究动机；§3 模型结构；§6 消融；§7 贡献边界。
- **比较或使用边界**：occupancy–action 联合 FM 和 scene-to-ego 设计已有先例；应重点比较显式邻居变量、动作干预响应和风险校准。

### [3] OccWorld — 自回归 occupancy 基线

Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, Jiwen Lu. **[OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](https://arxiv.org/abs/2311.16038)**. arXiv:2311.16038；预印本首发 2023-11-27。

- **与方案的关系**：使用 occupancy tokenizer 和时空生成 Transformer 联合预测未来场景与自车运动，支撑采用 occupancy 作为世界状态表示的动机。
- **建议引用位置**：§1 研究动机；§3.1 场景编码；§5.2 对比方法。
- **比较或使用边界**：这是离散 token 自回归基线，不应归类为 Flow Matching。

### [4] Copilot4D — 离散扩散世界模型基线

Lunjun Zhang, Yuwen Xiong, Ze Yang, Sergio Casas, Rui Hu, Raquel Urtasun. **[Copilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion](https://arxiv.org/abs/2311.01017)**. arXiv:2311.01017；预印本首发 2023-11-02。

- **与方案的关系**：将点云观测 token 化并通过离散扩散预测未来，适合比较不同生成范式在未来状态预测中的质量、效率与误差累积。
- **建议引用位置**：§1 相关工作；§3.3 生成模型；§5.2 对比方法。
- **比较或使用边界**：主要表示为点云观测的离散 token；与语义 occupancy 指标对比时需对齐数据、预测时长和表示。

### [5] DOME — 可控 occupancy 扩散基线

Songen Gu, Wei Yin, Bu Jin, Xiaoyang Guo, Junming Wang, Haodong Li, Qian Zhang, Xiaoxiao Long. **[DOME: Taming Diffusion Model into High-Fidelity Controllable Occupancy World Model](https://arxiv.org/abs/2410.10429)**. arXiv:2410.10429；预印本首发 2024-10-14。

- **与方案的关系**：研究时空扩散 Transformer 和轨迹重采样控制，直接关联本方案的轨迹条件未来 occupancy 生成。
- **建议引用位置**：§3.3 ReactFlow；§4.1 动作条件；§5.2 对比方法。
- **比较或使用边界**：需检验创新是否超出已有轨迹控制机制；轨迹重采样本身不提供未执行动作对应的真实交通响应标签。

### [6] Drive-OccWorld — 动作条件与 occupancy 规划参考

Yu Yang, Jianbiao Mei, Yukai Ma, Siliang Du, Wenqing Chen, Yijie Qian, Yuxiang Feng, Yong Liu. **[Driving in the Occupancy World: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving](https://arxiv.org/abs/2408.14197)**. arXiv:2408.14197；预印本首发 2024-08-26。

- **与方案的关系**：结合历史 BEV、动作条件、未来 occupancy/运动场预测和 occupancy-based planner，为从世界预测接到规划提供工程参考。
- **建议引用位置**：§3.1 场景编码；§4.3 轨迹选择；§5.2 对比方法。
- **比较或使用边界**：文中的 occupancy flow 是场景运动场，不等于生成模型的 Flow Matching 速度场。

### [7] IR-WM — 残差预测与预测–规划耦合参考

Jianbiao Mei, Yu Yang, Xuemeng Yang, Licheng Wen, Jiajun Lv, Botian Shi, Yong Liu. **[Vision-Centric 4D Occupancy Forecasting and Planning via Implicit Residual World Models](https://arxiv.org/abs/2510.16729)**. arXiv:2510.16729；预印本首发 2025-10-19。

- **与方案的关系**：通过动作和场景条件预测隐式未来残差，并研究预测与规划的耦合，可用于设计静态信息复用和动态变化建模。
- **建议引用位置**：§3.3 世界状态；§4.2 时序一致性；§5.2 对比方法。
- **比较或使用边界**：残差建模、对齐模块和预测–规划耦合已有先例；不能笼统写成新提出的机制。

### [8] OccSim — 长时域仿真相关工作

Tianran Liu, Shengwen Zhao, Mozhgan Pourkeshavarz, Weican Li, Nicholas Rhinehart. **[OccSim: Multi-kilometer Simulation with Long-horizon Occupancy World Models](https://arxiv.org/abs/2603.28887)**. arXiv:2603.28887；预印本首发 2026-03-30。

- **与方案的关系**：研究 occupancy 世界模型驱动的长时域环境生成，并通过静态世界模型与动态布局模块构建仿真场景。
- **建议引用位置**：§1 研究动机；§5.1 仿真数据；§9 数据风险。
- **比较或使用边界**：可作为仿真研究参照；实际数据生成方案仍取决于可用代码、动态参与者控制接口和标签导出能力。

### [9] AutoWorld — 多智能体生成的重点对照

Mozhgan Pourkeshavarz, Tianran Liu, Nicholas Rhinehart. **[AutoWorld: Learning Multi-Agent Traffic Simulation with Self-Supervised World Models](https://arxiv.org/abs/2603.28963)**. arXiv:2603.28963；预印本首发 2026-03-30。

- **与方案的关系**：结合 LiDAR occupancy 世界模型和多智能体运动生成，并研究潜在动态监督及多样性采样，直接关联 occupancy 与 actor motion 联合建模。
- **建议引用位置**：§4.1 显式邻居反应；§5.2 对比方法；§6 消融。
- **比较或使用边界**：已有世界状态到多智能体行为建模，且讨论规划器控制自车时的行为；不能将本方案泛称为首次连接 occupancy 与邻居行为。

### [10] PhysWAM — 几何一致性 world-action 对照

Dhruv Parikh, Fengcheng Yu, Quankai Gao, Jiawei Yang, Junjie Ye, Maulik Bhatt, Thang Vu, Charles Ochoa, Rowan McAllister, Igor Vasiljevic, Rajgopal Kannan, Viktor Prasanna, Vitor Guizilini, Yue Wang. **[PhysWAM: Physically Consistent World Action Model for Autonomous Driving](https://arxiv.org/abs/2609.37970)**. arXiv:2609.37970；预印本首发 2026-09-29。

- **与方案的关系**：以统一 Flow Matching 模型生成视频、深度和自车运动，使用几何投影约束耦合世界和动作，支撑本方案对物理一致性的讨论。
- **建议引用位置**：§4.2 一致性损失；§5.2 对比方法。
- **比较或使用边界**：几何一致性已有先例；本方案需证明 occupancy–actor 一致性带来的额外收益，并对齐观测模态和计算预算。

### [11] DriveWAM — 视频–动作联合 FM 相关工作

Chen Shi, Jinrui Xu, Shaoshuai Shi, Kehua Sheng, Bo Zhang, Li Jiang. **[DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving](https://arxiv.org/abs/2605.28544)**. arXiv:2605.28544；预印本首发 2026-05-27。

- **与方案的关系**：使用联合视频–动作 Flow Matching 和自回归 rollout，可用于说明本方案与视频 world-action 模型在表示和任务上的差别。
- **建议引用位置**：§1 相关工作；§3 联合生成；§5.3 效率。
- **比较或使用边界**：视频与动作的联合 FM 本身已有研究；其长时域缓存设计也不应误写为本方案的新贡献。

### [12] WA-JEPA — 未来 latent–动作生成对照

Xinlin Wang, Yujiao Xiang, Yuheng Zhou, Jingqi Wang, Minqing Huang, Jiajie Huang, Dongxu Wei, Tingguang Zhou, Xiyang Wang, Gong Chen, Zhi Xu, Feiyang Tan, Hangning Zhou, Mu Yang. **[WA-JEPA: Rethinking the Video JEPA Paradigm for World-Action Modeling in Autonomous Driving](https://arxiv.org/abs/2608.20974)**. arXiv:2608.20974；预印本首发 2026-08-21。

- **与方案的关系**：将视频 JEPA 改造为未来 latent 预测与动作联合建模，适合比较隐式特征空间与可度量 occupancy 表示。
- **建议引用位置**：§1 相关工作；§3.3 latent 预测；§5.2 对比方法。
- **比较或使用边界**：比较应落在显式几何、交互条件和风险评估上，避免将 JEPA 一概描述为确定性回归。

## 二、Flow Matching 理论与轨迹生成（4 篇）

### [13] Flow Matching — FM 理论基础

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le. **[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)**. arXiv:2210.02747；预印本首发 2022-10-06。

- **与方案的关系**：给出通过回归条件向量场学习生成流的基础方法，是方案中线性噪声–数据插值与速度回归目标的主要理论引用。
- **建议引用位置**：§3.2 PlanFlow；§3.3 FM 目标。
- **比较或使用边界**：需分别定义物理预测时间与生成流时间；FM 时间参数不是未来驾驶帧的时间戳。

### [14] Rectified Flow — 直线路径与少步采样基础

Xingchao Liu, Chengyue Gong, Qiang Liu. **[Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)**. arXiv:2209.03003；预印本首发 2022-09-07。

- **与方案的关系**：提出 Rectified Flow，用于解释直线概率路径、速度场学习，以及 reflow 与推理效率之间的关系。
- **建议引用位置**：§3.3 FM 路径；§8 效率优化。
- **比较或使用边界**：采用直线插值不自动保证任意少步采样下的精度；需实测 NFE、时延和预测质量。

### [15] Conditional Flow Matching / OT-CFM — CFM 与路径耦合基础

Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, Yoshua Bengio. **[Improving and generalizing flow-based generative models with minibatch optimal transport](https://arxiv.org/abs/2302.00482)**. arXiv:2302.00482；预印本首发 2023-02-01。

- **与方案的关系**：研究广义 CFM 和小批量最优传输耦合，可支持 source–target 配对方式及训练路径选择的设计。
- **建议引用位置**：§3.3 CFM 目标；§6 路径耦合消融。
- **比较或使用边界**：小批量 OT 配对用于生成训练，不等同于交通参与者身份匹配或轨迹数据关联。

### [16] GoalFlow — PlanFlow 的直接参考

Zebin Xing, Xingyu Zhang, Yang Hu, Bo Jiang, Tong He, Qian Zhang, Xiaoxiao Long, Wei Yin. **[GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving](https://arxiv.org/abs/2503.05689)**. arXiv:2503.05689；预印本首发 2025-03-07。

- **与方案的关系**：通过目标点引导 Flow Matching 生成多模态轨迹并进行候选选择，为自车候选轨迹模块提供直接参考。
- **建议引用位置**：§3.2 PlanFlow；§4.3 候选选择。
- **比较或使用边界**：方案内 PlanFlow 是暂定模块名；候选轨迹 FM 不能单独作为主要创新，需连接动作条件世界预测。

## 三、交互预测、反应式仿真与预测–规划耦合（6 篇）

### [17] TrafficSim — 多智能体仿真基础

Simon Suo, Sebastian Regalado, Sergio Casas, Raquel Urtasun. **[TrafficSim: Learning to Simulate Realistic Multi-Agent Behaviors](https://arxiv.org/abs/2101.06557)**. arXiv:2101.06557；预印本首发 2021-01-17。

- **与方案的关系**：学习联合多智能体行为并在训练中展开可微仿真，适合支撑交互一致性、长时域训练和真实感评测的讨论。
- **建议引用位置**：§4.1 多智能体交互；§4.2 交互损失；§5.1 仿真。
- **比较或使用边界**：需保留真实危险行为的统计特征；将预测结果强行约束为零碰撞并不能证明世界模型真实。

### [18] M2I — 显式交互因子分解参考

Qiao Sun, Xin Huang, Junru Gu, Brian C. Williams, Hang Zhao. **[M2I: From Factored Marginal Trajectory Prediction to Interactive Prediction](https://arxiv.org/abs/2202.11884)**. arXiv:2202.11884；预印本首发 2022-02-24。

- **与方案的关系**：将交互主体区分为 influencer 和 reactor，并通过边缘预测与条件预测构造联合轨迹分布，直接支持邻居反应分支设计。
- **建议引用位置**：§4.1 邻居反应；§6 条件交互消融。
- **比较或使用边界**：influencer/reactor 结构是预测建模，不能仅凭条件相关性声称识别了自车动作的因果效应。

### [19] GameFormer — 交互预测–规划联合参考

Zhiyu Huang, Haochen Liu, Chen Lv. **[GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving](https://arxiv.org/abs/2303.05760)**. arXiv:2303.05760；预印本首发 2023-03-10。

- **与方案的关系**：基于层级博弈和 Transformer 迭代建模多主体响应，并联合推理自车规划与其他主体行为。
- **建议引用位置**：§3.3 actor tokens；§4.1 交互；§5.2 交互预测基线。
- **比较或使用边界**：可作为“独立预测”与“交互预测”之间的结构对照，说明 FM 的收益是否超出交互模块本身。

### [20] DIPP — 可微预测–规划与代价函数参考

Zhiyu Huang, Haochen Liu, Jingda Wu, Chen Lv. **[Differentiable Integrated Motion Prediction and Planning with Learnable Cost Function for Autonomous Driving](https://arxiv.org/abs/2207.10422)**. arXiv:2207.10422；预印本首发 2022-07-21。

- **与方案的关系**：使用可微优化连接周围主体预测、自车规划和可学习代价，适合支撑安全距离、舒适性和运动学约束的设计。
- **建议引用位置**：§4.2 约束；§4.3 规划代价；§6 消融。
- **比较或使用边界**：需区分约束改善轨迹优化与世界分布预测改善带来的收益，避免仅靠新增优化器获得提升。

### [21] DTPP — 自车条件预测和树策略规划的关键对照

Zhiyu Huang, Peter Karkus, Boris Ivanovic, Yuxiao Chen, Marco Pavone, Chen Lv. **[DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning in Autonomous Driving](https://arxiv.org/abs/2310.05885)**. arXiv:2310.05885；预印本首发 2023-10-09。

- **与方案的关系**：联合学习 ego-conditioned 预测与规划代价，并采用树结构策略规划，是检验“候选自车动作影响邻居未来”主张的重要参照。
- **建议引用位置**：§2 核心问题；§4.1 反应式建模；§4.3 决策。
- **比较或使用边界**：动作条件预测和交互规划早已有研究；本方案差异需具体到 occupancy–actor 联合流、干预评测或风险控制。

### [22] Contingencies from Observations — 多未来分支与闭环响应参考

Nicholas Rhinehart, Jeff He, Charles Packer, Matthew A. Wright, Rowan McAllister, Joseph E. Gonzalez, Sergey Levine. **[Contingencies from Observations: Tractable Contingency Planning with Learned Behavior Models](https://arxiv.org/abs/2104.10558)**. arXiv:2104.10558；预印本首发 2021-04-21。

- **与方案的关系**：研究面对多种他车行为的 contingent planning，并在 CARLA 交互场景中进行闭环评估，支持多未来世界样本的决策动机。
- **建议引用位置**：§2 多模态未来；§4.3 风险决策；§5.1 交互场景。
- **比较或使用边界**：该文使用条件自回归 normalizing flow，不应归类为现代 Flow Matching。

## 四、概率校准、风险控制与概率评价（4 篇）

### [23] Temperature Scaling — 分类概率校准基础

Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. **[On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)**. International Conference on Machine Learning (ICML), 2017, PMLR；pp. 1321–1330。

- **与方案的关系**：介绍现代神经网络的校准问题和温度缩放，为 occupancy 分类概率或碰撞判别头的后处理校准提供基础。
- **建议引用位置**：§4.3 风险校准；§5.3 可靠性图、ECE。
- **比较或使用边界**：temperature scaling 的输入通常为 logits；不能直接把未经定义的多样本碰撞频率当作分类 logits。

### [24] Calibrated Regression — 连续预测分布校准基础

Volodymyr Kuleshov, Nathan Fenner, Stefano Ermon. **[Accurate Uncertainties for Deep Learning Using Calibrated Regression](https://proceedings.mlr.press/v80/kuleshov18a.html)**. International Conference on Machine Learning (ICML), 2018, PMLR；pp. 2796–2804。

- **与方案的关系**：研究回归预测区间和预测分布的校准，可支持轨迹位置、速度、距离及 TTC 等连续量的不确定性评估。
- **建议引用位置**：§4.3 连续风险量；§5.3 校准指标。
- **比较或使用边界**：回归区间校准与二元碰撞事件的概率校准是不同目标，应分别定义评价指标。

### [25] Conformal Risk Control — 风险阈值选择的方法依据

Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, Tal Schuster. **[Conformal Risk Control](https://arxiv.org/abs/2208.02814)**. arXiv:2208.02814；预印本首发 2022-08-04。

- **与方案的关系**：将 conformal 方法扩展到单调损失的期望风险控制，可为独立校准集上的风险阈值选择提供更严谨的参照。
- **建议引用位置**：§4.3 风险阈值；§5.3 风险–覆盖率；§9 分布偏移。
- **比较或使用边界**：相关保证依赖校准/测试条件和损失假设，不能直接宣称为任意闭环分布偏移下的逐场景安全保证。

### [26] Proper Scoring Rules — 概率预测评价基础

Tilmann Gneiting, Adrian E Raftery. **[Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437)**. Journal of the American Statistical Association, 2007, 102(477): 359–378. DOI: 10.1198/016214506000001437。

- **与方案的关系**：系统讨论严格适当评分规则，可作为 Brier score 和多变量 energy score 等概率评价指标的理论依据。
- **建议引用位置**：§5.3 概率预测与 energy score；§6 校准消融。
- **比较或使用边界**：采样方差描述预测分布的离散程度，不自动等于模型认知不确定性，也不等于已经校准的风险。

## 五、数据集、Occupancy 标注与闭环仿真（4 篇）

### [27] nuScenes — 真实数据集

Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, Oscar Beijbom. **[nuScenes: A multimodal dataset for autonomous driving](https://arxiv.org/abs/1903.11027)**. arXiv:1903.11027；预印本首发 2019-03-26。

- **与方案的关系**：提供多传感器驾驶观测和目标框等标注，支持真实日志中的世界预测、目标运动构建和开放环评测。
- **建议引用位置**：§5.1 数据集与划分。
- **比较或使用边界**：单条记录只有已执行自车行为对应的未来；不能把未执行候选动作与原日志未来直接配成反事实真值。

### [28] Occ3D — occupancy 标注与评测

Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, Hang Zhao. **[Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving](https://arxiv.org/abs/2304.14365)**. arXiv:2304.14365；预印本首发 2023-04-27。

- **与方案的关系**：提供 nuScenes/Waymo 的稠密可见性相关语义 occupancy 标签与评测，用于训练 occupancy 表示并报告 mIoU 等指标。
- **建议引用位置**：§3.1 occupancy 数据；§5.1 标签；§5.3 mIoU。
- **比较或使用边界**：0.5 m 的方案设置需要与所用标注和基线网格对齐；应声明分辨率、有效范围、类别、可见性 mask 和重采样规则。

### [29] Bench2Drive — 闭环驾驶能力评测

Xiaosong Jia, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, Junchi Yan. **[Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving](https://arxiv.org/abs/2406.03877)**. arXiv:2406.03877；预印本首发 2024-06-06。

- **与方案的关系**：提供覆盖多种交互场景的闭环端到端驾驶评测，支持检验开放环预测提升是否转化为实际驾驶收益。
- **建议引用位置**：§5.1 闭环平台；§5.3 闭环指标；§6 消融。
- **比较或使用边界**：从 nuScenes occupancy 模型迁移到该平台需要传感器、坐标、动作接口和数据适配，不能作为直接可运行的既有能力。

### [30] CARLA — 可控仿真与反事实试验平台

Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, Vladlen Koltun. **[CARLA: An Open Urban Driving Simulator](https://arxiv.org/abs/1711.03938)**. arXiv:1711.03938；预印本首发 2017-11-10。

- **与方案的关系**：为自车动作干预、相同初始条件下的多次交互 rollout，以及传感器和环境控制提供基础仿真平台。
- **建议引用位置**：§5.1 反事实数据；§8 实施路线；§9 数据风险。
- **比较或使用边界**：需固定初始状态、随机种子和 NPC 控制策略，并确认 NPC 会随自车动作反应；仿真证据不等同于真实交通因果保证。

## 写入方案和论文时的引用组合建议

| 要论证的内容 | 建议文献 |
|---|---|
| Occupancy 作为可度量世界状态的价值 | [1]、[3]、[5]、[28] |
| 已有 occupancy–action 联合生成，需明确区别 | [2]、[6]、[7] |
| 与视频/latent world-action 方法的表示差异 | [10]、[11]、[12] |
| CFM 训练、线性插值、少步 ODE | [13]、[14]、[15] |
| 自车多模态候选轨迹生成 | [16] |
| 周围主体随自车候选动作响应 | [9]、[18]、[19]、[21]、[22] |
| 联合预测与规划、可微代价 | [6]、[7]、[20]、[21] |
| 多智能体仿真与交互 rollout | [8]、[9]、[17]、[30] |
| 概率校准与风险阈值选择 | [23]、[24]、[25] |
| Brier score、energy score 等概率指标 | [26] |
| 真实日志、occupancy 标注和闭环能力评测 | [27]、[28]、[29]、[30] |

## 文献对本方案的三点直接启示

1. **收窄创新主张。** [1] 已有未来轨迹条件，[2] 已有 occupancy–ego action 联合 FM，[9] 已有 world-model-grounded 多主体行为，[21] 已有 ego-conditioned 预测与规划。因此，“联合生成”或“显式条件于自车轨迹”应作为基础设计；新的研究问题应落在可检验的 occupancy–actor 耦合、动作干预响应及校准后决策收益。
2. **区分条件预测与反事实证据。** [18]、[21]、[22] 提供了交互和动作条件预测的参照，但仅增加邻居轨迹 head 或扰动 ego trajectory，无法自动获得未执行动作的真实响应。建议借助 [30] 构造固定初始条件下的交互试验，再用 [29] 评价闭环驾驶。
3. **区分风险校准与安全保证。** [23]、[24] 支持校准方法，[25] 支持满足其假设时的风险控制，[26] 支持概率分布评分。应在独立校准集上选择参数，并在测试集及闭环分布变化下分别报告校准与安全指标；不能仅凭一个概率阈值宣称获得严格的逐场景安全保证。

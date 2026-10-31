# Flow Matching × 自动驾驶世界模型：在线论文清单

检索截止：2026-10-26。来源以 arXiv 公开号页、HTML/PDF 正文和作者项目页为主；本清单不把论文标题中的 “flow” 自动解释为 flow matching。严格核心要求 FM/RF/CFM/MeanFlow 目标直接作用于未来世界状态，或与世界状态联合生成；planning 类单独列出。

## 结论

本轮整理 113 条去重记录：core 20篇；world_action 24篇；planning 15篇；scenario 4篇；extension 9篇；adjacent 12篇；synthesis 7篇；uncertain 3篇；excluded 14篇。最早的严格核心起点是 GAIA-2、Orbis、OccFM；2026 年迅速扩展到 occupancy WAM、video WAM、geometry WAM 和 traffic simulation。

## 严格核心：未来世界状态/占据/视频/几何由 FM/RF/CFM/MeanFlow 生成

| arXiv | 论文 | 首发日期 | 生成对象 / FM 用法 | 证据定位 |
|---|---|---:|---|---|
| [2502.11663](https://arxiv.org/abs/2502.11663) | MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction | 2025-02-17 | 未来视频生成采用Rectified Flow | §3.1 式1 |
| [2503.15875](https://arxiv.org/abs/2503.15875) | MiLA: Multi-view Intensive-fidelity Long-term Video Generation World Model for Autonomous Driving | 2025-03-20 | 通过rectified-flow视频生成及纠错扩展预测时长 | §3.1；§3.3–3.5 |
| [2503.20523](https://arxiv.org/abs/2503.20523) | GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving | 2025-03-26 | 世界预测主干本身FM；§2.2.2线性高斯插值与速度训练。 | https://arxiv.org/html/2503.20523 ; 2 Model > 2.2 World Model ; /tmp/fm_search_online/papers/2503.20523.txt:522 |
| [2504.19077](https://arxiv.org/abs/2504.19077) | Learning to Drive from a World Model | 2025-04-27 | 世界模型视觉预测以RF训练，规划头另用MHP；用于on-policy驾驶策略训练 | §4.2.2 式4–5 |
| [2506.23434](https://arxiv.org/abs/2506.23434) | Towards foundational LiDAR world models with efficient latent flow matching | 2025-06-30 | CFM/RF预测历史条件下的未来occupancy latent | https://arxiv.org/html/2506.23434v2#S3.SS2.p1.1 |
| [2507.13162](https://arxiv.org/abs/2507.13162) | Orbis: Overcoming Challenges of Long-Horizon Prediction in Driving World Models | 2025-07-17 | Orbis连续世界模型使用FM；Orbis-MG主干是MaskGIT，仅后处理refiner含FM，必须区分版本。 | https://arxiv.org/html/2507.13162 ; 4 Compatible Discrete and Continuous Prediction Models > 4.2 Latent Space World Model > 4.2.1 Flow Matching ; /tmp/fm_search_online/papers/2507.13162.txt:668 |
| [2510.18313](https://arxiv.org/abs/2510.18313) | OmniNWM: Omniscient Driving Navigation World Models | 2025-10-21 | 全景视频生成以RF速度场学习 | Training objectives / PDiT Backbone |
| [2602.02002](https://arxiv.org/abs/2602.02002) | UniDriveDreamer: A Single-Stage Multimodal World Model for Autonomous Driving | 2026-02-02 | camera与LiDAR两类future latent联合flow matching，双向跨模态交互 | https://arxiv.org/html/2602.02002v1#S3.SS2.p4.1 |
| [2603.17652](https://arxiv.org/abs/2603.17652) | VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs | 2026-03-18 | interval-conditioned MeanFlow + JVP单步masked scene outpainting，FM家族扩展 | https://arxiv.org/html/2603.17652v1#S3.SS4.SSS0.Px7.p1.1 |
| [2603.19979](https://arxiv.org/abs/2603.19979) | X-World: Controllable Ego-Centric Multi-Camera World Models for Scalable End-to-End Driving | 2026-03-20 | RF训练future multi-camera video，动作/相机/可选布局/文本条件；最终few-step模型还使用DMD。 | https://arxiv.org/html/2603.19979 ; 4 Training > 4.1 Stage-I: Bidirectional I2V Training for Accurate Controllability ; /tmp/fm_search_online/papers/2603.19979.txt:608 |
| [2603.28887](https://arxiv.org/abs/2603.28887) | OccSim: Multi-kilometer Simulation with Long-horizon Occupancy World Models | 2026-03-30 | 主W-DiT为线性噪声插值+速度回归FM；layout也为flow生成；agent motion另由2D-IDM控制 | https://arxiv.org/html/2603.28887v2#S3.SS1.p6.1 |
| [2603.28963](https://arxiv.org/abs/2603.28963) | AutoWorld: Learning Multi-Agent Traffic Simulation with Self-Supervised World Models | 2026-03-30 | 世界模型的未来occupancy预测使用rectified flow；后级轨迹生成用diffusion，勿混为全部FM | https://arxiv.org/html/2603.28963v2#S3.SS2.p1.1 |
| [2605.11596](https://arxiv.org/abs/2605.11596) | HorizonDrive: Self-Corrective Autoregressive World Model for Long-horizon Driving Simulation | 2026-05-12 | 世界模型基础训练明确future chunks FM；后续少步自回归蒸馏不能等同于MeanFlow。 | https://arxiv.org/html/2605.11596 ; 4 Methods > 4.1 Conditional driving world model training ; /tmp/fm_search_online/papers/2605.11596.txt:662 |
| [2605.18137](https://arxiv.org/abs/2605.18137) | Xiaomi Auto World Model: A Joint World Model Integrating Reconstruction and Generation for Autonomous Driving | 2026-05-18 | RF用于WorldGen生成分支，不是WorldRec 3DGS重建；§3.2.1有explicit linear interpolation和FM目标。 | https://arxiv.org/html/2605.18137 ; 3 Method > 3.2 World Generation > 3.2.1 Bidirectional Model Training ; /tmp/fm_search_online/papers/2605.18137.txt:714 |
| [2606.15341](https://arxiv.org/abs/2606.15341) | CausalDrive: Real-time Causal World Models for Autonomous Driving | 2026-06-13 | 动作与文本条件；不输入未来NPC布局。教师FM与最终DMD蒸馏目标需区分。 | https://arxiv.org/html/2606.15341 ; 5 Experiments > 5.1 Experimental Setup ; /tmp/fm_search_online/papers/2606.15341.txt:1311 |
| [2607.14005](https://arxiv.org/abs/2607.14005) | M$^\text{4}$World: A Multi-view Multimodal Driving World Model for Interactive Object Manipulation and Minute-long Streaming | 2026-07-15 | mid-training与teacher forcing均使用rectified-flow；另有后续流式训练 | https://arxiv.org/html/2607.14005v1#S4.SS1.p2.1 |
| [2607.15898](https://arxiv.org/abs/2607.15898) | Orbis 2: A Hierarchical World Model for Driving | 2026-07-17 | §3.2高层预测latent subgoal，低层生成中间细节帧；动作仅条件高层。 | https://arxiv.org/html/2607.15898 ; 3 Method > 3.1 Preliminary: Flow Matching ; /tmp/fm_search_online/papers/2607.15898.txt:330 |
| [2609.21712](https://arxiv.org/abs/2609.21712) | ZYT-World: A Real-Time Controllable World Model for Closed-Loop Autonomous-Driving Simulation | 2026-09-18 | FM teacher及fake-score明确；最终one-step蒸馏不等于MeanFlow；相机/ego/layout可独立控制。 | https://arxiv.org/html/2609.21712 ; 4 From bidirectional to autoregressive > 4.1 Causalization and TF ; /tmp/fm_search_online/papers/2609.21712.txt:1688 |
| [2610.07031](https://arxiv.org/abs/2610.07031) | Artemis: Geometry-Grounded Multi-Agent Driving World Models with Shared 3D State and Progressive Memory Update | 2026-10-04 | 视频训练使用FM；后续以DMD做少步蒸馏 | §5.1 Implementation |
| [2610.07381](https://arxiv.org/abs/2610.07381) | GeoWM: Efficient Direct World Modeling in Explicit Geometry | 2026-10-05 | 指定future horizon的depth直接生成，无需recursive rollout；clean-target prediction + velocity-space weighted L1 | https://arxiv.org/html/2610.07381v1#S3.SS3.p2.1 |

## 严格核心：世界状态与自车动作联合 FM

| arXiv | 论文 | 首发日期 | 生成对象 / FM 用法 | 证据定位 |
|---|---|---:|---|---|
| [2506.24113](https://arxiv.org/abs/2506.24113) | Epona: Autoregressive Diffusion World Model for Autonomous Driving | 2025-06-30 | next-frame latent与trajectory均使用RF目标 | §3；§4 实现 |
| [2601.04453](https://arxiv.org/abs/2601.04453) | UniDrive-WM: Unified Understanding, Planning and Generation World Model for Autonomous Driving | 2026-01-07 | 仅连续AR+diffusion分支用FM预测视觉latent；需与离散AR方案区分 | §3，式9–11 |
| [2602.06521](https://arxiv.org/abs/2602.06521) | DriveWorld-VLA: Unified Latent-Space World Modeling with Vision-Language-Action for Autonomous Driving | 2026-02-06 | FM用于第二阶段action-conditioned未来BEV，不是RGB视频模型；可转交BEV/occupancy分组。 | https://arxiv.org/html/2602.06521 ; 3 Method > 3.2 Action Controllability Fine-Tuning ; /tmp/fm_search_online/papers/2602.06521.txt:746 |
| [2603.19675](https://arxiv.org/abs/2603.19675) | DynFlowDrive: Flow-Based Dynamic World Modeling for Autonomous Driving | 2026-03-20 | 从当前latent的noised anchor到未来latent做FM，利用flow方向稳定性选择轨迹 | https://arxiv.org/html/2603.19675v2#S3.SS2.p8.1 |
| [2604.01765](https://arxiv.org/abs/2604.01765) | DriveDreamer-Policy: A Geometry-Grounded World-Action Model for Unified Generation and Planning | 2026-04-02 | §3.1规定continuous-target专家CFM；§3.2.2明确depth generator standard FM。视频段说Wan DiT未单列另一RF公式，证据强度应如实说明。 | https://arxiv.org/html/2604.01765 ; 3 Methodology > 3.1 Preliminaries ; /tmp/fm_search_online/papers/2604.01765.txt:286 |
| [2604.04198](https://arxiv.org/abs/2604.04198) | DriveVA: Video Action Models are Zero-Shot Drivers | 2026-04-05 | §4.3定义生成target包含视频和动作，§4.4以同一线性插值/速度回归目标训练。 | https://arxiv.org/html/2604.04198 ; 4 Method > 4.4 Flow Matching Objective ; /tmp/fm_search_online/papers/2604.04198.txt:1542 |
| [2605.10426](https://arxiv.org/abs/2605.10426) | CoWorld-VLA: Thinking in a Multi-Expert World Model for Autonomous Driving | 2026-05-11 | 未来视频片段FM监督塑造动态tokens；轨迹另有回归头 | §3，式4–6、11 |
| [2605.14696](https://arxiv.org/abs/2605.14696) | EponaV2: Driving World Model with Comprehensive Future Reasoning | 2026-05-14 | §3.1同时明确RF Future Frame Predictor和Planner，FM不只用于GRPO规划器。 | https://arxiv.org/html/2605.14696 ; 3 Methods > 3.1 Applying World Model to Driving with Flow Matching ; /tmp/fm_search_online/papers/2605.14696.txt:901 |
| [2605.28544](https://arxiv.org/abs/2605.28544) | DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving | 2026-05-27 | 视频和动作联合FM；自回归是时间分解，和FM不互斥。 | https://arxiv.org/html/2605.28544 ; 3 Method > 3.1 Autoregressive Video-Action Generation > Training objective. ; /tmp/fm_search_online/papers/2605.28544.txt:698 |
| [2606.15869](https://arxiv.org/abs/2606.15869) | Metis: A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation | 2026-06-14 | 同时研究自动驾驶与城市导航；§4明确两个expert均FM。 | https://arxiv.org/html/2606.15869 ; 3 Method > 3.4 Training Objective ; /tmp/fm_search_online/papers/2606.15869.txt:699 |
| [2606.16274](https://arxiv.org/abs/2606.16274) | GraphWorld: Long-Horizon Planning with World Models for End-to-End Autonomous Driving | 2026-06-15 | conditional FM refine latent world state并向motion/ego planning queries反馈；2-step生成 | https://arxiv.org/html/2606.16274v1#S4.SS2.p2.1 |
| [2606.28758](https://arxiv.org/abs/2606.28758) | X-Mind: Efficient Visual Chain-of-Thought via Predictive World Model for End-to-End Driving | 2026-06-27 | Layer Flow Matching把未来latent的速度预测与Euler更新内化在LLM深度中 | https://arxiv.org/html/2606.28758v1#S2.SS3.p1.1 |
| [2607.05133](https://arxiv.org/abs/2607.05133) | UNIVERSE: Unified Video Action Models for Autonomous Driving with Flexible Mask-Modulated Modality Generation | 2026-07-06 | 不是只对动作训练FM；视频与动作共同更新共享generative backbone。 | https://arxiv.org/html/2607.05133 ; 3 Method > 3.4 Training Objective |
| [2608.03084](https://arxiv.org/abs/2608.03084) | SUV: Future Scene Understanding as Video Generation for End-to-End Driving | 2026-08-04 | Method/Training and Inference/Multi-stream flow matching明确所有visual streams和action的线性插值与速度loss。 | https://arxiv.org/html/2608.03084 ; Method > Future Scene Understanding as Video Generation > Shared latent video space. ; /tmp/fm_search_online/papers/2608.03084.txt:594 |
| [2608.07468](https://arxiv.org/abs/2608.07468) | SimWAM: A Simple World Action Model for End-to-End Autonomous Driving | 2026-08-07 | 原文同段明确future frames are noised and reconstructed with flow matching；推理可去除未来视频分支。 | https://arxiv.org/html/2608.07468 ; 4 Method > 4.1 Model Architecture ; /tmp/fm_search_online/papers/2608.07468.txt:1403 |
| [2608.10107](https://arxiv.org/abs/2608.10107) | 4D-WAM: 4D Consistent World Modeling for Autonomous Driving | 2026-08-10 | 全网joint video-action flow matching；第二阶段加入4D一致性 | https://arxiv.org/html/2608.10107v1#Sx4.SSx2.p1.1 |
| [2608.12854](https://arxiv.org/abs/2608.12854) | BrainWAM: Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonomous Driving | 2026-08-13 | 视频和动作使用独立RF timestep；不是只有动作FM。 | https://arxiv.org/html/2608.12854 ; 3. Method > 3.1. WAM Branch ; /tmp/fm_search_online/papers/2608.12854.txt:371 |
| [2608.20974](https://arxiv.org/abs/2608.20974) | WA-JEPA: Rethinking the Video JEPA Paradigm for World-Action Modeling in Autonomous Driving | 2026-08-21 | 原文Method部分无数字节号；Stage 1显式linear interpolation+CFM，Stage 2联合未来场景与动作。 | https://arxiv.org/html/2608.20974 ; Method > Stage 1: Hybrid Future-Masked Pre-training > flow-based latent future prediction. ; /tmp/fm_search_online/papers/2608.20974.txt:729 |
| [2609.03602](https://arxiv.org/abs/2609.03602) | SV-WAM: An Efficient Surround-View World-Action Model for End-to-End Autonomous Driving | 2026-09-03 | 原文紧接on both future actions and future video latents；训练期视频监督，推理两步action-only。 | https://arxiv.org/html/2609.03602 ; Methodology > Surround-View World-Action Model > SV-WAM Training. ; /tmp/fm_search_online/papers/2609.03602.txt:722 |
| [2609.20377](https://arxiv.org/abs/2609.20377) | MM-Future: Multi-Mode Joint World-Action Modeling for Autonomous Driving | 2026-09-17 | 明确符合CFM：对scene/action独立流时间做线性source→target插值，clean endpoint预测转为速度，Best-of-Many联合速度回归，Euler积分；属于世界状态与动作共同FM核心，虽正文几乎未写flow matching术语。 | https://arxiv.org/html/2609.20377 ; Method > Multi-Mode Joint World–Action Flow > Joint conditional flow. ; /tmp/fm_search_online/papers/2609.20377.txt:956 |
| [2609.37970](https://arxiv.org/abs/2609.37970) | PhysWAM: Physically Consistent World Action Model for Autonomous Driving | 2026-09-29 | FM目标作用于RGB/depth VAE latent和motion pose coordinates，不是仅动作。 | https://arxiv.org/html/2609.37970 ; 3 Method > 3.1 Problem Formulation and Preliminaries ; /tmp/fm_search_online/papers/2609.37970.txt:672 |
| [2610.11060](https://arxiv.org/abs/2610.11060) | AffordDrive3D: Affordance-Aware World-Action Modeling with Spatial Understanding | 2026-10-08 | 未来RGB world branch与action branch分别采用FM；geometry用RGB-guided depth | https://arxiv.org/html/2610.11060v2#S3.SS2.SSS0.Px1.p2.3 |
| [2610.13781](https://arxiv.org/abs/2610.13781) | SimVTA: Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models | 2026-10-09 | §3.3视频FM损失用于模拟器给出的未来视频target；§4.1更新VideoDiT和shared encoder；推理只输出轨迹。 | https://arxiv.org/html/2610.13781 ; 3 Method > 3.3 Video Supervision > Video Loss. ; /tmp/fm_search_online/papers/2610.13781.txt:629 |
| [2610.17688](https://arxiv.org/abs/2610.17688) | OccWAM: Occupancy World-Action Models with Joint Scene-to-Ego Flow Matching | 2026-10-13 | 两阶段scene-centric到ego-centric的联合CFM，世界与动作每层交互 | https://arxiv.org/html/2610.17688v1#S1.p5.1 |

## 邻近：FM 只负责轨迹、动作或规划

| arXiv | 论文 | 首发日期 | 生成对象 / FM 用法 | 证据定位 |
|---|---|---:|---|---|
| [2503.05689](https://arxiv.org/abs/2503.05689) | GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving | 2025-03-07 | ego multimodal trajectory; FM trajectory decoder | arXiv abstract / method where available |
| [2506.08541](https://arxiv.org/abs/2506.08541) | TrajFlow: Multi-modal Motion Prediction via Flow Matching | 2025-06-10 | multimodal motion prediction | arXiv abstract / method where available |
| [2509.21961](https://arxiv.org/abs/2509.21961) | FlowDrive: moderated flow matching with data balancing for trajectory planning | 2025-09-26 | conditional rectified flow trajectory planner | arXiv abstract / method where available |
| [2510.11083](https://arxiv.org/abs/2510.11083) | Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling | 2025-10-13 | Flow Planner; interactive multimodal trajectory generation | arXiv abstract / method where available |
| [2511.18729](https://arxiv.org/abs/2511.18729) | GuideFlow: Constraint-Guided Flow Matching for Planning in End-to-End Autonomous Driving | 2025-11-24 | constrained FM + EBM for trajectory generation | arXiv abstract / method where available |
| [2512.06112](https://arxiv.org/abs/2512.06112) | WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving | 2025-12-05 | discrete FM over ego trajectory tokens | arXiv abstract / method where available |
| [2602.10285](https://arxiv.org/abs/2602.10285) | Adaptive Time Step Flow Matching for Autonomous Driving Motion Planning | 2026-02-10 | joint surrounding-agent motion + ego planning CFM | arXiv abstract / method where available |
| [2602.10476](https://arxiv.org/abs/2602.10476) | Driving Reaction Trajectories via Latent Flow Matching | 2026-02-11 | latent FM for reaction trajectories | arXiv abstract / method where available |
| [2602.20060](https://arxiv.org/abs/2602.20060) | MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving | 2026-02-23 | MeanFlow one-step trajectory generation | arXiv abstract / method where available |
| [2603.02613](https://arxiv.org/abs/2603.02613) | Real-Time Generative Policy via Langevin-Guided Flow Matching for Autonomous Driving | 2026-03-03 | FM policy / trajectory generation | arXiv abstract / method where available |
| [2605.14832](https://arxiv.org/abs/2605.14832) | Learning Direct Control Policies with Flow Matching for Autonomous Driving | 2026-05-14 | direct acceleration/curvature control sequence FM | arXiv abstract / method where available |
| [2606.00519](https://arxiv.org/abs/2606.00519) | DriveAnchor: Progressive Anchor-based Flow Learning for Autonomous Driving Planning | 2026-05-30 | flow-based trajectory planner | arXiv abstract / method where available |
| [2606.24231](https://arxiv.org/abs/2606.24231) | FlowR2A: Learning Reward-to-Action Distribution for Multimodal Driving Planning | 2026-06-23 | reward-conditioned action distribution FM | arXiv abstract / method where available |
| [2608.09098](https://arxiv.org/abs/2608.09098) | UnsDrive: Towards Robust End-to-End Autonomous Driving in Unstructured Scenes | 2026-08-10 | FM planner; occupancy is conditioning | arXiv abstract / method where available |
| [2608.10976](https://arxiv.org/abs/2608.10976) | XCoT-VLA: Executable Chain-of-Thought for Vision-Language-Action Policies | 2026-08-11 | FM action/trajectory branch | arXiv abstract / method where available |

## 邻近：交通/安全场景生成或多智能体模拟

| arXiv | 论文 | 首发日期 | 生成对象 / FM 用法 | 证据定位 |
|---|---|---:|---|---|
| [2506.08902](https://arxiv.org/abs/2506.08902) | Intention-Conditioned Flow Occupancy Models | 2025-06-10 | conditional flow occupancy distribution; not a full driving WM | arXiv abstract / method where available |
| [2605.04366](https://arxiv.org/abs/2605.04366) | Conditional Flow-VAE for Safety-Critical Traffic Scenario Generation | 2026-05-06 | conditional latent FM transforms nominal to safety-critical rollouts | arXiv abstract / method where available |
| [2607.04451](https://arxiv.org/abs/2607.04451) | CCFM: Collision-Constrained Flow Matching for Safety-Critical Scenario Generation | 2026-07-05 | collision-constrained scenario FM | arXiv abstract / method where available |
| [2607.06957](https://arxiv.org/abs/2607.06957) | Flow-ERD: Agent-type Aware Flow Matching with Entropy-Regularized Distillation for Diverse Traffic Simulation | 2026-07-08 | multi-agent traffic rollout AFM | arXiv abstract / method where available |

## 扩展：基础模型、蒸馏、采样引导或外部状态渲染

| arXiv | 论文 | 首发日期 | 生成对象 / FM 用法 | 证据定位 |
|---|---|---:|---|---|
| [2412.03758](https://arxiv.org/abs/2412.03758) | Auto-Regressive Models Need Structural Registers: Semantic Foresight Improves Coherent Driving Video Continuation | 2024-12-04 | FM用于super-resolution后处理；世界时序主干是AR | Super-resolution module |
| [2603.24587](https://arxiv.org/abs/2603.24587) | DreamerAD: Efficient Reinforcement Learning via Latent World Model for Autonomous Driving | 2026-03-25 | RF世界模型的少步蒸馏及想象策略训练 | §3 Shortcut Forcing |
| [2606.02800](https://arxiv.org/abs/2606.02800) | Cosmos 3: Omnimodal World Models for Physical AI | 2026-06-01 | 通用Physical AI基础模型，不能单列为自动驾驶专用贡献；language understanding并非因此全部变为FM。 | https://arxiv.org/html/2606.02800 ; 4 Training > 4.2 Generator Training > Training objective. ; /tmp/fm_search_online/papers/2606.02800.txt:2942 |
| [2606.03159](https://arxiv.org/abs/2606.03159) | NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation | 2026-06-02 | §3.1静态地图和动态agent轨迹预先得到；§4.1/4.2自身RF训练。闭环系统中的神经renderer，不宜称全动态自主预测。 | https://arxiv.org/html/2606.03159 ; 4 Training > 4.1 World-Scenario Control and Multi-view adaptation > Preamble |
| [2606.11155](https://arxiv.org/abs/2606.11155) | Mean Flow Distillation: Robust and Stable Distillation for Flow Matching Models | 2026-06-09 | Mean Flow Distillation用平均速度场匹配，把OccFM多步预测蒸馏为单步 | https://arxiv.org/html/2606.11155v1#S5.SS1.p2.1 |
| [2607.10781](https://arxiv.org/abs/2607.10781) | Is Energy Guidance All You Need? Training-Free Norm Injection for Driving World Models | 2026-07-12 | FM世界模型上的推理扩展；作者明确引导后的轨迹尚不能有效改变生成视频。 | https://arxiv.org/html/2607.10781 ; III Method > III-A Background: rectified-flow sampling ; /tmp/fm_search_online/papers/2607.10781.txt:258 |
| [2609.22762](https://arxiv.org/abs/2609.22762) | DriveReferee: Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models | 2026-09-19 | 保留视频和动作FM损失，叠加几何偏好蒸馏；属WAM后训练 | §III-C |
| [2609.28931](https://arxiv.org/abs/2609.28931) | HelloWorld: Towards Practical Applications of Generative Driving World Models | 2026-09-24 | §4.1明确ego trajectories/scene configurations为prescribed conditions，模型不推断low-level action或传播explicit dynamics；列观测生成器。 | https://arxiv.org/html/2609.28931 ; 4 Method > 4.2 Generative Backbone and Factorized Conditioning > Latent rectified-flow backbone. ; /tmp/fm_search_online/papers/2609.28931.txt:1016 |
| [2610.00859](https://arxiv.org/abs/2610.00859) | CtrlWAM: Controllable World Action Models with Aligned Intent and Foresight | 2026-10-01 | B.2.3监督初始化为standard FM，A.2保留recorded video-action FM anchors；自身新paired-state corrective targets偏离标准插值路径，不能把整个新目标简单称标准CFM。 | https://arxiv.org/html/2610.00859 ; Appendix A Aligned denoising and optimization > A.2 Loss weighting and recorded-data anchors ; /tmp/fm_search_online/papers/2610.00859.txt:1689 |

## 邻近：静态 LiDAR/occupancy/场景合成或动作侧 FM

| arXiv | 论文 | 首发日期 | 生成对象 / FM 用法 | 证据定位 |
|---|---|---:|---|---|
| [2412.02241](https://arxiv.org/abs/2412.02241) | Fast LiDAR Data Generation with Rectified Flows | 2024-12-03 | CFM/RF、reflow和timestep distillation用于无条件LiDAR样本生成 | https://arxiv.org/html/2412.02241v2#S3.SS1.p1.2 |
| [2507.02250](https://arxiv.org/abs/2507.02250) | FMOcc: TPV-Driven Flow Matching for 3D Occupancy Prediction with Selective State Space Model | 2025-07-03 | FM恢复few-frame输入缺失/遮挡特征，目标不是未来场景演化 | https://arxiv.org/html/2507.02250v1#S3.SS3.p1.1 |
| [2601.07692](https://arxiv.org/abs/2601.07692) | R3DPA: Leveraging 3D Representation Alignment and RGB Pretrained Priors for LiDAR Scene Generation | 2026-01-12 | latent FM生成单帧LiDAR场景并支持inpainting/mixing，非时间世界模型 | https://arxiv.org/html/2601.07692v2#S3.SS1.p5.1 |
| [2602.02232](https://arxiv.org/abs/2602.02232) | LiFlow: Flow Matching for 3D LiDAR Scene Completion | 2026-02-02 | 从单扫描构造的noisy points向完整点云流匹配，非未来预测 | https://arxiv.org/html/2602.02232v1#S4.SS2.p1.2 |
| [2603.24506](https://arxiv.org/abs/2603.24506) | Toward Physically Consistent Driving Video World Models under Challenging Trajectories | 2026-03-25 | §3.1输入所有agent未来轨迹和地图，偏物理一致的条件视频合成；自身视频generator确实RF。 | https://arxiv.org/html/2603.24506 ; 3 Method > 3.4 Physics-Enhanced Multi-view Video Generator ; /tmp/fm_search_online/papers/2603.24506.txt:945 |
| [2606.17536](https://arxiv.org/abs/2606.17536) | OmniDrive: An LLM-Choreographed Multi-Agent World Model with Unified Latent Co-Compression for Multi-View Driving Video Generation | 2026-06-16 | §3.5明确CFM加regularizers的训练目标；偏LLM编排的可控视频合成。注意与同名早期OmniDrive区分。 | https://arxiv.org/html/2606.17536 ; 3 Methodology > 3.1 Preliminaries: A Smooth Manifold for Choreography ; /tmp/fm_search_online/papers/2606.17536.txt:633 |
| [2607.20175](https://arxiv.org/abs/2607.20175) | PerceptDrive: Perception Prior World-Action Modeling with Adaptive Expert Routing for End-to-End Autonomous Driving | 2026-07-22 | 严格边界：future latent以L1回归训练；FM仅actor，不能计作FM世界状态生成主干。 | https://arxiv.org/html/2607.20175 ; 3 Method > 3.3 Prior-Preserving World-Action Planning with Adaptive Expert Routing ; /tmp/fm_search_online/papers/2607.20175.txt:218 |
| [2608.12203](https://arxiv.org/abs/2608.12203) | GeoFlow: Efficient Driving Video Generation via Geometry-Aligned Priors | 2026-08-12 | 条件含未来动态agent的3D boxes、逐帧相机pose和地图；列为可控驾驶视频合成，不能直接称反事实交互动态模型。 | https://arxiv.org/html/2608.12203 ; 3 Method > 3.1 Problem Formulation ; /tmp/fm_search_online/papers/2608.12203.txt:537 |
| [2608.26737](https://arxiv.org/abs/2608.26737) | Generative Semantic Scene Completion | 2026-08-27 | S²D²通过FM一步修正现有SSC输出，非未来世界状态 | https://arxiv.org/html/2608.26737v1#S3.SS4.p1.1 |
| [2609.33737](https://arxiv.org/abs/2609.33737) | MomWorld: Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving | 2026-09-27 | 新FM模块只把世界模型预测用于轨迹refinement；未据此证明世界状态rollout由FM生成。 | https://arxiv.org/html/2609.33737 ; 3 Method > 3.1 Overview of MomWorld ; /tmp/fm_search_online/papers/2609.33737.txt:376 |
| [2609.39245](https://arxiv.org/abs/2609.39245) | ReWAM: Reciprocal World Action Models for Interactive Autonomous Driving | 2026-09-30 | 自身新增CFM训练针对多agent动作；§3.1明确Video DiT冻结，不能把新增目标说成视频FM训练。 | https://arxiv.org/html/2609.39245 ; 2 Methodology > 2.5 Joint Learning and Inference |
| [2610.02726](https://arxiv.org/abs/2610.02726) | SymRegFlow: Symmetry-Regularized Flow Matching for Video World Models | 2026-10-02 | §2.2/2.5定义自身latent FM fine-tuning；核心任务是novel-view条件合成，不足证明动作交互动态模型。 | https://arxiv.org/html/2610.02726 ; 3 Experiments ; /tmp/fm_search_online/papers/2610.02726.txt:1905 |

## 邻近：驾驶视频/场景合成，缺少自主未来动力学闭环

| arXiv | 论文 | 首发日期 | 生成对象 / FM 用法 | 证据定位 |
|---|---|---:|---|---|
| [2411.13807](https://arxiv.org/abs/2411.13807) | MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control | 2024-11-21 | 以轨迹、框、地图等条件生成视频；CFM/RF训练，不等同于独立学习NPC动力学 | §3 式1–2；§4.1 |
| [2504.19614](https://arxiv.org/abs/2504.19614) | DiVE: Efficient Multi-View Driving Scenes Generation Based on Video Diffusion Transformer | 2025-04-28 | 采用RF生成驾驶视频，强调合成和加速 | §4 Implementation |
| [2506.07497](https://arxiv.org/abs/2506.07497) | Genesis: Multimodal Driving Scene Generation with Spatio-Temporal and Cross-Modal Consistency | 2025-06-09 | LiDAR生成分支采用rectified-flow schedule；不把全流水线都称FM | §3 LiDAR Generation |
| [2509.20251](https://arxiv.org/abs/2509.20251) | 4D Driving Scene Generation With Stereo Forcing | 2025-09-24 | multi-view video用FM损失；属于几何引导场景合成 | Implementation / Training pipeline |
| [2512.23180](https://arxiv.org/abs/2512.23180) | GaussianDWM: 3D Gaussian Driving World Model for Unified Scene Understanding and Multi-Modal Generation | 2025-12-29 | 视频生成训练阶段采用RF；属于理解/重建/生成混合框架 | §4.2 |
| [2606.01590](https://arxiv.org/abs/2606.01590) | Effective Multi-sensor Conditioning for Street-view Novel-view Synthesis | 2026-06-01 | FM生成目标视角视频，主要是multi-sensor条件渲染 | §3 式1 |
| [2608.02953](https://arxiv.org/abs/2608.02953) | RealWeather: Realistic and Scene-Faithful Weather Translation with Driving World Models | 2026-08-03 | RF/FM做天气转换；非动作条件的未来动力学预测 | Method / Model Architecture |

## 待复核：论文没有明确写出完整 FM 目标

| arXiv | 论文 | 首发日期 | 生成对象 / FM 用法 | 证据定位 |
|---|---|---:|---|---|
| [2509.16500](https://arxiv.org/abs/2509.16500) | RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation | 2025-09-20 | 文中提及FM采样过程；未由所读方法段锁定所有基模型训练目标，暂不纳入严格核心 | §3.3 |
| [2510.19195](https://arxiv.org/abs/2510.19195) | Rethinking Driving World Model as Synthetic Data Generator for Perception Tasks | 2025-10-22 | 正文称采用RF，但紧邻公式写epsilon/noise目标；训练参数化需以代码进一步核查 | Methods / generation objective |
| [2608.06008](https://arxiv.org/abs/2608.06008) | Adaptive-WAM: Quality-Guided Early-Exit Planning from Intermediate Video-Diffusion Features | 2026-08-06 | 自身论文明确继承native Wan video objective但未写FM/RF或对应插值；不能仅凭Wan名称确认为FM。可追加原始Wan目标/本方法代码核验后升级；目前不计FM核心。 | https://arxiv.org/html/2608.06008 ; 4 Method > Training > Video-domain adaptation. ; /tmp/fm_search_online/papers/2608.06008.txt:1197 |

## 排除：正文是 diffusion/AR、普通位移 flow/warp，或 FM 只出现在相关工作/动作头

| arXiv | 论文 | 首发日期 | 生成对象 / FM 用法 | 证据定位 |
|---|---|---:|---|---|
| [2501.03575](https://arxiv.org/abs/2501.03575) | Cosmos World Foundation Model Platform for Physical AI | 2025-01-07 | 该报告明确采用diffusion score matching，不能因Cosmos后续版本使用RF而回填到本篇 | Diffusion model formulation |
| [2603.24581](https://arxiv.org/abs/2603.24581) | Latent-WAM: Latent World Action Modeling for End-to-End Autonomous Driving | 2026-03-25 | §3.3未来world state直接自回归预测，§3.5 MSE直接拟合未来状态；velocity是ego物理速度监督而非概率transport速度。无source-noise插值/flow目标，不属FM；DLWM推理时移除。 | https://arxiv.org/html/2603.24581 ; 3. Method > 3.5. Training and Inference ; /tmp/fm_search_online/papers/2603.24581.txt:1595 |
| [2605.07326](https://arxiv.org/abs/2605.07326) | GEM: Generating LiDAR World Model via Deformable Mamba | 2026-05-08 | 平方根α加噪与epsilon预测损失；DDPM类型latent diffusion，不是RF/CFM | https://arxiv.org/html/2605.07326v2#S3.SS3.p8.1 |
| [2605.11550](https://arxiv.org/abs/2605.11550) | The DAWN of World-Action Interactive Models | 2026-05-12 | 所核方法未建立世界预测FM证据；§9.2轨迹planner为DPM-Solver++ diffusion。排除于FM核心，不因world-action名称纳入。 | https://arxiv.org/html/2605.11550 ; 2 Methodology > 2.2 DAWN Architecture ; /tmp/fm_search_online/papers/2605.11550.txt:503 |
| [2606.12987](https://arxiv.org/abs/2606.12987) | Diffusion Transformer World-Action Model for AV Scene Prediction | 2026-06-11 | §4.2明确1000-step cosine VP forward process和x0目标，推理DDIM；虽然称DiT/World-Action也不是FM方法。 | https://arxiv.org/html/2606.12987 ; 4 Methods > 4.2 Latent DiT World-Action Model ; /tmp/fm_search_online/papers/2606.12987.txt:801 |
| [2606.13460](https://arxiv.org/abs/2606.13460) | VISA: VLM-Guided Instance Semantic Auditing for 3D Occupancy World Models | 2026-06-11 | taxonomy/attribute/graph audit蒸馏，不是FM世界模型方法 | https://arxiv.org/html/2606.13460v3#S3.p13.1 |
| [2606.30421](https://arxiv.org/abs/2606.30421) | OWMDrive: Causality-Aware End-to-End Autonomous Driving via 4D Occupancy World Model | 2026-06-29 | 世界模型自回归，动作模型ε-prediction diffusion；未采用FM | https://arxiv.org/html/2606.30421v1#S3.SS2.p4.1 |
| [2607.17521](https://arxiv.org/abs/2607.17521) | GeoWorldAD: Geometry World Action Model for Autonomous Driving | 2026-07-20 | depth/trajectory监督回归，无FM生成目标 | https://arxiv.org/html/2607.17521v2#S3.SS2.p2.2 |
| [2608.20430](https://arxiv.org/abs/2608.20430) | RISE: Adaptive Imagination for World Action Models | 2026-08-20 | world predictor自回归latent监督；action planner明确VP diffusion与DPM-Solver++，所核方法无FM世界状态目标。 | https://arxiv.org/html/2608.20430 ; 4 RISE > 4.2 World Action Model Architecture > 4.2.3 Planner ; /tmp/fm_search_online/papers/2608.20430.txt:866 |
| [2608.23486](https://arxiv.org/abs/2608.23486) | GeoWAM: Visual Geometry World Action Models for Autonomous Driving | 2026-08-24 | 未来feature cosine + point regression；trajectory L1，正文无本方法FM | https://arxiv.org/html/2608.23486v3#S3.SS2.SSS0.Px3.p1.1 |
| [2609.18442](https://arxiv.org/abs/2609.18442) | Risk-Aware World Modeling with Flow-Guided Occupancy Evolution for Selective Trajectory Planning in Automated Driving | 2026-09-16 | 此flow是实际场景位移，不是生成分布CFM/RF | https://arxiv.org/html/2609.18442v1#S3.SS5.p1.1 |
| [2609.37098](https://arxiv.org/abs/2609.37098) | V2X-WAM: A Cooperative World Action Model for End-to-End Autonomous Driving | 2026-09-29 | flow为matched-actor空间位移与occupancy warp，非CFM/RF | https://arxiv.org/html/2609.37098v2#S3.SS5.SSS3.p10.1 |
| [2610.11382](https://arxiv.org/abs/2610.11382) | PlanWAM: Planning-Shaped Future Representations for End-to-End Autonomous Driving | 2026-10-08 | §3.4未来latent损失为预测与stop-gradient target的平方Frobenius误差，没有FM插值/速度目标；不属FM世界模型。 | https://arxiv.org/html/2610.11382 ; 3 Method > 3.4 Hindsight-to-Foresight Prediction and Planning ; /tmp/fm_search_online/papers/2610.11382.txt:1286 |
| [2610.14544](https://arxiv.org/abs/2610.14544) | InfraWorld: Motion-Aware World Modeling for Roadside 4D Occupancy Forecasting | 2026-10-10 | dense latent residual更新+语义与point监督，无FM | https://arxiv.org/html/2610.14544v1#S3.SS1.p2.1 |

## 判定边界

- `core/world_action`：FM 直接生成未来观测、latent、occupancy、LiDAR/几何状态，或世界状态与自车动作共同生成。
- `planning/scenario`：FM 生成轨迹、控制、动作、交通参与者 rollout 或安全场景；这些工作对规划和仿真很重要，但不能等同于未来视觉/occupancy 世界模型。
- `adjacent/synthesis`：FM 用于单帧 LiDAR 补全、静态 occupancy、视角/天气/街景合成，或需要外部 NPC 轨迹/布局，世界动力学证据不足。
- `excluded`：正文核验为 DDPM/扩散、AR、普通 displacement/warp/optical flow，或仅在 related work/动作对照头中出现 FM。

## 检索策略与遗漏风险

组合检索词包括 `flow matching`, `rectified flow`, `conditional flow matching`, `mean flow`, `world model`, `world-action`, `driving`, `occupancy`, `LiDAR`, `traffic simulation`, `trajectory planning`；随后逐篇检查摘要和方法段。论文可能只写 `velocity matching`、`flow-based` 或继承底座模型而不在摘要中写 FM，因此仍可能漏掉未公开、非 arXiv 或方法表述不清的工作。反过来，普通 optical flow、空间位移 flow 和 diffusion 论文不会因关键词命中而进入严格核心。

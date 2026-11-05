# DiT 与 Flow Matching 创新文献综述及发表场所

检索日期：**2026-11-02**。面向 ReactOccFlow / ICCV 选题查重与方法设计。本轮整理 **100 篇文献条目**，记录 **144 条实际检索式**；不是 DiT/FM 全领域的穷尽清单，不把有限检索当作“所有论文已经查完”。

- [可筛选 CSV 总表](docs/literature/2026-11-02-dit-flow-matching/papers.csv)
- [结构化论文记录 JSON](docs/literature/2026-11-02-dit-flow-matching/papers.json)
- [检索式与边界记录](docs/literature/2026-11-02-dit-flow-matching/search_strategy.json)
- 关联：[ICCV 修改建议](ReactOccFlow_ICCV修改建议.md)、[Baseline 方案](ReactOccFlow论文Baseline实验方案.md)。本文新增文献证据，不自动修改权威实施计划、模型或训练状态。

## 1. 检索边界与证据等级

这是定向整合性文献检索，覆盖 2022 年起的基础工作及截至检索日发现的 2026 年方法扩展：架构、条件交互、表示、训练目标、路径/先验耦合、少步生成、缓存、离散/流形状态、时序、控制及驾驶世界模型近邻。不是按 PRISMA 完成的数据库系统综述；没有全领域总命中数、召回率或“全球首次”结论。

检索先用 `diffusion transformer / DiT / flow matching / rectified flow` 扩展，再按 `representation alignment / autoencoder / optimal transport / shortcut / mean flow / manifold / discrete / multimodal / asynchronous / physical / world model / counterfactual / rare event` 等交叉查找，最后逐篇核对题名和发表信息。完整查询含来源域名限制，保存在检索记录中。

技术归纳优先使用 PMLR、CVF、ICLR/NeurIPS/ECVA 正式论文集、JMLR/TMLR、OpenReview 最终论文、IEEE 记录以及作者预印本/项目页。搜索引擎结果用于定位；非作者维护的列表或博客不单独作为技术结论依据。语义学术 API 访问未成功，采用规范化题名、arXiv ID/论文 URL 去重。

纳入提出明确方法改变的论文，以及与本项目直接相近的世界—动作、预测和规划工作。排除仅有 optical flow 同名词、无关物理流体词条和大多数只调用 DiT 的应用论文。医学、分子、语言、音频等应用没有逐领域穷尽；保留少数具有通用方法意义的例子。因此“没有出现在表里”不能理解为“不存在”。

| 状态 | 数量 | 含义 |
|---|---:|---|
| 主会 | 76 | 已核对正式论文集、会议目录或最终版论文明确标注 |
| 期刊 | 6 | 已核对期刊记录/最终版论文，含 JMLR、TMLR、IEEE TPAMI |
| 作者标注会议 | 1 | IR-WM 的作者 arXiv 元数据标注 ICRA 2026；证据级别单列 |
| Workshop | 1 | 与会议主会严格区分 |
| 仅核实预印本 | 14 | 本轮没有核实正式发表，不是断言该文从未录用 |
| 场所待核验 | 2 | 有发表线索但未通过出版社核验，不计入已核实期刊 |

**阅读范围：**逐篇核对摘要或作者公开方法概述及发表元数据，重点近邻另结合前期已读内容；没有对 100 篇全部全文精读、公式验证或实验复现。表中“创新点”是原文方法主张的简要归纳；“与项目关系”是我的阅读推论，不是原论文已经验证了 ReactOccFlow。不会仅凭摘要判定论文的隐藏缺陷。

**年份规则：**正式发表条目采用会议/期刊年份；仅预印本条目采用预印本年份。REPA、RAE、Latte 等不能把 arXiv 首发年份直接当作正式发表年份；under-review 不等于已录用。

## 2. 检索后对项目创新的判断

我同意需要加强创新辨识度，但原因不能只归结为模块数量少。当前方案的若干通用做法已有直接先例：

| 拟采用的通用做法 | 已有关键文献 | 对项目主张的约束 |
|---|---|---|
| U-Net 改为 DiT | [DiT](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)，ICCV 2023 | 主干替换本身已有充分先例 |
| VAE latent + RF/FM | [Latent FM](https://arxiv.org/abs/2307.08698)、[OccFM](https://openreview.net/pdf?id=OyX9cC9WaV) | 表示与目标的组合不能直接算新 |
| 模态分支与双向交互 | [SD3 / MMDiT](https://proceedings.mlr.press/v235/esser24a.html)、[OccWAM](https://arxiv.org/abs/2610.17688) | 需明确区别于通用多模态联合生成 |
| 世界—动作双流及异步采样 | [DUST](https://proceedings.mlr.press/v306/won26a.html)，ICML 2026 | 改成双流或不同噪声时间仍需实质增量 |
| 中间表征空间/物理对齐 | [iREPA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3929a7785bd56f57edcff0152ab41289-Abstract-Conference.html)、[REPA-P](https://proceedings.mlr.press/v306/jia26j.html) | 加几何监督或物理辅助头不能自动建立新颖性 |
| 用运动先验替代高斯噪声 | [STFlow](https://proceedings.mlr.press/v306/brinke26a.html)，ICML 2026 | 要证明驾驶联合条件问题中特有的收益和边界 |
| 一步生成或减少 NFE | [Shortcut Models](https://openreview.net/pdf?id=OlzB6LnXcS)、[MeanFlow](https://papers.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html)、[iMF](https://openaccess.thecvf.com/content/CVPR2026/html/Geng_Improved_Mean_Flows_On_the_Challenges_of_Fastforward_Generative_Models_CVPR_2026_paper.html) | 需要超过现有少步训练路线，不能只减少求解步数 |
| 连续运动与离散状态一起生成 | [EF-VFM / TabbyFlow](https://proceedings.mlr.press/v267/guzman-cordero25a.html)、[Purrception](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9d8cf1247786d6dfeefeeb53b8b5f6d7-Abstract-Conference.html)、[Sketch2Colab](https://openaccess.thecvf.com/content/CVPR2026/papers/Daiya_Sketch2Colab_Sketch-Conditioned_Multi-Human_Animation_via_Controllable_Flow_Distillation_CVPR_2026_paper.pdf) | 通用混合生成和离散交互耦合已有先例 |
| 多样性采样用于风险估计 | [Weighted Joint Sampling](https://proceedings.mlr.press/v337/liu26c.html)，UAI 2026 | 改变采样分布后需要权重/偏差分析 |
| 用 FM 做反事实生成 | [Counterfactual FM](https://arxiv.org/abs/2610.01193)、[Debiased Counterfactual FM](https://arxiv.org/abs/2605.07665) | 需要单独说明识别假设及真实干预证据 |

这不等于这些方向已经没有研究空间。它意味着论文需要提出**具体的失败现象、针对该现象的机制，以及强对照下的证据**，而不能把上述通用做法重新命名后作为贡献。

## 3. 按创新方向整理的文献总表

列中“关系”是项目适用性推论；论文的完整发表证据备注见 CSV/JSON。所有预印本和待核验条目都保持显式标签。

### 3.1 FM基础与概率路径（9 篇）

| 编号 | 论文与原文 | 发表场所 / 年份 / 状态 | 原文创新点概述 | 与 ReactOccFlow 的关系（推论） |
|---|---|---|---|---|
| F01 | [**Flow Matching**：Flow Matching for Generative Modeling](https://openreview.net/pdf?id=PqvMRDCJT9t) | ICLR 2023；主会 | 通过条件概率路径的速度场回归，无需训练时模拟连续流 | 你当前目标的理论基础；普通CFM不是新增贡献 |
| F02 | [**Rectified Flow**：Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://openreview.net/pdf?id=XVjTT1nw5z) | ICLR 2023；主会 | 线性插值速度回归与迭代rectification，研究高效分布传输 | 区分当前直线路径训练与尚未执行的reflow/蒸馏 |
| F03 | [**Multisample FM**：Multisample Flow Matching: Straightening Flows with Minibatch Couplings](https://proceedings.mlr.press/v202/pooladian23a.html) | ICML 2023；主会 | 以minibatch中非独立的噪声—数据耦合改善流路径 | 联合O/A重配对须整体保留条件与模态对应 |
| F04 | [**OT-CFM**：Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://openreview.net/pdf?id=HgDwiZrpVq) | TMLR 2024；期刊 | 统一条件FM，采用minibatch OT与熵正则耦合 | 可作为路径/耦合对照；增加OT匹配本身已有先例 |
| F05 | [**Stochastic Interpolants**：Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://jmlr.org/papers/v26/23-1605.html) | JMLR 2025；期刊 | 随机插值统一确定性流与随机扩散，并联系Schrödinger bridge | 为随机性、路径和推理ODE/SDE的分离设计提供基础 |
| F06 | [**SF²M**：Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://proceedings.mlr.press/v238/tong24a.html) | AISTATS 2024；主会 | 将score与flow matching结合以无模拟方式学习随机桥 | 研究有噪声的条件状态转移时可参考 |
| F07 | [**Functional FM**：Functional Flow Matching](https://proceedings.mlr.press/v238/kerrigan24a.html) | AISTATS 2024；主会 | 把FM推广到函数空间的概率测度与向量场 | 连续时空场建模参考，不等于直接替换当前网络 |
| F08 | [**Latent FM**：Flow Matching in Latent Space](https://arxiv.org/abs/2307.08698) | arXiv 2023；仅核实预印本 | 在预训练autoencoder latent中学习FM以降低高维建模成本 | VAE latent加FM已有先例 |
| F09 | [**Stochastic forecasting**：Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes](https://proceedings.mlr.press/v235/chen24n.html) | ICML 2024；主会 | 通过随机插值与Föllmer过程学习高维概率预测 | 预测多未来及随机动态，比图像生成FID更贴近任务 |

### 3.2 DiT结构与条件机制（18 篇）

| 编号 | 论文与原文 | 发表场所 / 年份 / 状态 | 原文创新点概述 | 与 ReactOccFlow 的关系（推论） |
|---|---|---|---|---|
| D01 | [**DiT**：Scalable Diffusion Models with Transformers](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html) | ICCV 2023；主会 | latent patches上的Transformer扩散主干与规模研究 | 当前主干基础；U-Net换成DiT不能单独算创新 |
| D02 | [**U-ViT**：All Are Worth Words: A ViT Backbone for Diffusion Models](https://openaccess.thecvf.com/content/CVPR2023/html/Bao_All_Are_Worth_Words_A_ViT_Backbone_for_Diffusion_Models_CVPR_2023_paper.html) | CVPR 2023；主会 | 时间、条件和图像均token化，浅深层采用长跳连 | 认识Transformer主干与U形拓扑可共存；仅作结构文献 |
| D03 | [**MDT**：Masked Diffusion Transformer is a Strong Image Synthesizer](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_Masked_Diffusion_Transformer_is_a_Strong_Image_Synthesizer_ICCV_2023_paper.html) | ICCV 2023；主会 | masked latent modeling增强语义部分间上下文学习 | 可研究部分观测鲁棒性，但普通masked training已有 |
| D04 | [**DiT-3D**：DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d6c01b025cad37d5c8bab4ba18846c02-Abstract.html) | NeurIPS 2023；主会 | 在体素化点云上执行Transformer扩散生成 | 3D状态采用DiT也已有先例；需突出动态与联合结构 |
| D05 | [**PixArt-α**：PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fe989bb038b5dcc44181255dd6913e43-Abstract-Conference.html) | ICLR 2024；主会 | 高效文图DiT训练与条件注入、分阶段训练设计 | 条件机制与训练效率参考 |
| D06 | [**PixArt-Σ**：PixArt-Sigma: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation](https://eccv2024.ecva.net/virtual/2024/poster/284) | ECCV 2024；主会 | weak-to-strong训练扩展DiT至高分辨率生成 | 分辨率扩展与训练迁移参考，不能照搬图像结论 |
| D07 | [**SD3 / MMDiT**：Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://proceedings.mlr.press/v235/esser24a.html) | ICML 2024；主会 | RF训练时间/噪声采样优化；模态独立权重与双向token交互 | 双模态分支及双向交换已有重要先例 |
| D08 | [**SiT**：SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9828_ECCV_2024_paper.php) | ECCV 2024；主会 | 固定DiT结构比较插值、目标、连续时间及ODE/SDE采样 | 比单纯主干变化更有解释力的FM设计消融 |
| D09 | [**FiT**：FiT: Flexible Vision Transformer for Diffusion Model](https://proceedings.mlr.press/v235/lu24k.html) | ICML 2024；主会 | 支持灵活分辨率和长宽比的视觉扩散Transformer | 可变网格与位置编码参考 |
| D10 | [**DiffiT**：DiffiT: Diffusion Vision Transformers for Image Generation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1231_ECCV_2024_paper.php) | ECCV 2024；主会 | 时间依赖的注意力与扩散Transformer架构 | 区分生成时间条件与物理预测时间条件 |
| D11 | [**U-DiTs**：U-DiTs: Downsample Tokens in U-Shaped Diffusion Transformers](https://proceedings.nips.cc/paper_files/paper/2024/hash/5d2e24df9cfaad3189833b819c40b392-Abstract-Conference.html) | NeurIPS 2024；主会 | U形DiT结合token降采样以改善质量—成本 | 结构比较参考，不表示恢复已删除的U-Net代码 |
| D12 | [**Lumina-T2X**：Lumina-T2X: Scalable Flow-Based Large Diffusion Transformer for Flexible Resolution Generation](https://openreview.net/pdf?id=EbWf36quzd) | ICLR 2025；主会 | 统一多模态序列表示的flow-based DiT与灵活分辨率 | 不能把统一token和RF Transformer泛称首次 |
| D13 | [**Lumina-Next**：Lumina-Next: Making Lumina-T2X Stronger and Faster with Next-DiT](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ed2dad593d87ca474a636cba610a29d3-Abstract-Conference.html) | NeurIPS 2024；主会 | 改进Flag-DiT架构、位置外推及训练/推理效率 | 位置外推与预算可控的时空建模参考 |
| D14 | [**SANA**：SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://openreview.net/pdf?id=N8Oj1XhtYZ) | ICLR 2025；主会 | 深压缩autoencoder与线性DiT实现高效高分辨率生成 | 长token瓶颈参考；不能仅比较理论复杂度 |
| D15 | [**Hunyuan-DiT**：Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding](https://arxiv.org/abs/2405.08748) | arXiv 2024；仅核实预印本 | 多分辨率DiT与细粒度双语文图条件设计 | 多条件融合参考，任务距离较远 |
| D16 | [**DDT**：DDT: Decoupled Diffusion Transformer](https://arxiv.org/abs/2504.05741) | CVPR 2026；主会 | 解耦低频语义编码与高频解码，缓解优化冲突 | 可检验全局场景结构与局部actor细节是否需要分工 |
| D17 | [**PixelDiT**：PixelDiT: Pixel Diffusion Transformers for Image Generation](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_PixelDiT_Pixel_Diffusion_Transformers_for_Image_Generation_CVPR_2026_paper.html) | CVPR 2026；主会 | patch级全局语义与pixel级细节的双层Transformer | 细粒度局部更新思路；不能直接把像素机制当occupancy新意 |
| D18 | [**JiT**：Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720) | CVPR 2026；主会 | 直接预测干净数据以利用数据流形，在原始数据空间运行Transformer | 端点预测与速度预测的参数化值得公平比较 |

### 3.3 表征与训练优化（8 篇）

| 编号 | 论文与原文 | 发表场所 / 年份 / 状态 | 原文创新点概述 | 与 ReactOccFlow 的关系（推论） |
|---|---|---|---|---|
| R01 | [**REPA**：Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://openreview.net/pdf?id=DJSZGGZYVi) | ICLR 2025；主会 | 对齐DiT中间特征与预训练视觉表征以加速生成训练 | 仅加教师特征对齐属于已有路线 |
| R02 | [**DC-AE**：Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models](https://openreview.net/pdf?id=wH8XXUOUZU) | ICLR 2025；主会 | 提高空间压缩比，并处理重建与高分辨率适配 | 减token数需同时量化小目标几何损失 |
| R03 | [**VA-VAE / LightningDiT**：Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2025/html/Yao_Reconstruction_vs._Generation_Taming_Optimization_Dilemma_in_Latent_Diffusion_Models_CVPR_2025_paper.html) | CVPR 2025；主会 | 视觉表征对齐VAE，缓解重建与生成优化矛盾，并给出LightningDiT | 不能只以VAE重建好判定latent适合未来生成 |
| R04 | [**REPA-E**：REPA-E: Unlocking VAE for End-to-End Tuning of Latent Diffusion Transformers](https://openaccess.thecvf.com/content/ICCV2025/html/Leng_REPA-E_Unlocking_VAE_for_End-to-End_Tuning_of_Latent_Diffusion_Transformers_ICCV_2025_paper.html) | ICCV 2025；主会 | 通过表征对齐实现VAE与latent DiT的端到端优化 | 联合微调VAE已有先例，且会改变当前缓存/预训练公平性 |
| R05 | [**RAE**：Diffusion Transformers with Representation Autoencoders](https://openreview.net/pdf?id=0u1LigJaab) | ICLR 2026；主会 | 预训练表征encoder配合训练decoder替代传统VAE表示 | 表示空间选择是研究变量，但RGB表征不能直接替代occupancy VAE |
| R06 | [**iREPA**：What matters for Representation Alignment: Global Information or Spatial Structure?](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3929a7785bd56f57edcff0152ab41289-Abstract-Conference.html) | ICLR 2026；主会 | 研究空间结构对齐，采用卷积投影与空间归一化 | 几何对应若仅做普通空间对齐，需要与此类方法区分 |
| R07 | [**RAEv2**：Improved Baselines with Representation Autoencoders](https://arxiv.org/abs/2605.18324) | arXiv 2026；仅核实预印本 | 多层表征聚合，研究RAE与REPA的互补及guidance简化 | 更换表征时应使用强RAE基线，不能只比较最早版本 |
| R08 | [**Scaling RAE**：Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2601.16208) | arXiv 2026；仅核实预印本 | 研究RAE向大规模文图训练扩展时的数据、噪声与架构取舍 | 小规模有效的设计未必在新域/规模有效 |

### 3.4 少步生成与推理效率（14 篇）

| 编号 | 论文与原文 | 发表场所 / 年份 / 状态 | 原文创新点概述 | 与 ReactOccFlow 的关系（推论） |
|---|---|---|---|---|
| A01 | [**InstaFlow**：InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4dc37a7bc61057252ce043fa3b83aac2-Abstract-Conference.html) | ICLR 2024；主会 | 以rectification/reflow改善噪声—样本映射，再实现单步生成 | 当前普通RF不等于已经完成单步蒸馏 |
| A02 | [**Improved RF training**：Improving the Training of Rectified Flows](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7343a5c976f8399880b695267f1f9e9f-Abstract-Conference.html) | NeurIPS 2024；主会 | U形训练时间分布和LPIPS-Huber等改进单轮RF训练 | 更换t分布或损失权重本身已有先例 |
| A03 | [**Rectified Diffusion**：Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4df9a5e6bad9e64ebcea453e031142bb-Abstract-Conference.html) | ICLR 2025；主会 | 研究重配对/rectification的作用，质疑直线路径是唯一关键 | 不要将路径更直直接等同于更好的任务预测 |
| A04 | [**Shortcut Models**：One Step Diffusion via Shortcut Models](https://openreview.net/pdf?id=OlzB6LnXcS) | ICLR 2025；主会 | 以噪声时间与步长共同条件化并学习跨步自一致映射 | 可变求解预算已有方法，需看联合分布是否保持 |
| A05 | [**MeanFlow**：Mean Flows for One-step Generative Modeling](https://papers.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html) | NeurIPS 2025；主会 | 学习区间平均速度，通过与瞬时速度的恒等关系训练 | 不能把减少Euler步数称作MeanFlow或新单步方法 |
| A06 | [**Flow Map Matching**：Flow map matching with stochastic interpolants: A mathematical framework for consistency models](https://openreview.net/pdf?id=cqDH0e6ak2) | TMLR 2025；期刊 | 直接学习双时间的flow map，统一直接训练与蒸馏目标 | 跨时间映射的理论参考，与语义几何一致性不是同一种一致性 |
| A07 | [**IMM**：Inductive Moment Matching](https://proceedings.mlr.press/v267/zhou25c.html) | ICML 2025；主会 | 以自一致插值及分布矩匹配学习少步生成 | 从分布层面检查少步训练，不只匹配单条轨迹 |
| A08 | [**iMF**：Improved Mean Flows: On the Challenges of Fastforward Generative Models](https://openaccess.thecvf.com/content/CVPR2026/html/Geng_Improved_Mean_Flows_On_the_Challenges_of_Fastforward_Generative_Models_CVPR_2026_paper.html) | CVPR 2026；主会 | 重新组织平均/瞬时速度训练目标并改进快速生成的guidance | MeanFlow本身已有后续强基线 |
| A09 | [**MF-RAE**：MeanFlow Transformers with Representation Autoencoders](https://openaccess.thecvf.com/content/CVPR2026/html/Hu_MeanFlow_Transformers_with_Representation_Autoencoders_CVPR_2026_paper.html) | CVPR 2026；主会 | 在RAE语义latent空间中设计MeanFlow的训练与采样 | RAE加MeanFlow也已有直接先例 |
| A10 | [**IGM**：One-Step Diffusion and Flow Distillation Through Implicit Generator Matching](https://ieeexplore.ieee.org/document/11454572/) | IEEE TPAMI 2026；期刊 | 通过隐式generator matching，将预训练扩散/流模型无训练图像地蒸馏为单步生成器 | 蒸馏加速已有期刊路线，须检验运动多模态与风险概率失真 |
| A11 | [**Consistency-FM**：Consistency Flow Matching: Defining Straight Flows with Velocity Consistency](https://arxiv.org/abs/2407.02398) | arXiv 2024；仅核实预印本 | 约束不同起点时间到相同端点的速度自一致性 | 与当前actor—occupancy端点几何一致性区分命名 |
| A12 | [**ToCa**：Accelerating Diffusion Transformers with Token-wise Feature Caching](https://openreview.net/pdf/211563413bc2e5e90bca8745796e3fb0aa9fcff9.pdf) | ICLR 2025；主会 | 按token选择更新与复用特征，减少DiT重复计算 | 缓存若影响小目标/突发响应需单独评估 |
| A13 | [**TeaCache**：Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](https://www.openaccess.thecvf.com/content/CVPR2025/html/Liu_Timestep_Embedding_Tells_Its_Time_to_Cache_for_Video_Diffusion_CVPR_2025_paper.html) | CVPR 2025；主会 | 用时间嵌入相关变化估计缓存时机，无需额外训练 | 不同候选或噪声样本之间不能随意共享缓存 |
| A14 | [**FasterCache**：FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality](https://proceedings.iclr.cc/paper_files/paper/2025/hash/518046d86bbc41a0707727c38301ad8e-Abstract-Conference.html) | ICLR 2025；主会 | 利用去噪过程特征冗余以免训练方式加速视频生成 | 属于部署优化方向，不自动解释联合建模新意 |

### 3.5 几何、离散与混合状态（10 篇）

| 编号 | 论文与原文 | 发表场所 / 年份 / 状态 | 原文创新点概述 | 与 ReactOccFlow 的关系（推论） |
|---|---|---|---|---|
| G01 | [**Riemannian FM**：Flow Matching on General Geometries](https://openreview.net/pdf?id=g7ohDlTITL) | ICLR 2024；主会 | 在流形上构造目标向量场，推广欧氏FM | 姿态角/刚体状态不必完全按独立欧氏量处理 |
| G02 | [**Discrete FM**：Discrete Flow Matching](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0d629a734b56a642701bba7bc8bb3ed-Abstract-Conference.html) | NeurIPS 2024；主会 | 在离散状态上定义概率路径及采样速度 | 类别/存在性采用离散生成已有理论基础 |
| G03 | [**Fisher-Flow**：Fisher Flow Matching for Generative Modeling over Discrete Data](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fadec8f2e65f181d777507d1df69b92f-Abstract-Conference.html) | NeurIPS 2024；主会 | 利用概率单纯形上的Fisher–Rao几何构造离散数据流 | 语义类别概率的几何建模参考 |
| G04 | [**Statistical FM**：Categorical Flow Matching on Statistical Manifolds](https://proceedings.neurips.cc/paper_files/paper/2024/hash/62a58f2130894e44e8a272c563a2c6f1-Abstract-Conference.html) | NeurIPS 2024；主会 | 在参数化概率测度流形上定义统计FM | 类别状态生成不局限于连续latent加分类损失 |
| G05 | [**VFM / CatFlow**：Variational Flow Matching for Graph Generation](https://papers.nips.cc/paper_files/paper/2024/hash/15b780350b302a1bf9a3bd273f5c15a4-Abstract-Conference.html) | NeurIPS 2024；主会 | 将FM改写为轨迹端点后验推断，并实现类别CatFlow | actor交互图与离散状态参考 |
| G06 | [**EF-VFM / TabbyFlow**：Exponential Family Variational Flow Matching for Tabular Data Generation](https://proceedings.mlr.press/v267/guzman-cordero25a.html) | ICML 2025；主会 | 以指数族VFM统一混合连续和离散特征 | 连续运动与离散存在性的联合建模已有通用先例 |
| G07 | [**Purrception**：Purrception: Variational Flow Matching for Vector-Quantized Image Generation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9d8cf1247786d6dfeefeeb53b8b5f6d7-Abstract-Conference.html) | ICLR 2026；主会 | codebook类别后验监督与连续embedding速度场结合 | 离散语义与连续传输结合也需具体超越已有方法 |
| G08 | [**Unbalanced FM**：Composing Unbalanced Flows for Flexible Docking and Relaxation](https://openreview.net/pdf?id=gHLWTzKiZV) | ICLR 2025；主会 | 放宽FM边缘约束并组合流以改善柔性对接 | 质量不平衡不等同于已经解决actor出生/消失，需要另行建模 |
| G09 | [**SE(n)-invariant FM**：SE(n)-Invariant Flow Matching: A General Framework with Application to Object Reassembly](https://proceedings.mlr.press/v306/heck26a.html) | ICML 2026；主会 | gauge fixing与商空间目标剔除整体刚体自由度 | 减少坐标系任意性，但道路坐标有语义约束，不能直接照搬 |
| G10 | [**γ-FM**：Implicit geometric regularization in flow matching via density weighted Stein operators](https://openreview.net/pdf?id=LBlkVBDRdu) | TMLR 2026；期刊 | 通过密度加权与Stein算子调整速度回归几何 | 几何加权目标已有路线，需区分概率空间与物理空间 |

### 3.6 视频、时序与滚动生成（7 篇）

| 编号 | 论文与原文 | 发表场所 / 年份 / 状态 | 原文创新点概述 | 与 ReactOccFlow 的关系（推论） |
|---|---|---|---|---|
| V01 | [**Latte**：Latte: Latent Diffusion Transformer for Video Generation](https://openreview.net/pdf?id=ntGPYNUF3t) | TMLR 2025；期刊 | 将视频latent token的空间与时间建模分解为高效Transformer变体 | 时空attention分解已有先例 |
| V02 | [**CogVideoX**：CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ce31378e9f41d8907e97dab172b6c559-Abstract-Conference.html) | ICLR 2025；主会 | 3D VAE与expert Transformer改善时空压缩及文图对齐 | 模态专属归一化/专家分工参考 |
| V03 | [**GenTron**：GenTron: Diffusion Transformers for Image and Video Generation](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_GenTron_Diffusion_Transformers_for_Image_and_Video_Generation_CVPR_2024_paper.html) | CVPR 2024；主会 | 研究文图DiT并扩展至视频生成 | 视频扩展与条件机制对照参考 |
| V04 | [**Goku**：Goku: Flow Based Video Generative Foundation Models](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Goku_Flow_Based_Video_Generative_Foundation_Models_CVPR_2025_paper.html) | CVPR 2025；主会 | RF Transformer联合图像/视频训练，配套数据与训练设计 | 多帧RF生成本身已有应用先例 |
| V05 | [**Diffusion Forcing**：Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2aee1c4159e48407d68fe16ae8e6e49e-Abstract-Conference.html) | NeurIPS 2024；主会 | 不同token独立噪声水平，统一因果序列与多token去噪 | 独立生成时间/异步噪声不是空白方向 |
| V06 | [**CausVid**：From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.pdf) | CVPR 2025；主会 | 将双向视频生成教师转化为快速因果自回归生成器 | 长时滚动与少步推理参考，因果attention不等于因果干预识别 |
| V07 | [**Self Forcing**：Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html) | NeurIPS 2025；主会 | 训练时基于自身生成历史滚动，减小训练—推理差异 | 若扩展长时world rollout，需控制历史误差积累 |

### 3.7 引导、控制与有限样本（7 篇）

| 编号 | 论文与原文 | 发表场所 / 年份 / 状态 | 原文创新点概述 | 与 ReactOccFlow 的关系（推论） |
|---|---|---|---|---|
| C01 | [**D-Flow**：D-Flow: Differentiating through Flows for Controlled Generation](https://proceedings.mlr.press/v235/ben-hamu24a.html) | ICML 2024；主会 | 对生成流微分以优化输入噪声，实现目标约束生成 | test-time控制已有先例；必须统计额外反传成本 |
| C02 | [**FM Guidance**：On the Guidance of Flow Matching](https://proceedings.mlr.press/v267/feng25s.html) | ICML 2025；主会 | 建立一般FM能量引导框架及精确/近似引导方法 | 不能直接把扩散score guidance公式当一般FM正确公式 |
| C03 | [**Guided Flows**：Guided Flows for Generative Modeling and Decision Making](https://arxiv.org/abs/2311.13443) | arXiv 2023；仅核实预印本 | 将classifier-free guidance引入FM并用于生成与离线决策 | CFG与FM规划已有先例，guided分布不一定等于原始预测分布 |
| C04 | [**Preference FM**：Preference Alignment with Flow Matching](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3df874367ce2c43891aab1ab23ae6959-Abstract-Conference.html) | NeurIPS 2024；主会 | 从偏好数据学习将较差样本传输至较好样本的flow | 偏好优化参考，不等同于概率校准或交互建模 |
| C05 | [**FlowEdit**：FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](https://openaccess.thecvf.com/content/ICCV2025/html/Kulikov_FlowEdit_Inversion-Free_Text-Based_Editing_Using_Pre-Trained_Flow_Models_ICCV_2025_paper.html) | ICCV 2025；主会 | 构造源—目标分布的编辑ODE，避免显式反演 | 改变动作条件时保持场景一致的编辑思路参考，仍需真实干预标签 |
| C06 | [**Diverse FM**：Letting Trajectories Spread: Quality-Preserving Control for Diverse Flow Matching](https://proceedings.mlr.press/v306/wu26be.html) | ICML 2026；主会 | 通过与质量方向解耦的几何控制与随机扰动促进采样多样性 | 多样性引导已有先例，需同时核验概率偏差 |
| C07 | [**Weighted Joint Sampling**：Score-Regularized Joint Sampling with Importance Weights for Flow Matching](https://proceedings.mlr.press/v337/liu26c.html) | UAI 2026；主会 | 非IID联合采样、score正则与沿流重要性权重，提高有限样本期望估计 | 小M风险估计必须考虑采样偏置和权重，不能只鼓励多样性 |

### 3.8 物理动态与2026扩展（7 篇）

| 编号 | 论文与原文 | 发表场所 / 年份 / 状态 | 原文创新点概述 | 与 ReactOccFlow 的关系（推论） |
|---|---|---|---|---|
| T01 | [**OTP-FM**：Multimarginal flow matching with optimal transport potentials](https://proceedings.mlr.press/v306/kansal26a.html) | ICML 2026；主会 | 以OT势约束多个中间观测边缘，扩展FM训练目标 | 多时刻分布一致不等于每条物理轨迹都正确 |
| T02 | [**STFlow**：STFlow: Data-Coupled Flow Matching for Geometric Trajectory Simulation](https://proceedings.mlr.press/v306/brinke26a.html) | ICML 2026；主会 | 以条件随机游走替代独立高斯先验，结合数据依赖耦合 | 观测锚定的运动先验值得比较，但替换Gaussian已有直接先例 |
| T03 | [**GSE-Flow**：Flow for Future: Geometric SE(3)-Equivariant Flow Matching for 3D Trajectory Prediction](https://proceedings.mlr.press/v306/wu26ak.html) | ICML 2026；主会 | 统一历史/生成流编码，通过等变几何特征融合预测3D轨迹 | 几何等变与历史—生成流融合已有方法 |
| T04 | [**Longitudinal FM**：Longitudinal Flow Matching for Trajectory Modeling](https://proceedings.mlr.press/v300/islam26a.html) | AISTATS 2026；主会 | 多观测时刻的插值FM，联合学习drift与扩散系数 | 不规则时间采样及多时间一致性参考 |
| T05 | [**REPA-P**：Learning to Think in Physics: Breaking Shortcut Learning in Scientific Diffusion via Representation Alignment](https://proceedings.mlr.press/v306/jia26j.html) | ICML 2026；主会 | 中间表征解码物理量并施加第一原理残差，推理去掉辅助头 | 中间层加物理监督也已有直接先例 |
| T06 | [**SAD-Flower**：SAD-Flower: Flow Matching for Safe, Admissible, and Dynamically Consistent Planning](https://proceedings.mlr.press/v306/huang26t.html) | ICML 2026；主会 | 向flow添加虚拟控制，以控制理论引导满足状态/动作及动态约束 | 约束规划与不确定外界未来预测是不同任务，不能给预测世界强加无碰撞 |
| T07 | [**RP Flow**：Random Process Flow Matching: Generative Implicit Representations of Multivariate Random Fields](https://proceedings.mlr.press/v306/lalanne26a.html) | ICML 2026；主会 | 神经隐式场与源空间GP后验，从稀疏观测生成随机场 | 连续查询与稀疏观测不确定性参考 |

### 3.9 驾驶世界模型与动作联合生成（10 篇）

| 编号 | 论文与原文 | 发表场所 / 年份 / 状态 | 原文创新点概述 | 与 ReactOccFlow 的关系（推论） |
|---|---|---|---|---|
| W01 | [**OccFM**：Towards foundational LiDAR world models with efficient latent flow matching](https://openreview.net/pdf?id=OyX9cC9WaV) | NeurIPS 2025；主会 | 高效LiDAR latent FM与跨域迁移；动作条件occupancy预测 | 直接基线，普通RF与trajectory condition已有 |
| W02 | [**DOME**：DOME: Taming Diffusion Model into High-Fidelity Controllable Occupancy World Model](https://arxiv.org/abs/2410.10429) | arXiv 2024；仅核实预印本 | 时空diffusion Transformer与轨迹重采样控制occupancy未来 | DiT加轨迹条件本身已有直接任务先例 |
| W03 | [**OccWAM**：OccWAM: Occupancy World-Action Models with Joint Scene-to-Ego Flow Matching](https://arxiv.org/abs/2610.17688) | arXiv 2026；仅核实预印本 | occupancy—ego动作联合CFM、双向交互与scene-to-ego两阶段 | 不能把联合flow与双向world-action注意力泛称首次 |
| W04 | [**AutoWorld**：AutoWorld: Learning Multi-Agent Traffic Simulation with Self-Supervised World Models](https://arxiv.org/abs/2603.28963v3) | arXiv 2026；仅核实预印本 | 以LiDAR世界模型预测构造粗细上下文，条件生成多agent行为 | occupancy→actor顺序方案的最相关对照 |
| W05 | [**GoalFlow**：GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving](https://openaccess.thecvf.com/content/CVPR2025/html/Xing_GoalFlow_Goal-Driven_Flow_Matching_for_Multimodal_Trajectories_Generation_in_End-to-End_CVPR_2025_paper.html) | CVPR 2025；主会 | 目标点引导FM以约束多模态自车轨迹生成 | 规划FM基线，与外界actor联合未来仍需区分 |
| W06 | [**DiffusionDrive**：DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html) | CVPR 2025；主会 | 截断扩散与结构化轨迹采样提升实时规划 | 邻近扩散规划对照，不归为FM方法 |
| W07 | [**DUST**：Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](https://proceedings.mlr.press/v306/won26a.html) | ICML 2026；主会 | 模态分流交互、独立噪声和解耦FM目标，以及视觉/动作异步采样 | 双流和异步生成已有关键先例；必须比较真实技术差别 |
| W08 | [**π0**：π0: A Vision-Language-Action Flow Model for General Robot Control](https://www.roboticsproceedings.org/rss21/p010.pdf) | RSS 2025；主会 | 将预训练视觉语言模型与连续动作flow expert结合 | 动作专家与预训练条件分工参考 |
| W09 | [**ManiFlow**：ManiFlow: A General Robot Manipulation Policy via Consistency Flow Training](https://proceedings.mlr.press/v305/yan25a.html) | CoRL 2025；主会 | 结合多模态DiT策略与consistency flow训练实现高效动作生成 | 不能将consistency flow加策略头泛称新颖 |
| W10 | [**IR-WM**：Vision-Centric 4D Occupancy Forecasting and Planning via Implicit Residual World Models](https://arxiv.org/abs/2510.16729v3) | ICRA 2026；作者标注会议 | 以历史BEV为先验预测残差，结合对齐与预测—规划耦合 | 邻近世界模型，非DiT/FM核心方法；残差建模本身已有 |

### 3.10 补充与发表场所待核验（4 篇）

| 编号 | 论文与原文 | 发表场所 / 年份 / 状态 | 原文创新点概述 | 与 ReactOccFlow 的关系（推论） |
|---|---|---|---|---|
| S01 | [**Sequential FM**：Accelerated Sequential Flow Matching: A Bayesian Filtering Perspective](https://openreview.net/pdf?id=tAk1Rzg72a) | ICLR ReALM-GEN Workshop 2026；Workshop | 从Bayesian filtering视角用已有预测更新流，减少重复从无信息先验采样 | 在线滚动预测的热启动已有先例；须区分workshop与主会 |
| S02 | [**AsyncCouple-Flow**：AsyncCouple-Flow: Asynchronous Cross-Modal Coupling and Flow Matching for Spatio-Temporal Forecasting](https://arxiv.org/abs/2609.16573) | arXiv 2026；仅核实预印本 | 异步跨模态时空耦合与条件FM联合多步预测 | 异步多模态预测已有预印本先例 |
| S03 | [**SAMPO++**：Action-Controlled Scale-Wise Flow Matching for Embodied World Models](https://sanmumumu.github.io/SAMPO_plus_plus/) | IEEE TPAMI（待出版社核验） 2026；场所待核验 | 作者项目提出多尺度时序规划与动作控制速度场，分离被动动态及动作残差 | 动作残差flow不能脱离此先例宣称新颖 |
| S04 | [**FlowCritic**：FlowCritic: Bridging Value Estimation with Flow Matching in Reinforcement Learning](https://arxiv.org/abs/2510.22686) | IEEE TPAMI（待出版社核验） 2026；场所待核验 | 用FM表示与采样回报价值分布，替代确定性价值回归 | 价值分布建模参考，与occupancy生成任务不同 |

### 3.11 反事实与事件建模补充（6 篇）

| 编号 | 论文与原文 | 发表场所 / 年份 / 状态 | 原文创新点概述 | 与 ReactOccFlow 的关系（推论） |
|---|---|---|---|---|
| X01 | [**Counterfactual FM**：Counterfactual Generation via Flow Matching: Coupling-Sensitive End-to-End Rates](https://arxiv.org/abs/2610.01193) | arXiv 2026；仅核实预印本 | 双重稳健训练与学习耦合，用FM生成干预分布并分析误差 | 反事实FM本身已有研究；驾驶中的识别假设和干预数据仍需独立论证 |
| X02 | [**Debiased Counterfactual FM**：Debiased Counterfactual Generation via Flow Matching from Observations](https://arxiv.org/abs/2605.07665) | arXiv 2026；仅核实预印本 | 利用观测与反事实分布间关系，构造去偏的FM反事实生成 | 日志动作条件预测不能直接等同于干预分布 |
| X03 | [**CounterFlow**：CounterFlow: Target-Conditioned Flow Matching for Visual Counterfactuals](https://arxiv.org/abs/2610.35679) | arXiv 2026；仅核实预印本 | 按目标类别将源图像传输为视觉反事实解释 | 分类解释意义的反事实与驾驶动作干预不同，不能混为证据 |
| X04 | [**Sketch2Colab**：Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation](https://openaccess.thecvf.com/content/CVPR2026/papers/Daiya_Sketch2Colab_Sketch-Conditioned_Multi-Human_Animation_via_Controllable_Flow_Distillation_CVPR_2026_paper.pdf) | CVPR 2026；主会 | 离散交互事件规划与连续RF姿态传输耦合，支持多实体接触控制 | 混合离散/连续交互生成也有具体应用先例 |
| X05 | [**EventFlow**：EventFlow: Forecasting Temporal Point Processes with Flow Matching](https://proceedings.mlr.press/v300/kerrigan26a.html) | AISTATS 2026；主会 | 非自回归学习不规则事件时刻的联合分布 | 可用于研究事件时间，不能直接当作actor出生/消失的现成实现 |
| X06 | [**PHINN**：PHINN: Persistent Homology Inspired Neural Network for Rare-Event Time Series Generation](https://arxiv.org/abs/2606.15452) | arXiv 2026；仅核实预印本 | 以动态Betti曲线条件和持久同调损失生成罕见事件时间序列 | 拓扑稀有事件建模参考，真实风险概率仍需独立验证 |

## 4. 针对当前项目的优先精读顺序

不是要求先读完 100 篇再做实验。优先读下面这些，形成逐篇方法与假设对照后再改架构。

| 顺序 | 论文 / 发表 | 精读目的 |
|---:|---|---|
| 1 | [OccFM](https://openreview.net/pdf?id=OyX9cC9WaV)；NeurIPS 2025 | 当前实现来源与直接 baseline，先弄清已有动作条件和局限 |
| 2 | [OccWAM](https://arxiv.org/abs/2610.17688)；arXiv 2026 | 核对联合flow、双向交换与ego动作生成的重叠 |
| 3 | [AutoWorld](https://arxiv.org/abs/2603.28963v3)；arXiv 2026 | 建立强 occupancy→actor 顺序生成对照 |
| 4 | [DUST](https://proceedings.mlr.press/v306/won26a.html)；ICML 2026 | 核对双流、解耦训练和异步采样，不重复提出通用机制 |
| 5 | [SD3 / MMDiT](https://proceedings.mlr.press/v235/esser24a.html)；ICML 2024 | 理解MMDiT模态交互与RF训练设计 |
| 6 | [SiT](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9828_ECCV_2024_paper.php)；ECCV 2024 | 区分架构收益、路径收益与采样收益 |
| 7 | [iREPA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3929a7785bd56f57edcff0152ab41289-Abstract-Conference.html)；ICLR 2026 | 检查空间对应如何影响生成，而不是仅换更强教师 |
| 8 | [STFlow](https://proceedings.mlr.press/v306/brinke26a.html)；ICML 2026 | 研究观测锚定先验与数据耦合 |
| 9 | [Purrception](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9d8cf1247786d6dfeefeeb53b8b5f6d7-Abstract-Conference.html)；ICLR 2026 | 理解类别后验与连续速度的结合 |
| 10 | [REPA-P](https://proceedings.mlr.press/v306/jia26j.html)；ICML 2026 | 区分输出约束与中间层物理表征约束 |
| 11 | [SAD-Flower](https://proceedings.mlr.press/v306/huang26t.html)；ICML 2026 | 区分自车可行规划约束与外界未来分布 |
| 12 | [Weighted Joint Sampling](https://proceedings.mlr.press/v337/liu26c.html)；UAI 2026 | 研究有限样本期望和重要性修正 |
| 13 | [MeanFlow](https://papers.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html)；NeurIPS 2025 | 若做低NFE，理解平均速度而非直接砍步数 |
| 14 | [Debiased Counterfactual FM](https://arxiv.org/abs/2605.07665)；arXiv 2026 | 若声称反事实，先核对观测数据到干预分布的条件 |

## 5. 更值得继续查重与实验验证的问题

以下是文献启发下的**候选研究问题**，尚未证明新颖性，也未作为作者已确认的论文贡献。不能把几篇方法组合起来就宣称获得创新。

### 5.1 多动作分支间的联合反应，而不只是单分支预测

当前主要学习 `p(O_future, A_future | H, tau)`。可以进一步研究：同一历史 H 下，不同候选动作 tau 的预测如何对应同一场景不确定性，同时保证每个动作条件下的边缘分布正确，并捕捉周围 actor 的实际响应。

重点不在“让两个输出都变化”，而在**同一交通参与者的响应变化、非交互背景稳定性，以及不同动作之间的联合依赖是否正确**。共享随机种子可以是诊断或对照，但不能自动识别真实反事实耦合。需要同历史实际执行不同动作的分支标签、可响应 NPC、共同噪声等简单基线，以及条件预测与真实干预的严格区分。

已有世界—动作生成 [OccWAM](https://arxiv.org/abs/2610.17688)、[DUST](https://proceedings.mlr.press/v306/won26a.html) 和反事实 FM [Counterfactual FM](https://arxiv.org/abs/2610.01193)、[Debiased Counterfactual FM](https://arxiv.org/abs/2605.07665) 均必须列入查重。可检验的增量必须具体到稠密 occupancy、实例 actor 与真实动作干预的联合问题，不能泛称“首次反事实 flow”。

### 5.2 连续几何与离散交互事件的联合未来

actor 的位置、速度、朝向是连续的；存在性、接触/让行等事件具有不同结构。值得检验：把当前独立存在分类与连续状态流，改为结构明确的混合生成，能否减少几何冲突、漏检和交互模式切换，同时维持概率质量。

[Discrete FM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0d629a734b56a642701bba7bc8bb3ed-Abstract-Conference.html)、[EF-VFM / TabbyFlow](https://proceedings.mlr.press/v267/guzman-cordero25a.html)、[Purrception](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9d8cf1247786d6dfeefeeb53b8b5f6d7-Abstract-Conference.html) 与 [Sketch2Colab](https://openaccess.thecvf.com/content/CVPR2026/papers/Daiya_Sketch2Colab_Sketch-Conditioned_Multi-Human_Animation_via_Controllable_Flow_Distillation_CVPR_2026_paper.pdf) 已覆盖离散、混合生成及交互事件耦合。因此“增加离散分支”本身不足以建立新意；需要驾驶任务中特定的状态定义、几何对应和联合概率训练/推理机制。

必须保留当前已观测 actor 的选取边界，不能用未来标签挑选将出现的 actor。若要显式生成新进入的 actor，需另行定义可变集合、身份和观测协议；不能悄悄改变当前任务来获得指标优势。

### 5.3 有限采样预算下的可靠事件概率

当前小 M 多世界采样与风险目标之间存在直接矛盾：稀有碰撞可能被漏采。可以研究借助稠密—稀疏联合结构的事件定向采样与概率修正，目标是在固定计算预算下改善碰撞概率估计，而不只是提高样本多样性。

[Diverse FM](https://proceedings.mlr.press/v306/wu26be.html) 和 [Weighted Joint Sampling](https://proceedings.mlr.press/v337/liu26c.html) 已分别研究多样性与非IID采样的重要性修正。使用guidance主动采出危险世界后，命中比例通常不能再直接当作原模型概率。需要明确目标分布、proposal/权重、估计偏差/方差和有效样本量；学习权重的误差也需要验证。

若这一路线只改进统计估计，对 ICCV 的视觉方法贡献可能仍显不足；应检验 occupancy—actor 几何是否提供了不可被通用采样器替代的作用。这是研究判断，不是会议硬性规则。

### 5.4 我会优先推进的次序

先用现有模型和强基线找出可量化的联合失败：配对打乱诊断、W2-SC 强并行、occupancy→actor 单向生成，以及各损失的独立训练消融。由证据决定优先研究 5.1 或 5.2。5.3 可在联合模型有效后作为下游用途，避免三个方向同时铺开。

如果最终决定研究几何/时间对齐，应先与 iREPA、REPA-P、GSE-Flow 等相关机制逐项比较。检索结果已经表明“加局部attention”“加物理loss”“异步去噪”“换先验”都不能直接作为未经查重的新颖性结论。

## 6. 发表信息与引用时容易写错的条目

- **Latte：TMLR 2025**，不是仅按 2024 预印本年份标注，也不能未经核实写成 ECCV。
- **RAE：ICLR 2026**；其 2025 预印本与正式会议版本只算一篇。
- **REPA-E：ICCV 2025**；**iREPA：ICLR 2026**，不同论文。
- **VA-VAE / LightningDiT：CVPR 2025**，正式题名为 *Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models*。
- **OT-CFM：TMLR 2024**；**Functional Flow Matching：AISTATS 2024**。
- **ManiFlow：CoRL 2025**，以 PMLR 正式记录为准。
- **OccFM：NeurIPS 2025**，本轮核实正式论文标注；**DOME** 的 under-review 文件不足以确认 ICLR 录用。
- **IR-WM：作者元数据标注 ICRA 2026**，本表保留与出版社核验不同的证据等级。
- **AutoWorld** 的早期题名使用 *Scaling*、当前 v3 使用 *Learning*，不是两篇独立论文。
- **S01 是 ICLR Workshop**；S03/S04 的 TPAMI 出版线索未通过出版社核验，不纳入已核实期刊计数。

部分网页在直接打开时出现 OpenReview 验证、IEEE JS 验证或 CVF 403；本轮没有绕过访问限制，具体情况在条目备注中说明。已核实的可检索最终 PDF/会议记录与未核实的出版线索分开呈现。

## 7. 使用这份清单的方式与限制

每确定一个技术方向，进一步精读最接近的论文正文，建立“状态变量—条件信息—先验/耦合—训练目标—信息交互—采样—干预数据—评价预算”的对照，之后才作新颖性判断。这份摘要级检索不能替代该步骤。

本文没有统一比较各论文的 FID、mIoU 或驾驶成绩；其任务、输入、预训练和计算预算不同。不能把它们跨表直接排名，也不能把文献报告结果写成项目已有实验。

本次工作为文献与研究建议，不启动训练、不改变现有 checkpoint、模型或参考仓库。权威工程计划仍以原计划及用户后续明确决定为准。


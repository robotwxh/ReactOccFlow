# 2026-11-07：标准DiT生成mIoU及与原CFM、OccFM对比

> 后续深入诊断已发现明确结构缺陷：当前400维patch通过hidden256的输入/输出线性瓶颈，使144个方向的初噪无法被直接速度场控制；8个固定tune场景的数值对照支持其严重影响。见 [完整结构与差异分析](2026-11-07-dit-cfm-gap-analysis.md)。下文的“具体根因仍未确定”是初次评测实现检查阶段的记录；官方指标未改，完整mIoU差距仍未逐因素分摊。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Verification Status: VERIFIED（完整生成与原始计数、身份核验；不表示模型质量合格或统计显著）
- Version Label: standard_dit_best100000_miou_v1

标准DiT联合occupancy–actor模型已于03:55完成100,000步训练。本次使用此前按完整tune联合损失选定的best100,000 EMA，对官方validation150场景、**4,219个窗口**进行生成评测。两张卡各完成2,110及2,109个窗口，退出码0/0，合并退出0；本次完整运行耗时38.87分钟。没有重新启动训练或执行优化器更新。

结果表明：**当前标准DiT配置的自由生成预测质量很差**。全网格平均mIoU仅2.5755%，低于原actor+U形CFM的28.6739%达26.0985个百分点；训练tune loss下降到0.5522不能替代生成质量评价。

| 模型 / 条件 | 1秒mIoU | 2秒mIoU | 3秒mIoU | 三时刻平均 |
|---|---:|---:|---:|---:|
| OccFM论文，已给定未来自车轨迹 | 36.42% | 27.39% | 21.66% | 28.49% |
| 原OccFM e196既有完整复现 | 35.8208% | 27.3686% | 21.6111% | 28.2669% |
| 上次actor+U形CFM，best52,000 | 33.4676% | 28.0838% | 24.4704% | 28.6739% |
| 本次actor+标准DiT，best100,000 | **2.8361%** | **2.5240%** | **2.3663%** | **2.5755%** |

标准DiT相对原actor+U形CFM在1/2/3秒分别为−30.6315 / −25.5598 / −22.1042个百分点，平均−26.0985个百分点；相对论文未来轨迹条件平均−25.9145个百分点。论文来自[OccFM v2表1](https://arxiv.org/html/2506.23434v2)。既有两组基线均复用原完整评测结果，没有伪称本次重新推理。

两个actor模型的评测使用相同窗口及顺序、oracle actor历史、mask_lidar历史occupancy、已给定的日志未来ego XY、同一冻结VAE、train-only归一化、FP32、NFE10、逐窗口身份固定seed2026、batch4/卡及encode/decode chunk4/2。18项采样/数值/计分设置和5,723个输入文件SHA-256完全相同，数据/噪声/预测/评分函数保持一致。协议版本1→2仅新增主干选择身份。每rank原始int64计数、最终合并计数、4,219窗口无重复完整覆盖及三组逐时刻逐类别GT计数均已严格核对；主表Avg是1、2、3秒mIoU的算术平均。

预测在固定当前E0坐标系生成，再由评分器用未来GT位姿做3D最近邻对齐；超出E0视野的预测填empty17并保留计分，不使用camera/LiDAR评分mask。原生E0已知区域Avg也只有**2.7856%**，因此不能把本次低分仅归因于未来坐标投影或视野覆盖。

## 针对低结果的有界检查

没有因为官方验证分数低而换checkpoint、改seed/NFE、重新评测挑选结果或开始重训。本次额外检查只判断实现和权重装配是否存在明显错误：

- best100,000的172个EMA张量与最终完整checkpoint的EMA去`module.`前缀后逐位相同；没有误用旧权重或raw参数。
- 评测源副本中的81个原训练文件与冻结训练清单SHA-256完全相同。模型确为hidden256/depth8/heads8/patch5的StandardDiT，10,878,667个参数，推理时全部冻结，EMA严格加载。
- 输出投影权重有限且非零；缓存latent未缩放，训练目标/历史乘10，采样历史也乘10，积分结果除10，与训练endpoint路径一致，未发现漏乘或重复缩放。
- 固定一个**tune工程样本**、seed617、NFE10，冻结训练源码与当前工作区实现产生的future latents、actor状态、存在性和解码语义逐位相同。这仅排查该样本上的实现一致性，不是多样本复现保证或官方验证集调参。
- 该工程样本standard口径的1/2/3秒mIoU也只有2.6665 / 2.8394 / 2.7491%。其legacy分数会因单窗口缺失类别记IoU=1而被抬高，不能把单窗口legacy高分当作生成质量改善。
- 全数据预测并非全部empty：1秒预测free占93.94%，GT free占94.75%；但多个语义类别IoU很低。此分布本身不能定位是何种训练或生成机制导致失败。

因此，当前结论是“本轮标准DiT实验生成质量差”，**具体根因仍未确定**。本次没有证明标准DiT架构普遍无效，也不能仅凭joint tune loss判断自由积分生成已收敛。后续若继续，应在train/tune上诊断条件利用、速度场与自由rollout误差，再另行制定实验；不能反复针对本次官方验证分数选择设置。

## 对比解释与文件

原U形模型使用OccFM主干初始化、best52,000/最终72,000及原学习率历史；标准DiT从随机初始化训练至100,000步。因此即使评测口径一致，也不是仅改变主干的同初始化、同预算消融。与原OccFM相比，还存在额外真值actor输入、E0/历史mask、轨迹条件表达、CFG、BF16/FP32与噪声等差异。单训练seed、单生成seed及重叠窗口不支持统计显著性或架构因果结论。

- [四组完整对照报告](../../outputs/evaluation/2026-11-07-standard-dit-best100000/标准DiT与CFM及OccFM_mIoU对照.md)
- [对照JSON](../../outputs/evaluation/2026-11-07-standard-dit-best100000/comparison.json)
- [逐类CSV（17类×3时刻）](../../outputs/evaluation/2026-11-07-standard-dit-best100000/per_class_comparison.csv)
- [全量计数](../../outputs/evaluation/2026-11-07-standard-dit-best100000/full/result.json)
- [实际命令和退出码](../../outputs/evaluation/2026-11-07-standard-dit-best100000/full/launch.json)
- [单tune样本源码一致性](../../outputs/evaluation/2026-11-07-standard-dit-best100000/engineering_probe_comparison.json)
- [有界独立实现审查](../../outputs/evaluation/2026-11-07-standard-dit-best100000/independent_low_score_review.json)
- [验收与参考目录保护记录](standard_dit_miou_acceptance_20261107.json)

11项生成评测器CPU测试及17项报告核验CPU测试通过。两个参考目录、四份冻结源码清单、全部检查点和既有基线结果的本次前后校验通过；所有数据均保留。

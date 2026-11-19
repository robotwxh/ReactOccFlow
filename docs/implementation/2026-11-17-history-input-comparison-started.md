# 相同10,000步预算：mask_lidar训练与双输入评测

> 最新恢复：用户已明确要求继续，控制器从完整6,842步状态恢复训练，保留10,000步总预算，之后继续正式评测与对比。详见[恢复记录](2026-11-17-history-input-comparison-resumed.md)。

> 同日后续暂停：用户要求“训练先暂停”。本轮mask模型已保存到6,842步并正常退出，后续正式评测/对比一并暂停。恢复时保留总预算10,000步，接续剩余3,158步；下文启动状态和ETA为历史记录。见[暂停与恢复记录](2026-11-17-history-input-comparison-paused.md)。

用户于2026-11-17要求：“按照mask_lidar输入进行10000步的模型训练，然后分别评估密集语义历史输入和mask_lidar输入，并进行对比”。

## 已启动

北京时间**2026-11-17 10:38:23**启动独立后台控制器，PID为`164292`；正式训练torchrun PID为`164293`。模型优化于10:39:01开始，已观测到第一步有限损失和两卡约92%利用率。

当前训练的是**U形CFM＋actor、mask_lidar历史输入、最多10,000优化步**。直接读取既有完整masked缓存，不重新生成缓存，不从旧joint模型或密集历史模型续训。VAE保持冻结。

工作根目录：`outputs/history_input_comparison/2026-11-17-seed2026/`。

## 固定训练对照

两轮均使用seed2026、原`occfm_history.ckpt`主干初始化、新actor/交互分支初始化、主干LR2e-5/新增分支LR1e-4、FP32、两卡每卡batch2/累积2（有效batch8）、warmup1000、总步数10000、完整tune间隔2000、workers2、decode-chunk1和相同EMA/损失。密集模型已有10,000步检查点，不重新训练。

实际新训练日志配置已与密集模型配置逐项核对，仅`history_visibility`不同。33份模型/损失/优化器相关Python源码完全相同；训练main的AST除退出清理及其初始变量外一致。

两组缓存均为train17,468/tune2,176窗口，窗口身份和顺序一致，源occupancy、官方表、标签、split、normalization、VAE身份及时间/网格/后验设置一致。每组各抽查4个均匀分布窗口，除历史latent外的全部数组逐值一致；未将抽查解释为全载荷逐条验证。

详见运行根目录`training_alignment.json`和`startup_health.json`。

## 自动执行顺序

1. `train_mask`：完整masked缓存训练至10,000步。
2. `export_step10000`：分别从两轮latest完整状态中严格加载并导出**预先固定的第10,000步EMA**，不依照正式验证mIoU挑选checkpoint。
3. `evaluate_dense_semantic`：密集模型使用密集历史，双卡完整生成官方验证集。
4. `aggregate_dense_semantic`：CPU汇总两rank原始计数。
5. `evaluate_mask_lidar`：mask模型使用mask历史，同协议完整生成。
6. `aggregate_mask_lidar`：CPU汇总。
7. `report`：审计并输出总体mIoU、逐类别结果与密集减mask的差值。

每个GPU阶段顺序使用两张卡，不同时运行两套双卡任务。控制器核验冻结源码；每阶段必须正常退出并产生完整产物才进入下一阶段。失败不自动重试；新运行的STOP_REQUESTED标记会阻止后续阶段，不影响保留的raw停止标记。

## 评测协议

- 官方validation150场景、4,219窗口，两个rank分别2,110/2,109窗口。
- 同一U形CFM＋actor、同一冻结VAE、同一数据/normalization；两者均评价最终第10,000步EMA。
- FP32、NFE10、seed2026、按窗口身份固定噪声、每条件1个世界。
- 每卡batch4、workers4、encode-chunk4、decode-chunk2。
- 主表：未来ego原始全网格、17个非空语义类别，忽略GT255，范围外预测empty17参与计分；报告1/2/3秒及算术平均。
- 保留原生E0、未来网格覆盖区域与standard/legacy完整计数；主表使用standard规则。
- actor从当前/历史观测选取，未来actor/occupancy真值不进入预测器；日志未来ego XY是显式动作条件，GT未来位姿只在评分对齐中使用。
- 汇总报告复算原始计数并核对精确rank游标、窗口、GT计数、覆盖、共享数据SHA和评分函数；所有训练/评测/汇总进程的实际退出码必须为0。

本次为单训练seed与单生成seed对照，窗口重叠；不宣称多seed显著性、物理一致性或规划收益。验证集已有诊断使用历史，不宣称独立未见确认性测试。

## 退出处理与验收

上轮训练完成后退出卡住的具体等待点未完全定位。本轮增加显式关闭persistent DataLoader worker及CUDA同步后再销毁进程组，不改变训练计算。

- 73项定向测试通过，包括真实CPU worker退出、模式/配置不一致拒绝、固定步数导出保护和流程推进。
- 双卡真实短跑完成2步优化、完整小集tune、生成探针、保存与暂停，实际退出0。
- 4窗口双卡评测短跑两个rank及CPU汇总退出均为0，仅作运行验证，不能作为正式mIoU成绩。
- 密集模型最终EMA导出与原best10000权重逐项完全一致。
- 使用192个独立普通文件冻结源码，指纹`ef94fdfa820d9f2043c6e464b020c4ab038e3cf925ef53f13a6d91acdd2742c8`。后续工作区修改不进入运行。
- 原密集训练源码/权重、旧缓存及raw STOP标记保持原状。来源与保护记录见[provenance](../provenance/history_input_comparison_sources_20261117.json)。

## 耗时与产物

预计mask模型训练约9.5–10.5小时，两轮完整评测及汇总约1–2小时，总计约**11–12小时**。以10:38启动计，预计北京时间11月17日晚至18日凌晨完成；实际进度以运行根目录`pipeline.json`和日志为准。

运行中的文件：`launch_config.json`保存全部确切命令，`comparison_config.json`保存两模型与报告路径，`pipeline.json`保存阶段、PID、进度与退出码，`train_mask/training.jsonl`保存训练/验证记录。

完成后自动生成：

- `checkpoints/{dense,mask}_step10000_ema.pt`：严格导出的固定步数EMA，原完整权重不改动。
- `evaluation/{dense,mask}/full/`：两轮identity、rank状态/结果和全量汇总。
- `comparison/comparison.json`：总体、逐类、全部评分口径和审计结果。
- `comparison/comparison.csv`：1/2/3秒及平均mIoU。
- `comparison/per_class_comparison.csv`：17类×3时刻IoU与差值。
- `comparison/密集历史与mask_lidar_10000步评测对比.md`：四列表格及结论边界。

此记录是启动状态，不包含尚未生成的正式对比数值。

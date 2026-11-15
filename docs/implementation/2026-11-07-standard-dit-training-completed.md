# 2026-11-07：标准DiT首seed训练完成

> 后续评测已完成：best100,000的官方4,219窗口Avg mIoU仅2.5755%，详见 [完整评测与实现核查](2026-11-07-standard-dit-miou-evaluation.md)。下文保留训练完成及评测启动时的记录。

标准DiT occupancy–actor joint seed=2026实验已于北京时间2026-11-07 03:55:01完成原定100,000步；最后完整tune loss=0.5522293311513242，本次继续改善，因此best EMA也是100,000步。共50次完整tune验证，bad_evaluations=0，结束原因是达到步数上限。torchrun及两个训练rank均已退出。

完整状态位于`outputs/reactflow_dit/seed_2026/latest.pt`，推理EMA位于同目录`best.pt`。模型、EMA、AdamW、scheduler、双rank RNG及训练游标均在完整检查点中保留；训练源副本仍为`outputs/reactflow_dit/source_seed_2026/`，没有改写或自动开始其他训练。

训练完成证据及检查点摘要见[standard_dit_training_completed_20261107.json](standard_dit_training_completed_20261107.json)。此前54,000步暂停备份继续保留；11月5日从54,000步恢复时，首个54,050步loss与暂停前相同步数一致。完成训练不意味着已证明mIoU或物理一致性提升。

用户随后要求评测并比较OccFM与上次actor+U形CFM。新输出目录为`outputs/evaluation/2026-11-07-standard-dit-best100000/`，固定此100,000步best EMA后再评测官方validation150场景/4,219窗口。评测使用原标准DiT训练源码的独立普通文件导出，新增通用评测器；NFE=10、FP32、逐窗口seed2026、batch=4/卡、world_size=2与上次CFM相同。16窗口双卡预检已通过，完整评测进行中。

本次评测对比预先保留训练长度、初始化、学习率及OccFM输入/表示/CFG/精度差异；不把结果差异自动解释为仅由主干改变带来的提升。权重与协议记录在新输出的`protocol.json`，所有实际命令与退出码在`pilot/launch.json`和`full/launch.json`。

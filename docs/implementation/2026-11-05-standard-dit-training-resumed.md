# 2026-11-05：从54,000步恢复标准DiT联合训练

> 后续状态：2026-11-07 03:55已完成100,000步训练，best EMA也是100,000步，tune loss=0.5522293311513242。见 [训练完成记录](2026-11-07-standard-dit-training-completed.md)。下文为恢复当时的状态。

按用户“继续训练吧”的指示，北京时间2026-11-05 17:18:43通过既有`scripts/resume_dit_training.py`恢复原seed=2026实验。训练器于17:20:02记录`started, global_step=54000`，说明完整缓存校验、配置/数据身份检查、模型/EMA/优化器/scheduler及双rank随机状态加载均通过；恢复后首批优化日志和GPU采样详见`standard_dit_resume_20261105_health.json`。

17:24:33已实际记录恢复后的第54,050步，loss=0.5349867269396782，与暂停前相同步数的日志数值完全一致。GPU活跃阶段25次采样平均利用率为90.88%/84.56%，峰值99%/100%，显存约9.7GB/卡；训练继续在后台运行，未见报错。此处训练损失不代表mIoU评价结果。

| 项目 | 本次恢复状态 |
|---|---|
| 实验 | 标准全局时空DiT occupancy–actor joint，seed=2026 |
| 架构 | hidden=256、depth=8、heads=8、patch=5，1,200时空token |
| 恢复文件 | `outputs/reactflow_dit/seed_2026/paused_step_054000_20261105.pt` |
| SHA-256 | `ce5f6b30882336007ff9b8e0191da30b757b181e78c11514f284807ff72df880` |
| 恢复步数与游标 | 54,000步，epoch=24、next_batch=3216 |
| 状态 | 模型、EMA、AdamW、scheduler、双rank RNG、数据游标及早停计数完整继续 |
| 原best tune loss | 0.5647131986610445，27次完整验证，bad_evaluations=0 |
| 训练配置 | 两张RTX A6000，每卡batch=2、累积=2、有效batch=8、workers=4、decode-chunk=4、FP32 |
| 训练目标与计划 | 保持原RF目标和一致性项；上限100,000步，每2,000步完整tune验证/保存，patience=10 |
| VAE | 冻结，沿用既有Swin VAE及缓存 |
| 源码 | 原`outputs/reactflow_dit/source_seed_2026/`普通文件副本；不使用修改后的工作区训练器 |
| 进程 | torchrun PID=25182；rank PID=25202、25203 |
| 自动重启 | 关闭；本次启动来自用户明确恢复指示 |

暂停前最后日志为54,600步，但完整检查点为54,000步；该区间及少量未打印步骤在本次恢复后重跑。日志按原文件追加，应结合最新`started`事件及UTC时间识别本轮，不能直接取整个日志的最大global_step。

日志：`outputs/reactflow_dit/seed_2026/training.jsonl`；本次console：`outputs/reactflow_dit/seed_2026.resume_20261105T171843194848.log`。完整命令与环境参数在`outputs/reactflow_dit/seed_2026.launch.json`，当前状态在同级`seed_2026.runtime.json`。

本次复用原恢复入口，未修改训练代码、架构或学习率计划。恢复前dry-run校验82个源码/检查点文件通过；训练器继续执行19,722项运行身份严格检查。暂停恢复文件、原CFM best/latest和冻结源码保持不变，参考仓库本次前后快照见`outputs/reactflow_dit/resume_20261105/`。目录整理迁移的基线评测产物不参与该训练恢复身份，保留于`outputs/audits/`。

来源与恢复验收记录：[dit_training_resume_20261105.json](../provenance/dit_training_resume_20261105.json)。原U形CFM已结束，本次仅继续标准DiT实验。

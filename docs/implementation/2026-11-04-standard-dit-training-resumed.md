# 2026-11-04：恢复标准 DiT 联合训练

> 后续状态：2026-11-05已按用户要求再次暂停，完整可恢复状态保存至54,000步，最后日志54,600步。见 [最新暂停与恢复记录](2026-11-05-dit-training-paused.md)。下文为本轮恢复启动时的记录。

按用户“按照标准DiT的架构进行训练”的指示，于北京时间2026-11-04 16:41:26通过既有手动入口恢复原seed=2026实验。使用完整22,000步检查点、原训练源码副本和原配置，原训练器于16:42:05记录`started, global_step=22000`，严格恢复检查已通过。最新优化步数和GPU检查见同目录`standard_dit_resume_20261104_health.json`。

实际续训已于16:44:29记录第**22,050步**，损失0.5724391490221024，有限且未见报错。20次GPU采样平均利用率为87.65% / 86.15%，峰值均100%，显存约9.7GB/卡。训练继续在后台运行，最近完整落盘状态仍为22,000步。原训练源码81文件、暂停检查点及CFM best/latest字节摘要均保持不变，两个参考目录的本次前后快照完全相同。

| 项目 | 配置 |
|---|---|
| 主干 | 标准全局时空DiT，hidden=256、depth=8、heads=8、patch=5、MLP ratio=4 |
| 时空token | 12帧槽 × 100空间patch = 1,200 |
| 联合分支 | occupancy–actor双向交互，joint模式，最多8个当前选定actor |
| 可训练部分 | DiT、actor及交互分支，约10.88M参数；单套Swin VAE冻结 |
| 训练目标 | 原CFM/RF目标及已有端点一致性项 |
| 起始完整状态 | 22,000步，epoch=10、next_batch=340，双rank RNG、EMA、优化器、scheduler及早停状态一并恢复 |
| 原计划 | 上限100,000步，每2,000步完整tune，patience=10 |
| 双卡 | 两张RTX A6000，DDP，每卡batch=2、累积=2，有效batch=8，FP32 |
| 数据与解码 | 完整train17,468 / tune2,176窗口；每rank workers=4、decode-chunk=4 |
| 研究条件 | oracle E0 occupancy/actor历史，日志ego未来XY为给定动作监督条件；未来actor/occupancy保留为target |

恢复前核验82个源码/检查点文件，以及原训练器全部19,722项运行身份；实际构造StandardDiT并严格加载保存的flow状态通过。检查点SHA-256：`b23778564e3098ae580a63979071b0f635127d24ccff70e373a03c2d696a4e68`。没有修改原训练源码、检查点身份或恢复检查。

上次暂停前日志曾达到22,550步，但完整落盘状态为22,000步；之后未保存区间重跑。原日志以追加方式保留，应依据最近一次`started`事件识别当前进度，不能直接取全日志历史最大值。

本次使用`outputs/reactflow_dit/source_seed_2026/`的原普通文件副本，后续工作区编辑不影响当前训练。U形CFM实验已在72,000步结束，其52,000步best权重及mIoU评测结果保留。此次没有使用U形主干权重初始化标准DiT。

## 运行与审计位置

- 恢复入口：`scripts/resume_dit_training.py`；启动前`--dry-run`通过。
- 原参数、绝对路径命令与恢复记录：`outputs/reactflow_dit/seed_2026.launch.json`。
- 当前状态：`outputs/reactflow_dit/seed_2026.runtime.json`。
- 实际训练日志：`outputs/reactflow_dit/seed_2026/training.jsonl`。
- 本次console：`outputs/reactflow_dit/seed_2026.resume_20261104T164126788165.log`。
- 恢复前日志/状态、严格预检和参考保护：`outputs/reactflow_dit/resume_20261104/`。
- 来源与验收：`docs/provenance/dit_training_resume_20261104.json`及`docs/implementation/standard_dit_resume_20261104_health.json`。

没有启动自动重启守护进程。完整训练检查点仍按原验证周期保存；当前训练器不支持信号触发的即时保存，日志步数与完整落盘步数应分开报告。

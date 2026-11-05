# 2026-11-05 标准 DiT 训练暂停与恢复记录

> 后续状态：同日17:18已按用户“继续训练吧”的要求，从本记录保存的54,000步完整状态恢复双卡训练，见 [本次续训记录](2026-11-05-standard-dit-training-resumed.md)。下文保留暂停当时的状态和保存边界。

按用户“训练先停掉，记得保存进度，以便于再次启动”的指示，已停止标准DiT joint seed=2026双卡实验。北京时间16:57完成暂停核验：torchrun、两个训练rank与数据加载共19个进程全部退出，GPU计算进程列表为空。运行状态标记为`paused_by_user`，自动重启关闭。原U形CFM实验已结束，本次未启动其他训练。

| 项目 | 已保存及核验的状态 |
|---|---|
| 完整可恢复步数 | **54,000 / 100,000（54%）** |
| 最后训练日志 | 54,600步，loss=0.5168503820896149 |
| 游标 | epoch=24，next_batch=3216 |
| 最佳tune损失 | 0.5647131986610445，第54,000步 |
| 验证与早停 | 已完成27次完整验证，bad_evaluations=0，patience=10 |
| 保存内容 | 模型、EMA、AdamW、scheduler、两rank RNG、数据游标、配置、源码/数据身份 |
| 训练配置 | 两卡，每卡batch=2、累积=2、有效batch=8，workers=4、decode-chunk=4；VAE冻结 |
| 恢复验证 | 19,722项运行身份完全匹配；模型和EMA严格加载、优化器和scheduler加载通过；scheduler步数54,000 |
| 独立副本 | 81个冻结源码文件不变；检查点与best EMA均为普通文件备份，SHA-256匹配 |

当前运行的冻结训练器不支持信号触发的即时保存。本次保留最近完整写盘的54,000步检查点；54,000步之后的内存更新没有保存，恢复时需重跑。最后日志比检查点多600步，停止前还可能执行了未打印的步骤（日志每50步一次）。不能将54,600步日志当作可恢复权重。console中的SIGTERM/SignalException由本次主动暂停产生。

## 保存位置

- 完整恢复文件：`outputs/reactflow_dit/seed_2026/paused_step_054000_20261105.pt`。
- SHA-256：`ce5f6b30882336007ff9b8e0191da30b757b181e78c11514f284807ff72df880`。
- 最佳EMA备份：`outputs/reactflow_dit/seed_2026/paused_best_step_054000_20261105.pt`，SHA-256为`59aa58c748fad7cf4c340a13c177de6174a80fbe23f32cc2a425986dc53d64f3`。
- 原`latest.pt`、`best.pt`保留；best EMA与完整检查点的EMA逐张量一致。继续训练使用完整恢复文件。
- 日志快照：`outputs/reactflow_dit/seed_2026/training.paused_20261105.jsonl`；原日志保留。
- 暂停审计：`outputs/reactflow_dit/seed_2026.paused_by_user.json`；此前暂停记录及运行元数据另存于`outputs/reactflow_dit/pause_20261105/`。
- 原冻结源码：`outputs/reactflow_dit/source_seed_2026/`，清单为同级`source_seed_2026.manifest.json`。
- 严格检查：`outputs/reactflow_dit/pause_20261105/strict_checkpoint_verification.json`。
- 来源及保护记录：[dit_training_pause_20261105.json](../provenance/dit_training_pause_20261105.json)。两个参考仓库本次前后快照与HEAD均一致，未修改参考源码或运行中的冻结源码。

## 后续恢复

已复用既有[手动恢复入口](../../scripts/resume_dit_training.py)，没有修改训练代码。只校验恢复条件：

```bash
/home/lthpc/wxh/envs/ReactOccFlow/bin/python \
  /home/lthpc/wxh/Paper/scripts/resume_dit_training.py --dry-run
```

本次仅执行该dry-run，82个源码/检查点文件校验通过，未启动训练。

用户后续明确要求继续训练时执行：

```bash
/home/lthpc/wxh/envs/ReactOccFlow/bin/python \
  /home/lthpc/wxh/Paper/scripts/resume_dit_training.py
```

入口使用原专用环境、固定训练副本和原双卡配置，从完整54,000步后台恢复，保留学习率计划、EMA、优化器、随机状态、训练游标及早停计数。保持固定源码、缓存、VAE权重不变；不要直接使用已修改的工作区训练入口。恢复后日志继续追加，重跑区间会出现重复global_step，应以最新`started`事件及时间区分训练阶段。

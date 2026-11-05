# 2026-11-01 标准 DiT 训练暂停与恢复记录

> 后续状态：2026-11-04 16:41已按用户明确指示，从本记录保存的22,000步完整状态恢复双卡训练，见 [标准DiT续训记录](2026-11-04-standard-dit-training-resumed.md)。下文保留暂停当时的记录。

按用户“训练暂时先停掉，记录保存，便于后面重新启动”的指示，已在北京时间 2026-11-01 12:53 发起停止。torchrun、两卡训练与数据加载共 19 个进程均已停止，GPU 训练显存已释放。12:56 完成检查点和恢复身份核验。不会自动重启本轮或其他种子。

## 保存的实际进度

| 项目 | 已核验状态 |
|---|---|
| 实验 | ReactFlow joint，标准 DiT，seed=2026，oracle E0、日志自车动作条件 |
| 最后已记录训练步数 | 22,550 |
| **完整可恢复检查点** | **22,000 步**，epoch=10、next_batch=340 |
| 最佳完整 tune 损失 | 0.7826553602260875，第 22,000 步；共完成 11 次完整验证 |
| 完整训练状态 | 模型、EMA、AdamW、scheduler、两 rank RNG、训练游标、配置、数据/源码身份摘要 |
| 恢复身份核验 | 19,722 项全部一致 |
| 原实验配置 | 两卡，每卡 batch=2、累积=2、有效 batch=8，workers=4、decode-chunk=4，上限 100,000 步 |

**当前加载的训练器没有信号触发的即时保存接口。** 本次立即停止后，保留最近已完整写盘的第 22,000 步状态；22,000 步之后到停止之间的内存进度未保存，至少 550 个已有日志的步骤需在恢复后重算，实际停止前可能还有未打印的步骤。原始日志保留，不能把日志中的 22,550 步当作可恢复权重进度。console 中的 SIGTERM/SignalException 来自本次主动停止。

## 文件位置

- 独立保留的完整状态：`outputs/reactflow_dit/seed_2026/paused_step_022000_20261101.pt`。
- SHA-256：`b23778564e3098ae580a63979071b0f635127d24ccff70e373a03c2d696a4e68`。
- 原 `latest.pt`、`best.pt` 均保留；`best.pt` 主要用于 EMA 推理，完整恢复应使用上述完整状态。
- 最佳 EMA 另存为 `outputs/reactflow_dit/seed_2026/paused_best_step_022000_20261101.pt`；暂停时日志快照为同目录 `training.paused_20261101.jsonl`。
- 暂停审计及全部配置：`outputs/reactflow_dit/seed_2026.paused_by_user.json`。
- 启动参数：`outputs/reactflow_dit/seed_2026.launch.json`；当前状态：同级 `seed_2026.runtime.json`。
- 原训练日志：`outputs/reactflow_dit/seed_2026/training.jsonl`；本次暂停另保留日志快照。
- 已固定的训练代码：`outputs/reactflow_dit/source_seed_2026/`，逐文件摘要在同级 `source_seed_2026.manifest.json`。

## 后续手动恢复

恢复入口为 [resume_dit_training.py](../../scripts/resume_dit_training.py)。脚本会使用原专用环境、原训练副本和原配置，从第 22,000 步后台启动双卡训练，保留优化器、学习率计划、EMA 与 RNG；完整缓存审计和严格身份匹配由原训练器继续执行。运行前会检查记录的旧进程均已退出，并验证训练副本和保留检查点的 SHA-256。

只检查恢复条件，不启动训练：

```bash
/home/lthpc/wxh/envs/ReactOccFlow/bin/python \
  /home/lthpc/wxh/Paper/scripts/resume_dit_training.py --dry-run
```

用户决定继续时，执行以下命令：

```bash
/home/lthpc/wxh/envs/ReactOccFlow/bin/python \
  /home/lthpc/wxh/Paper/scripts/resume_dit_training.py
```

命令返回后，训练在独立 session 中继续。新 console 日志位置和新 PID 会写回启动/状态文件。原始日志以追加方式保留，所以重跑未保存区间时可能再次出现相同 global_step；应结合时间和最近的 `started` 事件识别训练阶段。后续修改工作区时保持训练副本、对应配置、缓存与 VAE 权重不变，继续使用该恢复入口。

本次仅执行了 `--dry-run`，82 项训练副本/检查点文件校验通过，没有重新启动训练。来源与辅助脚本记录见 [暂停来源记录](../provenance/dit_training_pause_20261101.json)。

## 参考目录审计的额外记录

只读审计发现，`IR-WM/data` 相对 2026-10-31 的空目录快照变成了指向 `/home/lthpc/data` 的符号链接，入口时间戳为 2026-11-01 12:55，晚于本次暂停请求。该变化的来源未确定，本任务没有创建、删除或修改这个入口，并保留其当前状态。其余参考目录条目与两个 HEAD 均保持原快照状态。该入口不参与本轮训练恢复；训练使用项目自有缓存、权重和代码副本，19,722 项运行身份核验全部通过。细节保存在暂停来源记录中。

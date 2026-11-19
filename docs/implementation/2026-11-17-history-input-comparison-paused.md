# mask/dense对照流程暂停与恢复记录

> 后续恢复：用户同日明确要求“继续训练吧”。本轮已从下述6,842步状态恢复；本页保留暂停时的核验结果和SHA，暂停点权重另存为不可变快照，活动latest/best将继续更新。见[恢复记录](2026-11-17-history-input-comparison-resumed.md)。

用户于2026-11-17明确要求“训练先暂停”。已通过本运行STOP_REQUESTED标记，在同步优化步后保存并正常退出；不恢复训练或后续正式评测/对比，直到用户明确要求继续。

| 项目 | 保存结果 |
|---|---|
| 模型 | U形CFM＋actor，mask_lidar历史输入，VAE冻结 |
| 完整保存步数 | **6,842 / 10,000** |
| 剩余优化步数 | **3,158** |
| 暂停完成，北京时间 | 2026-11-17 17:22:17 |
| 训练进程退出码 | **0** |
| 已完成完整tune验证 | 2,000 / 4,000 / 6,000步 |
| 当前最佳EMA | 6,000步，完整tune联合损失1.3568509569 |
| 恢复游标 | epoch=3（零起始），next_batch=586 |
| 学习率调度器 | last_epoch=6,842；总计划10,000步、warmup1,000保持不变 |

## 保存与验证

`latest.pt`包括完整模型、AveragedModel EMA、276项优化器状态、scheduler和双rank的Python/NumPy/torch CPU/CUDA/flow generator RNG。模型、EMA、优化器和scheduler均已成功加载，权重有限；验证期间没有执行优化器更新。

- [6,842步完整状态快照](../../outputs/history_input_comparison/2026-11-17-seed2026/train_mask/pause_step_006842.latest.pt)：SHA-256 `a70d3269e81234cfdfd299da10d3f21c21c9f9cf8d4ec3ec6cbd2b7fe1b2a4f5`。
- [暂停时6,000步最佳EMA快照](../../outputs/history_input_comparison/2026-11-17-seed2026/train_mask/pause_step_006000.best.pt)：SHA-256 `29f9f7576e1696a66883cf9d4c985300a40e7b5446ab709f4d5ccfe607ff3370`。

本轮控制器、torchrun、两rank和DataLoader worker均已退出，无需强制结束，训练占用的显存已释放。停止标记保留；旧密集模型、缓存和raw停止标记保留。

## 后续恢复

已生成但**没有执行**[resume_plan_20261117.json](../../outputs/history_input_comparison/2026-11-17-seed2026/resume_plan_20261117.json)，保存确切训练命令、环境、冻结源码、检查点身份和控制器接续步骤。

获得明确恢复指令后，保留现有日志/运行状态的归档，使用相同冻结源码和完整latest状态恢复；训练命令使用`--resume`，不再重复提供初始化用的`--trunk-checkpoint`。继续沿用总预算10,000步及原学习率计划。停止标记只在明确恢复时归档，不自动移除。

正式评测尚未启动。恢复训练至10,000步后，才继续预先固定的最终EMA导出、密集模型评测、mask模型评测和总体/逐类别报告。

机器记录：[暂停状态](../../outputs/history_input_comparison/2026-11-17-seed2026/paused_by_user_20261117.json)、[检查点校验](../../outputs/history_input_comparison/2026-11-17-seed2026/pause_checkpoint_verified_20261117.json)。

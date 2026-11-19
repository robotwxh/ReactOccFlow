# 密集语义历史U形CFM＋actor：10,000步训练完成

2026-11-17核查：本轮模型优化、最终完整tune验证和检查点保存均已完成，进度为**10,000 / 10,000步，100%**，剩余训练时间为0。

| 项目 | 结果 |
|---|---|
| 模型 | U形CFM＋actor，密集语义历史输入，原OccFM VAE冻结 |
| 模型训练开始，北京时间 | 2026-11-16 20:02:25 |
| 模型训练计算完成，北京时间 | **2026-11-17 05:50:18** |
| 模型训练耗时 | 35,272.71秒，约9小时48分钟 |
| train/tune缓存 | 17,468 / 2,176窗口，均完整 |
| 最佳EMA步数 | 10,000 |
| 完整tune联合损失 | 0.7463341209 |

完整tune损失在2,000/4,000/6,000/8,000/10,000步分别为3.282028、2.018678、1.355934、0.969488、0.746334。该联合损失不是语义mIoU，不能与论文mIoU比较。

产物位于`outputs/reactflow_occfm_unet_dense_history/2026-11-16-seed2026/train/`：

- [best.pt](../../outputs/reactflow_occfm_unet_dense_history/2026-11-16-seed2026/train/best.pt)：10,000步最佳EMA，SHA-256为`aa53211b7ceee01b90e15e60fade2ac59aac4e581e961018ebeab2e37f963d3d`。
- [latest.pt](../../outputs/reactflow_occfm_unet_dense_history/2026-11-16-seed2026/train/latest.pt)：10,000步完整模型、EMA、优化器、scheduler、双rank RNG及游标，SHA-256为`9567dd7e4670eff3db46d822fd2526bb72cddab6999bc378a4c588d2b4192109`。

两个检查点均成功读取，模型与EMA按对应原始模型/AveragedModel结构严格加载，浮点权重有限；scheduler为10,000步，保存了两rank RNG及276个优化器状态项。首次校验把latest的AveragedModel EMA误当作原始模型状态加载，随后按其正确包装结构验证通过；未改动任何权重字节。

## 退出残留及处理

训练日志于05:50记录`finished`后，退出阶段卡住，控制器仍显示running，两张卡各约16GB显存仍被占用，但GPU利用率为0。具体退出等待点尚未完全定位。

核验最终检查点后，仅向本轮已完成的两个rank及其DataLoader worker发送SIGTERM，释放显存。torchrun的实际退出码为1，原始失败运行记录保留；不能将其写成正常退出0。计算已完成与进程退出异常分别记录，`pipeline.json`最终标记为`completed_with_teardown_cleanup`。清理后无本轮活跃训练进程，两卡仅剩约293/283MiB显示相关占用，检查点SHA不变。

证据：[完成记录](../../outputs/reactflow_occfm_unet_dense_history/2026-11-16-seed2026/completion_20261117.json)、[检查点校验](../../outputs/reactflow_occfm_unet_dense_history/2026-11-16-seed2026/checkpoint_verified_20261117.json)、[退出清理](../../outputs/reactflow_occfm_unet_dense_history/2026-11-16-seed2026/teardown_cleanup_20261117.json)。

本轮仅有8个固定tune窗口的原生E0生成探针，**尚未进行官方150场景、4,219窗口的正式预测mIoU评测**。没有启动新训练或扩展10,000步预算。

# 标准DiT512联合训练完成记录

> 同日后续评测已完成：按本文预选的best84,000 EMA，官方150场景/4,219窗口全网格Avg mIoU=18.2074%，高于旧hidden256的2.5755%，仍低于U形CFM＋actor的28.6739%。见 [全量评测记录](2026-11-11-dit512-miou-evaluation.md)。下文“未启动新评测”是清理训练进程时的历史状态。

2026-11-11核查：独立`outputs/reactflow_dit512/seed_2026/`实验于北京时间**2026-11-10 22:32:15**完成100,000/100,000步，达到预设上限；并未触发patience10早停。主干为标准DiT hidden512、depth8、heads8、patch5，与actor联合训练，Swin-VAE冻结。

完整tune共50次，每次2,176窗口。best EMA为**84,000步**，tune loss=0.19324544148902784；100,000步最终tune loss=0.19363859500943345，bad_evaluations=8。最终打印的训练loss=0.11520068906247616。从第50步恢复到结束的计时约75.38小时，包含验证、保存与固定tune生成监控。

- `latest.pt`为100,000步完整状态，含模型、EMA、AdamW、scheduler、双rank RNG、epoch/数据游标、配置和数据身份；scheduler last_epoch=100,000。SHA-256：`4b5950eb0b6817fdd8cec1723ac70600c3fb1eec9f67c670e8f18e262194399c`。
- `best.pt`为84,000步EMA，按完整tune联合损失选取。SHA-256：`f56c2569e6fd982bfe44bd1c8f6894875ec635c8d9c42a09168c75fc59636378`。
- 最后100,000步固定8场景tune原生E0 Avg mIoU=17.752520%；84,000步best对应同一小集为17.332341%。二者均非官方150场景/4,219窗口全网格结果，不能直接与历史CFM或论文指标比较。

最终`finished`日志、检查点元数据和源码153文件身份核对通过。训练结束后19个监督器/rank/DataLoader进程仍残留并占用每卡约10GB显存，具体退出阻塞位置未诊断。确认所有训练计算与状态保存结束后，通过既有torchrun监督器发送SIGTERM清理，19个进程均已停止，CUDA训练显存释放；未修改训练源码或权重。清理后的权重SHA再次匹配，日志中的后续终止消息不表示训练在100,000步前失败。没有自动恢复、重训或启动新评测。

完整元数据、哈希和清理过程见`outputs/reactflow_dit512/seed_2026.completed_20261111.json`；运行/启动/ETA记录均已同步为`finished_max_steps`，剩余训练时间为0。

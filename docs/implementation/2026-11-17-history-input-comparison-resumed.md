# mask/dense对照流程：从6,842步恢复

用户于2026-11-17明确要求“继续训练吧”。控制器于北京时间**17:41:53**重新启动，PID为`241825`；训练日志于**17:42:32**记录`started, global_step=6842`，严格恢复检查通过。

- 模型：U形CFM＋actor，mask_lidar历史输入，原VAE冻结。
- 继续总预算10,000步，恢复时剩余3,158步。
- 恢复模型、EMA、276项优化器状态、scheduler、双rank Python/NumPy/CPU/CUDA/flow RNG以及epoch=3、next_batch=586游标。
- 保留原学习率计划、seed2026、两卡有效batch8和全部训练/评测设置。
- 192份冻结源码与19,786项代码/数据指纹再次校验通过；没有更换训练数据、源码或归一化。

## 暂停点与历史记录保留

完整暂停状态复制为`train_mask/pause_step_006842.latest.pt`，SHA-256为`a70d3269e81234cfdfd299da10d3f21c21c9f9cf8d4ec3ec6cbd2b7fe1b2a4f5`。暂停时best6000复制为`train_mask/pause_step_006000.best.pt`，SHA-256为`29f9f7576e1696a66883cf9d4c985300a40e7b5446ab709f4d5ccfe607ff3370`。均为独立普通文件；活动latest/best可继续更新，暂停记录仍可追溯。

本运行的暂停标记、旧pipeline/控制器记录、控制台日志和原配置移入`history/before_resume_006842_20261117T094153Z/`。完整训练JSONL继续追加。原raw训练STOP标记未移除，其他旧实验没有恢复。

本次从完整状态接续。原初始化来源仍由最初launch配置、暂停前best权重的加载报告及恢复记录关联，不将恢复时临时构造模型的默认标签当作新的训练初始化。

## 后续流程

训练到10,000步后，自动接续预先批准的固定10,000步EMA导出、密集模型全量评测、mask模型全量评测，以及总体和17类IoU差值报告。评测仍为同一150场景/4,219窗口、FP32/NFE10/seed2026协议。

预计剩余模型训练约3小时，后续两轮正式评测/汇总约1–2小时；以实际日志为准。

运行根目录为`outputs/history_input_comparison/2026-11-17-seed2026/`。机器记录：[resumed_from_6842_20261117.json](../../outputs/history_input_comparison/2026-11-17-seed2026/resumed_from_6842_20261117.json)，当前阶段以同目录`pipeline.json`为准。

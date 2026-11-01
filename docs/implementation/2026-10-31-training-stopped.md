# 2026-10-31：停止训练并确认标准 DiT

按用户“停掉所有训练，将 U-Net 改为 DiT”的要求，于上海时间 2026-10-31 15:22:07 停止本项目全部训练。共识别并停止 12 个相关守护、启动、双卡训练和数据加载进程；复查没有剩余项目训练进程，GPU 计算进程列表中不再有训练进程。非训练的桌面/预览服务保留。

旧实验 `outputs/reactflow_joint/seed_2026` 最后日志为 51,400 步。实际读取检查点元数据确认，`latest.pt` 和 `best.pt` 均为 50,000 步，文件保留；50,000 步以后的内存训练状态没有保存。没有把停止操作写成模型完成训练或新的 checkpoint。

停止记录为 [seed_2026.stopped_by_user.json](../../outputs/reactflow_joint/seed_2026.stopped_by_user.json)，运行状态文件 [seed_2026.runtime.json](../../outputs/reactflow_joint/seed_2026.runtime.json) 已标记 `stopped_by_user`、`stage=stopped` 与停止时间，启动守护进程已经退出。

当前标准 DiT 代码已经在之前的迁移中接入，本次重新实际构造 `ReactFlow()` 并检查：默认主干为 `StandardDiT`，没有卷积 U-Net 的上下采样模块；patch=5、hidden=256、depth=8、heads=8，共 1,200 个全局时空 token。三个 `react_flow_*.yaml` 均选择 `backbone=dit`、主干可训练且无 U-Net 初始化权重；正式入口默认 `--backbone dit`。occupancy–actor 双向 cross-attention 与单套 VAE 保持当前设计。

停止操作当时没有启动新训练，原日志、权重与参考仓库继续保留。用户其后要求删除 U-Net，原可运行基线代码现已移除，见 [删除记录](2026-10-31-remove-unet.md)。新 DiT 需要独立训练，旧 U-Net 权重及其步数不用于 DiT 初始化或进度；当前包不提供其加载或续跑入口。

# 2026-11-03：恢复原 CFM 联合模型训练

按用户“用 CFM 架构继续进行训练”的要求，恢复原 `occfm_unet` 世界主干的 ReactFlow joint 实验，保留 occupancy–actor 双向交互、原 CFM/RF 目标及冻结 Swin VAE。标准 DiT 实验继续暂停在完整第 22,000 步；此次仅启动原 CFM 实验。

启动时间为北京时间 2026-11-03 10:19:52。原训练器通过全量检查后，10:21:11 确认从 50,000 步恢复；数据游标读取后进入优化，10:27:03 已记录第 **50,050 步**，训练损失 0.13152140006422997，有限且无报错。两张卡 30 秒平均利用率为 83.27% / 77.57%，峰值 98% / 100%，显存占用约 17,120 / 16,469 MiB。后台训练继续，实际步骤、PID 和检查时间见 [运行健康记录](cfm_training_resume_20261103_health.json)。当前最近完整 checkpoint 仍为 50,000 步；新训练日志不能当作新落盘状态。

## 恢复进度与配置

| 项目 | 内容 |
|---|---|
| 实验 | `outputs/reactflow_joint/seed_2026`，原 U 形时空 DiT，joint，seed=2026 |
| 完整恢复进度 | 第 50,000 步，epoch=22、next_batch=3948 |
| 原计划 | 上限 100,000 步，每 2,000 步完整 tune，连续 10 次无改善提前停止 |
| 数据 | 完整 train 17,468 / tune 2,176 窗口，oracle E0，日志自车动作监督 |
| 双卡 | 两张 RTX A6000，DDP，每卡 batch=2、累积=2，有效 batch=8 |
| 数据/解码 | 每 rank workers=2，decode-chunk=1，FP32；沿用原配置 |
| 学习率 | 原 trunk 初始 2e-5、交互/actor 初始 1e-4；恢复 AdamW 和原 warmup/cosine 当前位置 |
| 最佳历史 tune loss | 0.1905666558832447，第 50,000 步；不是 mIoU、规划或物理质量指标 |

原停止前最后日志为 51,400 步，但完整状态只保存至 50,000 步。之后未保存的计算需重跑，旧日志保留；应按最近一次 `started` 事件识别本次续训，不能用旧日志最大步数作为当前进度。最多剩余 50,000 步，实际结束还受提前停止和训练速度影响。

## 完整状态与源码身份

没有使用标准 DiT 的权重，也没有从随机权重重新开始。恢复模型、EMA、AdamW、scheduler、两 rank RNG、数据游标、最佳验证和提前停止计数。

原检查点绑定迁移前源码的绝对路径。为保持工作区的双架构实现，同时准确继续旧实验，本次把经过 SHA-256 验证的迁移前归档恢复成普通文件训练副本：`outputs/reactflow_joint/source_seed_2026/`，114 个归档文件逐一校验。训练从此目录正常导入自有包，不注入 `sys.path` 或 `PYTHONPATH`，不读取参考项目代码。

原检查点另存为 `outputs/reactflow_joint/seed_2026/stopped_step_050000_20261031.pt`，SHA-256 为 `244208facfb667b6153198a0d407dd6ec2ed9f6b575c96b52261c3696f272edf`。另生成 `resume_step_050000_source_relocated_20261103.pt`，**仅重定位 `data_fingerprint` 内 76 个源码路径键**；摘要值、配置及全部其他状态不变。1,395 次张量比较和 2,866 次其他值比较全部相等，19,726 项运行身份全部核验。原训练器逐字节不变，其严格配置/摘要匹配仍启用，并已用原 `_runtime_fingerprint` 实际验证完整字典相等。

普通文件副本独立导入与真实 checkpoint 严格加载通过，38 个已加载自有模块均来自该副本。参考仓库只读检索记录、来源提交、文件摘要、路径映射和状态比较见 [来源记录](../provenance/cfm_training_resume_20261103.json)。现有标准 DiT 冻结训练副本不变。后续编辑工作区不会改变本次训练副本，避免再次破坏恢复身份。

两参考仓库本次前后快照完全相同，参考 HEAD 均未改变；标准 DiT 的 `--dry-run` 再次核验 82 个冻结源码/检查点文件通过，没有启动该实验。CFM 启动入口的重复启动保护已实际检查，拒绝再次启动同一实验。

## 运行与审计文件

- 日志：`outputs/reactflow_joint/seed_2026/training.jsonl`；原历史日志追加保留。
- 本次 console：`outputs/reactflow_joint/seed_2026.cfm_resume_20261103T101952764948.log`。
- 当前运行状态：`outputs/reactflow_joint/seed_2026.runtime.json`。
- 本次绝对路径命令和环境：`outputs/reactflow_joint/seed_2026.resume_20261103.launch.json`。
- 冻结源码摘要：`outputs/reactflow_joint/source_seed_2026.manifest.json`。
- 恢复前日志、状态、参考保护快照、预检与路径映射：`outputs/reactflow_joint/resume_20261103/`。
- 启动入口：[resume_cfm_training.py](../../scripts/resume_cfm_training.py)。启动前 `--dry-run` 已通过，核验 115 个源码/检查点文件；同一续训已启动时再次运行会拒绝重复启动。

没有启动自动重启守护进程。检查点仍按原计划在完整验证后保存；当前训练器不支持信号触发的即时保存。后续停止时应区分最后日志步数和最近完整保存步数。

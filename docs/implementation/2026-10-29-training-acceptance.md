# 2026-10-29 ReactFlow 正式训练验收与运行记录

本轮已完成全量缓存审计、真实尺寸训练、32 窗口拟合、单卡/双卡恢复及隔离安装验收，并启动 seed=2026 的全量双卡联合训练。正式训练尚未完成，没有给出论文主实验或闭环效果结论。

## 已运行的验收

| 项目 | 实际结果 | 证据 |
|---|---|---|
| 完整 E0 数据 | train 17,468 窗口/560 场景；tune 2,176 窗口/70 场景 | [全量审计](e0_full_cache_audit_2026-10-29.json) |
| 数据完整性 | 全部 19,644 个载荷 SHA-256、record/manifest 身份、canonical settings 摘要、rank 清单、场景互斥和同一 train-only 归一化通过 | 同上；数值 schema 在实际读取时继续校验 |
| 完整测试 | 330 passed in 15.08s | [训练入口验收](training_runner_acceptance.json) |
| 真实 CUDA 更新 | batch=2、VAE 冻结、主干参与训练；276 个可训练张量全部有梯度，有限损失；峰值 13,432,840,704 bytes | [CUDA 记录](fullsize_cuda_step_2026-10-29.json) |
| 200 步单卡训练 | 完整缓存中显式选择 32 个窗口，新增分支和主干共同优化；200 步完成，峰值约 13.7 GB | `outputs/formal_training_acceptance/single_gpu_200/` |
| 固定噪声小样本拟合 | 同一 32 个重叠窗口/2 场景，raw flow probe 4.90875177235 → 0.80470655854，下降 83.6067% | [拟合记录](full_trunk_overfit_2026-10-29.json) |
| 单卡恢复 | 连续 4 步与 2 步保存后再训练 2 步比较：1,392 次张量比较全部逐位相同 | `outputs/formal_training_acceptance/deterministic_resume/training_state_acceptance.json` |
| 双卡 NCCL 恢复 | 每卡 batch=2、累积=2，有效 batch=8；连续 4 步与 2+2 步恢复比较，1,395 次张量比较全部逐位相同 | `outputs/formal_training_acceptance/ddp_resume/equivalence.json` |
| 独立安装 | 新 wheel 63 个条目，专用环境安装；临时目录 `python -I`、参考目录不存在时训练/推理/评价和连续 decoder 梯度通过 | [隔离验收](isolation_training_2026-10-29.json)、[wheel](training_wheel_2026-10-29.json) |

上述 32 窗口训练和 8 窗口 tune 检查只作为工程诊断，不是完整调参集评测或泛化结果。诊断使用 PyTorch Subset，保留原始 complete 缓存与 manifest。

## 修正与数值边界

1. 验证时分别汇总 occupancy FM、actor FM、existence、actor/occupancy 一致性、静态时序、速度一致性和航向对齐的分子与有效计数，再跨 rank 归约。速度与航向有独立分母。避免以 batch 大小近似有效元素数量，训练损失公式、梯度与 RNG 消耗保持原有行为。
2. 恢复已经触发 patience 的检查点时，不再额外优化或重复评价。
3. 丢弃每个打乱 epoch 尾部不足一次完整累积的 microbatch，固定每次更新的有效 batch=8。全量默认设置每 rank 使用 4,366 个 microbatch，尾部丢弃 1 个 microbatch，两卡合计每个 epoch 使用 17,464 个窗口；下一 epoch 重新打乱，全部窗口都保留被抽中的资格。
4. `--stop-after-steps` 只控制保存后暂停，不改变 max_steps、学习率计划或恢复配置。诊断窗口数量写入配置并明确标记 `diagnostic_subset`，正式训练为 `full_training`。
5. 首次 CUDA 严格比较未通过，两次独立运行在暂停前就已有数值差异。复制 OccFM common-utils 快照后，正式 runner 设置 deterministic cuDNN、禁用 benchmark、严格确定性算法与 cuBLAS workspace，并把数值设置绑定到恢复配置。

确定性设置后，模型、EMA、优化器、scheduler、全部 rank 的 RNG 及训练游标均逐位一致。派生验证损失仍存在浮点前向/归约差异：单卡约 4.01549×10⁻⁶，双卡约 1.26236×10⁻⁶。单卡记录的 scalar 容差为 rtol=1.3×10⁻⁶、atol=10⁻⁵；双卡比较通过其更严格的 rtol=10⁻⁶、atol=10⁻⁸。没有宣称验证 scalar 逐位一致。最初非确定性结果和后续 strict scalar 比较失败文件全部保留，没有覆盖成通过。

来源与复审见 [验证统计来源](../provenance/validation_statistics_sources.json)、[确定性来源](../provenance/deterministic_training_sources.json) 以及计划工作目录中的 `formal-runner-review-2026-10-29.md`、`formal-runner-rereview-2026-10-29.md`。

## 当前正式实验

| 设置 | 数值 |
|---|---|
| 随机种子 | 2026 |
| 输入与任务 | oracle E0 表示、日志实际 ego 动作条件下的世界预测监督 |
| 数据 | 完整 train=17,468；完整 tune=2,176 |
| GPU | 2 张 RTX A6000，NCCL DDP |
| batch | 每卡 2，累积 2，有效 8 |
| 优化器 | AdamW；主干 2×10⁻⁵，新增模块 10⁻⁴；weight decay=0.01；裁剪=1 |
| 计划 | 1,000 步 warmup；cosine；最多 100,000 步 |
| EMA 与选择 | EMA=0.9999；每 2,000 步在完整 tune 上评价；连续 10 次未改善停止 |
| 首次保存 | 第 50 步先保存 latest.pt，再由独立控制进程自动按原配置恢复 |
| 输出 | `outputs/reactflow_joint/seed_2026/` |

完整命令、进程信息和两个阶段的 console 日志位置分别记录在：

- `outputs/reactflow_joint/seed_2026.launch.json`
- `outputs/reactflow_joint/seed_2026.runtime.json`
- `outputs/reactflow_joint/seed_2026/training.jsonl`
- `outputs/reactflow_joint/seed_2026/latest.pt`

截至 2026-10-29T11:21:39+08:00，首50步检查点已实际保存，已成功自动恢复，最新训练日志到第100步，完整tune评估尚未到达首次2000步。两张GPU的训练进程仍在运行；恢复检查的19,726项源码/数据摘要全部一致。见 [运行健康记录](formal_training_health_2026-10-29.json)。进程由独立session控制程序管理，日志和检查点保留在项目目录。

## 尚待完成

正式训练收敛、完整研究预测指标、2027/2028 种子、消融和效率实验仍未完成。CARLA/Bench2Drive、反事实响应数据、独立风险校准、实际点云感知训练和闭环实验尚未具备真实运行结果。已有接口或本轮小样本损失下降不替代这些研究证据。

预训练权重接触过官方 700 个训练场景，新 tune/calibration 分组相对预训练不具有严格未见性；这一限制继续保留。参考源码、HEAD 与已有 Git 状态未变，原快照缺少的空 `IR-WM/data/` 目录继续保留，见 [来源保护审计](reference_audit_2026-10-29.json)。

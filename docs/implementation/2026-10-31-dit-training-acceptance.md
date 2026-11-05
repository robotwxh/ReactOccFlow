# 2026-10-31 标准 DiT 训练验收与续推

> 后续状态：2026-11-01 已按用户要求暂停训练，完整检查点为 22,000 步，最后训练日志为 22,550 步。下文保留启动阶段的历史验收；当前状态与恢复命令见 [暂停记录](2026-11-01-dit-training-paused.md)。

用户要求按实施计划继续，随后要求尽量充分使用两张 GPU。本次在删除 U-Net 后的纯 DiT 实现上重新进行训练验收，不使用旧 U-Net 权重或旧训练步数。200 步、32 窗口拟合、实际训练后交互与恢复验收已通过，吞吐设置已选定，新 seed=2026 双卡全量实验于 19:58 启动。

## 数据与实现

- 完整 E0 train：17,468 个窗口、560 个场景；tune：2,176 个窗口、70 个场景。每次正式 runner 启动前重新校验完整缓存的 manifest、settings、record、payload 校验值、场景互斥和同一 train-only 统计。
- 主干：patch=5、hidden=256、depth=8、heads=8，共 1,200 个时空 token；第 4 层后进行 actor self-attention 与双向 cross-attention。
- 仅冻结一套 occupancy VAE；训练读取 latent 缓存，decoder 用于连续预测端点的一致性损失。DiT 与 actor/interaction 从随机初始化共同训练。
- 新输出根目录为 `outputs/dit_training_acceptance/` 与 `outputs/reactflow_dit/`。历史 U-Net 实验仍停止，日志与检查点保留。

## 验收记录

| 项目 | 本轮状态与证据 |
|---|---|
| 200 步、32 窗口训练 | 已完成；固定噪声 probe 3.19560623169 → 1.25981795415，下降 60.5766%；200 步损失、梯度与参数均有限，168 个参数张量均有梯度，冻结 VAE 状态哈希未变。峰值分配显存 8,043,910,144 bytes。详见 `outputs/dit_training_acceptance/single_gpu_200/results.json`。32 个不同 anchor 窗口可能重叠，不能等同于 32 个独立场景。 |
| 双卡精确恢复 | 已通过；每卡 batch=2、累积=2，有效 batch=8；连续 4 步和保存后 2+2 步的模型、EMA、优化器、scheduler、两 rank RNG 与游标逐位一致，共比较 855 个张量。验证 scalar 差约 5.76e-8，不声称 scalar 逐位一致。详见 [恢复验收](dit_only_ddp_resume_acceptance.json)。诊断为 32 train/8 tune 窗口，不作为全量验证。 |
| 优化配置恢复复验 | decode-chunk=4、workers=4 与正式配置一致，重新比较连续 4 步和 2+2 步：855 个状态张量逐位相同，验证 scalar 差约 1.05e-9；见 [优化后恢复验收](dit_optimized_ddp_resume_acceptance.json)。 |
| 训练后交互 | 已通过；用实际 200 步权重，actor 扰动影响 occupancy，occupancy 扰动影响 actor；空 actor mask 有限、2-NFE 采样可复现。测试输入不使用未来标签。详见 [实际权重交互检查](dit_trained_exchange_acceptance.json)。 |
| 当前包与隔离 | 本轮开始前核对删除阶段的 109 个自有文件 SHA-256 无变化。沿用相同代码已通过的 337 项测试及 [独立安装验收](dit_only_isolation.json)，没有将旧模型的测试数混入。 |
| 参考保护 | 正式训练启动后再次审计：OccFM 368 项、IR-WM 389 项均无差异；58 个独立安装包源码文件与训练副本 SHA-256 一致。见 [来源保护](dit_training_source_protection.json)。 |

固定噪声 probe 降低用于检查小样本拟合；不据此声称完整过拟合收敛、泛化质量、反事实反应或闭环效果。

## 双卡吞吐选择

依据用户“尽量把两张卡跑满”的指示，用实际尺寸的 train 中段窗口分别压测每卡 batch=2、4、8 与 decoder chunk=1、4。batch=8 时两卡利用率达 97%–99%，但处理每个窗口更慢。GPU1 上 batch=2/chunk=4 约 2.018 窗口/秒，batch=4/chunk=4 约 1.660，batch=8/chunk=4 约 1.144；选择 batch=2/chunk=4。GPU0 对应 batch=4/chunk=4 约 1.472，batch=8/chunk=4 约 0.998。

正式训练保留每卡 batch=2、累积=2、有效 batch=8，decoder chunk 从 1 调至 4，每 rank 的数据加载进程从 2 调至 4。训练目标、样本范围、LR、warmup 与评价间隔保持计划配置。选择结果为 `outputs/dit_training_acceptance/throughput_selection.json`，原始 JSON、日志与 GPU 采样均保留在同一目录。压测未包含完整数据加载、DDP 和 tune 评价，且有并行工程活动，不作为 T16 效率论文结论。

全量正式 DDP 运行时另连续采样 30 秒：GPU0 平均利用率 90.83%、中位数 96%；GPU1 平均 87.83%、中位数 91%；两卡峰值均为 100%，设备显存约 9.5/9.4 GiB。记录见 [实际双卡采样](dit_formal_gpu_utilization.json)。该数值是当时设备利用率，包含图形/系统活动，不等同于模型理论 FLOPS 比例，也不保证后续数据加载、验证与保存期间保持峰值。

## 正式实验的恢复边界

已将训练所需的 81 个代码、配置及声明文件复制为普通独立文件，保存在 `outputs/reactflow_dit/source_seed_2026/`，逐文件 SHA-256 记录于同级 `source_seed_2026.manifest.json`。没有使用链接、参考目录导入或路径注入。新训练将从该副本自然导入项目包，使用项目自有的缓存和 VAE 绝对路径。

严格恢复会验证全部模型源码、配置、数据与权重身份。后续开发可继续修改工作区，正在训练的副本保持原样。恢复本次实验时，应使用启动记录中的工作目录和完整命令，不能把可变工作区的新源码直接用于旧检查点。

实际配置为双卡有效 batch=8，AdamW、全模型 LR=1e-4、weight decay=0.01、梯度裁剪=1、EMA=0.9999、warmup=1,000、cosine、上限 100,000 步；每 2,000 步进行完整 tune 验证，连续 10 次未改善提前停止。

首 50 步已于 20:01 保存 `latest.pt`，训练游标为 epoch=0、next_batch=100。使用训练副本重新计算的 19,722 项源码、配置、缓存身份和 VAE 权重摘要全部相同，两 rank RNG 状态均已保存；[50 步检查点验收](dit_formal_bootstrap_resume.json) 通过。20:02 已启动从该检查点恢复的长期训练，继续原 100,000 步上限与 warmup/cosine 计划，不使用新的小样本训练状态。另以普通文件保留 `outputs/reactflow_dit/seed_2026/bootstrap_step_000050.pt`，SHA-256 与已验证的第 50 步状态一致，避免后续 `latest.pt` 更新覆盖恢复证据。正式首次完整 tune 验证尚未到达。

实际启动/恢复命令记录于 `outputs/reactflow_dit/seed_2026.launch.json`，进程与阶段记录于同级 `seed_2026.runtime.json`；训练日志为 `outputs/reactflow_dit/seed_2026/training.jsonl`。[运行健康快照](dit_formal_training_health.json) 记录核验时刻，后续实际步数以追加日志为准。使用独立 session 的 torchrun 运行，不设置自动重启守护器。

## 后续边界

本轮为 oracle E0 表示和日志 ego 动作条件下的世界预测训练。其余随机种子、parallel/occupancy-only 对照、正式预测指标、CARLA 反事实采集、独立风险校准、实际点云感知和闭环仍需后续实验。预训练 VAE 曾接触官方 700 个训练场景，不能声称新 tune/calibration 相对该表示严格未见。

来源与操作记录见 [本轮 provenance](../provenance/dit_training_continuation.json)。

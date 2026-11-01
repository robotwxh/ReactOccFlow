# 标准 DiT 世界模型迁移

> 后续状态：用户已要求删除 U-Net，原先在迁移阶段保留的运行分支与加载/评测入口现已移除。当前工程仅提供标准 DiT；本页保留首次迁移的历史验收，最新删除范围与验证见 [U-Net 删除记录](2026-10-31-remove-unet.md)。所有训练仍停止。

2026-10-31，按用户要求将研究世界模型的默认主干从 U-Net/DiT 混合结构改为标准的 isotropic DiT。该变更不代表新模型已经完成训练或优于旧实验。

## 实际结构

`reactoccflow/models/standard_dit.py` 实现空间 patch 线性编码 → 连续堆叠 DiT Block → adaLN 输出层 → unpatchify。所有层的 token 数与宽度保持固定，进行全局时空注意力；没有 U-Net 编码/解码层、下采样、上采样或跨尺度拼接。

默认输入为 6 个历史槽和 6 个未来生成槽，每槽 latent 为 16×50×50。patch=5，因此共有 12×10×10=1,200 个 token。默认 hidden=256、depth=8、heads=8、MLP ratio=4，使用固定空间/物理时间位置编码；生成时间与候选自车轨迹经编码后共同调制 adaLN。输出保持原来的 occupancy 速度场张量契约。默认主干 10,051,728 参数，联合模型 10,878,667 参数。

actor 分支保留当前/历史观测、存在性预测和有效 mask。默认 8 个 actor×6 个未来槽组成 48 个 128 维 token。两层交互在第 4 个 DiT Block 后、同一 token 分辨率下交换信息，然后继续剩余 Transformer Block。VAE 保持独立冻结，仍采用原 Swin 编解码器；本次替换的是世界演化网络。

### Self-attention 与 cross-attention 的分工

occupancy 与 actor 是两条独立 token 序列。联合状态公式 `[Z,A]` 表示两组共同生成的随机变量，不表示将两组 token 拼接后送入同一个 self-attention。

| 注意力 | Q 来源 | K/V 来源 | 默认配置与作用 |
| --- | --- | --- | --- |
| occupancy self-attention | occupancy token | occupancy token | 1,200 个 token、256 维、8 头；8 层 DiT 全局建模空间和时间 |
| actor self-attention | actor token | actor token | 48 个 token、128 维、4 头；每层交互中先更新 actor，使用当前观测有效 mask |
| occupancy → actor cross-attention | actor token | occupancy token | 场景上下文更新 actor |
| actor → occupancy cross-attention | occupancy token | actor token | actor 状态更新场景；无有效 actor 时贡献为零 |

每层交互按“masked actor self-attention → 同步双向 cross-attention → actor FFN”执行。两路 cross-attention 使用同一轮的归一化特征，并经各自可学习门控与残差更新；一方向的结果不会先作为另一方向的输入。occupancy 在交互前 `256→128` 投影，经过两层交互后仅将增量 `128→256` 回投，再加回原 DiT token。此处没有额外的空间/时间分轴注意力：DiT self-attention 直接覆盖全部时空 token。

`joint` 启用上述跨模态交换；`parallel` 保留各自 self-attention 和 actor FFN、关闭跨模态交换；`occupancy_only` 只执行 occupancy DiT。候选 ego 轨迹与生成时间通过 DiT 的 adaLN 条件和 actor 条件编码注入，不是第三组 cross-attention token。

源码对应：`standard_dit.py:StandardDiT`、`joint_transition.py:JointTransition._dit_forward`、`TokenInteraction.forward`。删除 U-Net 时层间交互类统一命名为 `TokenInteraction`；交互位置和参数张量命名保持原 DiT 设计。

### 单套 VAE 与 patch 投影的边界

- 离线缓存：历史 occupancy 与未来训练标签各自经过同一 VAE encoder，以后验均值保存 latent；不是将一次编码输出再送入另一个 VAE。
- 训练 step：直接读取缓存 latent，VAE encoder 调用次数为零。预测未来端点经冻结 decoder 按帧/块解码，连续 logits 用于一致性损失，梯度仍可回传到 latent。
- DiT 内部：每帧 `16×50×50` latent 切成 `5×5` patch，每块 `16×5×5=400` 个数线性投影为 256 维；输出投影和 unpatchify 还原 latent 形状。这些操作不学习第二套 VAE 后验。
- 推理接口：`ReactFlow.sample` 接收历史 latent。如果从 occupancy 开始，需要先用同一 VAE 编码当前/历史观测；ODE 在 latent 空间迭代，结束后按需解码未来 latent。actor 使用独立状态编码，不经过 occupancy VAE，未来真值不进入采样接口。

DiT Block、注意力与位置/条件编码复用已迁入的 OccFM 组件。采用已核对 head 合并顺序的 math 注意力分支；默认启用非重入 activation checkpointing。adaLN 门控和输出预测层使用零初始化，因此初始化时 occupancy 速度为零；应在优化后验证条件依赖，不能用初始化输出判断网络是否接上。原始 DiT 思路见 [作者项目页](https://www.wpeebles.com/DiT)。

## 独立训练与恢复

专用环境为 `/home/lthpc/wxh/envs/ReactOccFlow`。新训练命令：

```bash
torchrun --standalone --nproc_per_node=2 --module scripts.train_react_flow \
  --backbone dit --dit-config configs/dit.yaml \
  --train-cache data/research/e0_full/train \
  --tune-cache data/research/e0_full/tune \
  --output outputs/reactflow_dit/seed_2026 --seed 2026
```

`configs/dit.yaml` 被训练器实际读取；三个 `react_flow_*.yaml` 给出对应模型/消融接口配置。默认 DiT 主干可训练，主干与新增分支初始学习率均为 1e-4。其余训练规范沿用现有入口，包括 complete-cache 审计、DDP、EMA、梯度累积、验证与逐 rank RNG 恢复。

新 DiT 从随机初始化训练；旧 `--trunk-checkpoint` 参数已经删除。`--resume` 只接受相同架构、配置、数据及源码身份的检查点，不能把 U-Net 的训练步数、优化器或 EMA 当作 DiT 的进展。恢复 DiT 时沿用原命令并增加 `--resume <本次DiT输出目录>/latest.pt`。本次删代码会改变源码指纹，因此不能将迁移前短跑检查点直接视为可无条件续跑的检查点。

长期正式训练前仍须完成计划规定的 200 步与 32 窗口训练验收；本次工程检查的范围以验收 JSON 为准。

## 已停止的 U-Net 历史实验

旧 `outputs/reactflow_joint/seed_2026` 的训练及守护进程已于 2026-10-31 15:22（上海时间）按用户要求全部停止。目录、日志和已有检查点保留：最后日志 51,400 步，`latest.pt`/`best.pt` 均为 50,000 步，后续内存进度未保存。新模型的输出默认进入 `outputs/reactflow_dit/seed_<seed>`，不复用旧实验目录；没有自动启动新的 DiT 训练。停止时已再次核验默认模型与三个研究配置均为标准 DiT，见[停止记录](2026-10-31-training-stopped.md)。

迁移前的 114 份包源码、脚本、配置和依赖声明归档于 `outputs/architecture_migration/2026-10-31-standard-dit/legacy_unet_sources.tar.gz`，其内容摘要在同目录 `legacy_unet_sources.json`。迁移时未重装专用环境内的包，旧进程及其 worker 当时保持原环境；这些训练进程现已停止。

旧 runner 的检查点绑定了项目绝对路径和全部源码哈希。发生中断时，直接在修改后的 checkout 上执行旧命令会因身份不匹配而拒绝恢复。需要先用归档恢复对应的旧源码/配置身份，或在单独环境中完成逐文件验证和显式路径迁移后再恢复；本次没有改写旧 checkpoint、关闭身份验证或自动重启旧训练。

## 来源与验收

两个参考目录已在实现前搜索；已有 DiTBlock 和依赖闭包复用项目内的独立复制件，无标准完整 DiT 的部分按检索记录实现。五份新的原始源快照及 SHA-256、源 commit、复用/新增说明位于 `docs/provenance/standard_dit_sources.json`。旧代码版权和未解决的 OccFM 授权限制保留在 `THIRD_PARTY_NOTICES.md`。

结构/条件/梯度/保存加载和三种模式的回归测试位于 `tests/test_standard_dit.py`。本次实际验收结果：

| 检查 | 结果与证据 |
| --- | --- |
| 完整 CPU 测试 | 350 passed，16.43 秒；包含旧基线与新 DiT 回归 |
| 真实尺寸 CUDA | 默认 8 层模型、真实 E0 窗口、带解码一致性损失的 3 次优化；168 个可训练参数张量均有有限梯度，VAE 保持冻结。固定噪声损失 3.2753 → 3.0296 → 2.8739；重复采样一致，新增进程峰值显存 4,027,846,656 字节。见 [CUDA 记录](standard_dit_cuda_acceptance.json) |
| 双卡 NCCL 与恢复 | 8 个真实 train/4 个 tune 窗口；连续 4 步与 2 步保存后恢复再 2 步比较，模型、EMA、优化器、调度器、两 rank RNG 共 855 个张量逐位一致。验证汇总损失存在约 6.5e-8 浮点差异，单独保留。见 [双卡恢复记录](standard_dit_ddp_resume_acceptance.json) |
| wheel 隔离 | 新建临时环境安装 wheel，复制独立真实样本和 VAE 权重，从 `/tmp` 用 `python -I` 完成 CPU 优化、采样和 VAE 解码；所有项目模块加载自该环境，未加载 checkout 或参考目录。未重装旧训练环境。见 [隔离记录](standard_dit_isolation_acceptance.json) |
| 参考保护 | 对照实现前快照：OccFM 368 个条目、IR-WM 389 个条目，文件/元数据、Git HEAD 与工作区状态均无变化。见 [审计记录](standard_dit_reference_audit.json) |
| 旧训练历史快照 | 迁移验收时旧进程仍存活，日志为 51,100 步，见 [当时快照](standard_dit_legacy_health.json)；其后已按用户要求在日志 51,400 步时停止，当前状态以[停止记录](2026-10-31-training-stopped.md)为准 |

CPU 单元测试最终命令为 `python -m pytest -q`。CUDA、双卡恢复及隔离检查的可复查脚本和日志保存在 `outputs/architecture_migration/2026-10-31-standard-dit/`。DiT 的短跑不替代原计划的 200 步及 32 窗口验收，也不能作为世界预测质量、反应式或闭环收益结论。

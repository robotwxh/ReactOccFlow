# ReactOccFlow

面向候选自车动作的 occupancy–actor 联合 Flow Matching 研究工程。2026-10-31 起，研究模型使用标准 DiT：patch 编码、固定分辨率全局时空 Transformer、输出还原，并在 Transformer 层间加入 actor 双向交互。新 DiT 的 200 步、32 窗口拟合与双卡恢复验收已通过，独立的 seed=2026 全量双卡训练已启动，见[本轮训练记录](docs/implementation/2026-10-31-dit-training-acceptance.md)。旧 U-Net 实验继续停止，其权重不适用于 DiT；旧实验停止时日志到 51,400 步，保留的最近检查点为 50,000 步，详见[停止记录](docs/implementation/2026-10-31-training-stopped.md)。反事实采集、风险校准及实际感知/闭环结果仍待完成。

架构、训练命令、旧实验边界和验收见 [标准 DiT 迁移说明](docs/implementation/2026-10-31-standard-dit.md)。

按用户后续要求，U-Net 实现、运行分支、专用配置、旧 CFM 包装器和基线命令已从当前工程移除，仅保留标准 DiT。旧实验与来源快照只作历史记录，见 [U-Net 删除记录](docs/implementation/2026-10-31-remove-unet.md)。

当前网络采用“模态内 self-attention＋模态间双向 cross-attention”：1,200 个 occupancy token 在 8 层 DiT 中进行全局时空 self-attention；48 个 actor token 使用 masked self-attention；第 4 层 DiT 后的两层交互分别以 actor 为 Q、occupancy 为 K/V 更新 actor，以 occupancy 为 Q、actor 为 K/V 更新 occupancy。两组 token 保持独立，没有拼接成一条共同做 self-attention 的序列。

只使用一套冻结的 occupancy VAE。缓存构建时分别编码历史观测与未来训练标签；训练 step 直接读取缓存 latent，并用 VAE decoder 计算连续端点一致性。DiT 的 patch 线性投影及 unpatchify 不构成第二次 VAE 编解码，actor 状态也不经过 occupancy VAE。

实施计划：[完整计划](docs/superpowers/plans/2026-10-27-reactoccflow-implementation-plan.md)。科学问题：[问题与解决机制](ReactOccFlow科学问题与解决机制.md)。进展与已知限制见 [实施记录](docs/implementation/progress.md)。

## 已实现

- 独立 Python 包、专用环境、参考项目快照与复制来源记录。
- 独立 occupancy VAE 编解码、标准 DiT occupancy–actor 联合 CFM；联合模型使用完整 state_dict/EMA 保存与恢复。
- future-only ODE 采样、局部随机数生成器、训练损失及反向传播。
- 缓存迁移为 NumPy 分片和场景/token 索引；历史与未来标签分离。
- occupancy/actor/风险评价接口；旧基线的已完成评测报告按历史保留，运行旧模型的命令已删除。
- 官方 JSON 轻量索引、当前 E0 下 actor/ego 标签、3D 语义与可见性 warp 后重新编码。
- 官方训练场景按固定哈希分为 560/70/70，官方 val150 场景保留评测；仅训练观测拟合归一化。

## 运行

本机专用环境：`/home/lthpc/wxh/envs/ReactOccFlow`。代码和资源不依赖两个参考项目的运行时路径。

```bash
conda activate /home/lthpc/wxh/envs/ReactOccFlow
python -m reactoccflow doctor
python -m reactoccflow inspect-cache data/latents/validation
python -m pytest -q
python -m reactoccflow audit-references
```

旧 `history`／`future` 模型的报告属于带日志真实 ego 动作条件的历史预测实验，不是规划效果或新 DiT 的结果；其专用命令和模型加载接口已删除。E0 研究管线继续使用独立标签和缓存。

## 安装和资源

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1
python -m pip install .
```

GPU 依赖采用 PyTorch 2.7.1+cu128；实际安装清单见 `requirements-lock.txt`。CUDA 软件环境与硬件检查在 `docs/implementation/environment.json`。

资源结构：

```text
assets/checkpoints/occfm_vae.ckpt
assets/checkpoints/occfm_history.ckpt
assets/checkpoints/occfm_future.ckpt
data/latents/training/{manifest.json,trajectory.npy,latents-*.npy}
data/latents/validation/{manifest.json,trajectory.npy,latents-*.npy}
```

真实数据在独立数据根目录配置，不通过参考仓库内的数据链接访问。导入已获授权的可信本地 pickle 缓存时使用：

```bash
python -m reactoccflow convert-cache input.pkl data/latents/new_split \
  --info scene_infos.pkl --shard-size 512
```

默认拒绝重复 token；已有验证缓存含一个重复 token，按原加载器首次出现规则显式使用 `--duplicate-policy first`，所有忽略记录写入缓存 manifest。

## 参考项目保护

`OccFM/`、`IR-WM/` 均为只读参考目录。每项功能先查找、后复制、再在自有包内修改。不使用软链接、硬链接、参考目录导入或配置继承。不删除、改名或清理参考仓库。复制来源与资源校验值保存在 `docs/provenance/`。许可证状态见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 原始数据与研究管线

用户数据根目录为 `/home/lthpc/data`（即 `~/data`）。当前使用 `nuscenes/v1.0-trainval` 官方表及 `nuscenes/gts` Occ3D 标签；路径集中记录在 `configs/data.yaml`。原始数据只读，标签写入项目 `data/research/`。

```bash
python -m scripts.prepare_research
python -m scripts.visualize_research
python -m scripts.build_e0_examples --device cuda:0
```

第一条构建按 train/tune/calibration/evaluation 分开的 actor/ego 标签与 train-only 归一化。第二条输出至少20个不同训练场景的离线 oracle 标签图。第三条构建这20个示例的E0重新编码验收缓存，**不是全量E0训练缓存**。可视化为了覆盖转弯等情况使用离线标签挑选例子，不影响模型的当前观测 actor 选择规则。

旧基线的完整/中途评测状态与指标保留在原输出和文档中，不迁移为新 DiT 成绩。移除旧评测入口后，本项目不再提供运行其检查点的命令；后续 DiT 质量与效率需要独立实测。

研究阶段的 GT occupancy/actor 为 **oracle 表示实验**；尚未实现实际 LiDAR 感知。未来目标仅在 `target`，推理观测不读取未来帧。后续6个关键帧是标称0.5–3秒，实测时间保存在 `future_time_s`，不伪造严格等间隔。

预训练 OccFM 权重已接触官方700个训练场景。因此新划出的70调参/70校准场景不能称为这些权重从未见过的数据；严格独立校准需另行按560场景重训/控制预训练暴露。此阶段没有校准效果或反应式因果结论。

## ReactFlow、缓存与计划边界

研究分支接口位于 `reactoccflow/models/react_flow.py`，连续 endpoint 一致性约束位于 `reactoccflow/losses/consistency.py`。真实 E0 标签的全量缓存由下面的命令生成；双 rank 任务完成并写入 `complete: true` 前，读取器默认拒绝缓存：

```bash
python -m scripts.cache_research --device cuda:0 --rank 0 --world-size 2 --output data/research/e0_full/train
python -m scripts.cache_research --device cuda:1 --rank 1 --world-size 2 --output data/research/e0_full/train
```

train 的 17,468 个窗口与 tune 的 2,176 个窗口已全部生成并通过全量载荷与 rank 清单审计。历史 U-Net 的 200 步、32 窗口、恢复和 330 项测试证据见 [原训练验收](docs/implementation/2026-10-29-training-acceptance.md)，该训练现已停止；新 DiT 的验收单独记录。T8 的分支协议、T9/T11 的指标接口和 T10 的候选规划器不会替代 CARLA 反事实采集、独立校准集或闭环评测。

标准 DiT 正式入口使用以下命令，默认从随机初始化训练，继续使用独立的冻结 VAE 和完整 E0 缓存。新输出目录与旧 U-Net 实验分开：

```bash
torchrun --standalone --nproc_per_node=2 --module scripts.train_react_flow \
  --backbone dit --dit-config configs/dit.yaml \
  --train-cache data/research/e0_full/train \
  --tune-cache data/research/e0_full/tune \
  --output outputs/reactflow_dit/seed_2026 --seed 2026
```

诊断可显式同时设置 `--diagnostic-train-windows 32 --diagnostic-tune-windows 8`，保留原 complete 缓存不变，并在配置中标记工程子集。`--stop-after-steps N` 会保存后暂停，恢复时沿用原学习率计划。

DiT 恢复时增加 `--resume outputs/reactflow_dit/seed_2026/latest.pt` 并保留相同配置与源码身份。本机本轮实际运行使用 `outputs/reactflow_dit/source_seed_2026/` 的独立普通文件副本，每卡 batch=2、累积=2、decode-chunk=4、workers=4；恢复工作目录与完整绝对路径命令以 `outputs/reactflow_dit/seed_2026.launch.json` 为准。后续工作区改动不进入该训练副本。入口仅接受 `--backbone dit`，旧的主干权重加载参数、U-Net 分支和基线命令已删除。历史 U-Net 检查点不能由当前包加载或续跑，其日志、权重及源码归档保留作审计与实验记录。

## Observed-only perception frontend

`reactoccflow/perception/` now contains a CPU reference frontend for current/past LiDAR and detector outputs. It validates E0 transforms and timestamps, builds unclamped voxel/BEV count-height-intensity features, and associates observed actor boxes with deterministic masks and IDs. The bundle intentionally stops before learned occupancy logits; see [perception notes](docs/implementation/perception.md). Oracle labels and future fields are rejected by the inference boundary.

## Bridge and control boundary

`reactoccflow/simulation/bridge.py` and `controller.py` define the versioned two-process message and control boundary. They reject future/GT payloads, stale episodes and invalid arrays, interpolate selected plans on measured timestamps, and use explicit deceleration fallback reasons. They do not start CARLA or claim closed-loop performance.

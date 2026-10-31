# ReactOccFlow

面向候选自车动作的 occupancy–actor 联合 Flow Matching 研究工程。当前已实现独立基线、完整 E0 训练缓存、联合生成和连续一致性损失，并启动 seed=2026 的正式双卡训练；反事实采集、风险校准及实际感知/闭环结果仍待完成。

实施计划：[完整计划](docs/superpowers/plans/2026-10-27-reactoccflow-implementation-plan.md)。科学问题：[问题与解决机制](ReactOccFlow科学问题与解决机制.md)。进展与已知限制见 [实施记录](docs/implementation/progress.md)。

## 已实现

- 独立 Python 包、专用环境、参考项目快照与复制来源记录。
- OccFM VAE 编解码、历史轨迹与未来轨迹条件 CFM，显式 state_dict/EMA 权重选择。
- future-only ODE 采样、局部随机数生成器、训练损失及反向传播。
- 缓存迁移为 NumPy 分片和场景/token 索引；历史与未来标签分离。
- 单窗口与全验证集评测，standard/legacy 指标分开；按窗口身份固定噪声、数据内容哈希和原子断点续跑。
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
python -m reactoccflow baseline \
  --data-root /home/lthpc/data/nuscenes \
  --device cuda:0 --nfe 10 --seed 2026 \
  --output outputs/baseline_initial
```

最后一条命令仅评价单个窗口，用于检查链路，不是完整验证集结果。两种兼容模型均沿用原缓存中每帧的下一步真实 ego 位移：`history` 最后一个条件含当前到下一帧的信息；`future` 还包含未来帧的真实位移。它们是带真实动作条件的预测基线，不能声称无未来信息的实际规划性能。兼容缓存保留原逐帧 ego 坐标；E0 研究管线使用独立标签与重新编码。

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

```bash
python -m reactoccflow.evaluation.baseline   --data-root /home/lthpc/data/nuscenes --variant history   --device cuda:0 --batch-size 8 --decode-chunk 8   --output outputs/baseline_validation/history_v2
# future 模型对应 --variant future 和独立输出目录。
```

同目录和完全相同设置会续跑；cache/label/checkpoint 内容变化会拒绝续跑。`metrics.json` 的 `status` 和 `processed_windows/total_windows` 决定是否全量完成，不能把中途输出当最终成绩。当前并行工程运行时记录的耗时不是隔离的公平性能实验。

研究阶段的 GT occupancy/actor 为 **oracle 表示实验**；尚未实现实际 LiDAR 感知。未来目标仅在 `target`，推理观测不读取未来帧。后续6个关键帧是标称0.5–3秒，实测时间保存在 `future_time_s`，不伪造严格等间隔。

预训练 OccFM 权重已接触官方700个训练场景。因此新划出的70调参/70校准场景不能称为这些权重从未见过的数据；严格独立校准需另行按560场景重训/控制预训练暴露。此阶段没有校准效果或反应式因果结论。

## ReactFlow、缓存与计划边界

研究分支接口位于 `reactoccflow/models/react_flow.py`，连续 endpoint 一致性约束位于 `reactoccflow/losses/consistency.py`。真实 E0 标签的全量缓存由下面的命令生成；双 rank 任务完成并写入 `complete: true` 前，读取器默认拒绝缓存：

```bash
python -m scripts.cache_research --device cuda:0 --rank 0 --world-size 2 --output data/research/e0_full/train
python -m scripts.cache_research --device cuda:1 --rank 1 --world-size 2 --output data/research/e0_full/train
```

train 的 17,468 个窗口与 tune 的 2,176 个窗口已全部生成并通过全量载荷与 rank 清单审计。200 步、32 个重叠窗口拟合、单/双卡恢复、隔离安装及 330 项测试通过。正式 seed=2026 已启动，尚无正式收敛或主实验结论，详见 [本轮训练验收](docs/implementation/2026-10-29-training-acceptance.md)。T8 的分支协议、T9/T11 的指标接口和 T10 的候选规划器不会替代 CARLA 反事实采集、独立校准集或闭环评测。

正式入口使用以下命令；当前正在运行的实验、实际命令和进程状态以 `outputs/reactflow_joint/seed_2026.launch.json`、`seed_2026.runtime.json` 为准。已有实验恢复时增加 `--resume outputs/reactflow_joint/seed_2026/latest.pt`，保留其余配置与数据/源码身份。

```bash
torchrun --standalone --nproc_per_node=2 --module scripts.train_react_flow \
  --train-cache data/research/e0_full/train \
  --tune-cache data/research/e0_full/tune \
  --output outputs/reactflow_joint/seed_2026 --seed 2026
```

诊断可显式同时设置 `--diagnostic-train-windows 32 --diagnostic-tune-windows 8`，保留原 complete 缓存不变，并在配置中标记工程子集。`--stop-after-steps N` 会保存后暂停，恢复时沿用原学习率计划。

## Observed-only perception frontend

`reactoccflow/perception/` now contains a CPU reference frontend for current/past LiDAR and detector outputs. It validates E0 transforms and timestamps, builds unclamped voxel/BEV count-height-intensity features, and associates observed actor boxes with deterministic masks and IDs. The bundle intentionally stops before learned occupancy logits; see [perception notes](docs/implementation/perception.md). Oracle labels and future fields are rejected by the inference boundary.

## Bridge and control boundary

`reactoccflow/simulation/bridge.py` and `controller.py` define the versioned two-process message and control boundary. They reject future/GT payloads, stale episodes and invalid arrays, interpolate selected plans on measured timestamps, and use explicit deceleration fallback reasons. They do not start CARLA or claim closed-loop performance.

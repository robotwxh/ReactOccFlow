> 2026-11-15：已按用户要求撤回原始 LiDAR 输入及其后续改造，恢复 **语义 occupancy → 冻结原 VAE → latent／actor 状态缓存 → U形 CFM 或标准 DiT512**。新增点云感知/actor 检测/从头分阶段训练路线已移出活动代码，历史产物保留。**全部训练继续暂停，不自动恢复。**详见 [回退记录](docs/implementation/2026-11-15-rollback-raw-input.md)。下文运行状态按日期保留为历史。

# ReactOccFlow

> 2026-11-11 差距诊断：当前hidden512已无旧秩瓶颈。固定8个tune场景中，actor→occupancy注入倍率1/0.5/0对应DiT mIoU=17.3353/19.2051/20.5856%，U形为29.1259/29.0881/28.9461%；发现当前DiT注入退化、速度场与语义误差及latent loss失配。预训练与粗patch结构的独立贡献尚未分摊。所有数值为诊断小集原生E0口径，不替代下方官方成绩；没有改权重或重训。见 [完整诊断与修复优先级](docs/implementation/2026-11-11-dit512-cfm-gap-analysis.md)。

> 2026-11-11 全量评测完成：标准DiT hidden512＋actor的best84,000 EMA在官方150场景/4,219窗口上的1/2/3秒全网格mIoU为21.1581 / 17.7828 / 15.6813%，Avg **18.2074%**；相对hidden256提高15.6320个百分点，仍低于U形CFM＋actor的28.6739%和OccFM论文未来轨迹条件28.49%。训练、生成及汇总均已完成，历史权重和结果完整保留。见 [评测记录与完整对照](docs/implementation/2026-11-11-dit512-miou-evaluation.md)。下方按日期保留历史状态。

> 2026-11-11 最新核查：修正后的标准DiT512＋actor已于11月10日22:32跑满100,000步；best EMA为84,000步，完整tune loss=0.19324544，最终100,000步固定8场景tune Avg mIoU=17.7525%（非官方全量成绩）。完整权重与状态已保存，结束后残留进程及CUDA占用已清理，无训练继续运行。见 [完成记录](docs/implementation/2026-11-11-dit512-training-completed.md)。

> 2026-11-07 修复进展：标准DiT配置已改为patch5/hidden512，并在构造时拒绝低于400维的直接线性速度头；1,200个token、8层、actor交互与冻结VAE保持原设计。新增分项tune损失及固定tune场景自由生成监控。512维独立全量实验已完成50步保存并恢复双卡训练，最终mIoU待测；旧256维模型的2.5755%为历史结果。验收与实验状态见 [修复记录](docs/implementation/2026-11-07-dit-rank-repair.md)。

> 原256维实验结果（2026-11-07）：标准DiT joint完成100,000步训练及官方150场景/4,219窗口生成评测。best100,000 EMA的1/2/3秒全网格mIoU为2.8361 / 2.5240 / 2.3663%，平均**2.5755%**；大幅低于上次actor+U形CFM的28.6739%和OccFM论文未来轨迹条件28.49%。实现、EMA、计数和单样本源码一致性核查未发现明确评测装配错误，具体低分原因尚未确定。训练及评测均已结束，见 [完整评测与低结果核查](docs/implementation/2026-11-07-standard-dit-miou-evaluation.md)。

> 修复前诊断：已发现原patch5/16通道/hidden256配置的输入400→256与速度头256→400瓶颈；实际输出秩256，144个方向的初噪无法被固定速度场控制，投影守恒检验通过。8个事先选定tune场景中，真值latent重建mIoU93.68%，仅加这部分噪声即降至4.77%；终点去噪仍不能修好当前模型。详见 [结构与差距诊断](docs/implementation/2026-11-07-dit-cfm-gap-analysis.md)。这更新了此前“尚未定位”的判断，但未完成官方26.10个百分点差距的逐因素归因；本次未修改架构或重训。

原U形CFM已在72,000步提前停止，其52,000步best EMA完成官方val150/4,219窗口评测，未来帧全网格Avg mIoU=28.67%（论文未来轨迹条件28.49%）；条件与表示差异见 [mIoU评测记录](docs/implementation/2026-11-04-reactflow-miou-evaluation.md)。该实验和评测结果保留。

产物统一保存在`outputs/`：[OccFM基线评测与核查](outputs/audits/README.md)、[项目A4概览](outputs/pdf/ReactOccFlow_项目概览_A4.pdf)、标准DiT参数与训练状态`outputs/reactflow_dit/seed_2026/`。2026-11-05已将旧`output/`全部196个文件完整迁移并移除空目录，六份全量预测与FVD特征均保留，详见 [迁移记录](docs/implementation/2026-11-05-output-directory-migration.md)。

面向候选自车动作的 occupancy–actor 联合 Flow Matching 研究工程。2026-11-02 按最新要求恢复原 OccFM 式 U 形时空 DiT 为默认主干，同时保留标准全局时空 DiT。两种主干均支持 joint、parallel、occupancy_only，使用相同 RF 目标和单套冻结 VAE。

| 主干选择 | 结构与 actor 交互 | 配置 |
|---|---|---|
| `occfm_unet`（默认） | 多尺度 down/up/skip；原瓶颈后两层双向 cross-attention | `configs/unet.yaml`、`configs/react_flow_joint.yaml` |
| `dit` | 1,200 token、8 层；第 4 层后两层双向 cross-attention | `configs/dit.yaml`、`configs/react_flow_joint_dit.yaml` |

恢复范围、命令、历史检查点边界和验证见 [双架构恢复说明](docs/implementation/2026-11-02-restore-unet-and-keep-dit.md)。之前的 [标准 DiT 迁移](docs/implementation/2026-10-31-standard-dit.md)及 [U-Net 删除](docs/implementation/2026-10-31-remove-unet.md)保留为历史记录；删除时的单架构限制已被本次要求替代。

两条分支均采用 actor self-attention 与 occupancy–actor 双向 cross-attention；两组 token 保持独立，没有拼接成共同 self-attention 序列。U形实验已在72,000步提前停止；两轮标准DiT均完成100,000步及全量生成评测，hidden256的Avg mIoU为2.5755%，修复后hidden512为18.2074%，仍低于U形模型的28.6739%。反事实采集、风险校准及实际感知/闭环结果仍待完成。

只使用一套冻结的 occupancy VAE。缓存构建时分别编码历史观测与未来训练标签；训练 step 直接读取缓存 latent，并用 VAE decoder 计算连续端点一致性。DiT 的 patch 线性投影及 unpatchify 不构成第二次 VAE 编解码，actor 状态也不经过 occupancy VAE。

实施计划：[完整计划](docs/superpowers/plans/2026-10-27-reactoccflow-implementation-plan.md)。科学问题：[问题与解决机制](ReactOccFlow科学问题与解决机制.md)。进展与已知限制见 [实施记录](docs/implementation/progress.md)。

论文实验：[Baseline 与实验对照方案](ReactOccFlow论文Baseline实验方案.md)，包含外部论文、同骨干对照、评测协议、风险校准和结果表模板。

## 已实现

- 独立 Python 包、专用环境、参考项目快照与复制来源记录。
- 独立 occupancy VAE 编解码、U 形/标准 DiT occupancy–actor 联合 CFM；联合模型使用完整 state_dict/EMA 保存与恢复。
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

train 的 17,468 个窗口与 tune 的 2,176 个窗口已全部生成并通过全量载荷与 rank 清单审计。历史 U-Net 的 200 步、32 窗口、恢复和 330 项测试证据见 [原训练验收](docs/implementation/2026-10-29-training-acceptance.md)，此次已从其 50,000 步完整状态续训；标准 DiT 的验收单独记录。T8 的分支协议、T9/T11 的指标接口和 T10 的候选规划器不会替代 CARLA 反事实采集、独立校准集或闭环评测。

原 U 形主干的新实验使用 `--backbone occfm_unet --unet-config configs/unet.yaml`，默认输出 `outputs/reactflow_occfm_unet/seed_<seed>`。修正后的512维标准 DiT 使用以下配置，从随机初始化训练，继续使用独立的冻结 VAE 和完整 E0 缓存。新输出目录与旧256维DiT、U形实验分开；目录已有实验时应使用对应冻结源码及记录的恢复命令：

```bash
torchrun --standalone --nproc_per_node=2 --module scripts.train_react_flow \
  --backbone dit --dit-config configs/dit.yaml \
  --train-cache data/research/e0_full/train \
  --tune-cache data/research/e0_full/tune \
  --output outputs/reactflow_dit512/seed_2026 --seed 2026 \
  --batch-size 2 --grad-accum 2 --workers 4 --decode-chunk 4 \
  --probe-tune-windows 8 --probe-nfe 10
```

诊断可显式同时设置 `--diagnostic-train-windows 32 --diagnostic-tune-windows 8`，保留原 complete 缓存不变，并在配置中标记工程子集。`--stop-after-steps N` 会保存后暂停，恢复时沿用原学习率计划。

原256维标准DiT训练使用 `outputs/reactflow_dit/source_seed_2026/` 的独立普通文件副本，每卡batch=2、累积=2、decode-chunk=4、workers=4；实际命令见 `outputs/reactflow_dit/seed_2026.launch.json`。该轮已于2026-11-07跑满100,000步，不重复执行恢复入口。512维修正版的独立归档位于 `outputs/reactflow_dit512/source_seed_2026/`，验收和当前训练状态见 [修复记录](docs/implementation/2026-11-07-dit-rank-repair.md)。入口接受 `--backbone occfm_unet|dit`；仅U形主干支持显式 `--trunk-checkpoint` 初始化。历史joint权重须由匹配架构和历史源码严格加载；完整续训还需要原配置、源码、优化器和RNG身份，不能直接把旧checkpoint交给已修改的工作区runner。

本次 CFM 续训使用 `outputs/reactflow_joint/source_seed_2026/` 的原始归档副本，完整命令以 `outputs/reactflow_joint/seed_2026.resume_20261103.launch.json` 为准。源码摘要与检查点中仅源码路径键的显式重定位均已核验，严格恢复检查保留；后续工作区修改不进入该训练副本。原单模型 baseline CLI 仍不加入当前开发包。

## Observed-only perception frontend

`reactoccflow/perception/` now contains a CPU reference frontend for current/past LiDAR and detector outputs. It validates E0 transforms and timestamps, builds unclamped voxel/BEV count-height-intensity features, and associates observed actor boxes with deterministic masks and IDs. The bundle intentionally stops before learned occupancy logits; see [perception notes](docs/implementation/perception.md). Oracle labels and future fields are rejected by the inference boundary.

## Bridge and control boundary

`reactoccflow/simulation/bridge.py` and `controller.py` define the versioned two-process message and control boundary. They reject future/GT payloads, stale episodes and invalid arrays, interpolate selected plans on measured timestamps, and use explicit deceleration fallback reasons. They do not start CARLA or claim closed-loop performance.

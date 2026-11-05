# ReactOccFlow 实施进展

> 2026-11-05 17:18 最新指示：按用户“继续训练吧”从完整54,000步恢复标准DiT joint双卡实验，原训练器已严格恢复模型、EMA、优化器、scheduler、双rank RNG、数据游标及早停计数，并记录本轮`started`。保持原固定源码及100,000步/patience=10计划，VAE冻结；暂停前未落盘的约600步重跑。详见 [本次续训记录](2026-11-05-standard-dit-training-resumed.md)。下方暂停描述为此前状态。

> 2026-11-05 最新指示：用户要求先暂停并保存恢复进度。标准DiT joint的19个训练及加载进程已全部退出，GPU训练显存释放；完整状态另存至54,000步（epoch24、next_batch3216），best tune loss=0.5647131986610445。最后日志54,600步，54,000步后的未落盘更新恢复时重跑。19,722项运行身份及模型/EMA/优化器/scheduler加载核验通过，手动入口dry-run通过；不自动重启。详见 [暂停与恢复记录](2026-11-05-dit-training-paused.md)。以下运行状态按日期保留为历史。

> 2026-11-04 16:41 最新训练决定：按用户要求恢复标准DiT joint实验，从完整22,000步继续双卡训练；原配置、优化器、EMA、scheduler与双rank RNG一并恢复，19,722项运行身份严格匹配。主干为256维/8层/8头/patch5，VAE冻结，沿用100,000步上限和patience=10。详见 [标准DiT续训记录](2026-11-04-standard-dit-training-resumed.md)。原U形CFM已结束及其mIoU结果保留，下方“DiT仍暂停”均为此前历史状态。

> 2026-11-04 最新实测：原CFM joint在72,000步因patience=10提前停止，best EMA为52,000步。官方validation150场景、4,219窗口完整生成评测通过：原未来帧全网格mIoU1/2/3秒=33.4676 / 28.0838 / 24.4704%，平均28.6739%；对OccFM论文未来轨迹条件28.49%为+0.1839个百分点。1秒低于论文、2/3秒高于论文；当前条件和表示不同，不据此声称受控结构提升。E0已知区域平均30.5889%仅作原生口径诊断。详见 [评测记录](2026-11-04-reactflow-miou-evaluation.md)。当前没有训练或评测GPU进程，标准DiT仍暂停。

> 2026-11-03 最新训练决定：按用户要求恢复原 CFM（`occfm_unet`）joint 实验，从完整 50,000 步继续双卡训练，上限 100,000 步，沿用原提前停止规则。标准 DiT 保持暂停。原源码归档与完整训练状态已校验，使用独立普通文件副本严格恢复。详见 [CFM 续训记录](2026-11-03-cfm-training-resumed.md)。下面的暂停、删除和切换记录按日期保留为历史。

> 2026-11-02 最新架构决定：恢复原 OccFM 式 U 形 DiT 为默认主干，同时保留 `backbone=dit` 的标准 DiT。两个分支均支持三种研究模式；原删除限制已被本次决定替代。训练继续暂停，既有 DiT 冻结源码和恢复入口保持不变。详见 [双架构恢复说明](2026-11-02-restore-unet-and-keep-dit.md)。


更新日期：2026-11-04。依据[完整实施计划](../superpowers/plans/2026-10-27-reactoccflow-implementation-plan.md)。本文件只记录已经运行并留有证据的工程结果；计划中的研究指标、反事实响应、校准和闭环结果不会提前写成结论。

## 2026-11-01：按用户要求暂停 DiT 训练并保留恢复状态

当前全部训练暂停。标准 DiT seed=2026 最后日志为 22,550 步，实际完整检查点为 22,000 步；最佳完整 tune 损失 0.7826553602260875。19 个训练、加载和启动进程已停止，训练 GPU 显存已释放。模型、EMA、优化器、scheduler、双卡 RNG 与游标均保存，19,722 项恢复身份摘要完全匹配，并以普通文件另存完整检查点。当前训练器不支持运行中信号保存，22,000 步之后未落盘的区间需重跑；原日志保留。已提供仅手动执行的恢复入口，dry-run 通过，没有重启训练。具体路径与命令见 [暂停与恢复记录](2026-11-01-dit-training-paused.md)。以下运行描述均为此前阶段的历史记录。

## 2026-10-31：按计划继续标准 DiT 训练

根据用户最新指示，纯 DiT 的 200 步、32 窗口小样本训练与有效 batch=8 的双卡断点恢复验收已通过。固定噪声 probe 从 3.19560623169 降到 1.25981795415，下降 60.5766%；855 个恢复状态张量逐位一致，实际训练后双向交互与空 mask 检查通过。按用户充分使用两卡的要求比较 batch=2/4/8 后，选择吞吐最高的 batch=2、decode-chunk=4，保持累积=2、有效 batch=8，并将 workers 调至每 rank 4；该配置也重新通过双卡精确恢复。独立全量 seed=2026 实验已于 19:58 启动，20:01 保存第 50 步完整状态，19,722 项恢复身份摘要全部匹配，20:02 启动恢复续训；首次完整 tune 验证尚未到达。实际双卡 30 秒平均利用率约 91%/88%，峰值均为 100%。新训练使用独立输出目录及普通文件代码副本，旧 U-Net 实验继续停止。当前证据和恢复边界见 [DiT 训练验收](2026-10-31-dit-training-acceptance.md)。下方的停止记录、旧模型训练数值及迁移初期测试数均保留为历史记录。

## 2026-10-31：删除 U-Net 运行代码

按后续指示，U-Net 主干、分支、专用配置、旧 CFM 包装器与基线训练/评价入口已从当前工程移除，仅保留标准 DiT 和独立 VAE。通用流插值、时间网格及优化诊断工具已解耦保留，`TokenInteraction` 继续负责 actor self-attention 和双向 cross-attention。旧训练保持停止，历史输出与来源快照保留；本次没有启动模型训练。删除范围与最终验证见 [删除记录](2026-10-31-remove-unet.md)。下面的迁移阶段“保留兼容入口”描述现已由本次删除决定替代。

## 2026-10-31 15:22：按用户要求停止全部训练

旧 U-Net 的守护/启动进程、双卡训练及其数据加载进程已全部停止，训练 GPU 显存已释放；未启动新的 DiT 训练。最后日志步数为 51,400，实际保留的 `latest.pt` 和 `best.pt` 均为 50,000 步；之后的内存进度没有保存。当前 `ReactFlow()`、三个研究配置和正式训练入口的默认主干均已核验为标准 DiT。运行状态文件已标记 `stopped_by_user`，详见[停止与架构确认记录](2026-10-31-training-stopped.md)。

## 2026-10-31：研究主干切换为标准 DiT

按用户明确要求，ReactFlow 默认主干和三个研究模式配置现为标准 DiT：patch=5、hidden=256、depth=8、heads=8，1,200 个全局时空 token，actor 双向交互在固定分辨率的 Transformer 层间完成。迁移初期曾保留 U-Net 兼容分支，后按用户要求移除。新 DiT 从头训练，训练入口与输出目录已区分架构。

注意力分工为 occupancy self-attention、masked actor self-attention 和双向 cross-attention。默认第 4 个 DiT Block 后进行两层交互，actor 更新的 Q 来自 actor、K/V 来自 occupancy；occupancy 更新的 Q 来自 occupancy、K/V 来自 actor。两条序列未拼接成共同 self-attention。VAE 仍只有一套，训练读取已缓存 latent，DiT patch 投影不构成第二次 VAE 编码。详见迁移记录中的注意力与 VAE 小节。

实际验收为 350 项 CPU 测试通过、真实尺寸 CUDA 带一致性损失优化及可复现采样通过、双卡 NCCL 4 步与 2+2 步恢复的模型/EMA/优化器/调度器/RNG 状态逐位一致、安装 wheel 后的隔离真实样本训练/推理/解码通过；两个参考仓库相对本次实现前快照无变化。新 DiT 尚未开始全量训练。旧 seed=2026 U-Net 训练在迁移验收时曾继续运行，现已按用户要求停止，其曲线和权重不属于 DiT。旧源码已归档；旧检查点的源码身份约束及恢复限制见 [完整迁移记录](2026-10-31-standard-dit.md)。

下面的阶段表和累计数字主要记录迁移前的基线与接口验收；新架构实际结果以本节和迁移记录为准。

## 已完成并验证的阶段

| 任务 | 状态 | 证据与边界 |
|---|---|---|
| T0–T2 | 完成 | 独立包、专用环境、普通文件资源、参考快照、复制来源和 SHA-256 记录。OccFM/IR-WM 未作为运行时依赖。 |
| T3 | 完成 | 历史/未来兼容基线各 4,219 个验证窗口全部完成；camera-mask pooled mIoU 分别为 0.3281336254598147 和 0.38052406869179267。它们是带日志真实 ego 位移条件的预测基线，不是无未来信息规划结果。 |
| T4 | 完成 | 官方 850 场景、34,149 帧的 E0 actor/ego/occupancy 标签；当前可见 actor 按当前状态选取，未来标签隔离；原始 3D warp 后再编码。 |
| T5 | 完成 | 固定560/70/70/150场景与train-only统计；train17,468、tune2,176个完整E0窗口已全部生成，全部载荷SHA-256、rank清单、settings摘要及场景互斥审计通过。 |
| T6 | 双架构工程验收完成；原CFM首seed已测，标准DiT续训 | 原CFM joint在72,000步提前停止，52,000步best EMA在官方val150/4,219窗口的未来帧全网格Avg mIoU=28.6739%。标准DiT已从完整54,000步恢复，当前best tune loss=0.5647131986610445；多seed、同信息消融和其他质量指标仍待完成。 |
| T7 | DiT 小样本验收完成 | 连续 decoder endpoint 的一致性、静态时序和类别运动学约束可反传；DiT 200 步/32 个重叠窗口 probe 为 3.19560623169 → 1.25981795415（下降 60.5766%），仅说明小样本拟合。仅 VAE 冻结，DiT/actor/interaction 均训练；VAE 状态哈希未变。 |
| T8 | 协议实现完成，仿真待条件 | 分支数据契约、origin/seed 分组、重置和失败保留规则已实现；没有 CARLA Python 模块或 server，因此没有真实 CARLA rollout。 |
| T9/T11 | 指标接口完成 | 响应差异、footprint/occupancy 风险、平滑频率、校准、ECE/Brier/NLL 与阈值选择接口通过独立复审；缺少反事实标签，尚无响应或校准效果结论。 |
| T10 | 候选规划接口完成 | PlanFlow 候选生成、显式刹停候选、实测时间运动学检查和离线选择器通过独立复审；尚无训练轨迹 head 或规划质量结论。 |
| T12/T13 | 观测前端接口完成 | LiDAR E0 voxel/BEV 特征与 observed-only actor 检测/关联接口通过独立复审；没有训练后的 occupancy 感知精度或 actor 检测结果。 |
| T14 | Bridge/controller 协议完成 | 消息、真值拒绝、新鲜度、时间插值、执行器范围和减速兜底均通过 14 项聚焦测试及独立复审；CARLA/Bench2Drive 闭环仍未运行。 |
| T15 | 等待仿真环境 | 真实闭环和 Dev10/220-route 评测依赖 CARLA/Bench2Drive，当前环境不具备。 |
| T16/T17 | 未完成 | 首个正式seed2026已启动；其余种子、效率消融、主实验表和闭环数据仍待完成，接口测试不替代研究结果。 |

## 数据与坐标边界

用户的 `~/data` 解析为 `/home/lthpc/data`。研究标签使用官方 nuScenes JSON 和 `nuscenes/gts`，写入项目自己的 `data/research/`；原始数据只读。E0 约定为 x 向前、y 向左、z 向上，网格 200×200×16、0.4 m。未来语义和 actor 状态只在训练 target 中出现，推理观测接口会拒绝 future/target 字段。

完整研究缓存采用普通 NPZ/JSON 文件、每条 payload 校验值、源码/权重/统计绑定和原子发布。完整 train/tune manifest 均为 complete，且全量载荷/身份审计通过。pilot 与诊断子集仍单独标记，不能冒充全量数据。

预训练 OccFM 权重曾接触官方 700 个训练场景。因此新划出的 tune/calibration 场景相对于这些权重不具备严格未见性，尚未给出独立校准结论。原始兼容缓存与 E0 研究管线是两套数据口径，不能混用。

## 隔离与来源保护

桥接层独立验收记录于 [`isolation_final6.json`](isolation_final6.json)。本轮增加训练 package 后重新构建并安装 wheel（SHA-256 见 `docs/provenance/joint_training_sources.json`），从 `/tmp` 用 `python -I` 导入训练 adapter/engine；wheel 清单无 OccFM/IR-WM 路径。完整 CPU 测试现为 330 项通过。参考审计记录于 [`reference_audit_final6.json`](reference_audit_final6.json)：OccFM 无差异；IR-WM 只有原快照缺少的空 `data/` 目录，原有文件、HEAD 和 Git 工作区状态保持一致，没有删除或重置。

## 历史运行快照与后续

最新运行快照（2026-10-29T11:21:39+08:00）：正式训练第100步已记入日志；第50步检查点已保存并成功自动恢复，两张GPU继续运行。19,726项恢复身份摘要全部一致；尚未到达首次完整tune评价。见 [运行健康记录](formal_training_health_2026-10-29.json)。

截至 2026-10-29T11:11:17+08:00，全量缓存审计、200步训练、32窗口固定噪声拟合、真实单/双卡恢复与隔离安装验收均已通过。拟合 probe 从4.90875177235降至0.80470655854（83.6067%），只证明小样本拟合。正式双卡seed2026已启动，实际使用train17,468/tune2,176、有效batch8、warmup1000、上限100,000步、每2000步全量tune评估、patience10；首50步保存检查点后自动按原计划续跑。运行状态与全部证据见 [2026-10-29训练验收](2026-10-29-training-acceptance.md)。CARLA、反事实校准、实际感知训练及闭环结果仍待相应环境与实验。

主要证据位于 `docs/implementation/baseline_validation.{json,md}`、`docs/implementation/2026-10-28-data-pipeline.md`、`docs/implementation/isolation_p3.json` 和 `.superpowers/sdd/2026-10-27-reactoccflow-implementation-plan/`。

T8、T9/T11、T10 的接口和修复已通过各自独立复审：分支协议 10 项、候选规划器 16 项、响应/风险 11 项聚焦测试通过；全项目 CPU 测试共 266 项通过。分支协议仍明确记录嵌套字段名过滤和无法仅凭 metadata 证明 NPC 反应性的限制。

T12/T13 的 observed-only LiDAR/actor frontend已完成接口验收：6项聚焦测试和全项目 272 项 CPU 测试通过。它输出 E0 voxel/BEV 特征和检测轨迹，不输出训练好的 occupancy logits，也没有报告感知精度；真实传感器训练与 GT/oracle 切换实验仍待后续阶段。

T12/T13 复审发现的两处边界问题已修复：检测帧必须显式声明 E0，track ID 门控断裂后使用确定性唯一后缀；大坐标在索引前校验。聚焦 9 项、全项目 275 项 CPU 测试通过，独立复审 PASS。

T14 的 bridge/controller 契约已完成：最终 14 项聚焦测试和独立复审 PASS。新增回归覆盖拼接、命名空间及大小写混合的 GT 字段，字符串/布尔执行器值也会被拒绝；合法预测世界字段仍可通过。全项目测试 289 项通过，最终 wheel 隔离验收见 `isolation_final6.json`。CARLA/Bench2Drive 尚未安装或启动，因此没有闭环结果。

Formal joint trainer continuation: project-owned AdamW groups, warmup/cosine, EMA, DDP gradient accumulation, exact-once tune sharding, per-rank RNG checkpointing and resume cursor are implemented. The joint endpoint adapter now lives in `reactoccflow.training` and is shared by smoke and full-run entrypoints. Runner tests cover early rejection of partial caches, scene disjointness, train-only normalization reuse, validation partition coverage, cache payload integrity, output collision broadcast and mid-epoch resume state. Full CPU suite 310 passed; installed engine/adapter imports under `python -I` passed. Train cache 8,350/8,734 and 8,420/8,734; tune complete at 2,176/2,176. Settings wrapper contract fixed; real tune records 0/1088/2175 pass payload/schema checks. Formal optimization not started.

以上较早的累计测试数和缓存计数为历史阶段记录；以本次更新和 training_runner_acceptance.json 为当前状态。

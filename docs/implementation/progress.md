# ReactOccFlow 实施进展

更新日期：2026-10-29。依据[完整实施计划](../superpowers/plans/2026-10-27-reactoccflow-implementation-plan.md)。本文件只记录已经运行并留有证据的工程结果；计划中的研究指标、反事实响应、校准和闭环结果不会提前写成结论。

## 已完成并验证的阶段

| 任务 | 状态 | 证据与边界 |
|---|---|---|
| T0–T2 | 完成 | 独立包、专用环境、普通文件资源、参考快照、复制来源和 SHA-256 记录。OccFM/IR-WM 未作为运行时依赖。 |
| T3 | 完成 | 历史/未来兼容基线各 4,219 个验证窗口全部完成；camera-mask pooled mIoU 分别为 0.3281336254598147 和 0.38052406869179267。它们是带日志真实 ego 位移条件的预测基线，不是无未来信息规划结果。 |
| T4 | 完成 | 官方 850 场景、34,149 帧的 E0 actor/ego/occupancy 标签；当前可见 actor 按当前状态选取，未来标签隔离；原始 3D warp 后再编码。 |
| T5 | 完成 | 固定560/70/70/150场景与train-only统计；train17,468、tune2,176个完整E0窗口已全部生成，全部载荷SHA-256、rank清单、settings摘要及场景互斥审计通过。 |
| T6 | 工程验收完成，正式训练中 | joint/parallel/occupancy-only接口可运行；主干参与训练的200步单卡与32窗口拟合检查通过，双卡NCCL及恢复通过；seed2026全量联合训练已启动，质量结论尚待正式评测。 |
| T7 | 完成小样本验收 | 连续 decoder endpoint 上的一致性损失、静态时序和类别运动学约束可反传；200 步、32 个重叠窗口的 debug 训练 probe 从 4.58998449519 降到 3.09713234752（32.5241%），仅说明小样本拟合链路工作。冻结 VAE/预训练主干的梯度与状态校验通过。 |
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

## 当前运行与后续

最新运行快照（2026-10-29T11:21:39+08:00）：正式训练第100步已记入日志；第50步检查点已保存并成功自动恢复，两张GPU继续运行。19,726项恢复身份摘要全部一致；尚未到达首次完整tune评价。见 [运行健康记录](formal_training_health_2026-10-29.json)。

截至 2026-10-29T11:11:17+08:00，全量缓存审计、200步训练、32窗口固定噪声拟合、真实单/双卡恢复与隔离安装验收均已通过。拟合 probe 从4.90875177235降至0.80470655854（83.6067%），只证明小样本拟合。正式双卡seed2026已启动，实际使用train17,468/tune2,176、有效batch8、warmup1000、上限100,000步、每2000步全量tune评估、patience10；首50步保存检查点后自动按原计划续跑。运行状态与全部证据见 [2026-10-29训练验收](2026-10-29-training-acceptance.md)。CARLA、反事实校准、实际感知训练及闭环结果仍待相应环境与实验。

主要证据位于 `docs/implementation/baseline_validation.{json,md}`、`docs/implementation/2026-10-28-data-pipeline.md`、`docs/implementation/isolation_p3.json` 和 `.superpowers/sdd/2026-10-27-reactoccflow-implementation-plan/`。

T8、T9/T11、T10 的接口和修复已通过各自独立复审：分支协议 10 项、候选规划器 16 项、响应/风险 11 项聚焦测试通过；全项目 CPU 测试共 266 项通过。分支协议仍明确记录嵌套字段名过滤和无法仅凭 metadata 证明 NPC 反应性的限制。

T12/T13 的 observed-only LiDAR/actor frontend已完成接口验收：6项聚焦测试和全项目 272 项 CPU 测试通过。它输出 E0 voxel/BEV 特征和检测轨迹，不输出训练好的 occupancy logits，也没有报告感知精度；真实传感器训练与 GT/oracle 切换实验仍待后续阶段。

T12/T13 复审发现的两处边界问题已修复：检测帧必须显式声明 E0，track ID 门控断裂后使用确定性唯一后缀；大坐标在索引前校验。聚焦 9 项、全项目 275 项 CPU 测试通过，独立复审 PASS。

T14 的 bridge/controller 契约已完成：最终 14 项聚焦测试和独立复审 PASS。新增回归覆盖拼接、命名空间及大小写混合的 GT 字段，字符串/布尔执行器值也会被拒绝；合法预测世界字段仍可通过。全项目测试 289 项通过，最终 wheel 隔离验收见 `isolation_final6.json`。CARLA/Bench2Drive 尚未安装或启动，因此没有闭环结果。

Formal joint trainer continuation: project-owned AdamW groups, warmup/cosine, EMA, DDP gradient accumulation, exact-once tune sharding, per-rank RNG checkpointing and resume cursor are implemented. The joint endpoint adapter now lives in `reactoccflow.training` and is shared by smoke and full-run entrypoints. Runner tests cover early rejection of partial caches, scene disjointness, train-only normalization reuse, validation partition coverage, cache payload integrity, output collision broadcast and mid-epoch resume state. Full CPU suite 310 passed; installed engine/adapter imports under `python -I` passed. Train cache 8,350/8,734 and 8,420/8,734; tune complete at 2,176/2,176. Settings wrapper contract fixed; real tune records 0/1088/2175 pass payload/schema checks. Formal optimization not started.

以上较早的累计测试数和缓存计数为历史阶段记录；以本次更新和 training_runner_acceptance.json 为当前状态。

# 新mask：10,000步训练、正式评测与三组对比

> 后续暂停：用户要求停止，已于2026-11-18 11:36:06暂停在缓存阶段，保留11,027条训练缓存，正式训练0步。后续训练/评测不自动启动；见[暂停与恢复记录](2026-11-18-ground-mask-training-paused.md)。下文保留启动时状态。

启动时间：2026-11-18 10:21:43（北京时间）。控制器PID：264199。

用户已明确授权切换新mask训练10,000步，随后评测并与前两轮对比。已启动独立自动流程；本记录写入时处于新历史缓存生成阶段，正式优化尚未开始。

运行目录：`outputs/history_input_comparison/2026-11-18-ground-mask-seed2026/`。

## 输入与公平对照

- 新模式为`mask_lidar_temporal_ground` v3：两次直接历史观测＋当前四邻居同类别支持的地面补全，仅使用过去/当前观测。
- 语义occupancy、冻结原OccFM VAE、annotation actor和U形CFM保持既有路线。未使用raw训练权重。
- 独立新训练从与前两轮相同的`assets/checkpoints/occfm_history.ckpt`、`state_dict`初始化主干；不从旧10,000步模型继续累加。
- seed2026、双卡FP32、每卡batch2、累积2、有效batch8、workers2、decode-chunk1；主干LR2e-5，新分支LR1e-4。
- 总预算10,000 optimizer steps，warmup1,000、每2,000步tune、patience10、EMA0.9999。正式成绩预先固定第10,000步EMA。
- 原两轮的模型/损失/训练模块36个Python文件及训练入口保持逐字节一致；仅输入处理和独立任务编排新增。

新缓存：`data/research/e0_mask_lidar_temporal_ground/{train,tune}`，窗口数17,468/2,176，顺序与原缓存一致。

复用旧mask缓存中已经验证的未来目标，重新编码新历史。复用前校验源occupancy、元数据、VAE、normalization、几何、数值设置及窗口身份；每条读取验证旧payload SHA，并逐项核对非latent观测。每条新记录绑定旧记录SHA。训练未来目标不进入观测编码。

## 自动执行顺序

1. 双卡生成新train历史缓存。
2. 双卡生成新tune历史缓存。
3. 双卡训练新mask U形CFM＋actor至总预算10,000步。
4. 严格导出第10,000步EMA。
5. 双卡评测官方validation150场景、4,219窗口。
6. 汇总并复核rank原始计数。
7. 与前两轮固定step10000 EMA结果对比，生成总体及17类×3时刻表格。

评测采用与前两轮相同的FP32、NFE10、seed2026、batch4/卡、encode-chunk4/decode-chunk2。旧两轮完整结果在SHA/窗口/GT/计数/评分代码核验后复用；仅允许声明的历史输入代码差异。

旧两轮平均mIoU：密集历史27.8944%，原mask27.9574%；新mask结果尚未产生。小规模重建收益不能预先当作未来预测收益。

## 已完成的启动核验

- 183项定向测试通过，覆盖目标复用、信息边界、三组配置对齐、评分代码限制及流程推进。
- 双卡短跑完成2步优化、完整4窗口小集tune、生成探针和checkpoint保存，实际退出0。首次短跑命令因warmup/两步上限冲突被参数检查拒绝，原日志保留；修正为10,000步调度、第2步显式保存暂停后通过。正式命令未加入短跑参数。
- 4窗口评测短跑两rank退出0/0、汇总退出0，仅作运行验证。
- train和tune各抽样重编码完整未来目标，与原缓存逐项完全相同；所有短跑记录非latent观测一致。
- 195个独立普通文件冻结源码，SHA：`c6c6664c2fdf0bfe1c12c970aef73caa52f37c2391f47bcfc223326011080c07`。
- 普通wheel安装到专用环境，`/tmp`下`python -I -B`隔离检查通过，禁止读取两个参考目录。
- 参考仓库无变更；18项受保护历史文件SHA无变更，raw STOP标记保留。

## 预计耗时与监控

按此前同设备训练和本轮短跑估计：缓存约2–3小时、训练约10小时、单轮评测及报告约1小时，总体约13–15小时。实际速度以正式缓存和训练日志更新，不能保证固定结束时刻。

- `launch_config.json`：固定命令、环境、停止标记、预算及全部后续阶段。
- `pipeline.json`：当前阶段、PID、心跳、进度和真实退出码。
- `train/training.jsonl`：正式优化/验证/保存记录。
- `startup_health.json`、`cache_preflight.json`、`training_alignment.json`：启动证据。
- `comparison_config.json`：三组检查点、既有结果和输出身份。

失败停止，不自动重试；本轮停止标记为运行目录下的`STOP_REQUESTED.json`。此前raw停止标记与历史运行保持独立。

最终产物：`checkpoints/ground_step10000_ema.pt`、`evaluation/new/full/`、`comparison/comparison.json`、`comparison/comparison.csv`、`comparison/per_class_comparison.csv`和`comparison/三种历史输入_10000步评测对比.md`。

来源：[新mask实现与诊断](2026-11-18-mask-lidar-ground-completion.md)、[来源记录](../provenance/ground_mask_training_sources_20261118.json)。

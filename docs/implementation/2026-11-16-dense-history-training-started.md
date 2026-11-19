# 密集语义历史：U形CFM＋actor训练启动

> 2026-11-17更新：本轮已于北京时间05:50完成10,000步训练及最终检查点保存。核查时发现退出阶段残留，验证权重后已清理并释放显存；实际非零进程退出码和原始运行状态保留。剩余训练时间为0，详见[完成记录](2026-11-17-dense-history-training-completed.md)。下文启动状态与ETA为历史记录。

用户于2026-11-16明确要求“按照新的密集语义历史输入进行训练，训练大概需要多长时间”。本次授权包括新train/tune缓存生成及其后续世界模型训练；早先的代码验收暂停范围对这一个新运行解除。其他历史实验、raw训练与raw STOP_REQUESTED标记保持原状。

> 预算更新：用户随后要求将本轮上限改为**10,000步**。北京时间16:40提出变更时模型训练尚未开始；16:42重启控制流程应用新配置，已完成缓存经校验后复用。旧控制器为127583，新控制器为132475。冻结源码及缓存生成参数不变，原日志和配置保存在`history/before_budget_10000/`，变更记录为`budget_update_10000.json`。

## 已启动的流程

北京时间2026-11-16 **15:10:23**启动独立后台控制器，PID为`127583`。当前阶段为`cache_train`，两张RTX A6000分别承担一个缓存rank，初始子进程PID为`127584`、`127585`。模型优化训练尚未开始；流程顺序为：

1. 生成密集语义历史train缓存：560个场景、17,468个窗口。
2. 校验完整性后生成tune缓存：70个场景、2,176个窗口。
3. 两组缓存均通过校验后，自动启动双卡U形CFM＋actor联合训练。

流程只为本次明确授权的实验自动推进。失败不会自动重试，不跳过失败阶段；收到新运行的停止标记后不启动后续阶段。cache和训练阶段使用两张卡顺序执行，不同时开启两套DDP训练。

## 数据、初始化与预算

| 项目 | 本轮设置 |
|---|---|
| 输入模式 | `dense_semantic`，历史有效区域为`in_bounds & semantics!=255` |
| 数据来源 | 原有Occ3D语义标签和nuScenes标注，不训练原始点云感知 |
| train/tune缓存 | `data/research/e0_dense_history/{train,tune}`，保留旧`e0_full` |
| 世界模型 | U形CFM＋actor，联合Rectified Flow目标 |
| occupancy VAE | 原`assets/checkpoints/occfm_vae.ckpt`，冻结，不重新训练 |
| 主干初始化 | 原`assets/checkpoints/occfm_history.ckpt`的`state_dict` |
| actor/交互分支 | seed2026的新初始化，随联合模型训练 |
| 学习率 | 主干`2e-5`，新增分支`1e-4`，沿用原U形实验设置 |
| 每卡batch / 梯度累积 | 2 / 2，两卡有效batch为8 |
| 精度 / workers / decode chunk | FP32 / 每rank 2 / 1 |
| 最大训练步数 | **10,000（用户后续指定，原100,000已撤销）** |
| warmup / 完整tune间隔 | 1,000 / 2,000步 |
| 提前停止 | 完整tune loss连续10次不改善 |
| EMA | 0.9999 |
| 生成监控 | 8个固定tune窗口，NFE10；不属于正式官方评测 |

不从旧joint训练状态原地续训，也不使用raw实验权重。与旧U形实验尽量保持相同初始化、学习率和有效batch，主要改变历史输入。输入修复后的实际mIoU收益仍需训练完成后的配对评测确认。

## 耗时估计（10,000步预算）

| 阶段 | 预计耗时 | 依据 |
|---|---:|---|
| train＋tune缓存，总耗时 | 4–5小时 | 已完成部分保留，新运行初始稳定速度约1.59–1.61秒/窗口/卡 |
| 世界模型训练至10,000步 | **约10–11小时** | 上次同机U形模型中位约3.6115秒/优化步，另计5次完整tune与保存开销 |
| 从最初流程启动到结束 | **约14–16小时** | 缓存与10,000步训练合计 |
| 预算变更时的剩余时间 | **约13–15小时** | 剩余缓存、断点校验及后续训练，待训练实际开始后校准 |

变更后预计于**北京时间11月17日清晨至上午**结束。模型训练耗时尚不是本轮实测，缓存速度也可能随场景与I/O变化。此前基于100,000步的4–5天估计不再适用。

预计缓存于11月16日晚完成，之后自动进入模型优化。上述估计不包含后续正式mIoU评测。

学习率计划的总步数同步为10,000，warmup仍为1,000步，验证间隔仍为2,000步，其余超参数保持原配置。本轮现在是10,000步预算实验，训练长度与历史72,000步U形实验不同。

## 验证与恢复边界

- 新增主干学习率选项和本运行停止标记支持，原CLI学习率默认值保持不变，本轮显式指定`2e-5`。停止请求会在同步优化步之后保存完整状态，覆盖“刚好遇到验证间隔边界”的情况。
- 启动前34项定向测试通过，包括完整保存后暂停、冻结源码校验、已有停止标记禁止启动、失败不推进以及完成缓存后才推进。
- 使用190个独立普通文件冻结本轮源码，指纹为`dbf24c7d982ce7b188997afb269a7883822cc1077fdd785decceccce73bf5e60`；后续工作区修改不进入本运行。
- 检查新缓存模式为dense，VAE SHA-256仍为`85c5a2566b3fa8a1ebb785af3113c6674fe9811e5eda2402b1959ce6cd66ef10`。
- 对2个已完成窗口与旧缓存配对检查：历史latent改变，actor、未来目标、mask和时间等其余29个数组逐值相同。该抽查不代表已逐条验证完整新缓存。
- 启动时两个参考仓库快照零差异，9个原权重/旧缓存清单/设置/raw停止标记的SHA-256不变。
- 新运行使用自己的`STOP_REQUESTED.json`；原raw运行的同名标记不删除，也不用于启动新运行。

## 产物位置

运行根目录：[outputs/reactflow_occfm_unet_dense_history/2026-11-16-seed2026](../../outputs/reactflow_occfm_unet_dense_history/2026-11-16-seed2026/)。

- [pipeline.json](../../outputs/reactflow_occfm_unet_dense_history/2026-11-16-seed2026/pipeline.json)：当前阶段、控制器/子进程PID、退出码及心跳。
- [launch_config.json](../../outputs/reactflow_occfm_unet_dense_history/2026-11-16-seed2026/launch_config.json)：所有确切命令、环境及授权边界。
- [source.manifest.json](../../outputs/reactflow_occfm_unet_dense_history/2026-11-16-seed2026/source.manifest.json)：冻结源码逐文件SHA-256。
- [startup_health.json](../../outputs/reactflow_occfm_unet_dense_history/2026-11-16-seed2026/startup_health.json)：初步速度、缓存抽查及源保护结果。
- `cache_train.rank-{0,1}.log`、`cache_tune.rank-{0,1}.log`：各缓存进程日志。
- `training.rank-0.log`：后续torchrun双rank训练日志；`train/training.jsonl`为结构化记录。
- `train/latest.pt`：后续完整模型/EMA/优化器/scheduler/双rank RNG/游标状态；`train/best.pt`为完整tune loss最优EMA。

后续暂停/恢复应以此新运行的配置、冻结源码和检查点为准，不使用旧mask输入实验的恢复入口。此次没有启动正式官方验证集预测或更改任何历史成绩。

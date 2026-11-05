# ReactFlow 首次完整预测 mIoU 评测

2026-11-04，按用户要求测试已训练CFM joint世界模型的生成预测，并对照OccFM论文。训练于北京时间09:08:40在72,000步触发patience=10提前停止；本次使用先前按完整tune联合损失选定的52,000步best EMA，不根据官方验证mIoU挑选权重。

官方validation的150个场景、4,219个预测窗口已全部完成。两卡均退出0，正式运行耗时56.70分钟，结束时间11:28:06。使用原训练源码的独立普通文件导出、冻结VAE和模型参数、NFE=10、一个生成未来、FP32、逐窗口固定噪声seed=2026。没有执行优化器步骤或重新启动任何训练。

| 模型 / 口径 | 1s mIoU | 2s mIoU | 3s mIoU | 三时刻平均 |
|---|---:|---:|---:|---:|
| OccFM论文，未来轨迹条件 | 36.42% | 27.39% | 21.66% | 28.49% |
| 本机既有完整OccFM e196复现 | 35.8208% | 27.3686% | 21.6111% | 28.2669% |
| 本项目，转换至未来帧全网格 | 33.4676% | 28.0838% | 24.4704% | **28.6739%** |
| 本项目减论文（百分点） | -2.9524 | +0.6938 | +2.8104 | **+0.1839** |

论文来自[OccFM表1](https://arxiv.org/html/2506.23434v2)。当前模型使用已给定的日志未来ego XY和真值actor历史，因此主要对照未来轨迹条件行。数值上平均值有小幅提升，但1秒下降、2/3秒提高；目前不能将单seed的0.18个百分点差异归因于结构创新。

评测17个非空语义类，排除empty17、忽略GT255，不使用camera/LiDAR评分mask；按全数据集累计int64计数，再计算每时刻mIoU及1/2/3秒平均。全网格GT逐时刻逐类别计数与原OccFM复现严格相同；另外对全部6,019帧原始GT与复现资源逐体素核对，全部相等。standard和公开代码legacy协议本次分数相同。

模型在固定当前E0坐标系输出。为对照原未来帧GT，评分器用未来GT位姿做3D最近邻坐标对齐，预测视野外填empty17并保留计分；未来occupancy/位姿不进入预测器。E0对未来1/2/3秒GT网格的覆盖率为90.26 / 82.25 / 74.98%。原生E0已知区域的Avg mIoU=30.5889%，仅覆盖区域重采样后的Avg=30.5887%；这些诊断分数不能替代全网格论文对比。

当前训练的历史occupancy经过mask_lidar限制，另有oracle actor历史；原OccFM采用其原密集语义latent、轨迹表达和CFG。joint微调560场景，其初始化和VAE此前接触过官方700训练场景。既有OccFM复现使用bf16/seed1000，本次是FP32/逐窗口seed2026。表示、输入、训练和采样设置尚未统一，因此该结果不是同信息、同预算重训的消融，也不包含多seed显著性、物理一致性或规划提升结论。

完整报告、每类IoU、累计计数和实际命令保存在：

- [详细对照报告](../../outputs/evaluation/2026-11-04-reactflow-best52000/mIoU评测与OccFM对照.md)
- [汇总JSON](../../outputs/evaluation/2026-11-04-reactflow-best52000/comparison.json)
- [逐类CSV](../../outputs/evaluation/2026-11-04-reactflow-best52000/per_class_comparison.csv)
- [全部评分计数](../../outputs/evaluation/2026-11-04-reactflow-best52000/full/result.json)
- [运行命令与退出码](../../outputs/evaluation/2026-11-04-reactflow-best52000/full/launch.json)
- [协议与条件边界](../../outputs/evaluation/2026-11-04-reactflow-best52000/protocol.json)

独立评测实现见`reactoccflow/evaluation/reactflow_generation.py`，8项CPU测试通过，覆盖时间均值、假阳性/未知mask、精确分片、固定噪声、非平凡坐标变换与计数合并。16窗口双卡预检通过后执行全量评测。来源、隔离导出和参考保护验收以同目录`reactflow_miou_acceptance_20261104.json`及`docs/provenance/reactflow_generation_sources_20261104.json`为准。

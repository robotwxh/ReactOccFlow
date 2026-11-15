# 2026-11-11：DiT512全量mIoU评测

> 后续诊断已完成，见 [为什么仍落后U形CFM](2026-11-11-dit512-cfm-gap-analysis.md)：固定tune干预定位到actor→occupancy注入退化及语义生成/latent loss失配，旧秩瓶颈已解除。下文官方成绩与原权重保持不变，诊断小集分数不替代正式表格。

按用户要求评测修复后的标准DiT hidden512＋actor，并对照OccFM论文、既有OccFM本机复现、上次U形CFM＋actor和hidden256＋actor。使用完整tune联合损失预选的**84,000步best EMA**，总训练100,000步；不是100,000步最终权重，也不是8场景tune探针。

官方validation共150场景、4,219个窗口。FP32、NFE10、每窗口固定身份噪声seed2026、每条件1个世界，两个rank分别2,110/2,109个窗口；batch4/卡、workers4、encode-chunk4/decode-chunk2。未来ego原始全网格评分，17个非空语义类，忽略GT255；E0范围外预测empty17参与计分。Avg为1/2/3秒mIoU算术平均。

| 模型 | 1s mIoU (%) | 2s mIoU (%) | 3s mIoU (%) | Avg (%) |
|---|---:|---:|---:|---:|
| OccFM论文，未来轨迹条件 | 36.4200 | 27.3900 | 21.6600 | 28.4900 |
| OccFM e196本机已审计复现 | 35.8208 | 27.3686 | 21.6111 | 28.2669 |
| U形CFM＋actor，best52,000 | 33.4676 | 28.0838 | 24.4704 | 28.6739 |
| 标准DiT hidden256＋actor，best100,000 | 2.8361 | 2.5240 | 2.3663 | 2.5755 |
| 标准DiT hidden512＋actor，best84,000 | **21.1581** | **17.7828** | **15.6813** | **18.2074** |

论文行来自[OccFM原文表1的Ours†](https://arxiv.org/html/2506.23434v2)。hidden512相对hidden256为**+15.6320个百分点**，相对U形CFM＋actor为**−10.4665个百分点**，相对论文未来轨迹条件为**−10.2826个百分点**。修复后结果明显恢复，仍未达到既有U形模型或OccFM论文数值。

hidden512原生E0 Avg为19.2426%，未来网格仅覆盖区域Avg为19.2431%；U形模型分别为30.5889%/30.5887%。未来1/2/3秒网格覆盖率为90.2632%/82.2534%/74.9774%，各actor模型相同，视野范围不能单独解释剩余差距。17类的三时刻平均IoU均高于hidden256，也均低于U形CFM；差距同时涉及静态和动态类别，详见完整报告。

两rank生成分别于北京时间11:35:47/11:35:35完成，较慢rank计时2,263.81秒，约37.73分钟。聊天中断使原监督启动器提前退出，未自动汇总；原始rank操作系统退出码不可追溯，不将其记成0。两个rank均保留completed结果、finished状态和完整精确游标。15:30后恢复时只执行既有CPU聚合入口，退出0，未重新生成；原启动记录另存，恢复记录绑定其SHA及两rank原始结果/状态SHA。汇总计数与两rank之和完全一致。

与hidden256及U形CFM分别核对18项推理设置、5,723份共享数据/VAE/归一化SHA和10个生成评分函数AST，均一致。三个actor模型各时刻各类GT计数、mask和覆盖率一致；全网格GT与原OccFM复现混淆矩阵一致。旧检查点及三个基线结果等9个关键产物SHA未变化，6份源码归档共698文件均与清单匹配；参考仓库370/389项前后快照零差异，独立评测导入检查通过。计数/协议审计测试17项通过。

本轮只度量occupancy mIoU。三者都有actor，不能量化actor相对无actor的独立收益，也没有测量actor ADE/FDE、碰撞或物理一致性。U形主干使用OccFM初始化，两个标准DiT随机初始化；参数量、优化轨迹与best步数不同。原OccFM输入、CFG、精度与随机噪声协议也有差异。此前验证失败已被用于诊断hidden256结构，本轮属于诊断性复用；单seed、重叠窗口结果不能支持架构因果、显著性或独立未见测试结论。

完整报告：[DiT512与hidden256、CFM、OccFM评测对照](../../outputs/evaluation/2026-11-11-dit512-best84000/DiT512与hidden256_CFM_OccFM评测对照.md)。机器结果、五行总表CSV、51行逐类别CSV、固定协议、原始计数、身份、恢复记录和acceptance均在同目录。可从项目根目录执行 `PYTHONDONTWRITEBYTECODE=1 /home/lthpc/wxh/envs/ReactOccFlow/bin/python -B -m scripts.compare_dit512_evaluation` 重建对照报告，命令不重新生成或训练模型。

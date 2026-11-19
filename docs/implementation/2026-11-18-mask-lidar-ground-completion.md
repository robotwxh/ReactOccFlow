# mask_lidar地面类诊断与保守历史补全

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate，当前上下文内核验
- Origin Date: 2026-11-18
- Verification Status: ANALYZED；代码测试和训练子集诊断已执行，未来预测收益尚未验证
- Version Label: observed_ground_v3

## 发现与解释

两轮10,000步的3秒结果中，密集历史在四个地面类均略好于mask：可行驶区域+0.38、other_flat+0.79、人行道+0.40、地形+0.18个百分点。other_flat并非唯一受影响类别，该差距也不能视为单seed下的显著结论。

均匀抽取128个具有完整历史的**训练集**窗口，当前帧共81,920,000个体素，统计原始标签与mask_lidar：

| 类别 | 完整语义体素数 | mask保留体素数 | 保留比例 |
|---|---:|---:|---:|
| car | 165,614 | 165,611 | 99.9982% |
| pedestrian | 15,357 | 15,357 | 100.0000% |
| driveable_surface | 895,599 | 794,428 | 88.7035% |
| other_flat | 21,970 | 19,258 | 87.6559% |
| sidewalk | 267,407 | 242,388 | 90.6439% |
| terrain | 277,364 | 254,923 | 91.9092% |

直接证据是：other_flat有2,712个体素（12.3441%）没有作为当前可见语义传入编码器；相同抽样中，它的完整体素数约为可行驶区域的1/40.8。原冻结VAE没有unknown通道，`OccupancyCodec.encode`把不可见体素替换为空类17。因而地面语义缺口会影响latent，并造成与密集语义VAE输入的分布差异。

“地面可见信息不足及其编码近似影响了表现”有输入统计和下述重建检查支持；类别频率、远近分布、类间混淆、优化随机性各自对未来IoU的贡献尚未分离，不能仅由本次分数差作因果定论。

## 已实现的独立输入模式

新增显式选项：`--history-visibility mask_lidar_temporal_ground`。原始dense和mask模式保留，既有权重/缓存/正式评测不覆盖、不重新标记。

处理在E0三维最近邻对齐后、冻结VAE编码前进行，缓存与在线评测共用`complete_masked_ground_history`：

1. 仅遍历所选的最多4个过去/当前帧，按时间顺序建立观测记忆。原mask可见的标签，包括空类和车辆，始终优先。
2. 候选类别只包括地面11/12/13/14。相同E0体素必须至少两次直接观测到同一类别；新的可见冲突标签清除旧证据。补出的体素不计为新观测。
3. 当前帧候选体素前后左右四个邻居都必须直接可见，在同一高度上具有相同类别；排除边界、单方向外推和大块无观测区域。
4. 只在当前几何有效、不可见的位置填入**过去实际观测的标签**。不读取隐藏位置的当前语义来选择补全类别，不把未来帧引入条件，不补动态物体。

这是一项基于语义观测的保守插补，补全值属于推断，不是新增激光测量；项目原有oracle语义/actor性质保持明确。没有把mask直接膨胀后读取隐藏GT，也没有按验证集GT放开other_flat掩码。

新缓存默认路径为`data/research/e0_mask_lidar_temporal_ground/{train,tune}`。设置、样本metadata、checkpoint模式和在线生成身份绑定该策略；原mask缓存不能冒充此模式。未来语义、未来已知掩码和actor数据不改变。既有checkpoint仍按原模式推理，不能通过改字段冒充新训练模型。

## 真实训练数据检查

最终版本使用均匀抽取的128个训练窗口检查覆盖；其中均匀选择64个窗口进行冻结VAE posterior mean重建。三个输入条件使用相同当前帧全网格known GT评分。没有读取未来occupancy进行这项诊断。

| 指标 | 原mask_lidar | 改进模式 | 差值（百分点） |
|---|---:|---:|---:|
| other_flat正确输入覆盖率，128窗口 | 87.6559% | 87.8152% | +0.1593 |
| other_flat VAE重建IoU，64窗口 | 86.0405% | 86.1301% | +0.0896 |
| 17类VAE重建mIoU，64窗口 | 92.0712% | 92.0986% | +0.0274 |

相同64个重建窗口的密集语义other_flat IoU为98.8162%。这揭示了mask输入下的重建损失，不等于CFM未来预测上限。

最终版本给other_flat补了57个体素，其中35个匹配当前完整语义，22个有误，插补精度61.40%；正确补回量仅占原来2,712个缺失体素的1.29%。**增益小，插补并不完美**。地面原始高度量化和对齐仍会带来误差；这项保守模式不能恢复从未被观测到的大面积地面。

开发期间的负结果也保留：最初仅要求两次历史观测的v1，在64窗口覆盖检查、32窗口重建中，other_flat重建从88.0755%下降至82.7703%，已弃用；允许一个相反方向邻居对支持的v2为88.2603%，但误补较多，因此最终使用四邻居支持。阈值在训练子集探索，未使用官方validation选择策略；不能把这些开发结果写成独立确认实验或统计显著性证据。

**上述表格是输入覆盖和VAE重建，均不是1/2/3秒预测IoU。** 新输入模式尚未生成完整train/tune缓存、尚未训练新CFM，也没有新的官方预测结果。需要独立训练与相同固定step10000 EMA评测，才能判断是否改善未来other_flat并保持其他类别。

## 使用方法

以下是后续实验命令，本次没有执行全量缓存生成和训练。两个cache rank需在两个终端分别执行；train与tune各自完成且manifest为complete后才能训练。

```bash
# 训练缓存：rank 0；另一终端将rank设为1、device设为cuda:1
PYTHONDONTWRITEBYTECODE=1 /home/lthpc/wxh/envs/ReactOccFlow/bin/python -B -m scripts.cache_research \
  --history-visibility mask_lidar_temporal_ground --group train \
  --label-manifest data/research/train/manifest.json --rank 0 --world-size 2 --device cuda:0

# tune缓存：同样分别运行rank 0和rank 1
PYTHONDONTWRITEBYTECODE=1 /home/lthpc/wxh/envs/ReactOccFlow/bin/python -B -m scripts.cache_research \
  --history-visibility mask_lidar_temporal_ground --group tune \
  --label-manifest data/research/tune/manifest.json --rank 0 --world-size 2 --device cuda:0

# 独立新实验，沿用前两轮训练设置，总预算10,000步
PYTHONDONTWRITEBYTECODE=1 /home/lthpc/wxh/envs/ReactOccFlow/bin/python -B -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module scripts.train_react_flow \
  --history-visibility mask_lidar_temporal_ground --backbone occfm_unet --unet-config configs/unet.yaml \
  --train-cache data/research/e0_mask_lidar_temporal_ground/train \
  --tune-cache data/research/e0_mask_lidar_temporal_ground/tune \
  --codec-checkpoint assets/checkpoints/occfm_vae.ckpt \
  --trunk-checkpoint assets/checkpoints/occfm_history.ckpt --trunk-weights state_dict --trunk-lr 2e-5 \
  --seed 2026 --max-steps 10000 --warmup-steps 1000 --eval-interval 2000 --patience 10 \
  --batch-size 2 --grad-accum 2 --workers 2 --decode-chunk 1 \
  --probe-tune-windows 8 --probe-nfe 10 --device cuda \
  --stop-request-file outputs/reactflow_mask_lidar_temporal_ground/2026-11-18-seed2026/STOP_REQUESTED.json \
  --output outputs/reactflow_mask_lidar_temporal_ground/2026-11-18-seed2026
```

正式运行时仍应使用项目既有冻结源码/身份核验流程，预先固定step10000 EMA，不按官方validation挑权重；比较时保留原mask与密集输入结果。训练时间/收益均不能由这项小规模重建检查推断。

## 核验与证据

代码测试覆盖：隐藏语义变化不影响条件、前缀因果性、两次直接观测、四邻居支持、可见冲突、动态类别不补、无效范围不补、缓存/在线一致、未来目标不变、缓存模式隔离。182项相关测试及额外1项缓存身份测试通过。普通wheel安装到专用环境后，从`/tmp`使用`python -I -B`执行独立导入和补全检查，并阻止读取两个参考目录。

统计/方法谬误检查11/11：本报告保留全类别结果和失败候选，报告样本基数与误补；探索性训练子集、同seed、重复开发与体素相关性均保留，不报告p值或显著性，不把重建收益外推为未来预测或规划收益。未做场景分层推断、总体因果分摊或多seed验证。

- [最终128窗口诊断与完整计数](../../outputs/verification/2026-11-18-mask-ground/train128_ground_four_neighbours.json)
- [原128帧可见性统计](../../outputs/verification/2026-11-18-mask-ground/baseline_visibility_train128.json)
- [弃用的v1结果](../../outputs/verification/2026-11-18-mask-ground/train64_ground_reconstruction.json)
- [v2结果](../../outputs/verification/2026-11-18-mask-ground/train64_ground_spatial_support.json)
- [源搜索与来源记录](../provenance/masked_ground_sources_20261118.json)
- [隔离执行](../../outputs/verification/2026-11-18-mask-ground/isolated_execution.json)


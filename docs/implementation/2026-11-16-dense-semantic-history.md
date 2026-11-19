# 密集语义历史输入修复

日期：2026-11-16。范围按用户确认限定为代码与验证；缓存生成、真实训练和正式mIoU评测继续暂停，原STOP_REQUESTED标记保留。

> 后续授权更新：用户随后明确要求“按照新的密集语义历史输入进行训练”。本页保留此前代码验收快照；新的独立运行已获准生成train/tune缓存并继续U形CFM＋actor训练，见[启动记录](2026-11-16-dense-history-training-started.md)。其他旧实验及raw训练保持暂停。

## 修改结果

新实验默认把历史语义occupancy中的全部已知、网格内体素送入原冻结OccFM VAE，不再额外使用LiDAR可见性过滤。只改变历史输入策略，保留原actor选择与状态编码、四帧观测/六个latent槽位、E0三维变换、未来目标、损失、采样及U形CFM/标准DiT512结构。

| 设置 | 新默认 | 历史实验 |
|---|---|---|
| 模式名 | `dense_semantic` | `mask_lidar` |
| 历史有效区域 | `in_bounds & semantics!=255` | `mask_lidar & in_bounds & semantics!=255` |
| 编码API参数 | `visibility_key=None` | `visibility_key='mask_lidar'` |
| 缓存目录 | `data/research/e0_dense_history/{train,tune}` | `data/research/e0_full/{train,tune}` |
| 当前缓存状态 | 尚未生成 | 原有缓存保留 |

密集模式不读取传感器mask，未知255和越界体素依然屏蔽。原VAE没有unknown通道，这些无效体素仍以empty17近似嵌入，不能视为有效空闲证据。这是语义GT的oracle表示实验，不是新增原始LiDAR感知能力。

实现复用项目已有的完整已知语义编码路径和三维warp，增加共享历史可见性规则，使离线缓存和在线评测采用相同有效区域。参考来源与检索记录见[provenance](../provenance/dense_history_sources_20261116.json)。

## 模式与兼容性

- 缓存CLI新增`--history-visibility dense_semantic|mask_lidar`，默认密集模式，默认输出目录同时按模式和`--group`选择。
- 缓存设置、样本元数据和新检查点记录模式；缓存指纹绑定模式与VAE身份。新生成记录另存`codec_checkpoint_sha256`。模式变化不能在旧目录中续写。
- 新训练入口默认选择密集train/tune目录。训练前检查两组缓存、样本记录与请求模式一致，不会自动回退到旧缓存。
- 在线评测从检查点配置读取`history_visibility`；没有此字段的历史joint检查点使用原`mask_lidar`规则，未知模式拒绝运行。
- 评测协议版本为3，输入模式进入评测身份；新版评测应使用新输出目录。旧版计数/断点评测和旧训练严格续训使用对应冻结源码。
- 新输入不改变模型state_dict结构，但旧训练状态不能作为更换数据后的原地恢复。是否以旧模型作新实验初始化需要单独指定，本次没有加载权重开展新训练。

## 待后续授权执行的缓存命令

以下命令仅供准备，没有执行。数据仍来自`/home/lthpc/data/nuscenes/gts`的既有语义occupancy与官方标注；不读取原始点云训练，不使用已归档raw实验权重。

在项目根目录使用专用环境。先在两个终端分别运行train的rank0/rank1；两者完成后，再运行tune的两条命令，避免四个编码进程争用两张卡。

```bash
# 每个终端先设置此变量。
REACTOCCFLOW_PY=/home/lthpc/wxh/envs/ReactOccFlow/bin/python
```

train，共17,468个预期窗口：

```bash
# 终端1
"$REACTOCCFLOW_PY" -B -m scripts.cache_research \
  --group train --label-manifest data/research/train/manifest.json \
  --history-visibility dense_semantic --device cuda:0 --rank 0 --world-size 2

# 终端2
"$REACTOCCFLOW_PY" -B -m scripts.cache_research \
  --group train --label-manifest data/research/train/manifest.json \
  --history-visibility dense_semantic --device cuda:1 --rank 1 --world-size 2
```

tune，共2,176个预期窗口：

```bash
# 终端1
"$REACTOCCFLOW_PY" -B -m scripts.cache_research \
  --group tune --label-manifest data/research/tune/manifest.json \
  --history-visibility dense_semantic --device cuda:0 --rank 0 --world-size 2

# 终端2
"$REACTOCCFLOW_PY" -B -m scripts.cache_research \
  --group tune --label-manifest data/research/tune/manifest.json \
  --history-visibility dense_semantic --device cuda:1 --rank 1 --world-size 2
```

输出目录自动选择新模式对应的数据组。保留同设置即可断点续建；只有完整窗口计数、样本校验和两个rank均通过，manifest才发布`complete: true`。旧模式可显式使用`--history-visibility mask_lidar`；修改后的生成器源码身份与旧缓存不同，不能用它原地续建旧归档，旧实验复现仍使用匹配的冻结源码。

后续新训练应明确使用新缓存和新的实验目录。当前没有提供自动启动或“缓存完成后继续训练”的机制。

## 验证与结论边界

合成数据覆盖密集历史保留不可见但已知体素、缺失/改变sensor mask无影响、未知/越界屏蔽、原mask模式数值回归、actor与未来目标不变、未来信息隔离、在线/离线一致、混合缓存拒绝及旧检查点模式回退。源保护与独立安装验证产物统一保存在`outputs/verification/2026-11-16-dense-history/`。

最终全量测试**405项通过，失败/错误/跳过均为0**。专用环境安装的新wheel在`/tmp`下以`python -I -B`运行，116份包源码与工作区校验一致；禁止参考目录和原始数据目录读取的审计钩子未发现访问尝试。原冻结VAE完成合成密集输入编码，历史U形CFM与DiT512 EMA权重均严格加载成功。旧train/tune缓存仍可读取，窗口数分别为17,468/2,176。

两个参考仓库前后快照零差异；保护范围内79,918个既有文件的大小、修改时间和inode未变，另核验的原VAE、旧缓存清单/设置及STOP标记SHA-256不变。当前没有训练或缓存生成进程，新密集历史缓存目录不存在。证据见[最终验收](../../outputs/verification/2026-11-16-dense-history/acceptance.json)、[全量测试](../../outputs/verification/2026-11-16-dense-history/pytest-final.xml)、[隔离执行](../../outputs/verification/2026-11-16-dense-history/isolated_execution.json)。

首次定向测试有两个小网格fixture未覆盖warp函数默认网格的失败，修正fixture后通过；原始失败报告保留，不将它们解释为模型质量问题。

本次验收只证明输入规则及兼容性正确。没有新生成真实缓存、没有启动真实训练任务或正式预测评测，不能报告第一秒mIoU提升，也不改变原先各模型的历史分数。

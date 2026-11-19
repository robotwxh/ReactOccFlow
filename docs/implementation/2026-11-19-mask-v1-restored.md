# 恢复第一版mask_lidar

日期：2026-11-19。按用户“将mask还原到第一版”的要求，活动默认输入恢复为原始`mask_lidar`。

## 恢复后的行为

历史有效区域严格为：

```text
mask_lidar & in_bounds & (semantics != 255)
```

默认不做历史地面补全、邻域插补或mask扩张。不可见位置仍由原冻结VAE按已有协议在embedding前置为空类17；这只是编码近似，不把不可见区域视为真实free证据。

- 编码API默认`mask_lidar`；显式`visibility_key=None`仍代表密集历史。
- 缓存CLI默认`mask_lidar`，默认目录`data/research/e0_full/<group>`。
- 训练CLI默认`mask_lidar`，默认train/tune均选择`e0_full`；显式选择其他历史模式时，自动匹配对应缓存目录，显式缓存路径仍优先。
- 在线评测继续使用checkpoint记录的输入模式；无模式字段的旧joint权重默认原mask。历史密集/新mask权重保持原身份，不能通过改变默认值冒充原mask。
- 未来监督仍使用完整known/in-bounds语义，不施加历史mask；actor数据、冻结VAE、U形CFM与标准DiT结构不变。

新mask实验的显式兼容入口及冻结源码保留用于复现，默认流程已经退出该实验。三轮缓存、权重、评测和报告全部保留。

## 使用原mask结果

现有`data/research/e0_full/train`与`tune`缓存分别有17,468与2,176个窗口，可以在匹配的原mask输入模式下使用。本次未重新生成缓存、训练或评测。

原mask第10,000步EMA：

`outputs/history_input_comparison/2026-11-17-seed2026/checkpoints/mask_step10000_ema.pt`

它的正式Avg mIoU仍为27.9574%，本次代码恢复不改变已有分数。三轮完整对比见[评测记录](2026-11-19-ground-mask-three-way-evaluation.md)。

旧完整缓存绑定原生成源码与配置。需要重生成时应显式选用新的输出目录；现有身份校验继续阻止静默覆盖。历史训练恢复继续使用对应冻结源码与原状态。

## 核验

- 183项相关测试通过；覆盖默认路径确实保持隐藏地面不可见、显式密集输入、默认与显式原mask编码一致、未来监督不变、缓存/在线一致及checkpoint模式隔离。
- 原mask交集函数与第一版冻结源码的AST一致。
- 普通wheel安装到专用环境，并在`/tmp`下隔离执行；真实样本当前帧的默认VAE编码与原mask缓存比较，结果见机器核验。
- 31项受保护历史文件、三轮冻结源码及两个参考仓库相对本次操作前的状态均保持原状。raw STOP保持原位。

来源：[源码搜索与恢复记录](../provenance/mask_v1_restore_sources_20261119.json)。核验目录：`outputs/verification/2026-11-19-mask-v1-restore/`。

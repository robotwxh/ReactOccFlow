# 2026-11-15：回退原始 LiDAR 输入改造

用户要求回退以“输入原始 nuScenes、增加 actor 编码、occ/actor 同步输出”为起点的后续改造，
恢复此前的 occupancy / latent 输入路线。该决定撤销本轮 raw-only 输入和禁止旧权重初始化的限制。
全部训练继续暂停，本次没有启动或恢复训练、预处理或自动阶段推进。

## 恢复后的路径

```text
已有语义 occupancy → 原 OccupancyCodec（冻结 OccFM Swin-VAE）→ 历史 latent
已有 actor 标注/轨迹缓存 → 原 actor history MLP / 类别与尺寸编码
两路表示 → U形 CFM＋actor 或标准 DiT512＋actor → 未来 occupancy 与 actor
```

正式训练重新读取 `data/research/e0_full/train` 和 `data/research/e0_full/tune`，
分别保留 17,468 / 2,176 个窗口；使用原 train-only normalization。
编码器恢复为语义 occupancy 输入，actor 恢复为标注派生状态与历史编码。
原始点云检测/编码前端、原生 lidarseg 射线监督、raw scratch VAE 及其分阶段自动训练入口已从活动代码中移出。
旧缓存构建工具仍可显式读取原先的数据标注源；已有缓存训练和本次验证不读取 `~/data/nuscenes` 原始数据。

## 回退依据与保留范围

Git 最后提交 `86d6ef1` 日期为11月5日，比目标边界更早，不能用整仓 reset 来覆盖11月7日至11日的修改。
本次以 `outputs/evaluation/2026-11-11-dit512-best84000/evaluation_source` 的普通文件快照作为恢复依据。
该快照创建于11月11日10:57，早于原始 LiDAR 改造，153 个来源文件 SHA 已核验。

- 11 个代码/说明文件逐字节恢复，包括 actor 编码实现、perception 导出及七个旧训练/缓存/恢复入口。
- 恢复旧 CLI 的公开 `main`，移除 raw-only 拦截；修复对应测试调用。
- 保留此前的标准 DiT512 维度修复、评测修复、generation probe、两种主干及所有历史评测。
- `.gitignore` 恢复原规则；方案、模型说明、README、实施计划及进展中的 raw-only 路线说明已撤回。
- 新增 raw 模块、配置、测试、相关说明/来源记录和 metadata 移入回退档案。
- `data/nuscenes` 软链接已移除，`/home/lthpc/data/nuscenes` 实际数据目录及已修复点云保持原状。
- 专用环境重新安装回退后的普通 wheel，移除 raw 阶段引入的 `py-spy`；原环境文件已备份。

回退前524个项目文件及工作区差异保存在：
`outputs/rollback/2026-11-15-before-raw-input/before/` 和 `before.manifest.json`。

移出内容及原路径映射位于同目录的 `removed/`、`changes.json`；其中 raw corpus 和 sensor splits 作为历史 metadata 保存。
外部原始数据、历史权重、评测结果、旧源码导出和新 raw 训练输出均未删除。
raw 世界模型的9,422步 checkpoint 继续保存在原输出目录，停止标记也保留；不会自动混入已恢复的旧路线。
历史 raw 实验若需重现，必须先根据档案恢复对应源码和 metadata，再取得用户明确运行指示。

## 验证

- 387 项原路线 CPU 回归测试通过。
- 专用环境使用新普通 wheel，隔离执行验证记录为回退档案中的 `isolated_cache_inference_check.json`。
- 只读取旧 E0 缓存、项目自有 VAE/世界模型权重，验证原 occupancy VAE 和两种主干；不访问原始数据目录或参考仓库。
- 参考保护、历史产物、源码导出、暂停状态及最终回退状态的核验见 `acceptance.json`。

本文件只记录回退；不将单样本接口验证视为新的性能评测。

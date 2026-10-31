# 数据接入与研究管线验收

本轮根据用户给出的 `~/data`，使用独立的 `/home/lthpc/data/nuscenes`。项目不借助参考仓库的数据链接，也不修改原始数据。

## 完整原始标签与划分

| 用途 | 场景 | 原始关键帧 |
|---|---:|---:|
| 模型训练 |560|22508|
| 调参 |70|2806|
| 校准预留 |70|2816|
| 最终评测 |150|6019|
| 总计 |850|34149|

官方700/150场景集不变；700训练场景按`sha256("2026:"+scene)`排序后80/10/10划分。每个场景独占一个组。850个数值NPZ分片均验证SHA256、形状、有限值和观测/目标边界；未来有效actor状态1306692、存在性标签1389911。所有34149个occupancy文件存在。

`data/research/normalization.json`只读取训练组的有效历史观测，未读取调参/校准/评测数组。数据源9张官方表的内容哈希在`data/research/source_manifest.json`。逐帧actor存在性标签表示官方标注轨迹在该后续帧是否有记录，不是物理世界中目标是否彻底消失的证明。

## E0与时间

- x向前、y向左、z向上，米/秒；网格200×200×16、0.4m，范围[-40,40]×[-40,40]×[-1,5.4]。
- 本地Occ3D使用ego坐标；当前帧标注box对齐与ego/lidar对照检验支持该约定，不额外叠加LiDAR外参。支持显式的其他来源frame选项，不能自动混淆。
- 先做原始3D语义和可见性nearest warp，再VAE后验均值编码，禁止将2Dlatent warp冒充精确3D变换。出界为unknown255、不可见；编码时以empty17替代未知是预训练embedding限制，不是自由空间证据。future_known_mask单独保留。
- 当前/过去信息决定最近8个可观测actor，距离与token稳定排序，类别及LWH来自当前；速度使用向后差分。未来状态/存在性/真实ego轨迹均分离在target。
-33299条官方相邻边仅一条为0.750945s，超过最初0.75s阈值。默认上限改为0.8s，保留实测dt，不删除帧或伪造0.5s。未来6关键帧标称0.5–3s，实际future_time_s用于物理计算。

## 可视化与编码产物

`outputs/research_qa/`包含20个不同训练场景及总览，覆盖直行、转弯、行人、低可见性和无当前actor。coding agent检查了20格总览及3张细图；**没有冒充研究者人工复核**，检查记录在inspection.json。

20个样例的原始3D warp与重新编码已完成，保存在`data/research/e0_acceptance/`。全部数值有限、缺失槽与mask对应，未来字段不进入observation。131个与网格相交的当前actor box全部含有对应类别体素；这是坐标QA，不是预测性能。

## 独立性与边界

- 安装wheel后在无参考仓库的临时目录，以`python -I`运行真实权重训练/推理、几何、官方表标签、分组与E0编码通过；不使用editable install或路径注入。
- 参考源码与HEAD/原有Git状态未变。对Oct27原始快照：OccFM完全匹配；IR-WM多出一个mtime为Oct27 20:41的空data目录，早于本轮续跑。没有删除、重置或掩盖该差异。完整审计在reference_audit_p2.json。
- 所有观测仍是annotation/GT occupancy oracle，不是实际LiDAR前端。实际感知、反事实动作数据、风险校准和闭环均未实现。
- 原始兼容latent缓存恰好每场景缺少末帧；新研究标签含所有原始关键帧，两种数据口径分开。
- 现有预训练权重接触过官方700训练场景；新70/70组不能称为相对于这些权重独立未见。严格校准结论需要控制模型训练/预训练暴露。

## 全量E0缓存

完整4历史+6未来的训练anchor共17468个。新增生成器支持两rank各8734个，按anchor序号取模分工；每条记录普通NPZ+JSON、内容哈希、设置/源码/权重/统计与源GT绑定，原子发布，断点验证后续跑。只有所有rank的完整census验证通过才写complete:true，默认loader拒绝未完成缓存。

20个验收例子、4条初始pilot以及32条v2调试缓存都不能称为全量缓存。实际完成状态以对应manifest.json和rank日志为准；v2显式允许小样本调试后才能读未完成缓存。完整缓存生成属于后续训练的数据准备，不等于模型训练完成。

证据：research_data_audit.json、e0_encoding_audit.json、keyframe_timing_audit.json、isolation_p2.json、reference_audit_p2.json；详细实现与独立复审在`.superpowers/sdd/2026-10-27-reactoccflow-implementation-plan/`。

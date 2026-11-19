# 新mask流程恢复记录

> 后续完成：正式训练于2026-11-19 04:39达到10,000步，评测及三组对比于05:33完成。全部阶段退出0；新mask未改善other_flat或总体Avg，见[完整对比](2026-11-19-ground-mask-three-way-evaluation.md)。下文保留恢复时状态。

用户明确要求“训练继续进行”。已于**2026-11-18 17:14:33（北京时间）**恢复同一独立流程，控制器PID **586194**。

运行目录：`outputs/history_input_comparison/2026-11-18-ground-mask-seed2026/`。

## 已核验并保留的状态

- 暂停时的11,027条训练缓存逐条通过完整payload SHA、数值schema、模式和记录身份检查，没有损坏。完整检查耗时约190秒。
- 195个冻结源码文件、专用环境已安装包、源数据和缓存设置均一致，冻结指纹仍为`c6c6664c2fdf0bfe1c12c970aef73caa52f37c2391f47bcfc223326011080c07`。
- 18项受保护旧权重/结果保持原状，两个参考仓库无变化；隔离执行检查再次通过。
- 原`launch_config.json`字节不变，训练总预算10,000步，warmup1,000步；正式优化此前0步，不使用工程短跑2步权重。

恢复入口先复核并复用已有11,027条记录，补齐剩余6,441条训练缓存，再生成2,176条tune缓存，之后按原配置进行双卡U形CFM＋actor训练。训练从原OccFM主干初始化，VAE冻结。

达到10,000步后自动导出预先固定的最终EMA，执行官方150场景、4,219窗口评测，并与既有密集历史和原mask_lidar两个10,000步模型生成1/2/3秒总体及17类对比。

## 暂停历史与停止控制

此前pipeline、controller、阶段日志和本轮STOP标记已归档至：

`history/before_resume_cache_20261118T091433Z/`

暂停机器记录、缓存清单和未执行恢复计划保留；缓存rank原进度文件也另存于归档的`cache_state/`。仅归档本轮STOP，**raw流程的STOP标记保持原位且SHA不变**。

后续再次暂停仍使用本轮运行目录的`STOP_REQUESTED.json`；控制器遇到失败停止，不自动重试。

机器证据：[恢复回执](../../outputs/history_input_comparison/2026-11-18-ground-mask-seed2026/resumed_from_cache_11027_20261118.json)、[完整缓存核验](../../outputs/history_input_comparison/2026-11-18-ground-mask-seed2026/resume_cache_verified_20261118.json)、[源码与环境核验](../../outputs/history_input_comparison/2026-11-18-ground-mask-seed2026/resume_static_verified_20261118.json)、[数据核验](../../outputs/history_input_comparison/2026-11-18-ground-mask-seed2026/resume_data_verified_20261118.json)。

当前阶段和进度以`pipeline.json`为准；正式优化开始后读取`train/training.jsonl`。预计剩余缓存、训练和评测合计约12–13小时，随实际速度更新。

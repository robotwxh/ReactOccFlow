# 新mask流程暂停记录

> 后续恢复：用户明确要求继续。已于2026-11-18 17:14:33完成11,027条缓存完整核验并恢复原流程；暂停时STOP与运行日志已归档，raw STOP保留。见[恢复记录](2026-11-18-ground-mask-training-resumed.md)。下文保留暂停时状态。

用户要求“训练先停掉”。流程已于**2026-11-18 11:36:06（北京时间）**暂停，状态为`paused_by_user`。

- 暂停阶段：`cache_train`，正式优化尚未开始，**0 / 10,000步**。
- 保留新训练缓存**11,027 / 17,468**条已提交记录对：rank0为5,480条，rank1为5,547条。
- 控制器及两个缓存进程（264199、264201、264202）均已退出；缓存子进程按停止请求终止，实际退出码为−15/−15。
- tune缓存生成、正式训练、固定EMA导出、全量评测和三组对比均未启动，停止标记阻止后续推进。
- 所有已生成文件保留，没有孤立payload或临时文件。此处数量基于已提交的JSON/NPZ记录对；恢复时须按冻结缓存器重新执行payload校验和schema验证，不能仅凭数量跳过核验。
- 工程短跑的2步checkpoint保留在`smoke/`，不计入正式训练进度，也不作为正式训练初始化。
- 冻结源码、18项受保护历史文件及两个参考仓库保持不变；raw STOP标记保留。

运行目录：`outputs/history_input_comparison/2026-11-18-ground-mask-seed2026/`。本轮`STOP_REQUESTED.json`保持原位，**不自动恢复**。

获得明确继续指令后，先归档暂停时的pipeline、controller/阶段日志与本轮STOP标记，再核验同一冻结源码和数据身份。重启同一配置，复核并复用已有缓存、补齐剩余记录，然后依次生成tune缓存、从原初始化训练总计10,000步并完成后续评测/对比。无正式模型checkpoint需要恢复；不会从短跑2步继续。raw STOP不移动。

机器记录：[暂停状态](../../outputs/history_input_comparison/2026-11-18-ground-mask-seed2026/paused_by_user_20261118.json)、[缓存清单](../../outputs/history_input_comparison/2026-11-18-ground-mask-seed2026/paused_cache_inventory_20261118.json)、[未执行的恢复计划](../../outputs/history_input_comparison/2026-11-18-ground-mask-seed2026/resume_plan_20261118.json)。

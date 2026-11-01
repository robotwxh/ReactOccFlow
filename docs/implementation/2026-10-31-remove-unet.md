# 2026-10-31：移除 U-Net，仅保留标准 DiT

按用户要求，从当前项目的可运行代码、配置、CLI 和安装包中删除 U-Net。所有模型训练继续保持停止；本次只执行 CPU 单元测试与隔离推理，没有启动正式或诊断训练任务。

## 删除范围

- 删除 `FlowMatchingV2.py` 及其注册目录，移除 `default_trunk`、`_middle_forward` 和世界模型的多尺度 down/up/skip 路径。
- 删除旧 `ConditionalFlowMatching` 包装器和 `load_pretrained_trunk`。`models/flow.py` 仅保留仍由 DiT 使用的线性插值与 shifted 时间网格数学工具。
- 删除 `occfm_history.yaml`、`occfm_future.yaml`、旧单模型训练脚本、原基线评测模块及其专用测试。`python -m reactoccflow baseline` 不再是有效命令。
- 训练入口只接受 `--backbone dit`，删除旧主干初始化参数和分支。三个研究配置统一使用 `freeze_backbone: false`，不再含预训练世界主干路径。
- 删除已核对仅供旧网络使用的 ResNet/注意力/相对位置与包装层；VAE 所需的卷积、Swin、残差及时间层保留。
- 清除相关过期字节码及生成包副本，重新构建并安装干净 wheel，避免旧文件被重新打包。

精确文件与类清单、删除前 SHA-256、源提交、依赖检查和归档位置见 [来源记录](../provenance/remove_unet_sources.json)。原代码在删除前已归档于 `outputs/architecture_migration/2026-10-31-remove-unet/before_removal.tar.gz`；归档和原始源快照只作审计，不参与运行。

## 保留并适配的部分

世界模型 API 为 `ReactFlow` 与 `StandardDiT`，默认 1,200 个 token、256 维、8 层、8 头。`TokenInteraction` 接替原交互类名，在第 4 层 DiT 后保持 actor self-attention 和双向 cross-attention，Q/K/V 与残差更新逻辑不变。`joint`、`parallel` 和 `occupancy_only` 都继续支持。

单套 `OccupancyCodec`、独立 VAE 权重、E0 数据/缓存、连续端点一致性、规划/风险/感知/桥接接口继续使用。VAE decoder 可反传到 latent，VAE 参数保持冻结。`JointTrainingAdapter` 默认支持可训练 DiT；显式冻结检查只用于调用者指定的消融。

通用优化/探针/原子发布工具从旧脚本迁入 `reactoccflow.training.diagnostics`。联合训练 smoke 与接口/一致性检查脚本现构造随机初始化的标准 DiT，并使用独立的 `dit_*` 输出目录。它们没有在本次清理中启动训练。

历史 U-Net 的 50,000 步检查点、51,400 步日志、原有指标和来源记录保留；当前包不提供加载或续跑旧主干的入口。之前迁移验收的 JSON 与源码摘要仍属于当时版本，最新状态以本页与本次来源记录为准。

## 验证

- 完整 CPU 套件：337 passed，14.40 秒；覆盖标准 DiT、跨模态交互、mask/防泄漏、Flow Matching、VAE 和训练/数据接口。旧模型专用测试随旧接口删除，通用数学/优化测试已经适配。
- wheel：62 个条目；检查旧主干、旧包装器及基线评测模块均不存在。已安装到专用 ReactOccFlow 环境，没有修改参考项目环境。
- 隔离 CPU 推理：在 `/tmp` 用 `python -I` 加载安装包和独立复制的真实历史观测，完成固定随机数采样、严格 state_dict 保存加载、真实 VAE 编码和预测 latent 解码。没有运行优化器。见 [隔离记录](dit_only_isolation.json)。
- 源保护、最终源码摘要、CLI 检查和停止状态以 [清理来源/验收记录](../provenance/remove_unet_sources.json) 为准。旧的 350 项验收是初次 DiT 迁移的历史记录，不作为本次仍保留旧接口的证明。

后续实际训练继续使用 [README 中的 DiT 命令](../../README.md)。任何代码清理都会改变严格恢复所绑定的源码身份，不绕过 checkpoint 验证。新模型的研究预测、反应式、风险校准和闭环效果仍需独立实验。

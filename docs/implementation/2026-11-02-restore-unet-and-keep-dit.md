# 恢复原 U 形 DiT，并保留标准 DiT

> 后续状态：2026-11-03 已按用户新指示恢复原 CFM joint 双卡训练，从完整 50,000 步继续；标准 DiT 仍暂停。见 [续训记录](2026-11-03-cfm-training-resumed.md)。下文“训练暂停”和未执行命令描述为本次架构恢复当时的状态。

2026-11-02，按用户最新要求恢复原 OccFM 式 U 形时空 DiT，同时保留标准 isotropic DiT。此决定覆盖 2026-10-31 删除 U-Net 的运行时限制。全部训练继续暂停，没有自动启动旧实验、新实验或守护进程。

## 架构选择

| 选择 | 主干 | actor 双向交互位置 | 配置 |
|---|---|---|---|
| `occfm_unet`（恢复后的默认值） | 原卷积残差块、空间/时间 DiT、多尺度 down/up 与 skip | 原 `mid_block2` 之后、解码器之前 | `configs/unet.yaml`；`react_flow_{joint,parallel,occupancy_only}.yaml` |
| `dit` | 固定宽度、固定 token 数的全局时空 DiT | 中间 DiT Block 之后 | `configs/dit.yaml`；`react_flow_{joint,parallel,occupancy_only}_dit.yaml` |

两种主干共用一套冻结 Swin VAE、数据边界、RF 目标、actor 编码、存在预测、端点一致性和 future-only 求解器。CFM/RF 是训练目标，DiT/U 形是主干结构，恢复主干没有更换生成目标。

API 示例：

```python
from reactoccflow.models import ReactFlow

original = ReactFlow()  # backbone='occfm_unet'
standard = ReactFlow(backbone='dit')
```

显式传入标准 DiT trunk 或仅提供 `dit_config` 时仍可推断 DiT，以保留现有调用。主干与配置不匹配会报错。三个模式均支持两种主干。

## 训练入口（供后续明确恢复训练时使用）

原 U 形主干：

```bash
torchrun --standalone --nproc_per_node=2 --module scripts.train_react_flow \
  --backbone occfm_unet --unet-config configs/unet.yaml \
  --output outputs/reactflow_occfm_unet/seed_2026
```

标准 DiT：

```bash
torchrun --standalone --nproc_per_node=2 --module scripts.train_react_flow \
  --backbone dit --dit-config configs/dit.yaml \
  --output outputs/reactflow_dit_new/seed_2026
```

以上命令未执行。默认新实验输出按主干分目录，已有暂停目录不能作为新实验目录覆盖。正式 runner 会记录主干和对应的完整架构配置，严格检查恢复身份。

U 形主干默认随机初始化。若需要原 OccFM 世界主干初始化，显式增加 `--trunk-checkpoint assets/checkpoints/occfm_history.ckpt`；VAE/actor 权重不会因此被覆盖。`--trunk-weights state_dict|ema_model` 选择上游权重分区。该操作是初始化，不能与 `--resume` 同用；标准 DiT 拒绝加载 OccFM U 形主干。

## 历史权重与暂停实验

- U 形主干保留原 204 个 state_dict 键。历史 joint 权重也使用严格加载；不同主干的完整 state_dict 不混用。
- 原 U 形代码从已核验 SHA-256 的项目自有迁移前归档恢复，结合 `git log` 中 `3169d59`（原实现）和 `ee95563`（DiT 迁移）定位。未执行 git reset/checkout 覆盖当前工作。
- 为保留历史数值行为，U 形主干默认恢复旧 `flash` 注意力分支，包含历史 head-layout 行为；`attention_mode: math` 可用于另行记录的新实验。标准 DiT 的既有 math 分支未改变。切换后端属于数值实现变更，不能据此声称历史输出逐位不变。
- 标准 DiT 的 22,000 步暂停检查点及 `outputs/reactflow_dit/source_seed_2026/` 冻结源码没有修改。原 `scripts/resume_dit_training.py` 仍只使用该冻结副本，必须等用户明确恢复指示。
- 旧 U 形 50,000 步和 DiT 22,000 步的权重可分别用于对应架构的加载验证。完整续训仍受原配置、源码、缓存、优化器和 RNG 身份限制；工作区恢复架构不等于已恢复旧训练进程。

未恢复不属于本次联合架构所需的旧单模型 CFM wrapper 与 baseline CLI。恢复范围是原 U 形世界主干、中间交互路径、主干初始化接口，以及双架构训练/推理选择。

## 来源与验收

来源提交、归档/文件 SHA-256、两仓检索和依赖闭包见 [来源记录](../provenance/restore_unet_sources_20261102.json)。参考目录只读；所有运行依赖均为项目自有普通文件，保留原版权及未解决的 OccFM 授权边界。

- 完整 CPU 测试：347 passed（15.22 秒）。覆盖两个主干、三种模式、双向依赖、无 actor 退化路径、严格权重分区加载、架构错配拒绝和训练 runner。
- 安装隔离、真实权重加载、源保护及暂停身份检查结果见同目录的 `dual_backbone_isolation.json` 和 `dual_backbone_restoration_acceptance.json`；以实际生成的结果为准。

真实权重验收还包括：原 OccFM 主干 204 键、原 U 形 joint 50,000 步、标准 DiT 22,000 步均严格加载成功；两个主干在真实尺寸 CUDA 输入上通过有限梯度及重复采样检查，没有执行优化器步骤。安装隔离在临时环境、项目目录外使用真实 tune 观测及独立复制 VAE 完成；新 wheel 的 111 个运行时源码文件与工作区一致，并已仅重装到专用环境。参考目录当前快照和 HEAD 均保持原样。

这些是工程验收，不构成训练完成、预测精度提升或物理一致性改善的证据。

> 2026-11-15：已按用户要求撤回原始 LiDAR 输入及其后续改造，恢复 **语义 occupancy → 冻结原 VAE → latent／actor 状态缓存 → U形 CFM 或标准 DiT512**。新增点云感知/actor 检测/从头分阶段训练路线已移出活动代码，历史产物保留。**全部训练继续暂停，不自动恢复。**详见 [回退记录](../../docs/implementation/2026-11-15-rollback-raw-input.md)。下文运行状态按日期保留为历史。

# U-shaped / standard DiT and the occupancy VAE

`ReactFlow` supports `backbone='occfm_unet'` (restored default) and `backbone='dit'` (isotropic). The restored trunk is the independently migrated `FLOW_MATCHING_DOWN_X4_DiT`: convolutional residual blocks, spatial/temporal attention, down/up stages and skip connections, with actor exchange immediately after `mid_block2`. Both architectures retain the same observation/target boundary and three model modes. One frozen `OccupancyCodec` is shared. See [restoration](../../docs/implementation/2026-11-02-restore-unet-and-keep-dit.md) for configuration, legacy weights and numerical compatibility.

## Standard DiT branch and actor interaction

`ReactFlow(backbone="dit")` constructs a standard isotropic DiT: patch projection → eight
fixed-width DiT blocks → conditioned output projection → unpatchify. Default
inputs are 12 temporal slots of 16×50×50 latents. Spatial patch size 5 gives
1,200 tokens, width 512, eight heads and MLP ratio 4. All occupancy tokens
participate in global space-time self-attention. There are no down/up stages
or encoder-to-decoder skips in the world model.

The direct linear patch-velocity head requires `hidden_size >= 16*patch_size**2`.
The constructor rejects narrower tokens: for patch5 the patch dimension is400,
so historical width256 permanently trapped144 independent Gaussian directions.
Width512 removes that forced rank bound; it does not guarantee semantic quality.
Standard adaLN/output zero initialization is retained. Initial rank0 is expected
and differs from the dimension bound of a trained output head.

The trainer logs globally reduced per-component tune losses and a fixed,
identity-seeded eight-scene tune generation probe (native E0 mIoU at1/2/3s,
latent moments and learned output rank). This diagnostic is not the official
held-out benchmark and preserves training RNG, module modes and gradient flags.

Actor history/state/condition embeddings form 48 tokens (8 actors×6 future
slots), width 128. Two `TokenInteraction` modules follow DiT block 4:

| Attention | Query | Keys and values | Updated modality |
| --- | --- | --- | --- |
| Occupancy self-attention | occupancy | occupancy | occupancy |
| Masked actor self-attention | actor | actor | actor |
| Occupancy-to-actor cross-attention | actor | occupancy | actor |
| Actor-to-occupancy cross-attention | occupancy | actor | occupancy |

Each interaction runs actor self-attention, simultaneous bidirectional
cross-attention with gated residuals, and an actor FFN. Occupancy is projected
512→128; only the exchanged delta is projected 128→512 into the original DiT
sequence, followed by blocks 5–8. Modalities are separate token sequences.
`joint` enables exchange, `parallel` disables exchange, and `occupancy_only`
skips actors. Valid actor keys come from current observations, never future
existence or target masks. Ego plans and generative time condition adaLN and
the actor embeddings.

DiT uses the copied block's correctly merged math attention path. Its adaLN
and output layers are zero-initialized, so initial occupancy velocity is zero.
Use `dit_config` or the training CLI's `--dit-config configs/dit.yaml` to set
architecture options. `freeze_backbone=False` is the default; the optional
`freeze_trunk()` method supports explicit inference/ablation setups. Only the restored U-shaped branch supports explicit pretrained trunk loading.

## One independent occupancy VAE

`OccupancyCodec.encode(occupancy, visibility=None)` accepts B[,T],200,200,16
semantic labels and returns posterior-mean B[,T],16,50,50 latents by default.
Optional stochastic posterior sampling requires `sample=True` and an explicit
local `torch.Generator`. A visibility mask replaces hidden/unknown labels with
empty class 17 for encoding; this is not evidence of free space.

`decode(latents)` returns semantic labels. `return_logits=True` returns
B[,T],200,200,16,18 continuous class similarities. The copied Swin codec,
its convolutional/residual/temporal layers and their weight names are retained;
these shared codec components are not the removed world-model U-Net.

Offline caching encodes history observations and future training labels using
the same VAE. Training reads cached latents and decodes predicted endpoints
for consistency losses. Patch projection maps 400 latent values to a 512-dim
token; unpatchify restores latent shape. Neither operation is a second VAE.
Actor tokens use state MLPs. The frozen VAE decoder still permits gradients
to its input latent; its parameters are not optimized.

## Flow and checkpoint boundary

`ReactFlow.sample` accepts observed history latents and a separately supplied
candidate plan. Future occupancy and actor states use independent Gaussian
noise and a shared generative clock; Euler updates leave observed history
fixed. Latents are scaled by 10 inside the flow and unscaled on output.
Random draws use explicit local generators. Training targets never enter the
sampling interface; predicted future existence comes from its own head.

The formal trainer saves complete DiT/actor model, EMA, optimizer, scheduler,
RNG and identity metadata. Resume requires identical configuration, data and
source identity. U-shaped weights require the matching U-shaped branch. Historical 256-wide DiT
checkpoints must be replayed with their original frozen source, while the
512-wide branch starts a separate experiment and cannot resume those weights.

See [removal and validation](../../docs/implementation/2026-10-31-remove-unet.md),
[standard DiT architecture](../../docs/implementation/2026-10-31-standard-dit.md),
[source provenance](../../docs/provenance/remove_unet_sources.json) and
[architecture configuration](../../configs/dit.yaml).

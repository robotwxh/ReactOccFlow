# Standard DiT and the occupancy VAE

The runnable world model is `ReactFlow`, exported with `StandardDiT` and
`OccupancyCodec` from `reactoccflow.models`. The former U-Net world model,
compatibility wrapper, dedicated configurations and baseline commands have
been removed. Historical weights, logs and source snapshots are audit material.

## DiT and actor interaction

`ReactFlow()` constructs a standard isotropic DiT: patch projection → eight
fixed-width DiT blocks → conditioned output projection → unpatchify. Default
inputs are 12 temporal slots of 16×50×50 latents. Spatial patch size 5 gives
1,200 tokens, width 256, eight heads and MLP ratio 4. All occupancy tokens
participate in global space-time self-attention. There are no down/up stages
or encoder-to-decoder skips in the world model.

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
256→128; only the exchanged delta is projected 128→256 into the original DiT
sequence, followed by blocks 5–8. Modalities are separate token sequences.
`joint` enables exchange, `parallel` disables exchange, and `occupancy_only`
skips actors. Valid actor keys come from current observations, never future
existence or target masks. Ego plans and generative time condition adaLN and
the actor embeddings.

DiT uses the copied block's correctly merged math attention path. Its adaLN
and output layers are zero-initialized, so initial occupancy velocity is zero.
Use `dit_config` or the training CLI's `--dit-config configs/dit.yaml` to set
architecture options. `freeze_backbone=False` is the default; the optional
`freeze_trunk()` method supports explicit inference/ablation setups. The removed
pretrained world-backbone loader is not part of the API.

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
for consistency losses. Patch projection maps 400 latent values to a 256-dim
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
source identity. Removed U-Net weights cannot be loaded by this package.
Historical source archives retain the old code solely for provenance.

See [removal and validation](../../docs/implementation/2026-10-31-remove-unet.md),
[standard DiT architecture](../../docs/implementation/2026-10-31-standard-dit.md),
[source provenance](../../docs/provenance/remove_unet_sources.json) and
[architecture configuration](../../configs/dit.yaml).

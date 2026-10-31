# Pretrained occupancy baseline

`OccupancyCodec` reproduces the copied OccFM VAE module names and operations.
`encode(occupancy, visibility=None)` returns the posterior mean by default.
Use `sample=True, generator=torch.Generator(device=...).manual_seed(seed)` for
the upstream stochastic posterior. Inputs use B[,T],H,W,Z labels; outputs use
B[,T],16,50,50 at the pretrained 200 by 200 resolution. `decode` returns labels;
`return_logits=True` returns B[,T],H,W,Z,18 similarities.
The optional visibility mask substitutes empty class 17 in hidden voxels,
without adding a channel absent from pretrained weights.

`ConditionalFlowMatching(trajectory_length=6)` uses historical ego trajectory;
length 12 loads the future-trajectory checkpoint. `sample(history, trajectory,
initial_noise=noise, nfe=10, device='cuda:0')` accepts unscaled codec latents,
multiplies history by 10, integrates future noise in scaled flow coordinates,
and divides future output by 10. Alternatively supply `future_shape` and an
explicit local generator. Sampling consumes no global RNG. History and future
are separate tensors, and every Euler step leaves the observed history fixed.
For the pretrained backbone, history + future must total 12 slots at 50x50;
the wrapper supports unequal lengths for other backbones but does not invent
new temporal embeddings for the checkpoint. Both batch 1 and batch 2 work.

Classifier-free guidance blanks history in the unconditional branch but
retains the exact same ego trajectory in both branches, as upstream does.
It is not classifier-free guidance over ego trajectory. Users must provide
observed ego trajectories for the historical checkpoint, or explicitly
supplied candidate ego trajectories for future-conditioned prediction.
No ground-truth future occupancy is accepted by the sampling API.

The shifted grid is exactly `(1 - alpha*t/(1+(alpha-1)*t)).flip(0)` on a linear
0..1 grid. Velocity is evaluated at current time multiplied by 1000.
The default guidance scale is 2 and alpha is 3. This is the compatibility
sampler itself, so a separately named compatibility sampler is unnecessary.

Checkpoint loading strips repeated `module.` and `model.` wrappers and then
loads strictly. CFM loading selects exactly `transition_model.*`, reports
excluded compressor keys, rejects unknown partitions, and strictly checks
all selected keys. EMA selection is explicit: pass `weights="ema_model"` to either
loader to select the EMA partition and discard only its `n_averaged` counter.
The default is `weights="state_dict"`, matching the upstream recovery API.
Use an explicit converted tensor dict for other formats. Checkpoints load on CPU with `weights_only=True`.

Known upstream numerical concerns are deliberately preserved:

* `DiTAttention` reshapes head-major SDPA output B,H,N,D directly into B,N,C.
  It does not transpose heads and tokens first. The CPU fallback retains the
  same reshape and has a regression test.
* The backbone adds `x + init_temporal_attn(x)` even though that submodule
  already includes a residual; the additional skip is preserved.
* Frozen spatial positional embeddings use a 51x51 grid truncated to 2500
  positions rather than constructing a 50x50 grid; the pretrained tensors and
  constructor behavior are preserved.
* Copied `VaeQuant.get_loss` stores posterior standard deviations in a field
  named `logvars`, making its KL expression suspect. Bootstrap inference uses
  explicit posterior mean/sample encoding and does not call this training loss.
* The original video latent decoder and height/semantic channel ordering are
  preserved, including the decoder's channel-last upsampling intermediates.

Adaptations are package-relative imports, dynamic embedding height indexing,
CPU-capable SDPA backend selection with the original output layout, explicit
local generators, future-only integration with batch-safe times, and inference
wrappers that avoid trainer, OpenMMLab and Lightning dependencies.

Source `.txt` snapshots and YAML files are documentation of copied reference
implementations only; no source snapshot is imported or executed. Original
commits and SHA256 values are in `docs/provenance/model_sources.json`.

For training smoke checks, `training_loss(history, target, trajectory=None,
generator=local_generator)` accepts unscaled history and training targets,
returns a differentiable scalar future-only MSE, and preserves upstream
sigmoid-normal time sampling, 0.2 per-example historical conditioning dropout,
scaled targets and noise velocity. Targets enter this training API only; the
sampling API has no target parameter. A supplied local generator controls every
random draw. Set constructor `uncond_p` to configure conditioning dropout.

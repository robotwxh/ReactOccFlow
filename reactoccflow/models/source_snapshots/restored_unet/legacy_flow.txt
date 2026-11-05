"""Future-only conditional flow matching with the original OccFM sampler math."""
import torch
from torch import nn
from .modules.transition_models.FlowMatchingV2 import FLOW_MATCHING_DOWN_X4_DiT
from ._utils import read_state_dict, strict_load


def interpolate(noise, target, time):
    """The upstream straight training path: t*target + (1-t)*noise."""
    if noise.shape != target.shape:
        raise ValueError('noise and target shapes must match')
    time = torch.as_tensor(time, device=noise.device, dtype=noise.dtype)
    if time.ndim == 1:
        time = time.reshape(-1, *([1] * (noise.ndim - 1)))
    return time * target + (1 - time) * noise


def shifted_time_grid(nfe, alpha=3., *, device=None, dtype=torch.float32):
    if not isinstance(nfe, int) or nfe < 1 or alpha <= 0:
        raise ValueError('nfe must be positive integer and alpha positive')
    time = torch.linspace(0, 1, nfe + 1, device=device, dtype=dtype)
    return (1 - alpha * time / (1 + (alpha - 1) * time)).flip(0)


class ConditionalFlowMatching(nn.Module):
    """Inputs are unscaled codec history; output is unscaled future latents.

    initial_noise is in the original flow coordinates (unit Gaussian), while
    history is multiplied by latent_scale. CFG blanks history, retaining ego
    trajectory in both branches exactly as the upstream sampler does.
    """
    def __init__(self, transition_model=None, *, trajectory_length=6, latent_scale=10.,
                 guidance_scale=2., alpha=3., time_scalar=1000., nfe=10, uncond_p=.2):
        super().__init__()
        self.transition_model = transition_model if transition_model is not None else FLOW_MATCHING_DOWN_X4_DiT(
            in_channels=16, out_channels=16, model_channels=128, channel_multi=[2, 4],
            input_size=[50, 50], trajectory_length=trajectory_length, init_kernel_size=7,
            init_3d_conv_channels=64, attn_dim=32, temporal_attn_head=8, spatial_attn_head=8)
        self.latent_scale = latent_scale
        self.guidance_scale = guidance_scale
        self.alpha = alpha
        self.time_scalar = time_scalar
        self.nfe = nfe
        self.uncond_p = uncond_p

    def load_pretrained(self, path, *, weights="state_dict"):
        state = read_state_dict(path, weights=weights)
        # Full upstream world-model checkpoints also contain compressor keys.
        # Select the named transition_model partition, then require exact keys.
        prefix = 'transition_model.'
        if any(key.startswith(prefix) for key in state):
            selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            unrelated = [key for key in state if not key.startswith(prefix)]
            unexpected = [key for key in unrelated if not key.startswith(('embedding.', 'encoder.', 'quantization.', 'decoder.'))]
            if unexpected:
                raise RuntimeError(f'Unexpected checkpoint partitions: {unexpected}')
        else:
            selected, unrelated = state, []
        report = strict_load(self.transition_model, selected)
        report['excluded_compressor_keys'] = unrelated
        return report

    def training_loss(self, history, target, trajectory=None, *, generator=None):
        """Source-equivalent CFM loss; targets are used only during training.

        The scalar MSE covers predicted future slots exclusively. All random
        draws use the supplied local generator on the input device.
        """
        if generator is None:
            raise ValueError('training_loss requires an explicit local generator')
        if history.ndim != 5 or target.ndim != 5 or history.shape[0] != target.shape[0] or history.shape[2:] != target.shape[2:]:
            raise ValueError('history and target must be B,T,C,H,W with matching batch and feature shapes')
        if history.device != target.device:
            raise ValueError('history and target must use the same device')
        batch = history.shape[0]
        keep = (torch.rand(batch, device=history.device, generator=generator) > self.uncond_p)
        keep = keep.reshape(batch, 1, 1, 1, 1).to(history.dtype)
        scaled_history = history * self.latent_scale * keep
        scaled_target = target * self.latent_scale
        time = torch.sigmoid(torch.randn((batch, 1, 1, 1, 1), device=history.device,
                                        generator=generator, dtype=history.dtype))
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        noised = interpolate(noise, scaled_target, time)
        inputs = torch.cat((scaled_history, noised), 1)
        if trajectory is not None:
            if trajectory.ndim != 3 or trajectory.shape[0] != batch or trajectory.shape[-1] != 2:
                raise ValueError('trajectory shape must be B,T,2')
            trajectory = trajectory.to(device=history.device, dtype=history.dtype)
        prediction = self.transition_model({'noised_sequence': inputs,
            'timesteps': time.reshape(batch) * self.time_scalar, 'trajectory': trajectory})['predicted_latent']
        if prediction.shape != inputs.shape:
            raise ValueError('transition model must predict velocity for every input slot')
        future_velocity = prediction[:, history.shape[1]:]
        return torch.nn.functional.mse_loss(future_velocity, scaled_target - noise)

    @torch.no_grad()
    def sample(self, history, trajectory=None, *, initial_noise=None, future_shape=None,
               generator=None, nfe=None, device=None):
        if history.ndim != 5:
            raise ValueError('history shape must be B,T,C,H,W')
        device = history.device if device is None else torch.device(device)
        self.to(device)
        history = history.to(device)
        if initial_noise is None:
            if generator is None or future_shape is None:
                raise ValueError('Provide initial_noise or both future_shape and a local generator')
            initial_noise = torch.randn(tuple(future_shape), generator=generator, device=device, dtype=history.dtype)
        future = initial_noise.to(device=device, dtype=history.dtype).clone()
        if future.ndim != 5 or future.shape[0] != history.shape[0] or future.shape[2:] != history.shape[2:]:
            raise ValueError('future noise and history must match in batch, channels and spatial shape')
        if trajectory is not None:
            if trajectory.ndim != 3 or trajectory.shape[0] != history.shape[0] or trajectory.shape[-1] != 2:
                raise ValueError('trajectory shape must be B,T,2')
            trajectory = trajectory.to(device=device, dtype=history.dtype)
            trajectory = torch.cat((trajectory, trajectory), 0)
        conditioned = history * self.latent_scale
        steps = self.nfe if nfe is None else nfe
        # Float32 steps reproduce the source time grid even under mixed precision.
        grid = shifted_time_grid(steps, self.alpha, device=device)
        history_length = history.shape[1]
        for current, following in zip(grid[:-1], grid[1:]):
            conditional = torch.cat((conditioned, future), 1)
            unconditional = torch.cat((torch.zeros_like(conditioned), future), 1)
            inputs = torch.cat((unconditional, conditional), 0)
            result = self.transition_model({
                'noised_sequence': inputs,
                'timesteps': (current * self.time_scalar).expand(inputs.shape[0]),
                'trajectory': trajectory})
            velocity = result['predicted_latent']
            if velocity.shape != inputs.shape:
                raise ValueError('transition model must predict velocity for every input slot')
            unconditional_v, conditional_v = velocity.chunk(2, 0)
            guided = unconditional_v + self.guidance_scale * (conditional_v - unconditional_v)
            future = future + (following - current) * guided[:, history_length:]
        return future / self.latent_scale

"""T7 differentiable consistency on estimated endpoints in the shared E0 frame.

These are research regularizers, not validated physical bounds. Geometry is
local to currently observed selected actors: a fixed target-box ROI must never
be interpreted as an exhaustive dynamic-world actor union. All target values
remain in this loss module, never in the inference interface. Search/provenance:
``docs/provenance/consistency_sources.json``.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from reactoccflow.geometry import GridSpec


def _tensor(data, key, shape, reference, *, kind='float'):
    value = data.get(key)
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{key} shape must be {tuple(shape)}')
    if kind == 'integer' and (value.is_floating_point() or value.is_complex()):
        raise ValueError(f'{key} must contain integer class IDs')
    dtype = {'float': reference.dtype, 'bool': torch.bool, 'integer': torch.long}[kind]
    return value.to(device=reference.device, dtype=dtype)


def _finite(value, mask, name):
    if not torch.isfinite(value[mask.expand_as(value)]).all():
        raise ValueError(f'Valid {name} values must be finite')


def _heading(state, eps):
    """Stable sin/cos unit heading with x-forward fallback for zero vectors."""
    vector = state[..., 2:4]
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    fallback = torch.stack((torch.zeros_like(norm[..., 0]), torch.ones_like(norm[..., 0])), -1)
    return torch.where(norm > eps, vector / norm.clamp_min(eps), fallback)


def _mean(values, mask, *, statistics=None, name=None, weight=1.):
    mask = mask.expand_as(values)
    # Large voxel/class reductions overflow half even when every term is <=1.
    if values.dtype in (torch.float16, torch.bfloat16):
        values = values.float()
    numerator = torch.where(mask, values, torch.zeros_like(values)).sum()
    count = mask.sum()
    if statistics is not None:
        statistics[name] = {'numerator': numerator.detach(), 'count': count.detach(),
                            'weight': float(weight)}
    return numerator / count.clamp_min(1)


def decode_endpoints(codec, latents, *, chunk=1):
    """Freeze/eval a codec and decode continuous logits, retaining latent grads.

    ``latents`` is B,F,C,H,W. Chunking limits decoder activation batch size but
    retains its backward graph; it is not inference/no_grad decoding.
    """
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError('chunk must be a positive integer')
    if not torch.is_tensor(latents) or not latents.is_floating_point() or latents.ndim != 5:
        raise ValueError('latents must be floating B,F,C,H,W')
    if min(latents.shape) < 1:
        raise ValueError('latents dimensions must be nonempty')
    codec.requires_grad_(False).eval()
    flat = latents.flatten(0, 1)
    logits = torch.cat([codec.decode(part, return_logits=True) for part in flat.split(chunk)], 0)
    return logits.reshape(*latents.shape[:2], *logits.shape[1:])


class ConsistencyLosses(nn.Module):
    """Continuous geometry, static temporal, and class-related kinematics.

    Forward inputs: logits B,F,X,Y,Z,18; actor states B,A,F,6 in metre/second
    coordinates ordered x,y,sin_yaw,cos_yaw,vx,vy; observed actor metadata; and
    separately supplied training targets. Required targets include aligned raw
    ``future_semantics``, ``future_known_mask``, ``future_latent_mask``,
    ``actor_future``, ``actor_future_z``, ``actor_target_mask``,
    ``actor_velocity_mask``, and measured ``future_time_s``. No nominal dt fallback.

    Geometry uses current observed LWH/z and selected semantic class, with a
    target future XYZ/heading ROI plus ``roi_margin`` metres. Each valid actor
    and time has equal weight irrespective of its box/ROI voxel count. Full
    vehicle lateral penalties apply to classes 3,4,5,9,10; cycle classes 2,6
    receive ``cycle_heading_weight``; pedestrians have no alignment penalty.
    Thresholds/scales/weights are editable hypotheses, not empirical limits.
    """
    def __init__(self, *, grid=GridSpec(), temperature=.4, roi_margin=1.,
                 velocity_scale=10., low_speed_threshold=.5, cycle_heading_weight=.5,
                 occ_actor_weight=.1, temporal_weight=.05, kinematics_weight=.01,
                 eps=1e-6):
        super().__init__()
        if not isinstance(grid, GridSpec):
            raise ValueError('grid must be a GridSpec')
        positive = (temperature, velocity_scale, eps)
        nonnegative = (roi_margin, low_speed_threshold, cycle_heading_weight,
                       occ_actor_weight, temporal_weight, kinematics_weight)
        if not all(math.isfinite(x) and x > 0 for x in positive) or not all(math.isfinite(x) and x >= 0 for x in nonnegative):
            raise ValueError('Temperatures/scales/eps must be positive; margins/thresholds/weights nonnegative')
        self.grid, self.temperature, self.roi_margin = grid, temperature, roi_margin
        self.velocity_scale, self.low_speed_threshold = velocity_scale, low_speed_threshold
        self.cycle_heading_weight = cycle_heading_weight
        self.occ_actor_weight, self.temporal_weight, self.kinematics_weight = occ_actor_weight, temporal_weight, kinematics_weight
        self.eps = eps

    def forward(self, occ_logits, actor_future, observation, target, *, return_statistics=False):
        """Optionally expose detached sum/count statistics for global validation.

        Geometry counts eligible actor-frame ROIs; temporal consistency counts
        static voxel-class entries. Velocity and heading alignment have distinct
        masks, so each is exported separately with ``kinematics_weight``.
        """
        if not torch.is_tensor(occ_logits) or not occ_logits.is_floating_point() or occ_logits.ndim != 6:
            raise ValueError('occ_logits must be floating B,F,X,Y,Z,18')
        batch, frames = occ_logits.shape[:2]
        if tuple(occ_logits.shape[2:]) != (*self.grid.shape, 18):
            raise ValueError('occ_logits XYZ grid must match GridSpec and have 18 semantic classes')
        if not torch.is_tensor(actor_future) or not actor_future.is_floating_point() or actor_future.ndim != 4 or actor_future.shape[0] != batch or actor_future.shape[2:] != (frames, 6):
            raise ValueError('actor_future must be floating B,A,F,6')
        if actor_future.device != occ_logits.device:
            raise ValueError('Actor endpoints and occupancy logits must share a device')
        if not torch.isfinite(occ_logits).all():
            raise ValueError('Occupancy logits must be finite')
        actors = actor_future.shape[1]
        observed = observation.get('actor_observed_mask')
        if not torch.is_tensor(observed) or observed.ndim != 3 or observed.shape[:2] != (batch, actors) or observed.shape[2] < 1:
            raise ValueError('actor_observed_mask must be B,A,H with a current observation slot')
        hist_frames = observed.shape[-1]
        observed = observed.to(device=occ_logits.device, dtype=torch.bool)
        selected = observed[..., -1]
        history = _tensor(observation, 'actor_history', (batch, actors, hist_frames, 6), actor_future)
        sizes = _tensor(observation, 'actor_size', (batch, actors, 3), actor_future)
        classes = _tensor(observation, 'actor_class', (batch, actors), actor_future, kind='integer')
        current_z = _tensor(observation, 'actor_z', (batch, actors), actor_future)
        poses = _tensor(target, 'actor_target_mask', (batch, actors, frames), actor_future, kind='bool') & selected[..., None]
        velocities = _tensor(target, 'actor_velocity_mask', poses.shape, actor_future, kind='bool') & poses
        target_actor = _tensor(target, 'actor_future', actor_future.shape, actor_future)
        target_z = _tensor(target, 'actor_future_z', poses.shape, actor_future)
        future_times = _tensor(target, 'future_time_s', (batch, frames), actor_future)
        frame_mask = _tensor(target, 'future_latent_mask', (batch, frames), occ_logits, kind='bool')
        known = _tensor(target, 'future_known_mask', occ_logits.shape[:-1], occ_logits, kind='bool') & frame_mask[..., None, None, None]
        semantics = _tensor(target, 'future_semantics', known.shape, occ_logits, kind='integer')
        if ((semantics[known] < 0) | (semantics[known] > 17)).any():
            raise ValueError('Known future_semantics class IDs must be in [0,17]')
        if ((classes[selected] < 0) | (classes[selected] > 17)).any():
            raise ValueError('Selected actor_class must be in [0,17]')
        _finite(sizes, selected[..., None], 'current actor_size')
        if (sizes[selected] <= 0).any():
            raise ValueError('Selected actor_size length,width,height must be positive')
        _finite(current_z, selected, 'current actor_z')
        _finite(history[..., -1, :2], selected[..., None], 'current actor position')
        _finite(target_actor[..., :4], poses[..., None], 'target actor pose')
        _finite(target_z, poses, 'target actor_future_z')
        _finite(actor_future[..., :4], poses[..., None], 'predicted actor pose')
        _finite(actor_future[..., 4:], velocities[..., None], 'predicted actor velocity')
        _finite(future_times, poses.any(1), 'future_time_s')
        # Sanitize masked values before subtraction, heading normalization or dt.
        safe_actor = torch.where(poses[..., None], actor_future, torch.zeros_like(actor_future))
        safe_actor = torch.cat((safe_actor[..., :4], torch.where(velocities[..., None], safe_actor[..., 4:], torch.zeros_like(safe_actor[..., 4:]))), -1)
        target_actor = torch.where(poses[..., None], target_actor, torch.zeros_like(target_actor))
        target_z = torch.where(poses, target_z, torch.zeros_like(target_z))
        history_current = torch.where(selected[..., None], history[..., -1, :2], torch.zeros_like(history[..., -1, :2]))
        # A single finite element anchors an empty gradient path without a
        # potentially overflowing full-tensor reduction in float16.
        zero = occ_logits.reshape(-1)[:1].sum() * 0 + safe_actor.reshape(-1)[:1].sum() * 0
        probabilities = occ_logits.softmax(-1)
        statistics = {} if return_statistics else None
        geometry = self._geometry(probabilities, safe_actor, target_actor, target_z,
                                  sizes, current_z, classes, poses, known, zero,
                                  statistics=statistics)
        static = (semantics[:, 1:] == semantics[:, :-1]) & (semantics[:, 1:] >= 11) & (semantics[:, 1:] <= 16)
        static = static & known[:, 1:] & known[:, :-1]
        temporal = _mean((probabilities[:, 1:, ..., 11:17] - probabilities[:, :-1, ..., 11:17]).square(),
                         static[..., None], statistics=statistics,
                         name='temporal_consistency', weight=self.temporal_weight) + zero
        kinematics = self._kinematics(safe_actor, history_current, selected, classes,
                                      poses, velocities, future_times, zero,
                                      statistics=statistics)
        total = self.occ_actor_weight * geometry + self.temporal_weight * temporal + self.kinematics_weight * kinematics
        losses = {'total': total, 'occ_actor_consistency': geometry,
                  'temporal_consistency': temporal, 'kinematics': kinematics}
        if return_statistics:
            losses['statistics'] = statistics
        return losses

    def _geometry(self, probabilities, actor, target_actor, target_z, sizes, current_z, classes, poses, known, zero, *, statistics=None):
        entries = []
        pred_heading, gt_heading = _heading(actor, self.eps), _heading(target_actor, self.eps)
        axes = [torch.arange(n, device=actor.device, dtype=actor.dtype).add(.5).mul(step).add(origin)
                for n, step, origin in zip(self.grid.shape, self.grid.voxel_size, self.grid.origin)]
        # Loop only over supervised actor/frames, never a union of eight slots.
        for b, a, f in poses.nonzero(as_tuple=False).tolist():
            half = sizes[b, a] / 2
            centre = torch.cat((target_actor[b, a, f, :2], target_z[b, a, f, None]))
            sin_gt, cos_gt = gt_heading[b, a, f].unbind()
            extent = torch.stack((cos_gt.abs() * half[0] + sin_gt.abs() * half[1],
                                  sin_gt.abs() * half[0] + cos_gt.abs() * half[1], half[2])) + self.roi_margin
            # These are *valid grid indices*, not clamped actor coordinates.
            indices = [((axis >= c - e) & (axis <= c + e)).nonzero().flatten()
                       for axis, c, e in zip(axes, centre, extent)]
            if any(index.numel() == 0 for index in indices):
                continue
            ix, iy, iz = torch.meshgrid(*indices, indexing='ij')
            points = torch.stack((axes[0][ix], axes[1][iy], axes[2][iz]), -1)
            offset = points - centre
            local_gt = torch.stack((offset[..., 0] * cos_gt + offset[..., 1] * sin_gt,
                                    -offset[..., 0] * sin_gt + offset[..., 1] * cos_gt,
                                    offset[..., 2]), -1)
            roi = (local_gt.abs() <= half + self.roi_margin).all(-1) & known[b, f, ix, iy, iz]
            if not roi.any():
                continue
            point = points[roi]
            sin_pred, cos_pred = pred_heading[b, a, f].unbind()
            dx, dy = (point[:, :2] - actor[b, a, f, :2]).unbind(-1)
            local = torch.stack((dx * cos_pred + dy * sin_pred,
                                 -dx * sin_pred + dy * cos_pred,
                                 point[:, 2] - current_z[b, a]), -1)
            support = torch.sigmoid((half - local.abs()) / self.temperature).prod(-1)
            semantic_probability = probabilities[b, f, ix[roi], iy[roi], iz[roi], classes[b, a]]
            # Accumulate Dice in float32 for mixed-precision voxel logits.
            if support.dtype in (torch.float16, torch.bfloat16):
                support = support.float()
            if semantic_probability.dtype in (torch.float16, torch.bfloat16):
                semantic_probability = semantic_probability.float()
            # Both sides remain continuous, so each branch receives gradients.
            dice = (2 * (support * semantic_probability).sum() + self.eps) / (support.sum() + semantic_probability.sum() + self.eps)
            entries.append(1 - dice)
        stacked = torch.stack(entries) if entries else None
        if statistics is not None:
            statistics['occ_actor_consistency'] = {
                'numerator': stacked.detach().sum() if entries else zero.detach(),
                'count': torch.tensor(len(entries), device=zero.device, dtype=torch.long),
                'weight': float(self.occ_actor_weight),
            }
        return stacked.mean() + zero if entries else zero

    def _kinematics(self, actor, current_xy, selected, classes, poses, velocities, times, zero, *, statistics=None):
        previous_pose = torch.cat((selected[..., None], poses[..., :-1]), -1)
        valid = poses & previous_pose & velocities
        safe_times = torch.where(poses.any(1), times, torch.zeros_like(times))
        elapsed = safe_times - torch.cat((torch.zeros_like(safe_times[:, :1]), safe_times[:, :-1]), -1)
        if (elapsed[:, None, :].expand_as(valid)[valid] <= 0).any():
            raise ValueError('future_time_s must increase with positive elapsed seconds for each valid transition')
        dt = torch.where(valid, elapsed[:, None, :], torch.ones_like(actor[..., 0]))
        previous_xy = torch.cat((current_xy[..., None, :], actor[..., :-1, :2]), -2)
        secant_velocity = (actor[..., :2] - previous_xy) / dt[..., None]
        velocity_error = ((secant_velocity - actor[..., 4:]) / self.velocity_scale).square()
        consistency = _mean(velocity_error, valid[..., None], statistics=statistics,
                            name='kinematic_velocity', weight=self.kinematics_weight)
        heading = _heading(actor, self.eps)
        lateral = (-heading[..., 0] * actor[..., 4] + heading[..., 1] * actor[..., 5]) / self.velocity_scale
        class_weight = torch.zeros_like(classes, dtype=actor.dtype)
        for semantic_class in (3, 4, 5, 9, 10):
            class_weight = torch.where(classes == semantic_class, torch.ones_like(class_weight), class_weight)
        for semantic_class in (2, 6):
            class_weight = torch.where(classes == semantic_class, torch.full_like(class_weight, self.cycle_heading_weight), class_weight)
        moving = torch.linalg.vector_norm(actor[..., 4:], dim=-1) >= self.low_speed_threshold
        alignment = _mean(lateral.square() * class_weight[..., None],
                          valid & moving & (class_weight[..., None] > 0),
                          statistics=statistics, name='heading_alignment',
                          weight=self.kinematics_weight)
        return consistency + alignment + zero

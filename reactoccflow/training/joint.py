"""Joint occupancy/actor endpoint loss adapter used by training entrypoints."""
from __future__ import annotations

from torch import nn

from reactoccflow.losses import ConsistencyLosses, decode_endpoints


class JointTrainingAdapter(nn.Module):
    """Expose a scalar training-only API while retaining endpoint gradients."""

    def __init__(self, flow, codec, *, consistency=None, decode_chunk=1,
                 require_frozen_trunk=True, record_components=True):
        super().__init__()
        self.flow = flow
        self.codec = codec.requires_grad_(False).eval()
        self.consistency = ConsistencyLosses() if consistency is None else consistency
        self.decode_chunk = decode_chunk
        self.require_frozen_trunk = require_frozen_trunk
        self.record_components = record_components
        self.last_components = {}
        self.frozen_checks = 0

    def forward(self, observation, target, candidate, *, generator):
        return self.training_loss(observation, target, candidate, generator=generator)

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval()
        return self

    def frozen_status(self):
        trunk = getattr(self.flow.transition_model, "trunk", None)
        groups = {"vae": list(self.codec.parameters())}
        if self.require_frozen_trunk:
            groups["trunk"] = [] if trunk is None else list(trunk.parameters())
        status = {}
        for name, parameters in groups.items():
            status[name] = {
                "parameter_tensors": len(parameters),
                "requires_grad_count": sum(parameter.requires_grad for parameter in parameters),
                "gradient_tensors": sum(parameter.grad is not None for parameter in parameters),
            }
            if status[name]["requires_grad_count"] or status[name]["gradient_tensors"]:
                raise RuntimeError(f"{name} must remain frozen with gradients None")
        return status

    def validation_statistics(self, observation, target, candidate, *, generator):
        """Return detached component sums/counts from one joint objective pass.

        Each component maps to ``numerator``, raw ``count`` and float ``weight``.
        Reduce sums and counts across batches/ranks, then sum the weighted means.
        Callers control eval mode and no-grad/autocast contexts as for training.
        """
        _, statistics = self.training_loss(
            observation, target, candidate, generator=generator, return_statistics=True
        )
        return statistics

    def training_loss(self, observation, target, candidate, *, generator,
                      return_statistics=False):
        """Return the unchanged scalar, or ``(scalar, statistics)`` when requested."""
        self.frozen_status()
        self.frozen_checks += 1
        options = {'return_statistics': True} if return_statistics else {}
        base = self.flow.training_loss(
            observation, target, candidate, generator=generator, return_endpoints=True,
            **options
        )
        endpoints = base["endpoints"]
        logits = decode_endpoints(self.codec, endpoints["future_latents"], chunk=self.decode_chunk)
        consistency = self.consistency(logits, endpoints["actor_future"], observation,
                                       target, **options)
        total = base["total"] + consistency["total"]
        if self.record_components:
            self.last_components = {
                **{f"base.{key}": float(value.detach()) for key, value in base.items() if key not in {"endpoints", "statistics"}},
                **{f"consistency.{key}": float(value.detach()) for key, value in consistency.items() if key != "statistics"},
                "total": float(total.detach()),
            }
        else:
            self.last_components = {}
        if return_statistics:
            return total, {**base['statistics'], **consistency['statistics']}
        return total

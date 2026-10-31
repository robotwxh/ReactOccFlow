import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from reactoccflow.training.engine import (
    build_react_flow_optimizer,
    build_warmup_cosine_scheduler,
    create_ema,
    optimizer_step,
    synchronize_buffers,
)


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.transition_model = nn.Module()
        self.transition_model.trunk = nn.Linear(1, 1, bias=False)
        self.head = nn.Linear(1, 1, bias=False)


def _ddp_accumulation_worker(rank, world_size, rendezvous, output_dir):
    from pathlib import Path

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        torch.set_num_threads(1)
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(float(rank + 1))
        model.register_buffer("rank_marker", torch.tensor(float(rank)))
        wrapped = DistributedDataParallel(model)
        synchronize_buffers(model)
        ema = create_ema(model, decay=0.5)
        torch.save(ema.module.weight.detach(), Path(output_dir) / f"ema-rank-{rank}.pt")
        torch.save(ema.module.rank_marker.detach(), Path(output_dir) / f"buffer-rank-{rank}.pt")
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
        batches = [torch.tensor([[float(rank + 1)]]), torch.tensor([[float(rank + 2)]])]

        def loss_fn(module, batch):
            return module(batch).square().mean()

        optimizer_step(wrapped, optimizer, batches, loss_fn=loss_fn, max_grad_norm=100.0)
        torch.save(model.weight.detach(), Path(output_dir) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_optimizer_has_separate_trunk_and_new_branch_learning_rates():
    flow = TinyFlow()
    optimizer = build_react_flow_optimizer(flow, trunk_lr=2e-5, new_lr=1e-4, weight_decay=0.01)
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["trunk"]["lr"] == 2e-5
    assert groups["new_branches"]["lr"] == 1e-4
    assert groups["trunk"]["params"] == [flow.transition_model.trunk.weight]
    assert groups["new_branches"]["params"] == [flow.head.weight]


def test_warmup_cosine_starts_warm_and_reaches_minimum_ratio():
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=6, warmup_steps=2, min_lr_ratio=0.1)
    assert optimizer.param_groups[0]["lr"] == 0.5
    observed = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    assert observed[0] == 0.5
    assert observed[1] == 1.0
    assert observed[-1] == 0.1


def test_optimizer_step_averages_microbatch_gradients_and_updates_once():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [
        (torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0], [1.0]])),
        (torch.tensor([[3.0], [4.0]]), torch.tensor([[1.0], [1.0]])),
    ]

    def loss_fn(module, batch):
        inputs, target = batch
        return (module(inputs) - target).square().mean()

    loss = optimizer_step(model, optimizer, batches, loss_fn=loss_fn, max_grad_norm=10.0)
    assert loss == 1.0
    # Mean gradient across the two equal-sized microbatches is -5.
    torch.testing.assert_close(model.weight, torch.tensor([[0.5]]))


def test_ema_tracks_trainable_model_parameters():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = create_ema(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update_parameters(model)
    torch.testing.assert_close(ema.module.weight, torch.tensor([[1.5]]))


def test_two_rank_ddp_accumulation_synchronizes_the_global_mean(tmp_path):
    mp.spawn(
        _ddp_accumulation_worker,
        args=(2, f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        nprocs=2,
        join=True,
    )

    weights = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
    torch.testing.assert_close(weights[0], torch.tensor([[0.91]]))
    torch.testing.assert_close(weights[1], weights[0])
    ema_weights = [torch.load(tmp_path / f"ema-rank-{rank}.pt", weights_only=True) for rank in range(2)]
    torch.testing.assert_close(ema_weights[0], torch.tensor([[1.0]]))
    torch.testing.assert_close(ema_weights[1], ema_weights[0])
    buffer_values = [torch.load(tmp_path / f"buffer-rank-{rank}.pt", weights_only=True) for rank in range(2)]
    torch.testing.assert_close(buffer_values[0], torch.tensor(0.0))
    torch.testing.assert_close(buffer_values[1], buffer_values[0])

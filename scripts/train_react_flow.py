"""Train ReactFlow jointly on complete, scene-disjoint E0 train/tune caches.

The training action is the logged ego motion used as a supervised condition.
This is an oracle-representation training workflow, not a future-free planning
benchmark. Launch multi-GPU runs with ``torchrun`` after both caches are complete.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

from reactoccflow.data.e0_dataset import E0ResearchDataset, payload_sha256, read_record, safe_cache_path
from reactoccflow.models import OccupancyCodec
from reactoccflow.models.react_flow import ReactFlow
from reactoccflow.provenance import sha256_file
from reactoccflow.training.engine import (
    build_react_flow_optimizer,
    build_warmup_cosine_scheduler,
    create_ema,
    optimizer_step,
    synchronize_buffers,
)
from reactoccflow.training.joint import JointTrainingAdapter


class NumericE0Dataset(Dataset):
    """Strip JSON metadata before DataLoader collation; tensors stay on CPU."""
    def __init__(self, dataset, *, checksums_verified=False):
        self.dataset = dataset
        self.verified = set(range(len(dataset))) if checksums_verified else set()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index not in self.verified:
            sample = read_record(self.dataset.root, self.dataset.records[index],
                                 self.dataset.manifest["settings_sha256"], verify_checksum=True)
            self.verified.add(index)
            return sample
        return self.dataset[index]


def collate_numeric(samples):
    return {
        section: {
            key: torch.from_numpy(np.stack([sample[section][key] for sample in samples]))
            for key in samples[0][section]
        }
        for section in ("observation", "target")
    }


def move_batch(batch, device):
    return {section: {key: value.to(device, non_blocking=True) for key, value in arrays.items()}
            for section, arrays in batch.items()}


class RankPartitionSampler(Sampler[int]):
    """Sequential non-padding shards for exact-once distributed validation."""
    def __init__(self, dataset, rank, world_size):
        self.length, self.rank, self.world_size = len(dataset), rank, world_size

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self):
        return max(0, (self.length - self.rank + self.world_size - 1) // self.world_size)


def validate_cache_pair(train_cache, tune_cache):
    train = E0ResearchDataset(train_cache, verify_checksums=False)
    tune = E0ResearchDataset(tune_cache, verify_checksums=False)
    for dataset in (train, tune):
        settings_path = safe_cache_path(dataset.root, "settings.json")
        if not settings_path.is_file():
            raise ValueError("Cache settings.json hash disagrees with its manifest")
        wrapper = json.loads(settings_path.read_text())
        settings = wrapper.get("settings") if isinstance(wrapper, dict) else None
        if not isinstance(settings, dict):
            raise ValueError("Cache settings.json hash requires a settings wrapper")
        canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        digest = hashlib.sha256(canonical).hexdigest()
        if digest != wrapper.get("settings_sha256") or digest != dataset.manifest["settings_sha256"]:
            raise ValueError("Cache settings.json hash disagrees with its manifest")
        if (settings.get("group") != dataset.manifest.get("group")
                or settings.get("normalization") != dataset.normalization):
            raise ValueError("Cache manifest group/normalization disagrees with settings.json")
    if train.manifest.get("group") != "train" or tune.manifest.get("group") != "tune":
        raise ValueError("Training requires group=train and group=tune E0 caches")
    if train.manifest.get("oracle") is not True or tune.manifest.get("oracle") is not True:
        raise ValueError("This runner is explicitly for oracle E0 representation training")
    if not train.records or not tune.records:
        raise ValueError("Train and tune caches must both contain samples")
    train_scenes = {entry["scene"] for entry in train.records}
    tune_scenes = {entry["scene"] for entry in tune.records}
    if train_scenes & tune_scenes:
        raise ValueError("Train and tune caches contain overlapping scenes")
    if train.normalization != tune.normalization:
        raise ValueError("Train and tune caches must use identical train-only normalization")
    normalization = train.normalization
    if not isinstance(normalization, dict) or normalization.get("fit_group") != "train" or normalization.get("coordinate_frame") != "E0":
        raise ValueError("Caches must bind train-only E0 normalization statistics")
    for dataset in (train, tune):
        _verify_cache_payloads(dataset)
    return train, tune


def _verify_cache_payloads(dataset):
    """Verify all committed records before model/output setup begins."""
    identity_fields = ("scene", "token", "file", "record_file", "sha256", "settings_sha256", "global_index")
    for entry in dataset.records:
        payload = safe_cache_path(dataset.root, entry["file"])
        record_path = safe_cache_path(dataset.root, entry["record_file"])
        record = json.loads(record_path.read_text())
        if any(record.get(key) != entry.get(key) for key in identity_fields):
            raise ValueError("Cache record identity disagrees with manifest")
        metadata = record.get("metadata", {})
        if (metadata.get("scene"), metadata.get("token")) != (entry["scene"], entry["token"]):
            raise ValueError("Cache record metadata identity disagrees with manifest")
        if payload_sha256(payload) != entry["sha256"]:
            raise ValueError(f"Cache payload checksum mismatch: {entry['scene']}/{entry['token']}")


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _capture_rng(generator, device):
    numpy_state = np.random.get_state()
    py_state = random.getstate()
    return {
        "python": {"version": py_state[0], "state": list(py_state[1]), "gauss": py_state[2]},
        "numpy": {"algorithm": numpy_state[0], "state": numpy_state[1].tolist(),
                  "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "flow_generator": generator.get_state(),
    }


def _restore_rng(state, generator, device):
    python_state = state["python"]
    random.setstate((python_state["version"], tuple(python_state["state"]), python_state["gauss"]))
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["state"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)
    generator.set_state(state["flow_generator"])


def _gather_rng(generator, device, rank, world_size):
    local = _capture_rng(generator, device)
    if world_size == 1:
        return [local]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)
    return gathered if rank == 0 else None


def _runtime_fingerprint(train_cache, tune_cache, codec_checkpoint):
    root = Path(__file__).resolve().parents[1]
    paths = {
        train_cache / "manifest.json", train_cache / "settings.json",
        tune_cache / "manifest.json", tune_cache / "settings.json",
        Path(codec_checkpoint),
        root / "scripts/train_react_flow.py", root / "scripts/train_joint_smoke.py",
        root / "reactoccflow/training/diagnostics.py",
        root / "reactoccflow/training/engine.py", root / "reactoccflow/training/joint.py",
        root / "pyproject.toml", root / "requirements-lock.txt",
        *root.joinpath("reactoccflow").rglob("*.py"),
        *root.joinpath("configs").rglob("*.yaml"),
        *root.joinpath("configs").rglob("*.yml"),
        *root.joinpath("configs").rglob("*.json"),
    }
    fingerprint = {str(path.resolve()): sha256_file(path) for path in sorted(paths)}
    for cache in (train_cache, tune_cache):
        manifest = json.loads((cache / "manifest.json").read_text())
        for entry in manifest["records"]:
            record_path = safe_cache_path(cache, entry["record_file"])
            fingerprint[str(record_path.resolve())] = sha256_file(record_path)
    return fingerprint


def _evaluate(objective, loader, device, generator):
    objective.eval()
    totals, names, weights = None, None, None
    samples = torch.zeros((), dtype=torch.float64, device=device)
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            candidate = batch["target"]["ego_future"][..., :2]
            statistics = objective.validation_statistics(
                batch["observation"], batch["target"], candidate, generator=generator)
            current_names = sorted(statistics)
            current_weights = [statistics[name]["weight"] for name in current_names]
            if names is None:
                names = current_names
                weights = torch.tensor(current_weights, dtype=torch.float64, device=device)
                totals = torch.zeros(len(names), 2, dtype=torch.float64, device=device)
            elif current_names != names or current_weights != weights.tolist():
                raise RuntimeError("Validation loss components or weights changed during evaluation")
            totals += torch.stack([torch.stack((statistics[name]["numerator"].double(),
                                               statistics[name]["count"].double())) for name in names])
            samples += batch["observation"]["history_latents"].shape[0]
    if totals is None:
        raise RuntimeError("Tune sampler produced no validation records")
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples, op=dist.ReduceOp.SUM)
    count = samples.item()
    if count < 1:
        raise RuntimeError("Tune sampler produced no validation records")
    loss = (weights * totals[:, 0] / totals[:, 1].clamp_min(1)).sum()
    if not torch.isfinite(loss):
        raise RuntimeError("Validation loss must be finite")
    return float(loss.item()), int(count)


def _load_resume(path, expected_config, data_fingerprint, flow, ema, optimizer, scheduler,
                 generator, device, rank, world_size):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("config") != expected_config or state.get("data_fingerprint") != data_fingerprint:
        raise ValueError("Resume checkpoint config/data identity differs from this run")
    if state.get("world_size") != world_size or len(state.get("rank_rng_states", [])) != world_size:
        raise ValueError("Resume requires the same distributed world size")
    flow.load_state_dict(state["flow_state_dict"], strict=True)
    ema.load_state_dict(state["ema_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    _restore_rng(state["rank_rng_states"][rank], generator, device)
    return {key: state[key] for key in ("global_step", "epoch", "next_batch", "best_tune_loss", "bad_evaluations", "evaluations")}


def _checkpoint(path, *, flow, ema, optimizer, scheduler, config, data_fingerprint,
                world_size, rng_states, progress, best_tune_loss, bad_evaluations, evaluations):
    _atomic_torch_save(path, {
        "version": 1, "config": config, "data_fingerprint": data_fingerprint,
        "world_size": world_size, "flow_state_dict": {k: v.detach().cpu() for k, v in flow.state_dict().items()},
        "ema_state_dict": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "rank_rng_states": rng_states, **progress,
        "best_tune_loss": best_tune_loss, "bad_evaluations": bad_evaluations,
        "evaluations": evaluations,
    })


def _make_loader(dataset, *, batch_size, workers, sampler, device, seed, drop_last=False):
    options = {
        "dataset": NumericE0Dataset(dataset, checksums_verified=True), "batch_size": batch_size, "sampler": sampler,
        "num_workers": workers, "pin_memory": device.type == "cuda", "collate_fn": collate_numeric,
        "persistent_workers": workers > 0, "worker_init_fn": _seed_worker, "drop_last": drop_last,
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(**options)


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _cursor_after_batch(epoch, batch_index, batch_count):
    if batch_index >= batch_count:
        return epoch + 1, 0
    return epoch, batch_index


def _diagnostic_datasets(train, tune, train_windows, tune_windows):
    """Select explicit engineering subsets without publishing partial caches."""
    if train_windows is None and tune_windows is None:
        return train, tune
    if (train_windows is None or tune_windows is None
            or not 1 <= train_windows <= len(train)
            or not 1 <= tune_windows <= len(tune)):
        raise ValueError("Both diagnostic window counts must be positive and within their complete caches")
    return Subset(train, range(train_windows)), Subset(tune, range(tune_windows))


def _usable_training_batches(batch_count, *, accumulation):
    """Drop only the shuffled epoch tail to keep every update's batch fixed."""
    usable = batch_count - batch_count % accumulation
    if usable < 1:
        raise ValueError("Training requires at least one full gradient accumulation group")
    return usable


def _configure_determinism():
    """Extend the copied OccFM seed policy to strict CUDA/BLAS determinism."""
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return {"deterministic_algorithms": True, "cudnn_deterministic": True,
            "cudnn_benchmark": False, "cublas_workspace_config": workspace,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def _prepare_output(output, *, resume, rank, world_size, device):
    """Let rank zero create output and broadcast any failure to all ranks."""
    result = [None]
    if rank == 0:
        try:
            if resume is None and output.exists() and any(output.iterdir()):
                raise FileExistsError("Use a fresh output directory or pass --resume; existing results are retained")
            output.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            result[0] = (type(error).__name__, str(error))
    if world_size > 1:
        dist.broadcast_object_list(result, src=0, device=device)
    if result[0] is not None:
        name, message = result[0]
        if name == "FileExistsError":
            raise FileExistsError(message)
        raise RuntimeError(f"Rank zero could not prepare output directory: {name}: {message}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="data/research/e0_full/train")
    parser.add_argument("--tune-cache", default="data/research/e0_full/tune")
    parser.add_argument("--backbone", choices=["occfm_unet", "dit"], default="occfm_unet")
    parser.add_argument("--dit-config", type=Path, help="Optional flat YAML of standard DiT architecture options")
    parser.add_argument("--unet-config", type=Path, help="Optional flat YAML of restored U-shaped DiT options")
    parser.add_argument("--trunk-checkpoint", type=Path, help="Explicit OccFM U-shaped trunk initialization; not resume")
    parser.add_argument("--trunk-weights", choices=["state_dict", "ema_model"], default="state_dict")
    parser.add_argument("--codec-checkpoint", default="assets/checkpoints/occfm_vae.ckpt")
    parser.add_argument("--output", help="Fresh run directory; defaults to outputs/reactflow_<backbone>/seed_<seed>")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-rank microbatch")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--decode-chunk", type=int, default=1)
    parser.add_argument("--diagnostic-train-windows", type=int, default=None,
                        help="Explicit engineering subset of a complete cache; both diagnostic counts are required")
    parser.add_argument("--diagnostic-tune-windows", type=int, default=None,
                        help="Engineering validation subset; results are not a full tune evaluation")
    parser.add_argument("--stop-after-steps", type=int, default=None,
                        help="Checkpoint and pause at this global step, preserving the configured schedule for resume")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if args.dit_config is not None and args.backbone != 'dit':
        parser.error('--dit-config requires --backbone dit')
    if (args.unet_config is not None or args.trunk_checkpoint is not None) and args.backbone != 'occfm_unet':
        parser.error('--unet-config and --trunk-checkpoint require --backbone occfm_unet')
    if args.resume and args.trunk_checkpoint is not None:
        parser.error('--resume cannot be combined with --trunk-checkpoint initialization')
    dit_options = None
    if args.dit_config is not None:
        dit_options = yaml.safe_load(args.dit_config.read_text())
        allowed = {'patch_size', 'hidden_size', 'depth', 'num_heads', 'mlp_ratio', 'activation_checkpointing'}
        if not isinstance(dit_options, dict) or set(dit_options) - allowed:
            parser.error('--dit-config must contain only patch_size, hidden_size, depth, num_heads, mlp_ratio, activation_checkpointing')
    if args.output is None:
        args.output = f'outputs/reactflow_{args.backbone}/seed_{args.seed}'
    unet_options = None
    if args.unet_config is not None:
        unet_options = yaml.safe_load(args.unet_config.read_text())
        allowed = {'model_channels', 'channel_multi', 'init_kernel_size', 'init_3d_conv_channels',
                   'attn_dim', 'temporal_attn_head', 'spatial_attn_head', 'attention_mode'}
        if not isinstance(unet_options, dict) or set(unet_options) - allowed:
            parser.error('--unet-config contains unknown U-shaped DiT architecture options')
    if min(args.max_steps, args.eval_interval, args.patience, args.batch_size, args.grad_accum, args.decode_chunk) < 1:
        parser.error("steps, intervals, batch sizes, patience and decode chunk must be positive")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps or args.workers < 0:
        parser.error("warmup must be in [0, max_steps) and workers nonnegative")
    if args.stop_after_steps is not None and not 1 <= args.stop_after_steps <= args.max_steps:
        parser.error("stop-after-steps must be in [1, max-steps]")

    train_cache, tune_cache = Path(args.train_cache).resolve(), Path(args.tune_cache).resolve()
    train_dataset, tune_dataset = validate_cache_pair(train_cache, tune_cache)
    normalization = train_dataset.normalization
    train_dataset, tune_dataset = _diagnostic_datasets(
        train_dataset, tune_dataset, args.diagnostic_train_windows, args.diagnostic_tune_windows)
    output = Path(args.output).resolve()
    numerical_settings = _configure_determinism()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.device == "cuda":
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise RuntimeError("Requested CUDA local rank is unavailable")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device, backend = torch.device("cpu"), "gloo"
    if world_size > 1:
        dist.init_process_group(backend=backend)
    if not 0 <= rank < world_size:
        raise RuntimeError("Invalid distributed rank configuration")

    try:
        _prepare_output(output, resume=args.resume, rank=rank, world_size=world_size, device=device)
        seed = args.seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(4)
        flow_generator = torch.Generator(device=device).manual_seed(args.seed + 100_003 * rank + 1)

        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=args.seed, drop_last=True)
        train_loader = _make_loader(train_dataset, batch_size=args.batch_size, workers=args.workers,
                                    sampler=sampler, device=device, seed=seed, drop_last=True)
        training_batch_count = _usable_training_batches(len(train_loader), accumulation=args.grad_accum)
        if len(train_loader) < 1 or len(tune_dataset) < world_size:
            raise ValueError("Train/tune cache is too small for the requested ranks and batch size")
        tune_loader = _make_loader(tune_dataset, batch_size=args.batch_size, workers=args.workers,
                                   sampler=RankPartitionSampler(tune_dataset, rank, world_size), device=device,
                                   seed=args.seed + 900_001 + rank)

        flow = ReactFlow(normalization=normalization, backbone=args.backbone,
                         dit_config=dit_options, unet_config=unet_options).to(device)
        loading = {'initialization': f'{args.backbone} from scratch'}
        if args.trunk_checkpoint is not None:
            loading = flow.load_pretrained_trunk(args.trunk_checkpoint, weights=args.trunk_weights)
            loading.update(initialization='explicit OccFM trunk checkpoint',
                           checkpoint=str(args.trunk_checkpoint.resolve()),
                           checkpoint_sha256=sha256_file(args.trunk_checkpoint))
        flow.freeze_trunk(frozen=False)
        codec = OccupancyCodec().to(device).eval()
        codec.load_pretrained(args.codec_checkpoint, weights="state_dict")
        objective = JointTrainingAdapter(flow, codec, decode_chunk=args.decode_chunk,
                                         require_frozen_trunk=False, record_components=False).to(device)
        trunk_lr = 1e-4
        optimizer = build_react_flow_optimizer(flow, trunk_lr=trunk_lr, new_lr=1e-4, weight_decay=0.01)
        scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=args.max_steps,
                                                  warmup_steps=args.warmup_steps, min_lr_ratio=0.01)
        if world_size > 1:
            wrapped = DistributedDataParallel(objective, device_ids=[local_rank] if device.type == "cuda" else None,
                                              broadcast_buffers=False, find_unused_parameters=False)
        else:
            wrapped = objective
        # DDP broadcasts rank zero's randomly initialized branches; EMA must
        # start from that synchronized model and buffer state on every rank.
        synchronize_buffers(flow)
        ema = create_ema(flow, decay=0.9999).to(device)

        config = {"mode": "joint", "max_steps": args.max_steps, "warmup_steps": args.warmup_steps,
                  "eval_interval": args.eval_interval, "patience": args.patience,
                  "batch_size_per_rank": args.batch_size, "gradient_accumulation": args.grad_accum,
                  "effective_batch_size": args.batch_size * args.grad_accum * world_size,
                  "workers": args.workers, "decode_chunk": args.decode_chunk,
                  "trunk_lr": trunk_lr, "new_branch_lr": 1e-4, "weight_decay": 0.01,
                  "max_grad_norm": 1.0, "ema_decay": 0.9999, "precision": "float32",
                  "minimum_lr_ratio": 0.01, "seed": args.seed, "world_size": world_size,
                  "device_type": args.device, "condition": "logged_ego_action_supervision",
                  "observation_representation": "oracle_E0",
                  "run_scope": "diagnostic_subset" if args.diagnostic_train_windows is not None else "full_training",
                  "diagnostic_train_windows": args.diagnostic_train_windows,
                  "diagnostic_tune_windows": args.diagnostic_tune_windows,
                  "validation_reduction": "global_per_component_valid_element_means",
                  "epoch_tail_policy": "drop_incomplete_accumulation_group",
                  "train_batches_per_rank": training_batch_count,
                  "dropped_tail_microbatches_per_rank": len(train_loader) - training_batch_count}
        config["numerical_settings"] = numerical_settings
        config['backbone'] = args.backbone
        config['dit'] = ((getattr(getattr(flow.transition_model, 'trunk', None), 'config', None)
                         if hasattr(flow, 'transition_model') else dit_options)
                         if args.backbone == 'dit' else None)
        if args.backbone == 'occfm_unet':
            config['unet'] = getattr(getattr(getattr(flow, 'transition_model', None), 'trunk', None), 'config', unet_options)
        data_fingerprint = _runtime_fingerprint(train_cache, tune_cache, args.codec_checkpoint)
        progress = {"global_step": 0, "epoch": 0, "next_batch": 0,
                    "best_tune_loss": math.inf, "bad_evaluations": 0, "evaluations": 0}
        if args.resume:
            progress = _load_resume(args.resume, config, data_fingerprint, flow, ema, optimizer, scheduler,
                                    flow_generator, device, rank, world_size)
        started = time.perf_counter()
        log_path = output / "training.jsonl"
        log_stream = log_path.open("a" if args.resume else "x", encoding="utf-8") if rank == 0 else None

        def log(record):
            if rank == 0:
                record.update({"utc": datetime.now(timezone.utc).isoformat(),
                               "elapsed_seconds": time.perf_counter() - started})
                log_stream.write(json.dumps(record, allow_nan=False) + "\n")
                log_stream.flush()
                print(json.dumps(record, allow_nan=False), flush=True)

        def save_latest(next_epoch, next_batch, best, bad, evaluations):
            rng_states = _gather_rng(flow_generator, device, rank, world_size)
            if rank == 0:
                _checkpoint(output / "latest.pt", flow=flow, ema=ema, optimizer=optimizer,
                            scheduler=scheduler, config=config, data_fingerprint=data_fingerprint,
                            world_size=world_size, rng_states=rng_states,
                            progress={"global_step": progress["global_step"], "epoch": next_epoch,
                                      "next_batch": next_batch}, best_tune_loss=best,
                            bad_evaluations=bad, evaluations=evaluations)
            if world_size > 1:
                dist.barrier()

        def evaluate_and_checkpoint(next_epoch, next_batch, best, bad, evaluations):
            eval_generator = torch.Generator(device=device).manual_seed(args.seed + 9_000_001 + rank)
            tune_loss, sample_count = _evaluate(
                JointTrainingAdapter(ema.module, codec, decode_chunk=args.decode_chunk,
                                     require_frozen_trunk=False, record_components=False).to(device),
                tune_loader, device, eval_generator)
            evaluations += 1
            improved = tune_loss < best
            if improved:
                best, bad = tune_loss, 0
                if rank == 0:
                    _atomic_torch_save(output / "best.pt", {
                        "version": 1, "ema_state_dict": {k: v.detach().cpu() for k, v in ema.module.state_dict().items()},
                        "global_step": progress["global_step"], "tune_loss": tune_loss,
                        "config": config, "data_fingerprint": data_fingerprint,
                        "trunk_loading_report": loading,
                    })
            else:
                bad += 1
            save_latest(next_epoch, next_batch, best, bad, evaluations)
            log({"event": "validation", "global_step": progress["global_step"],
                 "tune_loss_logged_action_conditioned": tune_loss, "tune_sample_count": sample_count,
                 "improved": improved, "bad_evaluations": bad, "best_tune_loss": best})
            return best, bad, evaluations

        log({"event": "started", "global_step": progress["global_step"], "config": config,
             "train_windows": len(train_dataset), "tune_windows": len(tune_dataset),
             "trainable_parameter_tensors": sum(p.requires_grad for p in flow.parameters()),
             "stop_after_steps": args.stop_after_steps})
        initial_step = progress["global_step"]
        stop_step = args.max_steps if args.stop_after_steps is None else args.stop_after_steps
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        flow.train()
        while progress["global_step"] < stop_step and progress["bad_evaluations"] < args.patience:
            epoch = progress["epoch"]
            sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            start_batch = progress["next_batch"]
            for _ in range(start_batch):
                next(iterator)
            batch_index = start_batch
            while batch_index < training_batch_count and progress["global_step"] < stop_step:
                group = []
                group_end = batch_index + args.grad_accum
                while batch_index < group_end:
                    group.append(move_batch(next(iterator), device))
                    batch_index += 1

                def loss_fn(model, batch):
                    candidate = batch["target"]["ego_future"][..., :2]
                    return model(batch["observation"], batch["target"], candidate, generator=flow_generator)

                local_loss = optimizer_step(wrapped, optimizer, group, loss_fn=loss_fn, max_grad_norm=1.0)
                scheduler.step()
                ema.update_parameters(flow)
                progress["global_step"] += 1
                next_epoch, next_batch = _cursor_after_batch(epoch, batch_index, training_batch_count)
                progress.update(epoch=next_epoch, next_batch=next_batch)
                if world_size > 1:
                    mean_loss = torch.tensor(local_loss, dtype=torch.float64, device=device)
                    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                    mean_loss = float((mean_loss / world_size).item())
                else:
                    mean_loss = local_loss
                if progress["global_step"] == 1 or progress["global_step"] % 50 == 0:
                    log({"event": "train", "global_step": progress["global_step"],
                         "loss": mean_loss, "lr_trunk": optimizer.param_groups[0]["lr"],
                         "lr_new_branches": optimizer.param_groups[-1]["lr"],
                         "rank0_peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
                if progress["global_step"] % args.eval_interval == 0:
                    best, bad, evaluations = evaluate_and_checkpoint(
                        next_epoch, next_batch, progress["best_tune_loss"],
                        progress["bad_evaluations"], progress["evaluations"])
                    progress.update(epoch=next_epoch, next_batch=next_batch, best_tune_loss=best,
                                    bad_evaluations=bad, evaluations=evaluations)
                    if bad >= args.patience:
                        break
            else:
                next_epoch, next_batch = _cursor_after_batch(epoch, batch_index, training_batch_count)
                progress.update(epoch=next_epoch, next_batch=next_batch)
                continue
            if progress["bad_evaluations"] >= args.patience or progress["global_step"] >= stop_step:
                break

        paused = progress["global_step"] < args.max_steps and progress["bad_evaluations"] < args.patience
        if paused and progress["global_step"] > initial_step and progress["global_step"] % args.eval_interval != 0:
            save_latest(progress["epoch"], progress["next_batch"], progress["best_tune_loss"],
                        progress["bad_evaluations"], progress["evaluations"])
        elif not paused and progress["global_step"] > initial_step and progress["global_step"] % args.eval_interval != 0:
            next_epoch, next_batch = progress["epoch"], progress["next_batch"]
            if next_batch >= training_batch_count:
                next_epoch, next_batch = next_epoch + 1, 0
            best, bad, evaluations = evaluate_and_checkpoint(
                next_epoch, next_batch, progress["best_tune_loss"],
                progress["bad_evaluations"], progress["evaluations"])
            progress.update(epoch=next_epoch, next_batch=next_batch, best_tune_loss=best,
                            bad_evaluations=bad, evaluations=evaluations)
        if world_size > 1:
            dist.barrier()
        log({"event": "paused" if paused else "finished", "global_step": progress["global_step"],
             "best_tune_loss_logged_action_conditioned": progress["best_tune_loss"] if math.isfinite(progress["best_tune_loss"]) else None,
             "training_seconds": time.perf_counter() - started,
             "formal_planning_evaluation": False})
        if log_stream is not None:
            log_stream.close()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

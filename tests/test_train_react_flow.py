import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.train_react_flow import (
    RankPartitionSampler,
    _configure_determinism,
    _cursor_after_batch,
    _diagnostic_datasets,
    _evaluate,
    _prepare_output,
    _usable_training_batches,
    main,
    validate_cache_pair,
)


def test_training_configures_strict_determinism_before_cuda_setup(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        settings = _configure_determinism()
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert settings["cublas_workspace_config"] == ":4096:8"
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def test_validation_combines_global_valid_element_statistics():
    class MaskedObjective:
        def eval(self):
            return self

        def validation_statistics(self, observation, target, candidate, *, generator):
            values, valid = observation["values"], target["valid"]
            return {"masked": {"numerator": torch.where(valid, values, 0.).sum(),
                               "count": valid.sum(), "weight": .5}}

        def __call__(self, observation, target, candidate, *, generator):
            stats = self.validation_statistics(observation, target, candidate, generator=generator)["masked"]
            return .5 * stats["numerator"] / stats["count"].clamp_min(1)

    batch = {"observation": {"history_latents": torch.zeros(2, 1),
                             "values": torch.tensor([[0., 0., 0.], [12., 0., 0.]])},
             "target": {"ego_future": torch.zeros(2, 1, 2),
                        "valid": torch.tensor([[True, True, True], [True, False, False]])}}
    split = [{section: {key: value[i:i+1] for key, value in arrays.items()}
              for section, arrays in batch.items()} for i in range(2)]
    together = _evaluate(MaskedObjective(), [batch], torch.device("cpu"), torch.Generator())
    separated = _evaluate(MaskedObjective(), split, torch.device("cpu"), torch.Generator())

    assert together == separated == (1.5, 2)


def test_resuming_early_stopped_state_does_not_optimize_again(tmp_path, monkeypatch):
    import scripts.train_react_flow as runner

    class TinyModel(nn.Linear):
        def __init__(self, **kwargs):
            super().__init__(1, 1)

        def freeze_trunk(self, **kwargs):
            return self

        def load_pretrained(self, *args, **kwargs):
            return {}

    train = _write_cache(tmp_path / "train", group="train", scenes=["train"])
    tune = _write_cache(tmp_path / "tune", group="tune", scenes=["tune"])
    monkeypatch.setattr(runner, "ReactFlow", TinyModel)
    monkeypatch.setattr(runner, "OccupancyCodec", TinyModel)
    monkeypatch.setattr(runner, "_runtime_fingerprint", lambda *args: {})
    monkeypatch.setattr(runner, "_load_resume", lambda *args: {
        "global_step": 4, "epoch": 0, "next_batch": 0, "best_tune_loss": 1.,
        "bad_evaluations": 2, "evaluations": 3,
    })
    def unexpected(*args, **kwargs):
        pytest.fail("early-stopped checkpoint must not train or validate again")
    monkeypatch.setattr(runner, "optimizer_step", unexpected)
    monkeypatch.setattr(runner, "_evaluate", unexpected)
    output = tmp_path / "run"

    main(["--train-cache", str(train), "--tune-cache", str(tune), "--output", str(output),
          "--resume", "terminal.pt", "--device", "cpu", "--batch-size", "1",
          "--grad-accum", "1", "--workers", "0", "--max-steps", "20", "--warmup-steps", "1", "--patience", "2"])

    records = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    assert records[-1]["event"] == "finished"
    assert records[-1]["global_step"] == 4


def test_diagnostic_limits_select_without_rewriting_complete_manifests(tmp_path):
    train = _write_cache(tmp_path / "train", group="train", scenes=["a", "b", "c"])
    tune = _write_cache(tmp_path / "tune", group="tune", scenes=["d", "e"])
    train_data, tune_data = validate_cache_pair(train, tune)
    original = [(cache / "manifest.json").read_bytes() for cache in (train, tune)]

    selected_train, selected_tune = _diagnostic_datasets(train_data, tune_data, 2, 1)

    assert list(selected_train.indices) == [0, 1]
    assert list(selected_tune.indices) == [0]
    assert [len(selected_train), len(selected_tune)] == [2, 1]
    assert [(cache / "manifest.json").read_bytes() for cache in (train, tune)] == original


@pytest.mark.parametrize("limits", [(2, None), (None, 1), (4, 1), (1, 0)])
def test_diagnostic_limits_require_two_valid_explicit_counts(limits):
    with pytest.raises(ValueError, match="diagnostic"):
        _diagnostic_datasets(list(range(3)), list(range(2)), *limits)


def test_no_diagnostic_limits_preserve_full_datasets():
    train, tune = [1, 2], [3]
    assert _diagnostic_datasets(train, tune, None, None) == (train, tune)


def test_full_training_sampler_preserves_effective_batch_eight():
    from torch.utils.data.distributed import DistributedSampler

    sampler = DistributedSampler(list(range(17468)), num_replicas=2, rank=0,
                                 shuffle=False, drop_last=True)
    microbatches = len(sampler) // 2
    usable = _usable_training_batches(microbatches, accumulation=2)

    assert usable == 4366
    assert usable * 2 * 2 == 17464
    assert (usable // 2) * 8 == 17464


def test_training_refuses_less_than_one_full_accumulation_group():
    with pytest.raises(ValueError, match="accumulation"):
        _usable_training_batches(1, accumulation=2)


def _output_worker(rank, rendezvous, output, results):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        try:
            _prepare_output(Path(output), resume=None, rank=rank, world_size=2, device="cpu")
            result = "created"
        except FileExistsError:
            result = "rejected"
        Path(results, f"rank-{rank}.txt").write_text(result)
    finally:
        dist.destroy_process_group()


def _write_cache(root, *, group, scenes, normalization=None, complete=True):
    root.mkdir(parents=True)
    normalization = normalization or {"fit_group": "train", "coordinate_frame": "E0"}
    settings = {"group": group, "normalization": normalization,
                "world_size": 1, "expected_count": len(scenes)}
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    settings_sha256 = hashlib.sha256(canonical).hexdigest()
    (root / "settings.json").write_text(json.dumps({
        "settings_sha256": settings_sha256, "settings": settings,
    }) + "\n")
    records = [
        {
            "scene": scene,
            "token": f"{scene}-token",
            "global_index": index,
            "settings_sha256": settings_sha256,
            "file": f"samples/{index}.npz",
            "record_file": f"records/{index}.json",
            "sha256": hashlib.sha256(f"payload-{index}".encode()).hexdigest(),
        }
        for index, scene in enumerate(scenes)
    ]
    (root / "manifest.json").write_text(json.dumps({
        "format": "reactoccflow_e0_research_cache",
        "version": 1,
        "coordinate_frame": "E0",
        "complete": complete,
        "expected_count": len(records),
        "sample_count": len(records),
        "records": records,
        "settings_sha256": settings_sha256,
        "normalization": normalization,
        "group": group,
        "oracle": True,
    }))
    for entry in records:
        payload = root / entry["file"]
        metadata = root / entry["record_file"]
        payload.parent.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(f"payload-{entry['global_index']}".encode())
        metadata.write_text(json.dumps({**entry, "metadata": {"scene": entry["scene"], "token": entry["token"]}}))
    return root


def test_runner_accepts_cache_generator_settings_wrapper(tmp_path):
    train = _write_cache(tmp_path / "train", group="train", scenes=["train-scene"])
    tune = _write_cache(tmp_path / "tune", group="tune", scenes=["tune-scene"])

    train_dataset, tune_dataset = validate_cache_pair(train, tune)

    assert len(train_dataset) == len(tune_dataset) == 1


def test_runner_refuses_partial_cache_before_creating_output(tmp_path):
    train = _write_cache(tmp_path / "train", group="train", scenes=["scene-a"], complete=False)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="incomplete"):
        main([
            "--train-cache", str(train),
            "--tune-cache", str(tmp_path / "missing-tune"),
            "--output", str(output),
            "--device", "cpu",
        ])

    assert not output.exists()


def test_runner_rejects_overlapping_train_and_tune_scenes(tmp_path):
    train = _write_cache(tmp_path / "train", group="train", scenes=["shared"])
    tune = _write_cache(tmp_path / "tune", group="tune", scenes=["shared"])

    with pytest.raises(ValueError, match="overlapping scenes"):
        validate_cache_pair(train, tune)


def test_runner_requires_the_same_train_only_normalization(tmp_path):
    train = _write_cache(tmp_path / "train", group="train", scenes=["train-scene"])
    tune = _write_cache(
        tmp_path / "tune", group="tune", scenes=["tune-scene"],
        normalization={"fit_group": "tune", "coordinate_frame": "E0"},
    )

    with pytest.raises(ValueError, match="identical train-only normalization"):
        validate_cache_pair(train, tune)


def test_runner_rejects_cache_settings_file_that_disagrees_with_manifest(tmp_path):
    train = _write_cache(tmp_path / "train", group="train", scenes=["train-scene"])
    tune = _write_cache(tmp_path / "tune", group="tune", scenes=["tune-scene"])
    (tune / "settings.json").write_text("{\"changed\": true}\n")

    with pytest.raises(ValueError, match="settings.json hash"):
        validate_cache_pair(train, tune)


def test_runner_rejects_manifest_normalization_not_bound_by_settings(tmp_path):
    train = _write_cache(tmp_path / "train", group="train", scenes=["train-scene"])
    tune = _write_cache(tmp_path / "tune", group="tune", scenes=["tune-scene"])
    altered = {"fit_group": "train", "coordinate_frame": "E0", "actor_scale": 42}
    for cache in (train, tune):
        manifest_path = cache / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["normalization"] = altered
        manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="disagrees with settings.json"):
        validate_cache_pair(train, tune)


@pytest.mark.parametrize("tamper", ["settings", "settings_sha256"])
def test_runner_rejects_tampered_settings_wrapper(tmp_path, tamper):
    train = _write_cache(tmp_path / "train", group="train", scenes=["train-scene"])
    tune = _write_cache(tmp_path / "tune", group="tune", scenes=["tune-scene"])
    path = tune / "settings.json"
    wrapper = json.loads(path.read_text())
    if tamper == "settings":
        wrapper["settings"]["world_size"] = 2
    else:
        wrapper["settings_sha256"] = "0" * 64
    path.write_text(json.dumps(wrapper))

    with pytest.raises(ValueError, match="settings.json hash"):
        validate_cache_pair(train, tune)


def test_runner_verifies_each_payload_before_accepting_complete_caches(tmp_path):
    train = _write_cache(tmp_path / "train", group="train", scenes=["train-scene"])
    tune = _write_cache(tmp_path / "tune", group="tune", scenes=["tune-scene"])
    (train / "samples/0.npz").write_bytes(b"changed")

    with pytest.raises(ValueError, match="payload checksum"):
        validate_cache_pair(train, tune)


def test_validation_rank_partitions_cover_every_record_once():
    shards = [list(RankPartitionSampler(list(range(11)), rank, 3)) for rank in range(3)]

    assert [len(shard) for shard in shards] == [4, 4, 3]
    assert sorted(index for shard in shards for index in shard) == list(range(11))


def test_max_step_checkpoint_retains_mid_epoch_resume_cursor():
    assert _cursor_after_batch(epoch=3, batch_index=5, batch_count=12) == (3, 5)
    assert _cursor_after_batch(epoch=3, batch_index=12, batch_count=12) == (4, 0)


def test_output_setup_preserves_existing_nonempty_results(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    evidence = output / "training.jsonl"
    evidence.write_text("existing evidence\n")

    with pytest.raises(FileExistsError, match="fresh output directory"):
        _prepare_output(output, resume=None, rank=0, world_size=1, device="cpu")

    assert evidence.read_text() == "existing evidence\n"


def test_distributed_output_collision_is_broadcast_to_all_ranks(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    evidence = output / "training.jsonl"
    evidence.write_text("existing evidence\n")
    results = tmp_path / "results"
    results.mkdir()

    mp.spawn(
        _output_worker,
        args=(f"file://{tmp_path / 'output-rendezvous'}", str(output), str(results)),
        nprocs=2,
        join=True,
    )

    assert [(results / f"rank-{rank}.txt").read_text() for rank in range(2)] == ["rejected", "rejected"]
    assert evidence.read_text() == "existing evidence\n"

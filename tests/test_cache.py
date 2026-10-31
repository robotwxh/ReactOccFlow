import pickle
import tempfile
from pathlib import Path
import numpy as np
import pytest


def fixtures(root, *, missing=False):
    records = []
    infos = {}
    for scene in ["scene-0001", "scene-0002"]:
        infos[scene] = []
        for i in range(4):
            token = f"{scene}-t{i}"
            infos[scene].append({"token": token, "timestamp": i * 500000})
            if missing and scene == "scene-0001" and i == 1:
                continue
            records.append({"x_sampled": np.full((1, 2, 3, 3), i, np.float32),
                            "gt_path": [f"/any/different/root/gts/{scene}/{token}/labels.npz"],
                            "gt_trajs": np.array([[i, -i]], np.float32)})
    cache = root / "input.pkl"
    with cache.open("wb") as f:
        pickle.dump(list(reversed(records)), f)
    info = root / "info.pkl"
    with info.open("wb") as f:
        pickle.dump({"infos": infos}, f)
    return cache, info


def test_cache_orders_tokens_and_never_crosses_scenes(tmp_path):
    from reactoccflow.data.cache import convert_legacy_cache, LatentSequenceDataset
    cache, info = fixtures(tmp_path)
    output = tmp_path / "shards"
    convert_legacy_cache(cache, output, info_path=info, shard_size=3)
    ds = LatentSequenceDataset(output, history=2, future=1, hist_last=1)
    assert len(ds) == 4
    first = ds[0]
    assert first["tokens"] == ["scene-0001-t0", "scene-0001-t1", "scene-0001-t2"]
    np.testing.assert_array_equal(first["history_mask"], [False, True])
    assert float(first["history_latents"][0].sum()) == 0
    assert float(first["history_latents"][1, 0, 0, 0]) == 1
    assert float(first["future_latents"][0, 0, 0, 0]) == 2
    assert ds[2]["scene"] == "scene-0002"
    first["future_latents"][:] = 999
    assert float(ds[0]["future_latents"][0, 0, 0, 0]) == 2
    assert not list(output.rglob("*.pkl"))


def test_missing_metadata_frame_does_not_become_consecutive(tmp_path):
    from reactoccflow.data.cache import convert_legacy_cache, LatentSequenceDataset
    cache, info = fixtures(tmp_path, missing=True)
    output = tmp_path / "shards"
    convert_legacy_cache(cache, output, info_path=info)
    ds = LatentSequenceDataset(output, history=2, future=1)
    assert len(ds) == 2
    assert all(ds[i]["scene"] == "scene-0002" for i in range(len(ds)))


def test_duplicate_tokens_are_rejected(tmp_path):
    from reactoccflow.data.cache import convert_legacy_cache
    cache, info = fixtures(tmp_path)
    with cache.open("rb") as f:
        records = pickle.load(f)
    records.append(records[0])
    with cache.open("wb") as f:
        pickle.dump(records, f)
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        convert_legacy_cache(cache, tmp_path / "shards", info_path=info)


def test_identity_parser_ignores_absolute_prefix():
    from reactoccflow.data.cache import parse_label_identity
    assert parse_label_identity([['data/nuscenes/gts/scene-0001/abc/labels.npz']]) == ("scene-0001", "abc")
    with pytest.raises(ValueError):
        parse_label_identity("invalid.npz")


def test_shape_mismatch_is_rejected_without_publishing_manifest(tmp_path):
    from reactoccflow.data.cache import convert_legacy_cache
    cache, info = fixtures(tmp_path)
    with cache.open("rb") as f:
        records = pickle.load(f)
    records[0]["x_sampled"] = np.zeros((1, 4, 3, 3), np.float32)
    with cache.open("wb") as f:
        pickle.dump(records, f)
    output = tmp_path / "shards"
    with pytest.raises(ValueError, match="shape"):
        convert_legacy_cache(cache, output, info_path=info)
    assert not (output / "manifest.json").exists()


def test_explicit_first_duplicate_policy_preserves_original_baseline_selection(tmp_path):
    from reactoccflow.data.cache import convert_legacy_cache, LatentSequenceDataset
    cache, info = fixtures(tmp_path)
    with cache.open('rb') as f:
        records = pickle.load(f)
    duplicate = dict(records[0])
    duplicate['x_sampled'] = np.full((1, 2, 3, 3), 999, np.float32)
    records.append(duplicate)
    with cache.open('wb') as f:
        pickle.dump(records, f)
    output = tmp_path / 'shards'
    manifest = convert_legacy_cache(cache, output, info_path=info, duplicate_policy='first')
    assert manifest['num_frames'] == 8
    assert len(manifest['duplicate_records']) == 1
    assert manifest['duplicate_records'][0]['same_latent'] is False
    ds = LatentSequenceDataset(output, history=2, future=1)
    assert float(ds[3]['future_latents'][0, 0, 0, 0]) == 3


def test_missing_metadata_is_rejected_without_publishing_manifest(tmp_path):
    from reactoccflow.data.cache import convert_legacy_cache
    cache, _ = fixtures(tmp_path)
    output = tmp_path / 'shards'
    with pytest.raises(ValueError, match='info_path.*metadata'):
        convert_legacy_cache(cache, output)
    assert not output.exists()


def test_missing_metadata_is_rejected_before_reading_cache(tmp_path):
    from reactoccflow.data.cache import convert_legacy_cache
    cache = tmp_path / 'invalid.pkl'
    cache.write_bytes(b'not a pickle')
    output = tmp_path / 'shards'
    with pytest.raises(ValueError, match='info_path.*metadata'):
        convert_legacy_cache(cache, output)
    assert not output.exists()


@pytest.mark.parametrize('timestamps', [[0, 1000000, 500000, 1500000], [0, None, 1000000, 1500000]])
def test_invalid_metadata_timestamps_are_rejected_without_publication(tmp_path, timestamps):
    from reactoccflow.data.cache import convert_legacy_cache
    cache, info = fixtures(tmp_path)
    with info.open('rb') as stream:
        metadata = pickle.load(stream)
    for frame, stamp in zip(metadata['infos']['scene-0001'], timestamps):
        if stamp is None:
            frame.pop('timestamp')
        else:
            frame['timestamp'] = stamp
    with info.open('wb') as stream:
        pickle.dump(metadata, stream)
    output = tmp_path / 'shards'
    with pytest.raises(ValueError, match='timestamp'):
        convert_legacy_cache(cache, output, info_path=info)
    assert not output.exists()

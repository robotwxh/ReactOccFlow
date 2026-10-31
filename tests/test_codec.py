"""Codec contracts use the actual migrated network at a small spatial size."""
import pytest
import torch
from reactoccflow.models.codec import OccupancyCodec


def small_codec():
    return OccupancyCodec(image_size=20, encoder_depth=(1, 1), decoder_depth=(1, 1)).eval()


def test_codec_mean_and_decode_shapes():
    torch.set_num_threads(1)
    codec = small_codec()
    occupancy = torch.zeros(1, 20, 20, 16, dtype=torch.long)
    with torch.no_grad():
        z = codec.encode(occupancy)
        assert z.shape == (1, 16, 5, 5)
        assert torch.equal(z, codec.encode(occupancy))
        assert codec.decode(z).shape == occupancy.shape
        assert codec.decode(z, return_logits=True).shape == (*occupancy.shape, 18)


def test_codec_video_batch_and_local_generator():
    codec = small_codec()
    occupancy = torch.zeros(2, 2, 20, 20, 16, dtype=torch.long)
    with torch.no_grad():
        z = codec.encode(occupancy)
        assert z.shape == (2, 2, 16, 5, 5)
        assert codec.decode(z).shape == occupancy.shape
        state = torch.random.get_rng_state()
        first = codec.encode(occupancy, sample=True, generator=torch.Generator().manual_seed(7))
        second = codec.encode(occupancy, sample=True, generator=torch.Generator().manual_seed(7))
        assert torch.equal(first, second)
        assert not torch.equal(first, z)
        assert torch.equal(state, torch.random.get_rng_state())


def test_codec_visibility_and_invalid_shape():
    codec = small_codec()
    occupancy = torch.zeros(1, 20, 20, 16, dtype=torch.long)
    occupancy[:, 0] = -100
    visible = occupancy >= 0
    with torch.no_grad():
        assert torch.isfinite(codec.encode(occupancy, visibility=visible)).all()
    with pytest.raises(ValueError, match='shape|spatial'):
        codec.encode(torch.zeros(1, 19, 20, 16))
    with pytest.raises(ValueError, match='generator'):
        codec.encode(torch.zeros(1, 20, 20, 16), sample=True)


def test_strict_checkpoint_mapping(tmp_path):
    codec = small_codec()
    state = {f'module.{key}': value for key, value in codec.state_dict().items()}
    path = tmp_path / 'codec.pt'
    torch.save({'state_dict': state}, path)
    report = codec.load_pretrained(path)
    assert report['missing_keys'] == [] and report['unexpected_keys'] == []
    state['module.unexpected_parameter'] = torch.zeros(1)
    torch.save({'state_dict': state}, path)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        codec.load_pretrained(path)


def test_explicit_ema_checkpoint_selection(tmp_path):
    codec = small_codec()
    state = {f'module.{key}': value for key, value in codec.state_dict().items()}
    path = tmp_path / 'ema.pt'
    torch.save({'state_dict': state, 'ema_model': {'n_averaged': torch.tensor(5), **state}}, path)
    assert codec.load_pretrained(path, weights='ema_model')['missing_keys'] == []
    with pytest.raises(ValueError, match='weights'):
        codec.load_pretrained(path, weights='unknown')

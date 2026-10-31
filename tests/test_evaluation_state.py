import numpy as np
import pytest


def test_seed_is_stable_for_identity_not_processing_order():
    from reactoccflow.evaluation.baseline import window_seed
    assert window_seed(2026, 'scene-1', ['t0','t1']) == window_seed(2026, 'scene-1', ['t0','t1'])
    assert window_seed(2026, 'scene-1', ['t0','t1']) != window_seed(2026, 'scene-2', ['t0','t1'])


def test_resume_roundtrips_counts_and_refuses_setting_changes(tmp_path):
    from reactoccflow.evaluation.baseline import EvaluationState
    state = EvaluationState({'nfe':10, 'seed':2026}, times=1)
    pred,truth=np.array([[0,1]]),np.array([[0,0]])
    state.update(pred,truth,np.ones_like(pred,bool))
    state.save(tmp_path/'state.npz')
    resumed=EvaluationState({'seed':2026,'nfe':10},times=1)
    resumed.load(tmp_path/'state.npz')
    assert resumed.processed==1
    assert resumed.masked.compute()['seen'][0][0]==2
    changed=EvaluationState({'nfe':4,'seed':2026},times=1)
    with pytest.raises(ValueError,match='settings'):
        changed.load(tmp_path/'state.npz')


@pytest.fixture
def evaluation_fixture(tmp_path, monkeypatch):
    """Exercise the evaluator with real inputs/state and lightweight model doubles."""
    import argparse
    import json
    import torch
    import reactoccflow.models as models

    cache = tmp_path / 'cache'
    cache.mkdir()
    np.save(cache / 'latents-00000.npy', np.zeros((13, 16, 50, 50), np.float32))
    np.save(cache / 'trajectory.npy', np.zeros((13, 2), np.float32))
    frames = [{'scene': 'scene-1', 'token': f't{i}', 'scene_position': i,
               'timestamp_us': i * 500000, 'shard': 0, 'row': i} for i in range(13)]
    (cache / 'manifest.json').write_text(json.dumps({
        'version': 1, 'frames': frames, 'shards': [{'file': 'latents-00000.npy', 'count': 13}]}))
    for i in range(6, 13):
        folder = tmp_path / 'data' / 'gts' / 'scene-1' / f't{i}'
        folder.mkdir(parents=True)
        np.savez_compressed(folder / 'labels.npz', semantics=np.zeros((200, 200, 16), np.uint8),
                            mask_camera=np.ones((200, 200, 16), bool))
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    for filename in ['occfm_history.ckpt', 'occfm_vae.ckpt']:
        (checkpoints / filename).write_bytes(b'fixture-weights')

    class SmallModel:
        def __init__(self, **kwargs):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_pretrained(self, *args, **kwargs):
            return {}

        def sample(self, history, trajectories, *, initial_noise, nfe):
            return initial_noise

        def decode(self, latents):
            return torch.zeros((len(latents), 200, 200, 16), dtype=torch.uint8)

    monkeypatch.setattr(models, 'ConditionalFlowMatching', SmallModel)
    monkeypatch.setattr(models, 'OccupancyCodec', SmallModel)
    return argparse.Namespace(cache=str(cache), data_root=str(tmp_path / 'data'),
        checkpoints=str(checkpoints), variant='history', weights='ema_model', device='cpu',
        nfe=1, seed=2026, batch_size=1, decode_chunk=6, cpu_threads=1, limit=1,
        output=str(tmp_path / 'output'))


@pytest.mark.parametrize('changed_payload', ['latents', 'trajectory', 'label'])
def test_resume_refuses_changed_payload_without_manifest_change(evaluation_fixture, changed_payload):
    from pathlib import Path
    from reactoccflow.evaluation.baseline import evaluate

    args = evaluation_fixture
    evaluate(args)
    output = Path(args.output) / 'metrics.json'
    old_report = output.read_bytes()
    if changed_payload == 'label':
        path = Path(args.data_root) / 'gts' / 'scene-1' / 't6' / 'labels.npz'
        np.savez_compressed(path, semantics=np.ones((200, 200, 16), np.uint8),
                            mask_camera=np.ones((200, 200, 16), bool))
    else:
        name = 'latents-00000.npy' if changed_payload == 'latents' else 'trajectory.npy'
        path = Path(args.cache) / name
        array = np.load(path)
        array.flat[0] = 1
        np.save(path, array)
    with pytest.raises(ValueError, match='settings'):
        evaluate(args)
    assert output.read_bytes() == old_report


def test_state_preserves_measured_peak_memory(tmp_path):
    from reactoccflow.evaluation.baseline import EvaluationState
    settings = {'device': 'cuda:0', 'gpu_name': 'test-gpu'}
    state = EvaluationState(settings)
    state.peak_cuda_bytes = 123456
    state.compute_seconds = 4.5
    state.batch_latencies = [4.5]
    state.save(tmp_path / 'state.npz')
    resumed = EvaluationState(settings)
    resumed.load(tmp_path / 'state.npz')
    assert resumed.peak_cuda_bytes == 123456
    assert resumed.compute_seconds == 4.5
    assert resumed.batch_latencies == [4.5]
    assert resumed.settings == settings


def test_resume_refuses_changed_timing_hardware_configuration(evaluation_fixture):
    from reactoccflow.evaluation.baseline import evaluate
    args = evaluation_fixture
    report = evaluate(args)
    assert report['device'] == 'cpu'
    args.cpu_threads = 2
    with pytest.raises(ValueError, match='settings'):
        evaluate(args)


def test_completed_resume_preserves_report_without_constructing_models(evaluation_fixture, monkeypatch):
    from pathlib import Path
    import reactoccflow.models as models
    from reactoccflow.evaluation.baseline import evaluate
    args = evaluation_fixture
    first = evaluate(args)
    old_report = (Path(args.output) / 'metrics.json').read_bytes()

    def unexpected_model(**kwargs):
        raise AssertionError('Completed evaluation should not load another model')

    monkeypatch.setattr(models, 'ConditionalFlowMatching', unexpected_model)
    resumed = evaluate(args)
    assert resumed == first
    assert (Path(args.output) / 'metrics.json').read_bytes() == old_report


def test_interrupted_resume_counts_each_window_once(evaluation_fixture, monkeypatch):
    from pathlib import Path
    import reactoccflow.models as models
    from reactoccflow.evaluation.baseline import evaluate

    args = evaluation_fixture
    args.limit = 2
    original_sample = models.ConditionalFlowMatching.sample
    noises = []

    def stop_on_second_window(self, history, trajectories, *, initial_noise, nfe):
        noises.append(initial_noise.clone())
        if len(noises) == 2:
            raise RuntimeError('interrupted fixture')
        return original_sample(self, history, trajectories, initial_noise=initial_noise, nfe=nfe)

    monkeypatch.setattr(models.ConditionalFlowMatching, 'sample', stop_on_second_window)
    with pytest.raises(RuntimeError, match='interrupted fixture'):
        evaluate(args)
    with np.load(Path(args.output) / 'state.npz') as state:
        assert int(state['processed']) == 1

    def finish_window(self, history, trajectories, *, initial_noise, nfe):
        assert np.array_equal(initial_noise.numpy(), noises[1].numpy())
        return original_sample(self, history, trajectories, initial_noise=initial_noise, nfe=nfe)

    monkeypatch.setattr(models.ConditionalFlowMatching, 'sample', finish_window)
    resumed = evaluate(args)
    assert resumed['status'] == 'complete'
    assert resumed['processed_windows'] == 2
    assert resumed['unmasked']['windows'] == 2
    assert resumed['unmasked']['valid_voxels'] == [1280000] * 6
    assert resumed['masked_camera']['standard']['mean_iou'] == 1.0

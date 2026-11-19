from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from reactoccflow.data.loader import shutdown_data_loaders
from reactoccflow.evaluation.history_input_comparison import export_final_ema, validate_training_configs


def test_shutdown_releases_live_persistent_workers_and_is_idempotent():
    loader = DataLoader(TensorDataset(torch.arange(8)), num_workers=2, persistent_workers=True)
    iterator = iter(loader)
    next(iterator)
    workers = list(iterator._workers)
    assert all(worker.is_alive() for worker in workers)
    shutdown_data_loaders(None, loader)
    assert loader._iterator is None
    assert not any(worker.is_alive() for worker in workers)
    shutdown_data_loaders(loader)


def test_shutdown_does_not_start_an_unused_loader():
    loader = DataLoader(TensorDataset(torch.arange(8)), num_workers=2, persistent_workers=True)
    shutdown_data_loaders(loader)
    assert loader._iterator is None


@pytest.mark.parametrize('tamper', ['seed', 'warmup_steps', 'field_missing'])
def test_comparison_rejects_training_changes_beyond_input_mode(tamper):
    dense = {'history_visibility': 'dense_semantic', 'seed': 2026, 'warmup_steps': 1000,
             'diagnostic_train_windows': None}
    masked = {**dense, 'history_visibility': 'mask_lidar'}
    assert validate_training_configs(dense, masked)['differing_field'] == 'history_visibility'
    if tamper == 'field_missing':
        del masked['diagnostic_train_windows']
    else:
        masked[tamper] += 1
    with pytest.raises(ValueError, match='differ beyond history input'):
        validate_training_configs(dense, masked)


@pytest.mark.parametrize('step,mode', [(9999, 'mask_lidar'), (10000, 'dense_semantic')])
def test_final_export_rejects_wrong_step_or_mode_before_model_loading(tmp_path, step, mode):
    source = tmp_path/'source.pt'
    torch.save({'global_step': step, 'config': {'max_steps': 10000, 'history_visibility': mode,
                                               'backbone': 'occfm_unet'}}, source)
    with pytest.raises(ValueError, match='exactly10000'):
        export_final_ema(source, tmp_path/'export.pt', {}, 'mask_lidar')
    assert not (tmp_path/'export.pt').exists()

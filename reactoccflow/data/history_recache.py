"""Reuse verified unchanged targets while recoding observed history only.

Project-owned glue after searches in ground_mask_training_sources_20261118.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .e0_dataset import E0ResearchDataset, payload_sha256


def verify_reuse_cache(root, settings, anchors):
    dataset = E0ResearchDataset(root, verify_checksums=True)
    wrapper = json.loads((Path(root)/'settings.json').read_text())
    previous = wrapper['settings']
    digest = hashlib.sha256(json.dumps(previous, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if digest != wrapper['settings_sha256'] or digest != dataset.manifest['settings_sha256']:
        raise ValueError('Target reuse cache settings identity mismatch')
    matched = ('group', 'source_occupancy_sha256', 'source_manifest_sha256', 'label_manifest_sha256',
               'split_sha256', 'normalization_sha256', 'checkpoint_sha256', 'source_frame',
               'history_slots', 'hist_last', 'future', 'posterior', 'dtype', 'include_future_semantics',
               'torch_version', 'cuda_version', 'allow_tf32_matmul', 'allow_tf32_cudnn', 'cudnn_benchmark')
    for key in matched:
        if key not in settings or key not in previous or settings[key] != previous[key]:
            raise ValueError('Target reuse setting differs: '+key)
    old = [(r['scene'], r['token'], r['global_index']) for r in dataset.records]
    new = [(r['scene'], r['token'], r['global_index']) for r in anchors]
    if old != new:
        raise ValueError('Target reuse window identities/order differ')
    return dataset, {'root': str(Path(root).resolve()), 'manifest_sha256': payload_sha256(Path(root)/'manifest.json'),
                     'settings_sha256': digest, 'matched_fields': list(matched), 'windows': len(old)}


def reuse_targets(previous, observed, *, source_record_sha256):
    before, after = previous['observation'], observed['observation']
    if set(before) != set(after):
        raise ValueError('Recoded observation fields differ')
    for key in before:
        if key != 'history_latents' and not np.array_equal(before[key], after[key]):
            raise ValueError('Non-latent observation differs: '+key)
    for key in ('scene', 'token', 'timestamp_us', 'history_sample_tokens', 'history_latent_sample_tokens',
                'coordinate_frame', 'source_frame', 'actor_instance_tokens'):
        if key not in observed['metadata'] or observed['metadata'][key] != previous['metadata'][key]:
            raise ValueError('Recoded history metadata differs: '+key)
    return {'observation': after, 'target': previous['target'],
            'metadata': {**previous['metadata'], **observed['metadata'],
                         'reused_target_source_record_sha256': source_record_sha256}}

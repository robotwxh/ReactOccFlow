"""Copy published artifacts and materialize an independent validation resource set."""
import codecs
import collections
import concurrent.futures
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import yaml

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE / 'resources/occfm_audit'


class NumpyReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            return getattr(np.core.multiarray, name)
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        if module == '_codecs' and name == 'encode':
            return codecs.encode
        if module in ('__builtin__', 'builtins') and name == 'bytes':
            return bytes
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        raise pickle.UnpicklingError(f'Unapproved pickle type: {module}.{name}')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def config(path):
    """Resolve only copied configuration resources, never reference-directory paths."""
    def merge(node):
        result = {}
        if '_BASE_CONFIG_' in node:
            base = (ROOT / 'source' / node['_BASE_CONFIG_']).resolve()
            if not base.is_relative_to((ROOT / 'source').resolve()):
                raise ValueError('Configuration escapes copied resources')
            result.update(merge(yaml.safe_load(base.read_text())))
        for key, value in node.items():
            if key == '_BASE_CONFIG_':
                continue
            if isinstance(value, dict):
                prior = result.get(key, {})
                result[key] = {**prior, **merge(value)}
            else:
                result[key] = value
        return result
    flat = merge(yaml.safe_load(path.read_text()))
    flat['MODEL'] = {key: flat.pop(key) for key in list(flat)
                     if key in ('COMPRESSOR_CONFIG', 'TRANSITION_MODEL_CONFIG', 'PLANNER_CONFIG')}
    flat['MODEL']['NAME'] = flat['NAME']
    # The runner strictly loads the full checkpoint; these paths are never used.
    flat['MODEL'].get('COMPRESSOR_CONFIG', {}).pop('PRETRAIN_WEIGHT', None)
    flat.pop('DATA_CONFIG', None)
    return flat


def main():
    reference = BASE / 'OccFM'
    ROOT.mkdir(parents=True, exist_ok=True)
    provenance = []
    def copy(source, relative):
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        source_hash, destination_hash = sha256(source), sha256(destination)
        if source_hash != destination_hash:
            raise ValueError(f'Copy checksum mismatch: {source}')
        provenance.append({'source': str(source), 'copy': str(relative), 'sha256': source_hash})
        return destination
    for source in (reference / 'tools/cfgs').rglob('*.yaml'):
        copy(source, Path('source') / source.relative_to(reference))
    checkpoints = []
    for source in sorted((reference / 'logs').rglob('*.ckpt')):
        relative = source.relative_to(reference)
        copied = copy(source, Path('source') / relative)
        yaml_path = next(source.parents[1].glob('*.yaml'))
        yaml_copy = copy(yaml_path, Path('source') / yaml_path.relative_to(reference))
        group = relative.parts[1]
        kind = {'occfm': 'flow_hist', 'occfm_fut': 'flow_future',
                'occfm_vae': 'vae2d', 'occfm_3dvae': 'vae3d'}[group]
        epoch = int(source.stem.split('=')[1])
        identifier = f'{kind}_e{epoch:03d}'
        flat = config(yaml_copy)
        flat_path = ROOT / 'configs' / f'{identifier}.json'
        flat_path.parent.mkdir(exist_ok=True)
        flat_path.write_text(json.dumps(flat, indent=2))
        checkpoints.append({'id': identifier, 'kind': kind, 'file_epoch': epoch,
                            'checkpoint': str(copied.relative_to(ROOT)),
                            'sha256': provenance[-2]['sha256'] if provenance[-2]['copy']==str(Path('source')/relative) else sha256(copied),
                            'config': str(flat_path.relative_to(ROOT))})
    info_path = copy(reference / 'data/nuscenes_infos_val_temporal_v3_scene.pkl', Path('input/validation_info.pkl'))
    cache_path = copy(reference / 'data/nusc_latent_vae/x16/nus_sem_occ_validation.pkl', Path('input/validation_latent.pkl'))
    with info_path.open('rb') as handle:
        infos = NumpyReader(handle).load()['infos']
    frames = [(scene, frame['token']) for scene, rows in infos.items() for frame in rows]
    tokens = [token for scene, token in frames]
    token_index = {token: index for index, token in enumerate(tokens)}
    gt_path = ROOT / 'occupancy.npy'
    if not gt_path.exists():
        occupancy = np.lib.format.open_memmap(gt_path, mode='w+', dtype=np.uint8, shape=(len(frames), 200, 200, 16))
        def load_one(item):
            index, (scene, token) = item
            path = Path('/home/lthpc/data/nuscenes/gts') / scene / token / 'labels.npz'
            with np.load(path, allow_pickle=False) as archive:
                value = archive['semantics']
                assert value.shape == (200, 200, 16) and value.min() >= 0 and value.max() < 18
                occupancy[index] = value
            return index
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for count, _ in enumerate(executor.map(load_one, enumerate(frames)), 1):
                if count % 1000 == 0:
                    print(f'Materialized GT {count}/{len(frames)}', flush=True)
        occupancy.flush()
        del occupancy
    with cache_path.open('rb') as handle:
        cache = NumpyReader(handle).load()
    by_token = {}
    for record in cache:
        path = record['gt_path'][0]
        while isinstance(path, (tuple, list)):
            path = path[0]
        by_token.setdefault(Path(path).parent.name, record)
    selected = [token for token in tokens if token in by_token]
    latent_path = ROOT / 'latents.npy'
    if not latent_path.exists():
        latents = np.lib.format.open_memmap(latent_path, mode='w+', dtype=np.float32, shape=(len(selected), 16, 50, 50))
        for index, token in enumerate(selected):
            latents[index] = by_token[token]['x_sampled'][0]
        latents.flush()
        del latents
    trajectories = np.stack([by_token[token]['gt_trajs'].reshape(2) for token in selected])
    np.save(ROOT / 'trajectories.npy', trajectories)
    latent_gt_indices = [token_index[token] for token in selected]
    flow_windows = []
    for index in range(len(selected)-11):
        gt_indices = latent_gt_indices[index:index+12]
        if len({frames[k][0] for k in gt_indices}) == 1:
            assert all(b == a+1 for a,b in zip(gt_indices, gt_indices[1:]))
            flow_windows.append(list(range(index, index+12)))
    vae_windows = [list(range(index,index+6)) for index in range(len(frames)-5)
                   if frames[index][0] == frames[index+5][0]]
    manifest = {'checkpoints': checkpoints, 'frames': frames, 'latent_gt_indices': latent_gt_indices,
                'flow_windows': flow_windows, 'vae3d_windows': vae_windows,
                'counts': {'vae2d_frames': len(frames), 'vae3d_windows': len(vae_windows),
                           'flow_windows': len(flow_windows), 'unique_latent_frames': len(selected)}}
    assert len(checkpoints)==10 and len(flow_windows)==4219 and len(frames)==6019
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (BASE / 'docs/provenance/occfm_audit_materials.json').write_text(json.dumps({'files': provenance, 'counts':manifest['counts']}, indent=2))
    print(json.dumps(manifest['counts']), flush=True)


if __name__ == '__main__':
    main()

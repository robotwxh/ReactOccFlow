"""Project-owned command line tools; run ``python -m reactoccflow --help``."""
import argparse
import json
from pathlib import Path


def _emit(value, path=None):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    print(text, end='')


def doctor(cpu_only=False):
    import platform
    import torch
    import numpy as np
    from reactoccflow import __version__
    report = {'package_version': __version__, 'python': platform.python_version(),
              'torch': torch.__version__, 'numpy': np.__version__,
              'cpu_tensor_check': bool((torch.arange(4.) @ torch.arange(4.)).item() == 14.),
              'cuda_available': None, 'devices': []}
    if not cpu_only:
        report['cuda_available'] = torch.cuda.is_available()
        for device in range(torch.cuda.device_count()):
            value = torch.arange(4., device=f'cuda:{device}')
            passed = bool((value @ value).item() == 14.)
            report['devices'].append({'index': device, 'name': torch.cuda.get_device_name(device),
                                      'tensor_check': passed})
    return report


def baseline(args):
    import time
    import numpy as np
    import torch
    from reactoccflow.data.cache import LatentSequenceDataset
    from reactoccflow.models import OccupancyCodec, ConditionalFlowMatching
    from reactoccflow.evaluation.occupancy import occupancy_metrics
    device = torch.device(args.device)
    dataset = LatentSequenceDataset(args.cache, history=6, future=6, hist_last=4)
    item = dataset[args.index]
    history = torch.from_numpy(item['history_latents']).unsqueeze(0).to(device)
    trajectory = torch.from_numpy(item['trajectory']).unsqueeze(0).to(device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    codec = OccupancyCodec().to(device).eval()
    codec_report = codec.load_pretrained(Path(args.checkpoints) / 'occfm_vae.ckpt')
    report = {'task': 'single_window_compatibility_smoke', 'not_dataset_benchmark': True,
              'scene': item['scene'], 'tokens': item['tokens'], 'seed': args.seed,
              'nfe': args.nfe, 'checkpoint_partition': args.weights, 'device': str(device),
              'coordinate_frame': 'legacy_per_frame_ego', 'codec_load': codec_report,
              'methods': {}}
    truths, camera_masks, lidar_masks = [], [], []
    for token in item['tokens'][6:]:
        path = Path(args.data_root) / 'gts' / item['scene'] / token / 'labels.npz'
        with np.load(path) as label:
            truths.append(label['semantics'].copy())
            camera_masks.append(label['mask_camera'].copy() if 'mask_camera' in label else None)
            lidar_masks.append(label['mask_lidar'].copy() if 'mask_lidar' in label else None)
    target = np.stack(truths)
    visibility = np.stack(camera_masks) if all(m is not None for m in camera_masks) else None
    report['metric_visibility'] = 'mask_camera' if visibility is not None else 'all_nonunknown_voxels'
    with torch.inference_mode():
        current_token = item['tokens'][5]
        with np.load(Path(args.data_root) / 'gts' / item['scene'] / current_token / 'labels.npz') as label:
            current = torch.from_numpy(label['semantics'].astype(np.int64)).unsqueeze(0).to(device)
        reconstructed = codec.decode(codec.encode(current)).cpu().numpy()[0]
        report['vae_current_reconstruction'] = occupancy_metrics(reconstructed, current.cpu().numpy()[0])
        np.savez_compressed(output / 'vae_reconstruction.npz', prediction=reconstructed)
        for name, length, condition in [('history', 6, trajectory[:, :6]), ('future', 12, trajectory)]:
            model = ConditionalFlowMatching(trajectory_length=length).to(device).eval()
            loaded = model.load_pretrained(Path(args.checkpoints) / f'occfm_{name}.ckpt', weights=args.weights)
            generator = torch.Generator(device=device).manual_seed(args.seed)
            noise = torch.randn((1, 6, 16, 50, 50), generator=generator, device=device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            predicted = model.sample(history, condition, initial_noise=noise, nfe=args.nfe)
            predictions = np.stack([codec.decode(predicted[:, t]).cpu().numpy()[0] for t in range(6)])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            result = {'load': loaded, 'prediction_and_decode_seconds': elapsed,
                      'peak_cuda_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                      'standard': occupancy_metrics(predictions, target, visibility=visibility),
                      'legacy_unmasked': occupancy_metrics(predictions, target, protocol='legacy'),
                      'per_time_standard': [occupancy_metrics(predictions[t], target[t], visibility=None if visibility is None else visibility[t]) for t in range(6)]}
            # List excluded key counts instead of flooding the user report.
            excluded = result['load'].pop('excluded_compressor_keys', [])
            result['load']['excluded_compressor_key_count'] = len(excluded)
            report['methods'][name] = result
            np.savez_compressed(output / f'{name}_prediction.npz', prediction=predictions,
                                latent=predicted.cpu().numpy(), tokens=np.asarray(item['tokens'][6:]))
            del model
    _emit(report, output / 'metrics.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cmd = commands.add_parser('doctor', help='Check the dedicated runtime and available tensor devices')
    cmd.add_argument('--cpu-only', action='store_true')
    cmd.add_argument('--output')
    cmd = commands.add_parser('inspect-cache', help='Inspect independent sharded latent windows')
    cmd.add_argument('cache')
    cmd.add_argument('--history', type=int, default=6)
    cmd.add_argument('--future', type=int, default=6)
    cmd = commands.add_parser('convert-cache', help='Import a trusted legacy pickle into independent shards')
    cmd.add_argument('source')
    cmd.add_argument('destination')
    cmd.add_argument('--info', required=True, help='Scene metadata with chronological frame timestamps')
    cmd.add_argument('--shard-size', type=int, default=512)
    cmd.add_argument('--duplicate-policy', choices=['error', 'first'], default='error')
    cmd = commands.add_parser('audit-references', help='Compare reference trees with their pre-migration snapshot')
    cmd.add_argument('--root', default='.')
    cmd.add_argument('--snapshot', default='docs/provenance/reference_snapshot.json')
    cmd = commands.add_parser('baseline', help='Run pretrained VAE and both CFM variants on one real cached window')
    cmd.add_argument('--cache', default='data/latents/validation')
    cmd.add_argument('--data-root', required=True)
    cmd.add_argument('--checkpoints', default='assets/checkpoints')
    cmd.add_argument('--index', type=int, default=0)
    cmd.add_argument('--device', default='cpu')
    cmd.add_argument('--nfe', type=int, default=10)
    cmd.add_argument('--seed', type=int, default=2026)
    cmd.add_argument('--weights', choices=['state_dict', 'ema_model'], default='ema_model')
    cmd.add_argument('--output', default='outputs/baseline')
    args = parser.parse_args()
    if args.command == 'doctor':
        _emit(doctor(args.cpu_only), args.output)
    elif args.command == 'inspect-cache':
        from reactoccflow.data.cache import LatentSequenceDataset
        ds = LatentSequenceDataset(args.cache, history=args.history, future=args.future)
        _emit({'frames': len(ds.frames), 'valid_windows': len(ds),
               'scenes': len({frame['scene'] for frame in ds.frames}),
               'coordinate_frame': ds.manifest['coordinate_frame']})
    elif args.command == 'convert-cache':
        from reactoccflow.data.cache import convert_legacy_cache
        result = convert_legacy_cache(args.source, args.destination, info_path=args.info,
                                      shard_size=args.shard_size, duplicate_policy=args.duplicate_policy)
        _emit({'num_frames': result['num_frames'], 'num_shards': len(result['shards']),
               'duplicate_records': result['duplicate_records']})
    elif args.command == 'audit-references':
        import os
        import subprocess
        from reactoccflow.provenance import snapshot_tree, compare_snapshot
        saved = json.loads(Path(args.snapshot).read_text())
        result = {}
        for name, before in saved.items():
            path = Path(args.root) / name
            differences = compare_snapshot(before['entries'], snapshot_tree(path))
            status = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain=v1', '-uall'],
                                             env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'), text=True)
            commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
            if status != before['status']:
                differences.append('Git working-tree status changed')
            if commit != before['commit']:
                differences.append('Git HEAD changed')
            result[name] = differences
        _emit(result)
        if any(result.values()):
            raise SystemExit(1)
    else:
        baseline(args)


if __name__ == '__main__':
    main()

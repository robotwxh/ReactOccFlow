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


if __name__ == '__main__':
    main()

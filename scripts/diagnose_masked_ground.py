"""Training-only coverage and optional frozen-VAE reconstruction diagnostic.

Project-owned after the reference searches in masked_ground_sources_20261118.
Reads observed history only. This is not a future-prediction benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reactoccflow.data.history_visibility import complete_masked_ground_history, history_visibility_mask
from reactoccflow.geometry import inverse_transform, warp_occupancy_nearest
from reactoccflow.provenance import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-manifest', type=Path, default=Path('data/research/train/manifest.json'))
    parser.add_argument('--occupancy-root', type=Path, default=Path('/home/lthpc/data/nuscenes/gts'))
    parser.add_argument('--windows', type=int, default=64)
    parser.add_argument('--codec-checkpoint', type=Path)
    parser.add_argument('--codec-windows', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.windows < 1 or args.codec_windows < 1:
        parser.error('window counts must be positive')
    manifest = json.loads(args.label_manifest.read_text())
    if manifest.get('group') != 'train':
        raise ValueError('This development diagnostic is restricted to training scenes')
    by_token = {frame['token']: frame for frame in manifest['frames']}
    frames = [frame for frame in manifest['frames'] if all(frame['history_sample_tokens'])]
    selected = np.linspace(0, len(frames) - 1, min(args.windows, len(frames)), dtype=int)
    codec_slots = set(np.linspace(0, len(selected)-1, min(args.codec_windows, len(selected)), dtype=int))
    codec = None
    if args.codec_checkpoint is not None:
        import torch
        from reactoccflow.models import OccupancyCodec
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(2026)
        codec = OccupancyCodec().to(args.device).eval()
        codec.load_pretrained(args.codec_checkpoint)
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
    counts = np.zeros((5, 18), dtype=np.int64)
    confusion = {mode: np.zeros((18, 18), dtype=np.int64)
                 for mode in ('dense_semantic', 'mask_lidar', 'mask_lidar_temporal_ground')}
    identities, sources = [], {}
    for number, frame_index in enumerate(selected):
        current = frames[int(frame_index)]
        transform = inverse_transform(np.asarray(current['e0_to_global']))
        grids = []
        for token in current['history_sample_tokens']:
            frame = by_token[token]
            if frame['scene'] != current['scene'] or frame['timestamp_us'] > current['timestamp_us']:
                raise ValueError('History must be within scene and at/before the prediction origin')
            path = args.occupancy_root / frame['scene'] / token / 'labels.npz'
            sources[str(path.resolve())] = sha256_file(path)
            with np.load(path, allow_pickle=False) as arrays:
                grids.append(warp_occupancy_nearest(arrays['semantics'],
                             transform @ np.asarray(frame['e0_to_global']), visibility=arrays['mask_lidar']))
        semantics = np.stack([grid['semantics'] for grid in grids])
        bounds = np.stack([grid['in_bounds'] for grid in grids])
        masks = np.stack([grid['visibility'] for grid in grids])
        base = np.stack([history_visibility_mask(s, b, m) for s, b, m in zip(semantics, bounds, masks)])
        labels, visible = complete_masked_ground_history(semantics, bounds, masks)
        assert np.array_equal(labels[base], semantics[base]) and visible[base].all()
        truth = semantics[-1]
        known = bounds[-1] & (truth != 255)
        added = visible[-1] & ~base[-1] & known
        correct = labels[-1] == truth
        for row, values in enumerate((truth[known], truth[base[-1]], truth[visible[-1] & correct & known],
                                      labels[-1][added], labels[-1][added & correct])):
            counts[row] += np.bincount(values, minlength=18)[:18]
        if codec is not None and number in codec_slots:
            inputs = {'dense_semantic': (truth, known), 'mask_lidar': (truth, base[-1]),
                      'mask_lidar_temporal_ground': (labels[-1], visible[-1])}
            with torch.inference_mode():
                for mode, (grid_labels, grid_visible) in inputs.items():
                    latent = codec.encode(torch.tensor(grid_labels[None], device=args.device).long(),
                                          torch.tensor(grid_visible[None], device=args.device).bool(), sample=False)
                    prediction = codec.decode(latent)[0].cpu().numpy()
                    confusion[mode] += np.bincount((18*truth[known].astype(np.int64)+prediction[known]).ravel(),
                                                   minlength=18*18).reshape(18, 18)
        identities.append({'scene': current['scene'], 'token': current['token'],
                           'history_tokens': current['history_sample_tokens'], 'codec': codec is not None and number in codec_slots})
        if (number+1) % 16 == 0:
            print(json.dumps({'windows_completed': number+1, 'windows_total': len(selected)}), flush=True)
    rows = []
    for category, name in enumerate(manifest['semantic_classes']):
        total, masked, completed, filled, correct = [int(value) for value in counts[:, category]]
        rows.append({'class_id': category, 'class': name, 'dense_voxels': total, 'mask_correct_voxels': masked,
                     'completed_correct_voxels': completed, 'filled_voxels': filled, 'filled_correct_voxels': correct,
                     'mask_coverage_percent': 100*masked/total if total else None,
                     'completed_coverage_percent': 100*completed/total if total else None,
                     'fill_precision_percent': 100*correct/filled if filled else None})
    reconstruction = {}
    if codec is not None:
        for mode, matrix in confusion.items():
            tp = np.diag(matrix)
            denominator = matrix.sum(0)+matrix.sum(1)-tp
            iou = np.divide(tp, denominator, out=np.zeros(18, float), where=denominator > 0)
            reconstruction[mode] = {'confusion': matrix.tolist(), 'class_iou_percent': (100*iou).tolist(),
                                    'miou17_percent': float(100*iou[:17].mean())}
    report = {'status': 'completed', 'group': 'train', 'selection': 'uniform manifest index; no class filtering',
              'windows': len(selected), 'identities': identities, 'coverage': rows,
              'codec_windows': sum(bool(item['codec']) for item in identities),
              'reconstruction': reconstruction, 'source_sha256': sources,
              'manifest_sha256': sha256_file(args.label_manifest),
              'policy_sha256': sha256_file(Path(__file__).resolve().parents[1]/'reactoccflow/data/history_visibility.py'),
              'script_sha256': sha256_file(__file__),
              'codec_sha256': sha256_file(args.codec_checkpoint) if codec is not None else None,
              'limits': ['Training subset only; no future labels read.', 'Coverage/reconstruction, not future IoU.',
                         'Two-observation threshold fixed before running; no validation-driven threshold selection.',
                         'A changed history mode requires new caches and a separately trained checkpoint.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'windows': len(selected), 'output': str(args.output),
                      'other_flat': rows[12], 'reconstruction_other_flat':
                      {mode: value['class_iou_percent'][12] for mode, value in reconstruction.items()}}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

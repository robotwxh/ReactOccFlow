"""Independent validation tensors; inference conditions are separate from targets."""
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2] / 'resources/occfm_audit'


class ValidationData(Dataset):
    def __init__(self, kind, limit=None):
        self.manifest = json.loads((ROOT/'manifest.json').read_text())
        self.kind = kind
        self.occupancy = np.load(ROOT/'occupancy.npy', mmap_mode='r')
        self.latents = np.load(ROOT/'latents.npy', mmap_mode='r') if kind.startswith('flow') else None
        self.trajectory = np.load(ROOT/'trajectories.npy', mmap_mode='r') if kind.startswith('flow') else None
        self.count = (len(self.manifest['flow_windows']) if kind.startswith('flow') else
                      len(self.manifest['vae3d_windows']) if kind=='vae3d' else len(self.manifest['frames']))
        if limit:
            self.count = min(limit, self.count)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.kind.startswith('flow'):
            window = self.manifest['flow_windows'][index]
            ids = [self.manifest['latent_gt_indices'][k] for k in window[6:]]
            history = np.array(self.latents[window[:6]], copy=True)
            trajectory = np.array(self.trajectory[window], copy=True)
            history[:2] = 0
            trajectory[:2] = 0
            return {'target': np.array(self.occupancy[ids], copy=True), 'history': history,
                    'trajectory': trajectory, 'index': index}
        ids = self.manifest['vae3d_windows'][index] if self.kind=='vae3d' else [index]
        return {'target': np.array(self.occupancy[ids], copy=True), 'index': index}

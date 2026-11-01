"""Check an installed DiT-only wheel with independent resources and inference.

Run in the dedicated ReactOccFlow environment after installing a clean wheel.
No optimizer, training process, reference execution or path injection is used.
Adapted from the project's existing isolation workflow after reference searches;
source/removal history is in docs/provenance/remove_unet_sources.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

from reactoccflow.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--output', type=Path, default=Path('docs/implementation/dit_only_isolation.json'))
    parser.add_argument('--research-cache', type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    wheel = (args.wheel or project / 'dist/reactoccflow-0.1.0-py3-none-any.whl').resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        forbidden = ('FlowMatchingV2', 'transition_models/', 'evaluation/baseline.py',
                     'OccFM/', 'IR-WM/')
        if any(any(part in name for part in forbidden) for name in names):
            raise RuntimeError('Wheel contains a removed world backbone or a reference repository')
        assert 'reactoccflow/models/standard_dit.py' in names
        assert 'reactoccflow/training/diagnostics.py' in names
    with tempfile.TemporaryDirectory(prefix='reactoccflow-dit-only-') as temporary:
        isolated = Path(temporary)
        resource = project / 'assets/checkpoints/occfm_vae.ckpt'
        copied = isolated / 'codec.ckpt'
        shutil.copyfile(resource, copied)
        assert copied.stat().st_ino != resource.stat().st_ino and copied.stat().st_nlink == 1
        assert sha256_file(copied) == sha256_file(resource)
        if args.research_cache is not None:
            from reactoccflow.data.e0_dataset import E0ResearchDataset
            data = E0ResearchDataset(args.research_cache, verify_checksums=True)
            sample = data[len(data) // 2]
            np.savez(isolated / 'observations.npz', **sample['observation'])
            (isolated / 'normalization.json').write_text(json.dumps(data.normalization))
        payload = r'''
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
from reactoccflow.models import OccupancyCodec, ReactFlow, StandardDiT
from reactoccflow.training.diagnostics import evaluate_probe

project = Path(sys.argv[1]).resolve()
torch.set_num_threads(2)
torch.manual_seed(2026)
if Path('observations.npz').exists():
    with np.load('observations.npz', allow_pickle=False) as arrays:
        obs = {k: torch.from_numpy(arrays[k]).unsqueeze(0) for k in arrays.files}
    normalization = json.loads(Path('normalization.json').read_text())
else:
    obs = {
        'history_latents': torch.zeros(1, 6, 16, 50, 50),
        'history_mask': torch.tensor([[False, False, True, True, True, True]]),
        'actor_history': torch.zeros(1, 8, 4, 6),
        'actor_observed_mask': torch.zeros(1, 8, 4, dtype=torch.bool),
        'actor_velocity_mask': torch.zeros(1, 8, 4, dtype=torch.bool),
        'actor_size': torch.zeros(1, 8, 3),
        'actor_class': torch.full((1, 8), -1, dtype=torch.long),
    }
    normalization = None
model = ReactFlow(normalization=normalization).eval()
assert isinstance(model.transition_model.trunk, StandardDiT)
assert not hasattr(model, 'load_pretrained_trunk')
assert not any(isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose3d))
               for m in model.modules())
assert importlib.util.find_spec('reactoccflow.evaluation.baseline') is None
assert not hasattr(__import__('reactoccflow.models', fromlist=['x']), 'ConditionalFlowMatching')
plan = torch.zeros(1, 6, 2)  # Explicit stationary candidate, independent of future labels.
codec = OccupancyCodec().eval()
codec.load_pretrained('codec.ckpt')
with torch.inference_mode():
    a = model.sample(obs, plan, generator=torch.Generator().manual_seed(7), nfe=1)
    b = model.sample(obs, plan, generator=torch.Generator().manual_seed(7), nfe=1)
    for key in a:
        assert torch.isfinite(a[key]).all() and torch.equal(a[key], b[key])
    torch.save(model.state_dict(), 'dit_state.pt')
    clone = ReactFlow(normalization=normalization).eval()
    clone.load_state_dict(torch.load('dit_state.pt', weights_only=True), strict=True)
    c = clone.sample(obs, plan, generator=torch.Generator().manual_seed(7), nfe=1)
    for key in a:
        assert torch.equal(a[key], c[key])
    latent = codec.encode(torch.full((1, 200, 200, 16), 17, dtype=torch.long))
    assert latent.shape == (1, 16, 50, 50) and torch.isfinite(latent).all()
    logits = codec.decode(a['future_latents'][:, 0, 0], return_logits=True)
    assert logits.shape == (1, 200, 200, 16, 18) and torch.isfinite(logits).all()
modules = {name: module.__file__ for name, module in sys.modules.items()
           if name.startswith('reactoccflow') and getattr(module, '__file__', None)}
assert all(not Path(file).resolve().is_relative_to(project) for file in modules.values())
assert not Path('OccFM').exists() and not Path('IR-WM').exists()
print(json.dumps({'passed': True, 'backbone': model.backbone,
                  'observations': 'real cached history' if Path('observations.npz').exists() else 'synthetic',
                  'sample_shapes': {k: list(v.shape) for k, v in a.items()},
                  'vae_encoded_shape': list(latent.shape), 'decoded_logits_shape': list(logits.shape),
                  'strict_checkpoint_round_trip': True, 'removed_runtime_modules_absent': True,
                  'loaded_modules': modules, 'training_started': False}, indent=2))
'''
        (isolated / 'run.py').write_text(payload)
        result = subprocess.run([sys.executable, '-I', 'run.py', str(project)],
                                cwd=isolated, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        report = json.loads(result.stdout)
        report.update(wheel=str(wheel), wheel_sha256=sha256_file(wheel), wheel_entries=len(names),
                      resource_sha256=sha256_file(resource), device='cpu')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'loaded_modules'}, indent=2))


if __name__ == '__main__':
    main()

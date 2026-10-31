"""Verify an installed wheel using independent resources outside the checkout.

Run with the dedicated environment after building/installing the wheel. The
reference repositories are never deleted, renamed, imported, or executed.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='docs/implementation/isolation.json')
    parser.add_argument('--research-cache', type=Path, help='Optional E0 cache; copy one independently usable debug fixture')
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    wheel = project / 'dist/reactoccflow-0.1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert all(not name.startswith(('OccFM/', 'IR-WM/')) for name in names)
        assert any(name == 'reactoccflow/data/cache.py' for name in names)
    with tempfile.TemporaryDirectory(prefix='reactoccflow-isolation-') as temporary:
        isolated = Path(temporary)
        assert not (isolated / 'OccFM').exists() and not (isolated / 'IR-WM').exists()
        (isolated / 'assets').mkdir()
        for name in ['occfm_vae.ckpt', 'occfm_history.ckpt', 'occfm_future.ckpt']:
            original = project / 'assets/checkpoints' / name
            target = isolated / 'assets' / name
            shutil.copyfile(original, target)
            assert not target.is_symlink() and original.stat().st_ino != target.stat().st_ino
        if args.research_cache is not None:
            source = args.research_cache.resolve()
            manifest = json.loads((source / 'manifest.json').read_text())
            entry = manifest['records'][0]
            fixture = isolated / 'research_cache'
            fixture.mkdir()
            for key in ('file', 'record_file'):
                original = source / entry[key]
                target = fixture / entry[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                assert not target.is_symlink() and original.stat().st_ino != target.stat().st_ino
            manifest.update(records=[entry], sample_count=1, complete=False)
            (fixture / 'manifest.json').write_text(json.dumps(manifest))
        payload = r'''
import json, os, sys
from pathlib import Path
import numpy as np
import torch
from reactoccflow.data.cache import LatentSequenceDataset
from reactoccflow.models import OccupancyCodec, ConditionalFlowMatching
from reactoccflow.evaluation.occupancy import occupancy_metrics

torch.set_num_threads(4)
project=Path(sys.argv[1]).resolve()
device=torch.device(sys.argv[2])
source=project/'data/latents/validation'
manifest=json.loads((source/'manifest.json').read_text())
frames=[dict(row) for row in manifest['frames'][:12]]
array=np.load(source/manifest['shards'][0]['file'], mmap_mode='r')[:12].copy()
trajectories=np.load(source/'trajectory.npy', mmap_mode='r')[:12].copy()
Path('data').mkdir()
np.save('data/latents-00000.npy', array)
np.save('data/trajectory.npy', trajectories)
for i,row in enumerate(frames):
    row['shard'],row['row']=0,i
manifest.update(frames=frames,num_frames=12,shards=[{'file':'latents-00000.npy','count':12}])
Path('data/manifest.json').write_text(json.dumps(manifest))
# All accesses after preparing this independent fixture use installed code and
# the temporary directory. Source pickle and reference projects are not needed.
del array, trajectories
sample=LatentSequenceDataset('data',hist_last=4)[0]
history=torch.from_numpy(sample['history_latents']).unsqueeze(0).to(device)
future=torch.from_numpy(sample['future_latents']).unsqueeze(0).to(device)
trajectory=torch.from_numpy(sample['trajectory'][:6]).unsqueeze(0).to(device)
codec=OccupancyCodec().to(device).eval()
codec.load_pretrained('assets/occfm_vae.ckpt')
model=ConditionalFlowMatching().to(device)
model.load_pretrained('assets/occfm_history.ckpt')
model.train()
optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-5)
loss=model.training_loss(history,future,trajectory,generator=torch.Generator(device=device).manual_seed(2026))
assert torch.isfinite(loss)
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
optimizer.step()
model.eval()
with torch.inference_mode():
    pred=model.sample(history,trajectory,future_shape=future.shape,generator=torch.Generator(device=device).manual_seed(2026),nfe=1)
    labels=codec.decode(pred[:,0]).cpu().numpy()
    reference=codec.decode(future[:,0]).cpu().numpy()
    assert labels.shape==(1,200,200,16) and torch.isfinite(pred).all()
metric=occupancy_metrics(labels,reference)
baseline_training_loss=float(loss.detach())
# Exercise new geometry, official-table labels, splits and E0 encoding using
# wholly independent numeric fixtures in this temporary directory.
from reactoccflow.geometry import transform_matrix, inverse_transform, transform_points
from reactoccflow.data.research import NuScenesLabelIndex
from reactoccflow.data.e0_cache import encode_e0_training_sample
from reactoccflow.data.splits import grouped_scene_split
raw=Path('raw'); table_root=raw/'v1.0-trainval'; table_root.mkdir(parents=True)
tables={
 'scene':[{'token':'sc','name':'scene-isolated','first_sample_token':'s0','last_sample_token':'s0','nbr_samples':1}],
 'sample':[{'token':'s0','scene_token':'sc','timestamp':1000000,'prev':'','next':''}],
 'sensor':[{'token':'sensor','channel':'LIDAR_TOP'}],
 'calibrated_sensor':[{'token':'cal','sensor_token':'sensor','translation':[0,0,0],'rotation':[1,0,0,0]}],
 'sample_data':[{'token':'lidar','sample_token':'s0','is_key_frame':True,'calibrated_sensor_token':'cal','ego_pose_token':'pose'}],
 'ego_pose':[{'token':'pose','translation':[10,20,0],'rotation':[1,0,0,0]}],
 'category':[{'token':'car','name':'vehicle.car'}],
 'instance':[{'token':'actor','category_token':'car'}],
 'sample_annotation':[{'token':'ann','sample_token':'s0','instance_token':'actor','translation':[15,20,0],'rotation':[1,0,0,0],'size':[2,4,1.5],'num_lidar_pts':3,'visibility_token':'4'}],
}
for table,rows in tables.items(): (table_root/(table+'.json')).write_text(json.dumps(rows))
label_path=raw/'gts/scene-isolated/s0';label_path.mkdir(parents=True)
semantics=np.full((200,200,16),17,np.uint8);semantics[110:115,98:102,0:5]=4
np.savez_compressed(label_path/'labels.npz',semantics=semantics,mask_lidar=np.ones_like(semantics,bool),mask_camera=np.ones_like(semantics,bool))
index=NuScenesLabelIndex(raw)
e0=encode_e0_training_sample(index,'scene-isolated','s0',codec,device,source_frame='ego')
assert e0['observation']['history_mask'].tolist()==[False,False,False,False,False,True]
assert e0['observation']['history_latents'].shape==(6,16,50,50)
assert e0['observation']['actor_history'][0,-1,0]==5
assert e0['target']['future_latents'].shape==(6,16,50,50)
assert not e0['target']['future_latent_mask'].any() and not e0['target']['future_known_mask'].any()
assert not any('future' in key for key in e0['observation'])
matrix=transform_matrix([1,2,3],[1,0,0,0]); points=np.array([[2,4,6.]])
np.testing.assert_allclose(transform_points(inverse_transform(matrix),transform_points(matrix,points)),points)
splits=grouped_scene_split([str(i) for i in range(10)],['val'])
assert len(splits['groups']['train'])==8 and splits['groups']['evaluation']==['val']
research = {}
if Path('research_cache').exists():
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    from reactoccflow.models.react_flow import ReactFlow
    from reactoccflow.losses import ConsistencyLosses, decode_endpoints
    try:
        E0ResearchDataset('research_cache')
    except ValueError as error:
        assert 'incomplete' in str(error)
    else:
        raise AssertionError('Partial-cache default refusal did not hold')
    data = E0ResearchDataset('research_cache', verify_checksums=True, allow_incomplete=True)
    real = data[0]
    observation = {key: torch.from_numpy(value).unsqueeze(0).to(device) for key,value in real['observation'].items()}
    target = {key: torch.from_numpy(value).unsqueeze(0).to(device) for key,value in real['target'].items()}
    del optimizer, model, loss
    joint = ReactFlow(normalization=data.normalization).to(device).train()
    loaded = joint.load_pretrained_trunk('assets/occfm_history.ckpt')
    assert loaded['trunk_key_count'] == 204 and len(loaded['new_branch_keys']) == 74
    base = joint.training_loss(observation, target, target['ego_future'][..., :2],
        generator=torch.Generator(device=device).manual_seed(2026), return_endpoints=True)
    endpoints = base['endpoints']
    continuous = decode_endpoints(codec, endpoints['future_latents'], chunk=1)
    regularizers = ConsistencyLosses()(continuous, endpoints['actor_future'], observation, target)
    total = base['total'] + regularizers['total']
    assert torch.isfinite(total)
    total.backward()
    grads = [parameter.grad for parameter in joint.parameters() if parameter.grad is not None]
    assert len(grads) == 74 and all(torch.isfinite(grad).all() for grad in grads)
    assert all(parameter.grad is None for parameter in codec.parameters())
    assert all(parameter.grad is None for parameter in joint.transition_model.trunk.parameters())
    del continuous, endpoints, base, regularizers, total
    joint.eval()
    # Inference uses an observed-state constant-speed candidate. Logged executed
    # future ego actions above are a separately supplied supervised condition.
    times = torch.arange(1, 7, device=device, dtype=torch.float32) * .5
    candidate = torch.stack((observation['ego_state'][:, 0, None] * times,
                             torch.zeros((1, 6), device=device)), dim=-1)
    with torch.inference_mode():
        worlds = joint.sample(observation, candidate, num_worlds=1, nfe=1,
            generator=torch.Generator(device=device).manual_seed(2026))
    assert worlds['future_latents'].shape == (1, 1, 6, 16, 50, 50)
    assert worlds['actor_future'].shape == (1, 1, 8, 6, 6)
    assert all(torch.isfinite(value).all() for value in worlds.values())
    research = dict(e0_numeric_cache_passed=True, partial_cache_default_refusal=True,
        joint_gradient_tensor_count=len(grads), differentiable_decoder_consistency_passed=True,
        joint_sampling_passed=True, frozen_codec_trunk_gradients_absent=True,
        diagnostic_only=True, independent_research_sample_count=len(data))
# Evaluation against a decoded latent reference is only an interface smoke,
# not ground-truth world-model performance.
loaded=[]
for name,module in sorted(sys.modules.items()):
    if name=='reactoccflow' or name.startswith('reactoccflow.'):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            assert project not in path.parents, f'Checkout import: {path}'
            assert '/OccFM/' not in str(path) and '/IR-WM/' not in str(path)
            loaded.append(str(path))
assert not any(p.is_symlink() or (p.is_file() and p.stat().st_nlink!=1) for p in Path('.').rglob('*'))
print(json.dumps({'passed':True,'reference_directories_present':False,'isolated_python':sys.flags.isolated==1,'loaded_package_files':loaded,'independent_checkpoint_count':3,'data_window_count':1,'training_one_step_passed':True,'training_loss':baseline_training_loss,'inference_shape':list(pred.shape),'evaluation_interface_passed':metric['valid_voxels']>0,'e0_encoding_passed':True,'official_schema_labels_passed':True,'geometry_roundtrip_passed':True,'grouped_split_passed':True,'device':str(device),**research}))
'''
        result = subprocess.run([sys.executable, '-I', '-c', payload, str(project), args.device],
                                cwd=isolated, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        report = json.loads(result.stdout)
        report['wheel_entry_count'] = len(names)
        report['stderr'] = result.stderr
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'loaded_package_files'}, indent=2))


if __name__ == '__main__':
    main()

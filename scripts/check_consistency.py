"""Actual full-resolution E0 endpoint/decoder/consistency gradient acceptance."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from reactoccflow.data.research import NuScenesLabelIndex
from reactoccflow.data.e0_cache import encode_e0_training_sample
from reactoccflow.models import OccupancyCodec
from reactoccflow.models.react_flow import ReactFlow
from reactoccflow.losses import ConsistencyLosses, decode_endpoints
from reactoccflow.provenance import sha256_file


def main():
    device=torch.device('cuda:0');torch.set_num_threads(4);torch.manual_seed(2026)
    output=Path('outputs/consistency_acceptance');output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path('data/research/e0_acceptance/manifest.json').read_text())
    selected=next(sample for sample in manifest['samples'] if sample['scene']=='scene-0002')
    index=NuScenesLabelIndex('/home/lthpc/data/nuscenes',scenes=[selected['scene']])
    codec=OccupancyCodec().to(device);codec.load_pretrained('assets/checkpoints/occfm_vae.ckpt')
    sample=encode_e0_training_sample(index,selected['scene'],selected['token'],codec,device,source_frame='ego',include_future_semantics=True)
    np.savez_compressed(output/'encoded_sample.npz',**{f'{group}.{key}':value for group in ('observation','target') for key,value in sample[group].items()})
    observation={key:torch.from_numpy(value).unsqueeze(0).to(device) for key,value in sample['observation'].items()}
    target={key:torch.from_numpy(value).unsqueeze(0).to(device) for key,value in sample['target'].items()}
    stats=json.loads(Path('data/research/normalization.json').read_text())
    model=ReactFlow(normalization=stats).to(device);model.load_pretrained_trunk('assets/checkpoints/occfm_history.ckpt')
    base=model.training_loss(observation,target,target['ego_future'][...,:2],generator=torch.Generator(device=device).manual_seed(2026),return_endpoints=True)
    endpoints=base['endpoints'];occ=endpoints['future_latents'];actor=endpoints['actor_future']
    assert occ.requires_grad and actor.requires_grad
    occ.retain_grad();actor.retain_grad()
    logits=decode_endpoints(codec,occ,chunk=1)
    losses=ConsistencyLosses()(logits,actor,observation,target)
    assert logits.shape==(1,6,200,200,16,18)
    total=base['total']+losses['total']
    assert torch.isfinite(total)
    geometry_grad=torch.autograd.grad(losses['occ_actor_consistency'],[occ,actor],retain_graph=True,allow_unused=True)
    total.backward()
    assert occ.grad is not None and actor.grad is not None
    assert torch.isfinite(occ.grad).all() and torch.isfinite(actor.grad).all()
    assert not any(parameter.grad is not None for parameter in codec.parameters())
    assert not any(parameter.grad is not None for parameter in model.transition_model.trunk.parameters())
    parameters=[p for p in model.parameters() if p.grad is not None]
    assert parameters and all(torch.isfinite(p.grad).all() for p in parameters)
    report={'passed':True,'scene':selected['scene'],'token':selected['token'],'oracle':True,
            'not_trained_model_evaluation':True,'source_frame':'ego','coordinate_frame':'E0','device':str(device),
            'base':{k:float(v.detach()) for k,v in base.items() if k!='endpoints'},
            'consistency':{k:float(v.detach()) for k,v in losses.items()},'combined_total':float(total.detach()),
            'logits_shape':list(logits.shape),'vae_frozen_no_grad':True,'trunk_frozen_no_grad':True,
            'new_parameter_gradient_tensors':len(parameters),
            'endpoint_gradient_abs_sum':{'occupancy':float(occ.grad.abs().sum()),'actor':float(actor.grad.abs().sum())},
            'geometry_gradient_abs_sum':{'occupancy':None if geometry_grad[0] is None else float(geometry_grad[0].abs().sum()),'actor':None if geometry_grad[1] is None else float(geometry_grad[1].abs().sum())},
            'encoded_sample_sha256':sha256_file(output/'encoded_sample.npz'),
            'semantics_source':'same raw3D E0 warp as target codec encoding; unknown255 and future-known masks preserved',
            'action_condition':'separate logged executed ego trajectory for supervised diagnostic, not generated planner',
            'note':'Losses/gradients only; no optimizer step or learned reaction/closed-loop/calibration claim.'}
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

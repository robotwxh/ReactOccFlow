"""Real oracle-E0 interface/gradient acceptance, never trained model evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from reactoccflow.models.react_flow import ReactFlow
from reactoccflow.provenance import sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('data/research/e0_acceptance'))
    parser.add_argument('--normalization',type=Path,default=Path('data/research/normalization.json'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--backbone', choices=['occfm_unet', 'dit'], default='occfm_unet')
    args=parser.parse_args()
    if args.output is None:
        args.output = Path(f'outputs/{args.backbone}_react_flow_acceptance.json')
    torch.set_num_threads(4);torch.manual_seed(2026)
    device=torch.device(args.device)
    stats=json.loads(args.normalization.read_text());manifest=json.loads((args.cache/'manifest.json').read_text())
    if stats['fit_group']!='train' or manifest['group']!='train':raise ValueError('Require train-only diagnostic inputs')
    report={'status':'running','not_trained_evaluation':True,'oracle':True,'device':str(device),
            'normalization_sha256':sha256_file(args.normalization),'cache_sha256':sha256_file(args.cache/'manifest.json'),
            'initialization':args.backbone+' from scratch','modes':{},
            'training_condition':'logged executed ego future is supplied separately as a supervised action condition',
            'sampling_condition':'constant speed from observed ego_state only; no future labels used',
            'timing':'concurrent engineering checks; no fair throughput claim'}
    for mode in ['joint','parallel','occupancy_only']:
        model=ReactFlow(mode=mode,normalization=stats,backbone=args.backbone).to(device).train()
        results=[]
        selected=manifest['samples'] if mode=='joint' else manifest['samples'][:5]
        for i,sample in enumerate(selected):
            with np.load(args.cache/sample['file'],allow_pickle=False) as data:
                obs={key.split('.',1)[1]:torch.from_numpy(data[key].copy()).unsqueeze(0).to(device) for key in data.files if key.startswith('observation.')}
                target={key.split('.',1)[1]:torch.from_numpy(data[key].copy()).unsqueeze(0).to(device) for key in data.files if key.startswith('target.')}
            model.zero_grad(set_to_none=True)
            loss=model.training_loss(obs,target,target['ego_future'][...,:2],generator=torch.Generator(device=device).manual_seed(2026+i))
            assert all(torch.isfinite(value).all() for value in loss.values())
            loss['total'].backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert any(p.grad is not None for p in model.transition_model.trunk.parameters() if p.requires_grad)
            current_mask=obs['actor_observed_mask'][...,-1]
            row={'scene':sample['scene'],'actors':int(current_mask.sum()),'loss':{k:float(v.detach()) for k,v in loss.items()},'gradient_tensors':len(grads)}
            if not current_mask.any():
                assert float(loss['actor_fm'])==0 and float(loss['actor_existence'])==0
            if i in [0,1]:
                model.eval()
                times=torch.arange(1,7,device=device,dtype=torch.float32)*.5
                candidate=torch.stack((obs['ego_state'][:,0,None]*times,torch.zeros((1,6),device=device)),dim=-1)
                with torch.inference_mode():
                    worlds=model.sample(obs,candidate,num_worlds=2,nfe=2,generator=torch.Generator(device=device).manual_seed(2026+i))
                assert worlds['future_latents'].shape==(1,2,6,16,50,50)
                assert worlds['actor_future'].shape==(1,2,8,6,6)
                assert worlds['actor_existence'].shape==(1,2,8,6)
                assert all(torch.isfinite(x).all() for x in worlds.values())
                assert ((worlds['actor_existence']>=0)&(worlds['actor_existence']<=1)).all()
                row['sampling_two_worlds_nfe2']=True
                model.train()
            results.append(row)
            print(json.dumps({'mode':mode,'scene':sample['scene'],'checked':len(results),'total':len(selected)}),flush=True)
        report['modes'][mode]={'backbone':model.backbone,'architecture':model.transition_model.trunk.config,'sample_count':len(results),'examples':results}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
        del model
    report['status']='passed';args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()

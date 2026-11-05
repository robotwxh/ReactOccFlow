"""Corrected published FVD feature pipeline using actual forecast predictions.

Uses the released 3D VAE, sampled 6x16x50x50 latents, adaptive 5x5 pooling,
flattening, FP32 feature inference and Gaussian Frechet distance. The public
script's prediction overwrite is removed. No unpublished FID/KID encoder is
substituted. GT features are reusable by preserving the interleaved posterior
noise stream: pred, GT, pred, GT, ... with seed 1000.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.linalg import eigh

from .data import ROOT, ValidationData
from .evaluate import make_model, atomic_json


def encode(model, occupancy):
    data={'semantic_occ':occupancy}
    for name in ('embedding','encoder','quantization'):
        data=getattr(model,name)(data)
    features=data['sampled_features']
    pooled=F.adaptive_avg_pool2d(features,(5,5)).reshape(len(occupancy),-1)
    return pooled


def posterior_noise_skip(device):
    torch.randn((6,16,50,50),device=device,dtype=torch.float32)


def statistics(features):
    x=np.asarray(features,dtype=np.float64)
    return x.mean(0),np.cov(x,rowvar=False)


def frechet(first,second):
    if max(len(first),len(second)) < min(first.shape[1],64):
        x,y=np.asarray(first,dtype=np.float64),np.asarray(second,dtype=np.float64)
        mx,my=x.mean(0),y.mean(0)
        x,y=x-mx,y-my
        trace1=(x*x).sum()/(len(x)-1)
        trace2=(y*y).sum()/(len(y)-1)
        cross=np.linalg.svd(x@y.T,compute_uv=False).sum()/np.sqrt((len(x)-1)*(len(y)-1))
        return max(float(np.sum((mx-my)**2)+trace1+trace2-2*cross),0.)
    mu1,cov1=statistics(first);mu2,cov2=statistics(second)
    values,vectors=eigh(cov1,check_finite=False,driver='evd')
    root=(vectors*np.sqrt(np.maximum(values,0)))@vectors.T
    product=root@cov2@root
    eigenvalues=eigh((product+product.T)/2,eigvals_only=True,check_finite=False,driver='evd')
    distance=float(np.sum((mu1-mu2)**2)+np.trace(cov1)+np.trace(cov2)-2*np.sqrt(np.maximum(eigenvalues,0)).sum())
    if distance < -1e-8:raise ValueError(f'Negative FVD beyond numerical tolerance: {distance}')
    return max(distance,0.)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['gt','pred'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--gt-features',type=Path)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=1000)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    resource_hash=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
    if args.mode=='gt' and args.limit is None:
        # An idle GPU may prepare this common input before all forecast jobs end.
        # The lock and bound cache prevent two controllers from recomputing it.
        gt_lock=(args.output/'gt_features.lock').open('a')
        fcntl.flock(gt_lock,fcntl.LOCK_EX)
        existing=args.output/'result.json'
        if existing.exists():
            cached=json.loads(existing.read_text())
            if (cached.get('status')=='completed' and cached.get('validation_scope')=='full'
                    and cached.get('seed')==args.seed and cached.get('resource_manifest_sha256')==resource_hash
                    and cached.get('precision')=='fp32' and cached.get('cudnn_deterministic') is True
                    and np.load(args.output/'features.npy',mmap_mode='r').shape==(4219,2400)):
                print('Reused verified full GT feature cache.',flush=True)
                return
    torch.set_num_threads(4)
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    entry=next(e for e in json.loads((ROOT/'manifest.json').read_text())['checkpoints'] if e['id']=='vae3d_e040')
    model,metadata=make_model(entry,device,False)
    dataset=ValidationData('flow_hist',args.limit)
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=args.workers,pin_memory=True,
                      persistent_workers=args.workers>0)
    predictions=np.load(args.predictions,mmap_mode='r') if args.mode=='pred' else None
    if predictions is not None:assert len(predictions)>=len(dataset)
    features=np.lib.format.open_memmap(args.output/'features.npy',mode='w+',dtype=np.float32,shape=(len(dataset),2400))
    start=time.monotonic();last=start
    with torch.inference_mode():
        for batch in loader:
            index=int(batch['index'][0])
            if args.mode=='gt':
                posterior_noise_skip(device)
                occupancy=batch['target'].to(device,non_blocking=True)
            else:
                occupancy=torch.from_numpy(np.array(predictions[index:index+1],copy=True)).to(device)
            encoded=encode(model,occupancy)
            features[index]=encoded[0].cpu().numpy()
            if args.mode=='pred':posterior_noise_skip(device)
            if time.monotonic()-last>=10 or index+1==len(dataset):
                elapsed=time.monotonic()-start
                state={'status':'running','mode':args.mode,'completed':index+1,'total':len(dataset),
                       'elapsed_seconds':elapsed,'estimated_remaining_seconds':elapsed*(len(dataset)-index-1)/(index+1),'pid':os.getpid()}
                atomic_json(args.output/'status.json',state);print(json.dumps(state),flush=True);last=time.monotonic()
    features.flush()
    result={'status':'completed','validation_scope':'full' if args.limit is None else 'pilot','samples':len(dataset),'resource_manifest_sha256':resource_hash,'feature_checkpoint':entry,'feature_state':metadata['state'],
            'precision':'fp32','seed':args.seed,'cudnn_deterministic':True,'cudnn_benchmark':False,'feature_dimensions':2400,'elapsed_seconds':time.monotonic()-start,
            'protocol':'released feature pipeline with prediction overwrite removed; posterior-sampled latent; adaptive 5x5 pooling; seed-preserving GT reuse'}
    if args.mode=='pred':
        real=np.load(args.gt_features,mmap_mode='r')
        if len(real)!=len(features):raise ValueError('GT/prediction sample counts differ')
        result['fvd_raw']=frechet(real,features)
        result['fvd_in_table3_units_x1e_minus3']=1000*result['fvd_raw']
    result['elapsed_seconds']=time.monotonic()-start
    atomic_json(args.output/'result.json',result)
    atomic_json(args.output/'status.json',{'status':'completed','completed':len(dataset),'total':len(dataset)})
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

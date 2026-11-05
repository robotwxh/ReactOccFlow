"""Evaluate copied OccFM checkpoints without importing the reference repositories.

The vendored architecture retains upstream numerical behavior, including its
attention implementation, CFG convention and repeated softmax. Evaluation
conditions follow the published configurations. Targets enter only the scorer.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict

from .data import ROOT, ValidationData
from .vendor.models import build_network


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def make_model(entry, device, use_ema=True):
    cfg = EasyDict(json.loads((ROOT/entry['config']).read_text()))
    model = build_network(model_cfg=cfg.MODEL, loss_cfg=cfg.LOSS,
                          cache_mode=cfg.CACHE_MODE)
    checkpoint = torch.load(ROOT/entry['checkpoint'], map_location='cpu', weights_only=True)
    ema = bool(use_ema and entry['kind'].startswith('flow') and checkpoint.get('ema_model') is not None)
    state = checkpoint['ema_model'] if ema else checkpoint['state_dict']
    state = {key.removeprefix('module.'): value for key,value in state.items() if key != 'n_averaged'}
    incompatible = model.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    metadata = {'state': 'ema_model' if ema else 'state_dict', 'checkpoint_internal_epoch':checkpoint.get('epoch'),
                'parameter_count':sum(p.numel() for p in model.parameters()), 'strict_load':True,
                'configuration':dict(cfg)}
    return model.to(device).eval(), metadata


def labels_from_maps(model, maps, batch, frames):
    n, _, h, w = maps.shape
    assert n == batch*frames
    embedding = model.embedding
    features = maps.reshape(n, embedding.height_num, embedding.cate, h, w).permute(0,3,4,1,2)
    similarity = features @ embedding.class_embeds.weight.T.detach()
    # Preserve the exact public inference path, including both softmax calls.
    probability = similarity.permute(0,4,1,2,3).softmax(dim=1)
    prediction = probability.softmax(dim=1).argmax(dim=1)
    return prediction.reshape(batch,frames,h,w,embedding.height_num)


def predict_flow(model, history, trajectory, noise_generator):
    batch = history.shape[0]
    history = history*model.rescale_factor
    # Generate one noise tensor per window to retain batch-size-one RNG ordering.
    noise = torch.cat([torch.randn((1,6,*history.shape[2:]),device=history.device,
                                  dtype=history.dtype,generator=noise_generator) for _ in range(batch)])
    data = {'x_sampled':torch.cat([history,noise],dim=1), 'trajectory':trajectory}
    latent = model.sample(data,batch)
    model.decoder.skip = False
    try:
        maps = model.decoder({'sampled_features':latent.reshape(-1,*latent.shape[2:])})['decoded_map']
    finally:
        model.decoder.skip = True
    return labels_from_maps(model,maps,batch,6)


def predict_vae(model, target, kind):
    batch, frames = target.shape[:2]
    data = {'semantic_occ':target[:,0] if kind=='vae2d' else target}
    for module in model.module_list:
        data = module(data)
    return labels_from_maps(model,data['decoded_map'],batch,frames)


class Counts:
    def __init__(self, frames, device):
        self.matrix = torch.zeros((frames,18,18),dtype=torch.int64,device=device)
        self.samples = 0

    def update(self, pred, target):
        for time_index in range(pred.shape[1]):
            truth = target[:,time_index].flatten().long()
            estimated = pred[:,time_index].flatten().long()
            self.matrix[time_index] += torch.bincount(truth*18+estimated,minlength=324).reshape(18,18)
        self.samples += len(pred)

    def compute(self):
        counts = self.matrix.cpu().numpy()
        per_time, classes, occupancy = [], [], []
        for matrix in counts:
            seen, positive = matrix.sum(1),matrix.sum(0)
            correct = np.diag(matrix)
            union = seen+positive-correct
            # Original evaluation assigns 1 to any class absent from GT.
            iou = np.divide(correct,union,out=np.ones(18,dtype=float),where=union>0)
            iou[seen==0] = 1.
            classes.append((100*iou[:17]).tolist())
            per_time.append(float(100*iou[:17].mean()))
            occupied_intersection = matrix[:17,:17].sum()
            occupied_union = matrix.sum()-matrix[17,17]
            occupancy.append(float(100*occupied_intersection/occupied_union))
        selected = [1,3,5] if len(per_time)==6 else [0]
        return {'samples':self.samples,'class_confusion_counts':counts.tolist(),
                'miou_percent_per_0p5s':per_time,'iou_percent_per_0p5s':occupancy,
                'class_iou_percent_per_time':classes,
                'miou_percent_at_1_2_3s':[per_time[i] for i in selected],
                'iou_percent_at_1_2_3s':[occupancy[i] for i in selected],
                'miou_percent_mean_1_2_3s':float(np.mean([per_time[i] for i in selected])),
                'iou_percent_mean_1_2_3s':float(np.mean([occupancy[i] for i in selected])),
                'miou_percent_mean_all_frames':float(np.mean(per_time)),
                'iou_percent_mean_all_frames':float(np.mean(occupancy))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    parser.add_argument('--no-ema',action='store_true')
    parser.add_argument('--save-predictions',action='store_true')
    parser.add_argument('--seed',type=int,default=1000)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    entry=next(e for e in json.loads((ROOT/'manifest.json').read_text())['checkpoints'] if e['id']==args.checkpoint)
    resource_hash=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
    run_lock=(args.output/'evaluation.lock').open('a')
    fcntl.flock(run_lock,fcntl.LOCK_EX)
    existing=args.output/'result.json'
    if args.limit is None and existing.exists():
        cached=json.loads(existing.read_text())
        expected_samples=4219 if entry['kind'].startswith('flow') else 6019 if entry['kind']=='vae2d' else 5269
        if (cached.get('status')=='completed' and cached.get('validation_scope')=='full'
                and cached.get('resource_manifest_sha256')==resource_hash
                and cached.get('checkpoint',{}).get('sha256')==entry['sha256']
                and cached.get('precision')==args.precision and cached.get('seed')==args.seed
                and cached.get('batch_size')==args.batch_size and cached.get('metrics',{}).get('samples')==expected_samples
                and not entry['kind'].startswith('flow')):
            print('Reused complete verified VAE evaluation.',flush=True)
            return
    try:
        device=torch.device('cuda:0')
        torch.cuda.set_device(device)
        torch.set_num_threads(4)
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        model,metadata=make_model(entry,device,not args.no_ema)
        dataset=ValidationData(entry['kind'],args.limit)
        loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers,
                          pin_memory=True,shuffle=False,persistent_workers=args.workers>0)
        count=Counts(6 if entry['kind']!='vae2d' else 1,device)
        noise_generator=torch.Generator(device=device).manual_seed(args.seed)
        predictions=None
        if args.save_predictions:
            predictions=np.lib.format.open_memmap(args.output/'predictions.npy',mode='w+',dtype=np.uint8,
                shape=(len(dataset),6 if entry['kind']!='vae2d' else 1,200,200,16))
        start=time.monotonic();last_log=start;gpu_seconds=0.
        atomic_json(args.output/'status.json',{'status':'running','checkpoint':entry['id'],'completed':0,'total':len(dataset),'pid':os.getpid()})
        with torch.inference_mode():
            for batch in loader:
                target=batch['target'].to(device,non_blocking=True)
                history=batch['history'].to(device,non_blocking=True) if entry['kind'].startswith('flow') else None
                trajectory=batch['trajectory'].to(device,non_blocking=True) if history is not None else None
                event_start,event_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                event_start.record()
                with torch.autocast('cuda',enabled=args.precision=='bf16',dtype=torch.bfloat16):
                    prediction=(predict_flow(model,history,trajectory,noise_generator) if history is not None else
                                predict_vae(model,target,entry['kind']))
                event_end.record();event_end.synchronize()
                gpu_seconds+=event_start.elapsed_time(event_end)/1000.
                count.update(prediction,target)
                if predictions is not None:
                    predictions[batch['index'].numpy()]=prediction.cpu().numpy().astype(np.uint8)
                elapsed=time.monotonic()-start
                if time.monotonic()-last_log>=10 or count.samples==len(dataset):
                    progress={'status':'running','checkpoint':entry['id'],'completed':count.samples,'total':len(dataset),
                              'elapsed_seconds':elapsed,'samples_per_second':count.samples/elapsed,
                              'estimated_remaining_seconds':elapsed*(len(dataset)-count.samples)/count.samples,
                              'gpu_memory_peak_bytes':torch.cuda.max_memory_allocated(),'pid':os.getpid()}
                    atomic_json(args.output/'status.json',progress)
                    print(json.dumps(progress),flush=True);last_log=time.monotonic()
        if predictions is not None:predictions.flush()
        result={'status':'completed','checkpoint':entry,'model':metadata,'precision':args.precision,
                'resource_manifest_sha256':resource_hash,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'batch_size':args.batch_size,'seed':args.seed,'hardware':torch.cuda.get_device_name(),
                'torch':torch.__version__,'validation_scope':'full' if args.limit is None else 'pilot',
                'elapsed_seconds':time.monotonic()-start,'gpu_forward_seconds':gpu_seconds,
                'gpu_forward_fps':count.samples*(6 if entry['kind']!='vae2d' else 1)/gpu_seconds,
                'input_protocol':'released configuration; all-voxel 17-class mIoU, empty=17; EMA for flow; historical trajectory retains published next-step displacement',
                'metrics':count.compute()}
        atomic_json(args.output/'result.json',result)
        atomic_json(args.output/'status.json',{'status':'completed','checkpoint':entry['id'],'completed':len(dataset),'total':len(dataset)})
        print('RESULT',json.dumps({'checkpoint':entry['id'],'metrics':{k:v for k,v in result['metrics'].items() if 'counts' not in k and 'class_' not in k},'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    except Exception:
        atomic_json(args.output/'status.json',{'status':'failed','checkpoint':entry['id'],'error':traceback.format_exc()})
        raise


if __name__=='__main__':
    main()

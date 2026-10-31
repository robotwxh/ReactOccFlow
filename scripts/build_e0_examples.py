"""Encode a small train-only E0 acceptance cache, not a full research dataset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from reactoccflow.data.research import NuScenesLabelIndex
from reactoccflow.data.e0_cache import encode_e0_training_sample
from reactoccflow.models import OccupancyCodec
from reactoccflow.provenance import sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples',type=Path,default=Path('outputs/research_qa/manifest.json'))
    parser.add_argument('--split',type=Path,default=Path('data/research/splits.json'))
    parser.add_argument('--root',type=Path,default=Path('/home/lthpc/data/nuscenes'))
    parser.add_argument('--checkpoint',type=Path,default=Path('assets/checkpoints/occfm_vae.ckpt'))
    parser.add_argument('--output',type=Path,default=Path('data/research/e0_acceptance'))
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(4)
    examples=json.loads(args.examples.read_text())
    split=json.loads(args.split.read_text())
    if examples.get('group')!='train': raise ValueError('Acceptance encodings use only training scenes')
    if any(e['scene'] not in split['groups']['train'] for e in examples['examples']):
        raise ValueError('Example scene is not in permitted training group')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    scenes=[e['scene'] for e in examples['examples']]
    index=NuScenesLabelIndex(args.root,scenes=scenes)
    codec=OccupancyCodec().to(args.device).eval();loaded=codec.load_pretrained(args.checkpoint)
    report={'version':1,'format':'reactoccflow_e0_acceptance_cache','oracle':True,'group':'train',
            'not_full_training_cache':True,'split_sha256':sha256_file(args.split),
            'examples_sha256':sha256_file(args.examples),'checkpoint_sha256':sha256_file(args.checkpoint),
            'vae_load':loaded,'device':args.device,'samples':[]}
    started=time.monotonic()
    for example in examples['examples']:
        result=encode_e0_training_sample(index,example['scene'],example['token'],codec,args.device,source_frame='ego')
        arrays={f'{group}.{key}':value for group in ('observation','target') for key,value in result[group].items()}
        for key,value in arrays.items():
            if value.dtype.kind=='O': raise ValueError(f'Object dtype not permitted: {key}')
            if value.dtype.kind=='f' and not np.isfinite(value).all(): raise ValueError(f'Nonfinite output: {key}')
        assert result['observation']['history_latents'].shape==(6,16,50,50)
        assert result['target']['future_latents'].shape==(6,16,50,50)
        assert result['observation']['actor_history'].shape==(8,4,6)
        filename=example['scene']+'.npz'
        np.savez_compressed(args.output/filename,**arrays)
        meta=result['metadata'];meta.update({'file':filename,'sha256':sha256_file(args.output/filename),'qa_category':example['category']})
        report['samples'].append(meta)
        (args.output/'manifest.partial.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'scene':example['scene'],'completed':len(report['samples']),'total':len(examples['examples']),
                          'history_valid':int(arrays['observation.history_mask'].sum()),'future_valid':int(arrays['target.future_latent_mask'].sum())}),flush=True)
    report['seconds']=time.monotonic()-started
    report['sample_count']=len(report['samples']);report['complete']=True
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'manifest.partial.json').unlink()
    print(json.dumps({'complete':True,'samples':len(report['samples']),'seconds':report['seconds']}),flush=True)


if __name__=='__main__': main()

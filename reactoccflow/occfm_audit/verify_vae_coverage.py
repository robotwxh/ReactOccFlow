"""Isolate the 150-frame coverage difference without rerunning all 6019 frames.

Skip the same posterior Gaussian draws for other frames, infer only the missing
frames, and subtract their exact confusion counts from the completed full run.
The RNG-skip equivalence is checked against actual forward passes in each dtype.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .data import ROOT
from .evaluate import Counts,make_model,predict_vae


def metrics(matrix):
    seen,positive=matrix.sum(1),matrix.sum(0)
    correct=np.diag(matrix);union=seen+positive-correct
    iou=np.divide(correct,union,out=np.ones(18,dtype=float),where=union>0);iou[seen==0]=1.
    return {'miou_percent':float(100*iou[:17].mean()),
            'iou_percent':float(100*matrix[:17,:17].sum()/(matrix.sum()-matrix[17,17])),
            'class_iou_percent':(100*iou[:17]).tolist()}


def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.manual_seed(1000);torch.cuda.manual_seed_all(1000)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    entry=next(e for e in manifest['checkpoints'] if e['id']=='vae2d_e100')
    model,_=make_model(entry,device,False)
    occupancy=np.load(ROOT/'occupancy.npy',mmap_mode='r')
    retained=set(manifest['latent_gt_indices']);missing=set(range(len(occupancy)))-retained
    assert len(missing)==150
    frames=manifest['frames']
    assert all(i==len(frames)-1 or frames[i+1][0]!=frames[i][0] for i in missing)
    output=Path(__file__).resolve().parents[2]/'outputs/audits/OccFM_weights_2026-11-02'
    results=[]
    with torch.inference_mode():
        for precision,result_file in [('bf16','full/vae2d_e100/result.json'),('fp32','diagnostics/vae2d_e100_fp32/result.json')]:
            dtype=torch.bfloat16 if precision=='bf16' else torch.float32
            def skip():torch.randn((1,16,50,50),device=device,dtype=dtype)
            def predict(index):
                target=torch.from_numpy(np.array(occupancy[index:index+1],copy=True)).to(device).unsqueeze(1)
                with torch.autocast('cuda',enabled=precision=='bf16',dtype=torch.bfloat16):
                    pred=predict_vae(model,target,'vae2d')
                return pred,target
            torch.cuda.manual_seed(1000)
            predict(0);predict(1);direct,_=predict(2)
            torch.cuda.manual_seed(1000);skip();skip();skipped,_=predict(2)
            assert torch.equal(direct,skipped),'Posterior draw skip changed predictions'
            torch.cuda.manual_seed(1000)
            excluded=Counts(1,device)
            for index in range(len(occupancy)):
                if index in missing:
                    pred,target=predict(index);excluded.update(pred,target)
                else:skip()
            original=json.loads((output/result_file).read_text())
            all_matrix=np.asarray(original['metrics']['class_confusion_counts'][0],dtype=np.int64)
            missing_matrix=excluded.matrix.cpu().numpy()[0]
            subset_matrix=all_matrix-missing_matrix
            assert np.all(subset_matrix>=0) and subset_matrix.sum()==len(retained)*200*200*16
            results.append({'precision':precision,'rng_skip_bitwise_verification':True,
                            'full_frames':len(occupancy),'retained_cache_frames':len(retained),'excluded_frames':len(missing),
                            'full':metrics(all_matrix),'cache_covered_subset':metrics(subset_matrix),
                            'excluded_scene_terminal_frames':metrics(missing_matrix),
                            'subset_confusion_counts':subset_matrix.tolist()})
            print(json.dumps({k:v for k,v in results[-1].items() if k!='subset_confusion_counts'}),flush=True)
    result={'status':'completed','method':'Use identical per-frame posterior random stream; subtract terminal-frame confusion counts from full 6019-frame results. This isolates coverage and is not a claim about the authors original VAE evaluation set.','results':results}
    (output/'vae_coverage_diagnostic.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()

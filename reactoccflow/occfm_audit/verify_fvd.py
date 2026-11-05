"""Validate reusable FVD features against the direct pred-then-GT sequence."""
from pathlib import Path
import json
import numpy as np
import torch

from .data import ROOT,ValidationData
from .evaluate import make_model
from .fvd import encode,frechet


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    base=Path(__file__).resolve().parents[2]
    output=base/'outputs/audits/OccFM_weights_2026-11-02'
    entry=next(e for e in json.loads((ROOT/'manifest.json').read_text())['checkpoints'] if e['id']=='vae3d_e040')
    torch.manual_seed(1000);torch.cuda.manual_seed_all(1000)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    model,_=make_model(entry,device,False)
    prediction=np.load(output/'full/flow_hist_e199/predictions.npy',mmap_mode='r')
    gt=ValidationData('flow_hist',1)[0]['target']
    with torch.inference_mode():
        first=encode(model,torch.from_numpy(np.array(prediction[:1],copy=True)).to(device))[0].cpu().numpy()
        second=encode(model,torch.from_numpy(gt[None]).to(device))[0].cpu().numpy()
    cached_pred=np.load(output/'pilot/fvd_pred_fp32/features.npy',mmap_mode='r')[0]
    cached_gt=np.load(output/'pilot/fvd_gt_fp32/features.npy',mmap_mode='r')[0]
    assert np.array_equal(first,cached_pred),'Prediction posterior noise sequence changed'
    assert np.array_equal(second,cached_gt),'GT posterior noise sequence changed'
    x=np.random.default_rng(55).normal(size=(20,7))
    delta=np.arange(1,8)*0.1
    distance=frechet(x,x+delta)
    assert np.isclose(distance,np.sum(delta*delta),atol=1e-10)
    assert frechet(x,x)<1e-10
    result={'status':'passed','cached_and_direct_pred_features_bitwise_equal':True,
            'cached_and_direct_gt_features_bitwise_equal':True,'frechet_known_translation':distance,
            'expected_translation_distance':float(np.sum(delta*delta))}
    (output/'fvd_implementation_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()

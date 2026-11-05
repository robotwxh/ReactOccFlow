"""Compare the isolated wrapper with the copied original cfm_eval implementation."""
from pathlib import Path
import json
import torch

from .data import ROOT,ValidationData
from .evaluate import make_model,predict_flow


def main():
    torch.set_num_threads(4)
    torch.manual_seed(1000);torch.cuda.manual_seed_all(1000)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    entry=next(e for e in json.loads((ROOT/'manifest.json').read_text())['checkpoints'] if e['id']=='flow_hist_e199')
    model,_=make_model(entry,device)
    item=ValidationData('flow_hist',1)[0]
    history=torch.from_numpy(item['history'][None]).to(device)
    trajectory=torch.from_numpy(item['trajectory'][None]).to(device)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        generator=torch.Generator(device=device).manual_seed(1000)
        adapted=predict_flow(model,history,trajectory,generator)
        torch.cuda.manual_seed(1000)
        # Original cfm_eval only needs future_clip for shape and a diagnostic MSE.
        # Supply zeros, so no future occupancy/latent target reaches inference.
        output,_=model.cfm_eval(torch.zeros_like(history),history*model.rescale_factor,{'trajectory':trajectory},1)
        original=output[2]['pred_occ']
    agreement=float((adapted==original).float().mean())
    assert torch.equal(adapted,original),agreement
    result={'status':'passed','prediction_voxels':adapted.numel(),'voxel_agreement':agreement,
            'adapted_wrapper_bitwise_equal_to_copied_original_cfm_eval':True,
            'future_target_supplied_to_oracle':'zeros only; no real future target'}
    destination=Path(__file__).resolve().parents[2]/'outputs/audits/OccFM_weights_2026-11-02/model_wrapper_verification.json'
    destination.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()

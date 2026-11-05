"""Check scorer against independent NumPy aggregation and protect reference sources."""
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch

from reactoccflow.evaluation.occupancy import OccupancyAccumulator
from .evaluate import Counts, predict_flow


def main():
    base=Path(__file__).resolve().parents[2]
    rng=np.random.default_rng(901)
    truth=rng.integers(0,18,size=(3,6,3,4,5),dtype=np.uint8)
    prediction=rng.integers(0,18,size=truth.shape,dtype=np.uint8)
    prediction[0,:2]=truth[0,:2]
    counter=Counts(6,'cpu')
    counter.update(torch.from_numpy(prediction),torch.from_numpy(truth))
    result=counter.compute()
    independent=OccupancyAccumulator()
    for p,t in zip(prediction,truth):independent.update(p,t)
    expected=independent.compute()['legacy']['per_time_mean_iou']
    assert np.allclose(result['miou_percent_per_0p5s'],np.asarray(expected)*100,atol=1e-12)
    for i in range(6):
        p,t=prediction[:,i]!=17,truth[:,i]!=17
        iou=100*np.logical_and(p,t).sum()/np.logical_or(p,t).sum()
        assert np.isclose(iou,result['iou_percent_per_0p5s'][i],atol=1e-12)
    # Chunking must preserve dataset-level aggregation, including absent classes.
    other=Counts(6,'cpu')
    for i in range(3):other.update(torch.from_numpy(prediction[i:i+1]),torch.from_numpy(truth[i:i+1]))
    assert torch.equal(counter.matrix,other.matrix)
    sources=json.loads((base/'docs/provenance/occfm_audit_sources.json').read_text())
    assert all(hashlib.sha256(Path(item['source']).read_bytes()).hexdigest()==item['source_sha256'] for item in sources['files'])
    assert 'target' not in inspect.signature(predict_flow).parameters
    imported=[]
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if name=='forecast' or name.startswith('forecast.'):
            raise AssertionError('Reference namespace imported')
        if path and '/Paper/OccFM/' in str(Path(path).resolve()):
            raise AssertionError(f'Reference source executed: {path}')
        if name.startswith('reactoccflow.occfm_audit.vendor'):
            imported.append({'module':name,'file':path})
    out={'status':'passed','scorer_matches_independent_numpy':True,'streaming_counts_match':True,
         'reference_source_files_unchanged':len(sources['files']),
         'forecast_prediction_accepts_no_future_occupancy_target':True,'isolated_imports':imported}
    destination=base/'outputs/audits/OccFM_weights_2026-11-02/implementation_verification.json'
    destination.write_text(json.dumps(out,indent=2))
    print('Scorer and isolated execution verification passed; protected source hashes unchanged.')


if __name__=='__main__':main()

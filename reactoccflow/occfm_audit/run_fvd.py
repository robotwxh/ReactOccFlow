"""Wait for full checkpoint evaluation, then compute corrected FVD for all forecasts."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .evaluate import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.full=args.full.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output/'controller.json',{'status':'waiting_for_forecast_results','pid':os.getpid()})
    # This stage is read-only with respect to the forecast outputs.
    while not (args.full/'completion.json').exists():time.sleep(15)
    completion=json.loads((args.full/'completion.json').read_text())
    if completion['status']!='completed':
        atomic_json(args.output/'controller.json',{'status':'blocked_by_failed_evaluation','evaluation':completion})
        return
    def run(mode,out,gpu,predictions=None):
        out.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-B','-m','reactoccflow.occfm_audit.fvd','--mode',mode,'--output',str(out)]
        if predictions is not None:command+=['--predictions',str(predictions),'--gt-features',str(args.output/'gt/features.npy')]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONDONTWRITEBYTECODE':'1',
             'OPENBLAS_NUM_THREADS':'8','OMP_NUM_THREADS':'4'}
        with (out/'run.log').open('w') as handle:
            process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
            try:code=process.wait(timeout=7200)
            except subprocess.TimeoutExpired:process.kill();process.wait();code=-9
        if code!=0:raise RuntimeError(f'FVD {mode} process failed at {out}: {code}')
    atomic_json(args.output/'controller.json',{'status':'computing_ground_truth_features','pid':os.getpid()})
    run('gt',args.output/'gt',1)
    groups={1:['flow_hist_e199','flow_hist_e171','flow_hist_e137'],
            0:['flow_future_e196','flow_future_e107','flow_future_e098']}
    atomic_json(args.output/'controller.json',{'status':'computing_prediction_fvd','pid':os.getpid()})
    def group(item):
        gpu,identifiers=item
        for identifier in identifiers:
            run('pred',args.output/identifier,gpu,args.full/identifier/'predictions.npy')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(group,groups.items()))
    atomic_json(args.output/'controller.json',{'status':'completed','pid':os.getpid()})


if __name__=='__main__':main()

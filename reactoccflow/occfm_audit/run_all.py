"""Run every published checkpoint once on two GPUs; retain failures without retries."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import ROOT
from .evaluate import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout-per-checkpoint',type=int,default=7200)
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    queues={1:['flow_hist_e199','flow_hist_e171','flow_hist_e137','vae3d_e040'],
            0:['vae2d_e100','flow_future_e196','flow_future_e107','flow_future_e098','vae3d_e039','vae3d_e038']}
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert {e['id'] for e in manifest['checkpoints']} == {e for q in queues.values() for e in q}
    def queue(gpu,checkpoints):
        completed=[]
        for identifier in checkpoints:
            target=args.output/identifier
            target.mkdir(exist_ok=True)
            command=[sys.executable,'-B','-m','reactoccflow.occfm_audit.evaluate','--checkpoint',identifier,
                     '--output',str(target),'--workers','4']
            if identifier.startswith('flow'):command+=['--save-predictions']
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONDONTWRITEBYTECODE':'1'}
            start=time.monotonic()
            with (target/'run.log').open('w') as handle:
                process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
                atomic_json(args.output/f'gpu_{gpu}.json',{'status':'running','current':identifier,'pid':process.pid,'completed':completed,'command':command})
                try:
                    code=process.wait(timeout=args.timeout_per_checkpoint)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait();code=-9
                    atomic_json(target/'status.json',{'status':'timeout','checkpoint':identifier,'timeout_seconds':args.timeout_per_checkpoint})
            completed.append({'checkpoint':identifier,'exit_code':code,'seconds':time.monotonic()-start})
            print(json.dumps({'gpu':gpu,**completed[-1]}),flush=True)
        atomic_json(args.output/f'gpu_{gpu}.json',{'status':'completed','completed':completed})
        return completed
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda pair:queue(*pair),queues.items()))
    atomic_json(args.output/'completion.json',{'status':'completed' if all(r['exit_code']==0 for g in results for r in g) else 'completed_with_failures','runs':[r for g in results for r in g]})


if __name__=='__main__':
    main()

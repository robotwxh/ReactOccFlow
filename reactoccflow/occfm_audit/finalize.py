"""Finalize comparison artifacts after every requested full run has completed."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback


def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def finite(value):
    if isinstance(value,float):assert math.isfinite(value)
    elif isinstance(value,dict):
        for child in value.values():finite(child)
    elif isinstance(value,list):
        for child in value:finite(child)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();root=args.root.resolve();base=Path(__file__).resolve().parents[2]
    ledger_script='/home/lthpc/.codex/skills/academic-research-suite/ars/scripts/run_ledger.py'
    started=time.monotonic()
    try:
        while time.monotonic()-started<28800:
            controller=root/'fvd/controller.json'
            if controller.exists() and json.loads(controller.read_text()).get('status')=='completed':break
            if controller.exists() and json.loads(controller.read_text()).get('status')=='blocked_by_failed_evaluation':
                raise RuntimeError('A main evaluation failed; inspect full/completion.json')
            time.sleep(15)
        else:raise TimeoutError('Finalization wait exceeded eight hours')
        completion=json.loads((root/'full/completion.json').read_text())
        assert completion['status']=='completed' and len(completion['runs'])==10
        for path in (root/'full').glob('*/result.json'):
            result=json.loads(path.read_text());finite(result)
            kind=result['checkpoint']['kind']
            expected=4219 if kind.startswith('flow') else 6019 if kind=='vae2d' else 5269
            assert result['metrics']['samples']==expected and result['model']['strict_load']
        for path in (root/'fvd').glob('flow_*/result.json'):
            result=json.loads(path.read_text());finite(result)
            assert result['samples']==4219 and result['validation_scope']=='full'
        sources=json.loads((base/'docs/provenance/occfm_audit_sources.json').read_text())
        unchanged=[]
        for record in sources['files']:
            assert hash_file(Path(record['source']))==record['source_sha256']
            assert hash_file(base/record['destination'])==record['destination_sha256']
            unchanged.append(record['source'])
        (root/'final_source_verification.json').write_text(json.dumps({'status':'passed','protected_reference_source_files_unchanged':len(unchanged),'copied_model_source_hashes_unchanged':len(unchanged)},indent=2))
        resources=base/'resources/occfm_audit'
        manifest=json.loads((resources/'manifest.json').read_text())
        resource_hashes={}
        for entry in manifest['checkpoints']:
            path=resources/entry['checkpoint'];actual=hash_file(path)
            assert actual==entry['sha256'];resource_hashes[str(path.relative_to(resources))]=actual
        for name in ('occupancy.npy','latents.npy','trajectories.npy','manifest.json'):
            resource_hashes[name]=hash_file(resources/name)
        (root/'evaluated_resource_hashes.json').write_text(json.dumps(resource_hashes,indent=2))
        subprocess.run([sys.executable,'-B','-m','reactoccflow.occfm_audit.write_report','--root',str(root)],check=True)
        def ledger(command,**paths):
            cmd=[sys.executable,'-B',ledger_script,command,'--passport-path',str(root/'material_passport.json')]
            for key,value in paths.items():cmd+=['--'+key.replace('_','-'),str(value)]
            result=subprocess.run(cmd,capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stdout+result.stderr)
            return json.loads(result.stdout) if result.stdout.strip() else {}
        trusted=ledger('show')['trusted_entries']
        recorded={entry['step'] for entry in trusted if entry['kind']=='tool_receipt'}
        def receipt(step,command,paths):
            if step in recorded:return
            entry={'kind':'tool_receipt','step':step,'command':command,'status':'passed','exit_status':0,'retries_used':0,'input_paths':[str(path) for path in paths]}
            file=root/f'ledger_entry_{step}.json';file.write_text(json.dumps(entry,indent=2))
            ledger('append',entry_file=file);recorded.add(step)
        for path in sorted((root/'full').glob('*/result.json')):
            identifier=path.parent.name
            step='full_vae2d_bf16' if identifier=='vae2d_e100' else 'full_'+identifier
            receipt(step,'python -B -m reactoccflow.occfm_audit.evaluate --checkpoint '+identifier,[path])
        for path in sorted((root/'fvd').glob('flow_*/result.json')):
            receipt('fvd_'+path.parent.name,'python -B -m reactoccflow.occfm_audit.fvd --mode pred',[path])
        receipt('fvd_ground_truth','python -B -m reactoccflow.occfm_audit.fvd --mode gt',[root/'fvd/gt/result.json'])
        receipt('final_source_protection','SHA-256 check of protected source files, copied model sources and evaluated resources',[root/'final_source_verification.json',root/'evaluated_resource_hashes.json'])
        receipt('comparison_report','python -B -m reactoccflow.occfm_audit.write_report',[root/'comparison.json',root/'权重评估与论文对照.md'])
        claims=root/'ledger_claims.json'
        claims.write_text(json.dumps({'steps':[{'step':step,'status':'passed'} for step in sorted(recorded)]},indent=2))
        checked=ledger('report',claims=claims)
        (root/'final_ledger_check.json').write_text(json.dumps(checked,indent=2))
        passport=root/'material_passport.json';p=json.loads(passport.read_text())
        p.update({'execution_status':'completed','analysis_status':'ANALYZED','full_checkpoints_completed':10,'corrected_fvd_completed':6,'verification_scope':'Measured inference and corrected FVD under recorded protocol; not authentication of every paper claim.'})
        passport.write_text(json.dumps(p,ensure_ascii=False,indent=2))
        state=root/'execution_state.json';p=json.loads(state.read_text());p['status']='completed';p['final_report']=str(root/'权重评估与论文对照.md');state.write_text(json.dumps(p,ensure_ascii=False,indent=2))
        (root/'FINAL_COMPLETION.json').write_text(json.dumps({'status':'completed','full_checkpoints':10,'corrected_fvd':6,'report':str(root/'权重评估与论文对照.md'),'source_protection':'passed','ledger':'ok'},ensure_ascii=False,indent=2))
        print('FINAL REPORT READY',root/'权重评估与论文对照.md',flush=True)
    except Exception:
        (root/'FINALIZATION_ERROR.json').write_text(json.dumps({'status':'failed','traceback':traceback.format_exc()},indent=2))
        raise


if __name__=='__main__':main()

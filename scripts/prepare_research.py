"""Build explicit split-scoped oracle labels from read-only official tables."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from reactoccflow.data.research import NuScenesLabelIndex, build_research_labels
from reactoccflow.data.splits import grouped_scene_split, fit_observation_normalization
from reactoccflow.provenance import sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/home/lthpc/data/nuscenes'))
    parser.add_argument('--output',type=Path,default=Path('data/research'))
    parser.add_argument('--official',type=Path,default=Path('docs/provenance/nuscenes_official_splits.json'))
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    start=time.monotonic()
    official=json.loads(args.official.read_text())
    split=grouped_scene_split(official['train'],official['val'],seed=args.seed)
    split['official_source_sha256']=sha256_file(args.official)
    split['metadata_version']='v1.0-trainval'
    split['pretraining_caveat']='Copied OccFM pretrained CFM/VAE used official700 training scenes. The70/70 groups are NOT unseen relative to these checkpoints. Strict held-out calibration claims require model refitting without tune/calibration scenes and separate evaluation.'
    scene_table=json.loads((args.root/'v1.0-trainval/scene.json').read_text())
    available={row['name'] for row in scene_table}
    expected=set(official['train'])|set(official['val'])
    if available!=expected: raise ValueError('Official split vs local metadata scene census differs')
    args.output.mkdir(parents=True,exist_ok=True)
    split_path=args.output/'splits.json'
    if split_path.exists() and json.loads(split_path.read_text())!=split:
        raise ValueError('Existing split manifest differs; choose another output')
    split_path.write_text(json.dumps(split,indent=2)+'\n')
    print(json.dumps({'stage':'index','groups':{k:len(v) for k,v in split['groups'].items()}}),flush=True)
    index=NuScenesLabelIndex(args.root)
    summary={'version':1,'source_root':str(args.root.resolve()),'oracle':True,'split_sha256':sha256_file(split_path),'groups':{}}
    for group,scenes in split['groups'].items():
        output=args.output/group
        print(json.dumps({'stage':'labels','group':group,'scenes':len(scenes)}),flush=True)
        if output.exists():
            manifest=json.loads((output/'manifest.json').read_text())
            if manifest.get('split_manifest_sha256')!=summary['split_sha256'] or manifest.get('group')!=group:
                raise ValueError(f'Cannot resume labels with unmatched split: {group}')
        else:
            manifest=build_research_labels(index,output,scenes)
            manifest.update({'group':group,'split_manifest_sha256':summary['split_sha256'],
                             'occupancy_source_frame':'ego','local_frame_evidence':'docs/implementation/research_label_design.md',
                             'source_scene_sha256':sha256_file(args.root/'v1.0-trainval/scene.json'),
                             'source_sample_sha256':sha256_file(args.root/'v1.0-trainval/sample.json')})
            for shard in manifest['shards']: shard['sha256']=sha256_file(output/shard['file'])
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        missing=[frame['token'] for frame in manifest['frames'] if not (args.root/'gts'/frame['scene']/frame['token']/'labels.npz').is_file()]
        if missing: raise FileNotFoundError(f'{group}: {len(missing)} missing occupancy frames; first={missing[0]}')
        summary['groups'][group]={'scenes':len(scenes),'frames':manifest['num_frames'],'manifest_sha256':sha256_file(output/'manifest.json'),'missing_occupancy':0}
        print(json.dumps({'stage':'group_complete','group':group,**summary['groups'][group]}),flush=True)
    stats=fit_observation_normalization(args.output/'train',split)
    (args.output/'normalization.json').write_text(json.dumps(stats,indent=2)+'\n')
    summary.update({'seconds':time.monotonic()-start,'normalization_sha256':sha256_file(args.output/'normalization.json'),
                    'pretraining_caveat':split['pretraining_caveat']})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'stage':'complete',**summary}),flush=True)


if __name__=='__main__': main()

"""Select offline label QA examples and render20 distinct training scenes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('data/research/train'))
    parser.add_argument('--root',type=Path,default=Path('/home/lthpc/data/nuscenes'))
    parser.add_argument('--output',type=Path,default=Path('outputs/research_qa'))
    parser.add_argument('--count',type=int,default=20)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from reactoccflow.visualization import render_label_panel
    manifest=json.loads((args.cache/'manifest.json').read_text())
    if manifest.get('group')!='train': raise ValueError('QA cache must be explicit training group')
    if args.count<20: raise ValueError('Plan acceptance requires at least20 distinct scenes')
    # Future targets may classify offline QA examples, never model actor selection.
    pool={key:[] for key in ('no_actor','pedestrian','occluded','turn','straight')}
    for shard in manifest['shards']:
        with np.load(args.cache/shard['file'],allow_pickle=False) as data:
            n=len(data['observation.actor_history'])
            for i in range(n):
                current=data['observation.actor_observed_mask'][i,:,-1]
                count=int(current.sum())
                future=data['target.ego_target_mask'][i]
                end=data['target.ego_future'][i,-1]
                classes=data['observation.actor_class'][i]
                visibility=data['observation.actor_visibility'][i]
                categories=[]
                if count==0: categories.append('no_actor')
                if ((classes==7)&current).any(): categories.append('pedestrian')
                if ((visibility<=2)&current).any(): categories.append('occluded')
                if future.all() and abs(end[1])>2 and abs(np.arctan2(end[2],end[3]))>.15: categories.append('turn')
                if future.all() and abs(end[1])<.5 and abs(end[0])>6: categories.append('straight')
                for category in categories: pool[category].append((shard['scene'],i,shard['file']))
    selected=[];seen=set()
    for category in pool:
        for scene,row,file in pool[category]:
            if scene not in seen:
                selected.append({'scene':scene,'row':row,'file':file,'category':category});seen.add(scene);break
    for shard in manifest['shards']:
        if len(selected)>=args.count: break
        if shard['scene'] not in seen:
            selected.append({'scene':shard['scene'],'row':min(10,shard['count']-1),'file':shard['file'],'category':'additional'});seen.add(shard['scene'])
    if len(selected)!=args.count: raise ValueError('Not enough distinct QA scenes')
    frames={(frame['scene'],frame['row']):frame for frame in manifest['frames']}
    args.output.mkdir(parents=True,exist_ok=True)
    rows=(args.count+4)//5
    figure,axes=plt.subplots(rows,5,figsize=(22,4.5*rows),squeeze=False)
    for selected_frame,axis in zip(selected,axes.flat):
        frame=frames[(selected_frame['scene'],selected_frame['row'])]; selected_frame['token']=frame['token']
        row=selected_frame['row']
        with np.load(args.cache/selected_frame['file'],allow_pickle=False) as data:
            obs={k.split('.',1)[1]:data[k][row] for k in data.files if k.startswith('observation.')}
            target={k.split('.',1)[1]:data[k][row] for k in data.files if k.startswith('target.')}
        with np.load(args.root/'gts'/frame['scene']/frame['token']/'labels.npz',allow_pickle=False) as label:
            semantics=label['semantics'].copy()
        title=f'{frame["scene"]} / {selected_frame["category"]}'
        render_label_panel(axis,semantics,obs,target,title=title)
        single,single_axis=plt.subplots(figsize=(8,8))
        render_label_panel(single_axis,semantics,obs,target,title=title+' / oracle labels')
        single.tight_layout();single.savefig(args.output/f'{frame["scene"]}.png',dpi=140);plt.close(single)
        selected_frame['observed_actors']=int(obs['actor_observed_mask'][:,-1].sum())
        selected_frame['current_classes']=obs['actor_class'][obs['actor_observed_mask'][:,-1]].tolist()
    for unused in list(axes.flat)[len(selected):]: unused.set_visible(False)
    figure.suptitle('Oracle E0 label QA: red current boxes; yellow history; cyan actor targets; orange ego targets',fontsize=15)
    figure.tight_layout(rect=(0,0,1,.97));figure.savefig(args.output/'overview.png',dpi=110);plt.close(figure)
    report={'oracle':True,'group':'train','scene_count':len(selected),
            'category_coverage':{k:sum(s['category']==k for s in selected) for k in pool},
            'selection_usage':'Offline QA only; future labels never select model actor slots.',
            'manual_inspection':'pending','examples':selected}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='examples'},indent=2))


if __name__=='__main__': main()

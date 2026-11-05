"""Build a reviewable comparison from completed full evaluations only."""
import argparse
import csv
import json
from pathlib import Path


PAPER={'flow_hist':{'miou':[33.17,21.09,15.64],'iou':[40.53,30.37,24.44],'miou_avg':23.33,'iou_avg':31.78,'fvd':7.80},
       'flow_future':{'miou':[36.42,27.39,21.66],'iou':[43.68,36.89,31.98],'miou_avg':28.49,'iou_avg':37.52,'fvd':7.68},
       'vae2d':{'miou':[92.8],'iou':[85.8]}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();rows=[]
    criteria=json.loads((args.root/'criteria.json').read_text())
    threshold=criteria['relative_error_threshold']
    for path in sorted((args.root/'full').glob('*/result.json')):
        result=json.loads(path.read_text())
        if result.get('status')!='completed' or result['validation_scope']!='full':continue
        entry=result['checkpoint'];m=result['metrics'];kind=entry['kind'];paper=PAPER.get(kind)
        row={'checkpoint':entry['id'],'kind':kind,'samples':m['samples'],'precision':result['precision'],
             'weight_state':result['model']['state'],'actual_internal_epoch':result['model']['checkpoint_internal_epoch'],
             'miou_percent':m['miou_percent_at_1_2_3s'],'iou_percent':m['iou_percent_at_1_2_3s'],
             'miou_average_percent':m['miou_percent_mean_1_2_3s'],'iou_average_percent':m['iou_percent_mean_1_2_3s'],
             'miou_all_six_frames_percent':m['miou_percent_mean_all_frames'],
             'paper':paper,'comparison_available':paper is not None,
             'source_result':str(path.relative_to(args.root))}
        if kind=='vae3d':
            row['miou_average_percent']=m['miou_percent_mean_all_frames']
            row['iou_average_percent']=m['iou_percent_mean_all_frames']
            row['average_definition']='mean of all six reconstruction frames'
        else:
            row['average_definition']='mean at 1s, 2s, 3s' if kind.startswith('flow') else 'single-frame reconstruction'
        if paper:
            row['miou_delta_percentage_points']=[a-b for a,b in zip(row['miou_percent'],paper['miou'])]
            row['iou_delta_percentage_points']=[a-b for a,b in zip(row['iou_percent'],paper['iou'])]
            for metric in ['miou','iou']:
                errors=[abs(a-b)/abs(b) for a,b in zip(row[metric+'_percent'],paper[metric])]
                row[metric+'_relative_error_percent']=[100*x for x in errors]
                row[metric+'_within_2pct_per_horizon']=[x<threshold for x in errors]
                target=paper.get(metric+'_avg',paper[metric][0])
                error=abs(row[metric+'_average_percent']-target)/abs(target)
                row[metric+'_average_relative_error_percent']=100*error
                row[metric+'_average_within_2pct']=error<threshold
        fvd_path=args.root/'fvd'/entry['id']/'result.json'
        if fvd_path.exists():
            fvd=json.loads(fvd_path.read_text());row['fvd_in_table_units']=fvd['fvd_in_table3_units_x1e_minus3']
            row['fvd_delta']=row['fvd_in_table_units']-paper['fvd']
            row['fvd_relative_error_percent']=100*abs(row['fvd_delta'])/paper['fvd']
            row['fvd_within_2pct']=row['fvd_relative_error_percent']<100*threshold
        rows.append(row)
    if args.require_complete:
        assert len(rows)==10,'Not all 10 full checkpoint evaluations completed'
        assert all('fvd_in_table_units' in r for r in rows if r['kind'].startswith('flow')),'Missing FVD results'
    (args.root/'comparison.json').write_text(json.dumps({'paper_version':'arXiv:2506.23434v2','criteria':criteria,'rows':rows},ensure_ascii=False,indent=2))
    fields=['checkpoint','kind','samples','precision','weight_state','actual_internal_epoch','average_definition','miou_1s_or_reconstruction','miou_2s','miou_3s','iou_1s_or_reconstruction','iou_2s','iou_3s','miou_average_percent','iou_average_percent','miou_delta_1s','miou_delta_2s','miou_delta_3s','iou_delta_1s','iou_delta_2s','iou_delta_3s','miou_relative_error_1s_percent','miou_relative_error_2s_percent','miou_relative_error_3s_percent','iou_relative_error_1s_percent','iou_relative_error_2s_percent','iou_relative_error_3s_percent','miou_average_relative_error_percent','miou_average_within_2pct','iou_average_relative_error_percent','iou_average_within_2pct','fvd_in_table_units','fvd_delta','fvd_relative_error_percent','fvd_within_2pct']
    with (args.root/'comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:
            output={k:row[k] for k in fields if k in row}
            for name,keys in [('miou_percent',['miou_1s_or_reconstruction','miou_2s','miou_3s']),('iou_percent',['iou_1s_or_reconstruction','iou_2s','iou_3s']),('miou_delta_percentage_points',['miou_delta_1s','miou_delta_2s','miou_delta_3s']),('iou_delta_percentage_points',['iou_delta_1s','iou_delta_2s','iou_delta_3s']),('miou_relative_error_percent',['miou_relative_error_1s_percent','miou_relative_error_2s_percent','miou_relative_error_3s_percent']),('iou_relative_error_percent',['iou_relative_error_1s_percent','iou_relative_error_2s_percent','iou_relative_error_3s_percent'])]:
                for key,value in zip(keys,row.get(name,[])):output[key]=value
            writer.writerow(output)
    print(json.dumps({'completed_weights':len(rows),'corrected_fvd_completed':sum('fvd_in_table_units' in r for r in rows)}))


if __name__=='__main__':main()

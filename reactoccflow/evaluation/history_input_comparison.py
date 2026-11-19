"""Fixed-step EMA export and audited dense/masked-history comparisons.

Reuses the owned architecture_report auditors and count metrics. Reference
searches and reuse decisions: history_input_comparison_sources_20261117.json.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from reactoccflow.data.research import OCC3D_CLASSES
from reactoccflow.evaluation.architecture_report import (
    SCORE_NAMES, read_json, sha256, validate_completed, validate_ground_truth, validate_shared_protocol,
)
from reactoccflow.evaluation.reactflow_generation import checkpoint_model_options
from reactoccflow.models.react_flow import ReactFlow
from reactoccflow.training.engine import create_ema


def export_final_ema(source, destination, normalization, expected_mode):
    """Export exactly step10000, retaining and strictly validating EMA wrappers."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    state = torch.load(source, map_location='cpu', weights_only=True)
    config = state['config']
    if (state['global_step'] != 10000 or config['max_steps'] != 10000
            or config.get('history_visibility') != expected_mode or config.get('backbone') != 'occfm_unet'):
        raise ValueError('Expected the matching U-CFM history mode and exactly10000 optimizer steps')
    model = ReactFlow(normalization=normalization, **checkpoint_model_options(state)).eval()
    model.load_state_dict(state['flow_state_dict'], strict=True)
    ema = create_ema(model)
    ema.load_state_dict(state['ema_state_dict'], strict=True)
    weights = {key: value.detach().cpu().clone() for key, value in ema.module.state_dict().items()}
    if not all(torch.isfinite(value).all() for value in weights.values() if value.is_floating_point()):
        raise ValueError('Nonfinite final EMA')
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'version': 1, 'config': config, 'global_step': 10000, 'ema_state_dict': weights,
                'data_fingerprint': state['data_fingerprint'], 'source_checkpoint_sha256': sha256(source),
                'selection': 'predeclared_final_step10000_EMA;not_selected_on_validation_mIoU'}, destination)
    return {'source': str(source), 'source_sha256': sha256(source), 'file': str(destination),
            'sha256': sha256(destination), 'global_step': 10000, 'history_visibility': expected_mode,
            'config': config, 'strict_load': True}


def validate_training_configs(dense, masked):
    if dense.get('history_visibility') != 'dense_semantic' or masked.get('history_visibility') != 'mask_lidar':
        raise ValueError('Expected dense_semantic versus mask_lidar')
    differences = [key for key in dense.keys() | masked.keys()
                   if key != 'history_visibility'
                   and (key not in dense or key not in masked or dense[key] != masked[key])]
    if differences:
        raise ValueError('Training configurations differ beyond history input: '+', '.join(sorted(differences)))
    return {'matched_training_fields': sorted(key for key in dense if key != 'history_visibility'),
            'differing_field': 'history_visibility'}


def prepare_checkpoints(config):
    torch.set_num_threads(2)
    normalization = read_json(config['normalization'])
    exported = {}
    for mode, values in config['models'].items():
        exported[mode] = export_final_ema(values['latest'], values['evaluation_checkpoint'], normalization, mode)
    audit = validate_training_configs(exported['dense_semantic']['config'], exported['mask_lidar']['config'])
    report = {'status': 'completed', 'checkpoint_step': 10000, 'models': exported, 'training_config_audit': audit}
    Path(config['export_manifest']).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


def build_report(config):
    runs = {mode: validate_completed(values['evaluation_dir'], 10000)
            for mode, values in config['models'].items()}
    dense, masked = runs['dense_semantic'], runs['mask_lidar']
    if (dense['identity'].get('history_visibility') != 'dense_semantic'
            or masked['identity'].get('history_visibility') != 'mask_lidar'):
        raise ValueError('Evaluation input does not match the trained history mode')
    protocol = validate_shared_protocol(dense['identity'], masked['identity'])
    if dense['identity']['model_options'] != masked['identity']['model_options']:
        raise ValueError('World-model architectures differ')
    for run in runs.values():
        if len({window[0] for window in run['identity']['windows']}) != 150:
            raise ValueError('Expected150 official validation scenes')
    baseline = read_json(Path(config['project_root'])/'outputs/audits/OccFM_weights_2026-11-02/full/flow_future_e196/result.json')
    gt = validate_ground_truth(dense['result'], masked['result'], baseline)
    exports = read_json(config['export_manifest'])
    training_audit = validate_training_configs(exports['models']['dense_semantic']['config'],
                                              exports['models']['mask_lidar']['config'])
    for mode, run in runs.items():
        checkpoint = exports['models'][mode]
        if (sha256(checkpoint['file']) != checkpoint['sha256']
                or run['identity']['input_sha256'].get(str(Path(checkpoint['file']).resolve())) != checkpoint['sha256']):
            raise ValueError('Evaluation checkpoint identity differs from fixed-step export')
    pipeline = read_json(config['pipeline_record'])
    for stage in pipeline['stages']:
        if stage['name'] == 'report':
            continue
        if stage['status'] != 'completed' or any(p['exit_code'] != 0 for p in stage['processes']):
            raise ValueError('A required training/evaluation stage did not exit successfully')
    scores = {mode: run['metrics']['paper_frame_full']['standard'] for mode, run in runs.items()}
    per_class = []
    for category, label in enumerate(OCC3D_CLASSES[:17]):
        for horizon, slot in zip((1, 2, 3), (1, 3, 5)):
            d = 100*scores['dense_semantic']['per_time_class_iou'][slot][category]
            m = 100*scores['mask_lidar']['per_time_class_iou'][slot][category]
            per_class.append({'class_id': category, 'class': label, 'horizon_s': horizon,
                              'dense_iou_percent': d, 'mask_iou_percent': m, 'dense_minus_mask_pp': d-m})
    rows = [{'history_input': mode, 'checkpoint_step': 10000,
             **{f'miou_{h}s_percent': value for h, value in zip((1, 2, 3), score['miou_percent_at_1_2_3s'])},
             'mean_1_2_3s_percent': score['miou_percent_mean_1_2_3s']} for mode, score in scores.items()]
    delta = {'miou_1_2_3s_pp': [a-b for a, b in zip(scores['dense_semantic']['miou_percent_at_1_2_3s'],
                                                  scores['mask_lidar']['miou_percent_at_1_2_3s'])],
             'mean_pp': scores['dense_semantic']['miou_percent_mean_1_2_3s']-scores['mask_lidar']['miou_percent_mean_1_2_3s']}
    report = {'status': 'completed', 'created_at': datetime.now(timezone.utc).isoformat(),
              'windows': 4219, 'scenes': 150, 'checkpoint_step': 10000, 'rows': rows,
              'dense_minus_mask': delta, 'per_class': per_class,
              'audits': {'training': training_audit, 'protocol': protocol, 'ground_truth': gt,
                         'rank_counts_recomputed': True, 'all_required_exit_codes_zero': True},
              'result_sha256': {mode: run['result_sha256'] for mode, run in runs.items()},
              'all_score_protocols': {mode: run['metrics'] for mode, run in runs.items()},
              'single_seed': True, 'statistical_significance_established': False}
    output = Path(config['comparison_dir'])
    output.mkdir(parents=True, exist_ok=True)
    for filename, values in [('comparison.csv', rows), ('per_class_comparison.csv', per_class)]:
        with (output/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    lines = ['# 相同10,000步预算：密集语义历史与mask_lidar对比', '',
             'U形CFM＋actor；原OccFM VAE冻结；两者均使用预先固定的第10,000步EMA。', '',
             '官方validation150场景、4,219窗口；FP32、NFE10、seed2026、每条件1个世界。主结果为未来ego原始全网格17类standard mIoU，忽略GT255；差值为密集历史减mask_lidar，单位为百分点。', '',
             '| 输入 | 1s mIoU | 2s mIoU | 3s mIoU | Avg |', '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+row['history_input']+' | '+' | '.join(f'{row[key]:.4f}' for key in
                     ('miou_1s_percent', 'miou_2s_percent', 'miou_3s_percent', 'mean_1_2_3s_percent'))+' |')
    lines.append('| 密集减mask | '+' | '.join(f'{value:+.4f}' for value in [*delta['miou_1_2_3s_pp'], delta['mean_pp']])+' |')
    for horizon in (1, 2, 3):
        lines += ['', f'## 预测{horizon}秒', '', '| 类别 | 密集语义历史 | mask_lidar | 差值 |', '|---|---:|---:|---:|']
        for row in per_class:
            if row['horizon_s'] == horizon:
                lines.append(f"| {row['class']} | {row['dense_iou_percent']:.2f} | {row['mask_iou_percent']:.2f} | {row['dense_minus_mask_pp']:+.2f} |")
    lines += ['', '## 审计与解释边界', '',
              '- 两模型除历史输入模式外的训练配置逐项相同；使用同一初始化、seed、预算、actor数据与冻结VAE。模型计算源码与输入数据的核验见运行协议。',
              '- 评测设置、源数据SHA、窗口身份、GT计数、坐标覆盖和评分函数一致；聚合计数与各rank原始计数逐项核对，全部命令实际退出0。',
              '- Avg为1/2/3秒mIoU算术平均，不是pooled IoU。原生E0及覆盖区域的诊断分数保留在comparison.json，不替代主表。',
              '- 单训练seed、单生成seed，窗口重叠；没有建立多seed统计显著性、物理一致性或规划收益结论。',
              '- 新实验与既有验证集的诊断历史有关，不宣称独立未见确认性测试。', '']
    (output/'密集历史与mask_lidar_10000步评测对比.md').write_text('\n'.join(lines))
    (output/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-checkpoints', action='store_true')
    args = parser.parse_args(argv)
    config = read_json(args.config)
    result = prepare_checkpoints(config) if args.prepare_checkpoints else build_report(config)
    print(json.dumps({'status': result['status'], 'checkpoint_step': 10000}), flush=True)


if __name__ == '__main__':
    main()

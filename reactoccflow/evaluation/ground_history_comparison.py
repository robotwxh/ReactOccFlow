"""Audited fixed-step comparison with the two completed history experiments.

Project-owned orchestration; see ground_mask_training_sources_20261118.json.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from reactoccflow.data.research import OCC3D_CLASSES
from .architecture_report import (
    SCORE_FUNCTIONS, _function_asts, baseline_metric, read_json, sha256,
    validate_completed, validate_ground_truth, validate_shared_protocol,
)
from .history_input_comparison import export_final_ema, validate_training_configs


MODE = 'mask_lidar_temporal_ground'
MODES = ('dense_semantic', 'mask_lidar', MODE)
LABELS = {'dense_semantic': '密集语义历史', 'mask_lidar': '原mask_lidar', MODE: '新mask_lidar'}
CHINESE = ('其他类', '护栏', '自行车', '公交车', '小汽车', '工程车辆', '摩托车', '行人',
           '交通锥', '拖挂车', '卡车', '可行驶区域', '其他平坦区域', '人行道', '地形', '人造结构', '植被')


def validate_training_pair(reference, candidate, reference_mode):
    if reference.get('history_visibility') != reference_mode or candidate.get('history_visibility') != MODE:
        raise ValueError('Wrong three-way training history mode')
    return validate_training_configs({**reference, 'history_visibility': 'dense_semantic'},
                                     {**candidate, 'history_visibility': 'mask_lidar'})


def validate_policy_only_evaluator_change(current, previous):
    """Allow only the explicit observed-history block to change in the scorer."""
    audit = validate_shared_protocol(current, previous, check_evaluator_code=False)
    paths = []
    for identity in (current, previous):
        matches = [p for p in identity['input_sha256'] if p.endswith('/reactoccflow/evaluation/reactflow_generation.py')]
        if len(matches) != 1 or sha256(matches[0]) != identity['input_sha256'][matches[0]]:
            raise ValueError('Evaluation source identity changed')
        paths.append(Path(matches[0]))
    now, old = (_function_asts(path) for path in paths)
    for name in SCORE_FUNCTIONS:
        if name not in now or name not in old:
            raise ValueError('Missing generation/scoring helper: '+name)
        if name != 'OnlineEvaluationData.__getitem__' and now[name] != old[name]:
            raise ValueError('Generation/scoring helper changed: '+name)
    text = paths[0].read_text()
    allowed = (
        ('history, visible, bounds = [], [], []', 'history, visible = [], []'),
        ("            bounds.append(transformed['in_bounds'])\n", ''),
        ("        if self.history_visibility == 'mask_lidar_temporal_ground':\n"
         '            history, visible = complete_masked_ground_history(history, bounds, visible)\n', ''),
    )
    for before, after in allowed:
        if text.count(before) != 1:
            raise ValueError('Unexpected history preprocessing source')
        text = text.replace(before, after)
    owner = next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == 'OnlineEvaluationData')
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == '__getitem__')
    if ast.dump(method, include_attributes=False) != old['OnlineEvaluationData.__getitem__']:
        raise ValueError('Changes beyond the declared history-input policy')
    return {**audit, 'unchanged_generation_and_score_functions': list(SCORE_FUNCTIONS),
            'allowed_change': 'observed_history_bounds_and_shared_temporal_ground_preprocessing_only'}


def prepare_checkpoint(config):
    previous = read_json(config['reference_export_manifest'])
    exported = previous['models'].copy()
    for mode, value in exported.items():
        if mode not in MODES[:2] or sha256(value['file']) != value['sha256']:
            raise ValueError('Historical fixed-step checkpoint changed')
    exported[MODE] = export_final_ema(config['latest'], config['evaluation_checkpoint'],
                                     read_json(config['normalization']), MODE)
    audits = {mode: validate_training_pair(exported[mode]['config'], exported[MODE]['config'], mode)
              for mode in MODES[:2]}
    result = {'status': 'completed', 'checkpoint_step': 10000, 'models': exported, 'training_config_audits': audits}
    Path(config['export_manifest']).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def build_report(config):
    if sha256(config['reference_comparison']) != config['reference_comparison_sha256']:
        raise ValueError('Historical comparison changed')
    runs = {mode: validate_completed(config['evaluation_dirs'][mode], 10000) for mode in MODES}
    exported = read_json(config['export_manifest'])
    protocols, ground_truth, training = {}, {}, {}
    original = read_json(config['original_occfm_result'])
    for mode, run in runs.items():
        checkpoint = exported['models'][mode]
        if (run['identity']['history_visibility'] != mode or checkpoint['global_step'] != 10000
                or sha256(checkpoint['file']) != checkpoint['sha256']
                or run['identity']['input_sha256'].get(str(Path(checkpoint['file']).resolve())) != checkpoint['sha256']):
            raise ValueError('Wrong history mode or fixed-step checkpoint identity')
        if len({window[0] for window in run['identity']['windows']}) != 150:
            raise ValueError('Expected150 official validation scenes')
        if run['identity']['model_options'] != runs[MODE]['identity']['model_options']:
            raise ValueError('Architectures differ')
    for mode in MODES[:2]:
        protocols[mode] = validate_policy_only_evaluator_change(runs[MODE]['identity'], runs[mode]['identity'])
        ground_truth[mode] = validate_ground_truth(runs[MODE]['result'], runs[mode]['result'], original)
        training[mode] = validate_training_pair(exported['models'][mode]['config'], exported['models'][MODE]['config'], mode)
    for stage in read_json(config['pipeline_record'])['stages']:
        if stage['name'] != 'report' and (stage['status'] != 'completed' or any(p['exit_code'] != 0 for p in stage['processes'])):
            raise ValueError('A required stage has not completed successfully')
    if not read_json(config['reference_comparison'])['audits']['all_required_exit_codes_zero']:
        raise ValueError('Historical comparison lacks successful exits')
    scores = {mode: run['metrics']['paper_frame_full']['standard'] for mode, run in runs.items()}
    rows = [{'history_input': mode, 'checkpoint_step': 10000,
             **{f'miou_{h}s_percent': value for h, value in zip((1, 2, 3), score['miou_percent_at_1_2_3s'])},
             'mean_1_2_3s_percent': score['miou_percent_mean_1_2_3s']} for mode, score in scores.items()]
    per_class = []
    baseline = baseline_metric(original)
    for category, name in enumerate(OCC3D_CLASSES[:17]):
        for horizon, slot in zip((1, 2, 3), (1, 3, 5)):
            values = {mode: 100*scores[mode]['per_time_class_iou'][slot][category] for mode in MODES}
            per_class.append({'class_id': category, 'class': name, 'horizon_s': horizon,
                              'original_occfm_percent': 100*baseline['class_iou'][slot][category],
                              **{mode+'_percent': value for mode, value in values.items()},
                              'new_minus_mask_pp': values[MODE]-values['mask_lidar'],
                              'new_minus_dense_pp': values[MODE]-values['dense_semantic']})
    output = Path(config['comparison_dir'])
    output.mkdir(parents=True, exist_ok=True)
    for filename, values in [('comparison.csv', rows), ('per_class_comparison.csv', per_class)]:
        with (output/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    lines = ['# 三轮10,000步：密集历史、原mask与新mask', '',
             'U形CFM＋actor，冻结原OccFM VAE；预先固定step10000 EMA。官方150场景/4,219窗口；FP32/NFE10/seed2026。', '',
             '| 输入 | 1s mIoU | 2s mIoU | 3s mIoU | Avg |', '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+LABELS[row['history_input']]+' | '+' | '.join(f'{row[k]:.4f}' for k in
                     ('miou_1s_percent', 'miou_2s_percent', 'miou_3s_percent', 'mean_1_2_3s_percent'))+' |')
    for horizon in (1, 2, 3):
        lines += ['', f'## 预测{horizon}秒', '',
                  '| 类别 | 原OccFM复现 | 密集历史 | 原mask | 新mask | 新减原mask | 新减密集 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for row in per_class:
            if row['horizon_s'] == horizon:
                values = [row['original_occfm_percent'], *[row[mode+'_percent'] for mode in MODES]]
                lines.append('| '+row['class']+'，'+CHINESE[row['class_id']]+' | '+' | '.join(f'{v:.2f}' for v in values)
                             +f" | {row['new_minus_mask_pp']:+.2f} | {row['new_minus_dense_pp']:+.2f} |")
        values = [baseline['miou_1_2_3s'][horizon-1], *[scores[mode]['miou_percent_at_1_2_3s'][horizon-1] for mode in MODES]]
        lines.append('| **17类mIoU** | '+' | '.join(f'**{v:.2f}**' for v in values)
                     +f' | **{values[3]-values[2]:+.2f}** | **{values[3]-values[1]:+.2f}** |')
    lines += ['', '## 口径与限制', '',
              '- 主分数为未来ego原始全网格17类standard mIoU，忽略GT255；差值为百分点，按原始精度计算。',
              '- 旧两轮复用已完成的正式结果，重新核验固定EMA、窗口、源数据、GT、rank原始计数与评分代码。仅允许声明的历史输入处理差异。',
              '- 原OccFM为本地e196复现，训练预算及CFG等不同，仅作外部参考。',
              '- 三轮同seed、窗口重叠，且新方案来自既有验证结果的诊断；未建立统计显著性或独立确认性结论。', '']
    (output/'三种历史输入_10000步评测对比.md').write_text('\n'.join(lines))
    report = {'status': 'completed', 'created_at': datetime.now(timezone.utc).isoformat(), 'windows': 4219,
              'scenes': 150, 'checkpoint_step': 10000, 'rows': rows, 'per_class': per_class,
              'audits': {'protocol': protocols, 'ground_truth': ground_truth, 'training': training,
                         'all_required_exit_codes_zero': True, 'rank_counts_recomputed': True},
              'all_score_protocols': {mode: run['metrics'] for mode, run in runs.items()},
              'result_sha256': {mode: run['result_sha256'] for mode, run in runs.items()},
              'single_seed': True, 'statistical_significance_established': False}
    (output/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-checkpoint', action='store_true')
    args = parser.parse_args(argv)
    config = read_json(args.config)
    result = prepare_checkpoint(config) if args.prepare_checkpoint else build_report(config)
    print(json.dumps({'status': result['status'], 'checkpoint_step': 10000}), flush=True)


if __name__ == '__main__':
    main()

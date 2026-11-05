"""Render a completed generation evaluation with explicit comparison limits.

Report formatting reuses the owned OccFM audit report pattern. Reference
searches and source ownership are in the generation evaluation provenance.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from reactoccflow.data.research import OCC3D_CLASSES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    result_path = root / 'full/result.json'
    result = json.loads(result_path.read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    launch = json.loads((root / 'full/launch.json').read_text())
    baseline_path = args.baseline_root / 'full/flow_future_e196/result.json'
    baseline = json.loads(baseline_path.read_text())
    gt_identity = json.loads((root / 'baseline_gt_identity.json').read_text())
    if gt_identity['status'] != 'passed' or gt_identity['checked_frames'] != 6019:
        raise ValueError('Raw GT equivalence to the original OccFM resources must be verified')
    reproduced = baseline['metrics']
    if result['status'] != 'completed' or result['windows'] != 4219:
        raise ValueError('Only the complete official4219 evaluation can produce this report')
    if baseline['status'] != 'completed' or reproduced['samples'] != 4219:
        raise ValueError('The original OccFM reproduction must also be complete')
    for rank in range(2):
        part = json.loads((root / f'full/rank_{rank:03d}/result.json').read_text())
        if part['checkpoint_step'] != 52000 or part['status'] != 'completed':
            raise ValueError('Wrong checkpoint or incomplete rank')
    scores = result['metrics']
    baseline_confusion = np.asarray(reproduced['class_confusion_counts'], dtype=np.int64)
    if not np.array_equal(np.asarray(scores['paper_frame_full']['seen']), baseline_confusion.sum(axis=2)):
        raise ValueError('Full-grid ground-truth counts differ from the original OccFM benchmark')
    actual = scores['paper_frame_full']['legacy']
    actual_values = actual['miou_percent_at_1_2_3s']
    actual_mean = actual['miou_percent_mean_1_2_3s']
    if not np.isclose(actual_mean, np.mean(actual_values), atol=1e-12):
        raise ValueError('Paper-horizon average is inconsistent')
    paper_future = protocol['paper']['future']
    paper_history = protocol['paper']['history']
    reproduced_values = reproduced['miou_percent_at_1_2_3s']
    reproduced_mean = reproduced['miou_percent_mean_1_2_3s']
    delta_paper = actual_mean - paper_future['avg']
    delta_reproduced = actual_mean - reproduced_mean
    conclusion = ('本次设置下，生成mIoU数值高于OccFM论文未来轨迹条件结果；由于输入与表示不同，尚不能认定为受控的结构提升。'
                  if delta_paper > 0 else
                  '本次设置下，生成mIoU没有超过OccFM论文未来轨迹条件结果，尚未体现整体预测mIoU提升。')
    comparison = {
        'status': 'completed', 'checkpoint_step': 52000, 'windows': 4219,
        'scenes': 150, 'reported_protocol': 'future_ego_full_grid_legacy17',
        'paper_future': paper_future, 'paper_history': paper_history,
        'reactflow_miou_1_2_3s_percent': actual_values,
        'reactflow_miou_average_percent': actual_mean,
        'delta_vs_paper_future_percentage_points': delta_paper,
        'delta_vs_paper_future_per_horizon': [a-b for a, b in zip(actual_values, paper_future['miou_1_2_3s'])],
        'reproduced_occfm_future_miou_1_2_3s_percent': reproduced_values,
        'reproduced_occfm_future_miou_average_percent': reproduced_mean,
        'delta_vs_reproduced_occfm_future_percentage_points': delta_reproduced,
        'numerically_exceeds_paper_future': delta_paper > 0,
        'numerical_conclusion': conclusion,
        'controlled_architecture_improvement_established': False,
        'full_grid_ground_truth_counts_match_original_occfm': True,
        'all6019_gt_grids_equal_original_occfm_resources_per_voxel': True,
        'representation_and_condition_limits': protocol['interpretation_limits'],
        'paper_frame_coverage': result['paper_frame_coverage'],
        'all_score_protocols': {name: {p: {key: row[p][key] for key in
            ('miou_percent_at_1_2_3s', 'miou_percent_mean_1_2_3s', 'pooled_miou_percent')}
            for p in ('standard', 'legacy')} for name, row in scores.items()},
        'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'baseline_result': str(baseline_path.resolve()),
        'baseline_result_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    }
    (root/'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2)+'\n')
    rows = [
        ('OccFM paper, historical trajectory', paper_history['miou_1_2_3s'], paper_history['avg']),
        ('OccFM paper, future trajectory', paper_future['miou_1_2_3s'], paper_future['avg']),
        ('OccFM future e196 reproduced', reproduced_values, reproduced_mean),
        ('ReactFlow best52000, future-frame full grid', actual_values, actual_mean),
    ]
    with (root/'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'miou_1s_percent', 'miou_2s_percent', 'miou_3s_percent', 'mean_1_2_3s_percent'])
        for name, values, average in rows:
            writer.writerow([name, *values, average])
    with (root/'per_class_comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['class_id', 'class', 'horizon_s', 'reactflow_full_iou_percent',
                         'occfm_reproduced_iou_percent', 'delta_percentage_points',
                         'reactflow_covered_iou_percent', 'target_class_coverage_percent'])
        for category, name in enumerate(OCC3D_CLASSES[:17]):
            for horizon, time_index in zip((1, 2, 3), (1, 3, 5)):
                current = 100*scores['paper_frame_full']['legacy']['per_time_class_iou'][time_index][category]
                previous = reproduced['class_iou_percent_per_time'][time_index][category]
                covered = 100*scores['paper_frame_covered']['legacy']['per_time_class_iou'][time_index][category]
                seen = scores['paper_frame_full']['seen'][time_index][category]
                support = scores['paper_frame_covered']['seen'][time_index][category]
                coverage = None if seen == 0 else 100*support/seen
                writer.writerow([category, name, horizon, current, previous, current-previous, covered, coverage])
    lines = [
        '# ReactFlow 预测 mIoU 与 OccFM 对照', '',
        '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run',
        '- Origin Date: '+datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        '- Verification Status: VERIFIED (completed generation/scoring, not a controlled novelty claim)',
        '- Version Label: reactflow_miou_best52000_v1', '',
        '## 实际结果', '',
        f'官方验证集 **150 个场景、4,219 / 4,219 个窗口**已完成。测试固定使用第52,000步最佳EMA权重；训练在第72,000步因连续10次验证损失未改善结束。', '',
        f'当前模型全网格平均 mIoU 为 **{actual_mean:.4f}%**，相对 OccFM 论文未来轨迹条件的28.49%为 **{delta_paper:+.4f}个百分点**；相对本机已复现同窗口OccFM为 **{delta_reproduced:+.4f}个百分点**。', '',
        '**'+conclusion+'**', '',
        '此处“平均”是1、2、3秒三个时刻的算术平均，不是全部六时刻平均或跨时间累计的pooled mIoU。mIoU为17个非空语义类的平均交并比，不是逐体素准确率。', '',
        '| 模型与口径 | 1s mIoU | 2s mIoU | 3s mIoU | Avg |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, values, average in rows:
        lines.append('| '+name+' | '+' | '.join(f'{x:.4f}' for x in [*values, average])+' |')
    lines += ['', '论文数值取自[OccFM，arXiv:2506.23434v2，表1](https://arxiv.org/html/2506.23434v2)。历史条件行保留论文印刷Avg=23.33；其三个印刷时刻直接重算得到23.30。主要对照为同样给定未来自车动作的28.49。', '',
              '## 三种评分视角', '',
              '| 评分范围 | 1s mIoU | 2s mIoU | 3s mIoU | Avg |',
              '|---|---:|---:|---:|---:|']
    for key, label in [('native_e0_known', '当前E0内已知目标区域'),
                       ('paper_frame_full', '未来帧原始全网格，超出E0预测视野按empty17计分'),
                       ('paper_frame_covered', '仅未来帧中被E0预测视野覆盖的区域')]:
        row = scores[key]['legacy']
        lines.append('| '+label+' | '+' | '.join(f'{x:.4f}' for x in
            [*row['miou_percent_at_1_2_3s'], row['miou_percent_mean_1_2_3s']])+' |')
    coverage = result['paper_frame_coverage']['fraction_per_horizon']
    lines += ['', 'E0预测视野对未来1/2/3秒原始网格的已知GT体素覆盖率为：'+
              ' / '.join(f'{100*coverage[i]:.2f}%' for i in (1, 3, 5))+'。覆盖区域和E0原生分数仅作诊断，不能替代全网格论文对比。', '',
              '## 复现设置与结论边界', '',
              '- 使用原训练冻结源码的独立普通文件导出，保留原数值实现；未修改原训练副本、检查点或参考仓库。当前开发代码与冻结代码的兼容性检查另见工程probe记录。',
              '- FP32张量、关闭autocast；确定性与TF32设置沿用训练配置。NFE=10、一个生成未来、逐窗口身份固定噪声seed=2026；双卡分片，无重复窗口。',
              '- 未来occupancy和未来位姿仅进入评分；日志未来ego XY作为明确给定的动作条件。历史actor由当前观测选取。没有将未来actor状态输入预测器。',
              '- 各未来帧原始全网格不使用camera/LiDAR评分mask，忽略GT255、排除empty17，累计完整int64计数后计算IoU。另存standard与原公开代码legacy协议；缺失GT类别的处理不同。',
              '- 全网格逐时刻、逐类别GT计数与原OccFM完整复现实验严格相等，确认评分目标与窗口覆盖完全对应。',
              '- 另外独立读取全部6,019帧原始GT，与原OccFM复现的occupancy.npy逐体素比较，全部相等；见baseline_gt_identity.json。',
              '- E0结果经3D最近邻重采样到各未来帧；未来GT位姿仅用于评分坐标对齐。超出原E0范围的预测填empty17并保留计分，避免靠丢弃难区域提高分数。',
              '- 此模型使用mask_lidar限制的历史occupancy及额外真值actor历史；发布OccFM使用原密集语义latent与其轨迹条件。训练场景数、条件表达、表示、原CFG采样和噪声也不同，因此不是同信息、同预算重训的结构消融。',
              '- 本次joint微调使用560个训练场景；初始化OccFM主干与冻结VAE此前接触过官方700个训练场景。官方validation150用于此评测，未用于本次checkpoint选择。',
              '- 本机OccFM复现为此前完成的同4,219窗口EMA/bf16/seed1000结果；本轮为FP32/逐身份seed2026。该精度与噪声差异保留，不把复现行伪装成本轮配对实验。',
              '- 只测试预先按tune损失选定的best权重，没有根据官方验证mIoU挑选checkpoint、采样器或seed。窗口重叠，单seed结果没有统计显著性或架构因果结论。', '',
              '## 原始证据', '',
              '- `full/result.json`：三种评分范围、每时刻每类别累计计数及mIoU。',
              '- `comparison.json` / `comparison.csv`：论文与复现差值。',
              '- `per_class_comparison.csv`：17类在1/2/3秒的IoU、差值和GT类别覆盖率。',
              '- `full/identity.json`：源码、权重、标签、原始网格SHA-256与运行配置。',
              '- `full/launch.json`：实际命令、PID、退出码与耗时。',
              '- `evaluation_source.manifest.json` / `protocol.json`：独立评测副本、预先固定的协议及条件边界。', '',
              f'正式评测耗时：{launch.get("elapsed_seconds", 0)/60:.2f}分钟。源保护审计另存，不把参考目录无变化视作模型预测性能证据。', '']
    (root/'mIoU评测与OccFM对照.md').write_text('\n'.join(lines))
    print(json.dumps({key: comparison[key] for key in ('status', 'windows',
        'reactflow_miou_1_2_3s_percent', 'reactflow_miou_average_percent',
        'delta_vs_paper_future_percentage_points')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

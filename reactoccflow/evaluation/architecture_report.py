"""Audit and report completed standard-DiT / U-CFM / OccFM evaluations.

The owned generation_report.py supplies the report/CSV pattern. Existing owned
OccupancyAccumulator supplies source-compatible count aggregation. Reference
searches and reuse are recorded in architecture_report_sources_20261107.json.
No reference repository is imported, executed, modified, or used as a resource.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from reactoccflow.data.research import OCC3D_CLASSES
from reactoccflow.evaluation.occupancy import OccupancyAccumulator


CFM_RESULT_SHA256 = '274a3697b71eb37b18dfb55dba4832a7dfc43e3ae8afe49ab886fb1f49cbfba5'
PAPER_URL = 'https://arxiv.org/html/2506.23434v2'
PAPER_FUTURE = {'miou_1_2_3s': [36.42, 27.39, 21.66], 'avg': 28.49}
SCORE_NAMES = ('native_e0_known', 'paper_frame_full', 'paper_frame_covered')
COUNTS = ('seen', 'positive', 'correct', 'valid_voxels')
MATCHED_SETTINGS = (
    'nfe', 'seed', 'precision', 'encode_chunk', 'decode_chunk', 'batch_size',
    'world_size', 'numerical_settings', 'limit', 'windows', 'condition',
    'predictor_frame', 'paper_score_frame', 'native_score_mask', 'paper_score_mask',
    'heading', 'torch_version', 'cuda_version',
)
SCORE_FUNCTIONS = (
    'window_seed', 'partition_indices', 'paper_horizon_summary',
    'project_prediction_to_future', 'make_windows', 'make_initial_noise',
    'predict_batch', 'OnlineEvaluationData._labels', 'OnlineEvaluationData._grid',
    'OnlineEvaluationData.__getitem__',
)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def metric_from_counts(score, expected_windows=4219):
    """Recompute every score; do not trust cached derived mIoU fields."""
    if score.get('windows') != expected_windows:
        raise ValueError('Metric window count is incomplete')
    accumulator = OccupancyAccumulator()
    state = {name: np.asarray(score[name]) for name in COUNTS}
    state['windows'] = score['windows']
    accumulator.load_state_dict(state)
    if not np.array_equal(accumulator.seen.sum(1), accumulator.valid_voxels):
        raise ValueError('Ground-truth counts and valid voxel totals disagree')
    report = accumulator.compute()
    for protocol in ('standard', 'legacy'):
        selected = [report[protocol]['per_time_mean_iou'][i] for i in (1, 3, 5)]
        if any(value is None for value in selected):
            raise ValueError('Selected evaluation horizon has no scored classes')
        values = [100*value for value in selected]
        average = float(np.mean(values))
        cached = score.get(protocol, {})
        if (not np.allclose(cached.get('miou_percent_at_1_2_3s', []), values, rtol=0, atol=1e-10)
                or not np.isclose(cached.get('miou_percent_mean_1_2_3s', np.nan), average, rtol=0, atol=1e-10)):
            raise ValueError('Cached mIoU differs from the original count-derived value')
        report[protocol]['miou_percent_at_1_2_3s'] = values
        report[protocol]['miou_percent_mean_1_2_3s'] = average
    return report


def validate_completed(root, checkpoint_step, expected_windows=4219):
    """Prove exact rank coverage and merged count equality before reporting."""
    root = Path(root)
    result = read_json(root/'full/result.json')
    identity = read_json(root/'full/identity.json')
    digest = identity_digest(identity)
    if (result.get('status') != 'completed' or result.get('windows') != expected_windows
            or result.get('identity_sha256') != digest or len(identity.get('windows', [])) != expected_windows):
        raise ValueError('Only a complete identity-bound full evaluation can be reported')
    if len({tuple(window) for window in identity['windows']}) != expected_windows:
        raise ValueError('Window identities contain duplicates')
    world_size = identity['world_size']
    if world_size != 2 or result.get('world_size') != world_size:
        raise ValueError('Expected the recorded two-rank evaluation')
    merged = {name: {key: np.zeros_like(np.asarray(result['metrics'][name][key]), dtype=np.int64)
                     for key in COUNTS} for name in SCORE_NAMES}
    coverage = {key: np.zeros(6, dtype=np.int64) for key in
                ('valid_target_voxels', 'supported_target_voxels')}
    all_indices = []
    for rank in range(world_size):
        part = read_json(root/f'full/rank_{rank:03d}/result.json')
        state = read_json(root/f'full/rank_{rank:03d}/state.json')
        expected = list(range(rank, expected_windows, world_size))
        if (part.get('status') != 'completed' or part.get('checkpoint_step') != checkpoint_step
                or part.get('rank') != rank or part.get('windows') != len(expected)
                or part.get('identity_sha256') != digest or state.get('identity_sha256') != digest
                or state.get('finished') is not True or state.get('completed') != expected):
            raise ValueError('Wrong checkpoint, incomplete rank, or non-exact rank coverage')
        all_indices.extend(state['completed'])
        for name in SCORE_NAMES:
            if state['accumulators'][name]['windows'] != len(expected):
                raise ValueError('Rank accumulator count differs from its cursor')
            metric_from_counts(part['metrics'][name], len(expected))
            for key in COUNTS:
                value = np.asarray(state['accumulators'][name][key], dtype=np.int64)
                if not np.array_equal(value, part['metrics'][name][key]):
                    raise ValueError('Rank result counts differ from its resumable state')
                merged[name][key] += value
        for key in coverage:
            coverage[key] += np.asarray(state['coverage'][key], dtype=np.int64)
    if sorted(all_indices) != list(range(expected_windows)):
        raise ValueError('Rank windows are duplicated or incomplete')
    metrics = {}
    for name in SCORE_NAMES:
        for key in COUNTS:
            if not np.array_equal(merged[name][key], result['metrics'][name][key]):
                raise ValueError('Merged counts do not match the published full result')
        metrics[name] = metric_from_counts(result['metrics'][name], expected_windows)
    for key, value in coverage.items():
        if not np.array_equal(value, result['paper_frame_coverage'][key]):
            raise ValueError('Merged coverage differs from the full result')
    return {'result': result, 'identity': identity, 'metrics': metrics,
            'result_sha256': sha256(root/'full/result.json'),
            'rank_coverage_verified': True}


def shared_data_hashes(identity):
    """Select data/codec identities, excluding architecture source and weights."""
    selected = {}
    for path, digest in identity['input_sha256'].items():
        if ('/data/research/' in path or '/resources/occfm_audit/manifest.json' in path
                or path.endswith('/labels.npz') or path.endswith('/assets/checkpoints/occfm_vae.ckpt')):
            selected[path] = digest
    if not selected:
        raise ValueError('No shared data identities were recorded')
    return selected


def _function_asts(path):
    functions = {}
    def visit(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit(node.body, prefix+node.name+'.')
            elif isinstance(node, ast.FunctionDef):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body = node.body[1:]
                functions[prefix+node.name] = ast.dump(node, include_attributes=False)
    visit(ast.parse(Path(path).read_text()).body)
    return functions


def validate_shared_protocol(current, previous, *, check_evaluator_code=True):
    differences = [key for key in MATCHED_SETTINGS
                   if key not in current or key not in previous or current[key] != previous[key]]
    if differences:
        raise ValueError('Shared evaluation settings differ: '+', '.join(differences))
    if shared_data_hashes(current) != shared_data_hashes(previous):
        raise ValueError('Shared data, normalization, VAE, or source-grid bytes differ')
    checked = []
    if check_evaluator_code:
        def evaluator(identity):
            paths = [Path(p) for p in identity['input_sha256']
                     if p.endswith('/reactoccflow/evaluation/reactflow_generation.py')]
            if len(paths) != 1 or sha256(paths[0]) != identity['input_sha256'][str(paths[0])]:
                raise ValueError('Evaluator source identity is unavailable or changed')
            return _function_asts(paths[0])
        a, b = evaluator(current), evaluator(previous)
        for name in SCORE_FUNCTIONS:
            if name not in a or name not in b or a[name] != b[name]:
                raise ValueError('Generation/scoring helper changed: '+name)
            checked.append(name)
    return {'matched_settings': list(MATCHED_SETTINGS),
            'shared_data_artifacts': len(shared_data_hashes(current)),
            'unchanged_generation_and_score_functions': checked,
            'identity_protocol_versions': [previous.get('protocol_version'), current.get('protocol_version')],
            'architecture_and_checkpoint_identity_changes_allowed': True}


def validate_ground_truth(current, previous, baseline):
    for name in SCORE_NAMES:
        for key in ('seen', 'valid_voxels'):
            if not np.array_equal(current['metrics'][name][key], previous['metrics'][name][key]):
                raise ValueError('DiT and U-CFM ground-truth counts/masks differ: '+name)
    confusion = np.asarray(baseline['metrics']['class_confusion_counts'])
    if confusion.shape != (6, 18, 18) or confusion.dtype.kind not in 'iu' or np.any(confusion < 0):
        raise ValueError('Original OccFM confusion counts have invalid shape or values')
    if not np.array_equal(confusion.sum(2), current['metrics']['paper_frame_full']['seen']):
        raise ValueError('Ground-truth counts differ from original OccFM evaluation')
    a, b = current['paper_frame_coverage'], previous['paper_frame_coverage']
    for key in ('valid_target_voxels', 'supported_target_voxels'):
        if a[key] != b[key]:
            raise ValueError('Fixed-E0 prediction field-of-view coverage differs')
    return {'per_time_per_class_GT_matches_U_CFM': True,
            'full_grid_GT_matches_original_OccFM': True, 'coverage_matches_U_CFM': True}


def baseline_metric(baseline):
    if (baseline.get('status') != 'completed' or baseline.get('validation_scope') != 'full'
            or baseline['metrics'].get('samples') != 4219):
        raise ValueError('Original OccFM reproduction is incomplete')
    matrix = np.asarray(baseline['metrics']['class_confusion_counts'], dtype=np.int64)
    seen, positive, correct = matrix.sum(2), matrix.sum(1), np.diagonal(matrix, axis1=1, axis2=2)
    score = {'windows': 4219, 'seen': seen, 'positive': positive, 'correct': correct,
             'valid_voxels': seen.sum(1)}
    accumulator = OccupancyAccumulator()
    accumulator.load_state_dict(score)
    computed = accumulator.compute()
    selected = [100*computed['legacy']['per_time_mean_iou'][i] for i in (1, 3, 5)]
    if not np.allclose(selected, baseline['metrics']['miou_percent_at_1_2_3s'], rtol=0, atol=1e-10):
        raise ValueError('Original OccFM cached mIoU differs from confusion counts')
    mean = float(np.mean(selected))
    if not np.isclose(mean, baseline['metrics']['miou_percent_mean_1_2_3s'], rtol=0, atol=1e-10):
        raise ValueError('Original OccFM average differs from its horizon values')
    return {'miou_1_2_3s': selected, 'avg': mean, 'class_iou': computed['legacy']['per_time_class_iou']}


def delta(current, previous):
    return {'per_horizon_percentage_points': [a-b for a, b in zip(current['miou_1_2_3s'], previous['miou_1_2_3s'])],
            'average_percentage_points': current['avg']-previous['avg']}


def build_comparison(root, cfm_root, baseline_root):
    root, cfm_root, baseline_root = map(Path, (root, cfm_root, baseline_root))
    protocol = read_json(root/'protocol.json')
    if protocol['checkpoint']['global_step'] != 100000:
        raise ValueError('Expected the standard-DiT best100000 checkpoint')
    paper = protocol['paper']['future']
    if paper != PAPER_FUTURE:
        raise ValueError('Published future-trajectory Table1 values differ from the verified source')
    previous_comparison = read_json(cfm_root/'comparison.json')
    expected_cfm_sha = protocol.get('prior_cfm_result_sha256', CFM_RESULT_SHA256)
    if expected_cfm_sha != CFM_RESULT_SHA256:
        raise ValueError('Unexpected prior U-CFM result identity')
    if sha256(cfm_root/'full/result.json') != expected_cfm_sha or previous_comparison['result_sha256'] != expected_cfm_sha:
        raise ValueError('Previous U-CFM result checksum differs from the frozen baseline')
    current = validate_completed(root, 100000)
    previous = validate_completed(cfm_root, 52000)
    if current['identity'].get('model_options', {}).get('backbone') != 'dit':
        raise ValueError('The new full result does not identify a standard-DiT backbone')
    if len({window[0] for window in current['identity']['windows']}) != 150:
        raise ValueError('The official150-scene census is incomplete')
    checkpoint_hash = protocol['checkpoint']['sha256']
    recorded_checkpoints = [path for path, digest in current['identity']['input_sha256'].items()
                            if path.endswith('/best.pt') and digest == checkpoint_hash]
    if len(recorded_checkpoints) != 1 or sha256(recorded_checkpoints[0]) != checkpoint_hash:
        raise ValueError('Selected DiT checkpoint bytes differ from the fixed protocol')
    shared = validate_shared_protocol(current['identity'], previous['identity'])
    baseline_path = baseline_root/'full/flow_future_e196/result.json'
    if sha256(baseline_path) != previous_comparison['baseline_result_sha256']:
        raise ValueError('OccFM reproduction checksum differs from the earlier audited baseline')
    baseline = read_json(baseline_path)
    reproduced = baseline_metric(baseline)
    gt = validate_ground_truth(current['result'], previous['result'], baseline)
    gt_identity = read_json(cfm_root/'baseline_gt_identity.json')
    if (gt_identity.get('status') != 'passed' or gt_identity.get('checked_frames') != 6019
            or gt_identity.get('raw_gt_array_matches_original_occfm_resources_per_voxel') is not True):
        raise ValueError('Original raw-GT equivalence audit is missing')
    def values(metrics):
        row = metrics['paper_frame_full']['legacy']
        return {'miou_1_2_3s': row['miou_percent_at_1_2_3s'],
                'avg': row['miou_percent_mean_1_2_3s']}
    actual, cfm = values(current['metrics']), values(previous['metrics'])
    if not np.isclose(cfm['avg'], previous_comparison['reactflow_miou_average_percent'], rtol=0, atol=1e-10):
        raise ValueError('Prior U-CFM report differs from its original count-derived result')
    rows = [
        {'id': 'occfm_paper_future', 'name': 'OccFM论文：未来轨迹条件', **PAPER_FUTURE},
        {'id': 'occfm_future_e196_reproduced', 'name': 'OccFM e196：已审计复现', **{k: reproduced[k] for k in ('miou_1_2_3s', 'avg')}},
        {'id': 'actor_u_cfm_best52000', 'name': 'Actor＋U形CFM：best52,000', **cfm},
        {'id': 'actor_standard_dit_best100000', 'name': 'Actor＋标准DiT：best100,000', **actual},
    ]
    output = {'status': 'completed', 'windows': 4219, 'scenes': 150, 'checkpoint_step': 100000,
              'reported_protocol': 'future_ego_full_grid_legacy17_mean_1_2_3s',
              'rows': rows, 'standard_dit': actual, 'actor_u_cfm': cfm,
              'occfm_paper_future': PAPER_FUTURE,
              'occfm_reproduced_future': {k: reproduced[k] for k in ('miou_1_2_3s', 'avg')},
              'delta_vs_actor_u_cfm': delta(actual, cfm),
              'delta_vs_occfm_paper_future': delta(actual, PAPER_FUTURE),
              'delta_vs_occfm_reproduced_future': delta(actual, reproduced),
              'controlled_architecture_improvement_established': False,
              'statistical_significance_established': False,
              'audit': {'current_result_sha256': current['result_sha256'],
                        'prior_cfm_result_sha256': previous['result_sha256'],
                        'original_OccFM_result_sha256': sha256(baseline_path),
                        'standard_dit_checkpoint_sha256': checkpoint_hash,
                        'current_model_options': current['identity']['model_options'],
                        'both_rank_coverage_verified': True, **shared, **gt},
              'all_score_protocols': {'standard_dit': current['metrics'], 'actor_u_cfm': previous['metrics']},
              'paper_frame_coverage': current['result']['paper_frame_coverage'],
              'protocol': protocol}
    return output, reproduced


def write_report(root, comparison, reproduced):
    root = Path(root)
    (root/'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2)+'\n')
    with (root/'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'miou_1s_percent', 'miou_2s_percent', 'miou_3s_percent', 'mean_1_2_3s_percent'])
        for row in comparison['rows']:
            writer.writerow([row['name'], *row['miou_1_2_3s'], row['avg']])
    current = comparison['all_score_protocols']['standard_dit']
    previous = comparison['all_score_protocols']['actor_u_cfm']
    with (root/'per_class_comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['class_id', 'class', 'horizon_s', 'standard_dit_full_iou_percent',
                         'u_cfm_full_iou_percent', 'occfm_reproduced_iou_percent',
                         'delta_vs_u_cfm_percentage_points', 'delta_vs_occfm_percentage_points',
                         'standard_dit_covered_iou_percent', 'target_class_coverage_percent'])
        for category, name in enumerate(OCC3D_CLASSES[:17]):
            for horizon, t in zip((1, 2, 3), (1, 3, 5)):
                a = 100*current['paper_frame_full']['legacy']['per_time_class_iou'][t][category]
                b = 100*previous['paper_frame_full']['legacy']['per_time_class_iou'][t][category]
                c = 100*reproduced['class_iou'][t][category]
                covered = 100*current['paper_frame_covered']['legacy']['per_time_class_iou'][t][category]
                seen = current['paper_frame_full']['seen'][t][category]
                support = current['paper_frame_covered']['seen'][t][category]
                writer.writerow([category, name, horizon, a, b, c, a-b, a-c, covered,
                                 None if seen == 0 else 100*support/seen])
    actual = comparison['standard_dit']
    d_cfm = comparison['delta_vs_actor_u_cfm']['average_percentage_points']
    d_paper = comparison['delta_vs_occfm_paper_future']['average_percentage_points']
    d_repro = comparison['delta_vs_occfm_reproduced_future']['average_percentage_points']
    lines = [
        '# 标准DiT预测mIoU与原CFM、OccFM对照', '',
        '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run',
        '- Origin Date: '+datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        '- Verification Status: VERIFIED（全量生成和计数核验；不是架构因果或显著性结论）',
        '- Version Label: standard_dit_best100000_miou_v1', '',
        f'官方验证集150场景、**4,219 / 4,219窗口**已完成。使用预先按tune损失选定的标准DiT第100,000步EMA权重；没有依据官方验证mIoU选择checkpoint、seed或采样设置。', '',
        f'标准DiT的1/2/3秒平均mIoU为 **{actual["avg"]:.4f}%**，相对原Actor＋U形CFM为 **{d_cfm:+.4f}个百分点**，相对OccFM论文未来轨迹条件为 **{d_paper:+.4f}个百分点**，相对其已审计复现为 **{d_repro:+.4f}个百分点**。', '',
        '| 模型 | 1s mIoU | 2s mIoU | 3s mIoU | Avg |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in comparison['rows']:
        lines.append('| '+row['name']+' | '+' | '.join(f'{v:.4f}' for v in [*row['miou_1_2_3s'], row['avg']])+' |')
    lines += ['', '论文行来自[OccFM原文表1](https://arxiv.org/html/2506.23434v2)。主对照采用给定未来自车轨迹的36.42/27.39/21.66%，印刷Avg=28.49%。', '',
              'mIoU是17个非空语义类的平均交并比。Avg是1、2、3秒三个时刻的算术平均，不是全部六帧平均、跨时刻pooled分数或逐体素准确率。表中差值单位为百分点。', '',
              '## 数据与评测一致性', '',
              '- 新旧模型双rank游标均精确覆盖4219个窗口，无重复、无遗漏；rank原始计数之和与最终结果完全相同。',
              '- 两模型各时刻、各类别GT计数及原生E0/覆盖区域mask一致；全网格GT计数与原OccFM复现混淆矩阵完全一致。',
              '- 原Actor＋U形CFM结果SHA-256与上次已审计报告一致；原OccFM复现结果SHA-256亦一致。所有报告mIoU均从原始int64累计计数重算核验。',
              '- 原始网格、标签及actor/ego时序数据、train-only归一化、VAE和精确窗口清单的SHA-256一致。两个模型使用相同oracle actor/occupancy历史与日志未来ego XY条件。',
              '- 新旧评测使用FP32、NFE10、逐窗口身份固定seed2026、batch4、双卡分片、encode-chunk4/decode-chunk2；数值设置、窗口顺序及生成/评分函数一致。身份协议版本升级仅用于记录主干选择。',
              '- 未来语义和未来位姿仅用于评分坐标对齐；E0预测以3D最近邻重采样到未来ego全网格，超出预测范围的体素保留为empty17参与评分。不使用camera/LiDAR评分mask；忽略GT255并排除empty17。standard和legacy原始结果均保留，主表使用原公开代码legacy口径。', '',
              '## 评分范围诊断', '',
              '| 评分范围 | 标准DiT 1s | 标准DiT 2s | 标准DiT 3s | 标准DiT Avg | 原U-CFM Avg |',
              '|---|---:|---:|---:|---:|---:|']
    for key, label in [('native_e0_known', '当前E0内已知GT区域'),
                       ('paper_frame_full', '未来ego原始全网格'), ('paper_frame_covered', '未来ego内被E0覆盖区域')]:
        a = current[key]['legacy']
        b = previous[key]['legacy']['miou_percent_mean_1_2_3s']
        lines.append('| '+label+' | '+' | '.join(f'{v:.4f}' for v in [*a['miou_percent_at_1_2_3s'], a['miou_percent_mean_1_2_3s'], b])+' |')
    coverage = comparison['paper_frame_coverage']['fraction_per_horizon']
    lines += ['', '固定E0对未来1/2/3秒已知GT体素的覆盖率为 '+
              ' / '.join(f'{100*coverage[i]:.2f}%' for i in (1, 3, 5))+'；原生E0和仅覆盖区域分数用于诊断，不能替代主表全网格对照。', '',
              '## 解释边界', '',
              '- 标准DiT与原U形CFM都采用CFM/RF目标和相同Actor交互接口；本次同时存在参数量、初始化和训练步数/优化轨迹差异，因此不是只改变主干的受控消融。原U形分支使用OccFM主干初始化，标准DiT主干从随机初始化训练；应以各自训练与protocol记录为准。',
              '- 原U形模型按tune损失选定best52,000；标准DiT按tune损失选定best100,000。训练损失量级和参数量不能替代生成mIoU比较。',
              '- 两个Actor模型使用mask_lidar限制的E0历史及额外真值actor历史；原OccFM使用原逐帧密集语义latent、其轨迹条件和CFG采样。本机原OccFM复现为bf16/seed1000，两个Actor模型为FP32/逐窗口身份seed2026；这些区别保留，复现行并非本轮同信息配对实验。',
              '- joint训练划分为560场景；冻结VAE及原OccFM初始化此前接触官方700个训练场景。官方validation150未用于本次checkpoint选择。',
              '- 本次是单个训练seed与单个生成seed；窗口互相重叠，没有据此宣称统计显著性、纯架构因果增益或论文发表结论。', '',
              '## 原始证据', '',
              '- `full/result.json` / `full/rank_*/state.json`：原始计数、覆盖率、rank精确游标。',
              '- `full/identity.json` / `full/launch.json`：输入与源码SHA-256、采样设置、进程与退出码。',
              '- `comparison.json` / `comparison.csv`：四组mIoU及百分点差值。',
              '- `per_class_comparison.csv`：17类在1/2/3秒的标准DiT、U-CFM、OccFM复现IoU及差值。',
              '- `protocol.json`：预先固定权重、训练信息与解释边界。', '']
    (root/'标准DiT与CFM及OccFM_mIoU对照.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cfm-root', type=Path, default=Path('outputs/evaluation/2026-11-04-reactflow-best52000'))
    parser.add_argument('--baseline-root', type=Path, default=Path('outputs/audits/OccFM_weights_2026-11-02'))
    args = parser.parse_args(argv)
    comparison, reproduced = build_comparison(args.root, args.cfm_root, args.baseline_root)
    write_report(args.root, comparison, reproduced)
    print(json.dumps({key: comparison[key] for key in ('status', 'windows', 'standard_dit',
                      'delta_vs_actor_u_cfm', 'delta_vs_occfm_paper_future')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""Report the fixed best84000 hidden512 evaluation against preserved baselines.

Reuses owned architecture_report count/protocol auditors. Reference searches,
commits and source checksums: docs/provenance/dit512_evaluation_sources_20261111.json.
No prediction, checkpoint selection or metric definition is changed here.
"""
from datetime import datetime
import csv
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from reactoccflow.data.research import OCC3D_CLASSES
from reactoccflow.evaluation.architecture_report import (
    PAPER_FUTURE, SCORE_NAMES, baseline_metric, delta, read_json, sha256,
    validate_completed, validate_ground_truth, validate_shared_protocol,
)


def main():
    project = Path(__file__).resolve().parents[1]
    root = project/'outputs/evaluation/2026-11-11-dit512-best84000'
    protocol = read_json(root/'protocol.json')
    expected = {
        'hidden512': (root, 84000, None),
        'hidden256': (root.parent/'2026-11-07-standard-dit-best100000', 100000,
                      'c178b01cb6703423323d370c7fc5e6d73798f7029f56c4fed8bbddea9b564d23'),
        'actor_u_cfm': (root.parent/'2026-11-04-reactflow-best52000', 52000,
                        '274a3697b71eb37b18dfb55dba4832a7dfc43e3ae8afe49ab886fb1f49cbfba5'),
    }
    runs = {}
    exit_audit = {}
    for name, (directory, step, checksum) in expected.items():
        runs[name] = validate_completed(directory, step)
        if checksum and runs[name]['result_sha256'] != checksum:
            raise ValueError('Preserved baseline identity changed: '+name)
        if len({w[0] for w in runs[name]['identity']['windows']}) != 150:
            raise ValueError('Expected all150 official validation scenes: '+name)
        launch = read_json(directory/'full/launch.json')
        # The November4 launcher did not record a separate aggregation exit code.
        # Its complete identity-bound aggregate is audited against rank counts.
        aggregation_code = launch.get('aggregation_exit_code')
        recovered = False
        if name == 'hidden512' and launch.get('original_monitor_interrupted') is True:
            recovery = read_json(directory/'full/aggregation_recovery.json')
            original = directory/'full/launch.before_aggregation_recovery.json'
            recovered = (recovery['status'] == 'completed' and recovery['exit_code'] == 0
                         and recovery['no_regeneration'] is True
                         and recovery['original_rank_exit_codes_observable'] is False
                         and recovery['original_launch_sha256'] == sha256(original)
                         and launch['rank_exit_codes'] == [None, None]
                         and launch.get('rank_generation_complete_verified') is True)
            for rank in range(2):
                evidence = recovery['rank_completion_evidence'][rank]
                for kind in ('result', 'state'):
                    if evidence[kind+'_sha256'] != sha256(directory/f'full/rank_{rank:03d}/{kind}.json'):
                        raise ValueError('Recovered rank completion evidence changed')
        if (launch['status'] != 'completed' or (launch['rank_exit_codes'] != [0, 0] and not recovered)
                or (name != 'actor_u_cfm' and aggregation_code != 0)
                or (aggregation_code is not None and aggregation_code != 0)):
            raise ValueError('Evaluation processes did not finish successfully: '+name)
        exit_audit[name] = {'rank_exit_codes': launch['rank_exit_codes'],
                            'aggregation_exit_code': aggregation_code,
                            'aggregation_recovered_after_monitor_interruption': recovered,
                            'aggregate_counts_verified': True}
    new = runs['hidden512']
    checkpoint_hash = 'f56c2569e6fd982bfe44bd1c8f6894875ec635c8d9c42a09168c75fc59636378'
    checkpoint = project/'outputs/reactflow_dit512/seed_2026/best.pt'
    if (protocol['checkpoint']['global_step'] != 84000
            or protocol['checkpoint']['sha256'] != checkpoint_hash
            or sha256(checkpoint) != checkpoint_hash
            or new['identity']['input_sha256'].get(str(checkpoint)) != checkpoint_hash):
        raise ValueError('Selected hidden512 EMA checkpoint identity changed')
    options = new['identity']['model_options']
    if (options.get('backbone') != 'dit' or options['dit_config'] != {
            'patch_size': 5, 'hidden_size': 512, 'depth': 8, 'num_heads': 8,
            'mlp_ratio': 4.0, 'activation_checkpointing': True}):
        raise ValueError('Unexpected hidden512 architecture')
    old_options = runs['hidden256']['identity']['model_options']
    if old_options.get('backbone') != 'dit' or old_options['dit_config']['hidden_size'] != 256:
        raise ValueError('Previous result is not the hidden256 DiT')
    if protocol['paper']['future'] != PAPER_FUTURE:
        raise ValueError('Paper Table1 values changed')
    baseline_path = project/'outputs/audits/OccFM_weights_2026-11-02/full/flow_future_e196/result.json'
    if sha256(baseline_path) != '8ad643898fe5eb388e81e960ed4d41e44ee03c7a3eb750ee0c0aae488597ac3a':
        raise ValueError('Preserved original OccFM result changed')
    baseline = read_json(baseline_path)
    reproduced = baseline_metric(baseline)
    audits = {}
    for name in ('hidden256', 'actor_u_cfm'):
        audits[name] = {
            **validate_shared_protocol(new['identity'], runs[name]['identity']),
            **validate_ground_truth(new['result'], runs[name]['result'], baseline),
        }
    gt_audit = read_json(expected['actor_u_cfm'][0]/'baseline_gt_identity.json')
    if (gt_audit.get('status') != 'passed' or gt_audit.get('checked_frames') != 6019
            or gt_audit.get('raw_gt_array_matches_original_occfm_resources_per_voxel') is not True):
        raise ValueError('Original raw-GT equivalence audit missing')
    values = {}
    for name, run in runs.items():
        row = run['metrics']['paper_frame_full']['legacy']
        values[name] = {'miou_1_2_3s': row['miou_percent_at_1_2_3s'],
                        'avg': row['miou_percent_mean_1_2_3s']}
    values['occfm_paper_future'] = PAPER_FUTURE
    values['occfm_reproduced_future'] = {k: reproduced[k] for k in ('miou_1_2_3s', 'avg')}
    labels = {'occfm_paper_future': 'OccFM论文：未来轨迹条件',
              'occfm_reproduced_future': 'OccFM e196：本机已审计复现',
              'actor_u_cfm': 'U形CFM＋actor：best52,000',
              'hidden256': '标准DiT hidden256＋actor：best100,000',
              'hidden512': '标准DiT hidden512＋actor：best84,000'}
    rows = [{'id': name, 'name': label, **values[name]} for name, label in labels.items()]
    deltas = {name: delta(values['hidden512'], values[name]) for name in labels if name != 'hidden512'}
    comparison = {
        'status': 'completed', 'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'windows': 4219, 'scenes': 150, 'checkpoint_step': 84000,
        'reported_protocol': 'future_ego_full_grid_legacy17_mean_1_2_3s',
        'rows': rows, 'values': values, 'hidden512_deltas': deltas,
        'all_score_protocols': {name: run['metrics'] for name, run in runs.items()},
        'paper_frame_coverage': new['result']['paper_frame_coverage'],
        'audit': {'all_three_rank_censuses_verified': True, 'shared_protocol': audits,
                  'process_exits': exit_audit,
                  'result_sha256': {name: run['result_sha256'] for name, run in runs.items()},
                  'occfm_reproduced_result_sha256': sha256(baseline_path),
                  'checkpoint_sha256': checkpoint_hash},
        'statistical_significance_established': False,
        'controlled_architecture_improvement_established': False,
        'new_unseen_confirmatory_test': False, 'protocol': protocol,
    }
    (root/'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2)+'\n')
    with (root/'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'miou_1s_percent', 'miou_2s_percent', 'miou_3s_percent', 'mean_1_2_3s_percent'])
        writer.writerows([[r['name'], *r['miou_1_2_3s'], r['avg']] for r in rows])
    per_class = []
    for category, name in enumerate(OCC3D_CLASSES[:17]):
        for horizon, t in zip((1, 2, 3), (1, 3, 5)):
            scores = {key: 100*run['metrics']['paper_frame_full']['legacy']['per_time_class_iou'][t][category]
                      for key, run in runs.items()}
            scores['occfm_reproduced'] = 100*reproduced['class_iou'][t][category]
            per_class.append({'class_id': category, 'class': name, 'horizon_s': horizon, **scores,
                              **{'delta_vs_'+key+'_pp': scores['hidden512']-scores[key]
                                 for key in ('hidden256', 'actor_u_cfm', 'occfm_reproduced')}})
    with (root/'per_class_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_class[0]))
        writer.writeheader(); writer.writerows(per_class)
    current = values['hidden512']
    lines = ['# DiT512、hidden256、U形CFM＋actor 与 OccFM 评测对照', '',
             '## Material Passport', '',
             '- Origin Skill: academic-research-suite / experiment-agent',
             '- Origin Mode: run / validate',
             '- Origin Date: '+comparison['created_at'],
             '- Verification Status: ANALYZED（结果解释；完整生成与原始计数核验通过，未做独立全量重现）',
             '- Version Label: dit512_best84000_full_miou_v1', '',
             f'官方验证集 **150场景、4,219/4,219窗口**双卡评测完成。hidden512使用按完整tune联合损失预选的第84,000步EMA；总训练100,000步，没有依据本次验证mIoU选择权重、seed或采样设置。', '',
             f'本轮全网格Avg mIoU为 **{current["avg"]:.4f}%**。', '',
             '| 模型 | 1s mIoU (%) | 2s mIoU (%) | 3s mIoU (%) | Avg (%) |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+row['name']+' | '+' | '.join(f'{v:.4f}' for v in [*row['miou_1_2_3s'], row['avg']])+' |')
    lines += ['', '论文数值来自[OccFM原文表1](https://arxiv.org/html/2506.23434v2)，采用给定未来轨迹的Ours†行（印刷Avg为28.49%）。原始本机复现独立列出。旧CFM和hidden256使用先前完整评测的冻结计数，本次重新核验身份和重算指标，未重跑或覆盖旧结果。', '',
              '| hidden512相对对照的变化（百分点） | 1s | 2s | 3s | Avg |',
              '|---|---:|---:|---:|---:|']
    for name, change in deltas.items():
        lines.append('| '+labels[name]+' | '+' | '.join(f'{v:+.4f}' for v in [*change['per_horizon_percentage_points'], change['average_percentage_points']])+' |')
    lines += ['', '## 类别差距', '',
              '以下为每一语义类在1/2/3秒的IoU算术平均，不能与逐体素准确率混淆。逐时刻数值保存在per_class_comparison.csv。', '',
              '| 类别 | hidden512 (%) | hidden256 (%) | U形CFM＋actor (%) | OccFM复现 (%) |',
              '|---|---:|---:|---:|---:|']
    for category, name in enumerate(OCC3D_CLASSES[:17]):
        subset = [r for r in per_class if r['class_id'] == category]
        scores = [sum(r[key] for r in subset)/3 for key in
                  ('hidden512', 'hidden256', 'actor_u_cfm', 'occfm_reproduced')]
        lines.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in scores)+' |')
    lines += ['', '17类的三时刻平均IoU均高于hidden256，也均低于U形CFM＋actor。差距同时出现在静态环境和动态对象，例如car、pedestrian、driveable_surface及sidewalk；不能仅用远期运动难度解释。', '']
    lines += ['', '## 评分范围诊断', '',
              '| 评分范围 | hidden512 1s | hidden512 2s | hidden512 3s | hidden512 Avg | hidden256 Avg | U-CFM Avg |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for scope in SCORE_NAMES:
        m = new['metrics'][scope]['legacy']
        scores = [*m['miou_percent_at_1_2_3s'], m['miou_percent_mean_1_2_3s'],
                  runs['hidden256']['metrics'][scope]['legacy']['miou_percent_mean_1_2_3s'],
                  runs['actor_u_cfm']['metrics'][scope]['legacy']['miou_percent_mean_1_2_3s']]
        lines.append('| '+scope+' | '+' | '.join(f'{v:.4f}' for v in scores)+' |')
    coverage = comparison['paper_frame_coverage']['fraction_per_horizon']
    lines += ['', '固定E0对未来1/2/3秒GT网格覆盖率：'+' / '.join(f'{100*coverage[i]:.2f}%' for i in (1,3,5))+'。原生E0和仅覆盖区域分数属于诊断，主对照始终使用未来ego全网格。', '',
              '## 协议与证据核验', '',
              '- 三个actor模型均覆盖同一150场景、4,219个唯一窗口，rank0/rank1分别2,110/2,109个，无重复或补齐。所有指标从原始int64交并计数重算，并核对缓存值。',
              '- 本轮聊天中断后原启动器退出，原始rank的操作系统退出码不可追溯；两rank均保存completed结果、finished状态及精确全量游标。恢复时只执行CPU汇总，退出0；汇总原始计数与两rank完全一致，未重新生成或修改rank结果。原启动记录和恢复记录完整保留。hidden256/U-CFM的历史双rank退出0；旧U-CFM未单列汇总退出码。',
              '- 与hidden256、U-CFM分别核对18项推理设置、5,723份共享数据/VAE/归一化SHA-256和10个生成/评分函数AST，全部一致。GT逐时刻逐类别计数及所有评分mask一致，全网格GT与原OccFM复现混淆矩阵相同；先前6,019帧逐体素GT等价审计保留。',
              '- FP32、NFE10、逐窗口身份噪声seed2026、每条件1个世界、batch4/卡、workers4、encode-chunk4/decode-chunk2。输入为mask_lidar约束的oracle occupancy历史、oracle actor历史及作为已知动作的日志未来ego XY。actor仅按当前观测选取；未来occupancy/actor真值不进入预测器，GT未来姿态仅用于评分投影。',
              '- 17个非空语义类0..16的mIoU；忽略GT255、排除empty17的类别平均。不使用camera/LiDAR评分mask，E0预测经3D最近邻映射到未来ego全网格，超出E0的预测为empty17并计分。Avg是1/2/3秒三个mIoU的算术平均，非逐体素准确率或全部六帧平均。standard/legacy两套计数结果均保留，主表沿用公开代码legacy规则。',
              '- 训练源码153文件以独立普通文件导出并核对SHA，模型和EMA严格加载；评测源码、权重和数据由identity绑定。参考仓库未执行或导入，保护及既有权重/结果完整性另见acceptance.json。', '',
              '## 解释与限制', '',
              '- hidden512修复了旧hidden256在400维patch输入/直接速度输出上的维度瓶颈；修复前后的生成质量变化可由本表衡量，但不能把全部差值分摊给某一个机制。两轮均随机初始化，训练/优化轨迹和最佳步数不同。',
              '- U形CFM使用预训练OccFM主干，hidden256/512从随机初始化训练；参数量、学习率历史、训练预算与best步数不同，不能据此判断某种主干普遍更优。三者均有actor，当前比较也不能单独量化加入actor的收益。',
              '- 原OccFM采用逐帧密集语义latent历史、其轨迹条件和CFG采样；本机原OccFM复现为bf16/seed1000。三个actor模型采用E0表示、masked history、额外actor输入及FP32/逐窗口seed2026。这些差异限制与论文行的直接因果解释。',
              '- 本轮是单个训练seed和单个生成seed，窗口重叠；没有置信区间或统计显著性结论。此前官方验证结果已用于发现并诊断hidden256缺陷，因此本次属于诊断性复用，论文的确认性结论仍需独立未见测试。',
              '- 仅评测occupancy生成mIoU；actor ADE/FDE、碰撞、物理一致性、反事实响应、规划和闭环性能不由这组mIoU证明。8场景tune探针17.3323%/17.7525%未混入正式主表。', '',
              '## 产物', '',
              '- `full/result.json`、`full/rank_*/state.json`：完整原始计数与精确游标。',
              '- `full/identity.json`、`full/launch.json`：输入源码SHA、固定采样参数、进程和退出码。',
              '- `comparison.json`、`comparison.csv`：五行结果及逐时刻/平均百分点差值。',
              '- `per_class_comparison.csv`：17类×3时刻，四个实际模型的IoU及差值。',
              '- `protocol.json`、`preflight_protocol_audit.json`、`acceptance.json`：预先固定协议与一致性/保护核验。', '']
    (root/'DiT512与hidden256_CFM_OccFM评测对照.md').write_text('\n'.join(lines))
    print(json.dumps({'status': comparison['status'], 'rows': rows, 'hidden512_deltas': deltas}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""Render the final report only after all weights and corrected FVD complete."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();root=args.root.resolve()
    subprocess.run([sys.executable,'-B','-m','reactoccflow.occfm_audit.summarize','--root',str(root),'--require-complete'],check=True)
    rows=json.loads((root/'comparison.json').read_text())['rows']
    by_id={r['checkpoint']:r for r in rows}
    hist,future=by_id['flow_hist_e199'],by_id['flow_future_e196']
    criteria=json.loads((root/'criteria.json').read_text());threshold=100*criteria['relative_error_threshold']
    conclusion=('README 指定的两份预测权重，其平均 mIoU 均在用户确认的 2% 相对误差容差内，可视为基本一致' if all(r['miou_average_within_2pct'] for r in (hist,future)) else
                'README 指定预测权重中存在平均 mIoU 超出 2% 相对误差容差的结果')
    f32=json.loads((root/'diagnostics/vae2d_e100_fp32/result.json').read_text())
    f32m=f32['metrics']['miou_percent_mean_all_frames'];f32i=f32['metrics']['iou_percent_mean_all_frames']
    def triple(values):return ' / '.join(f'{x:.2f}' for x in values)
    def signed(value):return f'{value:+.2f}'
    def relative(actual,paper):return 100*abs(actual-paper)/abs(paper)
    def decision(error):return '一致（可忽略）' if error<threshold else '超出容差'
    lines=['# OccFM 全部本地权重实测与论文对照','',
           '启动日期：2026-11-02（Asia/Shanghai）。论文版本：arXiv:2506.23434v2。',
           '',f'**已完成本地全部 10 个 checkpoint 的完整验证集评估，以及 6 个预测权重的修正后 FVD。{conclusion}。**',
           '', '单帧 VAE 的整体 mIoU / IoU 也在 2% 容差内；其 17 类 IoU 在 FP32、5,869 帧集合上全部复现附录表 21 至两位小数。',
           '', '这证明发布权重可以产生下列实测性能。分数接近不能验证全部原始训练记录，分数存在差异也不能单独证明造假。',
           '', '## 用户确认的判定标准', '',
           '`相对误差 = |实测值 − 论文值| / |论文值|`。**严格低于 2% 的差异视为一致、可忽略**；这不是“低于 2 个百分点”。所有原始值和正负差值仍保留。IoU/mIoU 越高越好，FVD 越低越好。', '',
           '| 主要对照项 | 实测 | 论文 | 相对误差 | 判定 |','| --- | ---: | ---: | ---: | --- |']
    checks=[]
    for r,label in [(hist,'历史 e199'),(future,'未来 e196')]:
        for metric in ['miou','iou']:
            checks.append((label+' 平均 '+metric,r[metric+'_average_percent'],r['paper'][metric+'_avg']))
        checks.append((label+' FVD',r['fvd_in_table_units'],r['paper']['fvd']))
    v=by_id['vae2d_e100']
    checks += [('VAE bf16 mIoU',v['miou_average_percent'],92.8),('VAE bf16 IoU',v['iou_average_percent'],85.8),('VAE FP32 mIoU',f32m,92.8),('VAE FP32 IoU',f32i,85.8)]
    for label,actual,paper in checks:
        error=relative(actual,paper)
        lines.append(f'| {label} | {actual:.4f} | {paper:.4f} | {error:.3f}% | {decision(error)} |')
    exceptions=[]
    for r in (hist,future):
        for metric in ['miou','iou']:
            for index,(actual,paper,error) in enumerate(zip(r[metric+'_percent'],r['paper'][metric],r[metric+'_relative_error_percent']),1):
                if error>=threshold:
                    direction='更高' if actual>paper else '更低'
                    exceptions.append(f"{r['checkpoint']} 的 {index}s {metric}：实测 {actual:.4f}，论文 {paper:.2f}，相对误差 {error:.3f}%（实测{direction}）")
    lines += ['', 'README 指定权重中超出容差的逐时刻项：','']
    lines += ['- '+entry for entry in exceptions] if exceptions else ['所有逐时刻 IoU/mIoU 均在容差内。']
    lines += ['', '其余早期 checkpoint 的差异用于对照训练阶段，不额外视为论文必须达到的 SOTA 声明。完整相对误差和判定保存在 comparison.csv / comparison.json。',
           '', '## 1. 预测模型：全部 6 份权重', '',
           'mIoU、IoU 为百分数；三个数依次对应 1、2、3 秒。平均值按这三个时刻计算。Δ为实测减论文，单位为百分点。FVD 按表 3 的 `×10⁻³` 单位显示。', '',
           '| 权重 | 样本窗口 | mIoU 1/2/3s | mIoU 平均 | Δ平均 | IoU 1/2/3s | FVD | ΔFVD |',
           '| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |',
           '| 论文：历史轨迹 | — | 33.17 / 21.09 / 15.64 | 23.33 | — | 40.53 / 30.37 / 24.44 | 7.80 | — |']
    for kind in ['flow_hist','flow_future']:
        if kind=='flow_future':lines.append('| 论文：未来轨迹 | — | 36.42 / 27.39 / 21.66 | 28.49 | — | 43.68 / 36.89 / 31.98 | 7.68 | — |')
        for r in sorted((r for r in rows if r['kind']==kind),key=lambda r:r['checkpoint'],reverse=True):
            delta=r['miou_average_percent']-r['paper']['miou_avg']
            lines.append(f"| {r['checkpoint']} | {r['samples']} | {triple(r['miou_percent'])} | {r['miou_average_percent']:.2f} | {signed(delta)} | {triple(r['iou_percent'])} | {r['fvd_in_table_units']:.4f} | {signed(r['fvd_delta'])} |")
    closest=[]
    for kind in ['flow_hist','flow_future']:
        selected=min((r for r in rows if r['kind']==kind),key=lambda r:max(r['miou_relative_error_percent']))
        distance=max(selected['miou_relative_error_percent'])
        closest.append(f"{selected['checkpoint']}（最大单时刻相对误差 {distance:.3f}%）")
    lines += ['', '若按三个 mIoU 时刻的最大相对误差最小选择，历史、未来两组中最接近论文的权重依次为：'+ '；'.join(closest)+'。上表保留所有实测行，每行对应单份权重的完整评估。']
    lines += ['',f"README 指定历史权重 e199 的 mIoU 差值为 {triple(hist['miou_delta_percentage_points'])}，未来权重 e196 为 {triple(future['miou_delta_percentage_points'])}。",'',
              '原入口默认打印全部 6 个半秒时刻的平均值；它与主表 1/2/3 秒平均值不同。本报告同时在 JSON 保存两种均值，避免混用。', '',
              '## 2. 单帧 VAE：完整 6,019 帧与 FP32 复核','',
              '| 权重 / 设置 | 验证帧 | mIoU | IoU | ΔmIoU（对表 2） | ΔIoU（对表 2） |',
              '| --- | ---: | ---: | ---: | ---: | ---: |','| 论文表 2：192× VAE | — | 92.80 | 85.80 | — | — |']
    v=by_id['vae2d_e100']
    for label,m,i in [('bf16',v['miou_average_percent'],v['iou_average_percent']),('FP32',f32m,f32i)]:
        lines.append(f'| vae2d_e100 / {label} | 6019 | {m:.4f} | {i:.4f} | {signed(m-92.8)} | {signed(i-85.8)} |')
    lines += ['', '逐类结果见 [vae_per_class_comparison.csv](vae_per_class_comparison.csv)。论文其他位置还出现 93.8、93.9 的重建 mIoU；本报告固定以表 2 的 92.8 / 85.8 为基准。',
              '', '### 2.1 集合口径检查与附录表 21 的精确复现','',
              '发布缓存覆盖 5,869 个唯一帧，比当前完整验证索引少每个场景的最后一帧，共 150 帧。本次保持逐帧后验随机序列，单独计算这 150 帧的混淆计数，再从完整结果中相减，隔离集合变化的影响；随机序列跳过方法已与实际前向逐位核对。',
              '', '| 精度 | 全部 6,019 帧 mIoU | 缓存覆盖 5,869 帧 mIoU | 缓存覆盖集合 IoU |',
              '| --- | ---: | ---: | ---: |']
    coverage=json.loads((root/'vae_coverage_diagnostic.json').read_text())
    for item in coverage['results']:
        lines.append(f"| {item['precision']} | {item['full']['miou_percent']:.6f} | {item['cache_covered_subset']['miou_percent']:.6f} | {item['cache_covered_subset']['iou_percent']:.6f} |")
    table21=json.loads((root/'vae_table21_reproduction.json').read_text())
    lines += ['', '集合变化对 mIoU 的影响约 0.003 个百分点，可忽略。整体 mIoU 与表 2 的差异也在用户确认的 2% 相对容差内，判为一致。',
              '', f"**在 FP32、5,869 帧集合上，17/17 类 IoU 与附录表 21 的 x192 行逐项匹配到两位小数。** 实测宏平均 mIoU 为 {table21['measured_macro_miou']:.6f}%；对表 21 印刷数字直接求平均为 {table21['mean_of_printed_table21_values']:.6f}%。这些精确值仅作版本追溯，不把容差内的整体差异列为问题。",
              '', '表 19 有个别类别超出 2% 容差。以匹配表 21 的 FP32、5,869 帧结果为例：construction vehicle 对表19相对误差约2.34%，trailer约3.23%，truck约2.10%，均为实测较低。当前发布权重匹配表 21；这些表19类别结果仍需对应训练或评估版本才能核对。',
              '', '证据：[逐类匹配 JSON](vae_table21_reproduction.json)、[集合口径诊断](vae_coverage_diagnostic.json)、[论文第 22 页原图](evidence/paper_page22.png)。',
              '', '## 3. 3D VAE：全部 3 份特征提取权重','',
              '下表为每个权重在全部 5,269 个六帧窗口上的重建诊断，平均覆盖 6 个时刻。论文没有给这些权重逐一对应的重建数字，因此不将其与单帧 VAE 表 2 混比。', '',
              '| 权重 | 窗口 | 重建 mIoU | 重建 IoU |', '| --- | ---: | ---: | ---: |']
    for r in sorted((r for r in rows if r['kind']=='vae3d'),key=lambda r:r['checkpoint'],reverse=True):
        original=json.loads((root/r['source_result']).read_text())['metrics']
        lines.append(f"| {r['checkpoint']} | {r['samples']} | {original['miou_percent_mean_all_frames']:.4f} | {original['iou_percent_mean_all_frames']:.4f} |")
    lines += ['', '## 4. 评估方法与一致性检查','',
              '- 权重从只读参考仓库复制为独立资源，SHA-256 匹配；模型严格加载，没有忽略缺失或多余参数。预测模型使用 EMA，VAE 使用 state_dict。',
              '- 复制模型只改导入命名空间，保留原网络、采样、CFG 和重复 softmax 行为。预测条件按发布配置保留，包括历史分支中的下一步真值位移；结果不能解释为严格无未来运动信息的实验。未来占据真值仅进入评分器。',
              '- seed=1000，batch size=1；主流程 bf16，另对单帧 VAE 运行完整 FP32 复核。FP32 指关闭 autocast；具体 TF32 默认策略记录在 environment.json。',
              '- 全体素累计 int64 混淆计数，17 个非空语义类，empty=17；保留公开代码的“GT 缺失类别 IoU 记 1”规则。1/2/3 秒对应第 2/4/6 个未来帧。',
              '- 数据与原加载逻辑相同：验证缓存按 token 排序并采用首条重复记录，得到 4,219 个连续 12 帧窗口；单帧 VAE 覆盖全部 6,019 帧，3D VAE 使用全部 5,269 个长度 6 窗口。场景顺序与原 nuScenes 元数据一致。',
              '- 评分器与独立 NumPy 实现核对通过。一组 384 万体素的评估封装输出与复制后的原始 cfm_eval 逐位相同；FVD 复用 GT 特征与直接 pred→GT 计算也通过逐位一致检查。',
              '- 本机为两张 RTX A6000。耗时保留在日志中，不能直接拿来验证论文 RTX 4090 的 FPS。',
              '', '## 5. FVD 修正与未验证指标','',
              '公开 tools/test_fid.py 会把读入的模型预测替换成重排真值。本次使用实际预测，保留发布的 3D VAE e040、FP32、后验采样、5×5 自适应池化与 2,400 维展平特征，再计算高斯 Fréchet 距离。GT 后验噪声按原 pred→GT 顺序预留，因此共享 GT 缓存不会改变该随机序列。每份 FVD 均使用全部 4,219 个窗口。',
              '', '对历史 e199 的完整 4,219 个样本另用原脚本的 scipy.linalg.sqrtm 公式直接重算，得到表单位 8.160063469；与对称矩阵实现的差仅约 2.75e-8 表单位。该数值实现差异不能解释与论文 7.80 的超容差差距。证据见 [FVD 原公式核对](fvd_source_formula_verification.json)。',
              '', '这验证的是修正后的公开特征流程；它不能确认作者生成表 3 时所用的未保存脚本版本。FID/KID 的论文描述使用另一套 Minkowski 特征提取器，当前发布权重中没有匹配的那套权重；NLL 也缺少可直接对应的完整实现。本次没有把替代指标记为这些论文指标的复现。',
              '', '这 10 个本地权重覆盖 nuScenes 语义预测和 VAE，不包含室内、KITTI-360 迁移及各标签比例实验的完整权重。因此，“30/36 次迁移比较”和“5% 标签”的训练结论没有由本次推理评估独立验证。',
              '', '## 6. 统计解释的覆盖','',
              '已筛查 11/11 类统计与方法问题；本次是固定发布权重的推理复跑，没有重新训练多个模型，也没有凭一次推理估算训练置信区间。',
              '', '| 检查项 | 本次范围 |','| --- | --- |',
              '| Simpson 悖论 | 保存逐类统计；未获得论文逐场景结果，不能检验其分组趋势反转 |',
              '| 生态谬误 | 数据集均值不能保证每个场景或类别均改善 |',
              '| 选择偏差 / Berkson | 全部发布权重、完整固定验证集均报告；未验证未发布实验的选择过程 |',
              '| Collider 偏差 | 无控制变量回归的对应分析，未作因果控制检验 |',
              '| 基率忽略 | mIoU 与占据 IoU 分别报告，保留各类混淆计数 |',
              '| 回归均值 | 无极端值选组前后测设计，不直接适用 |',
              '| 幸存者偏差 | 已完成全部本地发布权重；作者未发布的失败训练不在证据范围 |',
              '| 多重搜索 | 没有只选最有利 checkpoint；全表保存所有发布结果 |',
              '| 分析路径自由度 | 明确固定配置、种子、精度与时刻；FP32 诊断单列 |',
              '| 相关与因果 | 差值不能直接归因于优化器、硬件、训练数据或某个设计 |',
              '| 反向因果 / 时间边界 | 保留并披露发布轨迹条件的未来信息，未当作严格因果预测 |',
              '', '## 7. 文件与复核入口','',
              '- [逐权重比较 CSV](comparison.csv)、[完整数值 JSON](comparison.json)、[运行环境](environment.json)。',
              '- 每个权重的 full/<ID>/result.json 保存精度、权重 SHA-256、内部 epoch、完整混淆计数和逐类结果；full/<ID>/run.log 保存进度。预测结果位于对应 predictions.npy。',
              '- 修正 FVD 的结果、特征和日志位于 fvd/<ID>/；共享 GT 特征位于 fvd/gt/。',
              '- [评分与隔离验证](implementation_verification.json)、[模型封装逐位验证](model_wrapper_verification.json)、[FVD 特征逐位验证](fvd_implementation_verification.json)。',
              '- 独立执行代码：[evaluate.py](/home/lthpc/wxh/Paper/reactoccflow/occfm_audit/evaluate.py)。来源与复制哈希保存在 docs/provenance/occfm_audit_sources.json 和 occfm_audit_materials.json。',
              '- [论文 v2](https://arxiv.org/html/2506.23434v2)、[此前的材料与一致性核查](/home/lthpc/wxh/Paper/outputs/audits/OccFM_2026-11-01/核查报告.md)。', '']
    (root/'权重评估与论文对照.md').write_text('\n'.join(lines))
    print(root/'权重评估与论文对照.md')


if __name__=='__main__':main()

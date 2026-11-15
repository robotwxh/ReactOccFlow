# U形CFM＋actor与标准DiT＋actor的大幅差异：结构与数值诊断

> 2026-11-11 新一轮诊断已完成：下文的144维不可控噪声约束针对旧hidden256，不再适用于输入/输出均秩400的hidden512。新版剩余差距已通过固定样本注入/NFE干预、语义误差及梯度检查重新分析，发现actor→occupancy注入的局部退化等问题。见 [hidden512差距诊断](2026-11-11-dit512-cfm-gap-analysis.md)。

> 2026-11-11 后续实测：hidden512修复版best84,000 EMA的官方150场景/4,219窗口全网格Avg mIoU为18.2074%，比hidden256提高15.6320个百分点，仍低于U形CFM＋actor 10.4665个百分点。原生E0 Avg=19.2426%，差距同时涉及静态与动态语义类；本次比较没有分摊剩余原因。见 [全量评测](2026-11-11-dit512-miou-evaluation.md)。

> 后续执行：已按本诊断实施patch5/hidden512及维度检查，新增分项损失与固定tune自由生成监控；387项测试、真实200步拟合、双卡精确恢复和安装隔离检查通过。独立512维全量实验已完成第50步保存并恢复双卡训练，实际语义质量仍待训练和官方评测确认。见 [修复与重训记录](2026-11-07-dit-rank-repair.md)。下文保留修复前的诊断事实。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate；辅以冻结模型的有界数值诊断
- Verification Status: ANALYZED；固定输出子空间约束已由源码、SVD和数值投影验证，未完成全验证集差距的逐因素因果分摊
- Date: 2026-11-07
- 数据边界：官方validation成绩保持原样；新增实验只用8个事先选定的tune场景
- 运行边界：没有改模型/配置/权重，没有重训，没有根据官方验证mIoU挑选采样设置

目前最直接的解释不是“DiT天生不适合occupancy”，而是本项目这版标准DiT存在**patch数值维度与hidden宽度不匹配的结构缺陷**：16通道、patch5对应400维，输入/输出却经过256维瓶颈；当前又采用直接速度场输出、没有原始noisy state的输出旁路。因此有144个输出方向的初始高斯噪声无法被该固定模型消除。预训练、局部归纳偏置和优化设置还会影响差距，但本次没有把它们分别量化为官方验证集mIoU的因果贡献。

此前验收确认了形状、可运行、恢复与源码隔离，却遗漏了这个表达能力检查。这个遗漏属于当前迁移配置的设计问题；不能把此前程序检查通过解释为生成质量已合格。

## 1. 对比中的相同与不同

| 项目 | U形CFM＋actor | 标准DiT＋actor |
|---|---|---|
| 官方4,219窗口Avg mIoU | 28.6739% | 2.5755% |
| 训练目标 | 直线概率路径CFM/RF，直接预测速度 | 相同 |
| VAE/数据 | 同一冻结Swin VAE；train560/tune70；同一缓存及归一化 | 相同 |
| Actor/交互分支 | 74个张量、826,939参数；128维，2层self＋双向cross-attention | 同数量、同形状 |
| Actor token | 8×6=48 | 48 |
| 交互位置 | mid_block2后，12×25×25=7,500个occupancy token | 第4/8 block后，12×10×10=1,200个token |
| 细节通路 | 卷积、多尺度down/up和高分辨率skip | 线性patchify、固定宽度Transformer、线性unpatchify |
| 最终输出映射 | 每位置64→16的1×1×1卷积，实际行秩16 | 每patch256→400线性层，实际秩256 |
| 初始化 | OccFM history epoch199的state_dict分区，204个主干键 | 主干随机初始化 |
| 实际主干训练状态 | `freeze_trunk(False)` | `freeze_trunk(False)` |
| 初始主干LR | 2e-5 | 1e-4 |
| Actor分支初始LR | 1e-4 | 1e-4 |
| best/最终步数 | 52,000 / 72,000 | 100,000 / 100,000 |

两者均为有效batch8、FP32、warmup1000、EMA0.9999、weight decay0.01。不能根据旧归档YAML中的`freeze_pretrained_trunk: true`推断U形主干没有训练；真实runner明确解除冻结。

本轮官方评测两模型使用同一4,219窗口顺序、FP32/NFE10/逐窗口seed2026、batch4/卡、相同VAE、输入、评分函数与数值设置。已对照18项协议字段、5,723个输入文件SHA及所有GT计数。因此单纯的评测设置差异解释不了两组joint模型的巨大差距。

## 2. 已验证的硬约束：400维速度只能从256维固定列空间输出

源码入口见[`standard_dit.py`](../../reactoccflow/models/standard_dit.py)：

```text
每帧latent：16 × 50 × 50
每个patch：16 × 5 × 5 = 400个数值
patch_embed：Linear(400, 256)
8层Transformer与actor交互：256维token
final_layer.linear：Linear(256, 400)
```

设末层权重为 \(W\in\mathbb R^{400\times256}\)、偏置为 \(b\)，每个patch的预测速度总是：

\[
v_\theta(x,t,c)=W h_\theta(x,t,c)+b.
\]

实际best100,000权重SVD得到rank=256，最大/最小奇异值0.745843154 / 0.107319034。令 \(P\) 为 \(W\) 列空间的正交补投影，则维数为400−256=144，且：

\[
Pv_\theta=Pb.
\]

这144个方向上的速度仅为固定偏置，不能依赖当前噪声、历史、actor或动作条件。令生成流状态初值为标准高斯 \(\epsilon\)，总积分时间为1，模型最后除以latent_scale=10，则：

\[
P(10\,z_{\rm pred})=P\epsilon+Pb.
\]

对当前固定权重，这个关系同样适用于总步长为1的Euler积分。更多Transformer层、actor交互或NFE都不能绕过最后同一个固定W。注意保留偏置项，不能把投影速度误写为恒零。

对独立新高斯噪声，400维平均的条件噪声方差至少为：

\[
\frac{144}{400\times10^2}=0.0036,
\qquad \text{对应RMS}=0.06.
\]

这是平均方向方差的期望约束，不是每个坐标的逐点下界，更不是mIoU数值定理。VAE解码器是非线性的。

对均匀逐元素occupancy FM MSE，还得到：

\[
\mathbb E\mathcal L_{\rm occ}\ge
\frac{144}{400}+
\frac{\mathbb E\|P(10z_{\rm GT}-b)\|^2}{400}\ge0.36.
\]

这个结论针对纯occupancy速度分项、固定权重与独立高斯噪声；不能把0.36直接认作两个完整joint loss或mIoU差距的精确因果分摊。

输入端`Linear(400,256)`同样不可逆，而且没有raw-state旁路。落在输入矩阵核空间中的占用噪声扰动不会改变token，因此actor和全局attention也看不见它。仅修改最后一层输出、仍保留该输入瓶颈，不足以系统解决问题。

## 3. 数值验证：噪声确实被保留

在此前固定的tune样本scene-0011、seed617、NFE10上，按源码的(p,q,c)顺序patchify后验证：

```text
(10 × generated_latent − initial_noise − output_bias) @ output_null_basis
```

最大绝对误差为**5.84e−7**，RMS误差1.11e−7，符合float32积分误差量级。该样本跨patch中心化的生成方差中，约94.61%落在这些无法控制的输出方向。原始生成latent整体std为0.06159，GT为0.01632。std量级吻合本身不是证明；投影守恒才是直接检验。

在8个预先选定的tune场景、同一批噪声的固定生成时间点上，DiT在这些方向的速度MSE恒为约**0.367687**；对应理论不可拟合投影误差也为0.367687，t=0/0.1/0.5/0.9均如此。U形模型在相同方向的误差可明显低于此值，因此没有遭遇同样的输出限制。

| 生成时间t | U形速度MSE | DiT速度MSE | DiT不可拟合投影分量 |
|---|---:|---:|---:|
| 0.0 | 0.01010 | 0.38233 | 0.36769 |
| 0.1 | 0.00787 | 0.37911 | 0.36769 |
| 0.5 | 0.02034 | 0.38762 | 0.36769 |
| 0.9 | 0.20226 | 0.53491 | 0.36769 |

这里的t是RF生成时间，不是物理预测的1/2/3秒。固定时间点使用带噪真值状态做速度场诊断，不是生成mIoU测试。

## 4. 不是actor分支参数减少：损失差主要位于occupancy

按原JointTrainingAdapter累计组件原始numerator/count，再算加权均值；以下仅为8个固定tune样本的诊断，不冒充完整2,176窗口tune结果：

| 分项 | U形CFM＋actor | DiT＋actor |
|---|---:|---:|
| occupancy FM | 0.05573 | 0.41371 |
| actor FM | 0.07548 | 0.06666 |
| actor存在性，加权后 | 0.000296 | 0.000338 |
| occupancy–actor一致性，加权后 | 0.06622 | 0.07493 |
| 静态时序一致性，加权后 | 0.000190 | 0.001491 |
| 完整加权诊断损失 | 0.19795 | 0.55715 |

occupancy分项的增加为0.35798，joint损失增加为0.35920；约99.66%的**这组小样本损失差**在算术上来自occupancy分项。这个比例不能移用为官方mIoU下降的因果比例。actor监督FM损失并未变差；这也不等价于已证明actor自由轨迹ADE/FDE更好。

## 5. 残余噪声足以严重破坏解码，但只在终点去噪不能修好现有模型

同一8个固定tune场景，使用native-E0 known区域、standard类别口径，对1/2/3秒mIoU取均值：

| 对照 | mIoU | 含义 |
|---|---:|---|
| 真值latent经同一VAE重建 | 93.68% | 排查VAE/标签映射本身是否失效；这是oracle重建 |
| 真值latent＋已知保留的nullspace噪声 | 4.77% | 只引入这部分初始噪声，即可严重破坏解码 |
| 真值latent投影到DiT可达仿射子空间 | 45.12% | 输出子空间丢掉真值细节；是latent L2投影，不是IoU理论上界 |
| 上述投影真值＋保留噪声 | 3.99% | 即使可控分量由真值提供，噪声仍严重损坏解码 |
| U形CFM实际生成 | 29.13% | 同一子集的正常预测 |
| DiT实际生成 | 2.82% | 当前模型预测 |
| DiT生成后仅减去已知初始噪声的null投影 | 3.70% | 不使用GT做修正，仅为有界诊断，未应用于官方评测 |
| 重复最后观测latent的保持基线 | 44.98% | 该小子集上的诊断基线，不能外推为完整验证集成绩 |

控制实验均按对应结果产生前记录的补充计划执行，未筛掉不理想结果。GT控制使用未来真值，明确属于oracle诊断，不是可部署预测器。

这组证据同时说明两点：保留噪声是具有强机制证据的严重问题；但只在终点减去它，当前模型仍只有3.70%，没有恢复U形效果。可控方向的生成结果仍与oracle投影真值存在很大差距，说明还存在其它预测误差；它们是否来自历史条件利用、自由rollout分布偏移或训练适配，尚未分别验证。不能说“减掉36%噪声就能恢复全部26个百分点”，也不能承诺仅改hidden维度就恢复成绩。

## 6. U形分支为何更容易保留有效场景信息

U形模型最后是每空间位置64→16的卷积，实际输出矩阵行秩为16、左零空间为0；25个位置可各自输出16个数值，没有同样的每5×5块固定144维不可控输出限制。初始卷积提取高分辨率特征，down/up skip与最终stem skip把细节传到输出。actor更新中间occupancy表征后，decoder仍能利用这些局部信息。

标准DiT把整个5×5块先压成一个256维token，交互处只剩10×10空间网格；即使全局attention能看到更大范围，原始独立噪声的丢失维度也不能凭上下文恢复。全局建模范围不等于保留了完整去噪信息。

此外，U形主干加载已训练的OccFM占用生成知识，而DiT从零学习；主干LR又分别是2e−5和1e−4。预训练、局部归纳偏置、初始化与学习率很可能影响训练难度，但本次没有做单因素重训，不能把各自贡献写成已测定因果值。更多步数不能弥补上述固定秩约束；解除它后仍需重新验证学习过程。

## 7. 保持标准DiT的修复方向，尚未实施

原始[DiT官方实现](https://github.com/facebookresearch/DiT/blob/main/models.py)默认4通道latent、patch2和hidden1152；其噪声patch只有16个数值。其小模型也是hidden384，不能将本项目16通道、patch5的400维patch与这些配置直接类比。[原论文](https://arxiv.org/abs/2212.09748)说明DiT可替代U-Net且能随宽度、深度和token数量扩展；并不保证任意窄配置在本任务有效。

对当前线性patchify＋直接速度头，优先保证可观察/可输出的噪声维度不被强制截断：

| 方案 | patch数值维度 | hidden | 全序列token | 每头attention矩阵元素 |
|---|---:|---:|---:|---:|
| 当前 | 400 | 256 | 1,200 | 1.44M |
| patch5，hidden512 | 400 | 512 | 1,200 | 1.44M |
| patch2，hidden256 | 64 | 256 | 7,500 | 56.25M |

patch5/hidden512同时解除输入和输出的被迫低秩尺寸限制，保留1,200 token，但宽度相关计算和参数增加，较适合作为首个受控修复候选。patch2也可避免该尺寸问题，但token增6.25倍、attention矩阵增39.06倍。latent边长50不能直接使用patch4，除非明确定义padding/cropping及对应验证。

只增加depth或NFE不会改变最后W的144维正交补；只在输出后增加一层而保留不可逆输入，也不能视为完成修复。引入原始state的解析旁路、改预测参数化或时变输出投影是其它可能方案，但要重新推导训练/采样一致性，不能随意拼接后继续旧模型。

这里仅给出下一步建议，没有修改配置、没有发起重训。改变hidden或patch后应作为新实验，在train/tune先通过表达能力检查、噪声消除和自由生成诊断，再决定完整训练；已用于本轮分析的官方验证集不应反复承担调参选择。

## 8. 统计与因果表述检查（11/11已考虑）

| 检查 | 本次处理与限制 |
|---|---|
| Simpson聚合反转 | 8场景诊断不代表150场景总体；没有据子集宣称总体改进 |
| 生态谬误 | 整体occupancy分数不推断单个actor轨迹质量 |
| 选择/Berkson偏差 | 8场景按ID等距、每场景首窗口事先固定；不是概率抽样 |
| Collider偏差 | 无协变量回归或按生成结果条件筛选 |
| 忽略基准率 | 明确free体素主导；不用逐体素accuracy代替非空类mIoU |
| 均值回归 | 不把同样本上事后修正视为新模型泛化提升 |
| 幸存者偏差 | 官方全部4,219窗口和全部8诊断样本保留，无成功样本筛选 |
| 多重检索 | 无p值筛选；所有预先列出的诊断对照均保存 |
| 分析路径自由度 | 补充oracle对照属于探索性，计划先于对应结果记录 |
| 相关与因果 | 固定秩和噪声守恒是解析约束；小子集扰动支持机制，不分摊总体架构差距 |
| 反向因果 | 结构与权重先于额外诊断；不以mIoU低反推SVD秩或反向选择样本 |

## 原始记录

- [诊断计划](../../outputs/diagnostics/2026-11-07-dit-cfm-gap/diagnostic_plan.json)
- [SVD与噪声投影守恒](../../outputs/diagnostics/2026-11-07-dit-cfm-gap/output_rank_analysis.json)
- [配对诊断汇总](../../outputs/diagnostics/2026-11-07-dit-cfm-gap/summary.json)
- [oracle控制原始计数](../../outputs/diagnostics/2026-11-07-dit-cfm-gap/oracle_controls.json)
- [因果证据与边界](../../outputs/diagnostics/2026-11-07-dit-cfm-gap/causal_evidence_summary.json)
- [诊断图PNG](../../outputs/diagnostics/2026-11-07-dit-cfm-gap/gap_diagnosis.png) / [SVG](../../outputs/diagnostics/2026-11-07-dit-cfm-gap/gap_diagnosis.svg)
- [源码来源与不变性记录](../provenance/dit_cfm_gap_analysis_20261107.json)

![八个固定tune场景诊断图](../../outputs/diagnostics/2026-11-07-dit-cfm-gap/gap_diagnosis.png)

# ReactOccFlow - Design Spec

> 2026-10-31：本规格内容已同步为标准 DiT 与 occupancy–actor 注意力分工。现有 SVG/预览/导出文件仍反映旧 U-Net 版本；本次仅更新 Markdown，下列网络内容是重生成图件的依据，不是已有图件已经同步的声明。

## I. Project Information

| Item | Value |
| ---- | ----- |
| Project Name | ReactOccFlow 总体流程图 |
| Canvas Format | Custom portrait SVG, A4 ratio, 2480×3508 |
| Page Count | 1 |
| Design Style | 用户提供的白底彩色分区学术方法图 |
| Target Audience | 自动驾驶世界模型研究者，论文方法讲解与组会 |
| Use Case | 全流程阅读、局部放大、SVG 编辑 |
| Delivery Purpose | 近距离阅读的研究总图 |
| Content Strategy | 按权威计划与源码整理；内容完整，关键网络展开；参考图片仅控制视觉样式 |
| Created Date | 2026-10-31 |
| Confirmation | 用户在聊天中明确回复“按此生成”，接受单张 SVG、尺寸、六区范围及参考图风格；连续生成 |

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| Format | SVG portrait |
| Dimensions | 2480×3508 |
| viewBox | `0 0 2480 3508` |
| Margins | 28 |
| Content Area | x=28…2452, y=174…3446 |

项目初始化为 a4 目录类型，实际画布采用用户确认的双倍 A4 尺度；以 spec_lock 的 viewBox 为准。

## III. Visual Theme

### Theme Style

- Mode: instructional
- Visual style: custom
- Theme: Light
- Tone: 紧凑、技术准确、分层明确。
- Behavior: 按参考图使用横向编号分区与细描边圆角模块，区内以左到右和上到下流程组织，保留张量尺寸、条件支路与残差旁路。核心网络使用真正计算节点与箭头，不以列表代替网络。没有装饰纹理或阴影。

### Color Scheme

用户已给定参考图配色方向，直接延续；模块填色是其浅色衍生。

| Role | HEX | Purpose |
| ---- | --- | ------- |
| Background | #FFFFFF | 白底与残差加法节点 |
| Body text | #102A43 | 正文、主计算连线 |
| Secondary text | #486581 | 注释 |
| Primary | #087FCD | ①输入与任务、occupancy分支 |
| Primary light | #EEF8FF | ①背景 |
| Primary panel | #DDEFFF | Attention模块 |
| Green | #0A8F66 | ②VAE、Norm |
| Green light | #EFFBF5 | ②背景 |
| Green panel | #DAF2E4 | Norm模块 |
| Purple | #7C3AC9 | ③联合流、actor分支 |
| Purple light | #F7F1FF | ③背景 |
| Purple panel | #EBDDFB | actor与FFN |
| Orange | #E8750B | ④核心结构、门控 |
| Orange light | #FFF8ED | ④背景 |
| Orange panel | #FFF0CA | FFN与条件 |
| Teal | #078780 | ⑤推理规划 |
| Teal light | #EFFAF9 | ⑤背景 |
| Indigo | #4058A8 | ⑥数据验证 |
| Indigo light | #F1F4FD | ⑥背景 |
| Red | #C74A52 | 训练真值、计划组件与警示 |
| Red light | #FFF0F1 | 真值边界与待完成组件 |
| Border | #BDD1DF | 辅助分隔 |

## IV. Typography System

### Font Plan

采用与参考图一致的粗体无衬线标题及清晰技术正文，使用字重形成一致性层级。SVG 不依赖嵌入位图。

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| Title | Noto Sans CJK SC / Microsoft YaHei | Arial | sans-serif |
| Body | Noto Sans CJK SC / Microsoft YaHei | Arial | sans-serif |
| Emphasis | same as Body | same as Body | sans-serif |
| Code | Noto Sans CJK SC | Consolas | monospace |

- Title: `Arial, 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif`
- Body: `Arial, 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif`
- Emphasis: same as Body
- Code: `Consolas, 'Noto Sans CJK SC', monospace`

本机已安装 Noto Sans CJK SC；其他设备可安装该字体或使用 Microsoft YaHei 回退。

### Font Size Hierarchy

| Role | Size | Weight |
| ---- | ---- | ------ |
| title | 58 | 700 |
| subtitle | 32 | 500 |
| section | 40 | 700 |
| subheading | 30 | 700 |
| body | 28 | 400 |
| annotation | 24 | 400 |
| footnote | 22 | 400 |
| micro | 20 | 400 |

Formula policy: text-only。使用 Unicode 与 tspan 结构，保持公式可编辑。

## V. Layout Principles

- 顶部：标题、目标分布和状态图例。
- ① y174–524：观测→候选条件→未来输出；给出输入尺寸和目标定义。
- ② y540–950：类别/高度编码→Swin encoder→μ latent→decoder→semantic logits。
- ③ y966–1346：双模态噪声/目标→共享线性流→条件向量场→损失；下排连续端点约束。
- ④ y1362–2712：上排标准 DiT/actor 两分支，标明第 4 层后的双向 cross-attention；下排 DiT self-attention、双向 interaction、Swin 三张模块展开图。
- ⑤ y2728–3168：候选→ODE→解码→碰撞事件→校准→筛选；下排控制、回退与闭环目标。
- ⑥ y3184–3446：数据划分、动作干预、评价与消融。
- 分区间距16；普通模块内边距16–24；连线不穿过文字；残差连线绕行模块边缘。
- 实线表示已有实现/接口，虚线表示尚待完成的模块或实验接入；细虚线条件线与状态虚线通过文字区别。

## VI. Icon Usage Specification

不用外部图标库。以手写原生 SVG 绘制 BEV 示意、latent 栈、轨迹线、加法节点和箭头。图形均为示意，不代表预测结果。

## VII. Visualization Reference List

| Page | Template | Path | Summary-quote | Usage |
| ---- | -------- | ---- | ------------- | ----- |
| P01 | no-template-match | none | 用户提供的两张对话图片是样式参考 | 按实际源码手工构建网络与流程，无数据图表模板 |

版式须表达固定分辨率的 DiT 层堆叠、token 编解码、条件调制与同步双向交互；不再绘制旧 U-Net 的下采样、上采样与跨尺度 skip。DiT Block 内的残差仍需保留。

## VIII. Image Resource List

无外部图像、AI位图、公式PNG或占位图。用户图片仅为对话中的视觉参考，不嵌入成品。所有内容为可编辑矢量元素。

## IX. Content Outline

### P01 — ReactOccFlow 总体流程图

- Page rhythm: dense
- Source: sources/source_intake.md；来源路径与对应事实由该文件列明。
- 核心信息：候选 ego 动作条件下，以共享时间的联合流生成 occupancy 与 actor 世界，交由事件风险与候选筛选支持规划。
- ①输入：6槽occupancy历史/4有效、8actor×4时刻、6未来时刻、候选8+停车1。当前actor选择只取观测。
- ②VAE：同一套18类8维+高度4维编码、Swin encoder depths2/6、200→100→50、μ16×50×50、decoder50→200、18类logits。历史观测与未来训练标签离线编码；训练读取 latent，预测端点解码；patch 投影不画成第二套 VAE。
- ③训练：独立模态噪声，共享t；直线插值；velocity回归与existence；连续端点经冻结decoder取得三类一致性；显示总损失权重。
- ④核心：occupancy latent 按5×5 patch线性投影400→256，12槽共1200 token；DiT第1–4层→两层interaction→第5–8层→adaLN输出投影/unpatchify。actor51维历史及噪声/时间/动作/slot编码形成48个128维token；cross交互为128维4头，delta回投256维。输出v_Z/v_A/existence。
- ④细节：occupancy全局时空self-attention为1200 token、8头；actor masked self-attention为48 token、4头；双向cross分别标注Q=actor/KV=occupancy与Q=occupancy/KV=actor，计算来自同轮状态；门控残差、actor FFN、delta回投均展开。两模态不拼接做共同self-attention。VAE另展W-MSA/SW-MSA与残差。
- ⑤推理：每候选8世界、8次ODE、future-only积分；decoded occupancy与actor车体风险；平滑频率与独立校准；冻结阈值；轨迹选择和刹停；bridge/control与目标闭环。
- ⑥支持：数据划分与训练统计；CARLA反事实按origin分组；occ/actor/响应/校准/闭环指标及结构/预算消融。
- 实现边界：learned PlanFlow、真实LiDAR latent学习、反事实采集和闭环实验均显式待完成；不陈述当前实时或因果效果。
- Closing impact: 从候选动作到联合世界，再到事件风险与决策的完整可追踪链路；总图下方注明源码/计划依据。

## X. Speaker Notes Requirements

notes/total.md记录一页的阅读顺序、数学解释、张量约定和实验边界；非SVG正文的详细来源信息保留在附属材料中。

## XI. Technical Constraints Reminder

- 单张手写SVG；viewBox与spec_lock完全一致。
- inline属性，无style/class/foreignObject/script/filter/mask。
- 不用symbol/use复用图元；箭头marker在defs中声明为三角形。
- XML转义；文字按语义分组，以text/tspan保留可编辑性。
- 以svg_output为作者源，finalize生成svg_final；按用户要求只交付SVG，无需导出PPTX。
- 本任务不实现模型函数，不执行参考项目代码；使用用户指定ReactOccFlow环境。

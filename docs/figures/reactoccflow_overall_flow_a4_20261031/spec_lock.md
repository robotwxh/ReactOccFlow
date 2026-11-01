# Execution Lock

## architecture_revision_2026_10_31

- status: Markdown synchronized; existing SVG/previews/exports require regeneration
- backbone: standard isotropic DiT; patch=5, hidden=256, depth=8, heads=8
- occupancy_tokens: 12×10×10=1200; global space-time self-attention
- actor_tokens: 8×6=48; hidden=128; masked self-attention
- exchange: two interaction modules after DiT block 4; four heads
- cross_actor_update: Q=actor, K/V=occupancy
- cross_occupancy_update: Q=occupancy, K/V=actor
- cross_order: simultaneous updates from the same normalized state, gated residuals, actor FFN
- token_sequences: separate modality sequences; no concatenated occupancy–actor self-attention
- occupancy_delta: 256→128 projection before exchange; delta 128→256 back-projection
- codec: one frozen occupancy VAE; DiT patch projection is not a second VAE encoder
- checkpoint_scope: scratch DiT; old U-Net training/weights and original export hashes remain historical
- retained_visual_spec: the canvas, colors and typography below are retained for a future redraw

## canvas
- viewBox: 0 0 2480 3508
- format: Custom portrait SVG 2480×3508, A4 ratio

## mode
- mode: instructional

## visual_style
- visual_style: custom
- visual_style_behavior: 用户参考图的紧凑学术架构图；编号横向分区，细描边圆角模块，清楚的箭头、残差旁路与条件支路，主干与局部模块分层，粗体无衬线标题，规则无衬线正文，紧凑留白，无纹理无阴影。

## colors
- bg: #FFFFFF
- text: #102A43
- text_secondary: #486581
- primary: #087FCD
- blue_light: #EEF8FF
- blue_panel: #DDEFFF
- green: #0A8F66
- green_light: #EFFBF5
- green_panel: #DAF2E4
- purple: #7C3AC9
- purple_light: #F7F1FF
- purple_panel: #EBDDFB
- accent: #E8750B
- orange_light: #FFF8ED
- orange_panel: #FFF0CA
- teal: #078780
- teal_light: #EFFAF9
- secondary_accent: #4058A8
- indigo_light: #F1F4FD
- red: #C74A52
- red_light: #FFF0F1
- border: #BDD1DF

## typography
- font_family: Arial, 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif
- code_family: Consolas, 'Noto Sans CJK SC', monospace
- body: 28
- title: 58
- subtitle: 32
- section: 40
- subheading: 30
- annotation: 24
- footnote: 22
- micro: 20

## output
- formula_policy: text-only
- delivery: SVG only

## page_rhythm
- P01: dense

## geometry
- margin: 28
- section_gap: 16
- module_radius: 10
- section_radius: 14
- section_header_height: 56
- stroke: 2
- arrow_stroke: 3
- section_1: y174 h350
- section_2: y540 h410
- section_3: y966 h380
- section_4: y1362 h1350
- section_5: y2728 h440
- section_6: y3184 h262

## icons
- library: none
- inventory: native SVG semantic diagrams only

## forbidden
- Rasterization of diagram text or formulas
- External runtime image dependencies
- Invented network modules or research results
- Future ground truth used as inference observation
- Diffusion noise-prediction substituted for flow velocity
- style, class, foreignObject, script, filter, mask, symbol, use
- Group opacity

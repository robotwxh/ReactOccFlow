# reactoccflow_overall_flow_a4_20261031

- Canvas format: a4
- Created: 20261031

## Architecture documentation update: 2026-10-31

Markdown source facts, layout specifications and speaker notes now describe the
standard DiT model: 1,200 occupancy tokens, eight 256-dimensional blocks with
eight attention heads, and two actor interaction modules after block 4. The
model uses modality-internal self-attention and bidirectional cross-attention
between occupancy and actor tokens, with one shared occupancy VAE.

**The existing SVG/rendered/exported artifacts were generated for the earlier
U-Net architecture and have not been regenerated in this Markdown update.**
They must not be presented as diagrams of the current DiT. Regeneration should
use the updated `sources/source_intake.md`, `design_spec.md`, `spec_lock.md` and
`notes/` documents. Original export/source-hash records remain historical evidence
for the earlier artifacts. Current code and acceptance are described in the
[DiT migration record](../../implementation/2026-10-31-standard-dit.md).

## Directories

- `svg_output/`: raw SVG output
- `svg_final/`: finalized SVG output
- `images/`: runtime image pool; converter assets keep their original short filenames when possible
- `icons/`: project icon set — selected library icons copied in (via icon_sync.py) plus any custom icons you add; embedded from here at export
- `notes/`: speaker notes
- `templates/`: project templates
- `live_preview/`: browser preview runtime files and history (lock.json, server.log, edits.jsonl, annotations.jsonl)
- `sources/`: source materials and normalized markdown
- `analysis/`: machine-extracted intermediate analysis (PPTX intake, image_analysis.csv) — the pipeline's canonical must-read source/asset facts
- `exports/`: main native pptx (timestamped); `_svg.pptx` sibling added with `--svg-snapshot`, `_native_charts.pptx` name with `--native-objects`
- `backup/<timestamp>/`: svg_output/ archive (always written in default-flow mode; safe to delete old timestamps)

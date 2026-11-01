# Third-party source and resource notices

## OccFM

The occupancy codec, conditional-flow network, sampler/training formulas and compatibility data/metric adapters were copied from the local OccFM reference at commit `64959840a9a4cb54d5b0f6cd4bc6779bb242a853` before adaptation. Original file paths and SHA-256 checksums are recorded in `docs/provenance/model_sources.json`, `data_sources.json` and `evaluation_sources.json`.

The inspected reference has no root LICENSE file. This project does not infer or grant public redistribution rights for OccFM-derived code or weights. Resolve upstream terms before public release. Local research migration does not change upstream copyright ownership.

Copied Swin source files retain their Microsoft copyright and MIT license headers. Other file-level notices remain in their respective files. Source-only snapshots under `reactoccflow/models/source_snapshots/` and `reactoccflow/training/source_snapshots/` are audit material, never imported. The optimizer snapshot is recorded in `docs/provenance/joint_training_sources.json`; the copied common-utils snapshot and adapted deterministic seeding policy are recorded in `docs/provenance/deterministic_training_sources.json`. These snapshots have no runtime role and remain subject to the unresolved OccFM terms above.

Pretrained resources were copied into independent ordinary files. Their original paths, sizes and content hashes are in `docs/provenance/resources.json`. Dataset and pretrained model access remain subject to the respective upstream terms.

The standard isotropic DiT added on 2026-10-31 reuses the independently migrated OccFM `DiTBlock`, attention, MLP and position/time/trajectory embedding dependency closure. Its source snapshots and adaptation details are in `docs/provenance/standard_dit_sources.json`. Project-owned patchification, output layer and actor integration do not remove the upstream licensing restrictions on reused components. No DiT image-generation weights or external DiT repository were imported.

At the user's later request, the copied U-Net world backbone, its wrapper and dedicated runtime commands/configurations were removed. The remaining VAE, DiT blocks and shared mathematical/diagnostic utilities retain their upstream attribution and restrictions. Removal inventory, pre-removal hashes and the helper relocation are recorded in `docs/provenance/remove_unet_sources.json`; earlier source manifests describe historical copies and are not declarations that removed files remain executable.

Academic reference: Tianran Liu, Shengwen Zhao, Nicholas Rhinehart, *Towards foundational LiDAR world models with efficient latent flow matching*, arXiv:2506.23434. Full references: `ReactOccFlow参考文献.md`.

## IR-WM

Trajectory traversal, coordinate/grid conversion and official split selection were copied from IR-WM at commit `94dbbf44d6c22e50db4080e986b3675eadbb4497` before adaptation into independent NumPy research-label components. Original snapshots and file notices are retained under `reactoccflow/geometry/source_snapshots/` and `reactoccflow/data/source_snapshots/`; checksums and adaptations are recorded in `docs/provenance/research_sources.json` and `split_sources.json`. The root Apache-2.0 license is copied verbatim to `docs/licenses/IR-WM-APACHE-2.0.txt`. File-level upstream OpenMMLab, Junyi Ma/OpenOccupancy/Cam4DOcc and other notices remain applicable.

Official nuScenes train/val scene-name literals were extracted without executing an installed devkit; the source SHA-256 and Holger Caesar notice are retained in `docs/provenance/nuscenes_official_splits.json`. Dataset use remains subject to the nuScenes terms. No reference directory is a runtime dependency.

## New infrastructure and dependencies

The resource verification helper and CLI orchestration are project-owned additions after searching both references for the required functionality. The data adapter and occupancy metric are explicit adaptations of the copied OccFM source, not independent clean-room implementations. Runtime dependencies are declared in `pyproject.toml`; `requirements-lock.txt` records the installed environment. No license for the complete new project is assigned by this notice.

T7 consistency-source audit snapshots are retained under `reactoccflow/losses/source_snapshots/`; source ownership, file notices, copy hashes, reuse limitations and unresolved terms are recorded in `docs/provenance/consistency_sources.json`. These snapshots are not runtime imports.

## T8–T11 project-owned evaluation infrastructure

Simulator branch contracts, candidate planning, reaction metrics, risk geometry and calibration helpers were implemented independently after read-only searches of both reference repositories; no reference implementation was copied into these modules. Their provenance and search checksums are recorded in `docs/provenance/branch_collection_sources.json`, `planflow_sources.json` and `reaction_risk_sources.json`. These modules do not include CARLA, simulator results or learned performance claims.

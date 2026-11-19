# Source searches before implementation

Date: 2026-10-27. Both references remain read-only.

## 2026-11-19 restore first-version mask defaults

Read-only searches of OccFM's dataset/cache and IR-WM's occupancy loader for
mask_lidar, history_visibility, visibility_key and semantics found no equivalent
project-specific default/mode binding. Reuse the original owned visibility
intersection from the completed mask experiment; restore default API and CLI
selection independently. Future targets and checkpoint identities remain
unchanged. No reference code copied or executed. Sources and SHA-256:
`mask_v1_restore_sources_20261119.json`.

## 2026-11-18 new-mask training and three-way comparison

Read-only searches inspected both references' cache, EMA, occupancy metric and
dataset/loader code. Neither supplies identity-bound recoding of history while
reusing unchanged E0 targets or this three-way fixed-step pipeline. Reuse the
owned encoder/EMA exporter and rank/count/GT/protocol validators; implement only
independent target-reuse and report glue. The evaluator audit admits exactly the
declared observed-history block and verifies unchanged scoring/generation ASTs.
No model/loss/optimizer changes. Sources, frozen identities and GPU smoke exits:
`ground_mask_training_sources_20261118.json`.

## 2026-11-18 observed-only masked ground history

Read-only searches in both references for `mask_lidar`, `temporal_ground`,
`ground_completion`, `inpaint`, `hole.fill` and `binary_closing`, including
OccFM's dataset/cache and IR-WM's occupancy-loading pipeline, found no causal
visible-only E0 ground completion. Reuse the already-owned warp/VAE closure;
implement the shared cache/online completion and training-split diagnostic in
the independent project. Only previously observed ground labels may fill
currently hidden cells; hidden semantic labels and future frames cannot supply
completion values. Source commits, SHA-256 and protection snapshots:
`masked_ground_sources_20261118.json`.

## 2026-11-17 matched history-input experiment

Read-only searches inspected OccFM training/EMA checkpoint logic and count-based
evaluation, plus IR-WM's EMA hook and persistent-worker dataset builder. Neither
reference supplies the independent persistent-loader shutdown or matched-history
fixed-step export/controller protocol. Reuse owned EMA and architecture-report
auditors; implement only project-specific lifecycle/orchestration glue. No new
reference runtime dependencies. Commits, source hashes, reuse decisions and GPU
smoke results: `history_input_comparison_sources_20261117.json`.

## 2026-11-16 dense semantic history

Read-only searches inspected OccFM's `forecast/datasets/nuscenes_dataset.py`
and `tools/cache_vae.py`, plus IR-WM's occupancy-loading pipeline and both
repositories for `history_visibility`, `input_visibility_policy`,
`settings_sha256`, and `dense_semantic`. OccFM directly loads semantic labels
without LiDAR visibility filtering. The necessary semantic/VAE/3-D warp closure
is already copied into the owned package; its dense-known path was previously
used for future targets. Reuse that path for observed history. Neither reference
provides the project's mode-bound cache/checkpoint protocol, so implement its
small shared resolver/mask helpers independently. No new reference imports or
source dependencies are introduced. Source commits/checksums and decisions:
`dense_history_sources_20261116.json`.

| Function | Searches and inspected sources | Reuse decision |
|---|---|---|
| VAE and semantic embedding | OccFM compressors/occfm_vae.py, modules embedding/encoders/decoders/quantization; IR-WM model tree | Copy OccFM dependency closure before adaptation; exact files in model_sources.json |
| Conditional FM, CFG, ODE and training path | OccFM worldmodels/occfm.py and transition_models/FlowMatchingV2.py; IR-WM forecasting models | Copy source formulas and layers; local-generator and future-only wrapper in own package |
| Latent cache and frame indexing | OccFM datasets/nuscenes_dataset.py and tools/cache_vae.py; IR-WM datasets and tools | Copy dataset source before adapting to explicit metadata/NumPy shards |
| Occupancy IoU | OccFM utils/eval_utils.py; IR-WM metric_stp3.py and planning_metrics.py | Copy OccFM evaluation source before extracting standard/legacy counting |
| Verified ordinary-file migration and source protection | rg for sha256, copy2, copyfile, is_symlink, samefile, hashlib across both repositories | No complete verified migration/audit helper found; add standard-library provenance infrastructure |
| Independent cache inspection and installation diagnostics | Both reference tools trees, existing train/eval commands | No matching independent-package CLI; add orchestration around copied runtime components |
| Distributed training orchestration | OccFM tools/train.py, train_utils and optimization; IR-WM tools/test.py and MMDistributedDataParallel entrypoint | Implement independent DDP runner and optimizer/scheduler/EMA helpers in project package; full optimization waits for complete E0 train/tune caches |
| Isolated installed execution | Both repositories' packaging/tools and path handling | No equivalent source-independent acceptance utility; add verification script; never delete or rename reference repos |

## 2026-10-28 formal ReactFlow training runner search

Read-only searches covered `OccFM/tools/train.py`, `tools/train_utils/train_utils.py`,
`tools/train_utils/optimization.py` and `forecast/datasets/__init__.py`, plus
`IR-WM/projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py` and its EMA hook.
OccFM provides the closest native DDP, optimizer, warmup/cosine and EMA patterns;
its train loop snapshot was already present and the optimization source was copied
verbatim into `reactoccflow/training/source_snapshots/` before adaptation. IR-WM
delegates to MMDetection3D/MMCV and is not a suitable dependency closure. Source
commits, SHA-256 values and reuse decisions are in `joint_training_sources.json`.
Neither reference was executed or modified.

### 2026-10-28 E0 settings contract integration

Read-only Python searches in both OccFM and IR-WM for `settings_sha256`,
`anchor_identity_sha256`, `validate_cache_pair` and canonical JSON fingerprint
helpers found no matching E0 settings protocol. The owned cache generator already
defines its digest as SHA-256 of canonical inner settings, stored inside a JSON
wrapper. The formal runner now validates that inner digest against both wrapper
and manifest; it does not hash the serialized wrapper file. A generator-format
regression reproduced the rejection before the correction, and tamper cases cover
both inner settings and the wrapper digest. No reference source was copied for
this project-specific protocol correction.

### 2026-10-29 formal training diagnostics and restart

Read-only searches covered `max_samples`, `stop_after`, `Subset`, `max_iter`,
`val_loss` and `loss_dict` in both reference training tool/API trees. Their
training frameworks do not provide the independent complete-cache subset and
configuration-preserving pause contract. Project-owned runner orchestration now
selects explicit diagnostic subsets using PyTorch Subset without modifying cache
manifests, checkpoints a controlled global-step pause, and respects a restored
early-stop state. Incomplete accumulation tails are dropped after epoch shuffle
so every optimizer update retains the declared effective batch. Existing copied
training patterns remain attributed in joint_training_sources.json; no new
reference code or runtime dependency was needed for these controls.

### 2026-10-29 CUDA deterministic continuation

Searched both references for `cudnn.deterministic`, `use_deterministic_algorithms`,
`CUBLAS_WORKSPACE_CONFIG`, `cudnn.benchmark` and seed helpers. OccFM's
`forecast/utils/common_utils.py:set_random_seed` supplies deterministic cuDNN and
disabled benchmarking; its entire ordinary-file snapshot was copied to
`reactoccflow/training/source_snapshots/occfm_common_utils.py.txt` before adapting
the runner policy. Existing standard-library/NumPy/PyTorch dependencies cover the
selected helper. IR-WM delegates seeding to MMCV and can enable benchmarking.
The owned runner extends OccFM's policy with strict deterministic algorithms and
a BLAS workspace configuration set before CUDA initialization, recording all
settings. Initial CUDA comparison failed before the pause at step2 despite equal
first loss/RNG; failed evidence is retained. This change requires fresh runs, not
loading the earlier numerically unbound diagnostic checkpoints.

Actor/geometry labels and scene splits are now being implemented; risk calibration and simulator components remain pending their source searches.

## Continuation 2026-10-28

| Function | Searches and inspected sources | Reuse decision |
|---|---|---|
| Dataset evaluation and restart state | OccFM tools/test.py val_model and forecast/utils/eval_utils.py; IR-WM planning metrics | Copied source test.py before extending independent dataset counter aggregation; explicit masks and per-window RNG |
| Official scene split | OccFM datasets/nuscenes_dataset.py uses supplied train/val caches; IR-WM tools/data_converter/nuscenes_converter.py uses official devkit splits | Copied converter snapshot, extract official scene literals as JSON; no source runtime imports |
| Stable 80/10/10 subgrouping and training observation normalization | Search hash/calib/split and statistics in both references | No complete equivalent; add deterministic whole-scene SHA-256 grouping and masked train-only streaming moments |

## 2026-10-28 evaluation resume fixes

Searched both reference tools/evaluation/helper trees for `sha256`,
`peak_memory`, `max_memory_allocated`, device identity and resumable evaluation.
Inspected OccFM `tools/test.py`, `forecast/utils/eval_utils.py`, train/test entry
points and IR-WM `tools/test.py`, `tools/train.py` plus its EMA resume hook. Neither
reference supplies payload-fingerprinted resumable occupancy counts or persisted
per-run memory/timing metadata. Added these small orchestration functions in
`reactoccflow/evaluation/baseline.py`, reusing the existing project-owned streaming
SHA-256 helper. No new reference code was copied; existing evaluation source
commits/checksums remain in `evaluation_sources.json`. Metric state protocol v2
intentionally rejects older states that cannot certify payload/hardware identity.
| E0 label QA render | Both reference visual/plot/render file/function searches; IR-WM bevformer/dense_heads/visualization.py and models/utils/visual.py | Copy lightweight BEV plot snapshot before semantic projection/box overlay adaptation; no existing complete 20-scene semantic/actor QA, add project orchestration |
| Real E0 model/decoder acceptance orchestration | Both reference train/test/cache tools inspected; existing owned wrappers reused | Add project-only check_react_flow.py and check_consistency.py around migrated/new APIs; they are diagnostic checks, not paper evaluations |

### 2026-10-28 installed P3 isolation verification

Searched both read-only reference trees for `isolation|site-packages|sys.flags.isolated|independent.*install` in Python/Markdown; no usable installed-package verification implementation. Extended the existing owned `scripts/verify_isolation.py` verification harness only: independently copied one E0 numeric fixture, default partial-cache refusal, strict204-key pretrained trunk loading plus74 new branch keys, joint endpoint/continuous decoder/consistency backprop, frozen codec/trunk, and future-free observed-state inference candidate. This is acceptance infrastructure, not model source or simulator evidence. No reference imports, execution or changes.

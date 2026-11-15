# ReactOccFlow development rules

- Latest user instruction on 2026-11-15: revert all project changes introduced after the raw-LiDAR/synchronized occupancy-actor request. The active route is restored to semantic occupancy input, the original frozen OccFM VAE, cached occupancy latents and annotation-derived actor states/history MLP. The raw-only dataset and no-pretrained-weight restrictions from 2026-11-11/14 are revoked for this rollback. Preserve both U-shaped CFM and repaired standard DiT512.

- All training remains paused by the user. Do not start/resume training, preprocessing, training benchmarks or automatic stage continuation without a new explicit request. Keep `outputs/raw_full_training/2026-11-14-scratch/STOP_REQUESTED.json` in place. Raw-training weights/results are archived historical experiments; do not initialize the restored route from them. Rollback record: `docs/implementation/2026-11-15-rollback-raw-input.md`.

- `OccFM/` and `IR-WM/` are read-only reference repositories. Do not modify, execute scripts in, install into, or generate bytecode/logs in either directory.
- Before implementing a function, search both reference repositories. If suitable code exists, copy the necessary files or functions and their dependency closure into `reactoccflow/` before adapting them. Record source commits and SHA-256 checksums in `docs/provenance/`.
- If neither project supplies the function, record the search and implement it in the independent package.
- Never use symlinks, hardlinks, reference-directory imports, editable installs of reference projects, sys.path/PYTHONPATH injection, configuration inheritance or resource paths into reference directories.
- Reference snapshots and provenance may name the source paths as documentation only.
- Work only in the project-owned package, configurations, tests, scripts and resource copies. Never delete, rename, reset or clean the reference repositories.
- Preserve original copyright/license headers. Treat unresolved upstream licensing explicitly in THIRD_PARTY_NOTICES.md.
- Keep future targets outside inference inputs. Use observed information to select actors.
- Validate source protection and isolated package execution before marking migration complete.
- Use the dedicated environment `/home/lthpc/wxh/envs/ReactOccFlow`; do not modify existing reference-project environments.
- The authoritative implementation plan is `docs/superpowers/plans/2026-10-27-reactoccflow-implementation-plan.md`.

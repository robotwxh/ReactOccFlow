# ReactOccFlow development rules

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

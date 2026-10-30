# Changelog

## 0.1.0 — public preview (2026-10-30)

- README showcase with a 30-second synthetic living-room comparison, a curated cover and public case notes.
- Public modeling workflow in Markdown and standalone HTML, with current task, resource and recovery semantics.
- Rolling refinement: default three object tasks, configurable one to four, persistent packages and serial scene saves.
- Shared model/Blender admission, process ownership, recoverable pre-execution failures and bounded connection retries.
- Expanded Python, retry/transport and browser regression checks; source exports validate local HTML dependencies.

### Initial release preparation

- Personal-project README in Chinese and English, installation/configuration guides and workflow documentation.
- Original synthetic tabletop demo using the scene repository, candidate validation and Blender/GLB export.
- `demo`, `check` and `workbench` CLI entry points.
- Local TOML/environment configuration for Blender, models and GPU worker paths.
- Per-project model settings preserved across restart and refinement.
- Dedicated ViPE workbench launcher, static resource declarations and explicit Node dependencies.
- Apache-2.0 project license, third-party notices, contributor guidance and source export allowlist.
- Offline regression CI definition, portable host fixtures and focused release/configuration checks.

See [validation scope](docs/RELEASE_VALIDATION.md) and [release procedure](docs/RELEASING.md).

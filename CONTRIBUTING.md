# Contributing

Small, reproducible improvements to installation, scene editing, validation and model adapters are welcome. The supported first-use path is `video2scene demo`; the complete workflow starts at `video2scene workbench`.

## Development setup

```shell
python -m pip install -e '.[dev]'
npm ci --prefix tools/pi-runtime --ignore-scripts --no-audit --no-fund
python -X utf8 -m pytest -q
node --test tests/pi_workflow/transport.test.mjs tests/pi_workflow/retry.test.mjs
node tests/editor_v1/test_state.cjs
python scripts/prepare_release.py
```

These tests do not call real models. Historical-data tests and Blender checks may skip without their optional prerequisites. The source export omits a legacy sample-specific evaluation test whose external experiment driver is not shipped. The maintained editor/host tests generate their own synthetic fixtures.

Browser checks use their own development dependency:

```shell
npm ci --prefix tests --ignore-scripts --no-audit --no-fund
npm exec --prefix tests -- playwright install chromium
node tests/pi_workflow/test_web.cjs
node tests/demo_browser.cjs
```

On Linux, Playwright may require OS browser dependencies; use its documented installation procedure. The workbench browser test uses isolated HTTP fixtures. The demo test runs a real synthetic scene and Blender, uses `.venv` by default, and accepts `V2S_TEST_PYTHON` / `V2S_BLENDER` overrides. CI runs the workbench fixture test; the real Blender demo is a separate local check. Live model calls and GPU inference should not run in ordinary CI.

## Changes and pull requests

- Describe the user-visible problem, the change, and checks actually run.
- Keep world/object coordinate conventions and source frame IDs explicit.
- Preserve candidate hashes, authorization checks, version conflict handling and interrupted-call records.
- Add focused regression coverage for behavior changes; use synthetic inputs for ordinary tests.
- Record third-party provenance and keep licenses when copying code or adding model dependencies.
- Update the public docs and `release-manifest.json` when adding a required runtime file.

Do not commit `video2scene.local.toml`, credential caches, model weights, private videos, raw conversations or `outputs/`. Issue attachments should be minimal and redacted. [Security notes](SECURITY.md)

Original contributions are accepted under the project's Apache-2.0 license. Submit only material you have the right to contribute. [Third-party notices](THIRD_PARTY_NOTICES.md)

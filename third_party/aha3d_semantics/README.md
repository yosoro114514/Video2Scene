# AHa-3D scene semantics (vendored subset)

Upstream: https://github.com/KevinXu02/aha-3d
Pinned commit: `9b80f78bd8ee293d4da14c86f307a73584aa3169` (inspected 2026-10-28).
`semantics.py` is copied unchanged from `src/aha3d/blender/semantics.py` under the included Apache-2.0 license.

This prototype calls `tag_root`, `tag_support`, `roots`, and `descendants`. It does not call `export_semantics`, which depends on the upstream orientation module not included here. No model weights or third-party assets from AHa-3D are bundled.

Our revisioned scene JSON is authoritative. The Blender custom properties written by this adapter are a compiled representation, a deliberate difference from upstream's Blender-authoritative convention. Browser edits, transactions, rendering, and checks are implemented locally, inspired by AHa-3D's object inspection and asset replacement workflow. This is not a reproduction of its whole pipeline.

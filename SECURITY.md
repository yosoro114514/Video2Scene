# Local execution and data

Video2Scene is a personal local tool. Its HTTP services bind to loopback by default. They do not provide a multi-user account system or a public hosting boundary.

## Data flow

- The synthetic demo uses generated geometry and a local reference render. It reads no model credentials and makes no model calls.
- The video workbench stores source references, selected frames, tool inputs/results, scene revisions, prompts and model responses under its configured output root.
- Model operations send selected images and text to the configured model service through Pi/Codex. Account policies and service terms apply.
- Object authoring executes model-authored Blender code through the configured Codex execution boundary. Treat that workflow as code execution on your machine.
- The Pi adapter reads an existing file-backed Codex login in memory. It does not write/refresh the shared login. Diagnostic commands check file presence without printing or copying credentials.

## Sharing and reports

Keep credential caches, local config, user videos and raw runtime logs out of public releases. The source allowlist and targeted scanner help check exports, but they do not establish that arbitrary user data is safe to publish.

For a suspected security issue, use the repository's private vulnerability reporting channel if enabled. If it is unavailable, open a minimal issue requesting a private contact without including exploit details, tokens or private data. This local release draft has no invented contact address or support guarantee.

For ordinary bugs, include versions, a minimal synthetic input, and redacted error text. [Contributing](CONTRIBUTING.md)

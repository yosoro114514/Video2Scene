"""Remove an unavailable nightly dependency from our short-video-only build.

At the pinned ViPE revision, gtsam is imported lazily solely by the long-sequence
pose graph. We do not expose that pipeline. No reconstruction code is changed.
"""

import sys
from pathlib import Path

path = Path(sys.argv[1]) / "pyproject.toml"
text = path.read_text()
old = '    "gtsam-develop==4.3a2.dev202609030306",\n'
new = "    # video2scene: unused long-sequence nightly dependency omitted.\n"
if old in text:
    path.write_text(text.replace(old, new))
    print("Omitted unavailable gtsam nightly; short-video ViPE pipeline unchanged.")
elif new not in text:
    raise SystemExit(
        "Unexpected ViPE metadata; review dependency patch before installing."
    )

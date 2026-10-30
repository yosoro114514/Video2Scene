"""Owned subprocess boundary for cancellable calls through the existing Codex login."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video2scene.codex_review import invoke


def main():
    path = Path(sys.argv[1])
    spec = json.loads(path.read_text(encoding='utf-8'))
    result = invoke(**spec)
    (path.parent / 'invoke_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

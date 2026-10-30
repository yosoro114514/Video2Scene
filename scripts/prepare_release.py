"""Export a reviewed source allowlist without touching the development tree.

The scan reports rule names and line numbers, never matching credential text.
It is a targeted release check, not a complete security or licensing audit.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import stat
import zipfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.py', '.md', '.toml', '.json', '.yml', '.yaml', '.js', '.mjs', '.cjs', '.html', '.css', '.sh', '.ps1', '.lock'}
RULES = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'openai_key': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'huggingface_token': re.compile(r'\bhf_[A-Za-z0-9]{30,}\b'),
    'jwt': re.compile(r'\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\b'),
    'developer_path': re.compile(r'(?:[A-Za-z]:[/\\]+Users[/\\]+wmc|/home[/]wmc|/mnt/[a-z]/personal_PRO|D:[/]Applications|[A-Za-z]:[/\\]+personal_PRO[/\\]+VIPE_gpt)', re.I),
}


def release_bytes(path):
    """Use the same LF text bytes in the source ZIP, Git blobs and hash table."""
    data = path.read_bytes()
    if path.suffix in TEXT_SUFFIXES or path.name in {'LICENSE', 'NOTICE', '.gitignore', '.gitattributes'}:
        return data.replace(b'\r\n', b'\n')
    return data


class DocumentLinks(HTMLParser):
    """Local public HTML must work without the private development output tree."""
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for key, value in attrs if key in {'href', 'src', 'poster'} and value)


def files_for_release(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'release-manifest.json').read_text(encoding='utf-8'))
    selected = set()
    for pattern in manifest['include']:
        found = [p for p in root.glob(pattern) if p.is_file()]
        if not found:
            raise ValueError('Required release entry missing: ' + pattern)
        selected.update(found)
    result = []
    for path in sorted(selected):
        relative = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in manifest['exclude']):
            continue
        if (set(part.lower() for part in path.relative_to(root).parts) & {'node_modules', '__pycache__', '.git', '.venv', '.codex', '.pi', 'ccfa-workfiles', 'outputs', 'data'}
                or path.name.lower() in {'auth.json', 'video2scene.local.toml', '.env'}
                or path.suffix.lower() in {'.pt', '.pth', '.safetensors', '.pem', '.key', '.sqlite'}):
            raise ValueError('Private/runtime file cannot enter source release: ' + relative)
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root and p.is_relative_to(root)):
            raise ValueError('Release symlinks are not supported: ' + relative)
        if not path.resolve().is_relative_to(root):
            raise ValueError('Release path escapes source tree: ' + relative)
        if path.stat().st_size > manifest['max_file_bytes']:
            raise ValueError('Release file exceeds size budget: ' + relative)
        result.append(path)
    return result


def check_release(root=ROOT):
    root = Path(root).resolve()
    paths = files_for_release(root)
    selected = {p.relative_to(root).as_posix() for p in paths}
    findings = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if path.suffix not in TEXT_SUFFIXES:
            continue
        value = path.read_text(encoding='utf-8-sig')
        for rule, regex in RULES.items():
            for match in regex.finditer(value):
                findings.append({'path': relative, 'line': value.count('\n', 0, match.start()) + 1, 'rule': rule})
        targets = []
        if path.suffix == '.md':
            targets = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', value)
        elif path.suffix == '.html' and relative.startswith('docs/'):
            parser = DocumentLinks(); parser.feed(value); targets = parser.targets
        # App HTML routes are resolved by the server; public document links are files.
        for target in targets:
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = unquote(parsed.path)
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(root) or resolved.relative_to(root).as_posix() not in selected:
                findings.append({'path': relative, 'rule': 'missing_release_link', 'target': target})
    return {'ok': not findings, 'file_count': len(paths), 'bytes': sum(p.stat().st_size for p in paths), 'findings': findings}, paths


def export_release(destination, root=ROOT):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or root.is_relative_to(destination):
        raise ValueError('Export destination must not contain the source tree')
    report, paths = check_release(root)
    if not report['ok']:
        raise ValueError('Release checks failed: ' + json.dumps(report['findings'], ensure_ascii=False))
    # Neither an existing output nor an existing ZIP is overwritten.
    archive = destination.with_name(destination.name + '.zip')
    if destination.exists() or archive.exists():
        raise FileExistsError('Choose a new export directory and ZIP name')
    destination.mkdir(parents=True)
    hashes = {}
    for source in paths:
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(release_bytes(source))
        hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / 'RELEASE_FILES.json').write_bytes((json.dumps(hashes, indent=2) + '\n').encode('utf-8'))
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(p for p in destination.rglob('*') if p.is_file()):
            relative = path.relative_to(destination).as_posix()
            info = zipfile.ZipInfo(destination.name + '/' + relative, date_time=(2026, 10, 30, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | (0o755 if path.suffix == '.sh' else 0o644)) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(info, path.read_bytes())
    return {**report, 'directory': str(destination), 'archive': str(archive),
            'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Create a fresh source directory and ZIP; omitted means check only')
    args = parser.parse_args()
    report = export_release(args.output) if args.output else check_release()[0]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

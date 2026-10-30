import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location('source_release', Path(__file__).resolve().parents[1] / 'scripts/prepare_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def source(tmp_path, text='Original project\n'):
    root = tmp_path / 'source';root.mkdir()
    (root / 'README.md').write_text(text, encoding='utf-8')
    (root / 'release-manifest.json').write_text(json.dumps({'include': ['README.md', 'release-manifest.json'],
        'exclude': [], 'max_file_bytes': 100000}), encoding='utf-8')
    return root


def test_secret_detection_reports_location_without_value(tmp_path):
    value = 'sk-' + 'x' * 40
    root = source(tmp_path, value)
    report, _ = release.check_release(root)
    assert not report['ok']
    assert report['findings'][0]['rule'] == 'openai_key'
    assert value not in json.dumps(report)
    with pytest.raises(ValueError, match='Release checks failed'):
        release.export_release(tmp_path / 'export', root)
    assert not (tmp_path / 'export').exists()


def test_export_is_complete_and_refuses_overwrite(tmp_path):
    root = source(tmp_path)
    report = release.export_release(tmp_path / 'video2scene-0.1.0', root)
    assert Path(report['archive']).name == 'video2scene-0.1.0.zip'
    with zipfile.ZipFile(report['archive']) as archive:
        assert 'video2scene-0.1.0/RELEASE_FILES.json' in archive.namelist()
    hashes = json.loads((Path(report['directory']) / 'RELEASE_FILES.json').read_text(encoding='utf-8'))
    assert 'README.md' in hashes
    with pytest.raises(FileExistsError):
        release.export_release(Path(report['directory']), root)


def test_accidental_allowlist_of_credentials_is_rejected(tmp_path):
    root = source(tmp_path)
    (root / 'auth.json').write_text('{}', encoding='utf-8')
    path = root / 'release-manifest.json'
    manifest = json.loads(path.read_text());manifest['include'].append('auth.json')
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='Private/runtime'):
        release.check_release(root)


def test_document_link_must_be_in_the_export(tmp_path):
    root = source(tmp_path, '[Installation](private.md)\n')
    (root / 'private.md').write_text('Not included', encoding='utf-8')
    report, _ = release.check_release(root)
    assert report['findings'][0]['rule'] == 'missing_release_link'


def test_public_html_cannot_depend_on_private_runtime_files(tmp_path):
    root = source(tmp_path)
    (root / 'docs').mkdir()
    (root / 'docs/workflow.html').write_text('<img src="../outputs/private.png"><a href="https://example.org">web</a>', encoding='utf-8')
    path = root / 'release-manifest.json'
    manifest = json.loads(path.read_text()); manifest['include'].append('docs/workflow.html')
    path.write_text(json.dumps(manifest), encoding='utf-8')
    report, _ = release.check_release(root)
    assert report['findings'] == [{'path': 'docs/workflow.html', 'rule': 'missing_release_link', 'target': '../outputs/private.png'}]


def test_public_html_resolves_escaped_files_and_allows_fragments(tmp_path):
    root = source(tmp_path)
    (root / 'docs').mkdir()
    (root / 'docs/workflow.html').write_text('<a href="../README.md#demo">read</a><img src="frame%20one.png"><a href="#top">top</a>', encoding='utf-8')
    (root / 'docs/frame one.png').write_bytes(b'fixture')
    path = root / 'release-manifest.json'
    manifest = json.loads(path.read_text()); manifest['include'].append('docs/*')
    path.write_text(json.dumps(manifest), encoding='utf-8')
    assert release.check_release(root)[0]['ok']


def test_export_hashes_match_canonical_git_line_endings(tmp_path):
    import hashlib
    root = source(tmp_path)
    (root / 'README.md').write_bytes(b'First line\r\nSecond line\r\n')
    result = release.export_release(tmp_path / 'export', root)
    exported = Path(result['directory'])
    expected = b'First line\nSecond line\n'
    assert (exported / 'README.md').read_bytes() == expected
    hashes = json.loads((exported / 'RELEASE_FILES.json').read_text())
    assert hashes['README.md'] == hashlib.sha256(expected).hexdigest()

"""Content-addressed capture storage. Only copies into the explicit run root."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from video2scene.contracts_v1 import ContractError, canonical_hash


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    try:
        with path.open('xb') as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ContractError('EVIDENCE_MISMATCH', 'Immutable capture record already exists with different content', details={'name': path.name})


def copy_artifact(source, storage_root, scene_id, role, *, relative_folder='capture/files'):
    source = Path(source)
    if not source.is_file():
        raise ContractError('FRAME_MISMATCH', 'Source artifact is missing', details={'name': source.name})
    digest = file_hash(source)
    suffix = source.suffix.lower()
    media = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.blend': 'application/x-blender',
             '.glb': 'model/gltf-binary', '.npy': 'application/octet-stream'}.get(suffix, 'application/octet-stream')
    key = f'{relative_folder}/{digest}{suffix}'
    reference = {'uri': f'artifact://{scene_id}/{key}', 'sha256': digest, 'media_type': media, 'role': role}
    destination = resolve_artifact(storage_root, reference, verify=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open('xb') as target, source.open('rb') as origin:
            for block in iter(lambda: origin.read(1024 * 1024), b''):
                target.write(block)
    except FileExistsError:
        pass
    if file_hash(destination) != digest:
        raise ContractError('STALE_ARTIFACT', 'Copied artifact hash differs from source')
    return reference


def resolve_artifact(storage_root, reference, *, verify=True):
    parsed = urlsplit(reference['uri'])
    if parsed.scheme != 'artifact' or not parsed.netloc or parsed.query or parsed.fragment:
        raise ContractError('INVALID_ARGUMENT', 'Invalid artifact URI')
    root = Path(storage_root).resolve()
    scene_root = (root / parsed.netloc).resolve()
    path = (scene_root / parsed.path.lstrip('/')).resolve()
    if not scene_root.is_relative_to(root) or scene_root == root or not path.is_relative_to(scene_root):
        raise ContractError('FORBIDDEN_SCOPE', 'Artifact path escapes scene storage')
    if verify and (not path.is_file() or file_hash(path) != reference['sha256']):
        raise ContractError('EVIDENCE_MISMATCH', 'Evidence artifact is missing or has changed', details={'uri': reference['uri']})
    return path


def checked_record(path, hash_key):
    value = read_json(path)
    payload = {key: item for key, item in value.items() if key != hash_key}
    if value.get(hash_key) != canonical_hash(payload):
        raise ContractError('EVIDENCE_MISMATCH', 'Stored evidence record hash is invalid')
    return value

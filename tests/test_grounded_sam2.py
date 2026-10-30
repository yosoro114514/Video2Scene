import numpy as np
import json
from types import SimpleNamespace
import pytest
from video2scene.grounded_sam2 import associate, load_masks, save_masks


def test_cache_rejects_changed_scale_inventory_and_incomplete_frames(tmp_path, monkeypatch):
    from video2scene import grounded_sam2 as backend

    root = tmp_path / 'vipe'
    for folder in ['pose', 'intrinsics']:
        (root / folder).mkdir(parents=True)
        (root / folder / 'sample.npz').write_bytes(b'fixture')
    rgb, depth = root / 'rgb.mp4', root / 'depth.zip'
    rgb.write_bytes(b'rgb')
    depth.write_bytes(b'depth')
    artifacts = SimpleNamespace(root=root, rgb_path=rgb, depth_path=depth, sequence='sample', indices=[5])
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps({'categories': [{'label':'book', 'prompt':'book'}]}))
    calls = []
    def generate(artifacts, categories, cache, manifest, settings):
        calls.append(1)
        save_masks(cache / '000005.npz', {})
    monkeypatch.setattr(backend, '_generate', generate)
    cache = tmp_path / 'cache'
    backend.prepare_masks(artifacts, inventory, cache, device='cpu')
    backend.prepare_masks(artifacts, inventory, cache, device='cpu')
    assert len(calls) == 1
    with pytest.raises(ValueError, match='cache'):
        backend.prepare_masks(artifacts, inventory, cache, device='cpu', scale=2)
    inventory.write_text(json.dumps({'categories':[{'label':'mouse', 'prompt':'mouse'}]}))
    with pytest.raises(ValueError, match='cache'):
        backend.prepare_masks(artifacts, inventory, cache, device='cpu')
    inventory.write_text(json.dumps({'categories':[{'label':'book', 'prompt':'book'}]}))
    (cache / '000005.npz').unlink()
    with pytest.raises(ValueError, match='cache'):
        backend.prepare_masks(artifacts, inventory, cache, device='cpu')


def test_mask_cache_preserves_overlapping_instances_and_empty_frames(tmp_path):
    a = np.zeros((9, 13), bool)
    b = a.copy()
    a[1:6, 1:6], b[3:8, 3:8] = True, True
    path = tmp_path / "masks.npz"
    save_masks(path, {17: a, 42: b})
    result = load_masks(path)
    assert set(result) == {17, 42}
    np.testing.assert_array_equal(result[17], a)
    np.testing.assert_array_equal(result[42], b)
    save_masks(path, {})
    assert load_masks(path) == {}


def test_same_class_objects_keep_distinct_ids_on_reentry():
    xyz = np.random.default_rng(7).uniform(0, 0.1, (100, 3))
    mask = np.ones((10, 10), bool)
    tracks = {
        11: dict(label="mouse", points=xyz),
        22: dict(label="mouse", points=xyz + 1),
    }
    candidates = [
        dict(label="mouse", points=xyz + 1.01, mask=mask),
        dict(label="mouse", points=xyz + 0.01, mask=mask),
    ]
    assert associate(candidates, tracks, {}) == {0: 22, 1: 11}
    # An incorrect 2D overlap must not override a large 3D disagreement.
    assert (
        associate([dict(label="mouse", points=xyz + 3, mask=mask)], tracks, {11: mask})
        == {}
    )


def test_association_is_one_to_one_and_class_aware():
    xyz = np.random.default_rng(2).uniform(0, 0.2, (100, 3))
    mask = np.ones((10, 10), bool)
    tracks = {3: dict(label="book", points=xyz)}
    candidates = [dict(label="book", points=xyz, mask=mask) for _ in range(2)]
    assert len(associate(candidates, tracks, {3: mask})) == 1
    candidates[0]["label"] = "keyboard"
    assert associate(candidates, tracks, {3: mask}) == {1: 3}

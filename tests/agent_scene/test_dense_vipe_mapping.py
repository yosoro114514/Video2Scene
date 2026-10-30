"""Dense camera/depth indices remain exact original-source frame identities."""
import copy
from pathlib import Path

import numpy as np
import pytest

from video2scene.artifacts import Frame
from video2scene.agent_scene import whitebox
from video2scene.agent_scene.planning import save, read
from video2scene.contracts_v1 import file_hash


@pytest.fixture
def dense(tmp_path, monkeypatch):
    observation = tmp_path / 'frames/manifest.json'
    original = {'source_sha256': '1' * 64, 'frames': [
        {'frame_index': i, 'source_frame_index': source, 'sha256': str(i + 2) * 64,
         'source_width': 640, 'source_height': 480, 'source_fps': 30., 'source_time_s': source / 30.}
        for i, source in enumerate([12, 36])]}
    save(observation, original)
    path = tmp_path / 'tools/vipe/dense_input/vipe_manifest.json'
    value = {'sampling_version': 'vipe_dense_v1', 'observation_manifest_path': str(observation),
             'observation_manifest_sha256': file_hash(observation), 'source_sha256': original['source_sha256'], 'frames': []}
    for i, source in enumerate([0, 12, 24, 36, 48]):
        index = {12: 0, 36: 1}.get(source)
        value['frames'].append({'frame_index': i, 'dense_frame_index': i, 'source_frame_index': source,
            'observation_frame_index': index, 'is_observation_frame': index is not None,
            'source_width': 640, 'source_height': 480, 'source_fps': 30., 'source_time_s': source / 30.,
            'observation_rgb_sha256': original['frames'][index]['sha256'] if index is not None else None})
    save(path, value)
    pointer_path = tmp_path / 'tools/vipe/latest.json'
    pointer = {'sampling_version': 'vipe_dense_v1', 'vipe_dir': str(path.parent / 'vipe'),
               'vipe_manifest_path': str(path), 'vipe_manifest_sha256': file_hash(path),
               'observation_manifest_sha256': file_hash(observation)}
    save(pointer_path, pointer)
    calls = []
    class Artifacts:
        indices = [0, 1, 2, 3, 4]
        def __init__(self, root):
            assert str(root) == pointer['vipe_dir']
        def frames(self, *, indices):
            calls.append(list(indices))
            for index in indices:
                yield Frame(index, np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)), np.array([1, 1, 1, 1]), np.eye(4))
    monkeypatch.setattr(whitebox, 'ViPEArtifacts', Artifacts)
    def update(value):
        save(path, value)
        pointer['vipe_manifest_sha256'] = file_hash(path)
        save(pointer_path, pointer)
    return tmp_path, value, pointer, calls, update, Artifacts


def test_dense_selects_only_exact_observation_indices_even_with_legacy_directory(dense):
    run, _, _, calls, _, _ = dense
    (run / 'vipe').mkdir()  # An explicit dense pointer wins over an old sparse directory.
    result = whitebox.frames_by_source(run)
    assert set(result) == {12, 36}
    assert calls == [[1, 3]]
    assert result[12].index == 1 and result[36].index == 3


@pytest.mark.parametrize('problem', ['missing_source', 'duplicate_source', 'duplicate_index', 'dense_index',
    'observation_index', 'observation_flag', 'observation_rgb', 'source_width', 'source_time', 'source_hash', 'observation_hash'])
def test_invalid_dense_mappings_fail_before_depth_decode(dense, problem):
    run, value, _, calls, update, _ = dense
    changed = copy.deepcopy(value)
    row = changed['frames'][1]
    if problem == 'missing_source':
        row.update(source_frame_index=13, observation_frame_index=None, is_observation_frame=False, observation_rgb_sha256=None)
    elif problem == 'duplicate_source': row['source_frame_index'] = 0
    elif problem == 'duplicate_index': row['frame_index'] = 0
    elif problem == 'dense_index': row['dense_frame_index'] = 0
    elif problem == 'observation_index': row['observation_frame_index'] = 1
    elif problem == 'observation_flag': row['is_observation_frame'] = False
    elif problem == 'observation_rgb': row['observation_rgb_sha256'] = 'f' * 64
    elif problem == 'source_width': row['source_width'] = 641
    elif problem == 'source_time': row['source_time_s'] += .1
    elif problem == 'source_hash': changed['source_sha256'] = 'f' * 64
    else: changed['observation_manifest_sha256'] = 'f' * 64
    update(changed)
    with pytest.raises(ValueError):
        whitebox.frames_by_source(run)
    assert calls == []


def test_manifest_hash_and_complete_artifact_index_set_are_checked(dense):
    run, value, pointer, calls, update, artifacts = dense
    save(pointer['vipe_manifest_path'], {**value, 'changed_after_binding': True})
    with pytest.raises(ValueError, match='hash mismatch'):
        whitebox.frames_by_source(run)
    update(value)
    artifacts.indices = [0, 1, 2, 3]
    with pytest.raises(ValueError, match='artifact frame indices'):
        whitebox.frames_by_source(run)
    assert calls == []


def test_duplicate_observation_indices_fail_explicitly(dense):
    run, _, _, _, _, _ = dense
    path = run / 'frames/manifest.json'; value = read(path)
    value['frames'][1]['frame_index'] = 0; save(path, value)
    with pytest.raises(ValueError, match='duplicate'):
        whitebox.frames_by_source(run)


def test_legacy_source_mapping_still_rejects_missing_frames(tmp_path, monkeypatch):
    save(tmp_path / 'frames/manifest.json', {'frames': [
        {'frame_index': 0, 'source_frame_index': 7}, {'frame_index': 1, 'source_frame_index': 9}]})
    class Artifacts:
        def __init__(self, root): pass
        def frames(self):
            yield Frame(0, np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)), np.ones(4), np.eye(4))
    monkeypatch.setattr(whitebox, 'ViPEArtifacts', Artifacts)
    with pytest.raises(ValueError, match='missing required'):
        whitebox.frames_by_source(tmp_path)


def test_old_proposal_bytes_and_point_clouds_are_preserved_before_dense_overwrite(tmp_path):
    proposal = tmp_path / 'geometry_proposals.json'
    save(proposal, {'status': 'proposed', 'floor_fits': [{'p90_residual_m': .12}], 'objects': []})
    original = proposal.read_bytes()
    points = tmp_path / 'geometry/points/chair.npy'; points.parent.mkdir(parents=True)
    np.save(points, np.array([[1., 2., 3.]]))
    old_points = points.read_bytes()
    result = whitebox.archive_geometry_proposals(tmp_path)
    save(proposal, {'status': 'proposed', 'vipe_evidence': {'sampling_version': 'vipe_dense_v1'}, 'objects': []})
    np.save(points, np.array([[3., 2., 1.]]))
    assert Path(result['path']).read_bytes() == original
    record = read(result['record_path'])
    assert record['legacy_origin_unrecorded'] is True and record['vipe_evidence'] is None
    archived_points = next(row for row in record['files'] if row['path'].endswith('chair.npy'))
    assert Path(archived_points['path']).read_bytes() == old_points


def test_actual_cpu_dense_manifest_joins_observation_sources(tmp_path, monkeypatch):
    import cv2
    from video2scene.pi_workflow.video_tools import prepare_video, prepare_vipe_dense_video
    source = tmp_path / 'source.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 30., (80, 60))
    for index in range(30):
        writer.write(np.full((60, 80, 3), index * 5, np.uint8))
    writer.release()
    run = tmp_path / 'project'
    prepare_video(run, source, frame_budget=6, keyframe_count=2)
    destination = run / 'tools/vipe/fixture_dense/prepared'
    value = prepare_vipe_dense_video(run, destination)
    path = destination / 'manifest.json'
    save(run / 'tools/vipe/latest.json', {'sampling_version': value['sampling_version'],
        'vipe_dir': str(destination.parent / 'vipe'), 'vipe_manifest_path': str(path),
        'vipe_manifest_sha256': file_hash(path), 'observation_manifest_sha256': value['observation_manifest_sha256']})
    selected = []
    class Artifacts:
        def __init__(self, root): self.indices = list(range(len(value['frames'])))
        def frames(self, *, indices):
            selected.extend(indices)
            for index in indices:
                yield Frame(index, np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)), np.ones(4), np.eye(4))
    monkeypatch.setattr(whitebox, 'ViPEArtifacts', Artifacts)
    result = whitebox.frames_by_source(run)
    expected = {row['source_frame_index'] for row in read(run / 'frames/manifest.json')['frames']}
    assert set(result) == expected
    assert len(selected) == 6 < len(value['frames'])

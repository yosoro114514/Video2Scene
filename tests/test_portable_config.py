import pytest

from video2scene.config import load_settings


def test_explicit_config_and_environment_override_without_changing_disk(tmp_path):
    path = tmp_path / 'settings.toml'
    path.write_text('[video2scene]\nmodel="from-file"\nsam_checkpoint="/models/checkpoint.pt"\n', encoding='utf-8')
    original = path.read_bytes()
    settings = load_settings(path, environ={'V2S_MODEL': 'from-env'})
    assert settings.model == 'from-env'
    assert settings.sam_checkpoint == '/models/checkpoint.pt'
    assert path.read_bytes() == original


def test_bad_or_missing_explicit_config_fails_before_launch(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / 'missing.toml', environ={})
    path = tmp_path / 'settings.toml'
    path.write_text('[video2scene]\nsam_checkpoint="C:/weights/model.pt"\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Linux path'):
        load_settings(path, environ={})
    path.write_text('[video2scene]\nblender_path="typo"\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Unknown configuration'):
        load_settings(path, environ={})


def test_project_model_is_pinned_across_config_change(tmp_path, monkeypatch):
    from video2scene.pi_workflow.service import WorkflowService
    path = tmp_path / 'config.toml';path.write_text('[video2scene]\n', encoding='utf-8')
    monkeypatch.setenv('V2S_CONFIG', str(path))
    monkeypatch.setenv('V2S_MODEL', 'first-model')
    source = tmp_path / 'fixture.mp4';source.write_bytes(b'test-only, not decoded')
    service = WorkflowService(tmp_path / 'workflow')
    try:
        first = service.create_project({'request_id':'first','source_path':str(source)})
    finally:
        service.close()
    monkeypatch.setenv('V2S_MODEL', 'second-model')
    service = WorkflowService(tmp_path / 'workflow')
    try:
        assert service.action(first['id']).model == 'first-model'
        second = service.create_project({'request_id':'second','source_path':str(source)})
        assert second['model'] == 'second-model'
        assert service.action(second['id']).model == 'second-model'
    finally:
        service.close()

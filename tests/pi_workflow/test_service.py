"""Temporary service integration with explicit mock Pi/GPU/author/build workers.

No production project, browser approval, model credential or GPU is touched.
The repository, evidence selection, human-event store and validators are real.
"""
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
from PIL import Image
import pytest
import trimesh

from video2scene.artifacts import Frame
from video2scene.agent_scene import whitebox
from video2scene.agent_scene.planning import save, read
from video2scene.contracts_v1 import ContractError, canonical_hash, file_hash
from video2scene.pi_workflow import scene_actions, service
from test_scene_actions import prior, layout


class MockPi:
    """Only records transport messages; never starts Node or a model."""
    def __init__(self, config, folder, *, token, on_event):
        self.config, self.folder, self.on_event = config, folder, on_event
        self.process = SimpleNamespace(poll=lambda: None)
        self.sent, self.sent_event = [], threading.Event()
    def start(self):
        value = {'type': 'ready', 'session_file': str(self.folder / 'mock_session.jsonl')}
        self.on_event(value)
        return value
    def send(self, value):
        self.sent.append(value)
        self.on_event({'type': 'settled', 'run_id': value.get('run_id')})
        self.sent_event.set()
    def close(self): pass


class ServiceHarness:
    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path
        monkeypatch.setattr(service, 'PiBridge', MockPi)
        self.service = service.WorkflowService(tmp_path / 'workflow')
        source = tmp_path / 'fixture.mp4'
        source.write_bytes(b'explicit mock video decoder input')
        self.project = self.service.create_project({'request_id': 'create_fixture', 'source_path': str(source)})
        self.pid = self.project['id']
        self.run = self.service.run_root(self.pid)
        self.actions = self.service.action(self.pid)
        self.count = 0
        self.author_calls = []
        self.prior, self.layout = prior(), layout()
        second = copy.deepcopy(self.prior['objects'][0]); second['object_id'] = 'chair_two'
        self.prior['objects'].append(second)
        second_layout = copy.deepcopy(self.layout['objects'][0]); second_layout['object_id'] = 'chair_two'
        second_layout['center_m'][0] = 1.5
        self.layout['objects'].append(second_layout)
        self.sam = self.run / 'tools/sam/mock/output'
        save(self.sam / 'result.json', {'status': 'complete', 'explicit_fixture': True})
        for target in ('chair', 'chair_two'):
            directory = self.sam / target / 'target_masks'; directory.mkdir(parents=True)
            np.savez_compressed(directory / '00007.npz', mask=np.ones((40, 60), dtype=bool), source_frame_index=np.int64(7))

        def prepare(run_root, source_path, **kwargs):
            image = self.run / 'frames/00000.jpg'; image.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (60, 40), '#777777').save(image)
            row = {'frame_index': 0, 'source_frame_index': 7, 'source_time_s': .7, 'source_fps': 10.,
                   'source_width': 120, 'source_height': 80, 'width': 60, 'height': 40,
                   'path': str(image), 'sha256': file_hash(image)}
            save(self.run / 'frames/manifest.json', {'prepared_fps': 10., 'source_sha256': file_hash(source_path), 'frames': [row]})
            return {'status': 'complete', 'keyframes': [row], 'manifest_path': str(self.run / 'frames/manifest.json'),
                    'explicit_mock_decoder': True}
        monkeypatch.setattr(service, 'prepare_video', prepare)
        def mock_dense_vipe(*args, **kwargs):
            observation = self.run / 'frames/manifest.json'
            original = read(observation)
            path = self.run / 'tools/vipe/fixture_dense/vipe_manifest.json'
            save(path, {'sampling_version': 'vipe_dense_v1', 'observation_manifest_path': str(observation),
                'observation_manifest_sha256': file_hash(observation), 'source_sha256': original['source_sha256'],
                'frames': [{**row, 'dense_frame_index': row['frame_index'], 'observation_frame_index': row['frame_index'],
                            'is_observation_frame': True, 'observation_rgb_sha256': row['sha256']} for row in original['frames']],
                'explicit_mock_dense_inference': True})
            pointer = {'vipe_dir': str(path.parent / 'vipe'), 'vipe_manifest_path': str(path),
                       'vipe_manifest_sha256': file_hash(path), 'observation_manifest_sha256': file_hash(observation),
                       'sampling_version': 'vipe_dense_v1'}
            save(self.run / 'tools/vipe/latest.json', pointer)
            return {'status': 'complete', 'explicit_mock_gpu': True, **pointer}
        monkeypatch.setattr(service, 'run_vipe_tool', mock_dense_vipe)
        monkeypatch.setattr(service, 'run_sam_tool', lambda *a, **k: {'status': 'complete', 'sam_root': str(self.sam), 'explicit_mock_gpu': True})
        frame = Frame(0, np.full((40, 60, 3), 120, np.uint8), np.full((40, 60), 3., np.float32),
                      np.array([45., 45., 30., 20.]), np.eye(4))
        monkeypatch.setattr(whitebox, 'frames_by_source', lambda _: {7: frame})
        def measure(root, sam):
            result = {'status': 'proposed', 'scene_from_vipe': np.eye(4).tolist(),
                'objects': [{'object_id': o['object_id'], 'status': 'observed_proposal',
                            'accepted_frame_indices': [7], 'center_m': o['center_m']} for o in self.layout['objects']],
                'prior_sha256': file_hash(root / 'prior.json'), 'sam_result_sha256': file_hash(sam / 'result.json')}
            save(root / 'geometry_proposals.json', result); return result
        monkeypatch.setattr(whitebox, 'propose', measure)
        def artifact(path, role, media='application/json'):
            return {'uri': 'artifact://' + path.relative_to(self.service.runtime).as_posix(),
                    'sha256': file_hash(path), 'media_type': media, 'role': role}
        self.artifact = artifact
        async def mock_export(snapshot, label):
            folder = self.service.runtime / self.pid / 'mock_exports' / canonical_hash(snapshot)[:16]
            folder.mkdir(parents=True, exist_ok=True)
            refs = []
            for filename, role, value in [('reopen.json', 'reopen_report', {'passed': True, 'explicit_mock_build': True}),
                ('result.json', 'build_manifest', {'input_fingerprint': canonical_hash(snapshot), 'explicit_mock_build': True}),
                ('input.json', 'build_input', {'input_fingerprint': canonical_hash(snapshot)})]:
                save(folder / filename, value); refs.append(artifact(folder / filename, role))
            for filename, role in [('scene.blend', 'scene_blend'), ('scene.glb', 'scene_glb')]:
                (folder / filename).write_bytes(b'explicit mock exporter fixture')
                refs.append(artifact(folder / filename, role, 'application/octet-stream'))
            return {'snapshot_hash': canonical_hash(snapshot), 'artifact_refs': refs, 'explicit_mock_export': True}
        monkeypatch.setattr(self.actions, '_export_snapshot', mock_export)

        class MockAuthor:
            def __init__(backend, *args, **kwargs): pass
            async def refine(backend, spec, snapshot, evidence, context):
                target = next(o for o in snapshot['objects'] if o['instance_id'] == spec['target_id'])
                self.author_calls.append(target['instance_id'])
                folder = self.service.runtime / self.pid / 'mock_assets'; folder.mkdir(parents=True, exist_ok=True)
                path = folder / (spec['target_id'] + '.glb')
                low, high = np.array(target['local_bounds_m']['min']), np.array(target['local_bounds_m']['max'])
                mesh = trimesh.creation.box(extents=high - low); mesh.apply_translation((low + high) / 2)
                mesh.visual.vertex_colors = [120, 70, 40, 255]
                path.write_bytes(trimesh.Scene(mesh).export(file_type='glb'))
                asset = copy.deepcopy(target['asset'])
                asset.update(asset_version='explicit_mock_fine_' + file_hash(path)[:20],
                             artifact=artifact(path, 'object_geometry', 'model/gltf-binary'))
                return {'input_fingerprint': spec['candidate']['candidate_hash'], 'artifact_refs': [asset['artifact']],
                        'asset_updates': [{'instance_id': target['instance_id'], 'asset': asset, 'local_bounds_m': target['local_bounds_m']}]}
        self.actions.backend_factory = MockAuthor
        # The actual provider proof verifier has separate live-backend tests.
        # This fixture bypass is explicit and never enabled by production code.
        def mock_proof(backend, build, candidate, *args):
            return {'explicit_mock_provider_proof': True, 'provider_call_receipts': ['fixture:' + candidate['candidate_id']],
                    'critic': {'status': 'accept'}, 'model_calls': 0}
        monkeypatch.setattr(scene_actions, 'collect_run_proof', mock_proof)

    def request(self, name, args=None):
        self.count += 1
        return {'project_id': self.pid, 'run_id': 'mock_pi_run', 'turn_id': self.count,
                'tool_call_id': 'mock_call_' + str(self.count), 'name': name, 'args': args or {}}

    def tool(self, name, args=None):
        payload = self.service.tool(self.request(name, args))
        return json.loads(payload['content'][0]['text']), payload

    def seed(self):
        inspected, response = self.tool('inspect_video')
        self.service._on_pi_event(self.pid, {'type': 'tool_receipt', 'run_id': 'mock_pi_run', 'result': response})
        self.tool('record_prior', {'prior': self.prior, 'observation_receipt': inspected['observation_receipt']})
        self.tool('estimate_geometry', {'reason': 'explicit fixture depth'})
        self.tool('segment_instances', {'reason': 'explicit fixture masks'})
        self.tool('measure', {'sam_root': str(self.sam)})
        result, _ = self.tool('build_whitebox', {'layout': self.layout, 'sam_root': str(self.sam)})
        self.tool('request_human_review', {'message': 'temporary fixture review only'})
        return result

    def accept(self, targets=('chair', 'chair_two')):
        p = self.service.status(self.pid)
        result = self.service.control(self.pid, {'action': 'approve_whitebox', **p['head'], 'target_ids': list(targets)})
        return result['acceptance']

    def refine(self, acceptance, target):
        result, _ = self.tool('refine_object', {'target_id': target, 'acceptance_id': acceptance['acceptance_id'],
                                             'operation_id': 'fixture_' + target})
        ref = {key: result['candidate'][key] for key in ('candidate_id', 'candidate_hash')}
        report, _ = self.tool('validate_candidate', ref)
        assert report['technical_pass'] and report['committable']
        result, _ = self.tool('commit_candidate', {**ref, 'validation_id': report['validation_id']})
        return result


@pytest.fixture
def h(tmp_path, monkeypatch):
    harness = ServiceHarness(tmp_path, monkeypatch)
    yield harness
    harness.service.close()


def test_observation_requires_actual_tool_receipt_acknowledgment(h):
    result, response = h.tool('inspect_video')
    assert len([c for c in response['content'] if c['type'] == 'image']) == 1
    with pytest.raises(ContractError, match='acknowledged'):
        h.tool('record_prior', {'prior': h.prior, 'observation_receipt': result['observation_receipt']})
    h.service._on_pi_event(h.pid, {'type': 'tool_receipt', 'run_id': 'mock_pi_run', 'result': response})
    value, _ = h.tool('record_prior', {'prior': h.prior, 'observation_receipt': result['observation_receipt']})
    assert value['objects'] == 2
    assert h.author_calls == []


def test_full_first_loop_schema_measure_preview_and_host_only_acceptance(h):
    result = h.seed()
    assert result['snapshot']['scene_id'] == h.pid
    assert h.service.status(h.pid)['phase'] == 'needs_review'
    assert (h.run / 'scene_actions/measurement_preview.png').is_file()
    with pytest.raises(ValueError, match='Unknown workflow tool'):
        h.tool('accept_whitebox', {})
    before = h.service.status(h.pid)
    with pytest.raises(ContractError, match='exact saved'):
        h.service.control(h.pid, {'action': 'approve_whitebox', 'revision': before['head']['revision'], 'head_seq': 999})
    with h.service.db() as db:
        assert db.execute('SELECT count(*) FROM human_events').fetchone()[0] == 0
    accepted = h.accept()
    assert accepted['head'] == before['head']
    assert accepted['target_ids'] == ['chair', 'chair_two']


def test_shared_host_evidence_remains_scene_bound(h):
    h.seed()
    source = h.root / 'second.mp4'; source.write_bytes(b'explicit second mock input')
    second = h.service.create_project({'request_id': 'create_second', 'source_path': str(source)})
    other = h.service.action(second['id'])
    assert other.host is h.actions.host
    assert h.actions.host.evidence.repository is other.host.repo
    view = asyncio.run(h.actions.inspect_scene())
    ev = asyncio.run(h.actions.host.evidence.select(h.actions._request('select_evidence', {
        'revision': view['head']['revision'], 'target_ids': ['chair'], 'purpose': 'review', 'n': 1}, 'shared_evidence'), h.actions.host.context()))
    assert ev['scene_id'] == h.pid
    assert all(f['mask'] is not None for f in ev['frames'])
    with pytest.raises(ContractError, match='verification record'):
        h.service._verify('human_events', second['id'], 'missing')


def test_two_objects_share_one_acceptance_only_through_verified_commits(h):
    h.seed(); accepted = h.accept()
    with pytest.raises((ValueError, ContractError)):
        h.tool('finish_workflow', {'summary': 'must not finish before refinement'})
    one = h.refine(accepted, 'chair')
    assert one['head']['head_seq'] == 1
    two = h.refine(accepted, 'chair_two')
    assert two['head']['head_seq'] == 2
    assert h.author_calls == ['chair', 'chair_two']
    exported, _ = h.tool('export_scene')
    delivery = asyncio.run(h.actions.verify_delivery(accepted['acceptance_id'], exported))
    assert delivery['refined_objects'] == 2 and delivery['layout_preserved']
    finished, _ = h.tool('finish_workflow', {'summary': 'explicit mocked provider integration complete'})
    assert finished['status'] == 'complete' and finished['head'] == two['head']


def test_delivery_rejects_forged_completion_and_stale_export(h):
    h.seed(); accepted = h.accept(('chair',))
    save(h.run / 'scene_actions/operations/forged/completion.json', {'kind': 'refine', 'status': 'complete'})
    exported, _ = h.tool('export_scene')
    with pytest.raises((ValueError, ContractError)):
        h.tool('finish_workflow', {'summary': 'forged success must fail'})
    h.refine(accepted, 'chair')
    with pytest.raises((ValueError, ContractError), match='stale|current|match'):
        h.tool('finish_workflow', {'summary': 'old whitebox export must fail'})
    with pytest.raises(ContractError):
        asyncio.run(h.actions.verify_delivery(accepted['acceptance_id'], exported))


def test_delivery_rejects_changed_export_bytes(h):
    h.seed(); accepted = h.accept(('chair',)); h.refine(accepted, 'chair')
    exported, _ = h.tool('export_scene')
    glb = next(ref for ref in exported['artifact_refs'] if ref['role'] == 'scene_glb')
    h.actions.host.resolver.resolve(glb, scene_id=h.pid).write_bytes(b'changed after export')
    with pytest.raises(ContractError):
        asyncio.run(h.actions.verify_delivery(accepted['acceptance_id'], exported))
    with pytest.raises((ValueError, ContractError)):
        h.tool('finish_workflow', {'summary': 'changed output must fail'})


def test_cached_tool_payload_must_match_persisted_hash(h):
    request = h.request('workflow_status')
    original = h.service.tool(request)
    assert h.service.tool(request) == original
    with h.service.db() as db:
        row = db.execute('SELECT data FROM jobs WHERE id=?', (h.pid + ':' + request['tool_call_id'],)).fetchone()
    path = Path(json.loads(row['data'])['result_path'])
    save(path, {'content': [{'type': 'text', 'text': 'forged cached response'}]})
    with pytest.raises((ValueError, ContractError)):
        h.service.tool(request)


@pytest.mark.parametrize('intent', ['edit_selected', 'refine_selected'])
def test_message_intent_records_only_actual_browser_scope_and_base(h, monkeypatch, intent):
    h.seed()
    sent, arrived = [], threading.Event()
    def capture_start(*args):
        sent.append(args); arrived.set()
    monkeypatch.setattr(h.service, '_start', capture_start)
    before = h.service.status(h.pid)
    text = '把选中椅子向右移 10cm' if intent == 'edit_selected' else '保持位置尺寸，精修这把椅子'
    result = h.service.message(h.pid, {'request_id': 'intent_' + intent, 'message': text, 'intent': intent,
        'selected_ids': ['chair'], 'base_revision': before['head']['revision'], 'base_head_seq': before['head']['head_seq']})
    assert result['accepted'] and arrived.wait(2)
    context = json.loads(sent[0][2].split('当前页面上下文：')[1])
    with h.service.db() as db:
        rows = db.execute('SELECT data FROM human_events WHERE project_id=?', (h.pid,)).fetchall()
    assert len(rows) == 1
    actual = json.loads(rows[0]['data'])
    view = asyncio.run(h.actions.inspect_scene())
    assert actual['actor'] == 'user' and actual['source'] == 'browser'
    assert actual['head'] == view['head'] and actual['snapshot_hash'] == canonical_hash(view['snapshot'])
    assert actual['payload']['user_instruction'] == text
    if intent == 'edit_selected':
        assert actual['kind'] == 'scene_patch'
        assert context['human_event'] == actual['event_id']
        assert actual['payload']['allowed_object_ids'] == ['chair']
        assert set(actual['payload']['allowed_fields']) == scene_actions.PATCH_FIELDS
        assert h.service.get(h.pid)['acceptance'] is None
    else:
        assert actual['kind'] == 'accept_whitebox'
        assert actual['payload']['target_ids'] == ['chair']
        assert context['acceptance']['head'] == before['head']
        saved = read(h.actions.root / 'acceptances' / (context['acceptance']['acceptance_id'] + '.json'))
        assert saved['human_event'] == actual
    assert h.author_calls == []


@pytest.mark.parametrize('invalid', ['missing_base', 'stale_base', 'unknown_object', 'busy'])
def test_message_change_refuses_bad_scope_before_human_event(h, invalid):
    h.seed()
    view = h.service.status(h.pid)
    request = {'request_id': 'bad_' + invalid, 'message': '移动选中对象', 'intent': 'edit_selected',
        'selected_ids': ['chair'], 'base_revision': view['head']['revision'], 'base_head_seq': view['head']['head_seq']}
    if invalid == 'missing_base': request.pop('base_revision')
    elif invalid == 'stale_base': request['base_head_seq'] += 1
    elif invalid == 'unknown_object': request['selected_ids'] = ['not_present']
    else:
        view['active_run'] = 'fixture_active'; h.service.put(view)
    with pytest.raises((ValueError, ContractError)):
        h.service.message(h.pid, request)
    with h.service.db() as db:
        assert db.execute('SELECT count(*) FROM human_events WHERE project_id=?', (h.pid,)).fetchone()[0] == 0


def test_pause_keeps_cancel_set_until_running_native_tool_has_finished(h):
    with h.service.db() as db:
        db.execute('INSERT INTO jobs VALUES(?,?,?,?)', ('fixture_native', h.pid, 'running', json.dumps({'tool': 'segment_instances'})))
    h.service.control(h.pid, {'action': 'pause'})
    h.service._on_pi_event(h.pid, {'type': 'aborted', 'run_id': 'fixture_old'})
    assert h.service.cancel[h.pid].is_set()
    with pytest.raises(ValueError, match='收尾|退出|stopped|running'):
        h.service.control(h.pid, {'action': 'resume', 'request_id': 'early_resume'})
    assert h.service.cancel[h.pid].is_set()
    with h.service.db() as db:
        db.execute('UPDATE jobs SET state=? WHERE id=?', ('cancelled', 'fixture_native'))
    h.service.control(h.pid, {'action': 'resume', 'request_id': 'finished_resume'})
    assert not h.service.cancel[h.pid].is_set()


def test_restart_requests_owned_supervisors_to_cancel_and_blocks_pending_resume(h):
    p = h.service.get(h.pid); p.update(phase='running', active_run='fixture_interrupted'); h.service.put(p)
    with h.service.db() as db:
        db.execute('INSERT INTO jobs VALUES(?,?,?,?)', ('fixture_restart_job', h.pid, 'running', json.dumps({'tool': 'estimate_geometry'})))
    started = h.run / 'tools/vipe/fixture_started'
    starting = h.run / 'tools/sam/fixture_starting'
    complete = h.run / 'tools/sam/fixture_complete'
    for folder in (started, starting, complete):
        save(folder / 'process.json', {'job_token': folder.name, 'argv': ['explicit', 'fixture']})
    save(started / 'linux_process.json', {'job_token': started.name, 'pid': 99999999})
    save(complete / 'process_result.json', {'job_token': complete.name, 'status': 'complete', 'returncode': 0})
    h.service.close()  # A restart ends the old coordinator before opening the same root.
    restarted = service.WorkflowService(h.service.root)
    try:
        assert restarted.get(h.pid)['phase'] == 'paused'
        assert restarted.get(h.pid)['active_run'] is None
        with restarted.db() as db:
            assert db.execute('SELECT state FROM jobs WHERE id=?', ('fixture_restart_job',)).fetchone()[0] == 'uncertain'
        for folder in (started, starting):
            assert read(folder / 'cancel.request')['reason'] == 'workflow_host_restarted'
        assert not (complete / 'cancel.request').exists()
        with pytest.raises(ValueError, match='收尾|退出|native|pending|process'):
            restarted.control(h.pid, {'action': 'resume', 'request_id': 'restart_pending'})
        for folder in (started, starting):
            save(folder / 'process_result.json', {'job_token': folder.name, 'status': 'cancelled', 'returncode': 130})
        restarted.control(h.pid, {'action': 'resume', 'request_id': 'restart_finished'})
    finally:
        restarted.close()


def test_tool_failure_and_followup_chat_restore_whitebox_review_interaction(h, monkeypatch):
    h.seed()
    monkeypatch.setattr(h.service, '_start', lambda *args: None)
    h.service.message(h.pid, {'request_id': 'ask_about_whitebox', 'message': '为什么墙有点歪？', 'selected_ids': []})
    failed = h.request('view_frames', {'source_frame_indices': [99999]})
    with pytest.raises(KeyError):
        h.service.tool(failed)
    h.service._on_pi_event(h.pid, {'type': 'assistant_message', 'run_id': 'fixture_explanation',
        'turn_id': 1, 'stopReason': 'stop', 'text': 'fixture: report missing view without changing scene'})
    h.service._on_pi_event(h.pid, {'type': 'settled', 'run_id': 'fixture_explanation'})
    p = h.service.status(h.pid)
    assert p['phase'] == 'needs_review' and p['active_run'] is None and p['active_tool'] is None
    assert p['head']['head_seq'] == 0 and p['acceptance'] is None
    # Same failed call is never replayed; a fresh status call remains usable.
    with pytest.raises(ValueError, match='already started'):
        h.service.tool(failed)
    status, _ = h.tool('workflow_status')
    assert status['head'] == p['head']


def test_pre_scene_tool_failure_settles_idle_and_can_resume(h, monkeypatch):
    monkeypatch.setattr(h.service, '_start', lambda *args: None)
    h.service.message(h.pid, {'request_id': 'start_fixture', 'message': '构建房间', 'selected_ids': []})
    with pytest.raises(FileNotFoundError):
        h.tool('view_frames', {'source_frame_indices': [7]})
    h.service._on_pi_event(h.pid, {'type': 'settled', 'run_id': 'fixture_initial'})
    assert h.service.get(h.pid)['phase'] == 'idle'
    result = h.service.control(h.pid, {'action': 'resume', 'request_id': 'retry_after_fixture_error'})
    assert result['phase'] == 'running' and result['active_run']


def test_model_retry_recovers_without_leaving_run_failed(h):
    p=h.service.get(h.pid);p.update(phase='running',active_run='fixture-reconnect',run_failed=False)
    h.service.put(p)
    def emit(kind,**fields):h.service._on_pi_event(h.pid,{'type':kind,'run_id':'fixture-reconnect',**fields})
    emit('assistant_message',turn_id=1,stopReason='error',error='fetch failed')
    emit('auto_retry_start',data={'attempt':1,'maxAttempts':5,'delayMs':2000,'errorMessage':'fetch failed'})
    p=h.service.get(h.pid)
    assert p['phase']=='running' and p['active_run']=='fixture-reconnect'
    assert p['model_retry']['status']=='waiting' and p['error'] is None and not p['run_failed']
    emit('turn_start',turn_id=2)
    assert h.service.get(h.pid)['model_retry']['status']=='retrying'
    emit('assistant_message',turn_id=2,stopReason='stop',text='Recovered fixture')
    emit('auto_retry_end',data={'attempt':1,'success':True})
    emit('settled')
    p=h.service.get(h.pid)
    assert p['phase']=='idle' and not p['active_run'] and not p['run_failed'] and p['error'] is None
    assert p['model_retry']['status']=='recovered'


@pytest.mark.parametrize('paused',[False,True])
def test_model_retry_exhaustion_and_pause_settle_without_advancing(h,paused):
    p=h.service.get(h.pid);p.update(phase='paused' if paused else 'running',active_run='fixture-reconnect')
    h.service.put(p)
    h.service._on_pi_event(h.pid,{'type':'auto_retry_start','run_id':'fixture-reconnect',
        'data':{'attempt':5,'maxAttempts':5,'delayMs':30000,'errorMessage':'fetch failed'}})
    h.service._on_pi_event(h.pid,{'type':'auto_retry_end','run_id':'fixture-reconnect',
        'data':{'attempt':5,'success':False,'finalError':'Retry cancelled' if paused else 'fetch failed'}})
    h.service._on_pi_event(h.pid,{'type':'settled','run_id':'fixture-reconnect'})
    p=h.service.get(h.pid)
    assert p['phase']==('paused' if paused else 'failed') and not p['active_run']
    assert p['model_retry']['status']==('cancelled' if paused else 'exhausted')
    if not paused:assert p['run_failed'] and p['error']=='fetch failed'


def test_sam_retry_is_worker_bound_bounded_and_requires_reason(h, monkeypatch, tmp_path):
    fake_root = tmp_path / 'worker_root'
    worker = fake_root / 'video2scene/agent_scene/sam_worker.py'; worker.parent.mkdir(parents=True)
    worker.write_text('# explicit fake worker v1\n', encoding='utf-8')
    monkeypatch.setattr(service, 'ROOT', fake_root)
    requests = [{'object_id': 'chair', 'prompt': 'chair'}]
    base = h.service._sam_attempt(h.run, requests)
    save(h.run / 'tools/sam' / base / 'receipt.json', {'status': 'failed'})
    with pytest.raises(ValueError, match='retry_reason'):
        h.service._sam_attempt(h.run, requests)
    with pytest.raises(ValueError, match='retry_reason'):
        h.service._sam_attempt(h.run, requests, '   ')
    assert h.service._sam_attempt(h.run, requests, 'corrected the worker API use') == base + '_retry1'
    save(h.run / 'tools/sam' / (base + '_retry1') / 'receipt.json', {'status': 'failed'})
    assert h.service._sam_attempt(h.run, requests, 'changed the bad object prompt') == base + '_retry2'
    save(h.run / 'tools/sam' / (base + '_retry2') / 'receipt.json', {'status': 'failed'})
    with pytest.raises(ValueError, match='budget exhausted'):
        h.service._sam_attempt(h.run, requests, 'cannot add unbounded retries')
    worker.write_text('# explicit fake worker v2\n', encoding='utf-8')
    changed = h.service._sam_attempt(h.run, requests)
    assert changed != base and '_retry' not in changed
    save(h.run / 'tools/sam' / changed / 'receipt.json', {'status': 'complete'})
    assert h.service._sam_attempt(h.run, requests) == changed


@pytest.mark.parametrize('status', ['running', 'uncertain'])
def test_sam_uncertain_attempt_never_receives_new_worker_job(h, monkeypatch, tmp_path, status):
    fake_root = tmp_path / 'worker_root'
    worker = fake_root / 'video2scene/agent_scene/sam_worker.py'; worker.parent.mkdir(parents=True)
    worker.write_text('# fixture worker', encoding='utf-8'); monkeypatch.setattr(service, 'ROOT', fake_root)
    requests = [{'object_id': 'chair', 'prompt': 'chair'}]
    base = h.service._sam_attempt(h.run, requests)
    save(h.run / 'tools/sam' / base / 'receipt.json', {'status': status})
    with pytest.raises(ValueError, match='running|interrupted|uncertain|未结束'):
        h.service._sam_attempt(h.run, requests, 'reason cannot convert uncertainty into a terminated run')


def test_sam_failed_receipt_with_pending_native_supervisor_still_blocks_retry(h, monkeypatch, tmp_path):
    fake_root = tmp_path / 'worker_root'
    worker = fake_root / 'video2scene/agent_scene/sam_worker.py'; worker.parent.mkdir(parents=True)
    worker.write_text('# fixture worker', encoding='utf-8'); monkeypatch.setattr(service, 'ROOT', fake_root)
    requests = [{'object_id': 'chair', 'prompt': 'chair'}]
    base = h.service._sam_attempt(h.run, requests)
    folder = h.run / 'tools/sam' / base
    save(folder / 'receipt.json', {'status': 'failed'})
    save(folder / 'process.json', {'job_token': 'fixture_worker'})
    with pytest.raises(ValueError, match='running|interrupted|收尾|process|结束'):
        h.service._sam_attempt(h.run, requests, 'known repair after timeout')

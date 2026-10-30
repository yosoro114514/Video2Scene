"""Mechanical recovery and remaining-critic gates; no model is invoked."""
import hashlib
import json

import pytest

from video2scene.agent_scene.recovery import (
    WORLD_LINE, inspect_source, patch_world_initialization, require_critic_accept,
)
from video2scene.contracts_v1 import ContractError


@pytest.mark.parametrize('newline', [b'\n', b'\r\n'])
def test_world_patch_is_only_the_exact_initialization_insertion(newline):
    original = newline.join([b'import bpy', b"geometry = 'unchanged'", WORLD_LINE, b"material = 'unchanged'", b''])
    patched = patch_world_initialization(original)
    insertion = (b'if bpy.context.scene.world is None:' + newline
                 + b"    bpy.context.scene.world = bpy.data.worlds.new('Recovered World')" + newline)
    assert patched.count(insertion) == 1
    assert patched.replace(insertion, b'', 1) == original
    assert hashlib.sha256(patched).digest() != hashlib.sha256(original).digest()
    with pytest.raises(ContractError):
        patch_world_initialization(patched)


@pytest.mark.parametrize('source', [b'import bpy\n', WORLD_LINE + b'\n' + WORLD_LINE + b'\n'])
def test_unexpected_script_does_not_receive_a_general_rewrite(source):
    with pytest.raises(ContractError) as error:
        patch_world_initialization(source)
    assert error.value.code == 'BUILD_FAILED'


@pytest.mark.parametrize('status,major', [('revise', False), ('unclear', False), ('accept', True)])
def test_last_critic_must_accept_without_major_issue(status, major):
    critic = {'status': status, 'summary': 'Explicit test fixture', 'limitations': [],
              'issues': [{'severity': 'major', 'evidence': 'fixture', 'requested_fix': 'fixture'}] if major else []}
    with pytest.raises(ContractError) as error:
        require_critic_accept(critic)
    assert error.value.code == 'BUILD_FAILED'


def test_last_critic_may_accept_with_documented_minor_uncertainty():
    require_critic_accept({'status': 'accept', 'summary': 'Explicit test fixture', 'limitations': [],
                          'issues': [{'severity': 'minor', 'evidence': 'fixture', 'requested_fix': 'fixture'}]})


@pytest.mark.parametrize('change', [{'model_calls': 4}, {'provider_calls': 4}, {'status': 'running'}, {'origin': 'local_fixture'}])
def test_source_budget_or_uncertain_state_is_rejected_before_any_call(tmp_path, change):
    ledger = {'origin': 'codex_image', 'status': 'failed', 'model_calls': 3, 'provider_calls': 3}
    ledger.update(change)
    (tmp_path / 'run.json').write_text(json.dumps(ledger), encoding='utf-8')
    with pytest.raises(ContractError) as error:
        inspect_source(tmp_path)
    assert error.value.code == 'FORBIDDEN_SCOPE'
    assert sorted(path.name for path in tmp_path.iterdir()) == ['run.json']

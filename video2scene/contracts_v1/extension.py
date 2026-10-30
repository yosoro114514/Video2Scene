"""Strict host-internal addition contract; not a public v1 ToolRequest route."""
from functools import lru_cache

from jsonschema import Draft202012Validator

from .core import CONTRACT_VERSION, ContractError, canonical_bytes, schema


@lru_cache
def _extension_validator():
    identity = {'type': 'string', 'pattern': r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$'}
    replacement = {'type': 'object', 'properties': {'instance_id': identity, 'asset': {'$ref': '#/$defs/AssetRef'}},
                   'required': ['instance_id', 'asset'], 'additionalProperties': False}
    params = {'type': 'object', 'properties': {
        'intent': {'enum': ['design_edit', 'reconstruction_correction']}, 'decision_ref': identity,
        'additions': {'type': 'array', 'items': {'$ref': '#/$defs/SceneObject'}, 'minItems': 1, 'maxItems': 32},
        'asset_replacements': {'type': 'array', 'items': replacement, 'maxItems': 64}},
        'required': ['intent', 'decision_ref', 'additions', 'asset_replacements'], 'additionalProperties': False}
    definition = {'$schema': schema()['$schema'], '$defs': schema()['$defs'], 'type': 'object',
                  'properties': {'contract_version': {'const': CONTRACT_VERSION}, 'request_id': identity,
                                 'scene_id': identity, 'tool': {'const': 'extend_scene'}, 'base_revision': identity,
                                 'base_head_seq': {'type': 'integer', 'minimum': 0}, 'params': params},
                  'required': ['contract_version', 'request_id', 'scene_id', 'tool', 'base_revision', 'base_head_seq', 'params'],
                  'additionalProperties': False}
    return Draft202012Validator(definition)


def validate_extension_request(request):
    try:
        canonical_bytes(request)
    except (TypeError, ValueError) as exc:
        raise ContractError('INVALID_ARGUMENT', 'Extension request must contain finite JSON values') from exc
    errors = list(_extension_validator().iter_errors(request))
    if errors:
        raise ContractError('INVALID_ARGUMENT', 'Invalid SceneExtensionRequest', {'errors': [error.message for error in errors[:5]]})
    return request

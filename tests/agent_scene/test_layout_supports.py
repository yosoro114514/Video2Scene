"""Layout support errors identify the exact input without normalizing a draft."""
import copy

import pytest

from video2scene.agent_scene.run import LAYOUT_ITEM, validate_layout


def inputs():
    common = {'yaw_deg': 0., 'basis': 'mixed', 'reason': 'fixture'}
    layout = {'summary': 'fixture', 'objects': [{**common, 'object_id': 'chair', 'category': 'furniture/chair', 'kind': 'object',
        'center_m': [0, .5, 0], 'dimensions_m': [1, 1, 1], 'support_object_id': 'floor'}],
        'structures': [{**common, 'object_id': 'floor', 'category': 'structure/floor', 'kind': 'floor',
        'center_m': [0, -.025, 0], 'dimensions_m': [4, .05, 4], 'support_object_id': 'unknown'}],
        'rejected_objects': [], 'uncertainties': [], 'recommended_refinement_target': 'chair'}
    prior = {'objects': [{'object_id': 'chair'}]}
    geometry = {'objects': [{'object_id': 'chair', 'status': 'observed_proposal'}]}
    return layout, prior, geometry


@pytest.mark.parametrize('invalid', ['', 'none', 'null', None, 'not_in_layout', 'floor', '<missing>'])
def test_support_error_names_object_bad_value_and_exact_unknown_sentinel(invalid):
    layout, prior, geometry = inputs()
    if invalid == '<missing>': layout['structures'][0].pop('support_object_id')
    else: layout['structures'][0]['support_object_id'] = invalid
    untouched = copy.deepcopy(layout)
    with pytest.raises(ValueError) as error:
        validate_layout(layout, prior, geometry)
    assert "object_id='floor'" in str(error.value)
    assert ('<missing>' if invalid == '<missing>' else repr(invalid)) in str(error.value)
    assert "exact sentinel 'unknown'" in str(error.value)
    assert layout == untouched


def test_known_support_and_unknown_remain_unchanged_and_schema_explains_required_choice():
    layout, prior, geometry = inputs()
    untouched = copy.deepcopy(layout)
    assert validate_layout(layout, prior, geometry) is layout
    assert layout == untouched
    assert 'support_object_id' in LAYOUT_ITEM['required']
    description = LAYOUT_ITEM['properties']['support_object_id']['description']
    assert 'Required' in description and "'unknown'" in description and 'another object_id' in description

import copy
import json
import pytest
from video2scene.contracts_v1 import canonical_hash
from video2scene.pi_workflow.service import WorkflowService


@pytest.fixture
def draft(tmp_path):
    service=WorkflowService(tmp_path/'workflow');video=tmp_path/'fixture.mp4';video.write_bytes(b'fixture')
    p=service.create_project({'request_id':'fixture','source_path':str(video)})
    args={'layout':{'objects':[{'object_id':'chair','support_object_id':'floor','center_m':[1,2,3]}],
                    'structures':[{'object_id':'floor','support_object_id':'none','center_m':[0,0,0]}]},'sam_root':'fixture-sam'}
    request={'project_id':p['id'],'name':'build_whitebox','args':args}
    record={'id':'failed_fixture','request':request,'request_hash':canonical_hash(request)}
    with service.db() as db:db.execute('INSERT INTO jobs VALUES(?,?,?,?)',('fixture-key',p['id'],'failed',json.dumps(record)))
    yield service,p['id'],args
    service.close()


def test_support_repair_preserves_the_exact_model_draft(draft):
    service,pid,original=draft
    repaired=service._repair_supports(pid,{'failed_job_id':'failed_fixture','support_corrections':[{'object_id':'floor','support_object_id':'unknown'}]})
    expected=copy.deepcopy(original);expected['layout']['structures'][0]['support_object_id']='unknown'
    assert repaired==expected
    assert original['layout']['structures'][0]['support_object_id']=='none'
    with service.db() as db:record=json.loads(db.execute('SELECT data FROM jobs WHERE id=?',('fixture-key',)).fetchone()['data'])
    assert record['request']['args']==original


@pytest.mark.parametrize('changes',[[{'object_id':'floor','support_object_id':'none'}],
    [{'object_id':'missing','support_object_id':'unknown'}],[{'object_id':'chair','support_object_id':'unknown'}]])
def test_support_repair_rejects_invalid_or_unrelated_changes(draft,changes):
    service,pid,_=draft
    with pytest.raises(ValueError):service._repair_supports(pid,{'failed_job_id':'failed_fixture','support_corrections':changes})


def test_support_repair_never_reads_another_project_draft(draft):
    service,pid,_=draft
    with pytest.raises(ValueError):service._repair_supports('another_project',{'failed_job_id':'failed_fixture','support_corrections':[{'object_id':'floor','support_object_id':'unknown'}]})

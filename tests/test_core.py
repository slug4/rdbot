import json
import pytest
from rxdx.core import Passage, answer_question, chunk_pages, validate_url

@pytest.mark.parametrize('url',['http://rxdx.in/','https://evil.test/','https://rxdx.in.evil.test/','https://user:password@rxdx.in/','https://rxdx.in:444/'])
def test_reject_unapproved_urls(url):
    with pytest.raises(ValueError): validate_url(url)

def test_chunks_preserve_source_and_tail():
    result=chunk_pages([('https://rxdx.in/','x'*2500+'TAIL')])
    assert result[-1].text.endswith('TAIL')
    assert all(p.url=='https://rxdx.in/' for p in result)
    assert len({p.id for p in result})==len(result)

def test_empty_evidence_never_calls_provider():
    def fail(*_): raise AssertionError('must not call provider')
    assert answer_question('Where is the clinic?',[],fail)['mode']=='abstain'

@pytest.mark.parametrize('question',['Diagnose my symptoms','What dosage should I take?','What medicine should I use?','I have my chest pain'])
def test_medical_requests_refused(question):
    assert answer_question(question,[])['mode']=='safety'

def test_retrieval_only_mode():
    p=Passage(1,'https://rxdx.in/','Public services')
    assert answer_question('services',[p])['mode']=='retrieval'

@pytest.mark.parametrize('response',[
    '{invalid',json.dumps({'answer':'Claim [9]','source_ids':[9]}),
    json.dumps({'answer':'Claim without citation','source_ids':[1]}),
    json.dumps({'answer':'Claim [1] https://evil.test','source_ids':[1]}),
    json.dumps({'answer':'Claim [1]','source_ids':[True]}),
])
def test_invalid_citations_fail_closed(response):
    p=Passage(1,'https://rxdx.in/','Public services')
    assert answer_question('services',[p],lambda *_:response)['mode']=='abstain'

def test_grounded_response_retains_actual_source():
    p=Passage(1,'https://rxdx.in/','Clinic offers health checks.')
    def generate(system,payload):
        assert 'untrusted' in system
        assert json.loads(payload)['passages'][0]['text']==p.text
        return json.dumps({'answer':'Health checks are offered [1].','source_ids':[1]})
    result=answer_question('services',[p],generate)
    assert result['mode']=='ai' and result['sources']==[p]

def test_question_limit():
    with pytest.raises(ValueError): answer_question('x'*2001,[])

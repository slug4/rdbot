import pytest
import app as web
from rag import Chunk, Retriever

@pytest.fixture
def client():
    web.app.config['TESTING']=True
    return web.app.test_client()


def test_vercel_entrypoint_and_home(client):
    assert web.app is not None
    response=client.get('/')
    assert response.status_code==200
    assert b'Public knowledge assistant' in response.data
    assert client.get('/static/app.js').status_code==200
    assert client.get('/health').json['status']=='ok'

@pytest.mark.parametrize('body',[{},[],{'question':None},{'question':''},{'question':'x'*2001}])
def test_input_rejected(client,body):
    assert client.post('/api/answer',json=body).status_code==400

def test_large_payload_rejected(client):
    assert client.post('/api/answer',data='x'*9000,content_type='application/json').status_code==413


def test_medical_refusal_never_calls_retrieval(client,monkeypatch):
    def fail(): raise AssertionError('must not retrieve')
    monkeypatch.setattr(web,'get_index',fail)
    result=client.post('/api/answer',json={'question':'What dosage should I take?'}).json
    assert result['mode']=='safety' and result['sources']==[]


def index_fixture():
    chunks=[Chunk('Synthetic clinic contact telephone information.','https://rxdx.in/')]
    return Retriever(chunks,web.KeywordEmbeddings(chunks)),[], 'keyword'


def test_source_grounded_excerpt_mode(client,monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    monkeypatch.delenv('RXDX_API_KEY',raising=False)
    monkeypatch.setattr(web,'get_index',index_fixture)
    response=client.post('/api/answer',json={'question':'clinic contact telephone'})
    assert response.status_code==200
    result=response.json
    assert result['mode']=='keyword'
    assert result['sources'][0]['url']=='https://rxdx.in/'
    assert 'No generation API configured' in result['answer']


def test_unrelated_question_abstains(client,monkeypatch):
    monkeypatch.setattr(web,'get_index',index_fixture)
    response=client.post('/api/answer',json={'question':'zebras spaceships'})
    assert response.json['sources']==[]
    assert 'verify' in response.json['answer']


def test_provider_failure_returns_clear_error(client,monkeypatch):
    def fail(): raise RuntimeError('provider failure with confidential detail')
    monkeypatch.setattr(web,'get_index',fail)
    response=client.post('/api/answer',json={'question':'clinic contact'})
    assert response.status_code==503
    assert 'confidential' not in response.get_data(as_text=True)


def test_generated_bad_citations_fail_closed(client,monkeypatch):
    monkeypatch.setenv('RXDX_API_KEY','test-placeholder')
    monkeypatch.setattr(web,'get_index',index_fixture)
    monkeypatch.setattr(web,'answer',lambda *_:'Unsupported claim [Source 99]')
    response=client.post('/api/answer',json={'question':'clinic contact'})
    assert 'unverified' in response.json['mode']
    assert 'Unsupported claim' not in response.json['answer']


def test_index_cached_only_after_success(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    monkeypatch.delenv('RXDX_API_KEY',raising=False)
    web._cache.clear()
    calls=[]
    def load(*_,**__):
        calls.append(1)
        return [Chunk('Synthetic public contact information.','https://rxdx.in/')],[]
    monkeypatch.setattr(web,'load_pages',load)
    first=web.get_index()
    second=web.get_index()
    assert first[0] is second[0] and len(calls)==1
    web._cache.clear()

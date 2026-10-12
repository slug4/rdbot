import pytest
import app as web
from faq import QUESTIONS
from rag import Chunk

@pytest.fixture
def client(monkeypatch):
    web.app.config['TESTING']=True
    monkeypatch.setenv('RXDX_API_KEY','not-a-real-key')
    return web.app.test_client()


def test_catalog_lists_questions_without_stored_answers(client):
    questions=client.get('/api/faqs').json['questions']
    assert len(questions)==30
    assert len({q['id'] for q in questions})==30
    assert all(set(q)=={'id','category','question'} for q in questions)
    assert client.get('/').get_data(as_text=True).count('class="faq-question"')==30

@pytest.mark.parametrize('faq',QUESTIONS,ids=[row['id'] for row in QUESTIONS])
def test_listed_questions_use_retrieval_and_generation(client,monkeypatch,faq):
    calls=[]
    class Index:
        def search(self,question):
            calls.append(('retrieval',question))
            return [(Chunk('Synthetic official information.','https://rxdx.in/'),0.9)]
    monkeypatch.setattr(web,'get_index',lambda:(Index(),[],'semantic'))
    def generate(question,hits):
        calls.append(('generation',question))
        assert hits[0][0].text=='Synthetic official information.'
        return 'Synthetic source-grounded answer [Source 1].'
    monkeypatch.setattr(web,'answer',generate)
    response=client.post('/api/answer',json={'question':faq['question'],'faq_id':faq['id'],'simple':True})
    assert response.status_code==200
    if response.json['mode']=='safety':
        assert not calls
    else:
        assert response.json['mode']=='semantic'
        assert [call[0] for call in calls]==['retrieval','generation']
        assert response.json['sources'][0]['url']=='https://rxdx.in/'


def test_diagnostic_services_are_not_mistaken_for_diagnosis(client,monkeypatch):
    class Index:
        def search(self,_): return []
    monkeypatch.setattr(web,'get_index',lambda:(Index(),[],'semantic'))
    response=client.post('/api/answer',json={'question':'What diagnostic tests are available?'})
    assert response.status_code==200 and response.json['mode']!='safety'


def test_missing_key_is_explicit_not_canned_answer(client,monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    monkeypatch.delenv('RXDX_API_KEY',raising=False)
    response=client.post('/api/answer',json={'question':'What are doctor timings?'})
    assert response.status_code==503 and response.json['error_code']=='provider_configuration'

from streamlit.testing.v1 import AppTest

def test_ui_starts_and_safety_reply_needs_no_index():
    app=AppTest.from_file('app.py').run(timeout=30)
    assert not app.exception
    assert app.title[0].value=='RxDx public information assistant'
    assert app.button[0].disabled
    app.chat_input[0].set_value('What dosage should I take?').run()
    assert not app.exception
    assert any('cannot provide individual medical advice' in m.value for m in app.markdown)

def test_retrieval_reply_shows_citable_source(monkeypatch):
    from rxdx.core import Passage
    class Index:
        def search(self,question):
            return [Passage(1,'https://rxdx.in/','Synthetic test service information.')]
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    monkeypatch.delenv('RXDX_API_KEY',raising=False)
    app=AppTest.from_file('app.py')
    app.session_state['index']=Index()
    app.run(timeout=30)
    app.chat_input[0].set_value('What services are listed?').run()
    assert not app.exception
    assert any('AI generation is not configured' in m.value for m in app.markdown)
    assert any('Synthetic test service information' in m.value for m in app.markdown)
    assert app.session_state['messages'][-1]['sources'][0].url=='https://rxdx.in/'

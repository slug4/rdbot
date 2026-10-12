import numpy as np
from rag import Chunk, Retriever, chunk_text, extract_text, validate_url, answer, REFUSAL


class FakeModel:
    def encode(self, texts):
        return np.array([[1, 0] if "contact" in x.lower() else [0, 1] for x in texts], dtype=float)


def test_url_allowlist():
    assert validate_url("https://rxdx.in/")
    assert validate_url("https://www.rxdx.in/path")
    assert not validate_url("http://rxdx.in/")
    assert not validate_url("https://rxdx.in.evil.example/")
    assert not validate_url("https://rxdx.in@evil.example/")


def test_extractor_excludes_navigation_scripts():
    html = "<html><body><nav>menu</nav><main><h1>Contact RxDx</h1><script>ignore</script></main></body></html>"
    assert extract_text(html) == "Contact RxDx"


def test_chunking_keeps_all_words():
    text = " ".join(f"word{i}" for i in range(120))
    result = chunk_text(text, size=80, overlap=12)
    assert result
    assert result[0].startswith("word0")
    assert result[-1].endswith("word119")


def test_retriever_orders_relevant_passage():
    chunks = [Chunk("About services", "https://rxdx.in/"), Chunk("Contact information", "https://rxdx.in/")]
    retriever = Retriever(chunks, model=FakeModel())
    assert retriever.search("contact", top_k=1)[0][0].text == "Contact information"


def test_empty_evidence_refuses(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert answer("Question", []) == REFUSAL


def test_generation_requests_short_grounded_answers(monkeypatch):
    from types import SimpleNamespace
    import openai
    monkeypatch.setenv('RXDX_API_KEY','test-placeholder')
    def create(**kwargs):
        instructions=kwargs['messages'][0]['content']
        assert '1–3 short sentences' in instructions
        assert 'published matching schedule' in instructions
        assert 'home-care service hours' in instructions
        assert 'untrusted content' in instructions
        assert 'Synthetic appointment information' in kwargs['messages'][1]['content']
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Synthetic source answer [Source 1].'))])
    monkeypatch.setattr(openai,'OpenAI',lambda **kwargs:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    result=answer('doctor timings',[(Chunk('Synthetic appointment information','https://rxdx.in/'),0.8)])
    assert '[Source 1]' in result

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


def test_short_answer_is_an_exact_source_extract():
    text='The clinic lists home sample collection services.'
    result=answer('home sample collection',[(Chunk(text,'https://rxdx.in/'),0.8)])
    assert text in result and '[Source 1]' in result
    assert len(result)<500


def test_doctor_schedule_not_inferred_from_home_visit_hours():
    text='Home visits: Mon–Sat, 10 am–6 pm.'
    result=answer('What are doctor timings?',[(Chunk(text,'https://rxdx.in/'),0.8)])
    assert 'doctor’s name' in result and '10 am' not in result

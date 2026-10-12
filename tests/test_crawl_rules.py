from rag import load_pages

class Response:
    status_code=200
    headers={'content-type':'text/html'}
    def __init__(self,text): self.text=text;self.content=text.encode()
    def raise_for_status(self): pass


def test_robots_denial_prevents_page_request(monkeypatch):
    calls=[]
    def get(url,**kwargs):
        calls.append(url)
        assert kwargs['allow_redirects'] is False
        return Response('User-agent: *\nDisallow: /')
    monkeypatch.setattr('rag.requests.get',get)
    chunks,errors=load_pages(['https://rxdx.in/'])
    assert not chunks and errors
    assert calls==['https://rxdx.in/robots.txt']


def test_robots_html_error_fails_closed(monkeypatch):
    monkeypatch.setattr('rag.requests.get',lambda *_,**__:Response('<html>error</html>'))
    chunks,errors=load_pages(['https://rxdx.in/'])
    assert not chunks and errors


def test_public_page_extracts_source(monkeypatch):
    def get(url,**kwargs):
        return Response('User-agent: *\nAllow: /' if url.endswith('robots.txt') else '<main>'+'Synthetic public clinic information. '*10+'</main>')
    monkeypatch.setattr('rag.requests.get',get)
    monkeypatch.setattr('rag.time.sleep',lambda _:None)
    chunks,errors=load_pages(['https://rxdx.in/'])
    assert chunks and not errors
    assert chunks[0].url=='https://rxdx.in/'

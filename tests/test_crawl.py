import pytest
from rxdx.crawl import crawl, fetch

class Response:
    encoding='utf-8'
    def __init__(self,text='',status=200,location=None):
        self.text=text; self.status_code=status
        self.headers={'Content-Type':'text/html'}
        if location: self.headers['Location']=location
    def __enter__(self): return self
    def __exit__(self,*_): pass
    def raise_for_status(self):
        if self.status_code>=400: raise RuntimeError('HTTP failure')
    def iter_content(self,_): yield self.text.encode()

class Session:
    def __init__(self, responses): self.responses=responses;self.calls=[]
    def get(self,url,**kwargs):
        self.calls.append(url)
        assert kwargs['allow_redirects'] is False
        return self.responses[url]

def test_robots_denial_never_fetches_page():
    session=Session({'https://rxdx.in/robots.txt':Response('User-agent: *\nDisallow: /')})
    with pytest.raises(ValueError): crawl(['https://rxdx.in/'],session)
    assert session.calls==['https://rxdx.in/robots.txt']

def test_missing_robots_fails_closed():
    session=Session({'https://rxdx.in/robots.txt':Response(status=404)})
    with pytest.raises(RuntimeError): crawl(['https://rxdx.in/'],session)
    assert len(session.calls)==1

def test_external_redirect_never_followed():
    session=Session({'https://rxdx.in/':Response(status=302,location='https://evil.test/')})
    with pytest.raises(ValueError): fetch('https://rxdx.in/',session)
    assert len(session.calls)==1

def test_robots_disallowed_redirect_never_followed(monkeypatch):
    monkeypatch.setattr('rxdx.crawl.time.sleep',lambda _:None)
    session=Session({'https://rxdx.in/robots.txt':Response('User-agent: *\nDisallow: /private'), 'https://rxdx.in/':Response(status=302,location='/private')})
    with pytest.raises(ValueError): crawl(['https://rxdx.in/'],session)
    assert 'https://rxdx.in/private' not in session.calls

def test_extracts_public_main_text(monkeypatch):
    monkeypatch.setattr('rxdx.crawl.time.sleep',lambda _:None)
    session=Session({'https://rxdx.in/robots.txt':Response('User-agent: *\nAllow: /'), 'https://rxdx.in/':Response('<nav>navigation</nav><main>'+ 'Public clinic information. '*10+'</main><script>secret</script>')})
    pages=crawl(['https://rxdx.in/'],session)
    assert pages[0][0]=='https://rxdx.in/'
    assert 'navigation' not in pages[0][1] and 'secret' not in pages[0][1]

def test_html_robots_is_not_permission():
    session=Session({'https://rxdx.in/robots.txt':Response('<html>Access denied</html>')})
    with pytest.raises(ValueError): crawl(['https://rxdx.in/'],session)
    assert len(session.calls)==1

"""Check the known production deployment without reading or logging credentials."""
import json
from urllib.request import Request, urlopen

BASE = 'https://rdbot-gamma.vercel.app'


def call(path, body=None):
    payload = json.dumps(body).encode() if body else None
    request = Request(BASE + path, data=payload, headers={'Content-Type': 'application/json', 'User-Agent': 'RxDx-Deployment-Smoke/1.0'})
    with urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f'Unexpected HTTP status: {response.status}')
        return json.load(response)


def main():
    health = call('/health')
    assert health.get('application') == 'rxdx-rag', 'Wrong application deployed'
    assert health.get('status') == 'ok', 'Application is not healthy'
    print('Production health: ok', flush=True)
    configured = health.get('generation_configured') is True
    print(f'::notice::Production API key configured: {configured}', flush=True)
    assert configured, 'Production does not receive OPENAI_API_KEY or RXDX_API_KEY. Configure the key for Production in Vercel and redeploy.'
    result = call('/api/answer', {'question': 'Does RxDx offer home visits?'})
    assert result.get('mode') == 'semantic', 'Hosted semantic retrieval or citation validation did not succeed'
    assert result.get('sources'), 'No supporting official source excerpts returned'
    assert all(source['url'].startswith(('https://rxdx.in/', 'https://www.rxdx.in/')) for source in result['sources']), 'Unexpected source domain'
    assert '[Source ' in result.get('answer', ''), 'AI answer is missing source citations'
    print('Live semantic retrieval and AI answer passed; cited sources:', len(result['sources']), flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'[:500].replace('%', '%25').replace('\n', '%0A').replace('\r', '%0D')
        print(f'::error::{message}', flush=True)
        raise

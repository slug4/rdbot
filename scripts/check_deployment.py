"""Check the known production deployment without reading or logging credentials."""
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE = 'https://rdbot-gamma.vercel.app'


def call(path, body=None):
    payload = json.dumps(body).encode() if body else None
    request = Request(BASE + path, data=payload, headers={'Content-Type': 'application/json', 'User-Agent': 'RxDx-Deployment-Smoke/1.0'})
    try:
        with urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(f'Unexpected HTTP status: {response.status}')
            return json.load(response)
    except HTTPError as exc:
        try:
            body = json.load(exc)
            code = body.get('error_code')
        except (ValueError, AttributeError):
            code = None
        known = {'insufficient_quota','provider_authentication','provider_rate_limit','provider_request','provider_connection','source_unavailable','request_failed','retrieval_unavailable'}
        safe_code = code if code in known else 'unclassified'
        raise RuntimeError(f'Production API returned HTTP {exc.code}, error category: {safe_code}') from None


def main():
    health = call('/health')
    assert health.get('application') == 'rxdx-rag' and health.get('status') == 'ok', 'Wrong or unhealthy application'
    assert health.get('requires_api_key') is False, 'Production still depends on an AI key'
    print('::notice::Production uses local semantic retrieval without an API key.',flush=True)
    questions = call('/api/faqs')['questions']
    assert len(questions) >= 30, 'Common question list missing'
    print('::notice::Production question catalog contains 30 suggested questions.',flush=True)
    for question in ['Does RxDx offer home visits?', 'What are the doctor timings?', 'What tests and diagnostic services are listed?']:
        result = call('/api/answer', {'question':question})
        assert result.get('mode') == 'local_semantic' and result.get('sources'), 'Local semantic retrieval failed'
        assert '[Source ' in result.get('answer', ''), 'Source-based answer is missing citations'
        assert all(source['url'].startswith(('https://rxdx.in/', 'https://www.rxdx.in/')) for source in result['sources']), 'Unexpected source'
    print('::notice::Live RAG answers and citations passed for home visits, doctor timings and tests.',flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'[:500].replace('%', '%25').replace('\n', '%0A').replace('\r', '%0D')
        print(f'::error::{message}', flush=True)
        raise

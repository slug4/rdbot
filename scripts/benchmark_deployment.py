"""Sequential functional/latency check of the known production application."""
import json
import time
from pathlib import Path
import requests

BASE='https://rdbot-gamma.vercel.app'
QUESTIONS=['Does RxDx offer home visits?','What are the doctor timings?',
           'What tests and diagnostic services are listed?','Does RxDx offer home visits?']


def main():
    session=requests.Session()
    results=[]
    for question in QUESTIONS:
        started=time.perf_counter()
        response=session.post(BASE+'/api/answer',json={'question':question},timeout=120)
        body=response.json()
        row={'question':question,'seconds':round(time.perf_counter()-started,3),'http_status':response.status_code,
             'mode':body.get('mode'),'timing':body.get('timing',{}),'source_count':len(body.get('sources',[])),
             'answer':body.get('answer'),'error_code':body.get('error_code')}
        results.append(row)
        print(json.dumps(row),flush=True)
        if response.status_code!=200:
            break
        assert body.get('mode')=='local_semantic' and body.get('sources'), 'Missing local retrieval evidence'
        assert '[Source ' in body.get('answer',''), 'Missing source citation'
    Path('/tmp/rdbot-benchmark.json').write_text(json.dumps(results,indent=2))
    assert len(results)==4 and all(row['http_status']==200 for row in results), 'Production answer check failed'
    return results


if __name__=='__main__':
    main()

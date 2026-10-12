"""Lightweight Flask entry point for Vercel; no Streamlit or Torch at import."""
import re
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from rag import Retriever, answer, load_pages, REFUSAL
from faq import public_faqs
from local_model import get_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8192
_cache = {}
_lock = threading.Lock()
_request_stats = threading.local()


def get_index():
    # Warm-instance cache only. Vercel may discard it between invocations.
    mode = 'local_semantic'
    with _lock:
        cached = _cache.get(mode)
        if cached and time.monotonic() - cached[0] < 600:
            _request_stats.value = {'cache_hit':True, 'index_seconds':0.0}
            return cached[1:]
        index_start = time.perf_counter()
        urls = [line.strip() for line in Path(__file__).with_name('sources.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]
        chunks, errors = load_pages(urls[:6], timeout=6)
        if not chunks:
            raise RuntimeError('No public source pages could be indexed.')
        # Keep source coverage fair instead of letting the first page fill the index.
        counts, selected = {}, []
        for chunk in chunks:
            count = counts.get(chunk.url, 0)
            if count < 8:
                selected.append(chunk)
                counts[chunk.url] = count + 1
        chunks = selected
        model = get_model()
        index = Retriever(chunks, model=model)
        _cache[mode] = (time.monotonic(), index, errors, mode)
        _request_stats.value = {'cache_hit':False, 'index_seconds':round(time.perf_counter()-index_start,3)}
        return index, errors, mode


@app.get('/')
def home():
    questions = public_faqs()
    return render_template('index.html', faqs=questions, categories=list(dict.fromkeys(row['category'] for row in questions)))


@app.get('/api/faqs')
def faqs():
    return jsonify(questions=public_faqs())


@app.get('/health')
def health():
    return jsonify(status='ok', application='rxdx-rag', requires_api_key=False, retrieval='local_semantic', answer_style='extractive')


@app.post('/api/answer')
def query():
    started=time.perf_counter()
    _request_stats.value={}
    body = request.get_json(silent=True)
    question = body.get('question') if isinstance(body, dict) else None
    if not isinstance(question, str) or not question.strip() or len(question)>2000:
        return jsonify(error='Enter a question of 1–2,000 characters.'), 400
    if re.search(r'\b(diagnos(?:e|is|ing|ed)|prescri\w*|dosage|dose|my symptoms|my chest pain|what medicine|treat my|suicid\w*|overdose)\b', question, re.I):
        return jsonify(answer='This assistant cannot provide individual medical advice. Consult a qualified clinician. If you believe this is an emergency, contact local emergency services immediately.', sources=[], mode='safety')
    try:
        index, warnings, mode = get_index()
        hits = [(chunk,score) for chunk,score in index.search(question) if score >= 0.3]
        if not hits:
            return jsonify(answer=REFUSAL, sources=[], mode=mode, partial_sources=bool(warnings))
        text = answer(question, hits)
        if hits:
            cited = {int(i) for i in re.findall(r'\[Source (\d+)\]', text)}
            if not cited or any(i<1 or i>len(hits) for i in cited) or re.search(r'https?://',text):
                text = 'The generated answer did not include valid source citations. Please review the retrieved excerpts.'
                mode = 'unverified_excerpt'
        return jsonify(answer=text, mode=mode, partial_sources=bool(warnings), sources=[{'id':i,'url':chunk.url,'text':chunk.text} for i,(chunk,_) in enumerate(hits,1)], timing={**_request_stats.value,'processing_seconds':round(time.perf_counter()-started,3)})
    except RuntimeError:
        return jsonify(error='Could not load approved public sources or the verified local model. Please try again later.', error_code='retrieval_unavailable'), 503
    except Exception:
        app.logger.warning('Local retrieval failed; no answer returned.')
        return jsonify(error='Unable to retrieve public sources or load the local model. Please try again later.', error_code='request_failed'), 503


@app.errorhandler(413)
def too_large(_):
    return jsonify(error='Request is too large.'), 413

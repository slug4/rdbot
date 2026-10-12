"""Lightweight Flask entry point for Vercel; no Streamlit or Torch at import."""
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import APIStatusError, APIConnectionError, APITimeoutError
from rag import Chunk, Retriever, answer, load_pages, REFUSAL

load_dotenv()
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8192
_cache = {}
_lock = threading.Lock()


def configured_key():
    return os.getenv('RXDX_API_KEY') or os.getenv('OPENAI_API_KEY')


class HostedEmbeddings:
    def __init__(self, key):
        from openai import OpenAI
        base = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        if not base.startswith('https://'):
            raise ValueError('AI endpoint must use HTTPS')
        self.client = OpenAI(api_key=key, base_url=base, timeout=20, max_retries=0)

    def encode(self, texts):
        response = self.client.embeddings.create(model='text-embedding-3-small', input=texts)
        return np.asarray([row.embedding for row in sorted(response.data, key=lambda row: row.index)])


class KeywordEmbeddings:
    """Explicit fallback, not semantic embeddings."""
    def __init__(self, chunks):
        self.vocabulary = sorted({word for chunk in chunks for word in self.tokens(chunk.text)})

    @staticmethod
    def tokens(text):
        stopwords = {'the','and','for','with','does','what','where','when','how','are','can','you','about','that','this','from'}
        return {word for word in re.findall(r'\w+', text.lower()) if len(word)>2 and word not in stopwords}

    def encode(self, texts):
        tokens = [self.tokens(text) for text in texts]
        return np.asarray([[int(word in row) for word in self.vocabulary] for row in tokens], dtype=np.float32)


def get_index():
    # Warm-instance cache only. Vercel may discard it between invocations.
    mode = 'semantic' if configured_key() else 'keyword'
    with _lock:
        cached = _cache.get(mode)
        if cached and time.monotonic() - cached[0] < 600:
            return cached[1:]
        urls = [line.strip() for line in Path(__file__).with_name('sources.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]
        chunks, errors = load_pages(urls[:4], timeout=6)
        if not chunks:
            raise RuntimeError('No public source pages could be indexed.')
        chunks = chunks[:60]
        model = HostedEmbeddings(configured_key()) if mode == 'semantic' else KeywordEmbeddings(chunks)
        index = Retriever(chunks, model=model)
        _cache[mode] = (time.monotonic(), index, errors, mode)
        return index, errors, mode


@app.get('/')
def home():
    return render_template('index.html')


@app.get('/health')
def health():
    return jsonify(status='ok', application='rxdx-rag', generation_configured=bool(configured_key()))


@app.post('/api/answer')
def query():
    body = request.get_json(silent=True)
    question = body.get('question') if isinstance(body, dict) else None
    if not isinstance(question, str) or not question.strip() or len(question)>2000:
        return jsonify(error='Enter a question of 1–2,000 characters.'), 400
    if re.search(r'\b(diagnos\w*|prescri\w*|dosage|dose|my symptoms|my chest pain|what medicine|treat my|suicid\w*|overdose)\b', question, re.I):
        return jsonify(answer='This assistant cannot provide individual medical advice. Consult a qualified clinician. If you believe this is an emergency, contact local emergency services immediately.', sources=[], mode='safety')
    try:
        index, warnings, mode = get_index()
        hits = [(chunk,score) for chunk,score in index.search(question) if score >= (0.3 if mode=='semantic' else 0.05)]
        if not hits:
            return jsonify(answer=REFUSAL, sources=[], mode=mode, partial_sources=bool(warnings))
        text = answer(question, hits)
        if configured_key():
            cited = {int(i) for i in re.findall(r'\[Source (\d+)\]', text)}
            if not cited or any(i<1 or i>len(hits) for i in cited) or re.search(r'https?://',text):
                text = 'The generated answer did not include valid source citations. Please review the retrieved excerpts.'
                mode += ' · unverified generation'
        return jsonify(answer=text, mode=mode, partial_sources=bool(warnings), sources=[{'id':i,'url':chunk.url,'text':chunk.text} for i,(chunk,_) in enumerate(hits,1)])
    except APIStatusError as exc:
        # Only known error categories are exposed; never provider bodies or credentials.
        code = getattr(exc, 'code', None)
        body = getattr(exc, 'body', None)
        if isinstance(body, dict):
            detail = body.get('error', body)
            if isinstance(detail, dict):
                code = code or detail.get('code') or detail.get('type')
        if code == 'insufficient_quota':
            return jsonify(error='Your AI provider account has no available API credits. Add API billing or credits, then try again.', error_code='insufficient_quota'), 503
        if exc.status_code == 401:
            return jsonify(error='The AI provider rejected the API key. Check the production key configuration.', error_code='provider_authentication'), 503
        if exc.status_code == 429:
            return jsonify(error='The AI provider rate limit was reached. Please try again later.', error_code='provider_rate_limit'), 503
        return jsonify(error='The AI provider rejected this request. Check the configured model and account access.', error_code='provider_request'), 503
    except (APIConnectionError, APITimeoutError):
        return jsonify(error='Could not reach the AI provider. Please try again later.', error_code='provider_connection'), 503
    except RuntimeError:
        return jsonify(error='No approved public website sources could be loaded. The source website may be blocking access or temporarily unavailable.', error_code='source_unavailable'), 503
    except Exception:
        app.logger.warning('Retrieval or provider request failed; no answer returned.')
        return jsonify(error='Unable to retrieve public sources or reach the AI provider. Please try again later.', error_code='request_failed'), 503


@app.errorhandler(413)
def too_large(_):
    return jsonify(error='Request is too large.'), 413

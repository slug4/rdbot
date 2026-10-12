"""Retrieval-augmented question answering over explicitly allowlisted public RxDx pages."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import time

import numpy as np
import requests
from bs4 import BeautifulSoup

ALLOWED_HOSTS = {"rxdx.in", "www.rxdx.in"}
HEADERS = {"User-Agent": "RxDx-RAG-EducationalDemo/0.1 (manual public-page fetch)"}
REFUSAL = "I couldn't verify that from the indexed RxDx sources. Please contact RxDx directly to confirm."


@dataclass(frozen=True)
class Chunk:
    text: str
    url: str


def validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS and not parsed.username and not parsed.password and parsed.port in {None, 443}
    except ValueError:
        return False


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "form", "noscript"]):
        tag.decompose()
    root = soup.select_one("main, article, [role=main]") or soup.body or soup
    return re.sub(r"\s+", " ", root.get_text(" ", strip=True)).strip()


def chunk_text(text: str, size: int = 850, overlap: int = 140) -> list[str]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size")
    words = text.split()
    result = []
    buffer = []
    for word in words:
        if buffer and len(" ".join(buffer + [word])) > size:
            result.append(" ".join(buffer))
            tail = []
            for prev in reversed(buffer):
                if len(" ".join([prev] + tail)) > overlap:
                    break
                tail.insert(0, prev)
            buffer = tail + [word]
        else:
            buffer.append(word)
    if buffer:
        result.append(" ".join(buffer))
    return result


def load_pages(urls: list[str], timeout: int = 12) -> tuple[list[Chunk], list[str]]:
    chunks, errors = [], []
    robots = {}
    for url in dict.fromkeys(urls):
        url = url.strip()
        if not validate_url(url):
            errors.append(f"Skipped unapproved URL: {url}")
            continue
        try:
            origin = f"https://{urlparse(url).netloc}"
            if origin not in robots:
                rules = requests.get(origin + '/robots.txt', headers=HEADERS, timeout=timeout, allow_redirects=False)
                rules.raise_for_status()
                if rules.status_code != 200 or '<html' in rules.text.lower() or '<!doctype' in rules.text.lower():
                    raise ValueError('Could not verify robots rules')
                robot = RobotFileParser()
                robot.parse(rules.text.splitlines())
                robots[origin] = robot
            robot = robots[origin]
            if not robot.can_fetch(HEADERS['User-Agent'], url):
                raise ValueError('Fetching is disallowed by robots.txt')
            delay = robot.crawl_delay(HEADERS['User-Agent']) or 1
            if delay > 10:
                raise ValueError('Use a scheduled indexer for the required crawl delay')
            time.sleep(delay)
            response = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=False)
            response.raise_for_status()
            if 300 <= response.status_code < 400:
                raise ValueError("Redirects are disabled; approve the destination explicitly")
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                raise ValueError("Page is not HTML")
            if len(response.content) > 2_000_000:
                raise ValueError("Page exceeds 2 MB limit")
            text = extract_text(response.text)
            if len(text) < 80:
                raise ValueError("Not enough readable page text")
            chunks.extend(Chunk(part, url) for part in chunk_text(text))
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{url}: {exc}")
    return chunks, errors


class Retriever:
    def __init__(self, chunks: list[Chunk], model=None):
        self.chunks = chunks
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.model = model
        self.vectors = self._embed([c.text for c in chunks]) if chunks else None

    def _embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.asarray(self.model.encode(texts), dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    def search(self, question: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        if self.vectors is None or not question.strip():
            return []
        query = self._embed([question])[0]
        scores = self.vectors @ query
        top = np.argsort(-scores)[:top_k]
        return [(self.chunks[int(i)], float(scores[i])) for i in top]


def answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    """Extract a short answer from retrieved evidence, without a generative provider."""
    if not hits:
        return REFUSAL
    query=question.lower()
    schedule=bool(re.search(r'\b(?:timings?|schedule|hours|available today)\b',query)) and 'doctor' in query
    home=bool(re.search(r'home[- ]?visits?|home doctor',query))
    if schedule and not home:
        if not re.search(r'\bdr[. ]+\w+',query):
            return 'Please give the doctor’s name and RxDx branch so their published timings can be checked. Current appointment slots need clinic confirmation. [Source 1]'
        return 'I cannot verify that doctor’s current schedule from the retrieved pages. Please confirm the timings with the RxDx branch. [Source 1]'
    terms=set(re.findall(r'\w+',query))-{'the','and','for','with','what','where','does','rxdx','are','can','how','when'}
    candidates=[]
    for source_id,(chunk,score) in enumerate(hits,1):
        for sentence in re.split(r'(?<=[.!?])\s+',chunk.text):
            if len(sentence)<30:
                continue
            words=set(re.findall(r'\w+',sentence.lower()))
            overlap=len(words & terms)
            rank=overlap + score
            candidates.append((rank,source_id,sentence))
    if not candidates:
        return REFUSAL
    _,source_id,text=max(candidates,key=lambda item:item[0])
    if len(text)>350:
        # A labelled extract, not a fabricated summary or medical instruction.
        positions=[m.start() for m in re.finditer(r'\w+',text.lower()) if m.group() in terms]
        start=max(0,(positions[0] if positions else 0)-60)
        if start:
            next_space=text.find(' ',start)
            start=next_space+1 if next_space>=0 else start
        end=min(len(text),start+320)
        if end<len(text):
            last_space=text.rfind(' ',start,end)
            end=last_space if last_space>start else end
        text=('…' if start else '')+text[start:end]+('…' if end<len(text) else '')
    return f'From the RxDx website: “{text}” [Source {source_id}]\nConfirm changing details directly with the clinic.'

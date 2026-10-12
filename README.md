# RxDx public-information Q&A

An independent source-grounded assistant for RxDx Clinics, Bengaluru. It retrieves approved public pages using local semantic embeddings and returns short extracts with citations. It is not affiliated with RxDx.

## No external AI key

The web application requires no OpenAI or other AI-provider key. A bundled, checksum-verified quantized `all-MiniLM-L6-v2` model runs locally with ONNX Runtime. Questions and website text are embedded locally, ranked by cosine similarity, and answered with short extracts from the retrieved evidence.

This is **semantic retrieval with extractive Q&A**, not generative LLM RAG. It does not rewrite arbitrary website passages into new explanations. The short quotes remain source text. Exact doctor timings require verified doctor/branch information; the assistant asks for clarification or clinic confirmation instead of inferring clinic schedules from home-care hours. Individual medical advice is out of scope.

The interface lists 30 question suggestions. `faq.py` stores questions only; answers are not prewritten FAQ entries. Every operational question searches the source index. The same route handles typed and clicked questions.

## Run and test

Requires Python 3.12.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
python -m flask --app app run --host 0.0.0.0 --port 8502
python -m pytest -q
```

Routes: `/` for the interface, `/health` for health and keyless architecture status, `/api/faqs` for question suggestions, and JSON POST `/api/answer` with `{"question":"Does RxDx offer home visits?"}`.

## Sources, caching and timing

Six approved official HTTPS pages in `sources.txt` cover doctors, branches, contacts, labs/radiology, home care and health packages. Robots checks and TLS verification remain enabled. Redirects are rejected; no portals or patient records are accessed. No full scraped website copies are committed.

The first request fetches pages and embeds up to eight chunks per page. The index is cached for 10 minutes within a warm server instance. A new instance or expired cache fetches sources again, so first responses are slower than repeated requests. Vercel can route requests to different instances; a repeat is not guaranteed to hit the same cache. Website access failures are reported and partial coverage is labelled. A scheduled ingestion process and durable index are appropriate for larger workloads.

The local model assets, upstream revision, license and SHA-256 values are under `models/`. They are bundled so requests never download model weights. Do not replace expected checksums or disable verification. The portable AVX2 quantized ONNX export uses the CPU runtime. No Torch or Streamlit is installed in the web runtime.

## Deployment

`app.py` exports a top-level Flask `app` for Vercel's Python runtime. Import the repository root on `main`; runtime dependencies are in `requirements.txt`. Run the manual Production smoke check workflow for live retrieval, source citations and question catalog validation. Mocked tests alone do not prove production behavior.

The optional `streamlit_app.py` interface needs `requirements-streamlit.txt` and a persistent local server; it is not the Vercel entry point.

## Limitations

This educational prototype may return imperfectly formatted excerpts or miss relevant content. Doctor rosters, test availability, charges and schedules can change; confirm directly with RxDx. The index is limited and does not guarantee answers to every possible question. Do not enter patient details or use it for diagnosis, prescriptions, emergencies or booking confirmation. Public production use needs access controls, request limits and suitable privacy measures.

# RxDx RAG Q&A Bot

An independent public-information assistant for RxDx Clinics, Bengaluru. Covers clinic locations, services and contacts using approved official public pages. It is not affiliated with RxDx and does not diagnose, prescribe, book appointments or confirm current availability.

## Vercel web application

`app.py` exports the top-level Flask `app` required by Vercel's Python runtime. Import this repository's **main** branch with the repository root as the root directory. Runtime dependencies in `requirements.txt` deliberately exclude Streamlit, Torch and local embedding weights. No custom build command is required. Python is pinned to 3.12.

Routes:

- `/`: browser Q&A interface
- `/health`: lightweight health/configuration status (no provider calls)
- `/api/answer`: JSON POST containing `{"question":"Does RxDx offer home visits?"}`

Add `OPENAI_API_KEY` in Vercel environment settings to enable hosted `text-embedding-3-small` semantic retrieval and chat generation. `RXDX_API_KEY` is also supported, particularly in Codex cloud where `OPENAI_` variable names are reserved. Optional `OPENAI_MODEL` defaults to `gpt-4.1-mini`; `OPENAI_BASE_URL` must be HTTPS and support both embeddings and chat completions.

Every operational answer goes through semantic retrieval and AI generation. A configured API key and usable provider access are required. There is no prewritten FAQ-answer or keyword-answer fallback. Public text chunks and user questions are sent to the provider, which may incur costs.

The first question fetches approved public pages and checks robots rules. The index is cached for 10 minutes within a warm server instance; Vercel can discard it between requests. Source failures are marked as partial coverage. The index is limited to six configured pages and up to eight chunks per page; this demo does not promise exhaustive clinic coverage. For production, use a scheduled ingestion pipeline and a persistent vector store to avoid request-time indexing and serverless timeouts.

## Develop and test

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Configure API credentials securely if desired.
python -m flask --app app run --host 0.0.0.0 --port 8502
python -m pytest -q
```

The tests cover the actual Flask entry point, UI/static routes, input limits, safety refusals, semantic retrieval, source citations, provider errors, index caching and robots enforcement. They use synthetic sources and mocked providers. Live model generation requires a separately configured API key and is not covered by those mocked tests.

## Optional local Streamlit interface

The original interface is preserved in `streamlit_app.py`. It uses local Sentence Transformers and requires the heavier optional dependencies:

```sh
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

Streamlit requires a persistent Python server and is not the Vercel entry point. Its local semantic model must be downloaded before use. Do not install the optional Streamlit requirements in the Vercel runtime.

## Sources and safety

Approved public HTTPS URLs are listed in `sources.txt`. Fetching rejects redirects and robots denials; it does not access portals or patient records. Review website terms for the intended use. The web application does not accept arbitrary source URLs from users.

Do not submit patient records or personal medical details. Provider-enabled requests send the question and public excerpts to the provider. Answers can be wrong: check supporting excerpts and confirm clinic details directly with RxDx. Citation validation checks source IDs and formatting, not semantic entailment of every generated claim. The keyword medical guard is a limited precaution, not a clinical safety classifier.

Before exposing provider-enabled public access, add authentication, durable rate limits, spending controls and appropriate privacy measures. This educational prototype has no such production access controls.

## Project 2

An AI recruitment assistant has been proposed; its implementation is not part of this deployment fix.

## Simple questions and answers

The interface lists 30 questions about doctor timings, doctor lists, appointments, tests, reports, health checks, home care, contacts and branches. `faq.py` contains question suggestions only, with no stored answers. Clicking a suggestion submits the question to the same RAG endpoint as a typed question. Answers use plain language, 1–3 short sentences or at most five bullet points, with source citations. Diagnostic-service questions are permitted; individual diagnosis and treatment requests are refused.

Six approved public source pages cover the doctor directory, centres, contacts, lab/radiology, home care and health packages. Exact doctor timings need a named doctor and branch plus a published matching schedule. Otherwise the bot must say it cannot verify them and ask the clinic to confirm. It must not generalize home-visit hours to all clinic doctors. The question list cannot promise every possible question or complete scheduling data. Add verified source documents for branch-specific coverage.

Run the manual Production smoke check workflow after deployment to verify the catalog and live cited RAG answers. The last observed hosted-provider rate limit remains an external blocker until provider access permits requests; a passing deployment or mocked unit suite alone does not verify generated live answers.

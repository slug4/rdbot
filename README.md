# RxDx RAG Q&A Bot

A source-grounded Q&A assistant for **public information about RxDx Clinics, Bengaluru**. Built with Streamlit, sentence-transformer embeddings, and a configurable OpenAI-compatible chat model.

**Scope:** clinic locations, services, appointments, contacts and other information published on the RxDx website. **Not** a diagnostic or patient-record chatbot; not affiliated with RxDx.

## Architecture

Official RxDx pages -> fetch/clean -> text chunks -> local embeddings -> cosine retrieval -> cited context -> chat-model answer -> source links.

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (runs locally, downloaded once)
- Generation: OpenAI-compatible chat API (set `OPENAI_API_KEY`; choose supported `OPENAI_MODEL`)
- UI: Streamlit
- Sources: `sources.txt` (allowlisted rxdx.in HTTPS URLs)
- No patient data or secrets are stored in Git.

## Run locally

```bash
git clone https://github.com/slug4/rdbot.git
cd rdbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY / OPENAI_MODEL
streamlit run app.py
```

On first run, click **Refresh knowledge base**, then ask a question. The initial model download needs internet. If you don't configure a chat API, the app shows retrieved excerpts and links rather than fabricating an answer.

## Example questions

- Where is the RxDx Whitefield clinic?
- Does RxDx offer home visits?
- How do I contact RxDx?
- What kinds of diagnostic services are listed?

## Safety and limitations

- Uses only public RxDx website pages; don't ingest patient records, protected portals or personal data.
- Information may change; confirm availability, doctor schedules, costs and booking status directly with RxDx.
- Do not use this bot for diagnosis, emergency advice, prescriptions, or individualized treatment.
- External chat providers receive the user's question and selected public website passages when enabled.
- Respect the website's robots rules/terms and avoid high-frequency crawling. Refresh manually, not continuously.
- Answers should cite page URLs; when sources do not support an answer, the bot must say it cannot verify it.
- This is an educational prototype, not an official RxDx service.

## Development

```bash
python -m pytest -q
```

## Next

Add approved FAQs or PDFs, retrieval quality evaluation, monitoring and a controlled deployment. Keep medical and booking claims source-grounded.

## Implementation and validation notes

Python 3.12 is the tested runtime. For CPU-only Linux development, install the CPU wheel first to avoid unnecessary GPU dependencies:

```sh
python -m pip install 'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

`rxdx/core.py` validates sources, chunks pages and checks generated citation IDs against retrieved passages. `rxdx/crawl.py` enforces HTTPS source allowlists, bounds downloads, checks robots rules and rejects disallowed redirects before following them. `rxdx/search.py` uses normalized sentence-transformer vectors and cosine similarity. The Streamlit UI requires explicit source-terms confirmation before indexing.

Tests use mocked website responses, a deterministic fake embedding model and a mocked generation provider. They test application behavior without representing live RxDx facts. Citation checks verify source identity and formatting; they do not prove every generated claim is entailed by a source. The keyword medical guard is an additional precaution, not a comprehensive clinical-safety classifier.

The index and conversation live in Streamlit server-session memory; restarting loses them. Refresh only after reviewing source terms and robots permissions. Only the official homepage is initially configured; add verified clinic/service URLs to `sources.txt` for useful coverage. No website content is bundled.

### Hosting

Streamlit requires a persistent Python process and WebSockets. Deploy this version to a Python/Streamlit-capable host such as Streamlit Community Cloud or a container host, with startup command `streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true`. Configure API credentials in host secret settings, never in Git. Public use additionally needs access controls, rate limits, privacy review and live retrieval/answer evaluation.

Vercel does not natively host this Streamlit server. A Vercel frontend would need a separately hosted Python backend and an API integration; that architecture is not included here.

### Project 2

An AI recruitment assistant has been proposed, but its scope and implementation are pending. It should provide job-related evidence and human review, avoiding automated hiring decisions or protected-characteristic inferences.

In Codex cloud settings, `OPENAI_` variable names are reserved. Use the supported `RXDX_API_KEY` secret binding to `api.openai.com`; the app accepts it before falling back to `OPENAI_API_KEY`. Never paste keys into chat. For another provider, configure its HTTPS endpoint and securely bind a credential to that provider's host.

### Current validation

The implemented suite passes 28 tests, including Streamlit startup/chat behavior, source display, crawler rules and citation guards. The running Streamlit server responds successfully to its health request. Live RxDx homepage extraction succeeded after a successful robots check. Full real-model semantic indexing is not yet validated: the Hugging Face weights CDN is currently blocked by the cloud network policy. Live AI generation is also untested because an API key is absent. No public deployment has been completed.

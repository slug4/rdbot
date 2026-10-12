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

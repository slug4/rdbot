# Codex instructions for rdbot

Build and improve the RxDx public-information RAG chatbot.

- Do not commit credentials, API keys, patient records, or scraped full website copies.
- Only crawl approved public URLs under rxdx.in; no portal logins, no personal data.
- Keep answers strictly grounded in cited passages. Say when evidence is insufficient.
- No diagnosis, prescriptions, emergency triage or medical decisions.
- Prefer small, testable functions and add tests for retrieval/safety changes.
- Run `python -m pytest -q` before proposing merges when dependencies are available.
- Keep the project demo runnable with documented env variables.
- Never claim affiliation with RxDx or guarantee prices/availability.

"""Streamlit front end for the RxDx public-information RAG demo."""
import streamlit as st
from dotenv import load_dotenv
from rag import Retriever, answer, load_pages

load_dotenv()
st.set_page_config(page_title="RxDx Q&A | RAG Demo", page_icon="🏥", layout="centered")
st.title("RxDx Q&A")
st.caption("Independent educational RAG demonstration • Public RxDx website information only")
st.warning("Not for diagnosis, prescriptions, emergencies, or personal medical questions. Confirm current details with RxDx.")

with open("sources.txt", encoding="utf-8") as source_file:
    sources = [line.strip() for line in source_file if line.strip() and not line.startswith("#")]


@st.cache_resource(show_spinner="Fetching approved pages and generating embeddings…")
def build_index(source_urls: tuple[str, ...]):
    chunks, errors = load_pages(list(source_urls))
    return Retriever(chunks) if chunks else None, errors, len(chunks)


if st.sidebar.button("Refresh knowledge base"):
    build_index.clear()
st.sidebar.write("Public source URLs:", len(sources))
try:
    retriever, errors, count = build_index(tuple(sources))
except Exception as exc:
    st.error(f"Index initialization failed: {exc}")
    st.stop()
st.sidebar.write("Indexed passages:", count)
if errors:
    with st.sidebar.expander("Page fetch warnings"):
        for error in errors:
            st.write(error)

question = st.chat_input("Ask about RxDx locations, services, appointments…")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        if not retriever:
            st.error("No valid source pages indexed. Review source URLs and network connectivity.")
        else:
            hits = retriever.search(question)
            try:
                st.write(answer(question, hits))
            except Exception as exc:
                st.error(f"Generation unavailable: {type(exc).__name__}. Review the cited source excerpts below.")
            with st.expander("Retrieved evidence and original links"):
                for i, (chunk, score) in enumerate(hits, 1):
                    st.markdown(f"**[Source {i}]({chunk.url})** — relevance {score:.2f}")
                    st.write(chunk.text)

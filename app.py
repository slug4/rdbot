import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from rxdx.core import answer_question, chunk_pages, medical_request
from rxdx.crawl import crawl

load_dotenv()
api_key = os.getenv('RXDX_API_KEY') or os.getenv('OPENAI_API_KEY')
st.set_page_config(page_title='RxDx · Public information assistant', page_icon='🌿', layout='wide')
st.title('RxDx public information assistant')
st.caption('Source-grounded answers about clinics, services and contacts. Educational prototype · not affiliated with RxDx.')
st.info('Do not enter patient information. This assistant cannot diagnose, prescribe, book appointments or verify live availability.')
with st.sidebar:
    st.header('Knowledge sources')
    urls = [line.strip() for line in Path(__file__).with_name('sources.txt').read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    for url in urls:
        st.write(url)
    approved = st.checkbox('I have reviewed source terms and have permission for this indexing use.')
    if st.button('Refresh knowledge base', disabled=not approved):
        try:
            with st.spinner('Checking robots rules, fetching pages and building semantic index…'):
                from rxdx.search import SemanticIndex
                index = SemanticIndex(chunk_pages(crawl(urls)))
                st.session_state.index = index
                st.session_state.messages = []
            st.success(f'Indexed {len(index.passages)} passages.')
        except Exception:
            st.error('Indexing failed. Check source URLs, robots permissions, network access and model availability. No incomplete index was installed.')
    st.caption('Manual refresh only. Documents and chat remain in this server session. Relevant excerpts and questions go to the configured AI provider when enabled.')
    st.write('AI answers enabled' if api_key else 'No API key: retrieval excerpts only')
for message in st.session_state.get('messages', []):
    with st.chat_message(message['role']):
        st.write(message['text'])
        for passage in message.get('sources', []):
            with st.expander(f'[{passage.id}] {passage.url}'):
                st.write(passage.text)
                st.link_button('Open official source', passage.url)
question = st.chat_input('Ask about RxDx locations, services or contacts', max_chars=2000)
if question:
    st.session_state.setdefault('messages', []).append({'role':'user','text':question})
    with st.chat_message('user'):
        st.write(question)
    with st.chat_message('assistant'):
        if 'index' not in st.session_state and not medical_request(question):
            st.warning('Refresh the knowledge base first. No RxDx facts are preloaded.')
        else:
            try:
                with st.spinner('Finding supporting sources…'):
                    passages = [] if medical_request(question) else st.session_state.index.search(question)
                    generate = None
                    if api_key:
                        base = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
                        if not base.startswith('https://'):
                            raise ValueError('The AI endpoint must use HTTPS.')
                        client = OpenAI(api_key=api_key, base_url=base, timeout=30, max_retries=1)
                        def generate(system, payload):
                            response = client.chat.completions.create(model=os.getenv('OPENAI_MODEL','gpt-4.1-mini'), messages=[{'role':'system','content':system},{'role':'user','content':payload}], response_format={'type':'json_object'}, max_tokens=800)
                            return response.choices[0].message.content
                    result = answer_question(question, passages, generate)
                st.write(result['answer'])
                st.caption('Mode: ' + result['mode'])
                for passage in result['sources']:
                    with st.expander(f'[{passage.id}] {passage.url}'):
                        st.write(passage.text)
                        st.link_button('Open official source', passage.url)
                st.session_state.messages.append({'role':'assistant','text':result['answer'],'sources':result['sources']})
            except Exception:
                st.error('Could not complete this request. Check provider configuration and try again. No answer was generated.')

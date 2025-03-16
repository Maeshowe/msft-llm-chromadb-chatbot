import streamlit as st
import yaml
import logging
import os
import time
from langchain_community.llms import LlamaCpp
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Loggolás beállítása
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(logs_dir, 'streamlit_chat.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    encoding='utf-8'
)

@st.cache_resource
def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_llm(config):
    model_conf = config['llm']['models'][config['llm']['active_model']]
    return LlamaCpp(
        model_path=model_conf["model_path"],
        temperature=model_conf["temperature"],
        max_tokens=model_conf["max_tokens"],
        n_ctx=model_conf["context_size"],
        n_batch=model_conf["n_batch"],
        n_threads=model_conf["n_threads"],
        verbose=config["llm"].get("verbose", False),
        n_gpu_layers=0
    )

@st.cache_resource
def load_retriever(config):
    embeddings = HuggingFaceEmbeddings(model_name=config["embedding"]["model"])
    db = Chroma(persist_directory=config['paths']['vectorstore_dir'], embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": config["similarity_search"]["top_k"]})

def load_chain(config):
    llm = load_llm(config)
    retriever = load_retriever(config)
    prompt_template = config['prompt']['template']
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    return RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=config["metadata"]["show_source_in_response"],
        chain_type_kwargs={"prompt": prompt}
    )

# Streamlit UI
st.set_page_config(page_title="Dokumentum-alapú Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Dokumentum-alapú Chatbot")

config = load_config()
qa_chain = load_chain(config)  # Fontos: itt "load_chain", nem "load_llm"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat előzmények megjelenítése
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kérdés bekérése
if query := st.chat_input("Írd be a kérdésed..."):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Gondolkodom..."):
            start_time = time.time()
            result = qa_chain.invoke(query)
            inference_time = time.time() - start_time

            # Fontos javítás itt történik:
            response = result['result'] if isinstance(result, dict) else result
            sources = [doc.metadata.get('source', 'n/a') for doc in result['source_documents']] if isinstance(result, dict) and 'source_documents' in result else []

            st.markdown(response)

            if sources and config["metadata"]["show_source_in_response"]:
                st.caption(f"Forrás(ok): {', '.join(set(sources))}")

            # Részletes logolás
            if not sources:
                logging.warning(f"Nincs találat: {query}")

            logging.info(f"Kérdés: {query}")
            logging.info(f"Válasz: {response}")
            logging.info(f"Források: {sources if sources else 'Nincs forrás'}")
            logging.info(f"Teljes válaszidő: {inference_time:.2f} másodperc")

            # Válasz hozzáadása az előzményekhez
            st.session_state.messages.append({"role": "assistant", "content": response})

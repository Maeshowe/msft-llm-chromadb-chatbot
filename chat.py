import yaml
import logging
import os
import time
from langchain_community.llms import LlamaCpp
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Loggolás beállítása
logs_dir = config['paths']['logs_dir']
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(logs_dir, 'chat.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    encoding='utf-8'
)

model_conf = config['llm']['models'][config['llm']['active_model']]
vectorstore_path = config['paths']['vectorstore_dir']

llm = LlamaCpp(
    model_path=model_conf["model_path"],
    temperature=model_conf["temperature"],
    max_tokens=model_conf["max_tokens"],
    n_ctx=model_conf["context_size"],
    n_batch=model_conf["n_batch"],
    n_threads=model_conf["n_threads"],
    verbose=config["llm"].get("verbose", False),
    n_gpu_layers=0
)

embeddings = HuggingFaceEmbeddings(model_name=config["embedding"]["model"])
db = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": config["similarity_search"]["top_k"]})

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=config['prompt']['template']
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

print("🚀 Chatbot elindult (kilépés: 'exit')")

while True:
    query = input("\n👉 Kérdésed: ")
    if query.lower() in ['exit', 'quit']:
        break

    start_time = time.time()  # ✅ hiányzó definíció hozzáadva
    result = qa_chain.invoke(query)
    inference_time = time.time() - start_time

    response = result['result']
    sources = [doc.metadata.get('source', 'n/a') for doc in result['source_documents']]

    if not sources:
        logging.warning(f"Nincs találat: {query}")

    logging.info(f"Kérdés: {query}")
    logging.info(f"Válasz: {response}")
    logging.info(f"Források: {sources}")
    logging.info(f"Inferencia idő: {inference_time:.2f}s")

    print(f"\n🤖 Válasz: {response}")
    print(f"📚 Forrás(ok): {', '.join(set(sources))}")

import yaml
import logging
import os
import time
from langchain_community.llms import LlamaCpp
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from tqdm import tqdm

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

chain_type = config["chain"]["type"]

if chain_type == "retrieval_qa":
    chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
elif chain_type == "conversational":
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
else:
    raise ValueError("Nem támogatott lánctípus van megadva a konfigurációban!")

print(f"🚀 Chatbot elindult [{config['chain']['type']}] (kilépés: 'exit')")

chat_history = []

while True:
    query = input("\n👉 Kérdésed: ")
    if query.lower() in ('exit', 'quit'):
        break

    with tqdm(total=1, desc="Válasz generálása") as pbar:
        start_time = time.time()

        if config["chain"]["type"] == "retrieval_qa":
            result = chain.invoke(query)
            response = result['result']
            sources = [doc.metadata.get('source', 'n/a') for doc in result['source_documents']]
        else:  # conversational
            result = chain.invoke({"question": query, "chat_history": chat_history})
            response = result['answer']
            sources = [doc.metadata.get('source', 'n/a') for doc in result.get('source_documents', [])]

        inference_time = time.time() - start_time
        pbar.update(1)

    if sources:
        response_with_sources = f"{response}\n📚 Forrás(ok): {', '.join(set(sources))}"
    else:
        response += "\n(Nincs elérhető forrás.)"

    print(f"\n🤖 Válasz: {response}")
    if sources:
        print(f"📚 Forrás(ok): {', '.join(set(sources))}")

    chat_history.append((query, response))

    logging.info(f"Kérdés: {query}")
    logging.info(f"Válasz: {response}")
    logging.info(f"Források: {sources if sources else 'Nincs forrás'}")
    logging.info(f"Inferencia idő: {inference_time:.2f}s")
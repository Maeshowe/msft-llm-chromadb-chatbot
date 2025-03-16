import os
import yaml
import logging
import time
from langchain_community.document_loaders import (
    PyMuPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

# Konfiguráció betöltése
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Loggolás beállítása
logs_dir = config['paths']['logs_dir']
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(logs_dir, 'process.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    encoding='utf-8'
)

docs_path = config['paths']['docs_dir']
vectorstore_path = config['paths']['vectorstore_dir']
embeddings = HuggingFaceEmbeddings(model_name=config["embedding"]["model"])

logging.info("🔄 Dokumentum feldolgozás elkezdődött.")

documents = []
start_time_total = time.time()
for root, dirs, files in os.walk(docs_path):
    for file in tqdm(files, desc="Dokumentumok betöltése"):
        start_time = time.time()
        file_path = os.path.join(root, file)
        if file.endswith('.pdf'):
            loader = PyMuPDFLoader(file_path)
        elif file.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path)
        elif file.endswith('.txt') or file.endswith('.md'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            continue

        loaded_docs = loader.load()
        elapsed = time.time() - start_time
        logging.info(f"📄 Betöltve: {file} | Dokumentumok: {len(loaded_docs)} | Idő: {elapsed:.2f} s")
        documents.extend(loaded_docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splitted_docs = text_splitter.split_documents(documents)

logging.info(f"🔄 {len(splitted_docs)} dokumentumrészlet generálva, embeddingelés kezdődik...")
start_embedding_time = time.time()
Chroma.from_documents(splitted_docs, embeddings, persist_directory=vectorstore_path)
embedding_duration = time.time() - start_embedding_time  # Ez a sor volt a hibás

logging.info(f"✅ Embeddingek elkészültek ({embedding_duration:.2f} másodperc).")
logging.info(f"✅ Teljes folyamat idő: {time.time() - start_time_total:.2f} másodperc.")
print("✅ Dokumentumok feldolgozva, embeddingek létrehozva és elmentve.")

# vector_manager.py

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import json
import os

CHROMA_DIR = "./vector_store"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class VectorManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        # load or initialize vector DB
        if os.path.exists(CHROMA_DIR):
            self.db = Chroma(persist_directory=CHROMA_DIR, embedding_function=self.embeddings)
        else:
            self.db = Chroma.from_texts([], embedding=self.embeddings, persist_directory=CHROMA_DIR)

    def index_documents(self, jsonl_path="data/parts.jsonl"):
        docs = []
        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                content = f"{item['name']}\nPart#: {item['part_number']}\nBrand: {item['brand']}\nCompatibility: {','.join(item['compatibility'])}\nInstallation: {item['installation_steps']}\nTroubleshooting: {','.join(item['troubleshooting'])}"
                docs.append(Document(page_content=content, metadata=item))
        self.db.add_documents(docs)
        self.db.persist()

    def get_retriever(self, k=4):
        return self.db.as_retriever(search_kwargs={"k": k})
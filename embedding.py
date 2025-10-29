from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import httpx
import urllib3
from dotenv import load_dotenv
import os
import requests

class EmbeddingPipeline:
    def __init__(self, model_name: str = "azure/genailab-maas-text-embedding-3-large", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # self.model = SentenceTransformer(model_name)

        # Load the API key from .env
        load_dotenv()
        API_KEY = os.getenv("sk-NeLegTs4FPkmxtUdqEzTRA")

        requests.packages.urllib3.disable_warnings()
        session = requests.Session()
        session.verify = False
        requests.get = session.get

        # Cache dir for tokens
        tiktoken_cache_dir = "./token"
        os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
        client = httpx.Client(verify=False)
        # Disable SSL warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Disable SSL verification in httpx client
        # client = httpx.Client(verify=False)
        self.model = OpenAIEmbeddings(
            base_url="https://genailab.tcs.in",
            model="azure/genailab-maas-text-embedding-3-large",
            api_key="sk-evrJ0zYSH6IaCJAOSQYxgQ",  # Replace with your actual API key
            http_client=client,
            )
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.embed_documents(texts)
        print(f"[INFO] Embeddings shape: {embeddings}")
        return embeddings

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import httpx
import urllib3
from dotenv import load_dotenv
import os
import requests


load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gemma2-9b-it"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

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

        self.llm = ChatOpenAI(
            base_url="https://genailab.tcs.in",
            model="azure/genailab-maas-gpt-4o-mini",
            api_key="sk-evrJ0zYSH6IaCJAOSQYxgQ",  # Replace with your actual API key
            http_client=client
        )
        
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    # query = "What is summary of susan_christopher?"
    # summary = rag_search.search_and_summarize(query, top_k=3)
    # print("Summary:", summary)
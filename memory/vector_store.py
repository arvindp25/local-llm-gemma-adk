import logging
from typing import Any, Dict, List, Optional
import httpx
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from config.settings import get_settings

logger = logging.getLogger(__name__)


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB compatible embedding function using local Ollama nomic-embed-text."""

    def __init__(self, model_name: Optional[str] = None, api_base: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.api_base = (api_base or settings.ollama_api_base).rstrip("/")

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        with httpx.Client(timeout=30.0) as client:
            for text in input:
                try:
                    resp = client.post(
                        f"{self.api_base}/api/embeddings",
                        json={"model": self.model_name, "prompt": text},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    embeddings.append(data.get("embedding", []))
                except Exception as e:
                    logger.error(f"Failed to generate embedding with Ollama for text: {e}")
                    # Return zero vector fallback (768 dim for nomic-embed-text)
                    embeddings.append([0.0] * 768)
        return embeddings


class SemanticMemoryStore:
    """Persistent local RAG vector store using ChromaDB and Ollama embeddings."""

    def __init__(self, collection_name: str = "agent_persistent_knowledge"):
        settings = get_settings()
        self.client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
        self.embedding_fn = OllamaEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Stores a document or learned fact into vector memory."""
        clean_metadata = metadata or {"source": "manual"}
        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[clean_metadata],
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs semantic similarity search to retrieve relevant long-term memories."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        hits = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            for doc, meta, dist in zip(docs, metas, distances):
                hits.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                })
        return hits

    def count(self) -> int:
        return self.collection.count()

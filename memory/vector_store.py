import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from core.interfaces import BaseVectorStore
from config.settings import get_settings

logger = logging.getLogger(__name__)

# Shared client pool to prevent concurrent lock conflicts on embedded Qdrant storage
_qdrant_client_pool: Dict[str, Any] = {}


def _cleanup_qdrant_clients():
    for client in list(_qdrant_client_pool.values()):
        try:
            client.close()
        except Exception:
            pass
    _qdrant_client_pool.clear()


import atexit
atexit.register(_cleanup_qdrant_clients)


def _get_shared_qdrant_client(
    path: Optional[str] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    in_memory: bool = False,
) -> Any:
    """Returns a shared or dedicated QdrantClient instance avoiding storage lock issues."""
    from qdrant_client import QdrantClient

    if in_memory:
        return QdrantClient(":memory:")

    if url:
        key = f"url:{url}"
        if key not in _qdrant_client_pool:
            _qdrant_client_pool[key] = QdrantClient(url=url, api_key=api_key)
        return _qdrant_client_pool[key]

    resolved_path = str(Path(path).resolve())
    if resolved_path not in _qdrant_client_pool:
        _qdrant_client_pool[resolved_path] = QdrantClient(path=resolved_path)
    return _qdrant_client_pool[resolved_path]


def _normalize_point_id(doc_id: str) -> str:
    """Ensures doc_id is converted into a valid UUID string required by Qdrant."""
    try:
        return str(uuid.UUID(doc_id))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))


class OllamaEmbeddingFunction:
    """Embeds text using local Ollama nomic-embed-text (compatible with ChromaDB and Qdrant)."""

    def __init__(self, model_name: Optional[str] = None, api_base: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.api_base = (api_base or settings.ollama_api_base).rstrip("/")

    def name(self) -> str:
        return "ollama_embedding_function"

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        # Fast 1.0s connection timeout so offline detection is instantaneous
        timeout = httpx.Timeout(10.0, connect=1.0)
        with httpx.Client(timeout=timeout) as client:
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
                    logger.debug(f"Failed to generate embedding with Ollama for text: {e}")
                    # Return zero vector fallback (768 dim for nomic-embed-text)
                    embeddings.append([0.0] * 768)
        return embeddings


class QdrantMemoryStore(BaseVectorStore):
    """Production vector store using Qdrant (supports embedded disk, in-memory, or remote server)."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        in_memory: bool = False,
        embedding_fn: Optional[OllamaEmbeddingFunction] = None,
        vector_size: int = 768,
    ):
        from qdrant_client import models

        settings = get_settings()
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.vector_size = vector_size
        self.embedding_fn = embedding_fn or OllamaEmbeddingFunction()

        target_path = str(path or settings.qdrant_path)
        self.client = _get_shared_qdrant_client(
            path=target_path,
            url=url or settings.qdrant_url,
            api_key=api_key or settings.qdrant_api_key,
            in_memory=in_memory,
        )

        # Ensure collection exists with proper dimensions
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name} (dim={self.vector_size})")

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Stores a document or learned fact into Qdrant."""
        from qdrant_client import models

        clean_metadata = metadata or {"source": "manual"}
        vectors = self.embedding_fn([text])
        vector = vectors[0] if vectors and len(vectors[0]) == self.vector_size else [0.0] * self.vector_size

        point_id = _normalize_point_id(doc_id)
        payload = {
            "text": text,
            "doc_id": doc_id,
            "metadata": clean_metadata,
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
            wait=True,
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs semantic similarity search to retrieve relevant memories."""
        query_vectors = self.embedding_fn([query])
        query_vec = query_vectors[0] if query_vectors and len(query_vectors[0]) == self.vector_size else [0.0] * self.vector_size

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
        )

        hits: List[Dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            content = payload.get("text", "")
            meta = payload.get("metadata", {})
            orig_doc_id = payload.get("doc_id", str(point.id))
            score = float(point.score) if point.score is not None else 0.0
            hits.append({
                "content": content,
                "doc_id": orig_doc_id,
                "metadata": meta,
                "score": score,
                "distance": max(0.0, 1.0 - score),
            })
        return hits

    def count(self) -> int:
        """Returns total documents stored in this collection."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count


class ChromaMemoryStore(BaseVectorStore):
    """ChromaDB implementation of BaseVectorStore."""

    def __init__(self, collection_name: str = "agent_persistent_knowledge"):
        import chromadb

        settings = get_settings()
        self.client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
        self.embedding_fn = OllamaEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        clean_metadata = metadata or {"source": "manual"}
        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[clean_metadata],
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        hits: List[Dict[str, Any]] = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            for doc, meta, dist in zip(docs, metas, distances):
                hits.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                    "score": max(0.0, 1.0 - dist),
                })
        return hits

    def count(self) -> int:
        return self.collection.count()


class SemanticMemoryStore(BaseVectorStore):
    """Unified Semantic Memory Store facade that delegates to Qdrant or ChromaDB based on settings."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        backend: Optional[str] = None,
        **kwargs,
    ):
        settings = get_settings()
        self.backend = backend or settings.vector_db_type
        self.collection_name = collection_name or settings.qdrant_collection_name

        if self.backend == "qdrant":
            self._store: BaseVectorStore = QdrantMemoryStore(
                collection_name=self.collection_name,
                **kwargs,
            )
        else:
            self._store = ChromaMemoryStore(
                collection_name=self.collection_name,
            )

    @property
    def store(self) -> BaseVectorStore:
        return self._store

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._store.add_document(doc_id=doc_id, text=text, metadata=metadata)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return self._store.search(query=query, top_k=top_k)

    def count(self) -> int:
        return self._store.count()


def migrate_chroma_to_qdrant(
    chroma_store: Optional[ChromaMemoryStore] = None,
    qdrant_store: Optional[QdrantMemoryStore] = None,
) -> int:
    """Migrates existing memories from ChromaDB into Qdrant."""
    src = chroma_store or ChromaMemoryStore()
    dst = qdrant_store or QdrantMemoryStore()

    data = src.collection.get()
    migrated_count = 0
    if data and data.get("ids"):
        ids = data["ids"]
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        for i, doc_id in enumerate(ids):
            text = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            if text:
                dst.add_document(doc_id=doc_id, text=text, metadata=meta)
                migrated_count += 1

    logger.info(f"Successfully migrated {migrated_count} records from ChromaDB to Qdrant.")
    return migrated_count

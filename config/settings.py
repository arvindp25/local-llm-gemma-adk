from pathlib import Path
from typing import Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Runtime & Inference
    ollama_api_base: str = Field(default="http://127.0.0.1:11434", description="Ollama API base URL")
    primary_model: str = Field(default="gemma4:e4b", description="Primary local LLM model tag")
    fallback_model: str = Field(default="gemma2:9b", description="Fallback local LLM model tag")
    embedding_model: str = Field(default="nomic-embed-text", description="Embedding model for RAG & vector memory")
    
    # Model Provider Selection
    provider_type: Literal["adk_litellm", "adk_ollama_native", "direct_ollama"] = Field(
        default="adk_litellm",
        description="Model integration provider: adk_litellm or adk_ollama_native",
    )

    # Context & Generation (Guarded for 8GB RTX 5050 VRAM)
    num_ctx: int = Field(default=8192, description="Guarded context window length in tokens")
    max_tokens: int = Field(default=2048, description="Maximum completion tokens per turn")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Nucleus sampling probability")

    # Hardware Optimizations (RTX 5050 8GB VRAM)
    ollama_num_parallel: int = Field(default=1, description="OLLAMA_NUM_PARALLEL slots")
    ollama_flash_attention: bool = Field(default=True, description="Enable Flash Attention")
    ollama_kv_cache_type: str = Field(default="q8_0", description="Quantized KV cache (q8_0 saves ~40% VRAM)")
    vram_headroom_mb: int = Field(default=1000, description="Minimum free VRAM safeguard buffer in MiB")

    # Storage Paths & Vector Database
    vector_db_type: Literal["qdrant", "chroma"] = Field(
        default="qdrant",
        description="Vector database backend: qdrant or chroma",
    )
    data_dir: Path = Field(default=BASE_DIR / ".data")
    chroma_db_dir: Path = Field(default=BASE_DIR / ".data" / "chroma")
    qdrant_path: Path = Field(default=BASE_DIR / ".data" / "qdrant", description="Local directory for embedded Qdrant")
    qdrant_url: Optional[str] = Field(default=None, description="Optional remote Qdrant URL (e.g. http://localhost:6333)")
    qdrant_api_key: Optional[str] = Field(default=None, description="Optional Qdrant API key")
    qdrant_collection_name: str = Field(default="agent_persistent_knowledge", description="Default Qdrant collection name")
    memory_db_path: Path = Field(default=BASE_DIR / ".data" / "episodic_memory.json")

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        if not self.qdrant_url:
            self.qdrant_path.mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings

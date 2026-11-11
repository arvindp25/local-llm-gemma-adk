from typing import Dict, Type, Optional
from .interfaces import BaseLLMProvider, BaseEmbeddingProvider, ModelMetadata
from .exceptions import ModelNotFoundError
from config.settings import get_settings


class ModelRegistry:
    """Registry of known local models, capabilities, and hardware footprints."""
    
    _MODELS: Dict[str, ModelMetadata] = {
        "gemma4:e4b": ModelMetadata(
            name="gemma4:e4b",
            advertised_context_window=131072,
            guarded_context_window=8192,
            supports_native_tools=True,
            supports_structured_outputs=True,
            supports_vision=True,
            vram_cost_mb=5200,
        ),
        "gemma4:e4b-it-q4_K_M": ModelMetadata(
            name="gemma4:e4b-it-q4_K_M",
            advertised_context_window=131072,
            guarded_context_window=8192,
            supports_native_tools=True,
            supports_structured_outputs=True,
            supports_vision=True,
            vram_cost_mb=5200,
        ),
        "gemma2:9b": ModelMetadata(
            name="gemma2:9b",
            advertised_context_window=8192,
            guarded_context_window=8192,
            supports_native_tools=True,
            supports_structured_outputs=True,
            supports_vision=False,
            vram_cost_mb=5400,
        ),
        "gemma3:latest": ModelMetadata(
            name="gemma3:latest",
            advertised_context_window=32768,
            guarded_context_window=8192,
            supports_native_tools=True,
            supports_structured_outputs=True,
            supports_vision=True,
            vram_cost_mb=5100,
        ),
    }

    @classmethod
    def get_metadata(cls, model_name: str) -> ModelMetadata:
        clean_name = model_name.split("/")[-1]
        if clean_name in cls._MODELS:
            return cls._MODELS[clean_name]
        # Return generic safe fallback metadata
        return ModelMetadata(
            name=model_name,
            advertised_context_window=32768,
            guarded_context_window=8192,
            supports_native_tools=True,
            supports_structured_outputs=True,
            vram_cost_mb=5000,
        )

    @classmethod
    def register_model(cls, metadata: ModelMetadata) -> None:
        cls._MODELS[metadata.name] = metadata


class ProviderFactory:
    """Factory for instantiating model providers."""
    
    _providers: Dict[str, Type[BaseLLMProvider]] = {}
    _embedding_providers: Dict[str, Type[BaseEmbeddingProvider]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[BaseLLMProvider]) -> None:
        cls._providers[name] = provider_cls

    @classmethod
    def register_embedding_provider(cls, name: str, provider_cls: Type[BaseEmbeddingProvider]) -> None:
        cls._embedding_providers[name] = provider_cls

    @classmethod
    def create_provider(
        cls,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseLLMProvider:
        settings = get_settings()
        p_name = provider_name or settings.provider_type
        m_name = model_name or settings.primary_model

        if p_name not in cls._providers:
            try:
                import agent.providers  # noqa: F401
            except Exception:
                pass

        if p_name not in cls._providers:
            raise ModelNotFoundError(
                f"Provider '{p_name}' not registered. Available: {list(cls._providers.keys())}"
            )
        
        provider_cls = cls._providers[p_name]
        return provider_cls(model_name=m_name, **kwargs)

    @classmethod
    def create_embedding_provider(
        cls,
        provider_name: str = "ollama",
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseEmbeddingProvider:
        settings = get_settings()
        m_name = model_name or settings.embedding_model

        if provider_name not in cls._embedding_providers:
            raise ModelNotFoundError(
                f"Embedding provider '{provider_name}' not registered. Available: {list(cls._embedding_providers.keys())}"
            )
        return cls._embedding_providers[provider_name](model_name=m_name, **kwargs)

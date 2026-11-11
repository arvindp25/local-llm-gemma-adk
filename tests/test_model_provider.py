import pytest
from core.interfaces import ChatMessage
from core.registry import ModelRegistry, ProviderFactory
from agent.providers.adk_litellm import ADKLiteLLMProvider
from agent.providers.adk_ollama_native import ADKOllamaNativeProvider


def test_model_registry_metadata():
    metadata = ModelRegistry.get_metadata("gemma4:e4b")
    assert metadata.name == "gemma4:e4b"
    assert metadata.guarded_context_window == 8192
    assert metadata.supports_native_tools is True
    assert metadata.supports_structured_outputs is True


def test_provider_factory():
    provider = ProviderFactory.create_provider("adk_litellm", model_name="gemma4:e4b")
    assert isinstance(provider, ADKLiteLLMProvider)
    assert provider.model_name == "gemma4:e4b"

    native_provider = ProviderFactory.create_provider("adk_ollama_native", model_name="gemma4:e4b")
    assert isinstance(native_provider, ADKOllamaNativeProvider)
    assert native_provider.model_name == "gemma4:e4b"


@pytest.mark.asyncio
async def test_litellm_provider_generation():
    provider = ProviderFactory.create_provider("adk_litellm", model_name="gemma4:e4b")
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant. Reply with only one word: 'CONFIRMED'."),
        ChatMessage(role="user", content="Ping"),
    ]
    resp = await provider.chat(messages=messages, temperature=0.0, max_tokens=150)
    assert resp.content is not None
    assert len(resp.content.strip()) > 0
    assert "CONFIRMED" in resp.content.upper()


@pytest.mark.asyncio
async def test_native_provider_generation():
    provider = ProviderFactory.create_provider("adk_ollama_native", model_name="gemma4:e4b")
    messages = [
        ChatMessage(role="system", content="Reply with only one word: 'READY'."),
        ChatMessage(role="user", content="Status check"),
    ]
    resp = await provider.chat(messages=messages, temperature=0.0, max_tokens=150)
    assert resp.content is not None
    assert len(resp.content.strip()) > 0
    assert "READY" in resp.content.upper()

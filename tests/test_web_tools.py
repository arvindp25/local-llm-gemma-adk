import pytest
from tools.web_tools import web_search, fetch_web_page
from core.interfaces import ChatMessage
from core.registry import ProviderFactory
from tools.base_tool import LocalTool


def test_web_search_direct():
    results = web_search(query="Python programming language", max_results=3)
    assert results is not None
    assert "URL:" in results or "http" in results
    assert len(results.strip()) > 20


def test_fetch_web_page_direct():
    text = fetch_web_page("https://httpbin.org/html", max_chars=500)
    assert text is not None
    assert "Herman Melville" in text or "Moby-Dick" in text or len(text) > 10


@pytest.mark.asyncio
async def test_gemma_web_search_tool_calling():
    search_tool = LocalTool(web_search)
    provider = ProviderFactory.create_provider("adk_litellm", model_name="gemma4:e4b")

    messages = [
        ChatMessage(
            role="system",
            content="You have access to live internet search via `web_search`. If real-time or online lookup is requested, invoke web_search.",
        ),
        ChatMessage(
            role="user",
            content="Use the web_search tool to find the official website of Python programming language.",
        ),
    ]

    resp = await provider.chat(
        messages=messages,
        tools=[search_tool.schema],
        temperature=0.0,
        max_tokens=300,
    )

    assert len(resp.tool_calls) > 0
    call = resp.tool_calls[0]
    assert call.name == "web_search"
    assert "query" in call.arguments

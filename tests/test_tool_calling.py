import pytest
from pydantic import BaseModel, Field
from core.interfaces import ChatMessage
from core.registry import ProviderFactory
from tools.local_tools import LOCAL_TOOLS, calculate


class SystemStatusReport(BaseModel):
    status: str = Field(description="Operational status, e.g. 'OK'")
    architecture: str = Field(description="Target GPU model or architecture")
    context_window_limit: int = Field(description="Configured context limit in tokens")


def multiply(a: float, b: float) -> float:
    """Multiplies two numbers.
    
    Args:
        a: First number.
        b: Second number.
    """
    return a * b


@pytest.mark.asyncio
async def test_tool_calling_execution():
    from tools.base_tool import LocalTool
    mult_tool = LocalTool(multiply)

    provider = ProviderFactory.create_provider("adk_litellm", model_name="gemma4:e4b")
    messages = [
        ChatMessage(
            role="system",
            content="You have access to tools. If calculation is requested, invoke multiply tool.",
        ),
        ChatMessage(
            role="user",
            content="What is 12.5 multiplied by 4? Call the multiply tool to find the exact result.",
        ),
    ]

    resp = await provider.chat(
        messages=messages,
        tools=[mult_tool.schema],
        temperature=0.0,
        max_tokens=100,
    )

    # Assert that Gemma 4 E4B selected the tool
    assert len(resp.tool_calls) > 0
    call = resp.tool_calls[0]
    assert call.name == "multiply"
    assert "a" in call.arguments or "b" in call.arguments

    # Execute tool and assert result
    args = call.arguments
    result = multiply(float(args.get("a", 12.5)), float(args.get("b", 4)))
    assert result == 50.0


@pytest.mark.asyncio
async def test_structured_output_pydantic():
    import json
    provider = ProviderFactory.create_provider("adk_litellm", model_name="gemma4:e4b")
    schema_json = json.dumps(SystemStatusReport.model_json_schema())
    messages = [
        ChatMessage(
            role="system",
            content=f"You are a system reporter. You must output JSON conforming strictly to this schema: {schema_json}. Do not output any explanation outside the JSON.",
        ),
        ChatMessage(
            role="user",
            content="Generate a status report for RTX 5050 with status 'OK', architecture 'RTX 5050', and context_window_limit 8192.",
        ),
    ]

    resp = await provider.chat(
        messages=messages,
        response_model=SystemStatusReport,
        temperature=0.0,
        max_tokens=512,
    )

    assert resp.structured_data is not None
    assert isinstance(resp.structured_data, SystemStatusReport)
    assert resp.structured_data.status.upper() == "OK"
    assert resp.structured_data.context_window_limit == 8192

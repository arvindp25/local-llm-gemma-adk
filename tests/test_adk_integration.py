import pytest
from google.adk.agents import Agent
from agent.gemma_agent import GemmaAgent
from agent.prompt_templates import DEFAULT_GEMMA_SYSTEM_INSTRUCTION
from tools.local_tools import calculate


@pytest.mark.asyncio
async def test_gemma_adk_agent_initialization():
    agent = GemmaAgent()
    assert agent.adk_agent is not None
    assert isinstance(agent.adk_agent, Agent)
    assert agent.adk_agent.name == "gemma_edge_agent"
    assert len(agent.adk_agent.tools) > 0


@pytest.mark.asyncio
async def test_multi_turn_conversation_context():
    agent = GemmaAgent(enable_reflection=False)
    agent.episodic_memory.clear()

    # Turn 1: Give personal secret
    r1 = await agent.execute_task("Remember this secret project code: GEMMA-EDGE-5050.")
    assert "GEMMA-EDGE-5050" in agent.episodic_memory.messages[-1].content or len(r1.answer) > 0

    # Turn 2: Query secret
    r2 = await agent.execute_task("What was the secret project code I just told you to remember?")
    assert "GEMMA-EDGE-5050" in r2.answer.upper() or "5050" in r2.answer


@pytest.mark.asyncio
async def test_multi_step_tool_workflow():
    agent = GemmaAgent(enable_reflection=False)
    result = await agent.execute_task(
        "Calculate the square root of 256 and add 40 using the calculate tool."
    )
    assert len(result.tool_calls) > 0 or len(result.observations) > 0
    # Final answer should contain 56 (sqrt(256)=16 + 40 = 56)
    assert "56" in result.answer or any("56" in str(obs) for obs in result.observations)

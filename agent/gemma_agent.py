import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel
from google.adk.agents import Agent
from config.settings import get_settings
from core.interfaces import BaseLLMProvider, ChatMessage, LLMResponse, ToolCall
from core.registry import ProviderFactory
from memory.episodic_memory import EpisodicMemory
from memory.vector_store import SemanticMemoryStore
from memory.reflection_engine import ReflectionEngine
from profiler.gpu_monitor import VRAMGuardian
from tools.local_tools import LOCAL_TOOL_FUNCTIONS, LOCAL_TOOLS
from tools.mcp_client import MCPManager
from .prompt_templates import DEFAULT_GEMMA_SYSTEM_INSTRUCTION
from .providers.adk_litellm import ADKLiteLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    answer: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    distilled_insight: Optional[str] = None
    vram_peak_used_mb: Optional[float] = None
    turn_count: int = 1


class GemmaAgent:
    """Production Agent integrating Google Gemma 4 E4B with Google ADK, MCP tools, and Long-Term Memory."""

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        tools: Optional[List[Callable]] = None,
        system_instruction: Optional[str] = None,
        enable_reflection: bool = True,
    ):
        self.settings = get_settings()
        self.provider = provider or ProviderFactory.create_provider()
        self.system_instruction = system_instruction or DEFAULT_GEMMA_SYSTEM_INSTRUCTION
        self.enable_reflection = enable_reflection

        # Guardian & Memory systems
        self.guardian = VRAMGuardian()
        self.episodic_memory = EpisodicMemory(max_context_tokens=self.settings.num_ctx)
        self.semantic_memory = SemanticMemoryStore()
        self.reflection_engine = ReflectionEngine(
            vector_store=self.semantic_memory,
            llm_provider=self.provider,
        )
        self.mcp_manager = MCPManager()

        # Tools setup
        self.tool_functions: List[Callable] = tools if tools is not None else list(LOCAL_TOOL_FUNCTIONS)
        self.tool_schemas: List[Dict[str, Any]] = [t.schema for t in LOCAL_TOOLS]
        self._tool_lookup: Dict[str, Callable] = {f.__name__: f for f in self.tool_functions}

        # Initialize Google ADK Agent instance
        adk_model = None
        if hasattr(self.provider, "get_adk_model"):
            adk_model = self.provider.get_adk_model()
            
        self.adk_agent = Agent(
            name="gemma_edge_agent",
            description="Autonomous local agent powered by Google Gemma 4 E4B",
            model=adk_model,
            instruction=self.system_instruction,
            tools=self.tool_functions,
        )

    def register_tool(self, func: Callable) -> None:
        from tools.base_tool import LocalTool
        tool = LocalTool(func)
        self.tool_functions.append(func)
        self.tool_schemas.append(tool.schema)
        self._tool_lookup[func.__name__] = func

    def connect_mcp_server(self, server_id: str, command: str, args: Optional[List[str]] = None) -> None:
        toolset = self.mcp_manager.connect_stdio_server(server_id=server_id, command=command, args=args)
        # Add to ADK Agent tools
        if self.adk_agent and hasattr(self.adk_agent, "tools"):
            self.adk_agent.tools.append(toolset)

    async def execute_task(
        self,
        user_prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
        max_turns: int = 5,
    ) -> WorkflowResult:
        """Executes multi-step reasoning, tool invocation, and long-term memory updates."""
        # Step 1: Assert VRAM headroom safety
        self.guardian.assert_vram_safety(estimated_need_mb=500.0)

        # Step 2: Retrieve relevant past learnings / facts from Semantic Memory (RAG)
        retrieved_memories = self.semantic_memory.search(query=user_prompt, top_k=2)
        memory_context = ""
        if retrieved_memories:
            fact_bullets = "\n".join(f"- {m['content']}" for m in retrieved_memories)
            memory_context = f"\n[Relevant Long-Term Memories & Past Solutions]:\n{fact_bullets}\n"

        augmented_prompt = f"{user_prompt}\n{memory_context}".strip()
        self.episodic_memory.add_message(role="user", content=augmented_prompt)

        all_tool_calls: List[ToolCall] = []
        all_observations: List[Dict[str, Any]] = []
        turn = 0
        final_answer = ""

        while turn < max_turns:
            turn += 1
            # Call provider with conversational history and active tool schemas
            response: LLMResponse = await self.provider.chat(
                messages=self.episodic_memory.get_messages(),
                tools=self.tool_schemas if self.tool_schemas else None,
                response_model=response_model if turn == 1 or not all_tool_calls else None,
                system_instruction=self.system_instruction,
            )

            # If model returned tool calls, execute them
            if response.tool_calls:
                self.episodic_memory.add_message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
                for tc in response.tool_calls:
                    all_tool_calls.append(tc)
                    fn = self._tool_lookup.get(tc.name)
                    if fn:
                        try:
                            obs = fn(**tc.arguments)
                        except Exception as e:
                            obs = f"Error executing {tc.name}: {e}"
                    else:
                        obs = f"Tool '{tc.name}' not found."

                    observation_dict = {"tool": tc.name, "arguments": tc.arguments, "output": obs}
                    all_observations.append(observation_dict)
                    
                    self.episodic_memory.add_message(
                        role="tool",
                        content=str(obs),
                        tool_call_id=tc.id or tc.name,
                    )
            else:
                # Terminal answer reached
                final_answer = response.content
                self.episodic_memory.add_message(role="assistant", content=final_answer)
                break

        # Step 3: Distill insights into long-term memory (Continuous Learning Reflection Loop)
        distilled_insight = None
        if self.enable_reflection and all_tool_calls:
            try:
                distilled_insight = await self.reflection_engine.reflect_and_store(
                    task_prompt=user_prompt,
                    tool_results=all_observations,
                    final_answer=final_answer,
                )
            except Exception as e:
                logger.warning(f"Reflection error: {e}")

        gpu_snap = self.guardian.get_snapshot()
        return WorkflowResult(
            answer=final_answer,
            tool_calls=all_tool_calls,
            observations=all_observations,
            distilled_insight=distilled_insight,
            vram_peak_used_mb=gpu_snap.used_mb if gpu_snap else None,
            turn_count=turn,
        )

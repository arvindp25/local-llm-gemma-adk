from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str = Field(default="", description="Call ID")
    name: str = Field(..., description="Function/tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Parsed arguments dict")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, assistant, tool")
    content: str = Field(default="", description="Message text")
    tool_calls: Optional[List[ToolCall]] = Field(default=None, description="Model tool calls")
    tool_call_id: Optional[str] = Field(default=None, description="Responding tool call id")


class LLMResponse(BaseModel):
    content: str = Field(default="", description="Assistant text response")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls generated")
    finish_reason: str = Field(default="stop", description="Finish reason: stop, tool_calls, length")
    usage: Dict[str, int] = Field(default_factory=dict, description="Token usage stats")
    raw_response: Optional[Any] = Field(default=None, description="Original engine response")
    structured_data: Optional[Any] = Field(default=None, description="Validated Pydantic instance if requested")


class ModelMetadata(BaseModel):
    name: str
    advertised_context_window: int = 131072
    guarded_context_window: int = 8192
    supports_native_tools: bool = True
    supports_structured_outputs: bool = True
    supports_vision: bool = True
    vram_cost_mb: int = 5200


class BaseLLMProvider(ABC):
    """Abstract interface for local LLM engines."""

    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_model: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Send chat messages and get response with optional tool calls and structured schema."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Simple single-turn generation."""
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Return model metadata and VRAM limits."""
        pass


class BaseEmbeddingProvider(ABC):
    """Abstract interface for local text embeddings."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings into vectors."""
        pass

class LocalLLMError(Exception):
    """Base exception for Local LLM and ADK operations."""
    pass

class ModelNotFoundError(LocalLLMError):
    """Raised when the specified model is not found in the local runtime."""
    pass

class VRAMOverflowError(LocalLLMError):
    """Raised when requested context window or batch exceeds available VRAM."""
    pass

class ToolExecutionError(LocalLLMError):
    """Raised when a local or MCP tool fails during execution."""
    pass

class StructuredOutputError(LocalLLMError):
    """Raised when LLM output fails validation against Pydantic schema."""
    pass

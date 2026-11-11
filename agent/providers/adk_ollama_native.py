import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Type
import httpx
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse as ADKLlmResponse
from config.settings import get_settings
from core.interfaces import BaseLLMProvider, ChatMessage, LLMResponse, ModelMetadata, ToolCall
from core.registry import ModelRegistry, ProviderFactory
from core.exceptions import StructuredOutputError

logger = logging.getLogger(__name__)


class OllamaNativeLlm(BaseLlm):
    """Custom Google ADK BaseLlm implementation interfacing directly with Ollama's native HTTP API."""

    model: str
    api_base: str = "http://127.0.0.1:11434"
    num_ctx: int = 8192
    temperature: float = 0.7

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[ADKLlmResponse, None]:
        """Translates ADK LlmRequest directly to Ollama /api/chat payload."""
        messages: List[Dict[str, Any]] = []

        # Convert contents from ADK request to Ollama format
        if llm_request.system_instruction:
            sys_text = ""
            if hasattr(llm_request.system_instruction, "parts"):
                sys_text = " ".join(p.text for p in llm_request.system_instruction.parts if hasattr(p, "text") and p.text)
            messages.append({"role": "system", "content": sys_text})

        for content in llm_request.contents:
            role = content.role or "user"
            text_parts = [p.text for p in content.parts if hasattr(p, "text") and p.text]
            messages.append({"role": role, "content": " ".join(text_parts)})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.api_base}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            reply_text = data.get("message", {}).get("content", "")

            # Return ADK LlmResponse
            yield ADKLlmResponse(
                raw_response=data,
                content=reply_text,
            )


class ADKOllamaNativeProvider(BaseLLMProvider):
    """Abstracted Provider using direct native Ollama HTTP interface."""

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        settings = get_settings()
        self.model_name = model_name or settings.primary_model
        self.api_base = settings.ollama_api_base
        self.adk_model = OllamaNativeLlm(
            model=self.model_name,
            api_base=self.api_base,
            num_ctx=settings.num_ctx,
            temperature=settings.temperature,
        )

    def get_adk_model(self) -> OllamaNativeLlm:
        return self.adk_model

    def get_metadata(self) -> ModelMetadata:
        return ModelRegistry.get_metadata(self.model_name)

    async def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_model: Optional[Type[Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        settings = get_settings()
        temp = temperature if temperature is not None else settings.temperature
        max_t = max_tokens or settings.max_tokens

        formatted_messages = []
        if system_instruction:
            formatted_messages.append({"role": "system", "content": system_instruction})

        for msg in messages:
            item: Dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "function": {"name": tc.name, "arguments": tc.arguments}
                    }
                    for tc in msg.tool_calls
                ]
            formatted_messages.append(item)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "num_ctx": settings.num_ctx,
                "temperature": temp,
                "num_predict": max_t,
            },
        }

        if tools:
            payload["tools"] = tools

        if response_model:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.api_base}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        msg_data = data.get("message", {})
        content = msg_data.get("content", "")
        thinking = msg_data.get("thinking", "")
        if not content and thinking:
            content = thinking

        # Extract tool calls
        parsed_tools: List[ToolCall] = []
        raw_tools = msg_data.get("tool_calls", [])
        for i, rt in enumerate(raw_tools):
            fn = rt.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}
            parsed_tools.append(ToolCall(id=f"call_{i}", name=name, arguments=args))

        # Handle structured validation
        structured_data = None
        if response_model:
            clean_json = content.strip()
            if clean_json.startswith("```"):
                clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
                clean_json = re.sub(r"\s*```$", "", clean_json)
            try:
                structured_data = response_model.model_validate_json(clean_json)
            except Exception as e:
                raise StructuredOutputError(
                    f"Failed to validate response against {response_model.__name__}: {e}\nRaw output: {content}"
                ) from e

        usage_dict = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        }

        return LLMResponse(
            content=content,
            tool_calls=parsed_tools,
            finish_reason=data.get("done_reason", "stop"),
            usage=usage_dict,
            raw_response=data,
            structured_data=structured_data,
        )

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = [ChatMessage(role="user", content=prompt)]
        resp = await self.chat(
            messages=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.content


# Register provider in factory
ProviderFactory.register_provider("adk_ollama_native", ADKOllamaNativeProvider)

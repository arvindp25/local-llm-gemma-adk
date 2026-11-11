import json
import os
import re
from typing import Any, Dict, List, Optional, Type
import litellm
from google.adk.models.lite_llm import LiteLlm
from config.settings import get_settings
from core.interfaces import BaseLLMProvider, ChatMessage, LLMResponse, ModelMetadata, ToolCall
from core.registry import ModelRegistry, ProviderFactory
from core.exceptions import StructuredOutputError


class ADKLiteLLMProvider(BaseLLMProvider):
    """Google ADK LiteLlm Provider wrapping local Ollama Gemma inference."""

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        settings = get_settings()
        self.model_name = model_name or settings.primary_model
        self.api_base = settings.ollama_api_base
        self.adk_model_id = f"ollama_chat/{self.model_name}"

        # Ensure environment variables for LiteLLM
        os.environ["OLLAMA_API_BASE"] = self.api_base

        # Instantiate ADK's LiteLlm model wrapper
        self.adk_model = LiteLlm(
            model=self.adk_model_id,
            **kwargs,
        )

    def get_adk_model(self) -> LiteLlm:
        """Returns the native Google ADK LiteLlm instance for use in ADK Agent."""
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

        # Format message payload for LiteLLM / Ollama chat
        formatted_messages = []
        if system_instruction:
            formatted_messages.append({"role": "system", "content": system_instruction})

        for msg in messages:
            item: Dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id or f"call_{i}",
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for i, tc in enumerate(msg.tool_calls)
                ]
            if msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            formatted_messages.append(item)

        # Prepare LiteLLM options
        kwargs: Dict[str, Any] = {
            "model": self.adk_model_id,
            "messages": formatted_messages,
            "temperature": temp,
            "max_tokens": max_t,
            "api_base": self.api_base,
            "num_ctx": settings.num_ctx,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if response_model:
            kwargs["response_format"] = {"type": "json_object"}

        response = await litellm.acompletion(**kwargs)
        choice = response.choices[0]
        msg_obj = choice.message
        content = msg_obj.content or ""
        if not content and hasattr(msg_obj, "reasoning_content") and msg_obj.reasoning_content:
            content = msg_obj.reasoning_content
        elif not content and hasattr(msg_obj, "thinking") and msg_obj.thinking:
            content = msg_obj.thinking

        # Extract tool calls
        parsed_tool_calls: List[ToolCall] = []
        if hasattr(msg_obj, "tool_calls") and msg_obj.tool_calls:
            for tc in msg_obj.tool_calls:
                fn = tc.function
                args = fn.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                parsed_tool_calls.append(ToolCall(id=tc.id or "", name=fn.name, arguments=args))

        # Handle structured output validation
        structured_data = None
        if response_model:
            clean_json = content.strip()
            # Clean markdown codeblocks if model enclosed json
            if clean_json.startswith("```"):
                clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
                clean_json = re.sub(r"\s*```$", "", clean_json)
            try:
                structured_data = response_model.model_validate_json(clean_json)
            except Exception as e:
                raise StructuredOutputError(
                    f"Failed to validate response against {response_model.__name__}: {e}\nRaw output: {content}"
                ) from e

        usage_dict = {}
        if hasattr(response, "usage") and response.usage:
            usage_dict = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage_dict,
            raw_response=response,
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
ProviderFactory.register_provider("adk_litellm", ADKLiteLLMProvider)

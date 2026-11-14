import logging
import uuid
from typing import Any, Dict, List, Optional
from core.interfaces import BaseLLMProvider, BaseVectorStore
from .vector_store import SemanticMemoryStore

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """Continuous learning engine that distills completed workflows into persistent semantic memory."""

    def __init__(self, vector_store: BaseVectorStore, llm_provider: Optional[BaseLLMProvider] = None):
        self.vector_store = vector_store
        self.llm_provider = llm_provider

    async def reflect_and_store(
        self,
        task_prompt: str,
        tool_results: List[Dict[str, Any]],
        final_answer: str,
        context_tags: Optional[List[str]] = None,
    ) -> str:
        """Analyzes task execution and indexes distilled lessons into ChromaDB."""
        # Synthesize a distillation prompt
        distill_prompt = (
            f"TASK: {task_prompt}\n"
            f"TOOL OBSERVATIONS: {tool_results}\n"
            f"FINAL ANSWER: {final_answer}\n\n"
            "INSTRUCTION: Extract 1-2 concise, reusable factual insights or procedural rules learned from this outcome. "
            "Focus on persistent facts, preferences, or technical rules. Keep it under 2 sentences."
        )

        distilled_insight = ""
        if self.llm_provider:
            try:
                distilled_insight = await self.llm_provider.generate(
                    prompt=distill_prompt,
                    system_instruction="You are a reflective agent distilling long-term memory.",
                    temperature=0.2,
                    max_tokens=256,
                )
            except Exception as e:
                logger.warning(f"LLM reflection generation failed: {e}. Falling back to rule-based memory.")

        if not distilled_insight:
            # Fallback distillation
            distilled_insight = f"Task '{task_prompt[:60]}' resolved with: {final_answer[:120]}"

        doc_id = f"insight_{uuid.uuid4().hex[:10]}"
        metadata = {
            "source": "reflection_engine",
            "task": task_prompt[:100],
            "tags": ",".join(context_tags or ["general"]),
        }

        self.vector_store.add_document(doc_id=doc_id, text=distilled_insight.strip(), metadata=metadata)
        logger.info(f"Distilled and stored long-term insight [{doc_id}]: {distilled_insight.strip()}")
        return distilled_insight.strip()

    def retrieve_relevant_insights(self, query: str, top_k: int = 2) -> List[str]:
        """Retrieves learned lessons from past experiences to ground future prompts."""
        hits = self.vector_store.search(query=query, top_k=top_k)
        return [h["content"] for h in hits]

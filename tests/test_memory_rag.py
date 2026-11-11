import pytest
from memory.vector_store import SemanticMemoryStore
from memory.episodic_memory import EpisodicMemory
from memory.reflection_engine import ReflectionEngine
from agent.gemma_agent import GemmaAgent


def test_episodic_memory_compaction():
    mem = EpisodicMemory(max_context_tokens=100)
    mem.clear()

    # Add system message
    mem.add_message(role="system", content="System core instruction.")

    # Add lots of messages to exceed 100 token threshold
    for i in range(25):
        mem.add_message(role="user", content=f"User turn message #{i} containing multiple words to fill space.")
        mem.add_message(role="assistant", content=f"Assistant turn response #{i} answering.")

    # Verify compaction occurred without losing system instruction
    messages = mem.get_messages()
    assert messages[0].role == "system"
    assert "System core instruction." in messages[0].content
    assert any("[Notice: Earlier turns" in m.content for m in messages)


def test_semantic_vector_store_rag():
    store = SemanticMemoryStore(collection_name="test_knowledge")
    unique_fact = "The RTX 5050 Laptop GPU in this system has 8151 MiB GDDR6 VRAM and uses CUDA compute 12.0."
    store.add_document(doc_id="fact_gpu_01", text=unique_fact, metadata={"topic": "gpu_specs"})

    results = store.search(query="How much VRAM does the RTX 5050 have?", top_k=1)
    assert len(results) > 0
    assert "8151 MiB" in results[0]["content"]


@pytest.mark.asyncio
async def test_continuous_learning_reflection_loop():
    store = SemanticMemoryStore(collection_name="test_reflection_knowledge")
    engine = ReflectionEngine(vector_store=store)

    # Simulate reflecting on a completed task
    insight = await engine.reflect_and_store(
        task_prompt="Find optimal context length for 8GB VRAM",
        tool_results=[{"tool": "gpu_inspect", "output": "Peak VRAM at 8192 tokens is 6.5 GB"}],
        final_answer="Set default context window to 8192 tokens to preserve 1.5 GB headroom.",
    )

    assert len(insight) > 0
    # Retrieve the insight from ChromaDB
    retrieved = engine.retrieve_relevant_insights(query="What is the safe context window for 8GB VRAM?", top_k=1)
    assert len(retrieved) > 0
    assert "8192" in retrieved[0] or "8GB" in retrieved[0] or "VRAM" in retrieved[0]

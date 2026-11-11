# Implementation Plan — Google Gemma Local LLM & Google ADK Architecture

This design and implementation plan establishes Google Gemma (specifically **Gemma 4 E4B**) as the primary local reasoning model on an **NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM)** with **24 GB system RAM** running **Windows 11**. The system is integrated with **Google Agent Development Kit (ADK)**, provides an abstracted model interface, supports **Model Context Protocol (MCP)** tool calling, adheres to VRAM safety budgets, and implements long-term learning via Memory and RAG rather than weight fine-tuning.

---

## 1. Goal Description & Architectural Overview

The goal is to build a production-grade, 100% local, privacy-first agent architecture that:
1. Runs **Gemma 4 E4B** via **Ollama** locally on Windows with zero cloud API dependencies.
2. Integrates directly with **Google Agent Development Kit (ADK)** (`google-adk`), evaluating both ADK's `LiteLlm` connector (using `ollama_chat/`) and a direct custom ADK `BaseLlm` provider.
3. Implements an abstracted `LLMProvider` interface to insulate agent workflows from model or runtime swaps.
4. Enforces strict hardware constraints for the **RTX 5050 8GB VRAM** laptop GPU, computing exact KV cache memory budgets to prevent Out-Of-Memory (OOM) crashes.
5. Implements reliable **Tool Calling**, **Structured Outputs** (Pydantic validation), **Multi-turn History**, and **MCP (Model Context Protocol)** client orchestration.
6. Realizes persistent intelligence through a two-tiered **Long-Term Memory & RAG** subsystem (Episodic + Semantic Vector Memory with post-task insight reflection).

### System Architecture Diagram

```mermaid
graph TD
    User([User / Application]) --> Agent[Google ADK Agent Engine]
    
    subgraph Agent_Subsystem [Google ADK & Agent Framework]
        Agent --> MultiTurn[Multi-Turn State & Turn Management]
        Agent --> SysInst[System Instructions & Prompt Templates]
        Agent --> ToolRouter[Tool Calling & Router]
        Agent --> StructOut[Structured Output Validator / Pydantic]
    end

    subgraph Memory_RAG_Subsystem [Long-Term Learning & RAG Subsystem]
        ToolRouter <--> MemEngine[Memory & Knowledge Engine]
        MemEngine --> EpisodicMem[Episodic Conversation Store]
        MemEngine --> VectorStore[(Local ChromaDB / Semantic Store)]
        MemEngine --> ReflectionLoop[Post-Task Insight Distillation]
        EmbeddingModel[Ollama Embedder: nomic-embed-text] --> VectorStore
    end

    subgraph Tool_Ecosystem [Tool & Integration Layer]
        ToolRouter --> LocalTools[Native Python Tools]
        ToolRouter --> McpToolset[ADK McpToolset Client]
        McpToolset --> McpServer[External MCP Servers stdio / sse]
    end

    subgraph Abstraction_Layer [Model Abstraction Layer]
        Agent --> ModelInterface[LLMProvider Interface]
        ModelInterface --> ProviderFactory[Provider Factory & Registry]
        ProviderFactory --> ADKLiteLLM[ADK LiteLlm Provider: ollama_chat]
        ProviderFactory --> ADKNative[ADK Custom OllamaNativeLlm Provider]
    end

    subgraph Hardware_Runtime [Hardware & Inference Layer: RTX 5050 8GB]
        ADKLiteLLM --> OllamaHost[Ollama Runtime: http://127.0.0.1:11434]
        ADKNative --> OllamaHost
        OllamaHost --> GemmaModel[Gemma 4 E4B Q4_K_M (~5.2 GB)]
        OllamaHost --> KVCache[Guarded KV Cache: 8K tokens FlashAttn (~0.8 GB)]
    end
```

---

## 2. User Review Required

> [!IMPORTANT]
> **Python 3.12 Virtual Environment via `uv`**:
> The system currently has Python 3.10 as the system default, but `google-adk` requires **Python >= 3.11**. `uv` is already installed on your machine and has a pre-cached Python 3.12.11 binary at `C:\Users\arvin\AppData\Roaming\uv\python\cpython-3.12.11-windows-x86_64-none\python.exe`.
> We will create a local `.venv` powered by Python 3.12 using `uv venv --python 3.12` to ensure seamless compatibility without affecting your global system Python.

> [!WARNING]
> **VRAM Context Window Safeguard (8 GB RTX 5050)**:
> Google Gemma 4 E4B advertises up to a **128K context window**. However, on an 8 GB VRAM GPU, allocating a 128K FP16 KV cache would require >16 GB of VRAM and immediately crash with CUDA Out-Of-Memory.
> We establish a hard default `num_ctx` limit of **8,192 tokens** (with option for 16,384 under Q8 KV cache), keeping total GPU memory around ~6.5 GB (weights + KV cache + CUDA runtime overhead), leaving ~1.5 GB headroom for Windows WDDM display buffers.

> [!NOTE]
> **Ollama Installation**:
> `ollama` is not yet present in your system PATH. Winget has the official `Ollama.Ollama` v0.40.13 package. We will install Ollama via winget (`winget install Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements`) or configure the portable binary.

---

## 3. Hardware Constraints & VRAM Budget Calculation

Target: **NVIDIA GeForce RTX 5050 Laptop GPU**
- **Total Physical VRAM**: 8,151 MiB (~8.0 GB)
- **Windows OS / WDDM Overhead**: ~300 MiB to 500 MiB (current idle usage: 276 MiB)
- **Available Usable VRAM**: ~7,650 MiB (~7.47 GB)
- **System RAM**: 24 GB (available for CPU offloading and local ChromaDB)

### Memory Allocation Matrix

| Component | Model Tag / Config | VRAM Footprint | System RAM Footprint | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Gemma 4 E4B Weights** | `gemma4:e4b-it-q4_K_M` | **~5,200 MiB** | ~0 MiB | 100% GPU offloaded (`num_gpu: 99`) |
| **KV Cache (4,096 tokens)** | FP16 / FlashAttn=1 | ~420 MiB | ~0 MiB | Ultra-safe, maximum inference speed |
| **KV Cache (8,192 tokens)** | FP16 / FlashAttn=1 | **~840 MiB** | ~0 MiB | **Default recommended configuration** |
| **KV Cache (16,384 tokens)**| Q8_0 KV Cache | ~1,100 MiB | ~0 MiB | High-context mode |
| **CUDA Activations & Headroom** | Windows Desktop / PyTorch | ~600 MiB | ~0 MiB | Prevents sudden OOM spikes |
| **Embedding Model** | `nomic-embed-text` | ~280 MiB | ~0 MiB | Loaded on-demand or in CPU RAM |
| **Total Peak GPU VRAM** | **(8K context mode)** | **~6,640 MiB / 8,151 MiB (81.4%)** | — | **Safe buffer of ~1.5 GB VRAM** |

### Critical Environment Configurations
To lock down memory stability on Windows:
```powershell
$env:OLLAMA_NUM_PARALLEL = "1"        # Prevent multiple concurrent KV caches
$env:OLLAMA_FLASH_ATTENTION = "1"     # Enable Flash Attention for reduced memory
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"    # Quantized KV cache saving ~40% VRAM
$env:OLLAMA_HOST = "127.0.0.1:11434"
```

---

## 4. Proposed Implementation Architecture

### Directory Structure

```
c:\Users\arvin\Desktop\local-llm\
├── .env.example
├── pyproject.toml
├── config/
│   ├── __init__.py
│   ├── settings.py              # Pydantic Settings for runtime, hardware, paths
│   └── default_config.yaml      # Model names, context limits, Ollama host, params
├── core/
│   ├── __init__.py
│   ├── interfaces.py            # BaseLLMProvider, EmbeddingProvider, ModelInfo
│   ├── registry.py              # Dynamic model registry & provider factory
│   └── exceptions.py            # Custom exceptions for model/tool execution
├── agent/
│   ├── __init__.py
│   ├── gemma_agent.py           # Core agent orchestrator
│   ├── adk_integration.py       # Google ADK agent wrapper & lifecycle
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── adk_litellm.py       # ADK LiteLlm wrapper (ollama_chat/gemma4:e4b)
│   │   └── adk_ollama_native.py # Direct ADK BaseLlm implementation for Ollama
│   └── prompt_templates.py      # System instructions & role definitions
├── tools/
│   ├── __init__.py
│   ├── base_tool.py             # Tool schema definitions & validators
│   ├── local_tools.py           # Native tools (math, file inspection, system)
│   └── mcp_client.py            # ADK McpToolset & stdio/SSE MCP client manager
├── memory/
│   ├── __init__.py
│   ├── episodic_memory.py       # Multi-turn history, window pruning, summarizer
│   ├── vector_store.py          # ChromaDB RAG vector store + Ollama embeddings
│   └── reflection_engine.py     # Distills task solutions into long-term memories
├── profiler/
│   ├── __init__.py
│   ├── gpu_monitor.py           # PyNVML / nvidia-smi VRAM monitor & alert
│   └── benchmark.py             # Evaluates TPS, latency, context & tool accuracy
└── tests/
    ├── __init__.py
    ├── test_model_provider.py   # Tests model abstraction & switching
    ├── test_adk_integration.py  # Tests ADK agent, multi-turn, system instructions
    ├── test_tool_calling.py     # Tests tool invocation & structured JSON output
    ├── test_mcp_integration.py  # Tests MCP server tool discovery and execution
    └── test_memory_rag.py       # Tests episodic memory & RAG retrieval loop
```

---

## 5. Detailed Component Specifications

### Component A: Project Setup & Dependency Management
- **File**: `pyproject.toml`
- **Python**: `>=3.11, <3.13`
- **Dependencies**:
  * `google-adk>=0.1.0` (Core Agent Development Kit)
  * `google-adk[mcp]` (Model Context Protocol tool support)
  * `litellm>=1.84.0` (Multi-provider model connector)
  * `ollama>=0.4.0` (Direct Ollama Python SDK)
  * `pydantic>=2.10.0` (Structured data schemas)
  * `chromadb>=0.6.0` (Local vector store for RAG)
  * `pynvml>=12.0.0` (GPU VRAM monitoring)
  * `pytest>=8.0.0`, `pytest-asyncio>=0.24.0` (Testing harness)
  * `rich>=13.0.0` (Formatted terminal output)

---

### Component B: Model Abstraction Layer (`core/`)

Insulates agent code from inference engines. Any model (Gemma 4 E4B, Gemma 2 9B, Gemma 3, or future models) can be swapped via configuration without touching agent or tool code.

```python
# core/interfaces.py
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class LLMResponse(BaseModel):
    content: str
    tool_calls: List[Dict[str, Any]] = []
    finish_reason: str
    usage: Dict[str, int] = {}
    structured_data: Optional[Any] = None

class BaseLLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Execute chat turn with tool support and structured outputs."""
        pass

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate text embeddings."""
        pass
```

---

### Component C: Google ADK Integration & Evaluation (`agent/`)

The prompt requires:
> *"Evaluate ADK's native Gemma model integration and LiteLLM/Ollama integration. Verify tool calling, structured outputs, multi-turn conversations and system instructions. Use the correct Ollama chat interface."*

#### 1. Evaluation Analysis
* **ADK Native Model Integration**: Google ADK's native built-in provider (`google.adk.models.google_genai.GoogleGenAi`) targets Google Cloud Vertex AI and Gemini APIs. For Gemma models hosted on Google Vertex AI endpoints, ADK provides native cloud connectivity. However, ADK does **not** package an internal C++/llama.cpp runtime for direct offline on-device inference.
* **ADK LiteLLM Integration (`LiteLlm`)**: ADK's official and documented pattern for local models is `google.adk.models.lite_llm.LiteLlm`. By configuring `model="ollama_chat/gemma4:e4b"`, LiteLLM translates ADK tool schemas, system prompts, and multi-turn turns into Ollama's `/api/chat` format.
* **Crucial Format Detail**: Using `ollama_chat/<model>` instead of `ollama/<model>` is required. `ollama_chat` correctly passes OpenAI-compatible tool schemas and messages without triggering format degradation or infinite tool-calling loops.
* **Direct ADK `BaseLlm` Provider (`OllamaNativeLlm`)**: To ensure maximum resilience and zero LiteLLM translation overhead, we will also implement a custom ADK `BaseLlm` subclass that communicates directly with Ollama's HTTP API, applying Gemma-specific chat templates and raw JSON schema constraints.

```python
# agent/providers/adk_litellm.py
import os
from google.adk.models.lite_llm import LiteLlm
from config.settings import settings

def create_adk_model():
    # Force local Ollama base URL
    os.environ["OLLAMA_API_BASE"] = settings.ollama_api_base
    
    # Prefix with 'ollama_chat/' to activate Ollama's chat & function-calling interface
    model_identifier = f"ollama_chat/{settings.primary_model}"
    
    return LiteLlm(
        model=model_identifier,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
```

---

### Component D: Tool Calling & Model Context Protocol (MCP) (`tools/`)

Gemma 4 E4B is natively trained for function/tool calling.
1. **Local Tool Definition**: Python functions with docstrings and type annotations automatically converted to JSON schema.
2. **MCP Integration**:
   - Uses `google.adk.tools.mcp_tool.McpToolset`
   - Uses `StdioConnectionParams` and `StdioServerParameters` to connect to local or external MCP servers (e.g. SQLite, filesystem, memory).
   - Validates that Gemma 4 E4B can parse MCP server tool catalogs, select the correct tool, supply formatted arguments, and incorporate the observation into final reasoning.
3. **Structured Outputs**:
   - Uses Pydantic schemas. If Ollama receives `format="json"`, it forces JSON mode.
   - Includes a self-correcting validation wrapper: if the model outputs JSON wrapped in markdown ` ```json ... ``` `, the parser cleanses and validates against the Pydantic model.

---

### Component E: Long-Term Learning: Memory & RAG Subsystem (`memory/`)

The prompt states:
> *"Do not confuse the LLM's persistent knowledge with its model weights. Implement long-term learning through memory and RAG before considering fine-tuning."*

1. **Episodic Memory (Short/Medium-Term)**:
   - Stores session turns in a structured buffer.
   - Summarization Trigger: When conversational history reaches 50% of `num_ctx` (e.g. >4,000 tokens), the agent runs a background summarization turn to condense prior turns into key points, freeing VRAM for new reasoning without losing context.
2. **Semantic Memory (Long-Term Vector RAG)**:
   - Built on **ChromaDB** running locally in system RAM (`C:\Users\arvin\Desktop\local-llm\.data\chroma`).
   - Uses local embedding model via Ollama (e.g., `nomic-embed-text` with 768-dim embeddings, ~280 MB RAM).
   - Stores facts, user guidelines, project context, and past solutions.
   - Context Retrieval: Before generating responses for complex prompts, the system performs a top-k semantic search and grounds Gemma with retrieved facts.
3. **Continuous Learning Reflection Loop**:
   - After completing a multi-step task or tool interaction, the agent invokes an internal reflection function:
     *"Extract 1-3 generalized facts or lessons learned from this task."*
   - These distilled lessons are embedded and indexed into the Semantic Vector Store.
   - On future tasks, similar prompts automatically retrieve these past learnings, effectively achieving permanent knowledge retention without modifying model weights!

---

### Component F: Hardware Profiling & VRAM Guardian (`profiler/`)

Provides real-time telemetry:
- Monitors GPU memory via PyNVML before, during, and after inference.
- Implements `VRAMGuardian`: If free VRAM drops below 800 MiB during generation, it triggers proactive context compaction or alerts the user.
- Benchmark script: Measures tokens/sec, time-to-first-token (TTFT), KV cache growth per turn, and tool execution latency.

---

## 6. Verification Plan

### Automated Test Suite (`pytest`)

1. **Model Abstraction Test** (`tests/test_model_provider.py`):
   - Verify `ProviderFactory` can instantiate local Ollama provider and switch models via configuration.
2. **ADK Integration Test** (`tests/test_adk_integration.py`):
   - Verify Google ADK `Agent` initializes with Gemma via `LiteLlm(model="ollama_chat/gemma4:e4b")`.
   - Verify multi-turn conversation retains context across 3 distinct turns.
   - Verify system instruction enforcement (e.g., persona or strict output format constraint).
3. **Tool Calling & Structured Output Test** (`tests/test_tool_calling.py`):
   - Define a test tool: `calculate_mortgage(loan_amount: float, interest_rate: float, years: int) -> float`.
   - Send a query requiring tool invocation. Verify Gemma 4 E4B generates exact tool call with parsed arguments.
   - Define a Pydantic schema: `AnalysisReport(summary: str, risk_score: int, recommendations: list[str])`.
   - Verify structured output passes Pydantic validation.
4. **MCP Server Integration Test** (`tests/test_mcp_integration.py`):
   - Spin up a mock/local stdio MCP server exposing a `lookup_inventory` tool.
   - Connect ADK `McpToolset` to the server.
   - Verify Gemma 4 E4B inspects MCP tools, calls `lookup_inventory`, and completes the goal.
5. **Memory & Continuous Learning RAG Test** (`tests/test_memory_rag.py`):
   - Store a custom unique fact into ChromaDB semantic memory.
   - Ask the agent a question answerable only via that fact.
   - Verify RAG retriever injects the context and Gemma provides the correct answer.
   - Test the reflection loop: run a mock task, distill insight, verify it is retrievable in subsequent query.
6. **VRAM Benchmark Test** (`profiler/benchmark.py`):
   - Run benchmark with 2k, 4k, and 8k context lengths.
   - Record and assert peak VRAM stays below 7,500 MiB on the RTX 5050.

### Manual Verification
- Run `uv run python -m agent.gemma_agent "What is the status of the RTX 5050 GPU and how many tokens can I process?"` in interactive mode.
- Inspect `nvidia-smi` in real-time during multi-step tool execution.

---

## 7. Execution Steps

1. **Environment Setup**: Initialize Python 3.12 virtual environment via `uv venv --python 3.12` and create `pyproject.toml`.
2. **Ollama Setup & Model Verification**: Install/verify Ollama runtime, configure GPU environment variables (`OLLAMA_NUM_PARALLEL=1`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`), and pull `gemma4:e4b` (or `gemma4:e4b-it-q4_K_M`) and `nomic-embed-text`.
3. **Implement Abstraction & Config**: Create `config/settings.py`, `core/interfaces.py`, and `core/registry.py`.
4. **Implement ADK Provider & Agent**: Implement `agent/providers/adk_litellm.py`, `agent/providers/adk_ollama_native.py`, and `agent/gemma_agent.py`.
5. **Implement Tooling & MCP**: Implement `tools/local_tools.py` and `tools/mcp_client.py` using `McpToolset`.
6. **Implement Long-Term Memory & RAG**: Implement `memory/episodic_memory.py`, `memory/vector_store.py`, and `memory/reflection_engine.py`.
7. **Implement Profiler & Benchmarking**: Implement `profiler/gpu_monitor.py` and `profiler/benchmark.py`.
8. **Automated Testing & VRAM Validation**: Run comprehensive test suite and verify stability within 8 GB VRAM.

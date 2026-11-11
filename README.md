# Google Gemma Local LLM & Google ADK Architecture

A 100% local, privacy-first reasoning system integrating **Google Gemma 4 E4B** with **Google Agent Development Kit (ADK)** on Windows 11 with an **NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM)** and **24GB system RAM**.

---

## Architecture Highlights

1. **Local Inference by Default**:
   - Primary model: `gemma4:e4b` (6.6 GB package with Q4_K_M weights) running on local Ollama runtime.
   - Zero external cloud API calls or API keys required.
   - Flash Attention (`OLLAMA_FLASH_ATTENTION=1`) and Q8 KV cache (`OLLAMA_KV_CACHE_TYPE=q8_0`).

2. **Google ADK Integration**:
   - Native integration with Google ADK (`google-adk==2.13.0`).
   - Supports both `ADKLiteLLMProvider` (using `ollama_chat/gemma4:e4b`) and custom direct `ADKOllamaNativeProvider` (`OllamaNativeLlm`).
   - Multi-turn conversation state, system instructions, and tool calling orchestration.

3. **Hardware Constraints & Safeguards (RTX 5050 8GB)**:
   - Hard default context window cap of **8,192 tokens** (`num_ctx = 8192`), avoiding the advertised 128k context window that would trigger CUDA OOM.
   - Total GPU memory footprint ~6.5 GB out of 8.0 GB, leaving ~1.5 GB VRAM headroom for Windows WDDM desktop display.
   - Proactive `VRAMGuardian` checks free VRAM before every execution turn.

4. **Model Abstraction Layer**:
   - `core.interfaces.BaseLLMProvider` abstracts all inference logic.
   - Seamless dynamic swapping between Gemma variants (`gemma4:e4b`, `gemma2:9b`, `gemma3`) via `config/default_config.yaml` or `.env`.

5. **Long-Term Learning via Memory & RAG**:
   - **Episodic Memory**: Short-term sliding-window turn compaction.
   - **Semantic Vector Memory**: Local ChromaDB instance with `nomic-embed-text` embeddings.
   - **Continuous Learning Reflection Loop**: Automatically distills completed task lessons into long-term memory.

6. **Tool Calling & Live Internet Access**:
   - Live internet query powers via `web_search` (DuckDuckGo integration with zero API keys required).
   - Deep web page reading via `fetch_web_page` (extracts clean, formatted text from URLs).
   - Local system tools for math evaluation, hardware telemetry, and file inspection.
   - Model Context Protocol (MCP) tool support via Google ADK `McpToolset`.
   - Structured JSON output validation with Pydantic.

---

## Quick Start

### 1. Check GPU Telemetry
```powershell
.venv\Scripts\python.exe main.py --vram
```

### 2. Run Inference Benchmark
```powershell
.venv\Scripts\python.exe main.py --benchmark
```

### 3. Query the Live Internet
```powershell
.venv\Scripts\python.exe main.py --query "Search the internet for Google Gemma 4 architecture and summarize what it is."
```

### 4. Ask a Calculation or Hardware Task
```powershell
.venv\Scripts\python.exe main.py --query "Inspect the system GPU hardware and calculate the remaining free VRAM percentage."
```

### 5. Interactive Chat Session
```powershell
.venv\Scripts\python.exe main.py --interactive
```

### 6. Run Test Suite
```powershell
.venv\Scripts\pytest.exe -v
```

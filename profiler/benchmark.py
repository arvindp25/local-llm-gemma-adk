import asyncio
import time
from typing import Any, Dict
from rich.console import Console
from rich.table import Table
from .gpu_monitor import VRAMGuardian
from core.interfaces import BaseLLMProvider, ChatMessage
from core.registry import ProviderFactory

console = Console()


async def run_benchmark(provider: BaseLLMProvider, iterations: int = 3) -> Dict[str, Any]:
    """Runs latency, throughput, and VRAM benchmark for the active model."""
    guardian = VRAMGuardian()
    initial_gpu = guardian.get_snapshot()

    prompt = "Explain why local LLM reasoning on an edge GPU with 8GB VRAM requires careful KV cache management in 3 bullet points."
    messages = [
        ChatMessage(role="system", content="You are a concise AI systems engineer."),
        ChatMessage(role="user", content=prompt),
    ]

    latencies = []
    token_counts = []
    
    console.print(f"[bold cyan]Starting Benchmark on model: {provider.get_metadata().name}[/bold cyan]")
    if initial_gpu:
        console.print(f"GPU: {initial_gpu.name} | Total VRAM: {initial_gpu.total_mb} MiB | Free: {initial_gpu.free_mb} MiB")

    for i in range(iterations):
        t0 = time.perf_counter()
        resp = await provider.chat(messages=messages, temperature=0.1, max_tokens=256)
        elapsed = time.perf_counter() - t0
        
        token_count = resp.usage.get("completion_tokens", len(resp.content.split()) * 4 // 3)
        latencies.append(elapsed)
        token_counts.append(token_count)
        console.print(f"  Iteration {i+1}: {elapsed:.2f}s, ~{token_count} tokens ({token_count/elapsed:.1f} tps)")

    post_gpu = guardian.get_snapshot()
    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(token_counts) / len(token_counts)
    tps = avg_tokens / avg_latency if avg_latency > 0 else 0

    results = {
        "model": provider.get_metadata().name,
        "avg_latency_sec": round(avg_latency, 2),
        "avg_tokens": round(avg_tokens, 1),
        "tokens_per_sec": round(tps, 1),
        "initial_vram_free_mb": initial_gpu.free_mb if initial_gpu else None,
        "peak_vram_used_mb": post_gpu.used_mb if post_gpu else None,
    }

    # Display Rich Table
    table = Table(title="Local LLM Inference Benchmark (RTX 5050 8GB)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model Name", str(results["model"]))
    table.add_row("Avg Generation Latency", f"{results['avg_latency_sec']} s")
    table.add_row("Avg Output Tokens", f"{results['avg_tokens']}")
    table.add_row("Throughput (Tokens/Sec)", f"{results['tokens_per_sec']} tps")
    if post_gpu:
        table.add_row("Peak VRAM Used", f"{results['peak_vram_used_mb']} MiB")
        table.add_row("Final Free VRAM", f"{post_gpu.free_mb} MiB")

    console.print(table)
    return results


if __name__ == "__main__":
    provider = ProviderFactory.create_provider()
    asyncio.run(run_benchmark(provider))

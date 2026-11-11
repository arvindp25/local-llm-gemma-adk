import argparse
import asyncio
import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from agent.gemma_agent import GemmaAgent
from profiler.gpu_monitor import VRAMGuardian
from profiler.benchmark import run_benchmark
from core.registry import ProviderFactory

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console()


async def run_interactive(agent: GemmaAgent):
    console.print(Panel("[bold green]Google Gemma 4 E4B + Google ADK Local Agent[/bold green]\nType 'exit' to quit.", title="Interactive Session"))
    while True:
        try:
            user_input = console.input("\n[bold cyan]User>[/bold cyan] ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                break

            with console.status("[yellow]Gemma is reasoning locally...[/yellow]"):
                res = await agent.execute_task(user_input)

            if res.tool_calls:
                console.print(f"[bold magenta]Tools Used:[/bold magenta] {[tc.name for tc in res.tool_calls]}")
            console.print(Markdown(res.answer))

            if res.distilled_insight:
                console.print(f"[dim italic][Learned Insight]: {res.distilled_insight}[/dim italic]")

        except (KeyboardInterrupt, EOFError):
            break


async def main():
    parser = argparse.ArgumentParser(description="Local LLM Google Gemma & ADK Agent Runner")
    parser.add_argument("--query", "-q", type=str, help="Single query to run")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive chat session")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run VRAM & inference throughput benchmark")
    parser.add_argument("--vram", action="store_true", help="Display current RTX 5050 GPU VRAM snapshot")
    args = parser.parse_args()

    guardian = VRAMGuardian()

    if args.vram:
        snap = guardian.get_snapshot()
        if snap:
            console.print(Panel(
                f"[bold cyan]GPU:[/bold cyan] {snap.name}\n"
                f"[bold cyan]Total VRAM:[/bold cyan] {snap.total_mb} MiB\n"
                f"[bold cyan]Used VRAM:[/bold cyan] {snap.used_mb} MiB\n"
                f"[bold green]Free VRAM:[/bold green] {snap.free_mb} MiB\n"
                f"[bold cyan]GPU Temp:[/bold cyan] {snap.temperature_c} °C\n"
                f"[bold cyan]Utilization:[/bold cyan] {snap.utilization_gpu_pct}%",
                title="Hardware Telemetry (RTX 5050)"
            ))
        else:
            console.print("[red]Could not query NVML GPU snapshot.[/red]")
        return

    if args.benchmark:
        provider = ProviderFactory.create_provider()
        await run_benchmark(provider)
        return

    agent = GemmaAgent()

    if args.query:
        with console.status("[yellow]Gemma is reasoning locally...[/yellow]"):
            res = await agent.execute_task(args.query)
        if res.tool_calls:
            console.print(f"[bold magenta]Tools Executed:[/bold magenta] {[tc.name for tc in res.tool_calls]}")
        console.print(Markdown(res.answer))
        if res.distilled_insight:
            console.print(f"[dim italic][Learned Insight]: {res.distilled_insight}[/dim italic]")
        return

    await run_interactive(agent)


if __name__ == "__main__":
    asyncio.run(main())

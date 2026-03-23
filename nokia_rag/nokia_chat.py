# ── nokia_rag/nokia_chat.py ─────────────────────────────
# Main entry point — run this to start the assistant

from FlagEmbedding import FlagModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich import box

import config
from indexer import load_or_build_index
from retriever import retrieve
from llm import generate_answer

console = Console()


def print_banner():
    banner = Text()
    banner.append("Nokia ISAM RAG Assistant\n", style="bold white")
    banner.append("Fine-tuned BGE-Large  +  BGE Reranker v2  +  Ollama\n", style="dim")
    banner.append("─" * 46 + "\n", style="dim")
    banner.append("Commands:\n", style="yellow")
    banner.append("  sources  ", style="cyan")
    banner.append("→ show reranked chunks from last query\n")
    banner.append("  reset    ", style="cyan")
    banner.append("→ clear conversation history\n")
    banner.append("  clear    ", style="cyan")
    banner.append("→ clear screen\n")
    banner.append("  quit     ", style="cyan")
    banner.append("→ exit\n")

    console.print(Panel(banner, box=box.ROUNDED, border_style="blue"))


def main():
    console.print("\n[bold blue]Initializing Nokia ISAM Assistant...[/bold blue]\n")

    # 1. Load embedding model
    console.print("[cyan]Loading Nokia fine-tuned BGE-Large...[/cyan]")
    model = FlagModel(
        config.MODEL_PATH,
        query_instruction_for_retrieval=config.QUERY_INSTRUCTION,
        use_fp16=True
    )
    console.print("[green]✅ Embedding model loaded[/green]")

    # 2. Load or build chunk index
    chunks, embeddings = load_or_build_index(model)

    console.clear()
    print_banner()
    console.print(
        f"[dim]Indexed {len(chunks)} chunks from Nokia ISAM Alarms Guide  |  "
        f"Model: {config.OLLAMA_MODEL}[/dim]\n"
    )

    # Session state
    last_sources: list[tuple[str, float]] = []
    history: list[dict] = []   # accumulates {"role", "content"} turns

    # 3. Chat loop
    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not query:
            continue

        # ── Commands ──
        if query.lower() == "quit":
            console.print("[dim]Bye.[/dim]")
            break

        if query.lower() == "clear":
            console.clear()
            print_banner()
            continue

        if query.lower() == "reset":
            history.clear()
            last_sources.clear()
            console.print("[dim]Conversation history cleared.[/dim]\n")
            continue

        if query.lower() == "sources":
            if not last_sources:
                console.print("[dim]Ask a question first.[/dim]\n")
            else:
                console.print("\n[bold yellow]Retrieved Chunks:[/bold yellow]")
                for i, (chunk, score) in enumerate(last_sources):
                    console.print(
                        f"\n[yellow]── Source {i+1}  "
                        f"[cyan](reranker score: {score:.3f})[/cyan] ──[/yellow]"
                    )
                    console.print(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                console.print()
            continue

        # ── Retrieval + Reranking ──
        with console.status("[dim]Searching and reranking Nokia documentation...[/dim]"):
            top_chunks, scores = retrieve(query, model, chunks, embeddings)
            last_sources = list(zip(top_chunks, scores))

        # ── Generation ──
        with console.status("[dim]Generating answer...[/dim]"):
            answer = generate_answer(query, top_chunks, history)

        # ── Display ──
        console.print(f"\n[bold green]Assistant:[/bold green]")
        console.print(Markdown(answer))
        turns = len(history) // 2
        console.print(
            f"[dim]  ↑ {len(top_chunks)} chunks (reranked from {config.TOP_K_RETRIEVAL})  "
            f"|  top score: {scores[0]:.3f}  "
            f"|  turn {turns}  "
            f"|  'sources' / 'reset' / 'quit'[/dim]\n"
        )


if __name__ == "__main__":
    main()

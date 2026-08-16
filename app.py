"""Typer CLI: collect, fetch, process, dedupe, export, stats, serve, prune.

Every command follows the same wiring: load_config -> setup_logging -> init_db ->
call one function from the relevant module. No business logic lives here.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.config import Config, ConfigError, load_config
from src.log import setup_logging
from src.stats import collect_stats, prune_rejected
from src.storage.db import init_db

app = typer.Typer(help="Funny Animals Enricher -- build an ML-ready dataset of animal videos.")
console = Console()


def _bootstrap() -> Config:
    """Shared wiring for every command: load_config -> setup_logging -> init_db."""
    try:
        cfg = load_config()
    except ConfigError as exc:  # a stack trace here only hides the actionable message
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1) from None
    setup_logging(level=cfg.app.log_level, log_dir=cfg.storage.log_path)
    init_db(cfg.storage.database)
    return cfg


@app.command()
def collect(
    source: str = typer.Option("all", help="pexels|pixabay|all"),
    query: str = typer.Option(..., help="Search query"),
    limit: int = typer.Option(20, help="Max videos to collect"),
) -> None:
    """Collect videos from an API-based source."""
    cfg = _bootstrap()
    from src.collectors import run_collect

    try:
        run_stats = asyncio.run(run_collect(cfg, source, query, limit))
    except ValueError as exc:  # unknown or disabled source — a traceback adds nothing
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    console.print(
        f"found={run_stats.found} downloaded={run_stats.downloaded} "
        f"skipped={run_stats.skipped} errors={run_stats.errors}"
    )


@app.command()
def fetch(
    urls: Path = typer.Option(None, help="Path to a text file with one URL per line"),
    from_queue: bool = typer.Option(False, "--from-queue", help="Take URLs from the browser queue"),
    watch: bool = typer.Option(False, "--watch", help="With --from-queue: keep polling for new URLs"),
    limit: int = typer.Option(50, help="Max URLs to process"),
) -> None:
    """Fetch videos via yt-dlp from a URL list or from the browser queue."""
    if (urls is None) == (not from_queue):
        console.print("[red]Pass either --urls or --from-queue, not both.[/red]")
        raise typer.Exit(1)
    cfg = _bootstrap()
    from src.collectors.ytdlp import run_fetch, run_fetch_from_queue

    try:
        if from_queue:
            result = run_fetch_from_queue(cfg, limit, watch)
        else:
            result = run_fetch(cfg, urls, limit)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:  # --watch runs until you stop it
        raise typer.Exit(130) from None
    console.print(
        f"found={result.found} downloaded={result.downloaded} "
        f"skipped={result.skipped} errors={result.errors}"
    )
    for reason, count in result.reasons.most_common():
        console.print(f"  skipped [{reason}]: {count}")


@app.command()
def process(
    detect_animals: bool = typer.Option(False, "--detect-animals", help="Run animal detection"),
    check_quality: bool = typer.Option(False, "--check-quality", help="Run quality assessment"),
) -> None:
    """Probe, normalize, dedupe and (optionally) detect animals in downloaded videos."""
    cfg = _bootstrap()
    from src.processors import run_processing

    run_processing(cfg, detect_animals, check_quality)


@app.command(name="compile")
def compile_short(
    category: str = typer.Option("", help="Only use clips of this category (dog, cat, ...)"),
    count: int = typer.Option(1, help="How many shorts to build, each from different clips"),
    keep_work: bool = typer.Option(False, "--keep-work", help="Keep the intermediate segments"),
) -> None:
    """Assemble processed clips into a 1080x1920 short, captioned by the local model."""
    cfg = _bootstrap()
    from src.compiler import CompileError, PlanError, build_short, keep, recall

    # what earlier runs spent, or a second `compile` remakes the first one's shorts
    used, themes = recall(cfg)
    try:
        for index in range(count):
            try:
                out = build_short(cfg, category or None, keep_work, used, themes)
            except (CompileError, PlanError) as exc:
                console.print(f"[red]{exc}[/red]")
                # a later short running out of unused clips is a stop, not a failure
                raise typer.Exit(0 if index else 1) from None
            console.print(f"[green]{out}[/green]")
    finally:
        keep(cfg, used, themes)


@app.command()
def dedupe(
    method: str = typer.Option("hash", help="hash|phash"),
) -> None:
    """Mark duplicate videos."""
    cfg = _bootstrap()
    from src.processors.dedupe import run_dedupe

    run_dedupe(cfg, method)


@app.command()
def export(
    format: str = typer.Option(..., "--format", help="coco|webdataset|huggingface"),
    output: Path = typer.Option(..., help="Output directory"),
    push: bool = typer.Option(False, help="Push to HuggingFace Hub"),
    include_unlicensed: bool = typer.Option(False, help="Include license='unknown' videos"),
) -> None:
    """Export the processed dataset."""
    cfg = _bootstrap()
    from src.export import run_export

    run_export(cfg, format, output, push, include_unlicensed)


@app.command()
def stats(
    show_categories: bool = typer.Option(False, "--show-categories", help="Break down by category"),
    show_sources: bool = typer.Option(False, "--show-sources", help="Break down by source"),
) -> None:
    """Print dataset statistics."""
    _bootstrap()
    data = collect_stats(show_categories, show_sources)

    table = Table(title="Dataset stats")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("total", str(data.get("total", 0)))
    for status_name, count in data.get("by_status", {}).items():
        table.add_row(f"status:{status_name}", str(count))
    if show_categories:
        for category, count in data.get("by_category", {}).items():
            table.add_row(f"category:{category or 'none'}", str(count))
    if show_sources:
        for source_name, count in data.get("by_source", {}).items():
            table.add_row(f"source:{source_name}", str(count))
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
) -> None:
    """Run the FastAPI server."""
    cfg = _bootstrap()
    import uvicorn
    from src.api import create_app

    uvicorn.run(create_app(cfg), host=host, port=port)


@app.command(name="browser-mode")
def browser_mode(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
) -> None:
    """Run the server with the queue endpoint the browser extension posts into."""
    cfg = _bootstrap()
    if not cfg.browser_mode.enabled:
        console.print("[red]browser_mode.enabled is false in config.yaml.[/red]")
        raise typer.Exit(1)
    if not cfg.browser_mode.ingest_token:
        import secrets

        console.print("[red]No ingest token.[/red] Add this line to .env and rerun:")
        console.print(f"  BROWSER_INGEST_TOKEN={secrets.token_urlsafe(32)}")
        raise typer.Exit(1)

    import uvicorn
    from src.api import create_app

    console.print(f"Extension: {Path(__file__).parent / 'browser_extension'}")
    console.print(f"Server:    http://{host}:{port}")
    console.print(f"Token:     {cfg.browser_mode.ingest_token}")
    console.print("Drain the queue elsewhere: app.py fetch --from-queue --watch")
    uvicorn.run(create_app(cfg), host=host, port=port)


@app.command()
def prune() -> None:
    """Delete on-disk files for rejected videos."""
    cfg = _bootstrap()
    count = prune_rejected(cfg)
    console.print(f"pruned {count} file(s)")


if __name__ == "__main__":
    app()

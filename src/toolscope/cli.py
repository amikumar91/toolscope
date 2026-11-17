from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from .config import Suite
from .metrics import GroupStats, RunScore, aggregate
from .report import print_console, render_html
from .runner import load_results, run_suite
from .schema import UnsupportedSchemaVersion, check_schema_version

app = typer.Typer(
    add_completion=False, help="Measure whether LLMs find and use your MCP tools effectively."
)
console = Console()


@app.command()
def run(
    suite_path: Path = typer.Argument(..., help="suite.yaml"),
    out: Path = typer.Option(Path("toolscope-results"), "--out", "-o"),
    repeats: int = typer.Option(None, "--repeats", "-k", help="override suite repeats"),
    agent: list[str] = typer.Option(None, "--agent", "-a", help="only these agent ids"),
    mode: list[str] = typer.Option(None, "--mode", "-m", help="only these modes: all | search"),
    fail_under: float = typer.Option(
        None, help="exit 1 if any group's pass^k is below this (0-1). For CI."
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="keep the runs already in --out (from an interrupted run) and run only the rest; "
        "infra-error runs are retried. Use the same suite and flags.",
    ),
):
    """Run a suite and write traces, scores, summary.json and report.html."""
    suite = Suite.load(suite_path)
    previous = None
    if resume:
        try:
            previous = load_results(out)
        except UnsupportedSchemaVersion as e:
            console.print(f"[red]ERROR: cannot resume: {escape(str(e))}[/red]")
            raise typer.Exit(1) from None
    if repeats:
        suite.repeats = repeats
    if agent:
        suite.agents = [a for a in suite.agents if a.id in agent]
    if mode:
        suite.exposure.modes = [m for m in suite.exposure.modes if m in mode] or list(mode)  # type: ignore
    total = (
        sum(len(a.modes or suite.exposure.modes) for a in suite.agents)
        * len(suite.tasks)
        * suite.repeats
    )
    out.mkdir(parents=True, exist_ok=True)
    with Progress(
        TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(), console=console
    ) as prog:
        bar = prog.add_task("running", total=total)
        result = asyncio.run(
            run_suite(
                suite,
                on_progress=lambda s: prog.advance(bar),
                server_log=str(out / "servers.log"),
                out_dir=out,
                resume_from=previous,
            )
        )
    result = result.redacted()
    path = result.save(out)
    (path / "report.html").write_text(
        render_html(
            result.stats,
            result.scores,
            suite_name=suite.name,
            tool_count=result.tool_count,
            catalogue_tokens=result.catalogue_tokens,
        ),
        encoding="utf-8",
    )
    print_console(result.stats, console)
    console.print(
        f"\n[dim]wrote {path}/report.html, summary.json, scores.jsonl, traces.jsonl, servers.log[/dim]"
    )
    if resume:
        console.print(f"[dim]resumed {result.resumed} completed run(s) from {out}[/dim]")
    if result.infra_errors:
        console.print(
            f"[yellow]{result.infra_errors} run(s) hit infrastructure errors (rate limits, timeouts, "
            f"dead servers) and were excluded from the metrics; rerun with --resume to retry them[/yellow]"
        )
    failed = False
    for agent_id, err in result.config_errors.items():
        console.print(
            f"[red]CONFIG ERROR: agent {agent_id!r} could not be built, "
            f"all its runs recorded as crash: {escape(err)}[/red]"
        )
        failed = True
    if fail_under is not None:
        bad = [g for g in result.stats if g.pass_hat_k < fail_under]
        if bad:
            console.print(
                f"[red]FAIL: {', '.join(f'{g.agent_id}/{g.mode} pass^k={g.pass_hat_k:.0%}' for g in bad)} "
                f"< {fail_under:.0%}[/red]"
            )
            failed = True
    if failed:
        raise typer.Exit(1)


@app.command()
def report(results_dir: Path = typer.Argument(Path("toolscope-results"))):
    """Re-render the report from saved scores.jsonl (any schema_version 1.x; files without one are 1)."""
    try:
        summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
        check_schema_version(summary, "summary.json")
        scores = []
        lines = (results_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            if line.strip():
                rec = json.loads(line)
                check_schema_version(rec, f"scores.jsonl line {i}")
                scores.append(RunScore.model_validate(rec))
    except UnsupportedSchemaVersion as e:
        console.print(f"[red]ERROR: {escape(str(e))}[/red]")
        raise typer.Exit(1) from None
    stats: list[GroupStats] = aggregate(scores)
    (results_dir / "report.html").write_text(
        render_html(
            stats,
            scores,
            suite_name=summary["suite"],
            tool_count=summary["tool_count"],
            catalogue_tokens=summary["catalogue_tokens"],
        ),
        encoding="utf-8",
    )
    print_console(stats, console)


@app.command()
def tools(suite_path: Path = typer.Argument(...)):
    """List the tools every server exposes, with schema token cost (sanity check before a run)."""
    from .mcp_pool import MCPPool
    from .tokens import count

    suite = Suite.load(suite_path)

    async def go():
        async with MCPPool(suite.servers, base_dir=str(suite.base_dir)) as pool:
            return list(pool.tools.values())

    from rich.table import Table

    specs = asyncio.run(go())
    t = Table(title=f"{len(specs)} tools")
    for c in ("tool", "tokens", "description"):
        t.add_column(c)
    for s in sorted(specs, key=lambda s: s.qualified):
        t.add_row(s.qualified, str(count(s.openai())), s.description[:90])
    console.print(t)
    console.print(f"catalogue total ≈ {count([s.openai() for s in specs]):,} tokens")


if __name__ == "__main__":
    app()

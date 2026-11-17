"""Console + self-contained HTML report."""

from __future__ import annotations

import html
from collections import Counter, defaultdict
from datetime import datetime

from rich.console import Console
from rich.table import Table

from .metrics import INFRA, STAGES, GroupStats, RunScore


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def _num(x: float | None, d: int = 0) -> str:
    return "—" if x is None else f"{x:,.{d}f}"


def print_console(stats: list[GroupStats], console: Console | None = None) -> None:
    c = console or Console()
    t = Table(title="toolscope — tool-use effectiveness", show_lines=False)
    for col in [
        "agent",
        "mode",
        "pass@1",
        "pass^k",
        "recall@k",
        "select",
        "args",
        "schema✓",
        "err/call",
        "calls",
        "schema tok",
        "tok/success",
        "top failure",
        "infra",
    ]:
        t.add_column(
            col, justify="right" if col not in ("agent", "mode", "top failure") else "left"
        )
    for g in stats:
        fails = {k: v for k, v in g.funnel.items() if v}
        top = max(fails, key=lambda k: fails[k]) if fails else "—"
        t.add_row(
            g.agent_id,
            g.mode,
            _pct(g.pass_at_1),
            _pct(g.pass_hat_k),
            _pct(g.recall_at_k),
            _pct(g.selection_accuracy),
            _pct(g.arg_accuracy),
            _pct(g.schema_valid_rate),
            _pct(g.call_error_rate),
            _num(g.avg_calls, 1),
            _num(g.avg_schema_tokens),
            _num(g.tokens_per_success),
            f"{top} ({fails[top]})" if fails else "—",
            str(g.infra_errors) if g.infra_errors else "—",
        )
    c.print(t)


# ---------------------------------------------------------------------------------------------------
_SEG = ["passed"] + STAGES  # fixed categorical order -> slots 1..7

_CSS = """
.viz-root{--surface-1:#fcfcfb;--surface-2:#f3f2ee;--line:#e4e2dc;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#7a7972;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;}
@media (prefers-color-scheme:dark){.viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242422;--line:#34332f;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#8f8e86;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#0ca30c;--s7:#9085e9;}}
*{box-sizing:border-box}body{margin:0}
.viz-root{background:var(--surface-1);color:var(--text-primary);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;padding:32px;min-height:100vh}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:36px 0 10px;letter-spacing:.01em}
.sub{color:var(--text-secondary);margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{background:var(--surface-2);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--text-secondary);font-size:12px}.tile .v{font-size:26px;font-weight:600;margin-top:2px}
.tile .n{color:var(--text-muted);font-size:12px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-weight:600;color:var(--text-secondary);font-size:12px}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
.scroll{overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:0 0 12px;color:var(--text-secondary);font-size:12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.frow{display:grid;grid-template-columns:220px 1fr 70px;align-items:center;gap:12px;margin:8px 0}
.frow .lbl{font-size:13px}.frow .lbl small{color:var(--text-muted)}
.bar{display:flex;gap:2px;height:18px}
.seg{position:relative;height:100%;min-width:3px}
.seg:first-child{border-radius:4px 0 0 4px}.seg:last-child{border-radius:0 4px 4px 0}.seg:only-child{border-radius:4px}
.seg:hover{outline:2px solid var(--text-primary);outline-offset:1px;z-index:2}
.seg .tip{display:none;position:absolute;bottom:26px;left:50%;transform:translateX(-50%);background:var(--text-primary);color:var(--surface-1);
padding:6px 9px;border-radius:6px;font-size:12px;white-space:nowrap;pointer-events:none}
.seg:hover .tip{display:block}
.frow .pct{color:var(--text-secondary);text-align:right;font-variant-numeric:tabular-nums}
.cell-ok{font-weight:600}.muted{color:var(--text-muted)}
.why{color:var(--text-muted);font-size:12px;display:block}
dl{display:grid;grid-template-columns:180px 1fr;gap:6px 16px;color:var(--text-secondary)}dt{color:var(--text-primary);font-weight:600}
"""

_GLOSSARY = [
    ("pass@1", "Share of individual runs that fully succeeded."),
    (
        "pass^k",
        "Share of tasks that succeeded on all k repeats: consistency, the number production cares about.",
    ),
    ("pass@k", "Share of tasks that succeeded on at least one of k repeats: capability ceiling."),
    (
        "recall@k (search)",
        "Expected tools that appeared in search results / expected tools. Low = retrieval problem, fix descriptions or retriever.",
    ),
    ("MRR (search)", "1 / rank of the first expected tool in search results."),
    ("skipped search", "Model called a tool before ever searching (search mode)."),
    ("selection", "Expected tools actually called / expected tools (tool tasks)."),
    (
        "precision",
        "Distinct tools called that were expected / distinct tools called. Low = wandering or decoy picks.",
    ),
    ("irrelevance", "On no-tool tasks, share of runs that correctly called nothing."),
    ("schema ✓", "Calls whose arguments validate against the tool's inputSchema."),
    ("args", "Expected calls that had a successful call with matching arguments."),
    ("err/call", "Tool calls that returned an error / all tool calls."),
    ("recovery", "Runs with an error that later got a successful call."),
    ("schema tok", "Tokens of tool schemas placed in context, summed over turns: the bloat cost."),
    ("tok/success", "All input+output tokens / successful runs."),
    (
        "infra",
        "Runs where infrastructure failed (rate limit or network/auth error after retries, dead MCP "
        "server, run timeout). Excluded from every other metric; rerun with --resume to retry them.",
    ),
]


def _seg_color(i: int) -> str:
    return f"var(--s{i + 1})"


def render_html(
    stats: list[GroupStats],
    scores: list[RunScore],
    *,
    suite_name: str,
    tool_count: int,
    catalogue_tokens: int,
) -> str:
    e = html.escape
    best = stats[0] if stats else None
    total_runs = sum(g.runs for g in stats)
    infra_runs = sum(g.infra_errors for g in stats)
    p = []
    p.append(
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>toolscope · {e(suite_name)}</title><style>{_CSS}</style></head><body><div class='viz-root'><div class='wrap'>"
    )
    p.append(
        f"<h1>Tool-use effectiveness · {e(suite_name)}</h1>"
        f"<p class='sub'>{tool_count} tools across the connected MCP servers · full catalogue ≈ {catalogue_tokens:,} tokens · "
        f"{len(stats)} agent×mode groups · {total_runs} runs"
        + (f" · {infra_runs} infra-error runs excluded" if infra_runs else "")
        + f" · generated {datetime.now():%Y-%m-%d %H:%M}</p>"
    )

    if best:
        fails = {k: v for k, v in best.funnel.items() if v}
        top = max(fails, key=lambda k: fails[k]) if fails else None
        p.append("<div class='tiles'>")
        for k, v, note in [
            ("Best pass@1", _pct(best.pass_at_1), f"{best.agent_id} · {best.mode}"),
            ("Its pass^k", _pct(best.pass_hat_k), f"k = {best.k}, all repeats pass"),
            ("Its selection accuracy", _pct(best.selection_accuracy), "expected tools called"),
            (
                "Its top failure stage",
                top or "none",
                f"{fails[top]} run(s)" if top else "every run passed",
            ),
        ]:
            p.append(
                f"<div class='tile'><div class='k'>{e(k)}</div><div class='v'>{e(v)}</div><div class='n'>{e(note)}</div></div>"
            )
        p.append("</div>")

    # ---- funnel chart -------------------------------------------------------------------------
    p.append("<h2>Where runs fail: earliest broken stage per run</h2><div class='legend'>")
    for i, seg in enumerate(_SEG):
        p.append(f"<span><i style='background:{_seg_color(i)}'></i>{e(seg)}</span>")
    p.append("</div>")
    for g in stats:
        counts = {"passed": g.runs - sum(g.funnel.values()), **g.funnel}
        p.append(
            f"<div class='frow'><div class='lbl'>{e(g.agent_id)} <small>· {e(g.mode)}</small></div><div class='bar'>"
        )
        for i, seg in enumerate(_SEG):
            n = counts.get(seg, 0)
            if not n:
                continue
            p.append(
                f"<div class='seg' style='flex:{n};background:{_seg_color(i)}' aria-label='{e(seg)} {n}'>"
                f"<span class='tip'>{e(seg)}: {n} of {g.runs} runs ({n / g.runs:.0%})</span></div>"
            )
        p.append(f"</div><div class='pct'>{_pct(g.pass_at_1)}</div></div>")

    # funnel table view (also the accessible alternative to the chart)
    p.append("<div class='scroll'><table><thead><tr><th>agent</th><th>mode</th><th>runs</th>")
    p += [f"<th>{e(s)}</th>" for s in _SEG]
    p.append(f"<th>{INFRA} (excluded)</th></tr></thead><tbody>")
    for g in stats:
        counts = {"passed": g.runs - sum(g.funnel.values()), **g.funnel}
        p.append(f"<tr><td>{e(g.agent_id)}</td><td>{e(g.mode)}</td><td>{g.runs}</td>")
        p += [f"<td>{counts.get(s, 0) or '<span class=muted>0</span>'}</td>" for s in _SEG]
        p.append(f"<td>{g.infra_errors or '<span class=muted>0</span>'}</td></tr>")
    p.append("</tbody></table></div>")

    # ---- leaderboard -------------------------------------------------------------------------
    cols = [
        ("pass@1", lambda g: _pct(g.pass_at_1)),
        ("pass^k", lambda g: _pct(g.pass_hat_k)),
        ("pass@k", lambda g: _pct(g.pass_at_k)),
        ("recall@k", lambda g: _pct(g.recall_at_k)),
        ("MRR", lambda g: _num(g.mrr, 2)),
        ("skipped search", lambda g: _pct(g.skipped_search_rate)),
        ("hallucinated", lambda g: _pct(g.hallucination_rate)),
        ("selection", lambda g: _pct(g.selection_accuracy)),
        ("precision", lambda g: _pct(g.precision)),
        ("irrelevance", lambda g: _pct(g.irrelevance_accuracy)),
        ("schema ✓", lambda g: _pct(g.schema_valid_rate)),
        ("args", lambda g: _pct(g.arg_accuracy)),
        ("err/call", lambda g: _pct(g.call_error_rate)),
        ("recovery", lambda g: _pct(g.recovery_rate)),
        ("calls/run", lambda g: _num(g.avg_calls, 1)),
        ("redundant", lambda g: _num(g.redundant_per_run, 2)),
        ("schema tok", lambda g: _num(g.avg_schema_tokens)),
        ("in tok", lambda g: _num(g.avg_input_tokens)),
        ("tok/success", lambda g: _num(g.tokens_per_success)),
        (
            "$/success",
            lambda g: "—" if g.cost_per_success is None else f"${g.cost_per_success:.4f}",
        ),
        ("p50 ms", lambda g: _num(g.p50_latency_ms)),
        ("retries", lambda g: _num(g.retries)),
        ("infra", lambda g: _num(g.infra_errors)),
    ]
    p.append(
        "<h2>All metrics by agent and exposure mode</h2><div class='scroll'><table><thead><tr><th>agent</th><th>mode</th>"
    )
    p += [f"<th>{e(c)}</th>" for c, _ in cols]
    p.append("</tr></thead><tbody>")
    for g in stats:
        p.append(
            f"<tr><td>{e(g.agent_id)}</td><td>{e(g.mode)}</td>"
            + "".join(f"<td>{e(fn(g))}</td>" for _, fn in cols)
            + "</tr>"
        )
    p.append("</tbody></table></div>")

    # ---- per-task matrix ----------------------------------------------------------------------
    groups = [(g.agent_id, g.mode) for g in stats]
    cells: dict[tuple[str, str, str], list[RunScore]] = defaultdict(list)
    task_ids: list[str] = []
    infra_cells: Counter[tuple[str, str, str]] = Counter()
    for s in scores:
        if s.failure_stage == INFRA:
            infra_cells[(s.task_id, s.agent_id, s.mode)] += 1
        else:
            cells[(s.task_id, s.agent_id, s.mode)].append(s)
        if s.task_id not in task_ids:
            task_ids.append(s.task_id)
    p.append(
        "<h2>Per task: passes / repeats, with the most common failure</h2><div class='scroll'><table><thead><tr><th>task</th><th></th>"
    )
    p += [f"<th>{e(a)}<br><span class='muted'>{e(m)}</span></th>" for a, m in groups]
    p.append("</tr></thead><tbody>")
    for t in task_ids:
        p.append(f"<tr><td>{e(t)}</td><td></td>")
        for a, m in groups:
            rs = cells.get((t, a, m), [])
            ok = sum(r.success for r in rs)
            why = Counter(
                f"{r.failure_stage}: {r.failure_reason}" for r in rs if not r.success
            ).most_common(1)
            cls = "cell-ok" if rs and ok == len(rs) else ""
            n_infra = infra_cells[(t, a, m)]
            p.append(
                f"<td class='{cls}'>{ok}/{len(rs)}"
                + (f"<span class='why'>{e(why[0][0])}</span>" if why else "")
                + (f"<span class='why'>+{n_infra} infra (excluded)</span>" if n_infra else "")
                + "</td>"
            )
        p.append("</tr>")
    p.append("</tbody></table></div>")

    p.append("<h2>Metric definitions</h2><dl>")
    for k, v in _GLOSSARY:
        p.append(f"<dt>{e(k)}</dt><dd style='margin:0'>{e(v)}</dd>")
    p.append("</dl></div></div></body></html>")
    return "".join(p)

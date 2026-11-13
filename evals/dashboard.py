"""Render the evaluation results as one self-contained HTML page."""

from __future__ import annotations

import html
import json

CSS = """
:root{--bg:#f6f8f7;--panel:#fff;--ink:#14241e;--mist:#5a7267;--line:#d5dfda;--ok:#0d8f6e;--okbg:#e3f4ee;--bad:#b3402b;--badbg:#fbe9e5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:1280px;margin:0 auto;padding:28px 20px 60px}h1{font-size:28px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 10px}
p.sub{color:var(--mist);margin:0 0 18px;max-width:900px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);padding:14px 16px}.card.me{border-color:var(--ok);box-shadow:inset 0 3px 0 var(--ok)}
.card .n{font-size:34px;font-weight:650;line-height:1.1}.card .l{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mist)}
.card small{color:var(--mist);display:block}table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line)}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--mist);background:#eef3f0}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.pill{display:inline-block;padding:1px 8px;font-size:12px;font-weight:600;border-radius:10px}
.pass{background:var(--okbg);color:var(--ok)}.fail{background:var(--badbg);color:var(--bad)}.ev{display:block;color:var(--mist);font-size:12px;margin-top:2px}
.tag{font-size:11px;color:var(--mist);border:1px solid var(--line);padding:0 6px;border-radius:3px;white-space:nowrap}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:#eef3f0;padding:0 4px}.wrap{overflow-x:auto}
.graph{background:var(--panel);border:1px solid var(--line);padding:14px;overflow-x:auto}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.two{grid-template-columns:1fr}}.exec{background:var(--panel);border:1px solid var(--line);padding:6px 16px 16px;margin:14px 0}.exec h3{margin:10px 0 2px;font-size:17px}
.exec table{margin-top:10px}details{margin-top:10px}summary{cursor:pointer;color:var(--mist)}.trace td:first-child{white-space:nowrap;font-weight:600}.trace .sup{color:var(--mist);font-size:13px}
"""


def e(x) -> str:
    return html.escape(str(x))


def _trace(run: dict) -> str:
    rows = []
    steps = run["steps"]
    routes = [None, *run["route_log"]]  # the planner has no supervisor decision
    for step, route in zip(steps, routes):
        sup = f"<div class='sup'>↳ supervisor → <b>{e(route['next'])}</b>: {e(route['reason'])}</div>" if route else ""
        rows.append(f"<tr><td>{e(step['agent'])}</td><td>{e(step['note'])}{sup}</td></tr>")
    return f"<table class='trace'>{''.join(rows)}</table>"


def executed_mermaid(run: dict) -> str:
    """The planned graph with this run's actual path drawn on it.

    Solid green edges were taken (×N = how many times), dotted edges were available but not
    used, green boxes ran, grey boxes were planned but never reached.
    """
    plan = run["plan"]
    counts: dict[tuple, int] = {}
    for r in run["route_log"]:
        counts[(r["after"], r["next"])] = counts.get((r["after"], r["next"]), 0) + 1
    ran = {s["agent"] for s in run["steps"]} - {"planner"}
    lines = ["flowchart LR", f"  planner([planner]) --> {plan['entry']}"]
    taken = [0]
    for i, (a, b) in enumerate(plan["edges"], start=1):
        n = counts.get((a, b), 0)
        lines.append(f"  {a} -- ×{n} --> {b}" if n else f"  {a} -.-> {b}")
        if n:
            taken.append(i)
    lines.append("  reporter --> END([END])")
    taken.append(len(plan["edges"]) + 1)
    skipped = [a for a in plan["agents"] if a not in ran]
    lines.append("  classDef ran fill:#e3f4ee,stroke:#0d8f6e,stroke-width:2px")
    lines.append("  classDef skipped fill:#f2f2f2,stroke:#c4c4c4,color:#9a9a9a")
    lines.append(f"  class planner,{','.join(sorted(ran))} ran")
    if skipped:
        lines.append(f"  class {','.join(skipped)} skipped")
    lines.append(f"  linkStyle {','.join(map(str, taken))} stroke:#0d8f6e,stroke-width:2.5px")
    return "\n".join(lines)


def _execution(title: str, what: str, run: dict) -> str:
    report = run.get("report") or {}
    trials = "".join(
        f"<tr><td>{e(t['experiment_id'])}</td><td>{e(t['model_family'])}</td><td>{e(t['status'])}</td>"
        f"<td class='num'>{t['attempts']}</td><td class='num'>{'—' if t['f1'] is None else format(t['f1'], '.3f')}</td>"
        f"<td class='num'>{'—' if t['roc_auc'] is None else format(t['roc_auc'], '.3f')}</td>"
        f"<td class='num'>{'—' if t['threshold'] is None else format(t['threshold'], '.2f')}</td></tr>"
        for t in run["trials"]
    ) or "<tr><td colspan='7'>no model was trained</td></tr>"
    h = run.get("holdout") or {}
    outcome = (f"holdout F1 {h['f1']:.3f}, ROC-AUC {h['roc_auc']:.3f} ({e(h.get('model_family'))})" if h else "no holdout score (nothing to evaluate)")
    repairs = "".join(f"<li>{e(x)}</li>" for x in report.get("repairs") or [])
    return f"""<div class="exec"><h3>{e(title)}</h3><p class="sub">{e(what)}</p>
<p><span class="pill pass">stop reason: {e(run['stop_reason'])}</span> &nbsp; {len(run['trials'])} of {report.get('trial_budget_requested')} trials ·
{report.get('retries', 0)} retries · {report.get('agent_steps')} agent steps · {outcome}</p>
{'<ul>' + repairs + '</ul>' if repairs else ''}
<div class="graph"><pre class="mermaid">{e(executed_mermaid(run))}</pre></div>
<table><tr><th>Trial</th><th>Model family</th><th>Status</th><th class="num">Attempts</th><th class="num">Val F1</th><th class="num">Val ROC-AUC</th><th class="num">Threshold</th></tr>{trials}</table>
<details><summary>Step-by-step trace with every supervisor decision</summary>{_trace(run)}</details></div>"""


def render(p: dict) -> str:
    systems, summary = p["systems"], p["summary"]
    cards = "".join(
        f"<div class='card{' me' if s == 'adaptive' else ''}'><div class='l'>{e(summary[s]['label'])}</div>"
        f"<div class='n'>{summary[s]['all']['passed']}/{summary[s]['all']['total']}</div>"
        f"<small>outcome cases {summary[s]['outcome']['passed']}/{summary[s]['outcome']['total']} · "
        f"capability cases {summary[s]['capability']['passed']}/{summary[s]['capability']['total']}</small>"
        f"<small>mean score {summary[s]['all']['mean_score']:.2f}</small></div>"
        for s in systems
    )
    head = "".join(f"<th>{e(summary[s]['label'])}</th>" for s in systems)
    body = ""
    for c in p["cases"]:
        cells = ""
        for s in systems:
            r = c["results"][s]
            cells += (f"<td><span class='pill {'pass' if r['passed'] else 'fail'}'>{'PASS' if r['passed'] else 'FAIL'} "
                      f"{r['score']:.2f}</span><span class='ev'>{e(r['evidence'])}</span></td>")
        inputs = "; ".join(
            f"{e(v['dataset'])}" + (f" + fault <code>{e(json.dumps(v['fault']))}</code>" if v["fault"] else "")
            + (f" · budget {v['budget']}" if v.get("budget", 3) != 3 else "") for v in c["input"].values()
        )
        body += (f"<tr><td><b>{e(c['title'])}</b><br><code>{e(c['id'])}</code> <span class='tag'>{e(c['kind'])}</span> "
                 f"<span class='tag'>{e(c['measures'])}</span><span class='ev'><b>Input:</b> {inputs}</span>"
                 f"<span class='ev'><b>Expected:</b> {e(c['expected'])}</span><span class='ev'><b>Scoring:</b> {e(c['scoring'])}</span></td>"
                 f"<td class='num'>≥ {c['threshold']:.2f}</td>{cells}</tr>")
    coord = "".join(
        f"<tr><td>{e(summary[s]['label'])}</td><td class='num'>{summary[s]['coordination']['mean_trials']}</td>"
        f"<td class='num'>{summary[s]['coordination']['mean_agent_steps']}</td><td class='num'>{summary[s]['coordination']['mean_seconds']}</td>"
        f"<td class='num'>{e(summary[s]['coordination']['fault_runs_survived'])}</td></tr>"
        for s in systems
    )
    show = p["showcase"]
    b = show["saudi_real"]
    executions = (
        _execution("Real station-day table, 6 trials allowed",
                   "Normal run: every model family is tried, one refinement, then the critic stops on a plateau.", show["saudi_real"])
        + _execution("Corrupted copy of the table",
                     "Null targets, infinite values and duplicated rows: the validator routes to the repairer and re-validates before modelling.", show["saudi_dirty"])
        + _execution("Training outage: the first 4 model fits crash",
                     "The probe is inconclusive, the trainer is retried within its limit, and the run still finishes.", show["training_outage"])
        + _execution("Target with a single class",
                     "Nothing can be learned: the validator sends the run straight to the reporter.", show["invalid_data"])
    )
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    pred = "".join(
        f"<tr><td>{e(summary[s]['label'])}</td><td>{e(p['predictive'][s]['model_family'] or '—')}</td>"
        + "".join(f"<td class='num'>{fmt(p['predictive'][s][k])}</td>" for k in ("f1", "precision", "recall", "roc_auc", "pr_auc"))
        + f"<td class='num'>{p['predictive'][s]['trials']}</td><td class='num'>{p['predictive'][s]['seconds']}</td></tr>"
        for s in systems
    )
    pers = next((p["predictive"][s]["persistence_f1"] for s in systems if p["predictive"][s]["persistence_f1"] is not None), None)
    reasons = "".join(f"<li><code>{e(k)}</code> — {e(v)}</li>" for k, v in b["plan"]["reasons"].items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Saudi dust-visibility forecast — adaptive workflow evaluation</title><style>{CSS}</style></head><body><main>
<h1>Saudi dust-visibility forecast — adaptive multi-agent workflow evaluation</h1>
<p class="sub">{len(p['cases'])} test cases run against {len(systems)} systems with the same table, trial budget and target, and no LLM calls.
Forecast quality is measured by one referee procedure on one shared chronological holdout (the most recent 20% of days). Outcome cases can be passed by any system;
capability cases need an intermediate output or a coordination decision that a system must expose to be scored.</p>
<div class="cards">{cards}</div>
<h2>Test cases</h2><div class="wrap"><table><tr><th>Case · input · expected behaviour · scoring</th><th class="num">Pass if</th>{head}</tr>{body}</table></div>
<h2>Predictive performance (same clean table, same future days)</h2>
<p class="sub">Each system's final model re-fitted and scored by the referee on <code>saudi_model_ready</code>.
Persistence forecast (tomorrow = today) on the same days: F1 {fmt(pers)}.</p>
<table><tr><th>System</th><th>Final model</th><th class="num">F1</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">ROC-AUC</th><th class="num">PR-AUC</th><th class="num">Trials</th><th class="num">Seconds</th></tr>{pred}</table>
<h2>Coordination cost</h2>
<table><tr><th>System</th><th class="num">Mean trials (clean runs)</th><th class="num">Mean agent steps</th><th class="num">Mean seconds</th><th class="num">Fault runs without a crash</th></tr>{coord}</table>
<h2>Workflow graph written by the planner</h2>
<p class="sub">Plan for the real station-day table. Dashed boxes are optional agents the planner chose to include:</p><ul>{reasons}</ul>
<div class="graph"><pre class="mermaid">{e(b['mermaid'])}</pre></div>
<h2>Adaptive executions (real runs)</h2>
<p class="sub">Each graph is the planned workflow with the path this run actually took: green boxes ran, green edges were taken
(×N = number of times), dotted edges were available but unused, grey boxes were never reached.</p>
{executions}
</main><script type="module">import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs';mermaid.initialize({{startOnLoad:true,theme:'neutral'}});</script></body></html>"""

"""Renders report.json into the Markdown summary. Leads with the metrics agreed with Eventum:
F1 per typology, alerts settled at high confidence, P90 latency, cost per 1,000 alerts."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from triage.schema import TYPOLOGIES

ORDER = ["jev", "gpt-4o-mini", "gpt-4o-mini-reasoning", "reference_model", "rules"]
TITLES = {
    "jev": "Jev",
    "gpt-4o-mini": "GPT-4o-mini (label only)",
    "gpt-4o-mini-reasoning": "GPT-4o-mini (reasoning first)",
    "reference_model": "Supervised ref. (ceiling)",
    "rules": "Analyst rules (floor)",
}
HEADLINE_TAU = 0.85


def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def _pct(x: Any, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.{nd}f}%"


def _table(header: List[str], rows: List[List[str]], align: str = "l") -> List[str]:
    sep = ["---"] + ["---:" if align == "r" else "---"] * (len(header) - 1)
    return ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"] + [
        "| " + " | ".join(r) + " |" for r in rows
    ]


def render_markdown(payload: Dict[str, Any]) -> str:
    meta, models, boot = payload["meta"], payload["models"], payload["paired_bootstrap"]
    names = [m for m in ORDER if m in models]
    api = [m for m in names if "latency" in models[m]]
    L: List[str] = ["# Fraud alert triage: Jev vs GPT-4o-mini", ""]

    if meta["mode"] != "live":
        L += ["> **DRY RUN — simulated model outputs. These numbers mean nothing. Do not publish.**", ""]
    elif not meta["publishable"]:
        L += ["> **Stand-in transaction data (not PaySim). Do not publish.**", ""]

    ds = meta["dataset"]
    L += _table(["Run", ""], [
        ["Timestamp (UTC)", meta["timestamp_utc"]],
        ["Dataset", f"`{ds['path']}` · n={ds['n']} · sha256 `{ds['sha256'][:12]}` · {', '.join(ds['sources'])}"],
        ["Criteria sha256", f"`{meta['criteria_sha256'][:12]}`"],
        ["Models served", "; ".join(f"{k}: {', '.join(v) or '–'}" for k, v in meta["models_served"].items()) or "–"],
        ["Concurrency / model", str(meta["concurrency_per_model"])],
        ["Client region", meta["client_region"]],
        ["Git", f"`{(meta['git']['commit'] or 'none')[:10]}`" + (" (dirty)" if meta["git"]["dirty"] else "")],
    ])

    # Headline
    L += ["", f"## Headline (settle when p(predicted class) ≥ {HEADLINE_TAU})", ""]
    rows = []

    def settle(m: str) -> Dict[str, Any]:
        return next(r for r in models[m]["settlement"] if math.isclose(r["tau"], HEADLINE_TAU))

    pm = boot["per_model"]
    rows.append(["Macro F1 [95% CI]"] + [
        f"{_f(models[m]['classification']['macro_f1'])} [{_f(pm[m]['macro_f1'][0])}, {_f(pm[m]['macro_f1'][1])}]"
        for m in names])
    rows.append(["Accuracy"] + [_pct(models[m]["classification"]["accuracy"]) for m in names])
    rows.append(["Settled automatically"] + [
        f"{settle(m)['settled']}/{models[m]['n']} ({_pct(settle(m)['coverage'])})" if "settlement" in models[m] else "–"
        for m in names])
    rows.append(["Accuracy of settled"] + [_pct(settle(m)["accuracy"]) if "settlement" in models[m] else "–" for m in names])
    rows.append(["Fraud auto-cleared as NORMAL"] + [
        str(settle(m)["fraud_auto_cleared"]) if "settlement" in models[m] else "–" for m in names])
    rows.append(["P50 / P90 / P99 latency (ms)"] + [
        f"{_f(models[m]['latency']['p50_ms'], 0)} / **{_f(models[m]['latency']['p90_ms'], 0)}** / {_f(models[m]['latency']['p99_ms'], 0)}"
        if m in api and models[m]["latency"] else "–" for m in names])
    rows.append(["P90 95% CI (ms)"] + [
        f"[{_f(models[m]['latency']['p90_ci95_ms'][0], 0)}, {_f(models[m]['latency']['p90_ci95_ms'][1], 0)}]"
        if m in api and models[m]["latency"] else "–" for m in names])
    rows.append(["Cost per 1,000 alerts (USD)"] + [
        f"${models[m]['cost']['usd_per_1k_alerts']:.4f}" if m in api and models[m]["cost"] else "–" for m in names])
    rows.append(["Failed calls"] + [str(models[m]["errors"]) if m in api else "–" for m in names])
    L += _table(["Metric"] + [TITLES[m] for m in names], rows, "r")

    if boot["differences"]:
        L += ["", "### Paired differences (bootstrap over the same alerts)", ""]
        drows = []
        for pair, d in boot["differences"].items():
            a, b = pair.split(" - ")
            if a in api or b in api:
                drows.append([
                    f"{TITLES.get(a, a)} − {TITLES.get(b, b)}",
                    f"[{_f(d['macro_f1']['ci95'][0])}, {_f(d['macro_f1']['ci95'][1])}]",
                    f"[{_pct(d['accuracy']['ci95'][0])}, {_pct(d['accuracy']['ci95'][1])}]",
                    _pct(d["accuracy"]["p_a_better"], 0),
                ])
        L += _table(["Comparison", "Δ macro F1 95% CI", "Δ accuracy 95% CI", "P(first better)"], drows, "r")

    # Per typology
    L += ["", "## F1 per typology", ""]
    trows = []
    for t in TYPOLOGIES:
        sup = models[names[0]]["classification"]["per_typology"][t]["support"]
        trows.append([f"`{t}`", str(sup)] + [
            f"{_f(models[m]['classification']['per_typology'][t]['f1'])} "
            f"(P {_f(models[m]['classification']['per_typology'][t]['precision'], 2)}, "
            f"R {_f(models[m]['classification']['per_typology'][t]['recall'], 2)})"
            for m in names])
    L += _table(["Typology", "n"] + [TITLES[m] for m in names], trows, "r")

    if "by_difficulty" in models[names[0]]:
        L += ["", "## Clear vs ambiguous alerts", "",
              f"Ambiguous = {meta['ambiguous_definition']}. These alerts are hard by construction; "
              "no model should be confidently right on all of them.", ""]
        cohorts = list(models[names[0]]["by_difficulty"])
        crows = [[c, str(models[names[0]]["by_difficulty"][c]["n"])] +
                 [_pct(models[m]["by_difficulty"][c]["accuracy"]) for m in names] for c in cohorts]
        L += _table(["Cohort", "n"] + [TITLES[m] for m in names], crows, "r")

    # Settlement curves
    prob_models = [m for m in names if "settlement" in models[m]]
    L += ["", "## Settlement curve (coverage / accuracy of settled / fraud auto-cleared)", ""]
    taus = [r["tau"] for r in models[prob_models[0]]["settlement"]]
    srows = []
    for i, tau in enumerate(taus):
        srows.append([f"{tau:.2f}"] + [
            f"{_pct(models[m]['settlement'][i]['coverage'], 0)} / {_pct(models[m]['settlement'][i]['accuracy'])} / "
            f"{models[m]['settlement'][i]['fraud_auto_cleared']}" for m in prob_models])
    L += _table(["τ"] + [TITLES[m] for m in prob_models], srows, "r")

    # Calibration
    L += ["", "## Calibration of typology probabilities", ""]
    L += _table(["Metric"] + [TITLES[m] for m in prob_models], [
        ["Multi-class Brier (↓)"] + [_f(models[m]["calibration"]["brier"]) for m in prob_models],
        ["Log loss (↓)"] + [_f(models[m]["calibration"]["log_loss"]) for m in prob_models],
        ["Top-label ECE, 10 bins (↓)"] + [_f(models[m]["calibration"]["ece_top_label"]) for m in prob_models],
        ["Probability source"] + [", ".join(models[m].get("probability_source", ["supervised"])) for m in prob_models],
    ], "r")

    # Hold gate
    hold_models = [m for m in api if "hold_gate" in models[m]]
    if hold_models:
        L += ["", "## Hold / release gate (`noul`)", "",
              "Release if p(hold) ≤ 0.30, hold if ≥ 0.70, otherwise send to an analyst.", ""]
        hrows = []
        for key, label in [("auc", "ROC AUC"), ("brier", "Brier (↓)"), ("accuracy_at_0_5", "Accuracy at 0.5")]:
            hrows.append([label] + [
                _pct(models[m]["hold_gate"][key]) if key == "accuracy_at_0_5" else _f(models[m]["hold_gate"][key])
                for m in hold_models])
        for key, label in [("released", "Released"), ("held", "Held"), ("sent_to_review", "Sent to review"),
                           ("fraud_released", "Fraud released"), ("legit_held", "Legitimate held")]:
            hrows.append([label] + [str(models[m]["hold_gate"][key]) for m in hold_models])
        L += _table(["Metric"] + [TITLES[m] for m in hold_models], hrows, "r")

    # Scenario breakdown
    L += ["", "## Accuracy by scenario", ""]
    scen = list(models[names[0]]["by_scenario"])
    L += _table(["Scenario", "n"] + [TITLES[m] for m in names], [
        [f"`{s}`", str(models[names[0]]["by_scenario"][s]["n"])] +
        [_pct(models[m]["by_scenario"][s]["accuracy"], 0) for m in names] for s in scen
    ], "r")
    return "\n".join(L) + "\n"

"""
Builds and executes the analysis notebooks:

    uv run --group notebooks python notebooks/build_notebooks.py

01_dataset_analysis.ipynb  - what the alerts look like, how separable they are, the
                             accuracy ceiling/floor, and which alerts are ambiguous.
02_results_analysis.ipynb  - deep dive into the latest benchmark run (results/live/...).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent


def md(s: str):
    return nbformat.v4.new_markdown_cell(s.strip())


def code(s: str):
    return nbformat.v4.new_code_cell(s.strip())


SETUP = """
%matplotlib inline
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from nbutil import *
style()
pd.set_option("display.precision", 3)
"""

DATASET_NB = [
    md("""
# 01 · Dataset analysis

What the benchmark alerts look like, before any model sees them:

1. class and scenario mix,
2. how the key signals differ by typology, including the deliberate **look-alike** scenarios,
3. the **accuracy ceiling** (a supervised model trained on 10k separate alerts) and the **floor** (analyst rules),
4. which test alerts are **ambiguous by construction**.

Labels come from the scenario that generated each alert, never from rules applied to its features.
"""),
    code(SETUP + """
from triage import baselines
from triage.schema import TYPOLOGIES
DATA = data_dir()
test = load_jsonl(DATA / "alerts_test.jsonl")
reference = load_jsonl(DATA / "alerts_reference.jsonl")
print(f"{len(test)} test alerts, {len(reference)} reference alerts from {DATA.relative_to(ROOT)}")
print("transaction source:", sorted({a['provenance']['source'] for a in test}))
"""),
    md("## 1 · Class and scenario mix"),
    code("""
def flat(alerts):
    rows = []
    for a in alerts:
        t, s = a["transaction"], a["signals"]
        rows.append({"typology": a["label"]["typology"], "scenario": a["label"]["scenario"].split("/")[1],
                     "type": t["type"], "amount": t["amount"], "drain_ratio": baselines.drain_ratio(t),
                     "pool": a["provenance"]["paysim_pool"], **s})
    return pd.DataFrame(rows)
df = flat(test)
mix = df.groupby(["typology", "scenario", "pool"]).size().rename("n").reset_index()
mix
"""),
    md("""
## 2 · Signals by typology

Each panel is one signal. Boxes show the middle 50% of alerts in that typology; whiskers the 5th–95th percentile.
Overlap between boxes is intended: real customers travel, change phones and make large payments.
"""),
    code("""
panels = [
    ("drain_ratio", "Share of balance moved", False),
    ("device_risk_score", "Device risk score", False),
    ("hours_since_credential_change", "Hours since credential change (log)", True),
    ("hours_since_inbound_funding", "Hours since inbound funding (log)", True),
    ("account_age_days", "Account age, days (log)", True),
    ("transfers_8k_to_10k_24h", "Transfers of \\\\$8k–\\\\$10k in 24h", False),
]
NICE = {"NORMAL": "Normal", "MONEY_MULE": "Money mule", "ACCOUNT_TAKEOVER": "Account takeover",
        "SYNTHETIC_ID": "Synthetic ID", "STRUCTURING_AML": "Structuring (AML)"}
fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), constrained_layout=True)
for ax, (col, title, log) in zip(axes.flat, panels):
    data = [df.loc[df.typology == t, col].clip(lower=0.1 if log else None) for t in TYPOLOGIES]
    bp = ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False, widths=0.55, patch_artist=True)
    for patch, t in zip(bp["boxes"], TYPOLOGIES):
        patch.set(facecolor=TYPOLOGY_COLORS[t], alpha=0.85, edgecolor="#fcfcfb")
    for k in ("medians",):
        for line in bp[k]: line.set(color=INK, linewidth=1.5)
    ax.set_yticks(range(1, 6), [NICE[t] for t in TYPOLOGIES] if ax in axes[:, 0] else [""] * 5)
    if log: ax.set_xscale("log")
    ax.set_title(title, loc="left"); ax.grid(axis="y", visible=False)
plt.show()
"""),
    md("""
### Look-alike pairs

Each pair shares most of its signal distribution; only weak cues separate them. These pairs are where we expect
honest uncertainty from a good model, and where confident answers are most likely to be wrong.
"""),
    code("""
pairs = [("NORMAL/new_device_large_payment", "ACCOUNT_TAKEOVER/domestic_proxy"),
         ("NORMAL/traveler_no_notice_large", "ACCOUNT_TAKEOVER/session_hijack"),
         ("NORMAL/payroll_batch", "STRUCTURING_AML/light"),
         ("NORMAL/new_customer", "SYNTHETIC_ID/sleeper"),
         ("MONEY_MULE/mule_new_phone", "ACCOUNT_TAKEOVER/domestic_proxy")]
ref_df = flat(reference); ref_df["full"] = ref_df.typology + "/" + ref_df.scenario
cols = ["drain_ratio", "is_known_device", "device_risk_score", "hours_since_credential_change",
        "failed_login_attempts_24h", "hours_since_inbound_funding", "recipient_is_first_time",
        "account_age_days", "transfers_8k_to_10k_24h", "distinct_counterparties_24h"]
summary = ref_df.groupby("full")[cols].median()
pd.concat({f"{a.split('/')[1]} vs {b.split('/')[1]}": summary.loc[[a, b]] for a, b in pairs}).round(2)
"""),
    md("""
## 3 · Ceiling and floor

**Ceiling:** gradient boosting trained on the reference split. The learning curve shows whether more training data would
raise it. A flat curve means the remaining errors are ambiguity in the data, not model capacity.

**Floor:** if-then rules an analyst would write from the same criteria text the models receive.
"""),
    code("""
from sklearn.metrics import accuracy_score, f1_score
y_test = [a["label"]["typology"] for a in test]
sizes = [250, 500, 1000, 2500, 5000, len(reference)]
curve = []
for n in sizes:
    m = baselines.train_reference_model(reference[:n])
    pred = [max(p, key=p.get) for p in baselines.reference_predict(m, test)]
    curve.append({"train_size": n, "accuracy": accuracy_score(y_test, pred),
                  "macro_f1": f1_score(y_test, pred, average="macro")})
curve = pd.DataFrame(curve)
rules_pred = [baselines.rules_predict(a) for a in test]
floor = accuracy_score(y_test, rules_pred)

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(curve.train_size, curve.accuracy, marker="o", markersize=6, color=MODEL_COLORS["reference_model"])
ax.axhline(floor, color=MODEL_COLORS["rules"], linestyle="--", linewidth=1.5)
ax.annotate(f"ceiling {curve.accuracy.iloc[-1]:.1%}", (curve.train_size.iloc[-1], curve.accuracy.iloc[-1]),
            textcoords="offset points", xytext=(-10, 8), ha="right", color=INK)
ax.annotate(f"analyst rules {floor:.1%}", (sizes[0], floor), textcoords="offset points", xytext=(0, 6), color=INK)
ax.set_xscale("log"); ax.set_ylim(min(floor, curve.accuracy.min()) - 0.05, 1.0)
ax.set_xlabel("training alerts (reference split)"); ax.set_ylabel("test accuracy")
ax.set_title("Supervised ceiling vs training size", loc="left")
plt.show()
curve
"""),
    code("""
ref_model = baselines.train_reference_model(reference)
ref_probs = baselines.reference_predict(ref_model, test)
p_true = np.array([p[y] for p, y in zip(ref_probs, y_test)])
df["ref_p_true"] = p_true
df["difficulty"] = np.where(p_true >= 0.9, "clear", "ambiguous")

from sklearn.inspection import permutation_importance
X_test, y_idx = baselines.to_xy(test)
imp = permutation_importance(ref_model, X_test, y_idx, n_repeats=10, random_state=0, scoring="accuracy")
imp = pd.Series(imp.importances_mean, index=baselines.FEATURE_NAMES).sort_values().tail(12)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp.index, imp.values, color=MODEL_COLORS["reference_model"], height=0.6)
ax.set_xlabel("accuracy drop when shuffled"); ax.grid(axis="y", visible=False)
ax.set_title("What the supervised ceiling relies on", loc="left")
plt.show()
"""),
    md("""
## 4 · Ambiguous alerts

An alert is **ambiguous** when the supervised model gives its true class less than 0.9 probability. These alerts sit in
the overlap between scenarios. Zero-shot models should be *uncertain* here, not confidently right or wrong.
"""),
    code("""
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(p_true, bins=np.linspace(0, 1, 21), color=MODEL_COLORS["reference_model"], edgecolor="#fcfcfb", linewidth=2)
ax.axvline(0.9, color=INK, linewidth=1, linestyle=":")
ax.set_yscale("log"); ax.set_xlabel("supervised model's probability for the true class"); ax.set_ylabel("alerts (log)")
ax.set_title(f"{(p_true < 0.9).sum()} of {len(p_true)} test alerts are ambiguous", loc="left")
plt.show()
(df.assign(full=df.typology + "/" + df.scenario)
   .groupby("full").agg(n=("difficulty", "size"), ambiguous=("difficulty", lambda s: (s == "ambiguous").sum()),
                        median_p_true=("ref_p_true", "median"))
   .sort_values("ambiguous", ascending=False))
"""),
    md("### Where the analyst rules go wrong"),
    code("""
pd.crosstab(pd.Series(y_test, name="truth"), pd.Series(rules_pred, name="rules"))
"""),
    md("""
## 5 · How much do the PaySim rows contribute?

The supervised model is retrained on one part of the input at a time. If the generated signals alone
reach nearly the full accuracy, the labels are mostly decided by the simulation written here, and PaySim
mainly contributes realistic transaction shapes.
"""),
    code("""
from sklearn.ensemble import HistGradientBoostingClassifier
X_ref, y_ref = baselines.to_xy(reference)
names = baselines.FEATURE_NAMES
txn_cols = [i for i, n in enumerate(names) if n in ("amount_log", "drain_ratio", "orig_balance_log",
            "dest_old_balance_zero", "dest_is_merchant") or n.startswith("type_")]
sig_cols = [i for i in range(len(names)) if i not in txn_cols]
rows = []
for label, cols in [("all fields", list(range(len(names)))), ("generated signals only", sig_cols),
                    ("PaySim transaction only", txn_cols)]:
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0).fit(X_ref[:, cols], y_ref)
    rows.append({"trained on": label, "test accuracy": float(np.mean(m.predict(X_test[:, cols]) == y_idx))})
display(pd.DataFrame(rows))

full_drain = df.drain_ratio.sub(1).abs() < 1e-4
print("share of alerts moving exactly the full balance, by PaySim pool:")
print(full_drain.groupby(df.pool).mean().round(2).to_string())
"""),
]

RESULTS_NB = [
    md("""
# 02 · Results analysis

Deep dive into one benchmark run. Set `RUN_DIR` to a specific folder under `results/live/` to pin it; by default the
most recent run is used. Every chart is computed from `predictions.jsonl`, so it can be checked against `report.md`.
"""),
    code(SETUP + """
from triage.schema import TYPOLOGIES
RUN_DIR = latest_run()
report = json.loads((RUN_DIR / "report.json").read_text())
rows = load_jsonl(RUN_DIR / "predictions.jsonl")
meta = report["meta"]
MODELS = [m for m in ["jev", "gpt-4o-mini", "gpt-4o-mini-reasoning", "reference_model", "rules"] if m in rows[0]]
API = [m for m in MODELS if m.startswith(("jev", "gpt"))]
PROB = [m for m in MODELS if m != "rules"]
print(RUN_DIR.relative_to(ROOT), "| mode:", meta["mode"], "| publishable:", meta["publishable"], "| n:", meta["dataset"]["n"])
if not meta["publishable"]:
    print("WARNING: dry-run or stand-in data. Numbers below are for plumbing only.")
truth = np.array([r["label"]["typology"] for r in rows])
def pred(m): return np.array([r[m]["predicted"] if r[m]["ok"] else "ERROR" for r in rows])
def p_pred(m): return np.array([r[m]["probs"].get(r[m]["predicted"], 0) if r[m]["ok"] else 0 for r in rows])
"""),
    md("## Headline"),
    code("""
print((RUN_DIR / "report.md").read_text().split("## F1 per typology")[0])
"""),
    md("## Confusion matrices (rows = truth)"),
    code("""
fig, axes = plt.subplots(1, len(MODELS), figsize=(4.2 * len(MODELS), 4), constrained_layout=True)
short = ["NORM", "CASH", "ATO", "SYN", "STRUC"]
for ax, m in zip(np.atleast_1d(axes), MODELS):
    cm = pd.crosstab(pd.Categorical(truth, TYPOLOGIES), pd.Categorical(pred(m), TYPOLOGIES + ["ERROR"]), dropna=False)
    cm = cm.loc[TYPOLOGIES, TYPOLOGIES].values
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(5), short); ax.set_yticks(range(5), short); ax.grid(False)
    ax.set_title(MODEL_TITLES[m], loc="left")
plt.show()
"""),
    md("""
## Calibration

Reliability diagram on the predicted class: of alerts predicted with probability ≈ p, how many were right?
Points on the diagonal are well calibrated; points below it are overconfident. Marker size shows how many alerts fall in the bin.
"""),
    code("""
fig, ax = plt.subplots(figsize=(5.6, 5.2))
ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
edges = np.linspace(0, 1, 11)
for m in PROB:
    conf, correct = p_pred(m), (pred(m) == truth)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum():
            xs.append(conf[mask].mean()); ys.append(correct[mask].mean()); ns.append(mask.sum())
    ax.plot(xs, ys, color=MODEL_COLORS[m], marker="o", markersize=0, label=MODEL_TITLES[m])
    ax.scatter(xs, ys, s=[12 + 3 * np.sqrt(n) * 4 for n in ns], color=MODEL_COLORS[m], edgecolor="#fcfcfb", linewidth=2, zorder=3)
ax.set_xlabel("probability of predicted class"); ax.set_ylabel("share correct"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2); ax.set_title("Reliability", loc="left")
plt.show()
pd.DataFrame({MODEL_TITLES[m]: report["models"][m]["calibration"] for m in PROB}).loc[["brier", "log_loss", "ece_top_label"]]
"""),
    md("""
## Settlement: how much can be cleared automatically, and at what cost?

Each curve sweeps the threshold τ on the predicted-class probability. Moving left means settling more alerts
automatically; the y-axis shows how many of those were right. The marked point is τ = 0.85.
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
taus = np.linspace(0.3, 0.999, 120)
for m in PROB:
    conf, correct = p_pred(m), (pred(m) == truth)
    is_fraud_cleared = (pred(m) == "NORMAL") & (truth != "NORMAL")
    cov = [(conf >= t).mean() for t in taus]
    acc = [correct[conf >= t].mean() if (conf >= t).any() else np.nan for t in taus]
    missed = [is_fraud_cleared[conf >= t].sum() for t in taus]
    axes[0].plot(cov, acc, color=MODEL_COLORS[m], label=MODEL_TITLES[m])
    k = np.argmin(np.abs(taus - 0.85))
    axes[0].scatter([cov[k]], [acc[k]], color=MODEL_COLORS[m], s=60, edgecolor="#fcfcfb", linewidth=2, zorder=3)
    axes[1].plot(cov, missed, color=MODEL_COLORS[m], label=MODEL_TITLES[m])
axes[0].set_xlabel("share of alerts settled automatically"); axes[0].set_ylabel("accuracy of settled alerts")
axes[0].set_title("Coverage vs accuracy", loc="left")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=len(PROB))
axes[1].set_xlabel("share of alerts settled automatically"); axes[1].set_ylabel("fraud alerts auto-cleared as NORMAL")
axes[1].set_title("The costly error: fraud waved through", loc="left")
plt.show()
"""),
    md("## Latency"),
    code("""
if API:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for m in API:
        lat = np.sort([r[m]["latency_ms"] for r in rows if r[m]["ok"]])
        axes[0].plot(lat, np.arange(1, len(lat) + 1) / len(lat), color=MODEL_COLORS[m], label=MODEL_TITLES[m])
        p90 = np.percentile(lat, 90)
        axes[0].axvline(p90, color=MODEL_COLORS[m], linewidth=1, linestyle=":")
        axes[0].annotate(f"P90 {p90:.0f} ms", (p90, 0.9), textcoords="offset points", xytext=(4, -12), color=INK)
        t = [r[m]["started_at_s"] for r in rows if r[m]["ok"]]
        l = [r[m]["latency_ms"] for r in rows if r[m]["ok"]]
        axes[1].scatter(t, l, s=10, color=MODEL_COLORS[m], alpha=0.6, label=MODEL_TITLES[m])
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    for a in axes:
        a.set_xscale("log") if a is axes[0] else a.set_yscale("log")
        axis = a.xaxis if a is axes[0] else a.yaxis
        axis.set_major_locator(FixedLocator([200, 300, 500, 1000, 2000, 3000, 5000]))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        axis.set_minor_locator(NullLocator())
    axes[0].set_xlabel("latency (ms, log)"); axes[0].set_ylabel("share of calls ≤ x")
    axes[0].set_title(f"Latency distribution (concurrency {meta['concurrency_per_model']}/model)", loc="left")
    axes[0].legend(loc="lower right")
    axes[1].set_xlabel("seconds since run start"); axes[1].set_ylabel("latency (ms, log)")
    axes[1].set_title("Latency over the run (drift, throttling)", loc="left")
    plt.show()
    display(pd.DataFrame({MODEL_TITLES[m]: report["models"][m]["latency"] for m in API}).T)
"""),
    md("## Accuracy by scenario"),
    code("""
scen = sorted({r["label"]["scenario"] for r in rows})
acc = pd.DataFrame({MODEL_TITLES[m]: [np.mean([(r[m]["ok"] and r[m]["predicted"] == r["label"]["typology"])
                                              for r in rows if r["label"]["scenario"] == s]) for s in scen] for m in MODELS},
                   index=scen)
acc.insert(0, "n", [sum(r["label"]["scenario"] == s for r in rows) for s in scen])
fig, ax = plt.subplots(figsize=(9, 0.38 * len(scen) + 1))
h = 0.8 / len(MODELS)
for i, m in enumerate(MODELS):
    ax.barh(np.arange(len(scen)) + i * h, acc[MODEL_TITLES[m]], height=h - 0.02, color=MODEL_COLORS[m], label=MODEL_TITLES[m])
ax.set_yticks(np.arange(len(scen)) + 0.4 - h / 2, scen); ax.invert_yaxis(); ax.set_xlim(0, 1)
ax.set_xlabel("accuracy"); ax.legend(loc="lower right", ncol=2); ax.grid(axis="y", visible=False)
plt.show()
acc
"""),
    md("## Hold / release gate"),
    code("""
if API:
    fig, axes = plt.subplots(1, len(API), figsize=(5.5 * len(API), 3.4), constrained_layout=True, squeeze=False)
    hold_truth = np.array([r["label"]["hold"] for r in rows])
    for ax, m in zip(axes[0], API):
        hp = np.array([r[m]["hold_prob"] if r[m]["ok"] else np.nan for r in rows])
        bins = np.linspace(0, 1, 21)
        ax.hist(hp[~hold_truth], bins=bins, color="#2a78d6", alpha=0.85, label="legitimate", edgecolor="#fcfcfb")
        ax.hist(hp[hold_truth], bins=bins, color="#eb6834", alpha=0.75, label="fraud / AML", edgecolor="#fcfcfb")
        for x in (0.3, 0.7): ax.axvline(x, color=INK, linewidth=1, linestyle=":")
        ax.set_title(f"{MODEL_TITLES[m]}: p(hold)", loc="left"); ax.set_xlabel("p(hold)"); ax.legend()
    plt.show()
    display(pd.DataFrame({MODEL_TITLES[m]: report["models"][m]["hold_gate"] for m in API}))
"""),
    md("## Where Jev and GPT-4o-mini (best configuration) disagree"),
    code("""
GPT_BEST = "gpt-4o-mini-reasoning" if "gpt-4o-mini-reasoning" in API else "gpt-4o-mini"
if "jev" in API and GPT_BEST in API:
    dis = [{"alert": r["alert_id"], "scenario": r["label"]["scenario"], "truth": r["label"]["typology"],
            "jev": pred("jev")[i], "p_jev": round(p_pred("jev")[i], 2),
            "gpt": pred(GPT_BEST)[i], "p_gpt": round(p_pred(GPT_BEST)[i], 2),
            "ref_p_true": round(r["reference_p_true"], 2) if r["reference_p_true"] is not None else None}
           for i, r in enumerate(rows) if pred("jev")[i] != pred(GPT_BEST)[i]]
    dis = pd.DataFrame(dis)
    print(f"{len(dis)} disagreements; Jev right on {(dis.jev == dis.truth).sum()}, GPT right on {(dis.gpt == dis.truth).sum()}")
    display(dis.head(40))
"""),
    md("## Jev's confident mistakes (p ≥ 0.85)"),
    code("""
if "jev" in API:
    test = {a["alert_id"]: a for a in load_jsonl(ROOT / meta["dataset"]["path"])}
    bad = [r for i, r in enumerate(rows) if pred("jev")[i] != truth[i] and p_pred("jev")[i] >= 0.85]
    out = []
    for r in bad:
        s = test[r["alert_id"]]["signals"]; t = test[r["alert_id"]]["transaction"]
        out.append({"alert": r["alert_id"], "scenario": r["label"]["scenario"], "jev": r["jev"]["predicted"],
                    "p": round(r["jev"]["probs"][r["jev"]["predicted"]], 2), "ref_p_true": r["reference_p_true"],
                    "drain": round(t["amount"] / t["oldbalanceOrg"], 2) if t["oldbalanceOrg"] else None,
                    "known_dev": s["is_known_device"], "risk": s["device_risk_score"],
                    "cred_h": s["hours_since_credential_change"], "fails": s["failed_login_attempts_24h"],
                    "fund_h": s["hours_since_inbound_funding"], "ip": s["ip_location"], "travel": s["travel_notice_active"]})
    display(pd.DataFrame(out))
"""),
]


def build(cells, name: str) -> None:
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
    nbformat.write(nb, HERE / name)
    print(f"[+] executed {name}")


if __name__ == "__main__":
    build(DATASET_NB, "01_dataset_analysis.ipynb")
    build(RESULTS_NB, "02_results_analysis.ipynb")

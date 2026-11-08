"""
Benchmark metrics. Conventions:
- Every alert counts. A failed API call is predicted as "ERROR": a miss for recall and
  F1, zero probability mass on the truth for Brier/ECE, and excluded only from latency
  and cost (which describe successful calls) — the error rate is reported next to them.
- Settlement thresholds use the probability of the predicted class (per the TypeSafe
  docs), not Jev's `confidence`, which measures how concentrated the answer is.
- Model comparisons use a *paired* bootstrap over alerts, so both models are scored on
  the same resampled alerts and the CI on the difference is meaningful.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

from triage.clients import Prediction
from triage.schema import TYPOLOGIES

TAUS = (0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)
HOLD_BAND = (0.30, 0.70)  # release <= 0.30 < review < 0.70 <= hold (TypeSafe docs' starting point)


def _labels(preds: Sequence[Prediction]) -> List[str]:
    return [p.predicted if p.ok else "ERROR" for p in preds]


def _prob_matrix(preds: Sequence[Prediction]) -> np.ndarray:
    return np.array([[p.probs.get(t, 0.0) if p.ok else 0.0 for t in TYPOLOGIES] for p in preds])


def classification(truth: Sequence[str], preds: Sequence[Prediction]) -> Dict[str, Any]:
    y_pred = _labels(preds)
    prec, rec, f1, sup = precision_recall_fscore_support(truth, y_pred, labels=TYPOLOGIES, zero_division=0)
    return {
        "accuracy": float(np.mean([a == b for a, b in zip(truth, y_pred)])),
        "macro_f1": float(f1_score(truth, y_pred, labels=TYPOLOGIES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, y_pred, labels=TYPOLOGIES, average="weighted", zero_division=0)),
        "per_typology": {
            t: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
            for i, t in enumerate(TYPOLOGIES)
        },
        "confusion": {
            t: {u: int(sum(1 for a, b in zip(truth, y_pred) if a == t and b == u)) for u in TYPOLOGIES + ["ERROR"]}
            for t in TYPOLOGIES
        },
    }


def calibration(truth: Sequence[str], preds: Sequence[Prediction], n_bins: int = 10) -> Dict[str, Any]:
    P = _prob_matrix(preds)
    Y = np.array([[float(t == y) for t in TYPOLOGIES] for y in truth])
    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
    p_true = np.array([P[i, TYPOLOGIES.index(y)] for i, y in enumerate(truth)])
    log_loss = float(np.mean(-np.log(np.clip(p_true, 1e-6, 1.0))))
    conf = np.array([p.p_pred for p in preds])
    correct = np.array([p.ok and p.predicted == y for p, y in zip(preds, truth)], dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if mask.any():
            gap = abs(correct[mask].mean() - conf[mask].mean())
            ece += mask.mean() * gap
            bins.append({"bin": f"({lo:.1f},{hi:.1f}]", "count": int(mask.sum()),
                         "mean_p_pred": float(conf[mask].mean()), "accuracy": float(correct[mask].mean())})
    return {"brier": brier, "log_loss": log_loss, "ece_top_label": float(ece), "reliability": bins}


def settlement(truth: Sequence[str], preds: Sequence[Prediction]) -> List[Dict[str, Any]]:
    """Alerts whose predicted-class probability >= tau are resolved without an analyst."""
    n = len(truth)
    rows = []
    for tau in TAUS:
        idx = [i for i, p in enumerate(preds) if p.ok and p.p_pred >= tau]
        correct = sum(preds[i].predicted == truth[i] for i in idx)
        fraud_auto_cleared = sum(preds[i].predicted == "NORMAL" and truth[i] != "NORMAL" for i in idx)
        rows.append({
            "tau": tau,
            "settled": len(idx),
            "coverage": len(idx) / n,
            "accuracy": correct / len(idx) if idx else float("nan"),
            "errors": len(idx) - correct,
            "fraud_auto_cleared": fraud_auto_cleared,
        })
    return rows


def hold_gate(truth_hold: Sequence[bool], preds: Sequence[Prediction]) -> Dict[str, Any]:
    y = np.array(truth_hold, dtype=int)
    ok = np.array([p.ok for p in preds])
    p = np.array([p.hold_prob if p.ok else np.nan for p in preds])
    p_scored = np.where(ok, p, 1.0 - y)  # a failed call counts as maximally wrong
    lo, hi = HOLD_BAND
    release = ok & (p <= lo)
    hold = ok & (p >= hi)
    review = ~(release | hold)  # includes failed calls: they would go to an analyst
    return {
        "auc": float(roc_auc_score(y, np.nan_to_num(p_scored))) if 0 < y.sum() < len(y) else float("nan"),
        "brier": float(np.mean((p_scored - y) ** 2)),
        "accuracy_at_0_5": float(np.mean(ok & ((p >= 0.5) == y.astype(bool)))),
        "band": {"release_le": lo, "hold_ge": hi},
        "released": int(release.sum()),
        "held": int(hold.sum()),
        "sent_to_review": int(review.sum()),
        "fraud_released": int((release & (y == 1)).sum()),
        "legit_held": int((hold & (y == 0)).sum()),
    }


def latency(preds: Sequence[Prediction], seed: int = 0) -> Dict[str, Any]:
    ok = [p for p in preds if p.ok]
    if not ok:
        return {}
    lat = np.array([p.latency_ms for p in ok])
    e2e = np.array([p.e2e_latency_ms for p in ok])
    rng = np.random.default_rng(seed)
    boot_p90 = [np.percentile(rng.choice(lat, size=len(lat)), 90) for _ in range(2000)]
    return {
        "n": len(ok),
        "p50_ms": float(np.percentile(lat, 50)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p90_ci95_ms": [float(np.percentile(boot_p90, 2.5)), float(np.percentile(boot_p90, 97.5))],
        "p99_ms": float(np.percentile(lat, 99)),
        "max_ms": float(lat.max()),
        "mean_ms": float(lat.mean()),
        "e2e_p90_ms": float(np.percentile(e2e, 90)),
        "retried_calls": int(sum(p.attempts > 1 for p in ok)),
    }


def cost(preds: Sequence[Prediction]) -> Dict[str, Any]:
    ok = [p for p in preds if p.ok]
    if not ok:
        return {}
    return {
        "mean_input_tokens": float(np.mean([p.input_tokens for p in ok])),
        "mean_output_tokens": float(np.mean([p.output_tokens for p in ok])),
        "usd_per_1k_alerts": float(np.mean([p.cost_usd for p in ok]) * 1000),
        "usd_total": float(sum(p.cost_usd for p in preds)),
    }


def paired_bootstrap(
    truth: Sequence[str],
    preds_by_model: Dict[str, Sequence[Prediction]],
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """95% CIs for each model's macro F1 / accuracy and for every pairwise difference."""
    rng = np.random.default_rng(seed)
    truth_arr = np.array(truth)
    labels = {m: np.array(_labels(p)) for m, p in preds_by_model.items()}
    n = len(truth_arr)
    samples: Dict[str, Dict[str, List[float]]] = {m: {"macro_f1": [], "accuracy": []} for m in labels}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        t = truth_arr[idx]
        for m, y in labels.items():
            samples[m]["macro_f1"].append(f1_score(t, y[idx], labels=TYPOLOGIES, average="macro", zero_division=0))
            samples[m]["accuracy"].append(float(np.mean(t == y[idx])))

    def ci(v: Sequence[float]) -> List[float]:
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out: Dict[str, Any] = {"per_model": {}, "differences": {}}
    for m, s in samples.items():
        out["per_model"][m] = {k: ci(v) for k, v in s.items()}
    names = list(labels)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = {}
            for k in ("macro_f1", "accuracy"):
                d = np.array(samples[a][k]) - np.array(samples[b][k])
                diff[k] = {"ci95": ci(d), "p_a_better": float(np.mean(d > 0))}
            out["differences"][f"{a} - {b}"] = diff
    return out


def cohort_accuracy(
    truth: Sequence[str], preds: Sequence[Prediction], cohort: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for c in sorted(set(cohort)):
        idx = [i for i, x in enumerate(cohort) if x == c]
        out[c] = {
            "n": len(idx),
            "accuracy": float(np.mean([preds[i].ok and preds[i].predicted == truth[i] for i in idx])),
        }
    return out


def model_report(
    alerts: List[Dict[str, Any]],
    preds: Sequence[Prediction],
    difficulty: Optional[Sequence[str]] = None,
    has_probabilities: bool = True,
    has_hold: bool = True,
    is_api: bool = True,
) -> Dict[str, Any]:
    truth = [a["label"]["typology"] for a in alerts]
    report: Dict[str, Any] = {
        "n": len(preds),
        "errors": int(sum(not p.ok for p in preds)),
        "classification": classification(truth, preds),
        "by_scenario": cohort_accuracy(truth, preds, [a["label"]["scenario"] for a in alerts]),
    }
    if difficulty is not None:
        report["by_difficulty"] = cohort_accuracy(truth, preds, difficulty)
    if has_probabilities:
        report["calibration"] = calibration(truth, preds)
        report["settlement"] = settlement(truth, preds)
    if has_hold:
        report["hold_gate"] = hold_gate([a["label"]["hold"] for a in alerts], preds)
    if is_api:
        report["latency"] = latency(preds)
        report["cost"] = cost(preds)
        sources = sorted({p.prob_source for p in preds if p.ok})
        report["probability_source"] = sources
        cov = [p.prob_coverage for p in preds if p.ok and not np.isnan(p.prob_coverage)]
        if cov:
            report["logprob_mass_mapped_mean"] = float(np.mean(cov))
    return report

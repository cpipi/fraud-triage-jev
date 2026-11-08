"""
evaluate.py
-----------
Runs the fraud-triage benchmark: TypeSafe Jev vs GPT-4o-mini, framed by a supervised
reference model (ceiling) and an analyst rules engine (floor).

    uv run evaluate.py --mode dry-run                       # plumbing check, no API calls
    uv run evaluate.py --mode live --split dev --limit 10   # smoke test
    uv run evaluate.py --mode live --concurrency 8          # full test split
    uv run evaluate.py --mode live --concurrency 1 --limit 150 --run-name sequential

Outputs go to results/live/<run-name>/ (or results/dry_run/... which is git-ignored):
report.md, report.json and predictions.jsonl, all stamped with the dataset hash,
criteria hash, git commit, pinned and served model versions, and concurrency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
import numpy as np

from generate_dataset import file_sha256
from triage import baselines
from triage.clients import (
    GPT_MODEL,
    JEV_MODEL,
    JevClient,
    OpenAIClient,
    Prediction,
    SimulatedClient,
)
from triage.metrics import model_report, paired_bootstrap
from triage.report import render_markdown
from triage.schema import TYPOLOGIES, criteria_sha256

AMBIGUOUS_BELOW = 0.90  # reference-model probability of the true class


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    except ImportError:
        pass


def read_key(*names: str) -> str:
    for n in names:
        raw = (os.getenv(n) or "").strip().strip('"').strip("'")
        if raw.lower().startswith("bearer "):
            raw = raw[7:].strip()
        if raw:
            return raw
    return ""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def git_info() -> Dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def reference_predictions(
    alerts: List[Dict[str, Any]], reference_path: Path
) -> Optional[List[Prediction]]:
    if not reference_path.exists():
        print(f"[!] {reference_path} missing: skipping the supervised reference model.")
        return None
    print(f"[*] Training supervised reference model on {reference_path} ...")
    model = baselines.train_reference_model(load_jsonl(reference_path))
    out = []
    for a, probs in zip(alerts, baselines.reference_predict(model, alerts)):
        predicted = max(probs, key=probs.get)
        out.append(Prediction(
            model="reference_model", alert_id=a["alert_id"], ok=True, predicted=predicted, probs=probs,
            hold_prob=1.0 - probs["NORMAL"], prob_source="supervised",
        ))
    return out


def rules_predictions(alerts: List[Dict[str, Any]]) -> List[Prediction]:
    out = []
    for a in alerts:
        label = baselines.rules_predict(a)
        out.append(Prediction(
            model="rules", alert_id=a["alert_id"], ok=True, predicted=label,
            probs={t: float(t == label) for t in TYPOLOGIES}, hold_prob=float(label != "NORMAL"),
            prob_source="hard_label",
        ))
    return out


async def run_api_models(
    clients: Sequence[Any], alerts: List[Dict[str, Any]], concurrency: int, seed: int
) -> Dict[str, List[Prediction]]:
    limits = httpx.Limits(max_connections=concurrency * len(clients) + 4, max_keepalive_connections=concurrency * len(clients))
    sems = {c.name: asyncio.Semaphore(concurrency) for c in clients}
    results: Dict[str, Dict[str, Prediction]] = {c.name: {} for c in clients}
    t_start = time.perf_counter()

    async with httpx.AsyncClient(limits=limits, http2=False) as http:
        # Symmetric warm-up: one unrecorded call per model to open the connection pool.
        await asyncio.gather(*(c.predict(http, alerts[0]) for c in clients))

        async def one(client: Any, alert: Dict[str, Any]) -> None:
            async with sems[client.name]:
                started = time.perf_counter() - t_start
                pred = await client.predict(http, alert)
                pred.started_at_s = started
                results[client.name][alert["alert_id"]] = pred

        jobs = [(c, a) for a in alerts for c in clients]
        order = np.random.default_rng(seed).permutation(len(jobs))  # interleave models in time
        done = 0

        async def tracked(c: Any, a: Dict[str, Any]) -> None:
            nonlocal done
            await one(c, a)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"    {done}/{len(jobs)} calls done ({time.perf_counter() - t_start:.1f}s)")

        await asyncio.gather(*(tracked(*jobs[i]) for i in order))

    return {name: [r[a["alert_id"]] for a in alerts] for name, r in results.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Jev vs GPT-4o-mini fraud triage benchmark")
    p.add_argument("--mode", choices=["dry-run", "live"], default="dry-run")
    p.add_argument("--split", choices=["test", "dev"], default="test")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Default: data/ if the PaySim-seeded split exists, else data/stand_in/")
    p.add_argument("--models", default="jev,gpt-4o-mini", help="Comma list of: jev, gpt-4o-mini")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=8, help="Max in-flight requests per model")
    p.add_argument("--run-name", default=None)
    p.add_argument("--client-region", default=os.getenv("CLIENT_REGION", "unspecified"),
                   help="Where the benchmark client runs, e.g. 'Almaty, KZ (home fiber)'")
    p.add_argument("--allow-stand-in", action="store_true",
                   help="Permit live calls on synthetic stand-in data (never publish such results)")
    args = p.parse_args()
    load_env()

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = Path("data") if Path(f"data/alerts_{args.split}.jsonl").exists() else Path("data/stand_in")
    data_path = data_dir / f"alerts_{args.split}.jsonl"
    if not data_path.exists():
        sys.exit(f"{data_path} not found. Run generate_dataset.py --split {args.split} first.")
    alerts = load_jsonl(data_path)
    if args.limit:
        alerts = alerts[: args.limit]
    sources = sorted({a["provenance"]["source"] for a in alerts})
    stand_in = "synthetic_stand_in" in sources
    if args.mode == "live" and stand_in and not args.allow_stand_in:
        sys.exit("Refusing live run on stand-in data. Generate PaySim-seeded data or pass --allow-stand-in.")

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    clients: List[Any] = []
    if args.mode == "live":
        if "jev" in wanted:
            key = read_key("TYPESAFE_API_KEY", "JEV_API_KEY")
            if not key:
                sys.exit("TYPESAFE_API_KEY is not set (add it to .env).")
            clients.append(JevClient(key))
        if "gpt-4o-mini" in wanted:
            key = read_key("OPENAI_API_KEY")
            if not key:
                sys.exit("OPENAI_API_KEY is not set. Add it, or pass --models jev for a Jev-only run.")
            clients.append(OpenAIClient(key))
    else:
        if "jev" in wanted:
            clients.append(SimulatedClient("jev", 300, 0.90, seed=1))
        if "gpt-4o-mini" in wanted:
            clients.append(SimulatedClient("gpt-4o-mini", 700, 0.85, seed=2))

    # Reference points (no API calls)
    ref = reference_predictions(alerts, data_dir / "alerts_reference.jsonl")
    difficulty: Optional[List[str]] = None
    ref_p_true: List[Optional[float]] = [None] * len(alerts)
    if ref is not None:
        ref_p_true = [r.probs[a["label"]["typology"]] for r, a in zip(ref, alerts)]
        difficulty = ["clear" if p >= AMBIGUOUS_BELOW else "ambiguous" for p in ref_p_true]
    rules = rules_predictions(alerts)

    print(f"[*] {len(alerts)} alerts from {data_path} ({', '.join(sources)}); "
          f"models={[c.name for c in clients]}; mode={args.mode}; concurrency={args.concurrency}")
    started = datetime.now(timezone.utc)
    api_preds = asyncio.run(run_api_models(clients, alerts, args.concurrency, seed=42)) if clients else {}
    wall_s = (datetime.now(timezone.utc) - started).total_seconds()

    reports: Dict[str, Any] = {}
    for name, preds in api_preds.items():
        reports[name] = model_report(alerts, preds, difficulty)
    if ref is not None:
        reports["reference_model"] = model_report(alerts, ref, difficulty, is_api=False)
    reports["rules"] = model_report(alerts, rules, difficulty, has_probabilities=False, is_api=False)

    all_preds = {**api_preds, "rules": rules, **({"reference_model": ref} if ref else {})}
    truth = [a["label"]["typology"] for a in alerts]
    bootstrap = paired_bootstrap(truth, all_preds)

    served = {n: sorted({p.served_model for p in ps if p.ok}) for n, ps in api_preds.items()}
    meta = {
        "mode": args.mode,
        "publishable": args.mode == "live" and not stand_in,
        "timestamp_utc": started.isoformat(timespec="seconds"),
        "wall_clock_s": round(wall_s, 1),
        "git": git_info(),
        "dataset": {"path": str(data_path), "sha256": file_sha256(data_path), "n": len(alerts),
                    "limit": args.limit, "sources": sources},
        "criteria_sha256": criteria_sha256(),
        "models_requested": {"jev": JEV_MODEL, "gpt-4o-mini": GPT_MODEL},
        "models_served": served,
        "concurrency_per_model": args.concurrency,
        "client_region": args.client_region,
        "python": platform.python_version(),
        "argv": sys.argv[1:],
        "ambiguous_definition": f"reference-model probability of the true class < {AMBIGUOUS_BELOW}",
    }

    run_name = args.run_name or started.strftime("%Y%m%dT%H%M%SZ") + f"_{args.split}"
    out_dir = Path("results") / ("live" if args.mode == "live" else "dry_run") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i, a in enumerate(alerts):
            row = {
                "alert_id": a["alert_id"],
                "label": a["label"],
                "reference_p_true": ref_p_true[i],
                "difficulty": difficulty[i] if difficulty else None,
                **{name: asdict(preds[i]) for name, preds in all_preds.items()},
            }
            f.write(json.dumps(row) + "\n")
    payload = {"meta": meta, "models": reports, "paired_bootstrap": bootstrap}
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = render_markdown(payload)
    (out_dir / "report.md").write_text(md, encoding="utf-8")

    print("\n" + md)
    print(f"\n[+] Wrote {out_dir}/report.md, report.json, predictions.jsonl")
    for name, preds in api_preds.items():
        errs = [p for p in preds if not p.ok]
        if errs:
            print(f"[!] {name}: {len(errs)} failed calls. First: {errs[0].error}")


if __name__ == "__main__":
    main()

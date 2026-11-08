"""
generate_dataset.py
-------------------
Builds reproducible fraud-triage alerts seeded from the public PaySim dataset
(Lopez-Rojas et al., 2016; Kaggle `ealaxi/paysim1`, CC BY-SA 4.0).

How an alert is made
1. A ground-truth typology and a scenario within it are drawn (e.g. NORMAL/traveler,
   ACCOUNT_TAKEOVER/domestic_proxy). The label is the scenario that generated the
   alert. It is never derived from rules applied to the features.
2. The `transaction` block is a real PaySim row, picked from a pool chosen by a
   published rule (see `PAYSIM_POOLS`). PaySim's own `isFraud` / `isFlaggedFraud`
   columns are dropped from the model input.
3. The `signals` block (session, device, identity, 24h velocity) is drawn from
   scenario-specific distributions that deliberately overlap: real customers travel,
   change phones and make large payments; fraudsters sometimes look ordinary. Some
   alerts are therefore truly ambiguous, and `baselines.py` estimates how many.

Splits
- `test`: the benchmark set reported in results (default 500 alerts, seed 42).
- `dev`: for wording criteria and smoke tests (default 100 alerts, seed 7).
- `reference`: a large set (default 10,000) used only to train the supervised
  reference model that estimates the accuracy ceiling. Never sent to an API.
Each split draws PaySim rows from a disjoint slice of every pool, so no transaction
appears in two splits.

Without the PaySim CSV, `--synthetic-core` builds a clearly labelled stand-in
transaction block for plumbing tests. Those alerts carry `source: "synthetic_stand_in"`
and must not be published as results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from triage.schema import TYPOLOGIES

DEFAULT_PAYSIM_CSV = Path("data/raw/PS_20174392719_1491204439457_log.csv")
POOL_SHUFFLE_SEED = 2017  # fixed so that split slices never change between runs

SPLIT_DEFAULTS: Dict[str, Tuple[int, int]] = {  # split -> (samples, seed)
    "test": (500, 42),
    "dev": (100, 7),
    "reference": (10_000, 1234),
}
SPLIT_SLICES: Dict[str, Tuple[float, float]] = {  # fraction of each shuffled pool
    "test": (0.00, 0.20),
    "dev": (0.20, 0.30),
    "reference": (0.30, 1.00),
}

CLASS_MIX: Dict[str, float] = {
    "NORMAL": 0.40,
    "MONEY_MULE": 0.15,
    "ACCOUNT_TAKEOVER": 0.15,
    "SYNTHETIC_ID": 0.15,
    "STRUCTURING_AML": 0.15,
}

# scenario -> (probability within its typology, PaySim pool; "a|b" picks one of them 50/50)
SCENARIOS: Dict[str, Dict[str, Tuple[float, str]]] = {
    "NORMAL": {
        "routine": (0.55, "normal_any"),
        "traveler": (0.10, "normal_any"),
        "new_device": (0.08, "normal_any"),
        "large_payment": (0.08, "normal_large"),
        "new_customer": (0.07, "normal_any"),
        # Look-alikes: legitimate activity that shares most signals with a fraud scenario.
        "new_device_large_payment": (0.05, "normal_large"),  # ~ ACCOUNT_TAKEOVER/domestic_proxy
        "traveler_no_notice_large": (0.03, "normal_large"),  # ~ ACCOUNT_TAKEOVER/session_hijack
        "payroll_batch": (0.04, "near_threshold"),  # ~ STRUCTURING_AML/light
    },
    "MONEY_MULE": {
        "mule_known_session": (0.80, "fraud"),
        "mule_new_phone": (0.20, "fraud"),  # ~ ACCOUNT_TAKEOVER/domestic_proxy
    },
    "ACCOUNT_TAKEOVER": {
        "foreign_login": (0.45, "fraud"),
        "domestic_proxy": (0.35, "fraud|normal_large"),  # takeovers do not always empty the account
        "session_hijack": (0.20, "fraud|normal_large"),  # stolen session cookie: device looks known
    },
    "SYNTHETIC_ID": {
        "bust_out": (0.70, "fraud"),
        "sleeper": (0.30, "normal_large"),  # partial drain, weaker identity signals
    },
    "STRUCTURING_AML": {
        "classic": (0.65, "near_threshold"),
        "light": (0.35, "near_threshold"),
    },
}
# Note: PaySim supplies transaction *shapes*. `isFraud` only decides which pool a scenario
# draws from; the label is always the scenario. E.g. structuring draws non-fraud PaySim
# rows in the $8k-$10k band, because PaySim does not simulate structuring.

NEAR_THRESHOLD_LOW = 8_000.0
NEAR_THRESHOLD_HIGH = 10_000.0  # FinCEN CTR threshold (exclusive)

PAYSIM_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
]
TRANSACTION_KEYS = PAYSIM_COLUMNS[:-1]  # isFraud is never shown to a model

SIGNAL_KEYS: Tuple[str, ...] = (
    "account_age_days",
    "credit_history_months",
    "credit_line_utilization_pct",
    "ssn_verification_status",
    "shared_kyc_attributes_count",
    "is_known_device",
    "device_risk_score",
    "home_location",
    "ip_location",
    "travel_notice_active",
    "hours_since_credential_change",
    "failed_login_attempts_24h",
    "hours_since_inbound_funding",
    "recipient_is_first_time",
    "transfers_8k_to_10k_24h",
    "total_8k_to_10k_usd_24h",
    "distinct_counterparties_24h",
    "tx_count_1h",
    "tx_count_24h",
)

US_LOCATIONS = [
    "New York, NY", "Austin, TX", "Chicago, IL", "Seattle, WA",
    "Miami, FL", "Charlotte, NC", "Denver, CO", "Phoenix, AZ",
]
FOREIGN_LOCATIONS = [
    "London, UK", "Frankfurt, DE", "Lagos, NG", "Singapore, SG",
    "Bucharest, RO", "Mexico City, MX", "Limassol, CY",
]


# --------------------------------------------------------------------------------------
# PaySim pools
# --------------------------------------------------------------------------------------

PAYSIM_POOLS: Dict[str, str] = {
    "fraud": "isFraud == 1 and oldbalanceOrg > 0 (PaySim's account-emptying fraud rows)",
    "normal_any": "isFraud == 0, uniform sample of 400k rows, all transaction types",
    "normal_large": (
        "isFraud == 0, type in {TRANSFER, CASH_OUT, PAYMENT}, oldbalanceOrg > 0, "
        "0.5 <= amount / oldbalanceOrg <= 1.0"
    ),
    "near_threshold": (
        "isFraud == 0, type in {TRANSFER, CASH_OUT}, 8,000 <= amount < 10,000, "
        "oldbalanceOrg >= amount"
    ),
}


class TransactionSource:
    """Hands out transaction rows for one split, without replacement."""

    def take(self, pool: str, rng: np.random.Generator) -> Tuple[Dict[str, Any], Optional[int]]:
        raise NotImplementedError

    name: str = "abstract"


class PaySimSource(TransactionSource):
    name = "paysim"

    def __init__(self, csv_path: Path, split: str):
        import pandas as pd

        print(f"[*] Loading PaySim from {csv_path} (this takes ~20s)...")
        df = pd.read_csv(
            csv_path,
            usecols=PAYSIM_COLUMNS,
            dtype={"type": "category", "nameOrig": "string", "nameDest": "string"},
        )
        df["row_index"] = np.arange(len(df))
        ratio = df["amount"] / df["oldbalanceOrg"].where(df["oldbalanceOrg"] > 0)
        non_fraud = df[df["isFraud"] == 0]
        pools = {
            "fraud": df[(df["isFraud"] == 1) & (df["oldbalanceOrg"] > 0)],
            "normal_any": non_fraud.sample(
                n=min(400_000, len(non_fraud)), random_state=POOL_SHUFFLE_SEED
            ),
            "normal_large": non_fraud[
                non_fraud["type"].isin(["TRANSFER", "CASH_OUT", "PAYMENT"])
                & ratio.loc[non_fraud.index].between(0.5, 1.0)
            ],
            "near_threshold": non_fraud[
                non_fraud["type"].isin(["TRANSFER", "CASH_OUT"])
                & (non_fraud["amount"] >= NEAR_THRESHOLD_LOW)
                & (non_fraud["amount"] < NEAR_THRESHOLD_HIGH)
                & (non_fraud["oldbalanceOrg"] >= non_fraud["amount"])
            ],
        }
        lo, hi = SPLIT_SLICES[split]
        self._rows: Dict[str, List[Dict[str, Any]]] = {}
        self._cursor: Dict[str, int] = {}
        for name, pool in pools.items():
            shuffled = pool.sample(frac=1.0, random_state=POOL_SHUFFLE_SEED)
            a, b = int(len(shuffled) * lo), int(len(shuffled) * hi)
            self._rows[name] = shuffled.iloc[a:b].to_dict(orient="records")
            self._cursor[name] = 0
            print(f"    pool {name:<15} {len(pool):>9,} rows, split '{split}' slice {b - a:>8,}")

    def take(self, pool: str, rng: np.random.Generator) -> Tuple[Dict[str, Any], Optional[int]]:
        rows = self._rows[pool]
        i = self._cursor[pool]
        if i >= len(rows):
            raise RuntimeError(f"PaySim pool '{pool}' exhausted for this split ({len(rows)} rows).")
        self._cursor[pool] = i + 1
        row = rows[i]
        txn = {k: row[k] for k in TRANSACTION_KEYS}
        txn["step"] = int(txn["step"])
        txn["type"] = str(txn["type"])
        txn["nameOrig"] = str(txn["nameOrig"])
        txn["nameDest"] = str(txn["nameDest"])
        for k in ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
            txn[k] = round(float(txn[k]), 2)
        return txn, int(row["row_index"])


class SyntheticStandInSource(TransactionSource):
    """Plumbing-test stand-in that mimics PaySim's columns. Not for published results."""

    name = "synthetic_stand_in"

    def take(self, pool: str, rng: np.random.Generator) -> Tuple[Dict[str, Any], Optional[int]]:
        dest_prefix = "C"
        if pool == "fraud":
            tx_type = str(rng.choice(["TRANSFER", "CASH_OUT"]))
            old_org = float(rng.lognormal(np.log(150_000), 1.0))
            amount = old_org
            old_dest = 0.0 if rng.random() < 0.6 else float(rng.lognormal(np.log(50_000), 1.2))
            new_dest = old_dest if tx_type == "TRANSFER" else old_dest + amount
        elif pool == "near_threshold":
            tx_type = str(rng.choice(["TRANSFER", "CASH_OUT"]))
            amount = float(rng.uniform(NEAR_THRESHOLD_LOW, NEAR_THRESHOLD_HIGH - 1))
            old_org = amount + float(rng.lognormal(np.log(40_000), 1.0))
            old_dest = float(rng.lognormal(np.log(200_000), 1.0))
            new_dest = old_dest + amount
        elif pool == "normal_large":
            tx_type = str(rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"]))
            old_org = float(rng.lognormal(np.log(40_000), 0.8))
            amount = old_org * float(rng.uniform(0.5, 1.0))
            old_dest = float(rng.lognormal(np.log(150_000), 1.0))
            new_dest = old_dest + amount
        else:
            tx_type = str(rng.choice(["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"],
                                     p=[0.34, 0.22, 0.35, 0.08, 0.01]))
            old_org = float(rng.lognormal(np.log(20_000), 1.5)) if rng.random() < 0.7 else 0.0
            amount = float(rng.lognormal(np.log(30_000), 1.3))
            if tx_type == "PAYMENT":
                amount = float(rng.lognormal(np.log(8_000), 1.0))
                dest_prefix = "M"
            old_dest = 0.0 if dest_prefix == "M" else float(rng.lognormal(np.log(300_000), 1.2))
            new_dest = old_dest if dest_prefix == "M" else old_dest + amount
        new_org = old_org + amount if tx_type == "CASH_IN" else max(0.0, old_org - amount)
        txn = {
            "step": int(rng.integers(1, 744)),
            "type": tx_type,
            "amount": round(amount, 2),
            "nameOrig": f"C{rng.integers(10**8, 10**10)}",
            "oldbalanceOrg": round(old_org, 2),
            "newbalanceOrig": round(new_org, 2),
            "nameDest": f"{dest_prefix}{rng.integers(10**8, 10**10)}",
            "oldbalanceDest": round(old_dest, 2),
            "newbalanceDest": round(new_dest, 2),
        }
        return txn, None


# --------------------------------------------------------------------------------------
# Signal generation
# --------------------------------------------------------------------------------------


def _clip_int(x: float, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, round(x))))


def _hours(rng: np.random.Generator, median: float, sigma: float, hi: float = 20_000.0) -> float:
    return round(float(min(hi, max(0.1, rng.lognormal(np.log(median), sigma)))), 1)


def _established_profile(rng: np.random.Generator, home: str) -> Dict[str, Any]:
    """An ordinary established customer. Scenarios override parts of this."""
    age = _clip_int(rng.lognormal(np.log(1200), 0.6), 120, 6000)
    return {
        "account_age_days": age,
        "credit_history_months": _clip_int(age / 30 + rng.normal(24, 18), 6, 400),
        "credit_line_utilization_pct": round(float(rng.beta(2, 5) * 100), 1),
        "ssn_verification_status": "MATCH" if rng.random() < 0.97 else "THIN_FILE",
        "shared_kyc_attributes_count": int(rng.random() < 0.05),  # e.g. family sharing an address
        "is_known_device": bool(rng.random() < 0.95),
        "device_risk_score": round(float(rng.beta(1.5, 12)), 2),
        "home_location": home,
        "ip_location": home if rng.random() < 0.97 else str(rng.choice(US_LOCATIONS)),
        "travel_notice_active": False,
        "hours_since_credential_change": _hours(rng, 2000, 1.0),
        "failed_login_attempts_24h": int(rng.poisson(0.15)),
        "hours_since_inbound_funding": _hours(rng, 120, 1.0, hi=2000),
        "recipient_is_first_time": bool(rng.random() < 0.20),
        "transfers_8k_to_10k_24h": 0,
        "total_8k_to_10k_usd_24h": 0.0,
        "distinct_counterparties_24h": 1 + int(rng.poisson(0.6)),
        "tx_count_1h": 1 + int(rng.poisson(0.3)),
    }


def _hijacked_session(rng: np.random.Generator, s: Dict[str, Any], device_beta: Tuple[float, float]) -> None:
    s["is_known_device"] = bool(rng.random() < 0.08)
    s["device_risk_score"] = round(float(rng.beta(*device_beta)), 2)
    if rng.random() < 0.75:
        s["hours_since_credential_change"] = _hours(rng, 8, 1.0)
    s["failed_login_attempts_24h"] = int(rng.poisson(2.5))
    s["recipient_is_first_time"] = bool(rng.random() < 0.95)
    s["tx_count_1h"] = 1 + int(rng.poisson(1.0))


def _quiet_new_session(rng: np.random.Generator, s: Dict[str, Any]) -> None:
    """New device and a recent credential reset with no other alarms. Shared on purpose by a
    legitimate phone upgrade, a mule on a new phone and a careful account takeover."""
    s["is_known_device"] = False
    s["hours_since_credential_change"] = _hours(rng, 24, 1.0)
    s["device_risk_score"] = round(float(rng.beta(3, 6)), 2)


def _near_threshold_batch(
    rng: np.random.Generator, s: Dict[str, Any], txn: Dict[str, Any], n_prior: int
) -> None:
    prior = rng.uniform(NEAR_THRESHOLD_LOW, NEAR_THRESHOLD_HIGH - 1, size=n_prior)
    s["transfers_8k_to_10k_24h"] = n_prior + 1
    s["total_8k_to_10k_usd_24h"] = round(float(prior.sum() + txn["amount"]), 2)


def apply_scenario(
    rng: np.random.Generator, typology: str, scenario: str, s: Dict[str, Any], txn: Dict[str, Any]
) -> None:
    home = s["home_location"]
    if typology == "NORMAL":
        if scenario == "traveler":
            s["ip_location"] = str(rng.choice(FOREIGN_LOCATIONS))
            s["travel_notice_active"] = bool(rng.random() < 0.80)  # some forget to file one
            s["device_risk_score"] = round(float(rng.beta(2, 10)), 2)
        elif scenario == "new_device":
            s["is_known_device"] = False
            s["hours_since_credential_change"] = _hours(rng, 30, 0.8)
            s["failed_login_attempts_24h"] = int(rng.poisson(0.8))
            s["device_risk_score"] = round(float(rng.beta(2.5, 8)), 2)
            s["ip_location"] = home if rng.random() < 0.9 else str(rng.choice(US_LOCATIONS))
        elif scenario == "large_payment":
            s["recipient_is_first_time"] = bool(rng.random() < 0.60)
            s["hours_since_inbound_funding"] = _hours(rng, 48, 1.0, hi=2000)
        elif scenario == "new_customer":
            s["account_age_days"] = int(rng.integers(20, 200))
            s["credit_history_months"] = int(rng.integers(0, 25))
            s["ssn_verification_status"] = "MATCH" if rng.random() < 0.75 else "THIN_FILE"
            s["shared_kyc_attributes_count"] = int(rng.random() < 0.10)
            s["credit_line_utilization_pct"] = round(float(rng.beta(2, 4) * 100), 1)
        elif scenario == "new_device_large_payment":
            _quiet_new_session(rng, s)
            s["ip_location"] = home if rng.random() < 0.6 else str(rng.choice(US_LOCATIONS))
            s["failed_login_attempts_24h"] = int(rng.poisson(0.8))
            s["recipient_is_first_time"] = bool(rng.random() < 0.80)
        elif scenario == "traveler_no_notice_large":
            s["ip_location"] = str(rng.choice(FOREIGN_LOCATIONS))
            s["device_risk_score"] = round(float(rng.beta(2.5, 7)), 2)
            s["recipient_is_first_time"] = bool(rng.random() < 0.75)
        elif scenario == "payroll_batch":
            _near_threshold_batch(rng, s, txn, n_prior=1 + int(rng.random() < 0.4))
            s["distinct_counterparties_24h"] = s["transfers_8k_to_10k_24h"]
            s["recipient_is_first_time"] = bool(rng.random() < 0.15)

    elif typology == "MONEY_MULE":
        s["hours_since_inbound_funding"] = _hours(rng, 3, 0.8, hi=2000)
        s["recipient_is_first_time"] = bool(rng.random() < 0.90)
        s["tx_count_1h"] = 1 + int(rng.poisson(1.5))
        s["distinct_counterparties_24h"] = 1 + int(rng.poisson(1.0))
        s["account_age_days"] = _clip_int(rng.lognormal(np.log(400), 0.8), 30, 6000)
        s["credit_history_months"] = _clip_int(s["account_age_days"] / 30 + rng.normal(12, 12), 1, 400)
        s["is_known_device"] = True
        s["device_risk_score"] = round(float(rng.beta(2, 8)), 2)
        if scenario == "mule_new_phone":
            _quiet_new_session(rng, s)
            s["failed_login_attempts_24h"] = int(rng.poisson(0.5))

    elif typology == "ACCOUNT_TAKEOVER":
        if scenario == "foreign_login":
            _hijacked_session(rng, s, (5, 3))
            s["ip_location"] = str(rng.choice(FOREIGN_LOCATIONS))
            s["travel_notice_active"] = bool(rng.random() < 0.05)  # coincidental real trip
        elif scenario == "domestic_proxy":
            _quiet_new_session(rng, s)
            s["ip_location"] = home if rng.random() < 0.4 else str(rng.choice(US_LOCATIONS))
            s["failed_login_attempts_24h"] = int(rng.poisson(1.6))
            s["recipient_is_first_time"] = bool(rng.random() < 0.90)
        else:  # session_hijack: stolen cookie, so the device fingerprint matches the owner's
            s["ip_location"] = (
                str(rng.choice(FOREIGN_LOCATIONS)) if rng.random() < 0.6 else str(rng.choice(US_LOCATIONS))
            )
            s["device_risk_score"] = round(float(rng.beta(3, 6)), 2)
            s["failed_login_attempts_24h"] = int(rng.poisson(0.3))
            s["recipient_is_first_time"] = bool(rng.random() < 0.90)
            s["tx_count_1h"] = 1 + int(rng.poisson(0.8))

    elif typology == "SYNTHETIC_ID":
        s["is_known_device"] = True  # the fraudster's own device, used since onboarding
        s["recipient_is_first_time"] = bool(rng.random() < 0.85)
        if scenario == "bust_out":
            s["account_age_days"] = int(rng.integers(15, 121))
            s["credit_history_months"] = int(rng.integers(0, 9))
            s["ssn_verification_status"] = str(
                rng.choice(["PARTIAL_MISMATCH", "THIN_FILE", "MATCH"], p=[0.60, 0.35, 0.05])
            )
            s["shared_kyc_attributes_count"] = 1 + int(rng.poisson(1.2))
            s["credit_line_utilization_pct"] = round(float(rng.beta(9, 1.5) * 100), 1)
            s["device_risk_score"] = round(float(rng.beta(3, 7)), 2)
        else:  # sleeper: aged longer, partial drain, weaker identity signals (~ NORMAL/new_customer)
            s["account_age_days"] = int(rng.integers(60, 301))
            s["credit_history_months"] = int(rng.integers(2, 20))
            s["ssn_verification_status"] = str(
                rng.choice(["THIN_FILE", "PARTIAL_MISMATCH", "MATCH"], p=[0.45, 0.25, 0.30])
            )
            s["shared_kyc_attributes_count"] = int(rng.poisson(0.6))
            s["credit_line_utilization_pct"] = round(float(rng.beta(4, 2.5) * 100), 1)
            s["device_risk_score"] = round(float(rng.beta(2, 8)), 2)

    elif typology == "STRUCTURING_AML":
        n_prior = 3 + int(rng.poisson(1.5)) if scenario == "classic" else 1 + int(rng.random() < 0.4)
        _near_threshold_batch(rng, s, txn, n_prior)
        s["distinct_counterparties_24h"] = max(2, n_prior + 1 - int(rng.poisson(0.7)))
        s["tx_count_1h"] = 1 + int(rng.poisson(1.0))
        s["recipient_is_first_time"] = bool(rng.random() < 0.50)

    else:
        raise ValueError(f"Unknown typology: {typology}")


def _finalize_velocity(rng: np.random.Generator, s: Dict[str, Any], txn: Dict[str, Any]) -> None:
    """Fills the 8k-10k counters for non-structuring alerts and enforces invariants."""
    if s["transfers_8k_to_10k_24h"] == 0 and NEAR_THRESHOLD_LOW <= txn["amount"] < NEAR_THRESHOLD_HIGH:
        s["transfers_8k_to_10k_24h"] = 1
        s["total_8k_to_10k_usd_24h"] = round(float(txn["amount"]), 2)
    floor = max(s["tx_count_1h"], s["transfers_8k_to_10k_24h"], s["distinct_counterparties_24h"])
    s["tx_count_24h"] = floor + int(rng.poisson(1.5))


# --------------------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertPlan:
    typology: str
    scenario: str
    pool: str


def make_plan(rng: np.random.Generator, n: int) -> List[AlertPlan]:
    """Exact class counts (largest remainder), scenarios sampled within class, then shuffled."""
    raw = {t: n * p for t, p in CLASS_MIX.items()}
    counts = {t: int(v) for t, v in raw.items()}
    for t in sorted(raw, key=lambda t: raw[t] - counts[t], reverse=True)[: n - sum(counts.values())]:
        counts[t] += 1
    plan: List[AlertPlan] = []
    for typology in TYPOLOGIES:
        names = list(SCENARIOS[typology])
        probs = np.array([SCENARIOS[typology][k][0] for k in names])
        for scenario in rng.choice(names, size=counts[typology], p=probs / probs.sum()):
            pool_options = SCENARIOS[typology][str(scenario)][1].split("|")
            plan.append(AlertPlan(typology, str(scenario), str(rng.choice(pool_options))))
    order = rng.permutation(len(plan))
    return [plan[i] for i in order]


def build_alert(
    rng: np.random.Generator, index: int, split: str, plan: AlertPlan, source: TransactionSource
) -> Dict[str, Any]:
    txn, row_index = source.take(plan.pool, rng)
    signals = _established_profile(rng, str(rng.choice(US_LOCATIONS)))
    apply_scenario(rng, plan.typology, plan.scenario, signals, txn)
    _finalize_velocity(rng, signals, txn)
    ordered = {k: signals[k] for k in SIGNAL_KEYS}
    assert tuple(ordered) == SIGNAL_KEYS and len(signals) == len(SIGNAL_KEYS), "schema drift"
    return {
        "alert_id": f"{split.upper()}-{index:05d}",
        "transaction": txn,
        "signals": ordered,
        "label": {
            "typology": plan.typology,
            "hold": plan.typology != "NORMAL",
            "scenario": f"{plan.typology}/{plan.scenario}",
        },
        "provenance": {"source": source.name, "paysim_pool": plan.pool, "paysim_row": row_index},
    }


def generate(
    split: str,
    n: int,
    seed: int,
    source: TransactionSource,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    plan = make_plan(rng, n)
    return [build_alert(rng, i + 1, split, p, source) for i, p in enumerate(plan)]


def validate(alerts: List[Dict[str, Any]]) -> None:
    for a in alerts:
        s, t = a["signals"], a["transaction"]
        assert tuple(s) == SIGNAL_KEYS, a["alert_id"]
        assert tuple(t) == tuple(TRANSACTION_KEYS), a["alert_id"]
        assert s["tx_count_24h"] >= max(
            s["tx_count_1h"], s["transfers_8k_to_10k_24h"], s["distinct_counterparties_24h"]
        ), a["alert_id"]
        assert "isFraud" not in t and "isFlaggedFraud" not in t
    ids = [a["alert_id"] for a in alerts]
    assert len(ids) == len(set(ids))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(alerts: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a in alerts:
            f.write(json.dumps(a, separators=(",", ":")) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split", choices=list(SPLIT_DEFAULTS), default="test")
    p.add_argument("--samples", type=int, default=None, help="Default depends on split")
    p.add_argument("--seed", type=int, default=None, help="Default depends on split")
    p.add_argument("--paysim-csv", type=Path, default=DEFAULT_PAYSIM_CSV)
    p.add_argument(
        "--synthetic-core",
        action="store_true",
        help="Use a synthetic stand-in for PaySim rows (plumbing tests only, never publish)",
    )
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    n_default, seed_default = SPLIT_DEFAULTS[args.split]
    n = args.samples or n_default
    seed = args.seed if args.seed is not None else seed_default

    source: TransactionSource
    if args.synthetic_core:
        source = SyntheticStandInSource()
        out_dir = Path("data/stand_in")
    else:
        if not args.paysim_csv.exists():
            raise SystemExit(
                f"PaySim CSV not found at {args.paysim_csv}.\n"
                "Download it from https://www.kaggle.com/datasets/ealaxi/paysim1 into data/raw/,\n"
                "or pass --synthetic-core for a plumbing test (not publishable)."
            )
        source = PaySimSource(args.paysim_csv, args.split)
        out_dir = Path("data")
    output = args.output or out_dir / f"alerts_{args.split}.jsonl"

    alerts = generate(args.split, n, seed, source)
    validate(alerts)
    write_jsonl(alerts, output)

    print(f"[+] {len(alerts)} alerts -> {output}  (source={source.name}, seed={seed})")
    print(f"[+] sha256 {file_sha256(output)}")
    for t, c in Counter(a["label"]["typology"] for a in alerts).most_common():
        print(f"    {t:<18} {c:>6}")
    if len(alerts) <= 1000:
        for sc, c in sorted(Counter(a["label"]["scenario"] for a in alerts).items()):
            print(f"      {sc:<36} {c:>4}")


if __name__ == "__main__":
    main()

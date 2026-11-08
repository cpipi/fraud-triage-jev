"""
Two non-API reference points that frame the Jev vs GPT-4o-mini comparison.

1. Supervised reference model (`oracle`): gradient boosting trained on the separate
   `reference` split (10k alerts from the same generator, disjoint PaySim rows).
   A zero-shot model has never seen the generating distribution, so this model's
   accuracy is a practical *ceiling*. Its probability for the true class on each test
   alert is also our model-independent difficulty score: alerts where even this model
   is unsure are ambiguous by construction, not mistakes by Jev or GPT.

2. Analyst rules engine (`rules`): the if-then rules a fraud analyst would write from
   the same criteria text the models receive. It is the *floor* any model should beat.
   It outputs hard labels only, so it is excluded from calibration metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from triage.schema import TYPOLOGIES

TX_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
SSN_LEVELS = ["MATCH", "THIN_FILE", "PARTIAL_MISMATCH"]

FEATURE_NAMES: List[str] = [
    "amount_log", "drain_ratio", "orig_balance_log", "dest_old_balance_zero",
    "dest_is_merchant", *[f"type_{t}" for t in TX_TYPES],
    "account_age_days", "credit_history_months", "credit_line_utilization_pct",
    "ssn_level", "shared_kyc_attributes_count", "is_known_device", "device_risk_score",
    "ip_away_from_home", "ip_foreign", "travel_notice_active",
    "hours_since_credential_change_log", "failed_login_attempts_24h",
    "hours_since_inbound_funding_log", "recipient_is_first_time",
    "transfers_8k_to_10k_24h", "total_8k_to_10k_usd_24h", "distinct_counterparties_24h",
    "tx_count_1h", "tx_count_24h",
]


def drain_ratio(txn: Dict[str, Any]) -> float:
    old = float(txn["oldbalanceOrg"])
    return min(1.5, float(txn["amount"]) / old) if old > 0 else 0.0


def featurize(alert: Dict[str, Any]) -> List[float]:
    t, s = alert["transaction"], alert["signals"]
    foreign = not str(s["ip_location"]).endswith(
        (", NY", ", TX", ", IL", ", WA", ", FL", ", NC", ", CO", ", AZ")
    )
    return [
        float(np.log1p(t["amount"])),
        drain_ratio(t),
        float(np.log1p(t["oldbalanceOrg"])),
        float(t["oldbalanceDest"] == 0),
        float(str(t["nameDest"]).startswith("M")),
        *[float(t["type"] == x) for x in TX_TYPES],
        float(s["account_age_days"]),
        float(s["credit_history_months"]),
        float(s["credit_line_utilization_pct"]),
        float(SSN_LEVELS.index(s["ssn_verification_status"])),
        float(s["shared_kyc_attributes_count"]),
        float(s["is_known_device"]),
        float(s["device_risk_score"]),
        float(s["ip_location"] != s["home_location"]),
        float(foreign),
        float(s["travel_notice_active"]),
        float(np.log1p(s["hours_since_credential_change"])),
        float(s["failed_login_attempts_24h"]),
        float(np.log1p(s["hours_since_inbound_funding"])),
        float(s["recipient_is_first_time"]),
        float(s["transfers_8k_to_10k_24h"]),
        float(s["total_8k_to_10k_usd_24h"]),
        float(s["distinct_counterparties_24h"]),
        float(s["tx_count_1h"]),
        float(s["tx_count_24h"]),
    ]


def to_xy(alerts: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([featurize(a) for a in alerts], dtype=float)
    y = np.array([TYPOLOGIES.index(a["label"]["typology"]) for a in alerts])
    return X, y


def train_reference_model(reference: List[Dict[str, Any]], seed: int = 0) -> HistGradientBoostingClassifier:
    X, y = to_xy(reference)
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=seed,
    )
    model.fit(X, y)
    return model


def reference_predict(model: HistGradientBoostingClassifier, alerts: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    X, _ = to_xy(alerts)
    proba = model.predict_proba(X)
    out = []
    for row in proba:
        full = np.zeros(len(TYPOLOGIES))
        full[model.classes_] = row
        out.append({t: float(full[i]) for i, t in enumerate(TYPOLOGIES)})
    return out


def rules_predict(alert: Dict[str, Any]) -> str:
    """Analyst rules, written from the criteria text (not from the generator)."""
    t, s = alert["transaction"], alert["signals"]
    drain = drain_ratio(t)
    if s["transfers_8k_to_10k_24h"] >= 3:
        return "STRUCTURING_AML"
    hijack_signs = sum(
        [
            not s["is_known_device"],
            s["ip_location"] != s["home_location"] and not s["travel_notice_active"],
            s["hours_since_credential_change"] < 48,
            s["failed_login_attempts_24h"] >= 2,
        ]
    )
    if hijack_signs >= 3 and drain >= 0.5:
        return "ACCOUNT_TAKEOVER"
    if s["ssn_verification_status"] != "MATCH" and s["account_age_days"] < 180:
        return "SYNTHETIC_ID"
    if s["hours_since_inbound_funding"] < 12 and drain >= 0.8:
        return "CASH_OUT_FRAUD"
    return "NORMAL"

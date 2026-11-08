"""
Single source of truth for the triage task: typologies, the questions sent to Jev,
and the identical criteria text given to the GPT-4o-mini baseline.

Rules that keep the comparison honest:
- Criteria describe each typology in domain terms. They never quote numeric cut-offs
  used by `generate_dataset.py` (the only number mentioned is the public $10,000
  FinCEN currency transaction reporting threshold, which is domain knowledge).
- Criteria are frozen before a test run. Any rewording is done against the `dev`
  split only, and every report records `criteria_sha256` so readers can see which
  wording produced which numbers.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

TYPOLOGIES: List[str] = [
    "NORMAL",
    "MONEY_MULE",
    "ACCOUNT_TAKEOVER",
    "SYNTHETIC_ID",
    "STRUCTURING_AML",
]

TYPOLOGY_CRITERIA: Dict[str, str] = {
    "NORMAL": (
        "Legitimate activity by the account holder. Unusual details such as travel, a new "
        "phone, a large one-off payment or a young account are explained by the rest of the "
        "profile rather than pointing to fraud."
    ),
    "MONEY_MULE": (
        "Money-mule pass-through: the account receives funds and quickly moves most of the "
        "balance on to a new counterparty. The session looks like the genuine owner; the "
        "suspicious part is the flow of money, not the login."
    ),
    "ACCOUNT_TAKEOVER": (
        "Someone other than the owner controls the session: signs of a hijacked login such "
        "as an unfamiliar device, an unexpected location, recent password or MFA changes, or "
        "failed login attempts, followed by moving the money out."
    ),
    "SYNTHETIC_ID": (
        "The identity itself is fabricated or stitched together: a young account with a thin "
        "or inconsistent identity record (SSN not fully verified, KYC attributes shared with "
        "other customers) that runs up its credit and drains the balance."
    ),
    "STRUCTURING_AML": (
        "Deliberately splitting money into several transfers just under the $10,000 currency "
        "transaction reporting threshold within a day, often to several counterparties, so "
        "that no single transfer triggers a report."
    ),
}

TYPOLOGY_INSTRUCTIONS = (
    "Which typology best describes this alert? `transaction` is the payment that raised the "
    "alert; `signals` holds session, device, identity and 24-hour velocity telemetry for the "
    "sending account."
)

HOLD_INSTRUCTIONS = (
    "Should this transaction be held for fraud or AML compliance review before it settles?"
)

HOLD_CRITERIA: Dict[str, str] = {
    "true": (
        "The transaction shows fraud (mule cash-out, account takeover, synthetic identity) or "
        "AML structuring and must not settle automatically."
    ),
    "false": "Legitimate customer activity that can settle straight through.",
}

# Question IDs are not sent to the model (per the TypeSafe docs); only content matters.
JEV_QUESTIONS: Dict[str, Any] = {
    "typology": {
        "type": "choice",
        "instructions": TYPOLOGY_INSTRUCTIONS,
        "criteria": TYPOLOGY_CRITERIA,
    },
    "hold": {
        "type": "noul",
        "instructions": HOLD_INSTRUCTIONS,
        "criteria": HOLD_CRITERIA,
    },
}


# Added to the GPT-4o-mini prompt in its reasoning-first configuration only.
REASONING_INSTRUCTION = (
    "\nFirst write 2-4 sentences in `reasoning` weighing the signals for and against each "
    "plausible typology, then give `typology` and `hold`."
)


def build_state(alert: Dict[str, Any]) -> Dict[str, Any]:
    """The exact model input. Labels, scenario names and data provenance are never included."""
    return {"transaction": alert["transaction"], "signals": alert["signals"]}


def build_llm_system_prompt() -> str:
    """Same instructions and criteria text as the Jev request, formatted for a chat model."""
    typology_lines = "\n".join(f"- {k}: {v}" for k, v in TYPOLOGY_CRITERIA.items())
    return (
        "You triage payment fraud alerts. The user message is a JSON object with "
        "`transaction` and `signals`. Answer two questions.\n\n"
        f"1. typology. {TYPOLOGY_INSTRUCTIONS} Options:\n{typology_lines}\n\n"
        f"2. hold. {HOLD_INSTRUCTIONS}\n"
        f"- true: {HOLD_CRITERIA['true']}\n"
        f"- false: {HOLD_CRITERIA['false']}\n\n"
        "Reply with JSON only."
    )


def criteria_sha256() -> str:
    """Fingerprint of everything that shapes the model's answer, recorded in every report."""
    blob = json.dumps(
        {
            "jev_questions": JEV_QUESTIONS,
            "llm_system_prompt": build_llm_system_prompt(),
            "llm_reasoning_instruction": REASONING_INSTRUCTION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

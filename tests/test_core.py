"""Offline tests: generator invariants, response parsers, and metric conventions."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import generate_dataset as gd
from triage.clients import Prediction, parse_gpt_response, parse_jev_response
from triage.metrics import calibration, classification, hold_gate, paired_bootstrap, settlement
from triage.schema import TYPOLOGIES, build_state


@pytest.fixture(scope="module")
def alerts():
    return gd.generate("test", 300, 42, gd.SyntheticStandInSource())


def test_generator_is_deterministic(alerts):
    again = gd.generate("test", 300, 42, gd.SyntheticStandInSource())
    assert json.dumps(alerts) == json.dumps(again)


def test_generator_schema_and_invariants(alerts):
    gd.validate(alerts)
    counts = {t: sum(a["label"]["typology"] == t for a in alerts) for t in TYPOLOGIES}
    assert counts == {"NORMAL": 120, "MONEY_MULE": 45, "ACCOUNT_TAKEOVER": 45,
                      "SYNTHETIC_ID": 45, "STRUCTURING_AML": 45}


def test_model_input_never_contains_labels(alerts):
    for a in alerts:
        blob = json.dumps(build_state(a))
        assert "label" not in blob and "scenario" not in blob and "isFraud" not in blob
        assert a["label"]["typology"] not in blob


def test_parse_jev_response():
    data = {
        "model": "jev-1.14.0",
        "answers": {
            "typology": {"type": "choice", "choice": "SYNTHETIC_ID", "confidence": 0.775,
                         "probabilities": {"NORMAL": 0.02, "MONEY_MULE": 0.16, "ACCOUNT_TAKEOVER": 0.0,
                                           "SYNTHETIC_ID": 0.82, "STRUCTURING_AML": 0.0}},
            "hold": {"type": "noul", "noul": 0.78},
        },
        "usage": {"input_tokens": 1072, "output_tokens": 93},
    }
    choice, probs, hold, conf = parse_jev_response(data)
    assert choice == "SYNTHETIC_ID" and math.isclose(probs["SYNTHETIC_ID"], 0.82) and hold == 0.78


def test_parse_jev_rejects_missing_answer():
    with pytest.raises(KeyError):
        parse_jev_response({"answers": {"typology": {"choice": "NORMAL", "confidence": 1,
                                                     "probabilities": {t: 0.2 for t in TYPOLOGIES}}}})


def _lp(token, alts):
    return {"token": token, "logprob": math.log(alts[0][1]),
            "top_logprobs": [{"token": t, "logprob": math.log(p) if p > 0 else -100.0} for t, p in alts]}


def test_parse_gpt_logprobs_real_token_layout():
    # Token layout observed from gpt-4o-mini-2024-07-18 with strict json_schema output
    # (the label is split into several tokens; probabilities live on the first one).
    content = '{"typology":"MONEY_MULE","hold":true}'
    toks = [
        _lp('{"', [('{"', 1.0)]), _lp("typ", [("typ", 1.0)]), _lp("ology", [("ology", 1.0)]),
        _lp('":"', [('":"', 1.0)]),
        _lp("M", [("M", 0.70), ("ACCOUNT", 0.20), ("S", 0.08), ("NORMAL", 0.02)]),
        _lp("ONEY", [("ONEY", 1.0)]), _lp("_M", [("_M", 1.0)]), _lp("ULE", [("ULE", 1.0)]),
        _lp('","', [('","', 1.0)]),
        _lp("hold", [("hold", 1.0)]), _lp('":', [('":', 1.0)]),
        _lp("true", [("true", 0.9), (" true", 0.02), ("false", 0.08)]), _lp("}", [("}", 1.0)]),
    ]
    data = {"choices": [{"message": {"content": content}, "logprobs": {"content": toks}}]}
    predicted, probs, hold, coverage, source = parse_gpt_response(data)
    assert predicted == "MONEY_MULE" and source == "logprobs"
    assert math.isclose(probs["MONEY_MULE"], 0.70, abs_tol=1e-6)
    assert math.isclose(probs["ACCOUNT_TAKEOVER"], 0.20, abs_tol=1e-6)
    assert math.isclose(probs["SYNTHETIC_ID"], 0.04, abs_tol=1e-6)  # "S" is split two ways
    assert math.isclose(probs["STRUCTURING_AML"], 0.04, abs_tol=1e-6)
    assert math.isclose(hold, 0.92, abs_tol=1e-6)  # " true" counts as true
    assert math.isclose(sum(probs.values()), 1.0)


def _pred(label, probs=None, ok=True, hold=0.5):
    probs = probs or {t: float(t == label) for t in TYPOLOGIES}
    return Prediction(model="m", alert_id="x", ok=ok, predicted=label if ok else "ERROR",
                      probs=probs if ok else {t: 0.0 for t in TYPOLOGIES}, hold_prob=hold if ok else float("nan"))


def test_errors_count_as_misses():
    truth = ["NORMAL", "ACCOUNT_TAKEOVER"]
    preds = [_pred("NORMAL"), _pred("", ok=False)]
    c = classification(truth, preds)
    assert c["accuracy"] == 0.5
    assert c["per_typology"]["ACCOUNT_TAKEOVER"]["recall"] == 0.0
    # Failed call carries no probability mass: its Brier term is 1 (the truth's (0-1)^2).
    assert calibration(truth, preds)["brier"] == pytest.approx(0.5)  # (0 + 1) / 2


def test_settlement_uses_predicted_class_probability():
    truth = ["NORMAL", "MONEY_MULE"]
    p1 = {"NORMAL": 0.9, "MONEY_MULE": 0.1, "ACCOUNT_TAKEOVER": 0, "SYNTHETIC_ID": 0, "STRUCTURING_AML": 0}
    p2 = {"NORMAL": 0.86, "MONEY_MULE": 0.14, "ACCOUNT_TAKEOVER": 0, "SYNTHETIC_ID": 0, "STRUCTURING_AML": 0}
    rows = {r["tau"]: r for r in settlement(truth, [_pred("NORMAL", p1), _pred("NORMAL", p2)])}
    assert rows[0.85]["settled"] == 2 and rows[0.85]["fraud_auto_cleared"] == 1
    assert rows[0.90]["settled"] == 1 and rows[0.90]["fraud_auto_cleared"] == 0


def test_hold_gate_bands():
    g = hold_gate([True, True, False, False], [_pred("X", hold=0.9), _pred("X", hold=0.2),
                                                _pred("X", hold=0.5), _pred("X", hold=0.8)])
    assert (g["held"], g["released"], g["sent_to_review"]) == (2, 1, 1)
    assert g["fraud_released"] == 1 and g["legit_held"] == 1


def test_paired_bootstrap_identical_models_have_zero_difference():
    rng = np.random.default_rng(0)
    truth = list(rng.choice(TYPOLOGIES, 60))
    preds = [_pred(str(t)) for t in rng.choice(TYPOLOGIES, 60)]
    out = paired_bootstrap(truth, {"a": preds, "b": preds}, n_boot=200)
    assert out["differences"]["a - b"]["macro_f1"]["ci95"] == [0.0, 0.0]


def test_parse_gpt_reasoning_first_reads_label_token_not_reasoning_text():
    content = '{"reasoning":"Mentions \\"typology\\": oddly.","typology":"NORMAL","hold":false}'
    prefix = '{"reasoning":"Mentions \\"typology\\": oddly.","typology":"'
    toks = [_lp(prefix, [(prefix, 1.0)]),
            _lp("NORMAL", [("NORMAL", 0.6), ("ACCOUNT", 0.4)]),
            _lp('","hold":', [('","hold":', 1.0)]),
            _lp("false", [("false", 0.75), ("true", 0.25)]), _lp("}", [("}", 1.0)])]
    data = {"choices": [{"message": {"content": content}, "logprobs": {"content": toks}}]}
    predicted, probs, hold, _, _ = parse_gpt_response(data)
    assert predicted == "NORMAL"
    assert math.isclose(probs["NORMAL"], 0.6, abs_tol=1e-6)
    assert math.isclose(hold, 0.25, abs_tol=1e-6)

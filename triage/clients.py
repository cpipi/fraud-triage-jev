"""
API clients for Jev (`POST /v1/systemone`) and the GPT-4o-mini baseline.

Fairness rules both clients follow:
- Same `state` JSON, same criteria text (see `triage.schema`).
- Same HTTP stack (one shared httpx.AsyncClient, HTTP/1.1), same timeout, same retry
  policy (429/5xx/529 with exponential backoff, honouring `retry-after`).
- Latency is wall-clock time of the successful attempt, measured on the client
  (request sent -> full body received). End-to-end time including retries and
  backoff is recorded separately.
- A failed call returns `ok=False`; metrics count it as a miss, never drop it.

GPT-4o-mini probabilities come from token logprobs at the position where the
typology value starts (and where the `hold` boolean starts), not from numbers
the model writes in its answer.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from triage.schema import JEV_QUESTIONS, TYPOLOGIES, build_llm_system_prompt, build_state

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

# Pinned versions (aliases move when new releases ship).
JEV_MODEL = "jev-1.14.0"
GPT_MODEL = "gpt-4o-mini-2024-07-18"

# USD per 1M tokens. Jev: docs.typesafe.ai/models (input only, output free).
JEV_INPUT_PER_M = 0.042
GPT_INPUT_PER_M = 0.15
GPT_CACHED_INPUT_PER_M = 0.075
GPT_OUTPUT_PER_M = 0.60

TIMEOUT_S = 30.0
MAX_ATTEMPTS = 4
RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}


@dataclass
class Prediction:
    model: str  # short name used in reports: "jev", "gpt-4o-mini", ...
    alert_id: str
    ok: bool
    predicted: str = "ERROR"
    probs: Dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in TYPOLOGIES})
    hold_prob: float = float("nan")
    prob_source: str = ""  # "native" (Jev), "logprobs" (GPT), "argmax_only", "simulated"
    prob_coverage: float = float("nan")  # GPT: share of top-logprob mass mapped to a label
    vendor_confidence: Optional[float] = None  # Jev's concentration statistic, for reference
    latency_ms: float = float("nan")
    e2e_latency_ms: float = float("nan")
    attempts: int = 0
    status_codes: List[int] = field(default_factory=list)
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    served_model: str = ""
    system_fingerprint: Optional[str] = None
    started_at_s: float = float("nan")  # seconds since run start, to check interleaving
    error: Optional[str] = None

    @property
    def p_pred(self) -> float:
        return float(self.probs.get(self.predicted, 0.0)) if self.ok else 0.0


class ApiError(Exception):
    pass


def _retry_after_s(resp: httpx.Response, attempt: int) -> float:
    header = resp.headers.get("retry-after")
    if header:
        try:
            return min(20.0, float(header))
        except ValueError:
            pass
    return 0.5 * (2**attempt) + random.uniform(0, 0.25)


async def _post_with_retries(
    http: httpx.AsyncClient, url: str, headers: Dict[str, str], payload: Dict[str, Any], pred: Prediction
) -> Tuple[Dict[str, Any], float]:
    """Returns (json, latency_ms of the successful attempt). Raises ApiError when exhausted."""
    last_err = "no attempt made"
    for attempt in range(MAX_ATTEMPTS):
        pred.attempts = attempt + 1
        t0 = time.perf_counter()
        try:
            resp = await http.post(url, headers=headers, json=payload, timeout=TIMEOUT_S)
            await resp.aread()
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except httpx.HTTPError as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(0.5 * (2**attempt))
            continue
        pred.status_codes.append(resp.status_code)
        if resp.status_code == 200:
            return resp.json(), latency_ms
        last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        if resp.status_code in RETRYABLE and attempt + 1 < MAX_ATTEMPTS:
            await asyncio.sleep(_retry_after_s(resp, attempt))
            continue
        break
    raise ApiError(last_err)


# --------------------------------------------------------------------------------------
# Jev
# --------------------------------------------------------------------------------------


def parse_jev_response(data: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, float]:
    """Strict parse of the documented response. Raises ValueError on any deviation."""
    answers = data["answers"]
    typ = answers["typology"]
    probs_raw = typ["probabilities"]
    if set(probs_raw) != set(TYPOLOGIES):
        raise ValueError(f"unexpected probability keys: {sorted(probs_raw)}")
    probs = {t: float(probs_raw[t]) for t in TYPOLOGIES}
    if not math.isclose(sum(probs.values()), 1.0, abs_tol=0.02):
        raise ValueError(f"probabilities sum to {sum(probs.values()):.3f}")
    choice = str(typ["choice"])
    if choice not in TYPOLOGIES:
        raise ValueError(f"invalid choice {choice!r}")
    hold = float(answers["hold"]["noul"])
    if not 0.0 <= hold <= 1.0:
        raise ValueError(f"noul out of range: {hold}")
    return choice, probs, hold, float(typ["confidence"])


class JevClient:
    name = "jev"

    def __init__(self, api_key: str, model: str = JEV_MODEL):
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def payload(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        return {"model": self.model, "state": build_state(alert), "questions": JEV_QUESTIONS}

    async def predict(self, http: httpx.AsyncClient, alert: Dict[str, Any]) -> Prediction:
        pred = Prediction(model=self.name, alert_id=alert["alert_id"], ok=False)
        t0 = time.perf_counter()
        try:
            data, latency = await _post_with_retries(http, JEV_ENDPOINT, self.headers, self.payload(alert), pred)
            choice, probs, hold, conf = parse_jev_response(data)
            usage = data["usage"]
            pred.ok = True
            pred.predicted, pred.probs, pred.hold_prob = choice, probs, hold
            pred.vendor_confidence = conf
            pred.prob_source = "native"
            pred.latency_ms = latency
            pred.input_tokens = int(usage["input_tokens"])
            pred.output_tokens = int(usage.get("output_tokens", 0))
            pred.cost_usd = pred.input_tokens * JEV_INPUT_PER_M / 1e6
            pred.served_model = str(data.get("model", ""))
        except (ApiError, KeyError, TypeError, ValueError) as exc:
            pred.error = f"{type(exc).__name__}: {exc}"
        pred.e2e_latency_ms = (time.perf_counter() - t0) * 1000.0
        return pred


# --------------------------------------------------------------------------------------
# GPT-4o-mini
# --------------------------------------------------------------------------------------

GPT_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "fraud_triage",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "typology": {"type": "string", "enum": TYPOLOGIES},
                "hold": {"type": "boolean"},
            },
            "required": ["typology", "hold"],
            "additionalProperties": False,
        },
    },
}


def _token_at(tokens: List[Dict[str, Any]], offset: int) -> Tuple[int, int]:
    """Index of the token whose text span contains character `offset`, and that token's start."""
    pos = 0
    for i, tok in enumerate(tokens):
        end = pos + len(tok["token"])
        if pos <= offset < end:
            return i, pos
        pos = end
    raise ValueError(f"offset {offset} beyond decoded text")


def _value_distribution(
    content: str, tokens: List[Dict[str, Any]], key: str, options: List[str]
) -> Tuple[Dict[str, float], float]:
    """
    Distribution over `options` for the JSON value of `key`, read from top_logprobs at the
    token where that value starts. Returns (normalised probabilities, mapped mass).
    Alternative tokens are mapped by prefix; a prefix shared by several options splits
    its mass equally between them.
    """
    marker = f'"{key}":'
    start = content.index(marker) + len(marker)
    while content[start] in ' "':
        start += 1
    idx, tok_start = _token_at(tokens, start)
    lead = content[tok_start:start]  # e.g. '":"' when the token also carries the opening quote
    mass = {o: 0.0 for o in options}
    for alt in tokens[idx]["top_logprobs"]:
        text = alt["token"]
        if not text.startswith(lead):
            continue
        frag = text[len(lead):].lstrip(' "')
        if not frag:
            continue
        matches = [o for o in options if o.startswith(frag) or frag.startswith(o)]
        for m in matches:
            mass[m] += math.exp(alt["logprob"]) / len(matches)
    total = sum(mass.values())
    if total <= 0:
        raise ValueError(f"no top_logprobs mapped to {key} options")
    return {o: v / total for o, v in mass.items()}, total


def parse_gpt_response(data: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, float, str]:
    choice = data["choices"][0]
    content = choice["message"]["content"]
    parsed = json.loads(content)
    predicted = str(parsed["typology"])
    if predicted not in TYPOLOGIES:
        raise ValueError(f"invalid typology {predicted!r}")
    lp = (choice.get("logprobs") or {}).get("content")
    if not lp:
        probs = {t: float(t == predicted) for t in TYPOLOGIES}
        return predicted, probs, float(bool(parsed["hold"])), float("nan"), "argmax_only"
    probs, coverage = _value_distribution(content, lp, "typology", TYPOLOGIES)
    hold_dist, _ = _value_distribution(content, lp, "hold", ["true", "false"])
    return predicted, probs, hold_dist["true"], coverage, "logprobs"


class OpenAIClient:
    name = "gpt-4o-mini"

    def __init__(self, api_key: str, model: str = GPT_MODEL):
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.system_prompt = build_llm_system_prompt()

    def payload(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": 0,
            "seed": 42,
            "max_tokens": 40,
            "logprobs": True,
            "top_logprobs": 20,
            "response_format": GPT_RESPONSE_FORMAT,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(build_state(alert), separators=(",", ":"))},
            ],
        }

    async def predict(self, http: httpx.AsyncClient, alert: Dict[str, Any]) -> Prediction:
        pred = Prediction(model=self.name, alert_id=alert["alert_id"], ok=False)
        t0 = time.perf_counter()
        try:
            data, latency = await _post_with_retries(http, OPENAI_ENDPOINT, self.headers, self.payload(alert), pred)
            predicted, probs, hold, coverage, source = parse_gpt_response(data)
            usage = data["usage"]
            cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
            pred.ok = True
            pred.predicted, pred.probs, pred.hold_prob = predicted, probs, hold
            pred.prob_source, pred.prob_coverage = source, coverage
            pred.latency_ms = latency
            pred.input_tokens = int(usage["prompt_tokens"])
            pred.cached_input_tokens = cached
            pred.output_tokens = int(usage["completion_tokens"])
            pred.cost_usd = (
                (pred.input_tokens - cached) * GPT_INPUT_PER_M
                + cached * GPT_CACHED_INPUT_PER_M
                + pred.output_tokens * GPT_OUTPUT_PER_M
            ) / 1e6
            pred.served_model = str(data.get("model", ""))
            pred.system_fingerprint = data.get("system_fingerprint")
        except (ApiError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            pred.error = f"{type(exc).__name__}: {exc}"
        pred.e2e_latency_ms = (time.perf_counter() - t0) * 1000.0
        return pred


# --------------------------------------------------------------------------------------
# Dry-run stand-in (plumbing only; numbers are meaningless and reports say so)
# --------------------------------------------------------------------------------------


class SimulatedClient:
    def __init__(self, name: str, latency_median_ms: float, accuracy: float, seed: int):
        self.name = name
        self.latency_median_ms = latency_median_ms
        self.accuracy = accuracy
        self.rng = np.random.default_rng(seed)

    async def predict(self, http: httpx.AsyncClient, alert: Dict[str, Any]) -> Prediction:
        rng = self.rng
        latency = float(rng.lognormal(np.log(self.latency_median_ms), 0.25))
        await asyncio.sleep(min(latency, 50.0) / 1000.0)
        truth = alert["label"]["typology"]
        predicted = truth if rng.random() < self.accuracy else str(rng.choice([t for t in TYPOLOGIES if t != truth]))
        raw = rng.dirichlet(np.ones(len(TYPOLOGIES)) * 0.3)
        raw[TYPOLOGIES.index(predicted)] += 2.0
        probs = {t: float(v) for t, v in zip(TYPOLOGIES, raw / raw.sum())}
        return Prediction(
            model=self.name, alert_id=alert["alert_id"], ok=True, predicted=predicted, probs=probs,
            hold_prob=float(1 - probs["NORMAL"]), prob_source="simulated", latency_ms=latency,
            e2e_latency_ms=latency, attempts=1, status_codes=[200], input_tokens=1000,
            cost_usd=1000 * JEV_INPUT_PER_M / 1e6, served_model=f"{self.name}-simulated",
        )

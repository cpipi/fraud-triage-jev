# Fraud alert triage on Jev

**Author:** Anuar Sultan · built with [Eventum AI](https://eventum.ai)

Payment fraud alerts need a decision in the payment path: settle, hold, or send to an analyst, and
which fraud typology to route to. This repository benchmarks TypeSafe's decision model **Jev**
(`POST https://api.typesafe.ai/v1/systemone`) against **GPT-4o-mini** on that task, measuring:

- **F1 per typology** (5 fixed classes),
- **alerts settled automatically at high confidence**, and how many of those were wrong,
- **P90 latency** under concurrent load, and
- **cost per 1,000 alerts**.

Both models receive the same alert JSON and the same criteria text. Results are in
[`results/live/`](results/live/); the analysis is in [`notebooks/`](notebooks/).

## The task

Each alert is one transaction plus 19 session, device, identity and velocity signals for the sending
account. Jev answers two questions in one request:

| Question | Jev type | Output |
|---|---|---|
| Which typology? `NORMAL`, `CASH_OUT_FRAUD`, `ACCOUNT_TAKEOVER`, `SYNTHETIC_ID`, `STRUCTURING_AML` | `choice` | label + probability per class |
| Hold for fraud/AML review before settlement? | `noul` | p(yes) |

GPT-4o-mini gets the same two questions through Structured Outputs. Its class probabilities are read from
token log-probabilities, not from numbers it writes.

## Data: how the alerts are built

A benchmark like this is only as convincing as its labels, so the design starts there.

- **Transactions are real PaySim rows.** [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
  is a public mobile-money simulation (Lopez-Rojas et al., 2016; CC BY-SA 4.0). Each scenario draws
  from a published pool, for example "PaySim rows flagged `isFraud`" or "non-fraud transfers of
  $8k–$10k". PaySim's fraud flags are never shown to a model.
- **Labels are the scenario that generated the alert.** They are never inferred by rules over the
  features. There are 17 scenarios across the 5 typologies (see
  [`generate_dataset.py`](generate_dataset.py)).
- **Some scenarios are deliberate look-alikes:** a legitimate phone upgrade followed by a large
  payment vs. a quiet account takeover; payroll split into near-threshold transfers vs. light
  structuring; a traveller who forgot the travel notice vs. a stolen session cookie. Some alerts are
  therefore ambiguous on purpose.
- **Two reference points frame the API models:**
  - **Supervised ceiling.** Gradient boosting trained on 10,000 separate alerts from disjoint PaySim
    rows. A zero-shot model has never seen this distribution, so this is a practical upper bound.
    Its probability for the true class also labels each test alert *clear* or *ambiguous*.
  - **Analyst rules floor.** The if-then rules an analyst would write from the same criteria text.
- **Splits:** `test` has 500 alerts (seed 42) and is the reported benchmark. `dev` has 100 alerts
  (seed 7) and is the only split used when wording criteria. `reference` has 10,000 alerts (seed
  1234) and is used only to train the ceiling model.

## Measurement rules

- **Failed calls count as misses** in F1. They are reported, never dropped.
- **Settlement threshold.** Auto-settle when the probability of the predicted class is ≥ τ (0.85
  headline, full curve reported). This follows TypeSafe's guidance; Jev's `confidence` field
  measures concentration, not correctness.
- **Hold gate.** Release at p ≤ 0.30, hold at p ≥ 0.70, anything in between goes to an analyst
  (TypeSafe's suggested starting band).
- **Same network window.** Both models run through one HTTP client with the same timeouts and
  retries. Their calls are interleaved in random order.
- **Latency** is client wall-clock time for the successful attempt. End-to-end time including
  retries is reported separately.
- **Paired bootstrap.** The Jev−GPT difference is bootstrapped over the same alerts (2,000 resamples).
- **Provenance.** Every report records the git commit, dataset SHA-256, criteria SHA-256, pinned and
  served model versions (`jev-1.14.0`, `gpt-4o-mini-2024-07-18`), concurrency and client region.

## Reproduce

```bash
uv sync --all-groups

# 1. Download PaySim (Kaggle login required) to:
#    data/raw/PS_20174392719_1491204439457_log.csv

# 2. Build the three splits
uv run generate_dataset.py --split test
uv run generate_dataset.py --split dev
uv run generate_dataset.py --split reference

# 3. Keys in .env (never committed):  TYPESAFE_API_KEY=...  OPENAI_API_KEY=...
uv run evaluate.py --mode dry-run                                   # plumbing, no API calls
uv run evaluate.py --mode live --split dev --limit 10               # smoke test
uv run evaluate.py --mode live --concurrency 8 --run-name main       # the benchmark
uv run evaluate.py --mode live --concurrency 1 --limit 150 --run-name sequential  # unloaded latency

# 4. Notebooks and tests
uv run --group notebooks python notebooks/build_notebooks.py
uv run pytest
```

Without the PaySim CSV, `generate_dataset.py --synthetic-core` builds a labelled stand-in
transaction block for plumbing tests. The runner refuses live calls on it unless
`--allow-stand-in` is passed, and reports from it are marked *not publishable*.

## Layout

```text
generate_dataset.py        PaySim-seeded alert generator (test / dev / reference splits)
evaluate.py                benchmark runner: Jev, GPT-4o-mini, ceiling, floor -> results/
triage/schema.py           typologies, questions and criteria (single source for both models)
triage/clients.py          Jev and OpenAI clients, strict parsing, logprob probabilities
triage/baselines.py        supervised ceiling and analyst rules
triage/metrics.py          F1, settlement, calibration, hold gate, latency, cost, paired bootstrap
triage/report.py           Markdown report
notebooks/                 01 dataset analysis, 02 results analysis
tests/                     offline tests (generator, parsers, metric conventions)
```

## Limitations

- **The signals are synthetic.** Only the transaction block is real PaySim data. The labels describe
  how this generator behaves, not how fraud is distributed at a real bank. The benchmark compares
  models on a shared, fully specified task; it does not predict production fraud-detection rates.
- **PaySim quirks.** Amounts are in simulated local currency, treated here as USD. Some balance
  fields are inconsistent in the original data, and those quirks are kept.
- **Latency depends on where the client runs.** Each report records the client region; tail
  latencies from 500 calls carry a confidence interval, also reported.
- **GPT-4o-mini probabilities come from the first token of the label.** Labels that share a first
  token split that token's probability equally. The average mass mapped to a label is reported.

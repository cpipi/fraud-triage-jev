# Fraud alert triage on Jev

**Author:** Anuar Sultan · built with [Eventum AI](https://eventum.ai)

Payment fraud alerts need a decision in the payment path: settle, hold, or send to an analyst, and
which fraud typology to route to. This repository benchmarks TypeSafe's decision model **Jev**
(`POST https://api.typesafe.ai/v1/systemone`) against **GPT-4o-mini** on that task.

It is a **controlled benchmark on simulated alerts**: it measures how well each model reads structured
fraud signals against a written rubric. It does not measure fraud detection on real bank data (see
[How synthetic is this?](#how-synthetic-is-this)). It measures:

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
| Which typology? `NORMAL`, `MONEY_MULE`, `ACCOUNT_TAKEOVER`, `SYNTHETIC_ID`, `STRUCTURING_AML` | `choice` | label + probability per class |
| Hold for fraud/AML review before settlement? | `noul` | p(yes) |

GPT-4o-mini gets the same two questions through Structured Outputs, in two configurations:
**label only** (answers immediately) and **reasoning first** (writes 2–4 sentences, then answers).
Its class probabilities are read from token log-probabilities, not from numbers it writes.

**GPT-4o-mini setup** (code in [`triage/clients.py`](triage/clients.py), prompt in [`triage/schema.py`](triage/schema.py)):

| | Label only | Reasoning first |
|---|---|---|
| Model | `gpt-4o-mini-2024-07-18` | `gpt-4o-mini-2024-07-18` |
| Endpoint | Chat Completions, Structured Outputs (strict `json_schema`) | same |
| System prompt | `build_llm_system_prompt()`: the same instructions and criteria text Jev receives | the same prompt plus `REASONING_INSTRUCTION` (below) |
| Output schema | `{typology, hold}` | `{reasoning, typology, hold}`, with `reasoning` first so it is written before the label |
| User message | the alert JSON (`transaction`, `signals`) | same |
| Sampling | `temperature=0`, `seed=42`, `logprobs=true`, `top_logprobs=20` | same |
| `max_tokens` | 40 | 400 |

The reasoning instruction, appended verbatim:

> First write 2-4 sentences in `reasoning` weighing the signals for and against each plausible
> typology, then give `typology` and `hold`.

Both prompts and the Jev criteria are hashed together into the `criteria_sha256` that every report records.

## Results (v2, 500 test alerts, client in Almaty, KZ)

| | Jev | GPT-4o-mini, label only | GPT-4o-mini, reasoning first | Analyst rules (floor) | Supervised (ceiling) |
|---|---:|---:|---:|---:|---:|
| Accuracy | **88.4%** | 47.8% | 58.2% | 82.8% | 95.6% |
| Macro F1 [95% CI] | **0.891** [0.864, 0.914] | 0.474 | 0.577 | 0.831 | 0.960 |
| Settled at p ≥ 0.85 (accuracy of settled) | 47% (96.6%) | 65% (55.2%) | 86% (59.3%) | – | 91% (98.9%) |
| Fraud auto-cleared at p ≥ 0.85 | **0** | 0 | 17 | – | 0 |
| P90 latency, concurrency 8 | **336 ms** | 1,153 ms | 3,890 ms | – | – |
| Cost per 1,000 alerts | **$0.045** | $0.113 | $0.190 | – | – |

### F1 per typology

| Typology | n | Jev | GPT-4o-mini, label only | GPT-4o-mini, reasoning first | Analyst rules (floor) | Supervised (ceiling) |
|---|---:|---:|---:|---:|---:|---:|
| `NORMAL` | 200 | 0.866 | 0.340 | 0.562 | 0.813 | 0.945 |
| `MONEY_MULE` | 75 | 0.839 | 0.364 | 0.406 | 0.937 | 0.993 |
| `ACCOUNT_TAKEOVER` | 75 | 0.821 | 0.698 | 0.667 | 0.635 | 0.887 |
| `SYNTHETIC_ID` | 75 | 0.966 | 0.273 | 0.500 | 0.874 | 1.000 |
| `STRUCTURING_AML` | 75 | 0.962 | 0.694 | 0.750 | 0.894 | 0.974 |

Jev's weak spots:
- **`MONEY_MULE`**: precision is 0.74. Recall is 0.97, so Jev catches nearly every mule, but it also labels some legitimate large transfers as mules. The analyst rules do better on this class (0.937).
- **`NORMAL`**: recall is 0.80. One in five legitimate alerts gets a fraud label. Payroll split into near-threshold transfers scored 0 of 6.
- **`ACCOUNT_TAKEOVER`**: lowest F1 (0.821). This is where the look-alike scenarios sit.

Precision and recall for every model are in [`report.md`](results/live/main/report.md#f1-per-typology).

**Repeatability** (same 150 alerts asked twice, in the `main` and `sequential` runs): Jev gave the same label
146/150 times (largest probability change 0.10). GPT-4o-mini at temperature 0 matched itself 132/150 times label-only
and 111/150 times reasoning-first.

Full report: [`results/live/main/report.md`](results/live/main/report.md). Unloaded latency (concurrency 1):
[`results/live/sequential/report.md`](results/live/sequential/report.md). Charts and error analysis:
[`notebooks/02_results_analysis.ipynb`](notebooks/02_results_analysis.ipynb).

### Revision history (disclosed)

- **v1** ([`results/live/v1/`](results/live/v1/)) used the label `CASH_OUT_FRAUD`. GPT-4o-mini scored 27% because
  it matched that label to PaySim's `CASH_OUT` transaction type (it labelled 36 `CASH_IN` deposits as cash-out
  fraud). Jev showed a milder version of the same confusion.
- **v2** renames the label to `MONEY_MULE` for both models and adds the reasoning-first GPT configuration. The
  change was decided on the `dev` split (Jev 88% → 91%, GPT 26% → 46% from the rename alone), then the test
  split was re-run once. The alerts themselves are byte-identical apart from the label string.

## Data: how the alerts are built

A benchmark like this is only as convincing as its labels, so the design starts there.

- **Transactions are unmodified PaySim rows.** [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
  is a public, *simulated* mobile-money dataset (Lopez-Rojas et al., 2016; CC BY-SA 4.0), calibrated
  on real transaction logs. Each scenario draws from a published pool, for example "PaySim rows
  flagged `isFraud`" or "non-fraud transfers of $8k–$10k". PaySim's fraud flags are never shown to a
  model.
- **The 19 signals are generated here.** Session, device, identity and velocity signals are drawn
  from distributions written for each scenario.
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

## How synthetic is this?

Mostly. Public fraud data labelled by typology (mule vs. takeover vs. synthetic identity vs.
structuring) is essentially unavailable, so the labels and signals here are simulated by design. To
show how much each part carries, the supervised model is retrained on one part of the input at a time
([`notebooks/01_dataset_analysis.ipynb`](notebooks/01_dataset_analysis.ipynb), section 5):

| Supervised model trained on | Test accuracy |
|---|---:|
| All 28 fields | 97.6% |
| The 19 generated signals only | 92.6% |
| The 9 PaySim transaction fields only | 66.6% |

So the generated signals decide most labels; PaySim contributes realistic transaction shapes. One
PaySim property also matters: 97% of its fraud rows move exactly the full balance, which no other
pool does. Every model can use that cue, and Jev is more accurate on alerts built from PaySim fraud
rows (96%) than on partial-drain alerts (73%).

What the benchmark supports: a like-for-like comparison of models on the same fully specified task
(accuracy, calibration, latency, cost, repeatability). What it does not support: claims about
fraud-detection rates at a real institution.

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

- **Simulated data.** The signals and labels are generated here and the transactions come from a
  simulator (see [How synthetic is this?](#how-synthetic-is-this)). Results describe this task, not
  production fraud-detection rates.
- **The hold band is untuned.** With TypeSafe's suggested 0.30 / 0.70 band, Jev sends 207 of 500
  alerts (41%) to an analyst. A band tuned on the `dev` split would likely review fewer.
- **Sample size.** 500 test alerts give per-typology F1 intervals of several points; the confidence
  intervals are reported.
- **GPT-4o-mini is the agreed baseline,** not the newest small LLM. Other models can be added as
  clients in `triage/clients.py`.
- **PaySim quirks.** Amounts are in simulated local currency, treated here as USD. Some balance
  fields are inconsistent in the original data, and those quirks are kept.
- **Latency depends on where the client runs.** Each report records the client region; tail
  latencies from 500 calls carry a confidence interval, also reported.
- **GPT-4o-mini probabilities come from the first token of the label.** Labels that share a first
  token split that token's probability equally. The average mass mapped to a label is reported.

## How this was built

Built with AI coding assistance (Claude). The design, the experiments, the review of results and the
decisions documented above (including the v1 → v2 correction) are by Anuar Sultan.

## License

Code: [MIT](LICENSE). The alert files in `data/` contain rows from PaySim and are shared under
[CC BY-SA 4.0](data/LICENSE.md), as PaySim's license requires.

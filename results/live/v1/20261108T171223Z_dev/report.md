# Fraud alert triage: Jev vs GPT-4o-mini

| Run |  |
| --- | --- |
| Timestamp (UTC) | 2026-11-08T17:12:23+00:00 |
| Dataset | `data/alerts_dev.jsonl` · n=10 · sha256 `742eb3e6cc13` · paysim |
| Criteria sha256 | `3ef0b060f246` |
| Models served | jev: jev-1.14.0; gpt-4o-mini: gpt-4o-mini-2024-07-18 |
| Concurrency / model | 8 |
| Client region | unspecified |
| Git | `none` |

## Headline (settle when p(predicted class) ≥ 0.85)

| Metric | Jev | GPT-4o-mini | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: |
| Macro F1 [95% CI] | 0.720 [0.260, 0.800] | 0.073 [0.000, 0.133] | 0.782 [0.360, 0.800] | 0.567 [0.280, 0.600] |
| Accuracy | 80.0% | 20.0% | 90.0% | 80.0% |
| Settled automatically | 5/10 (50.0%) | 10/10 (100.0%) | 9/10 (90.0%) | – |
| Accuracy of settled | 100.0% | 20.0% | 100.0% | – |
| Fraud auto-cleared as NORMAL | 0 | 0 | 0 | – |
| P50 / P90 / P99 latency (ms) | 389 / **403** / 404 | 1647 / **1891** / 1969 | – | – |
| P90 95% CI (ms) | [395, 404] | [1672, 1977] | – | – |
| Cost per 1,000 alerts (USD) | $0.0450 | $0.1149 | – | – |
| Failed calls | 0 | 0 | – | – |

### Paired differences (bootstrap over the same alerts)

| Comparison | Δ macro F1 95% CI | Δ accuracy 95% CI | P(first better) |
| --- | ---: | ---: | ---: |
| Jev − GPT-4o-mini | [0.165, 0.733] | [30.0%, 90.0%] | 100% |
| Jev − Analyst rules (floor) | [-0.145, 0.246] | [-30.0%, 30.0%] | 34% |
| Jev − Supervised ref. (ceiling) | [-0.175, 0.000] | [-30.0%, 0.0%] | 0% |
| GPT-4o-mini − Analyst rules (floor) | [-0.548, -0.189] | [-90.0%, -30.0%] | 0% |
| GPT-4o-mini − Supervised ref. (ceiling) | [-0.756, -0.250] | [-100.0%, -40.0%] | 0% |

## F1 per typology

| Typology | n | Jev | GPT-4o-mini | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `NORMAL` | 5 | 0.800 (P 0.80, R 0.80) | 0.000 (P 0.00, R 0.00) | 0.909 (P 0.83, R 1.00) | 0.833 (P 0.71, R 1.00) |
| `CASH_OUT_FRAUD` | 2 | 0.800 (P 0.67, R 1.00) | 0.364 (P 0.22, R 1.00) | 1.000 (P 1.00, R 1.00) | 1.000 (P 1.00, R 1.00) |
| `ACCOUNT_TAKEOVER` | 1 | 0.000 (P 0.00, R 0.00) | 0.000 (P 0.00, R 0.00) | 0.000 (P 0.00, R 0.00) | 0.000 (P 0.00, R 0.00) |
| `SYNTHETIC_ID` | 1 | 1.000 (P 1.00, R 1.00) | 0.000 (P 0.00, R 0.00) | 1.000 (P 1.00, R 1.00) | 1.000 (P 1.00, R 1.00) |
| `STRUCTURING_AML` | 1 | 1.000 (P 1.00, R 1.00) | 0.000 (P 0.00, R 0.00) | 1.000 (P 1.00, R 1.00) | 0.000 (P 0.00, R 0.00) |

## Clear vs ambiguous alerts

Ambiguous = reference-model probability of the true class < 0.9. These alerts are hard by construction; no model should be confidently right on all of them.

| Cohort | n | Jev | GPT-4o-mini | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ambiguous | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| clear | 9 | 88.9% | 22.2% | 100.0% | 88.9% |

## Settlement curve (coverage / accuracy of settled / fraud auto-cleared)

| τ | Jev | GPT-4o-mini | Supervised ref. (ceiling) |
| --- | ---: | ---: | ---: |
| 0.50 | 90% / 88.9% / 1 | 100% / 20.0% / 0 | 100% / 90.0% / 1 |
| 0.70 | 80% / 87.5% / 1 | 100% / 20.0% / 0 | 100% / 90.0% / 1 |
| 0.80 | 80% / 87.5% / 1 | 100% / 20.0% / 0 | 100% / 90.0% / 1 |
| 0.85 | 50% / 100.0% / 0 | 100% / 20.0% / 0 | 90% / 100.0% / 0 |
| 0.90 | 30% / 100.0% / 0 | 100% / 20.0% / 0 | 90% / 100.0% / 0 |
| 0.95 | 10% / 100.0% / 0 | 90% / 22.2% / 0 | 80% / 100.0% / 0 |
| 0.99 | 0% / – / 0 | 90% / 22.2% / 0 | 80% / 100.0% / 0 |

## Calibration of typology probabilities

| Metric | Jev | GPT-4o-mini | Supervised ref. (ceiling) |
| --- | ---: | ---: | ---: |
| Multi-class Brier (↓) | 0.263 | 1.575 | 0.130 |
| Log loss (↓) | 0.480 | 8.807 | 0.173 |
| Top-label ECE, 10 bins (↓) | 0.134 | 0.793 | 0.088 |
| Probability source | native | logprobs | supervised |

## Hold / release gate (`noul`)

Release if p(hold) ≤ 0.30, hold if ≥ 0.70, otherwise send to an analyst.

| Metric | Jev | GPT-4o-mini |
| --- | ---: | ---: |
| ROC AUC | 0.780 | 0.920 |
| Brier (↓) | 0.193 | 0.500 |
| Accuracy at 0.5 | 70.0% | 50.0% |
| Released | 2 | 0 |
| Held | 3 | 10 |
| Sent to review | 5 | 0 |
| Fraud released | 0 | 0 |
| Legitimate held | 1 | 5 |

## Accuracy by scenario

| Scenario | n | Jev | GPT-4o-mini | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ACCOUNT_TAKEOVER/domestic_proxy` | 1 | 0% | 0% | 0% | 0% |
| `CASH_OUT_FRAUD/mule_known_session` | 2 | 100% | 100% | 100% | 100% |
| `NORMAL/routine` | 4 | 100% | 0% | 100% | 100% |
| `NORMAL/traveler` | 1 | 0% | 0% | 100% | 100% |
| `STRUCTURING_AML/light` | 1 | 100% | 0% | 100% | 0% |
| `SYNTHETIC_ID/bust_out` | 1 | 100% | 0% | 100% | 100% |

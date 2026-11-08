# Fraud alert triage: Jev vs GPT-4o-mini

| Run |  |
| --- | --- |
| Timestamp (UTC) | 2026-11-08T19:29:53+00:00 |
| Dataset | `data/alerts_test.jsonl` · n=150 · sha256 `be1f0c3a1eca` · paysim |
| Criteria sha256 | `a32857963e83` |
| Models served | jev: jev-1.14.0; gpt-4o-mini: gpt-4o-mini-2024-07-18; gpt-4o-mini-reasoning: gpt-4o-mini-2024-07-18 |
| Concurrency / model | 1 |
| Client region | Almaty, KZ |
| Git | `16a8786111` (dirty) |

## Headline (settle when p(predicted class) ≥ 0.85)

| Metric | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Macro F1 [95% CI] | 0.862 [0.806, 0.910] | 0.447 [0.370, 0.523] | 0.617 [0.531, 0.694] | 0.955 [0.921, 0.982] | 0.832 [0.768, 0.885] |
| Accuracy | 84.7% | 44.7% | 60.0% | 94.7% | 82.7% |
| Settled automatically | 62/150 (41.3%) | 93/150 (62.0%) | 128/150 (85.3%) | 132/150 (88.0%) | – |
| Accuracy of settled | 96.8% | 49.5% | 57.8% | 99.2% | – |
| Fraud auto-cleared as NORMAL | 0 | 0 | 2 | 0 | – |
| P50 / P90 / P99 latency (ms) | 300 / **371** / 509 | 924 / **1106** / 1516 | 3249 / **3907** / 4840 | – | – |
| P90 95% CI (ms) | [337, 405] | [1072, 1213] | [3749, 4042] | – | – |
| Cost per 1,000 alerts (USD) | $0.0447 | $0.1129 | $0.1890 | – | – |
| Failed calls | 0 | 0 | 0 | – | – |

### Paired differences (bootstrap over the same alerts)

| Comparison | Δ macro F1 95% CI | Δ accuracy 95% CI | P(first better) |
| --- | ---: | ---: | ---: |
| Jev − GPT-4o-mini (label only) | [0.335, 0.491] | [32.0%, 48.0%] | 100% |
| Jev − GPT-4o-mini (reasoning first) | [0.170, 0.329] | [16.7%, 32.7%] | 100% |
| Jev − Analyst rules (floor) | [-0.030, 0.099] | [-4.0%, 8.7%] | 70% |
| Jev − Supervised ref. (ceiling) | [-0.154, -0.036] | [-16.0%, -4.0%] | 0% |
| GPT-4o-mini (label only) − GPT-4o-mini (reasoning first) | [-0.253, -0.081] | [-22.7%, -7.3%] | 0% |
| GPT-4o-mini (label only) − Analyst rules (floor) | [-0.477, -0.276] | [-47.3%, -27.3%] | 0% |
| GPT-4o-mini (label only) − Supervised ref. (ceiling) | [-0.589, -0.429] | [-58.0%, -41.3%] | 0% |
| GPT-4o-mini (reasoning first) − Analyst rules (floor) | [-0.307, -0.123] | [-32.0%, -12.7%] | 0% |
| GPT-4o-mini (reasoning first) − Supervised ref. (ceiling) | [-0.422, -0.263] | [-42.7%, -26.7%] | 0% |

## F1 per typology

| Typology | n | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `NORMAL` | 69 | 0.829 (P 0.94, R 0.74) | 0.376 (P 1.00, R 0.23) | 0.560 (P 0.90, R 0.41) | 0.941 (P 0.96, R 0.93) | 0.832 (P 0.78, R 0.90) |
| `MONEY_MULE` | 18 | 0.837 (P 0.72, R 1.00) | 0.275 (P 0.18, R 0.61) | 0.464 (P 0.34, R 0.72) | 1.000 (P 1.00, R 1.00) | 0.941 (P 1.00, R 0.89) |
| `ACCOUNT_TAKEOVER` | 23 | 0.692 (P 0.62, R 0.78) | 0.667 (P 0.58, R 0.78) | 0.606 (P 0.47, R 0.87) | 0.833 (P 0.80, R 0.87) | 0.571 (P 0.63, R 0.52) |
| `SYNTHETIC_ID` | 21 | 0.977 (P 0.95, R 1.00) | 0.250 (P 1.00, R 0.14) | 0.645 (P 1.00, R 0.48) | 1.000 (P 1.00, R 1.00) | 0.895 (P 1.00, R 0.81) |
| `STRUCTURING_AML` | 19 | 0.974 (P 0.95, R 1.00) | 0.667 (P 0.50, R 1.00) | 0.809 (P 0.68, R 1.00) | 1.000 (P 1.00, R 1.00) | 0.919 (P 0.94, R 0.89) |

## Clear vs ambiguous alerts

Ambiguous = reference-model probability of the true class < 0.9. These alerts are hard by construction; no model should be confidently right on all of them.

| Cohort | n | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ambiguous | 21 | 57.1% | 42.9% | 47.6% | 61.9% | 33.3% |
| clear | 129 | 89.1% | 45.0% | 62.0% | 100.0% | 90.7% |

## Settlement curve (coverage / accuracy of settled / fraud auto-cleared)

| τ | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) | Supervised ref. (ceiling) |
| --- | ---: | ---: | ---: | ---: |
| 0.50 | 92% / 87.7% / 2 | 94% / 42.6% / 0 | 94% / 58.2% / 3 | 99% / 95.3% / 2 |
| 0.70 | 65% / 95.9% / 0 | 77% / 45.2% / 0 | 89% / 58.2% / 3 | 93% / 97.8% / 0 |
| 0.80 | 51% / 96.1% / 0 | 68% / 45.1% / 0 | 87% / 58.5% / 2 | 89% / 99.2% / 0 |
| 0.85 | 41% / 96.8% / 0 | 62% / 49.5% / 0 | 85% / 57.8% / 2 | 88% / 99.2% / 0 |
| 0.90 | 24% / 97.2% / 0 | 55% / 52.4% / 0 | 83% / 59.2% / 1 | 87% / 99.2% / 0 |
| 0.95 | 10% / 100.0% / 0 | 49% / 55.4% / 0 | 80% / 60.0% / 1 | 85% / 99.2% / 0 |
| 0.99 | 0% / – / 0 | 33% / 70.0% / 0 | 74% / 61.3% / 1 | 79% / 100.0% / 0 |

## Calibration of typology probabilities

| Metric | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) | Supervised ref. (ceiling) |
| --- | ---: | ---: | ---: | ---: |
| Multi-class Brier (↓) | 0.249 | 0.961 | 0.806 | 0.066 |
| Log loss (↓) | 0.536 | 4.310 | 4.848 | 0.106 |
| Top-label ECE, 10 bins (↓) | 0.092 | 0.440 | 0.395 | 0.023 |
| Probability source | native | logprobs | logprobs | supervised |

## Hold / release gate (`noul`)

Release if p(hold) ≤ 0.30, hold if ≥ 0.70, otherwise send to an analyst.

| Metric | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) |
| --- | ---: | ---: | ---: |
| ROC AUC | 0.849 | 0.647 | 0.771 |
| Brier (↓) | 0.167 | 0.353 | 0.293 |
| Accuracy at 0.5 | 76.0% | 64.7% | 70.7% |
| Released | 21 | 16 | 31 |
| Held | 65 | 134 | 119 |
| Sent to review | 64 | 0 | 0 |
| Fraud released | 1 | 0 | 3 |
| Legitimate held | 11 | 53 | 41 |

## Accuracy by scenario

| Scenario | n | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ACCOUNT_TAKEOVER/domestic_proxy` | 12 | 83% | 83% | 100% | 92% | 58% |
| `ACCOUNT_TAKEOVER/foreign_login` | 6 | 100% | 83% | 83% | 100% | 83% |
| `ACCOUNT_TAKEOVER/session_hijack` | 5 | 40% | 60% | 60% | 60% | 0% |
| `MONEY_MULE/mule_known_session` | 13 | 100% | 54% | 85% | 100% | 92% |
| `MONEY_MULE/mule_new_phone` | 5 | 100% | 80% | 40% | 100% | 80% |
| `NORMAL/large_payment` | 9 | 67% | 22% | 78% | 100% | 100% |
| `NORMAL/new_customer` | 2 | 50% | 0% | 50% | 100% | 100% |
| `NORMAL/new_device` | 6 | 50% | 0% | 0% | 100% | 67% |
| `NORMAL/new_device_large_payment` | 6 | 33% | 0% | 0% | 17% | 33% |
| `NORMAL/payroll_batch` | 1 | 0% | 0% | 0% | 100% | 0% |
| `NORMAL/routine` | 32 | 94% | 44% | 62% | 100% | 100% |
| `NORMAL/traveler` | 10 | 70% | 0% | 0% | 100% | 100% |
| `NORMAL/traveler_no_notice_large` | 3 | 67% | 0% | 0% | 100% | 100% |
| `STRUCTURING_AML/classic` | 15 | 100% | 100% | 100% | 100% | 100% |
| `STRUCTURING_AML/light` | 4 | 100% | 100% | 100% | 100% | 50% |
| `SYNTHETIC_ID/bust_out` | 14 | 100% | 0% | 43% | 100% | 100% |
| `SYNTHETIC_ID/sleeper` | 7 | 100% | 43% | 57% | 100% | 43% |

# Fraud alert triage: Jev vs GPT-4o-mini

| Run |  |
| --- | --- |
| Timestamp (UTC) | 2026-11-08T18:00:07+00:00 |
| Dataset | `data/alerts_test.jsonl` · n=150 · sha256 `3482e6c42ec4` · paysim |
| Criteria sha256 | `3ef0b060f246` |
| Models served | jev: jev-1.14.0; gpt-4o-mini: gpt-4o-mini-2024-07-18 |
| Concurrency / model | 1 |
| Client region | unspecified |
| Git | `none` |

## Headline (settle when p(predicted class) ≥ 0.85)

| Metric | Jev | GPT-4o-mini | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: |
| Macro F1 [95% CI] | 0.824 [0.760, 0.879] | 0.286 [0.203, 0.367] | 0.955 [0.921, 0.982] | 0.832 [0.768, 0.885] |
| Accuracy | 81.3% | 28.7% | 94.7% | 82.7% |
| Settled automatically | 59/150 (39.3%) | 114/150 (76.0%) | 132/150 (88.0%) | – |
| Accuracy of settled | 98.3% | 23.7% | 99.2% | – |
| Fraud auto-cleared as NORMAL | 0 | 0 | 0 | – |
| P50 / P90 / P99 latency (ms) | 306 / **409** / 987 | 991 / **1440** / 1781 | – | – |
| P90 95% CI (ms) | [360, 411] | [1262, 1534] | – | – |
| Cost per 1,000 alerts (USD) | $0.0448 | $0.1143 | – | – |
| Failed calls | 0 | 0 | – | – |

### Paired differences (bootstrap over the same alerts)

| Comparison | Δ macro F1 95% CI | Δ accuracy 95% CI | P(first better) |
| --- | ---: | ---: | ---: |
| Jev − GPT-4o-mini | [0.456, 0.619] | [44.7%, 60.7%] | 100% |
| Jev − Analyst rules (floor) | [-0.064, 0.056] | [-7.3%, 5.3%] | 30% |
| Jev − Supervised ref. (ceiling) | [-0.197, -0.074] | [-20.0%, -7.3%] | 0% |
| GPT-4o-mini − Analyst rules (floor) | [-0.639, -0.450] | [-62.7%, -44.7%] | 0% |
| GPT-4o-mini − Supervised ref. (ceiling) | [-0.754, -0.583] | [-73.4%, -58.0%] | 0% |

## F1 per typology

| Typology | n | Jev | GPT-4o-mini | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `NORMAL` | 69 | 0.817 (P 0.96, R 0.71) | 0.296 (P 1.00, R 0.17) | 0.941 (P 0.96, R 0.93) | 0.832 (P 0.78, R 0.90) |
| `CASH_OUT_FRAUD` | 18 | 0.708 (P 0.57, R 0.94) | 0.254 (P 0.15, R 0.89) | 1.000 (P 1.00, R 1.00) | 0.941 (P 1.00, R 0.89) |
| `ACCOUNT_TAKEOVER` | 23 | 0.667 (P 0.61, R 0.74) | 0.378 (P 0.50, R 0.30) | 0.833 (P 0.80, R 0.87) | 0.571 (P 0.63, R 0.52) |
| `SYNTHETIC_ID` | 21 | 0.952 (P 0.95, R 0.95) | 0.091 (P 1.00, R 0.05) | 1.000 (P 1.00, R 1.00) | 0.895 (P 1.00, R 0.81) |
| `STRUCTURING_AML` | 19 | 0.974 (P 0.95, R 1.00) | 0.412 (P 0.47, R 0.37) | 1.000 (P 1.00, R 1.00) | 0.919 (P 0.94, R 0.89) |

## Clear vs ambiguous alerts

Ambiguous = reference-model probability of the true class < 0.9. These alerts are hard by construction; no model should be confidently right on all of them.

| Cohort | n | Jev | GPT-4o-mini | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ambiguous | 21 | 47.6% | 19.0% | 61.9% | 33.3% |
| clear | 129 | 86.8% | 30.2% | 100.0% | 90.7% |

## Settlement curve (coverage / accuracy of settled / fraud auto-cleared)

| τ | Jev | GPT-4o-mini | Supervised ref. (ceiling) |
| --- | ---: | ---: | ---: |
| 0.50 | 87% / 87.0% / 1 | 98% / 27.9% / 0 | 99% / 95.3% / 2 |
| 0.70 | 65% / 92.8% / 0 | 87% / 25.2% / 0 | 93% / 97.8% / 0 |
| 0.80 | 50% / 96.0% / 0 | 77% / 24.1% / 0 | 89% / 99.2% / 0 |
| 0.85 | 39% / 98.3% / 0 | 76% / 23.7% / 0 | 88% / 99.2% / 0 |
| 0.90 | 28% / 97.6% / 0 | 70% / 23.8% / 0 | 87% / 99.2% / 0 |
| 0.95 | 12% / 94.4% / 0 | 65% / 23.5% / 0 | 85% / 99.2% / 0 |
| 0.99 | 0% / – / 0 | 57% / 17.6% / 0 | 79% / 100.0% / 0 |

## Calibration of typology probabilities

| Metric | Jev | GPT-4o-mini | Supervised ref. (ceiling) |
| --- | ---: | ---: | ---: |
| Multi-class Brier (↓) | 0.280 | 1.343 | 0.066 |
| Log loss (↓) | 0.576 | 6.923 | 0.106 |
| Top-label ECE, 10 bins (↓) | 0.094 | 0.627 | 0.023 |
| Probability source | native | logprobs | supervised |

## Hold / release gate (`noul`)

Release if p(hold) ≤ 0.30, hold if ≥ 0.70, otherwise send to an analyst.

| Metric | Jev | GPT-4o-mini |
| --- | ---: | ---: |
| ROC AUC | 0.853 | 0.711 |
| Brier (↓) | 0.167 | 0.380 |
| Accuracy at 0.5 | 75.3% | 62.0% |
| Released | 21 | 12 |
| Held | 67 | 138 |
| Sent to review | 62 | 0 |
| Fraud released | 3 | 0 |
| Legitimate held | 11 | 57 |

## Accuracy by scenario

| Scenario | n | Jev | GPT-4o-mini | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ACCOUNT_TAKEOVER/domestic_proxy` | 12 | 83% | 25% | 92% | 58% |
| `ACCOUNT_TAKEOVER/foreign_login` | 6 | 100% | 50% | 100% | 83% |
| `ACCOUNT_TAKEOVER/session_hijack` | 5 | 20% | 20% | 60% | 0% |
| `CASH_OUT_FRAUD/mule_known_session` | 13 | 100% | 85% | 100% | 92% |
| `CASH_OUT_FRAUD/mule_new_phone` | 5 | 80% | 100% | 100% | 80% |
| `NORMAL/large_payment` | 9 | 67% | 11% | 100% | 100% |
| `NORMAL/new_customer` | 2 | 50% | 0% | 100% | 100% |
| `NORMAL/new_device` | 6 | 50% | 0% | 100% | 67% |
| `NORMAL/new_device_large_payment` | 6 | 33% | 0% | 17% | 33% |
| `NORMAL/payroll_batch` | 1 | 0% | 0% | 100% | 0% |
| `NORMAL/routine` | 32 | 91% | 34% | 100% | 100% |
| `NORMAL/traveler` | 10 | 70% | 0% | 100% | 100% |
| `NORMAL/traveler_no_notice_large` | 3 | 33% | 0% | 100% | 100% |
| `STRUCTURING_AML/classic` | 15 | 100% | 40% | 100% | 100% |
| `STRUCTURING_AML/light` | 4 | 100% | 25% | 100% | 50% |
| `SYNTHETIC_ID/bust_out` | 14 | 100% | 0% | 100% | 100% |
| `SYNTHETIC_ID/sleeper` | 7 | 86% | 14% | 100% | 43% |

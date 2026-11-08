# Fraud alert triage: Jev vs GPT-4o-mini

| Run |  |
| --- | --- |
| Timestamp (UTC) | 2026-11-08T17:20:19+00:00 |
| Dataset | `data/alerts_test.jsonl` · n=500 · sha256 `3482e6c42ec4` · paysim |
| Criteria sha256 | `3ef0b060f246` |
| Models served | jev: jev-1.14.0; gpt-4o-mini: gpt-4o-mini-2024-07-18 |
| Concurrency / model | 8 |
| Client region | unspecified |
| Git | `none` |

## Headline (settle when p(predicted class) ≥ 0.85)

| Metric | Jev | GPT-4o-mini | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: |
| Macro F1 [95% CI] | 0.861 [0.831, 0.888] | 0.246 [0.207, 0.284] | 0.960 [0.942, 0.975] | 0.831 [0.797, 0.861] |
| Accuracy | 85.4% | 27.0% | 95.6% | 82.8% |
| Settled automatically | 225/500 (45.0%) | 382/500 (76.4%) | 456/500 (91.2%) | – |
| Accuracy of settled | 97.8% | 23.3% | 98.9% | – |
| Fraud auto-cleared as NORMAL | 0 | 0 | 0 | – |
| P50 / P90 / P99 latency (ms) | 289 / **354** / 904 | 999 / **1316** / 1842 | – | – |
| P90 95% CI (ms) | [337, 381] | [1256, 1350] | – | – |
| Cost per 1,000 alerts (USD) | $0.0448 | $0.1144 | – | – |
| Failed calls | 0 | 0 | – | – |

### Paired differences (bootstrap over the same alerts)

| Comparison | Δ macro F1 95% CI | Δ accuracy 95% CI | P(first better) |
| --- | ---: | ---: | ---: |
| Jev − GPT-4o-mini | [0.573, 0.656] | [54.2%, 62.8%] | 100% |
| Jev − Analyst rules (floor) | [-0.005, 0.065] | [-1.0%, 6.2%] | 92% |
| Jev − Supervised ref. (ceiling) | [-0.129, -0.071] | [-13.4%, -7.2%] | 0% |
| GPT-4o-mini − Analyst rules (floor) | [-0.630, -0.539] | [-60.4%, -50.8%] | 0% |
| GPT-4o-mini − Supervised ref. (ceiling) | [-0.755, -0.673] | [-72.8%, -64.2%] | 0% |

## F1 per typology

| Typology | n | Jev | GPT-4o-mini | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `NORMAL` | 200 | 0.847 (P 0.96, R 0.76) | 0.246 (P 1.00, R 0.14) | 0.945 (P 0.95, R 0.94) | 0.813 (P 0.73, R 0.93) |
| `CASH_OUT_FRAUD` | 75 | 0.764 (P 0.63, R 0.97) | 0.292 (P 0.18, R 0.87) | 0.993 (P 1.00, R 0.99) | 0.937 (P 0.99, R 0.89) |
| `ACCOUNT_TAKEOVER` | 75 | 0.779 (P 0.78, R 0.77) | 0.367 (P 0.59, R 0.27) | 0.887 (P 0.88, R 0.89) | 0.635 (P 0.78, R 0.53) |
| `SYNTHETIC_ID` | 75 | 0.952 (P 0.99, R 0.92) | 0.026 (P 0.50, R 0.01) | 1.000 (P 1.00, R 1.00) | 0.874 (P 0.98, R 0.79) |
| `STRUCTURING_AML` | 75 | 0.962 (P 0.93, R 1.00) | 0.298 (P 0.32, R 0.28) | 0.974 (P 0.96, R 0.99) | 0.894 (P 0.95, R 0.84) |

## Clear vs ambiguous alerts

Ambiguous = reference-model probability of the true class < 0.9. These alerts are hard by construction; no model should be confidently right on all of them.

| Cohort | n | Jev | GPT-4o-mini | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ambiguous | 55 | 52.7% | 16.4% | 60.0% | 38.2% |
| clear | 445 | 89.4% | 28.3% | 100.0% | 88.3% |

## Settlement curve (coverage / accuracy of settled / fraud auto-cleared)

| τ | Jev | GPT-4o-mini | Supervised ref. (ceiling) |
| --- | ---: | ---: | ---: |
| 0.50 | 91% / 89.5% / 2 | 98% / 26.5% / 0 | 100% / 95.8% / 9 |
| 0.70 | 71% / 94.9% / 0 | 86% / 24.4% / 0 | 96% / 97.1% / 5 |
| 0.80 | 55% / 96.8% / 0 | 80% / 24.6% / 0 | 92% / 98.7% / 0 |
| 0.85 | 45% / 97.8% / 0 | 76% / 23.3% / 0 | 91% / 98.9% / 0 |
| 0.90 | 28% / 97.1% / 0 | 72% / 23.5% / 0 | 90% / 99.3% / 0 |
| 0.95 | 11% / 98.2% / 0 | 65% / 22.8% / 0 | 87% / 99.5% / 0 |
| 0.99 | 0% / 100.0% / 0 | 55% / 17.6% / 0 | 79% / 100.0% / 0 |

## Calibration of typology probabilities

| Metric | Jev | GPT-4o-mini | Supervised ref. (ceiling) |
| --- | ---: | ---: | ---: |
| Multi-class Brier (↓) | 0.227 | 1.348 | 0.061 |
| Log loss (↓) | 0.452 | 6.623 | 0.097 |
| Top-label ECE, 10 bins (↓) | 0.099 | 0.638 | 0.016 |
| Probability source | native | logprobs | supervised |

## Hold / release gate (`noul`)

Release if p(hold) ≤ 0.30, hold if ≥ 0.70, otherwise send to an analyst.

| Metric | Jev | GPT-4o-mini |
| --- | ---: | ---: |
| ROC AUC | 0.889 | 0.705 |
| Brier (↓) | 0.141 | 0.344 |
| Accuracy at 0.5 | 82.6% | 65.6% |
| Released | 67 | 28 |
| Held | 232 | 472 |
| Sent to review | 201 | 0 |
| Fraud released | 4 | 0 |
| Legitimate held | 25 | 172 |

## Accuracy by scenario

| Scenario | n | Jev | GPT-4o-mini | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ACCOUNT_TAKEOVER/domestic_proxy` | 25 | 80% | 20% | 84% | 60% |
| `ACCOUNT_TAKEOVER/foreign_login` | 32 | 100% | 41% | 100% | 78% |
| `ACCOUNT_TAKEOVER/session_hijack` | 18 | 33% | 11% | 78% | 0% |
| `CASH_OUT_FRAUD/mule_known_session` | 61 | 98% | 85% | 98% | 90% |
| `CASH_OUT_FRAUD/mule_new_phone` | 14 | 93% | 93% | 100% | 86% |
| `NORMAL/large_payment` | 19 | 58% | 16% | 100% | 95% |
| `NORMAL/new_customer` | 12 | 75% | 8% | 100% | 92% |
| `NORMAL/new_device` | 19 | 58% | 0% | 100% | 79% |
| `NORMAL/new_device_large_payment` | 10 | 40% | 0% | 20% | 40% |
| `NORMAL/payroll_batch` | 6 | 0% | 0% | 50% | 50% |
| `NORMAL/routine` | 108 | 90% | 22% | 100% | 100% |
| `NORMAL/traveler` | 21 | 86% | 0% | 100% | 100% |
| `NORMAL/traveler_no_notice_large` | 5 | 40% | 0% | 80% | 100% |
| `STRUCTURING_AML/classic` | 56 | 100% | 32% | 100% | 100% |
| `STRUCTURING_AML/light` | 19 | 100% | 16% | 95% | 37% |
| `SYNTHETIC_ID/bust_out` | 46 | 100% | 0% | 100% | 100% |
| `SYNTHETIC_ID/sleeper` | 29 | 79% | 3% | 100% | 45% |

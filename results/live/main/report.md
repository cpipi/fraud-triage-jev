# Fraud alert triage: Jev vs GPT-4o-mini

| Run |  |
| --- | --- |
| Timestamp (UTC) | 2026-11-08T19:25:55+00:00 |
| Dataset | `data/alerts_test.jsonl` · n=500 · sha256 `be1f0c3a1eca` · paysim |
| Criteria sha256 | `a32857963e83` |
| Models served | jev: jev-1.14.0; gpt-4o-mini: gpt-4o-mini-2024-07-18; gpt-4o-mini-reasoning: gpt-4o-mini-2024-07-18 |
| Concurrency / model | 8 |
| Client region | Almaty, KZ |
| Git | `4d9c4cda12` |

## Headline (settle when p(predicted class) ≥ 0.85)

| Metric | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Macro F1 [95% CI] | 0.891 [0.864, 0.914] | 0.474 [0.431, 0.514] | 0.577 [0.531, 0.619] | 0.960 [0.942, 0.975] | 0.831 [0.797, 0.861] |
| Accuracy | 88.4% | 47.8% | 58.2% | 95.6% | 82.8% |
| Settled automatically | 237/500 (47.4%) | 326/500 (65.2%) | 430/500 (86.0%) | 456/500 (91.2%) | – |
| Accuracy of settled | 96.6% | 55.2% | 59.3% | 98.9% | – |
| Fraud auto-cleared as NORMAL | 0 | 0 | 17 | 0 | – |
| P50 / P90 / P99 latency (ms) | 284 / **336** / 448 | 855 / **1153** / 1633 | 3127 / **3890** / 4701 | – | – |
| P90 95% CI (ms) | [321, 352] | [1084, 1214] | [3794, 3956] | – | – |
| Cost per 1,000 alerts (USD) | $0.0447 | $0.1130 | $0.1899 | – | – |
| Failed calls | 0 | 0 | 0 | – | – |

### Paired differences (bootstrap over the same alerts)

| Comparison | Δ macro F1 95% CI | Δ accuracy 95% CI | P(first better) |
| --- | ---: | ---: | ---: |
| Jev − GPT-4o-mini (label only) | [0.374, 0.461] | [35.8%, 45.0%] | 100% |
| Jev − GPT-4o-mini (reasoning first) | [0.270, 0.356] | [25.8%, 34.6%] | 100% |
| Jev − Analyst rules (floor) | [0.027, 0.093] | [2.0%, 9.0%] | 100% |
| Jev − Supervised ref. (ceiling) | [-0.098, -0.042] | [-10.0%, -4.4%] | 0% |
| GPT-4o-mini (label only) − GPT-4o-mini (reasoning first) | [-0.156, -0.050] | [-14.8%, -5.8%] | 0% |
| GPT-4o-mini (label only) − Analyst rules (floor) | [-0.410, -0.303] | [-40.4%, -29.2%] | 0% |
| GPT-4o-mini (label only) − Supervised ref. (ceiling) | [-0.529, -0.443] | [-52.2%, -43.0%] | 0% |
| GPT-4o-mini (reasoning first) − Analyst rules (floor) | [-0.304, -0.203] | [-29.6%, -19.4%] | 0% |
| GPT-4o-mini (reasoning first) − Supervised ref. (ceiling) | [-0.430, -0.340] | [-42.0%, -33.2%] | 0% |

## F1 per typology

| Typology | n | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `NORMAL` | 200 | 0.866 (P 0.95, R 0.80) | 0.340 (P 1.00, R 0.20) | 0.562 (P 0.81, R 0.43) | 0.945 (P 0.95, R 0.94) | 0.813 (P 0.73, R 0.93) |
| `MONEY_MULE` | 75 | 0.839 (P 0.74, R 0.97) | 0.364 (P 0.25, R 0.69) | 0.406 (P 0.32, R 0.55) | 0.993 (P 1.00, R 0.99) | 0.937 (P 0.99, R 0.89) |
| `ACCOUNT_TAKEOVER` | 75 | 0.821 (P 0.79, R 0.85) | 0.698 (P 0.63, R 0.79) | 0.667 (P 0.55, R 0.85) | 0.887 (P 0.88, R 0.89) | 0.635 (P 0.78, R 0.53) |
| `SYNTHETIC_ID` | 75 | 0.966 (P 0.99, R 0.95) | 0.273 (P 0.92, R 0.16) | 0.500 (P 1.00, R 0.33) | 1.000 (P 1.00, R 1.00) | 0.874 (P 0.98, R 0.79) |
| `STRUCTURING_AML` | 75 | 0.962 (P 0.93, R 1.00) | 0.694 (P 0.53, R 1.00) | 0.750 (P 0.60, R 1.00) | 0.974 (P 0.96, R 0.99) | 0.894 (P 0.95, R 0.84) |

## Clear vs ambiguous alerts

Ambiguous = reference-model probability of the true class < 0.9. These alerts are hard by construction; no model should be confidently right on all of them.

| Cohort | n | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ambiguous | 55 | 61.8% | 45.5% | 50.9% | 60.0% | 38.2% |
| clear | 445 | 91.7% | 48.1% | 59.1% | 100.0% | 88.3% |

## Settlement curve (coverage / accuracy of settled / fraud auto-cleared)

| τ | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) | Supervised ref. (ceiling) |
| --- | ---: | ---: | ---: | ---: |
| 0.50 | 93% / 90.8% / 4 | 94% / 47.8% / 0 | 97% / 57.0% / 19 | 100% / 95.8% / 9 |
| 0.70 | 74% / 96.2% / 0 | 79% / 51.0% / 0 | 90% / 57.1% / 18 | 96% / 97.1% / 5 |
| 0.80 | 58% / 96.5% / 0 | 69% / 54.0% / 0 | 87% / 59.0% / 17 | 92% / 98.7% / 0 |
| 0.85 | 47% / 96.6% / 0 | 65% / 55.2% / 0 | 86% / 59.3% / 17 | 91% / 98.9% / 0 |
| 0.90 | 26% / 97.7% / 0 | 58% / 57.7% / 0 | 84% / 59.8% / 17 | 90% / 99.3% / 0 |
| 0.95 | 11% / 96.3% / 0 | 52% / 61.2% / 0 | 82% / 60.5% / 14 | 87% / 99.5% / 0 |
| 0.99 | 1% / 100.0% / 0 | 34% / 70.9% / 0 | 76% / 62.0% / 13 | 79% / 100.0% / 0 |

## Calibration of typology probabilities

| Metric | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) | Supervised ref. (ceiling) |
| --- | ---: | ---: | ---: | ---: |
| Multi-class Brier (↓) | 0.205 | 0.885 | 0.814 | 0.061 |
| Log loss (↓) | 0.424 | 3.772 | 4.567 | 0.097 |
| Top-label ECE, 10 bins (↓) | 0.105 | 0.395 | 0.409 | 0.016 |
| Probability source | native | logprobs | logprobs | supervised |

## Hold / release gate (`noul`)

Release if p(hold) ≤ 0.30, hold if ≥ 0.70, otherwise send to an analyst.

| Metric | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) |
| --- | ---: | ---: | ---: |
| ROC AUC | 0.894 | 0.687 | 0.752 |
| Brier (↓) | 0.140 | 0.318 | 0.259 |
| Accuracy at 0.5 | 82.4% | 68.2% | 74.0% |
| Released | 64 | 41 | 101 |
| Held | 229 | 459 | 397 |
| Sent to review | 207 | 0 | 2 |
| Fraud released | 3 | 0 | 15 |
| Legitimate held | 28 | 159 | 114 |

## Accuracy by scenario

| Scenario | n | Jev | GPT-4o-mini (label only) | GPT-4o-mini (reasoning first) | Supervised ref. (ceiling) | Analyst rules (floor) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ACCOUNT_TAKEOVER/domestic_proxy` | 25 | 88% | 68% | 88% | 84% | 60% |
| `ACCOUNT_TAKEOVER/foreign_login` | 32 | 100% | 94% | 97% | 100% | 78% |
| `ACCOUNT_TAKEOVER/session_hijack` | 18 | 56% | 67% | 61% | 78% | 0% |
| `MONEY_MULE/mule_known_session` | 61 | 98% | 70% | 59% | 98% | 90% |
| `MONEY_MULE/mule_new_phone` | 14 | 93% | 64% | 36% | 100% | 86% |
| `NORMAL/large_payment` | 19 | 68% | 21% | 84% | 100% | 95% |
| `NORMAL/new_customer` | 12 | 75% | 8% | 42% | 100% | 92% |
| `NORMAL/new_device` | 19 | 63% | 0% | 0% | 100% | 79% |
| `NORMAL/new_device_large_payment` | 10 | 40% | 0% | 10% | 20% | 40% |
| `NORMAL/payroll_batch` | 6 | 0% | 0% | 0% | 50% | 50% |
| `NORMAL/routine` | 108 | 94% | 33% | 56% | 100% | 100% |
| `NORMAL/traveler` | 21 | 76% | 0% | 19% | 100% | 100% |
| `NORMAL/traveler_no_notice_large` | 5 | 60% | 0% | 0% | 80% | 100% |
| `STRUCTURING_AML/classic` | 56 | 100% | 100% | 100% | 100% | 100% |
| `STRUCTURING_AML/light` | 19 | 100% | 100% | 100% | 95% | 37% |
| `SYNTHETIC_ID/bust_out` | 46 | 100% | 11% | 46% | 100% | 100% |
| `SYNTHETIC_ID/sleeper` | 29 | 86% | 24% | 14% | 100% | 45% |

Fraud Alert Triage & TypeSafe Jev Benchmark — Project Specification

1. Objective
Build and benchmark a production-grade Fraud Alert Triage pipeline comparing
TypeSafe's Jev decision model (POST https://api.typesafe.ai/v1/systemone) against
GPT-4o-mini on speed (P50/P90/P99 latency), cost, and accuracy (F1-score per
typology and high-confidence settlement rate).

2. Dataset Specification
- Source: PaySim mobile money simulation (or any other public dataset) schema enriched with session telemetry,
  device metadata, and KYC checks.
- Reproducibility: Generated via generate_dataset.py with a fixed deterministic
  random seed (--seed 42) for a standard 250-alert dataset.
- Hard-Boundary Ratio: ~18% ambiguous or edge-case alerts to rigorously test
  calibrated probabilities and confidence thresholds (tau = 0.85).
- Fixed Typologies (5 classes):
  1. NORMAL: Routine transfers, payments, and cash-ins within user norms.
  2. CASH_OUT_FRAUD: Complete or near-total balance drain (80-100%).
  3. ACCOUNT_TAKEOVER: Foreign IP, recent password/MFA reset, high-value transfer.
  4. SYNTHETIC_ID: New account (<90 days), thin credit, SSN mismatch, utilization spike.
  5. STRUCTURING_AML: Multiple sub-threshold transfers ($8,400-$9,960) within 24h.

3. TypeSafe Jev API Integration (/v1/systemone)
Leverages Jev's native multi-question parallel execution in a single round trip:
- triage_typology (choice): Selects one of 5 fixed typologies + per-class probabilities.
- auto_block (noul): Emits a calibrated 0.0-1.0 probability gate for payment rail blocking.
- severity (score): Rates operational urgency on an ordered 4-level scale (0.0 to 3.0).

4. Evaluation Metrics & Deliverables
- Per-Typology F1-Score: Precision, recall, and F1 per class + macro/weighted averages.
- High-Confidence Settlement Rate: Percentage of alerts with confidence >= 0.85.
- Tail Latency: P50, P90, and P99 latency measured via asynchronous httpx concurrency.
- Cost Efficiency: Computed cost per 1,000 requests compared against GPT-4o-mini.
- Deliverable: Fully reproducible repository with generate_dataset.py, evaluate.py,
  and benchmark results.
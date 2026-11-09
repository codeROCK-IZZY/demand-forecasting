# ADR 2: Calibrate intervals on held-out blocks, and set stock from a quantile, not the median

## Context

Quantile models are trained to be calibrated, but in practice their raw 80% intervals cover only
about 72% of store x product outcomes here. The model is overconfident on noisy, low-volume series.
Separately, stock decisions are made from point forecasts, and a median forecast of right-skewed
demand sits below the mean. The result is chronic shortages.

## Decision

- **Conformalized quantile regression.** A model trained before six calibration blocks forecasts
  them. The nonconformity score is `max(q10 - y, y - q90) / scale`, and its finite-sample 80%
  quantile, per level of the hierarchy, widens the test-time intervals.
- **Reconciliation.** Bottom-up, WLS and MinT (shrunk error covariance from the calibration blocks)
  are all evaluated. The README reports which one wins rather than assuming MinT does.
- **Stock = the critical-ratio quantile.** With a shortage costing 4 and waste costing 1, the
  cost-minimising order is the 80th percentile of demand. Policies are compared on total cost over
  the backtest.

## Consequences

- Coverage at store x product moves from 72% to about 82% for an 80% target. At aggregate levels
  the calibrated intervals are slightly conservative.
- Ordering at the 80th percentile halves stock cost against a seasonal-naive point forecast and beats
  ordering the GBM median by about a quarter. Calibrating the 80th percentile helps coverage, but
  changes cost very little.
- Forecasts add up across stores and regions, so plans made at different levels agree.

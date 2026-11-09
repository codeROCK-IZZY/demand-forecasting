# ADR 1: One global model for every series, instead of one model per series

## Context

A chain has thousands of store-product series. Fitting a classical model (ETS, ARIMA) to each is the
traditional approach, but each series sees few promotions and holidays, a store that opened a year
ago has a short history, and slow movers are mostly zeros. Per-series models can't learn from what
the other series have seen.

## Decision

Train one gradient-boosted model on every series at every level of the hierarchy at once:

- **Scaled targets:** units divided by the series' recent mean.
- **Features known at the forecast origin only:** recent means, the same weekday last week, the same
  period last year, calendar, holidays, planned promotions and prices, plus the level, category and
  product as categorical features.
- **Direct forecasts:** one row per (series, horizon day), and no recursion, so errors don't compound
  over the four weeks.
- **Quantile loss:** separate models for the 10th, 50th, 80th and 90th percentiles. The quantiles
  are sorted after prediction so they never cross.

## Consequences

- Promotions, holidays and price effects are learned across all 136 series. Error falls by about 30%
  at store x product level against Holt-Winters fitted per series.
- The late-opening store borrows from the others and is forecast almost as well as the rest.
- A structural break (a competitor taking 40% of one product's demand in one region) is picked up
  through the recent-mean features. The baselines track it more slowly.
- A test checks that changing the future does not change any feature, which guards against leakage.

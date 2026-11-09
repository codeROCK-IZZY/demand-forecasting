# Known limitations

- **Synthetic sales.** The chain, its calendar and its demand are generated, with realistic structure
  (seasonality by category, planned promotions, holidays, price elasticity, slow movers, a late store,
  a structural break, negative binomial noise). Absolute error levels don't transfer to a real business.
- **Promotions are assumed known and executed as planned.** Real promotions change at short notice,
  and their effect on neighbouring products (cannibalisation) is not modelled.
- **Daily, perishable stock.** The stock simulation treats each day independently: leftovers are
  waste and shortages are lost sales. Real replenishment has lead times, carry-over stock and pack
  sizes.
- **Stockouts hide demand.** The data records demand, not sales capped by stock. In real data, days
  with a stockout understate demand and need to be corrected or excluded.
- **Median forecasts are biased low on purpose.** For skewed demand, the median minimises absolute
  error, so WAPE is computed on medians and reads about 7% below the mean. Planning in money terms
  should use the mean or a quantile.
- **Intervals are calibrated per level, not per series.** A series that is far noisier than its peers
  still gets intervals that are too narrow.

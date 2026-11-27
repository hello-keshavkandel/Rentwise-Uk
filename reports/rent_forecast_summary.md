# Rent forecasts — regional, 12 months ahead (Phase C)

*Generated 2026-11-26 from ONS PIPR, all properties, monthly since Jan 2015.*

| Model (log rent) | Rolling-origin MAPE, 1–12 months |
|---|---|
| ARIMA(1,1,1) | 1.45% |
| ARIMA(1,1,0)+drift | 1.49% |
| Damped-trend ETS **(chosen)** | 0.87% |

Error grows with the horizon: 0.19% one month ahead, 1.63% twelve months ahead.

**Bands are conformal**, built from the backtest's own errors at each horizon. Out-of-time check (calibrated on older origins, tested on newer): nominal 80% → **71%** actual coverage, nominal 95% → **93%** actual coverage.

| Area | Latest | Annual change | Forecast in 12 months (80% band) |
|---|---|---|---|
| London | £2,350 (2026-10) | +3.8% | £2,466 (£2,405–£2,526) |
| England | £1,472 (2026-10) | +4.0% | £1,542 (£1,505–£1,580) |
| South East | £1,448 (2026-10) | +4.0% | £1,543 (£1,505–£1,581) |
| United Kingdom | £1,411 (2026-10) | +3.8% | £1,472 (£1,436–£1,508) |
| East of England | £1,302 (2026-10) | +3.8% | £1,359 (£1,326–£1,392) |
| South West | £1,254 (2026-10) | +3.3% | £1,299 (£1,268–£1,331) |
| West Midlands | £983 (2026-10) | +3.7% | £1,015 (£990–£1,039) |
| North West | £979 (2026-10) | +5.6% | £1,032 (£1,007–£1,057) |
| East Midlands | £928 (2026-10) | +3.5% | £958 (£934–£981) |
| Yorkshire and The Humber | £872 (2026-10) | +4.7% | £906 (£884–£928) |
| Wales | £849 (2026-10) | +3.9% | £876 (£855–£898) |
| North East | £793 (2026-10) | +4.9% | £829 (£808–£849) |

*Forecasts are statistical extrapolations of the official series, not predictions of policy effects (e.g. the Renters' Rights Act). Indicative only.*

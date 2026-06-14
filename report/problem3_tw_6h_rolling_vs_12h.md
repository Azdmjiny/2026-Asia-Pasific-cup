# P3 TW: Two 6h Rolling Forecasts vs Direct 12h Forecast

## Comparison definition

- Direct 12h: forecast the target at `t+12h` using information available at `t`.
- Rolling 6h+6h: at `t+6h`, update the history with newly observed data and use the model's 6h horizon to forecast the same `t+12h` target.
- Both methods are evaluated on identical target timestamps and observed NTU values.
- The rolling method has a six-hour information advantage, so this is an operational accuracy comparison rather than a same-origin algorithm comparison.

## Overall results

| Evaluation | Paired points | Direct 12h MAE | Rolling MAE | MAE improvement | Direct 12h RMSE | Rolling RMSE | RMSE improvement | Direct R2 | Rolling R2 | Rolling win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final holdout | 457 | 0.045413 | 0.036536 | 19.55% | 0.116457 | 0.112311 | 3.56% | 0.634395 | 0.659963 | 68.49% |
| Blocked CV | 1782 | 0.082859 | 0.062845 | 24.15% | 0.190550 | 0.158584 | 16.78% | 0.477551 | 0.638138 | 66.78% |

The blocked-CV mean paired absolute-error reduction is 0.020014 NTU. Its 95% moving-block bootstrap interval is [0.013269, 0.027610], using 24-hour blocks.

## Blocked-CV results by fold

| Fold | Paired points | Direct 12h MAE | Rolling MAE | Direct 12h RMSE | Rolling RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 526 | 0.072858 | 0.060110 | 0.161198 | 0.155231 |
| 2 | 526 | 0.116388 | 0.082201 | 0.230023 | 0.173049 |
| 3 | 367 | 0.078400 | 0.061011 | 0.214723 | 0.175023 |
| 4 | 363 | 0.053274 | 0.040615 | 0.131566 | 0.119070 |

## Conclusion

Two-stage 6h rolling forecasting is better for operational use when observations can be updated after six hours. Direct 12h forecasting remains the appropriate method when a full 12-hour forecast must be issued once at the original time without later updates.

# Problem 3 TW-Qin GRU Feature Ablation

## 1. Experiment Objective

This experiment evaluates whether additional CSTR-inspired hydraulic and load
features improve the clipped TW-Qin residual GRU model.

The prediction target is outlet-water turbidity `NTU`. The model predicts NTU
increments for the next 2, 4, 6, 8, 10 and 12 hours, then adds the predicted
increments to the current NTU.

To match the clipped Problem 1 experiment:

- outlet-water `NTU` is capped at 2.0;
- `FILT. NTU` is capped at 2.0;
- originally missing target values are excluded from training loss and metrics;
- filled target values are used only as historical inputs.

The clipping step affected:

| Column | Valid rows | Values above 2 before clipping | Maximum before clipping |
|---|---:|---:|---:|
| `NTU` | 5124 | 96 | 11.9 |
| `FILT. NTU` | 5460 | 89 | 9.8 |

## 2. Controlled Settings

All ablation variants use the same:

- chronological 70%/15%/15% train-validation-test split;
- random seed: 42;
- 12-step lookback, corresponding to 24 hours;
- six direct forecast horizons;
- GRU hidden size: 64;
- batch size: 32;
- learning rate: 0.001;
- Smooth L1 loss with observed-target masking;
- early stopping patience: 15;
- maximum epochs: 100.

Features are standardized using training-block statistics only. Model selection
is primarily based on the minimum validation loss.

## 3. Candidate Features

Let:

- \(C_t\) be current outlet-water `NTU`;
- \(C_{in,t}\) be `FILT. NTU`;
- \(Q_t\) be `T/W FLOW`;
- \(L_t\) be `C/W WELL LEVEL`;
- \(\epsilon\) be a small positive constant.

The tested derived features are:

\[
InLoad_t = Q_t C_{in,t}
\]

\[
OutLoad_t = Q_t C_t
\]

\[
NetLoad_t = Q_t(C_{in,t}-C_t)
\]

\[
HRT_t^* = \frac{L_t}{Q_t+\epsilon}
\]

\[
ReplaceRate_t = \frac{Q_t}{L_t+\epsilon}
\]

\[
\Delta C_t^{proxy}
=
\frac{Q_t(C_{in,t}-C_t)}{L_t+\epsilon}
\]

The last three quantities are strongly related. In particular,
`ReplaceRate` is approximately the reciprocal of `HRT_PROXY`, while
`DeltaC_PROXY` is proportional to the difference between the original
`inflow_term` and `outflow_term`.

## 4. Feature Variants

### Baseline

The existing eight-feature model:

```text
current_NTU
inflow_term
outflow_term
HRT_PROXY
WELL_LEVEL_CHANGE
FILT. NTU
R/W FLOW
T/W FLOW
```

### plus_netload

Baseline plus:

```text
NetLoad
```

### plus_loads

Baseline plus:

```text
InLoad
OutLoad
```

### compact_net

Replace the original inflow/outflow terms with the compact net-load feature:

```text
current_NTU
NetLoad
HRT_PROXY
WELL_LEVEL_CHANGE
FILT. NTU
R/W FLOW
T/W FLOW
```

### replace_rate

Replace `HRT_PROXY` with its inverse-style hydraulic feature:

```text
current_NTU
inflow_term
outflow_term
ReplaceRate
WELL_LEVEL_CHANGE
FILT. NTU
R/W FLOW
T/W FLOW
```

### compact_proxy

Use the normalized net-concentration-change proxy:

```text
current_NTU
DeltaC_PROXY
HRT_PROXY
WELL_LEVEL_CHANGE
FILT. NTU
R/W FLOW
T/W FLOW
```

## 5. Results

Results are ordered by minimum validation loss.

| Variant | Best validation loss | Best epoch | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|---:|---:|
| `replace_rate` | **0.217481** | 10 | 0.036007 | 0.109962 | 0.673825 |
| `plus_loads` | 0.219330 | 11 | 0.040525 | 0.114405 | 0.646936 |
| `compact_net` | 0.220522 | 10 | 0.037194 | 0.109586 | 0.676050 |
| `compact_proxy` | 0.220556 | 10 | 0.037358 | 0.110870 | 0.668414 |
| `baseline` | 0.222596 | 10 | **0.035268** | **0.104443** | **0.705745** |
| `plus_netload` | 0.223269 | 6 | 0.037355 | 0.107848 | 0.686248 |

Relative to the baseline:

| Variant | MAE change | RMSE change | R2 change |
|---|---:|---:|---:|
| `replace_rate` | +2.10% | +5.28% | -0.0319 |
| `plus_loads` | +14.91% | +9.54% | -0.0588 |
| `compact_net` | +5.46% | +4.92% | -0.0297 |
| `compact_proxy` | +5.93% | +6.15% | -0.0373 |
| `plus_netload` | +5.92% | +3.26% | -0.0195 |

Positive MAE/RMSE changes indicate worse test performance.

## 6. Interpretation

`replace_rate` achieved the lowest validation loss, but its test RMSE and R2
were worse than the baseline. This indicates that its validation improvement
did not generalize to the final time block.

The added load variables did not improve test performance. The likely reasons
are:

1. `NetLoad` is algebraically related to `InLoad` and `OutLoad`.
2. `DeltaC_PROXY` is already represented by the difference between the
   existing `inflow_term` and `outflow_term`.
3. `ReplaceRate` and `HRT_PROXY` encode nearly reciprocal information.
4. The GRU already receives 24 hours of `FILT. NTU`, flow and current NTU
   history, so it can learn many of these interactions from the sequence.
5. Adding correlated variables increases estimation variance without adding
   much independent process information.

## 7. Final Decision

No proposed feature variant produced a reliable improvement over the existing
eight-feature clipped baseline on the test block. Therefore, the final
notebook keeps:

```text
FEATURE_VARIANT = "baseline"
```

The retained model has:

| Metric | Result |
|---|---:|
| MAE | 0.035268 |
| RMSE | 0.104443 |
| R2 | 0.705745 |

The negative ablation result is still useful: it shows that the current
formula terms already contain the information represented by the proposed
load and hydraulic-ratio features. Further improvement should focus on model
training, regime handling or genuinely new measurements rather than adding
algebraically redundant variables.

## 8. Reproducibility Artifacts

The ablation summary is stored at:

```text
codes/p3/outputs/problem3_formula_gru/
TW_Qin_clip2_ablation/ablation_summary.xlsx
```

Each variant directory contains:

- training history;
- metrics comparison;
- test predictions;
- requested-date predictions;
- sensitivity analysis;
- an executed notebook snapshot.

## 9. Blocked Time-Series Evaluation

In addition to the final 70%/15%/15% holdout evaluation, the retained baseline
feature model was evaluated using four expanding-window time blocks.

The split design was:

| Fold | Training period | Validation period | Test period |
|---:|---|---|---|
| 1 | 2025-01-01 07:00 to 2025-08-16 17:00 | 2025-08-16 19:00 to 2025-10-01 05:00 | 2025-10-01 07:00 to 2025-11-15 17:00 |
| 2 | 2025-01-01 07:00 to 2025-10-01 05:00 | 2025-10-01 07:00 to 2025-11-15 17:00 | 2025-11-15 19:00 to 2025-12-31 05:00 |
| 3 | 2025-01-01 07:00 to 2025-11-15 17:00 | 2025-11-15 19:00 to 2025-12-31 05:00 | 2025-12-31 07:00 to 2026-02-14 17:00 |
| 4 | 2025-01-01 07:00 to 2025-12-31 05:00 | 2025-12-31 07:00 to 2026-02-14 17:00 | 2026-02-14 19:00 to 2026-04-01 05:00 |

Each fold independently refits:

- feature standardization;
- target-increment standardization;
- GRU parameters;
- early-stopping epoch.

No future block is used to fit the preceding training block.

### 9.1 Metrics by fold

| Fold | Observed forecast points | MAE | RMSE | R2 |
|---:|---:|---:|---:|---:|
| 1 | 3174 | 0.057917 | 0.146425 | 0.528195 |
| 2 | 3174 | 0.084068 | 0.182001 | 0.690660 |
| 3 | 2217 | 0.058666 | 0.174225 | 0.269680 |
| 4 | 2196 | 0.040300 | 0.115669 | 0.610084 |

The performance variation between folds indicates temporal distribution shift.
In particular, Fold 3 has a substantially lower R2 even though its MAE is not
the worst. This occurs because R2 depends on the target variance within each
time block as well as prediction error.

### 9.2 Aggregated blocked-test metrics

All predictions from the four non-overlapping test blocks were combined:

| Horizon | Observed count | MAE | RMSE | R2 |
|---:|---:|---:|---:|---:|
| 2 h | 1796 | 0.031587 | 0.091820 | 0.877980 |
| 4 h | 1795 | 0.049333 | 0.131311 | 0.750534 |
| 6 h | 1794 | 0.062594 | 0.158075 | 0.638662 |
| 8 h | 1793 | 0.070147 | 0.173750 | 0.563756 |
| 10 h | 1792 | 0.076809 | 0.183041 | 0.516193 |
| 12 h | 1791 | 0.082762 | 0.190179 | 0.478127 |
| Overall | 10761 | 0.062189 | 0.158361 | 0.637533 |

The expected horizon effect is clear: error increases and R2 decreases as the
forecast horizon grows.

The blocked result is weaker than the single final-holdout result
(`MAE=0.035268`, `RMSE=0.104443`, `R2=0.705745`). Therefore, the final-holdout
score should not be presented as the only estimate of model performance.
The blocked evaluation provides a more conservative estimate across different
operating periods:

```text
Blocked overall MAE  = 0.062189
Blocked overall RMSE = 0.158361
Blocked overall R2   = 0.637533
```

The full blocked evaluation workbook is stored at:

```text
codes/p3/outputs/problem3_formula_gru/
TW_Qin_clip2_ablation/baseline/blocked_time_series_cv.xlsx
```

### 9.3 Comparison with Persistence

Persistence predicts every future horizon using the current observed outlet
NTU:

\[
\hat C_{t+h}^{Persistence}=C_t
\]

GRU and Persistence are evaluated on exactly the same blocked test samples and
the same original-target observation mask.

| Horizon | GRU MAE | Persistence MAE | GRU RMSE | Persistence RMSE | GRU R2 | Persistence R2 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 h | 0.031587 | **0.030947** | **0.091820** | 0.099630 | **0.877980** | 0.856340 |
| 4 h | **0.049333** | 0.050540 | **0.131311** | 0.145735 | **0.750534** | 0.692721 |
| 6 h | **0.062594** | 0.064009 | **0.158075** | 0.174922 | **0.638662** | 0.557539 |
| 8 h | **0.070147** | 0.072137 | **0.173750** | 0.190761 | **0.563756** | 0.474153 |
| 10 h | **0.076809** | 0.079644 | **0.183041** | 0.200679 | **0.516193** | 0.418463 |
| 12 h | **0.082762** | 0.086668 | **0.190179** | 0.209632 | **0.478127** | 0.365903 |
| Overall | **0.062189** | 0.063973 | **0.158361** | 0.174310 | **0.637533** | 0.560842 |

Relative to Persistence, the blocked GRU produces:

- 2.79% lower overall MAE;
- 9.15% lower overall RMSE;
- an R2 increase from 0.560842 to 0.637533.

The GRU has lower RMSE and higher R2 at every horizon. At 2 hours,
Persistence has a slightly lower MAE by approximately 2.07%. From 4 to 12
hours, the GRU also achieves lower MAE, and its MAE advantage increases with
the prediction horizon.

This pattern is consistent with strong short-term NTU inertia:

- Persistence is difficult to beat for small two-hour changes under MAE.
- The GRU reduces larger errors and is increasingly useful at longer horizons.
- RMSE and R2 show that the GRU captures temporal changes that Persistence
  cannot represent.

Across individual folds, the GRU has lower RMSE and higher R2 in all four
test blocks. Persistence has lower MAE only in Fold 4. Therefore, the GRU
improvement is not caused by a single favorable block, although its magnitude
varies with operating period.

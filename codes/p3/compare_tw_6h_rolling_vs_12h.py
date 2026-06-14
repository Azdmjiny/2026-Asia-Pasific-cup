from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "codes/p3/outputs/problem3_formula_gru/TW_Qin_clip2_ablation/baseline"
REPORT_PATH = ROOT / "report/problem3_tw_6h_rolling_vs_12h.md"
SUMMARY_PATH = OUTPUT_DIR / "rolling_6h_vs_direct_12h_summary.csv"
PAIRED_PATH = OUTPUT_DIR / "rolling_6h_vs_direct_12h_paired.csv"

FINAL_PATH = OUTPUT_DIR / "test_predictions_long.xlsx"
BLOCKED_PATH = OUTPUT_DIR / "blocked_time_series_cv.xlsx"

BOOTSTRAP_REPS = 5000
BOOTSTRAP_BLOCK_SIZE = 12
SEED = 42


def metrics(y_true, y_pred):
    error = y_true - y_pred
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(np.square(error)))
    denominator = np.sum(np.square(y_true - np.mean(y_true)))
    r2 = 1.0 - np.sum(np.square(error)) / denominator
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def moving_block_bootstrap_ci(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    block_size = min(BOOTSTRAP_BLOCK_SIZE, n)
    starts = np.arange(n - block_size + 1)
    blocks_per_sample = int(np.ceil(n / block_size))
    rng = np.random.default_rng(SEED)
    estimates = np.empty(BOOTSTRAP_REPS)

    for rep in range(BOOTSTRAP_REPS):
        sampled_starts = rng.choice(starts, size=blocks_per_sample, replace=True)
        sampled = np.concatenate(
            [values[start : start + block_size] for start in sampled_starts]
        )[:n]
        estimates[rep] = np.mean(sampled)

    return np.quantile(estimates, [0.025, 0.975])


def align_predictions(df, prediction_column, evaluation):
    df = df.copy()
    df["base_time"] = pd.to_datetime(df["base_time"])
    df["target_time"] = pd.to_datetime(df["target_time"])

    direct = df.loc[df["horizon_hour"] == 12].copy()
    rolling = df.loc[df["horizon_hour"] == 6].copy()
    rolling["origin_time"] = rolling["base_time"] - pd.Timedelta(hours=6)

    keys = ["origin_time", "target_time"]
    if "fold" in df.columns:
        keys.insert(0, "fold")

    direct = direct.rename(
        columns={
            "base_time": "origin_time",
            prediction_column: "direct_12h",
            "true_NTU": "true_direct",
        }
    )
    rolling = rolling.rename(
        columns={
            prediction_column: "rolling_6plus6",
            "true_NTU": "true_rolling",
        }
    )

    paired = direct.merge(
        rolling[keys + ["rolling_6plus6", "true_rolling"]],
        on=keys,
        how="inner",
    )
    paired = paired.loc[
        np.isfinite(paired["true_direct"])
        & np.isfinite(paired["direct_12h"])
        & np.isfinite(paired["rolling_6plus6"])
    ].copy()

    if not np.allclose(paired["true_direct"], paired["true_rolling"]):
        raise ValueError("Direct and rolling predictions are not aligned to the same target.")

    paired["evaluation"] = evaluation
    paired["direct_abs_error"] = np.abs(
        paired["true_direct"] - paired["direct_12h"]
    )
    paired["rolling_abs_error"] = np.abs(
        paired["true_direct"] - paired["rolling_6plus6"]
    )
    paired["abs_error_reduction"] = (
        paired["direct_abs_error"] - paired["rolling_abs_error"]
    )
    return paired


def summarize_group(paired, evaluation, fold="all"):
    y_true = paired["true_direct"].to_numpy()
    direct = paired["direct_12h"].to_numpy()
    rolling = paired["rolling_6plus6"].to_numpy()
    direct_metrics = metrics(y_true, direct)
    rolling_metrics = metrics(y_true, rolling)
    error_reduction = paired["abs_error_reduction"].to_numpy()
    ci_low, ci_high = moving_block_bootstrap_ci(error_reduction)

    return {
        "evaluation": evaluation,
        "fold": fold,
        "paired_count": len(paired),
        "direct_12h_MAE": direct_metrics["MAE"],
        "rolling_6plus6_MAE": rolling_metrics["MAE"],
        "MAE_improvement_pct": (
            (direct_metrics["MAE"] - rolling_metrics["MAE"])
            / direct_metrics["MAE"]
            * 100
        ),
        "direct_12h_RMSE": direct_metrics["RMSE"],
        "rolling_6plus6_RMSE": rolling_metrics["RMSE"],
        "RMSE_improvement_pct": (
            (direct_metrics["RMSE"] - rolling_metrics["RMSE"])
            / direct_metrics["RMSE"]
            * 100
        ),
        "direct_12h_R2": direct_metrics["R2"],
        "rolling_6plus6_R2": rolling_metrics["R2"],
        "rolling_win_rate": np.mean(
            paired["rolling_abs_error"] < paired["direct_abs_error"]
        ),
        "mean_abs_error_reduction": np.mean(error_reduction),
        "error_reduction_ci_low": ci_low,
        "error_reduction_ci_high": ci_high,
    }


def fmt(value, digits=6):
    return f"{value:.{digits}f}"


def build_report(summary):
    final = summary.loc[
        (summary["evaluation"] == "final_holdout") & (summary["fold"] == "all")
    ].iloc[0]
    blocked = summary.loc[
        (summary["evaluation"] == "blocked_cv") & (summary["fold"] == "all")
    ].iloc[0]
    folds = summary.loc[
        (summary["evaluation"] == "blocked_cv") & (summary["fold"] != "all")
    ]

    lines = [
        "# P3 TW: Two 6h Rolling Forecasts vs Direct 12h Forecast",
        "",
        "## Comparison definition",
        "",
        "- Direct 12h: forecast the target at `t+12h` using information available at `t`.",
        "- Rolling 6h+6h: at `t+6h`, update the history with newly observed data and use the model's 6h horizon to forecast the same `t+12h` target.",
        "- Both methods are evaluated on identical target timestamps and observed NTU values.",
        "- The rolling method has a six-hour information advantage, so this is an operational accuracy comparison rather than a same-origin algorithm comparison.",
        "",
        "## Overall results",
        "",
        "| Evaluation | Paired points | Direct 12h MAE | Rolling MAE | MAE improvement | Direct 12h RMSE | Rolling RMSE | RMSE improvement | Direct R2 | Rolling R2 | Rolling win rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Final holdout | {int(final['paired_count'])} | "
            f"{fmt(final['direct_12h_MAE'])} | {fmt(final['rolling_6plus6_MAE'])} | "
            f"{final['MAE_improvement_pct']:.2f}% | {fmt(final['direct_12h_RMSE'])} | "
            f"{fmt(final['rolling_6plus6_RMSE'])} | {final['RMSE_improvement_pct']:.2f}% | "
            f"{fmt(final['direct_12h_R2'])} | {fmt(final['rolling_6plus6_R2'])} | "
            f"{final['rolling_win_rate']:.2%} |"
        ),
        (
            f"| Blocked CV | {int(blocked['paired_count'])} | "
            f"{fmt(blocked['direct_12h_MAE'])} | {fmt(blocked['rolling_6plus6_MAE'])} | "
            f"{blocked['MAE_improvement_pct']:.2f}% | {fmt(blocked['direct_12h_RMSE'])} | "
            f"{fmt(blocked['rolling_6plus6_RMSE'])} | {blocked['RMSE_improvement_pct']:.2f}% | "
            f"{fmt(blocked['direct_12h_R2'])} | {fmt(blocked['rolling_6plus6_R2'])} | "
            f"{blocked['rolling_win_rate']:.2%} |"
        ),
        "",
        "The blocked-CV mean paired absolute-error reduction is "
        f"{fmt(blocked['mean_abs_error_reduction'])} NTU. Its 95% moving-block "
        f"bootstrap interval is [{fmt(blocked['error_reduction_ci_low'])}, "
        f"{fmt(blocked['error_reduction_ci_high'])}], using 24-hour blocks.",
        "",
        "## Blocked-CV results by fold",
        "",
        "| Fold | Paired points | Direct 12h MAE | Rolling MAE | Direct 12h RMSE | Rolling RMSE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in folds.iterrows():
        lines.append(
            f"| {int(row['fold'])} | {int(row['paired_count'])} | "
            f"{fmt(row['direct_12h_MAE'])} | {fmt(row['rolling_6plus6_MAE'])} | "
            f"{fmt(row['direct_12h_RMSE'])} | {fmt(row['rolling_6plus6_RMSE'])} |"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Two-stage 6h rolling forecasting is better for operational use when observations can be updated after six hours. Direct 12h forecasting remains the appropriate method when a full 12-hour forecast must be issued once at the original time without later updates.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    final_df = pd.read_excel(FINAL_PATH)
    final_df = final_df.loc[final_df["target_observed"]]
    final_paired = align_predictions(final_df, "pred_NTU_gru", "final_holdout")

    blocked_df = pd.read_excel(BLOCKED_PATH, sheet_name="predictions")
    blocked_paired = align_predictions(blocked_df, "pred_NTU_GRU", "blocked_cv")

    rows = [
        summarize_group(final_paired, "final_holdout"),
        summarize_group(blocked_paired, "blocked_cv"),
    ]
    for fold, fold_df in blocked_paired.groupby("fold"):
        rows.append(summarize_group(fold_df, "blocked_cv", int(fold)))

    summary = pd.DataFrame(rows)
    paired = pd.concat([final_paired, blocked_paired], ignore_index=True)

    summary.to_csv(SUMMARY_PATH, index=False)
    paired.to_csv(PAIRED_PATH, index=False)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nSaved report: {REPORT_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved paired predictions: {PAIRED_PATH}")


if __name__ == "__main__":
    main()

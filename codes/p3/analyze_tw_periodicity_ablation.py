from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = (
    ROOT
    / "codes/p3/outputs/problem3_formula_gru/TW_Qin_clip2_ablation/baseline"
)
PERIODIC_DIR = (
    ROOT
    / "codes/p3/outputs/problem3_formula_gru/TW_Qin_periodicity_ablation"
)
OUTPUT_DIR = PERIODIC_DIR
REPORT_PATH = ROOT / "report/problem3_tw_periodicity_ablation.md"
SUMMARY_PATH = OUTPUT_DIR / "periodicity_ablation_summary.csv"
HORIZON_PATH = OUTPUT_DIR / "periodicity_ablation_by_horizon.csv"
DIAGNOSTICS_PATH = OUTPUT_DIR / "periodicity_signal_diagnostics.csv"
PAIRED_PATH = OUTPUT_DIR / "periodicity_paired_error_comparison.csv"

VARIANTS = ("baseline", "daily", "weekly", "daily_weekly")
BOOTSTRAP_REPS = 5000
BOOTSTRAP_BLOCK_SIZE = 12
SEED = 42


def parse_datetime(df):
    date = pd.to_datetime(df["DATE"], errors="coerce").dt.normalize()
    numeric_time = pd.to_numeric(df["TIME"], errors="coerce")
    if numeric_time.notna().mean() >= 0.7:
        if numeric_time.max() <= 1:
            time = pd.to_timedelta(numeric_time, unit="D")
        elif numeric_time.max() <= 24:
            time = pd.to_timedelta(numeric_time, unit="h")
        else:
            hour = (numeric_time // 100).fillna(0)
            minute = (numeric_time % 100).fillna(0)
            time = pd.to_timedelta(hour, unit="h") + pd.to_timedelta(
                minute, unit="m"
            )
    else:
        hour = df["TIME"].map(lambda value: getattr(value, "hour", np.nan))
        minute = df["TIME"].map(lambda value: getattr(value, "minute", np.nan))
        time = pd.to_timedelta(hour, unit="h") + pd.to_timedelta(
            minute, unit="m"
        )
    return date + time


def eta_squared(values, groups):
    frame = pd.DataFrame({"value": values, "group": groups}).dropna()
    if len(frame) == 0:
        return np.nan
    grand_mean = frame["value"].mean()
    total_ss = np.square(frame["value"] - grand_mean).sum()
    if total_ss == 0:
        return 0.0
    between_ss = sum(
        len(group) * np.square(group["value"].mean() - grand_mean)
        for _, group in frame.groupby("group")
    )
    return between_ss / total_ss


def lag_correlation(values, lag):
    series = pd.Series(values)
    return series.corr(series.shift(lag))


def spectral_diagnostic(values, period_steps):
    series = pd.Series(values).interpolate(limit_direction="both")
    differenced = series.diff().dropna().to_numpy()
    differenced = differenced - differenced.mean()
    power = np.square(np.abs(np.fft.rfft(differenced)))
    frequencies = np.fft.rfftfreq(len(differenced), d=1.0)
    target_frequency = 1.0 / period_steps
    index = np.argmin(np.abs(frequencies - target_frequency))
    positive_power = power[1:]
    target_power = power[index]
    percentile = np.mean(positive_power <= target_power)
    return target_power / np.median(positive_power), percentile


def build_signal_diagnostics():
    raw = pd.read_excel(ROOT / "data/merged.xlsx")
    raw.columns = [str(column).strip() for column in raw.columns]
    raw["DATETIME"] = parse_datetime(raw)
    raw["NTU_CLIP2"] = pd.to_numeric(raw["NTU"], errors="coerce").clip(upper=2.0)
    raw = raw.sort_values("DATETIME").reset_index(drop=True)

    exact_gap = raw["DATETIME"].diff().eq(pd.Timedelta(hours=2))
    delta_2h = raw["NTU_CLIP2"].diff().where(exact_gap)
    delta_6h = raw["NTU_CLIP2"].diff(3).where(
        raw["DATETIME"].diff(3).eq(pd.Timedelta(hours=6))
    )
    delta_12h = raw["NTU_CLIP2"].diff(6).where(
        raw["DATETIME"].diff(6).eq(pd.Timedelta(hours=12))
    )

    daily_ratio, daily_percentile = spectral_diagnostic(raw["NTU_CLIP2"], 12)
    weekly_ratio, weekly_percentile = spectral_diagnostic(raw["NTU_CLIP2"], 84)

    rows = [
        {
            "diagnostic": "level_acf_24h",
            "value": lag_correlation(raw["NTU_CLIP2"], 12),
        },
        {
            "diagnostic": "level_acf_168h",
            "value": lag_correlation(raw["NTU_CLIP2"], 84),
        },
        {
            "diagnostic": "delta_2h_acf_24h",
            "value": lag_correlation(delta_2h, 12),
        },
        {
            "diagnostic": "delta_2h_acf_168h",
            "value": lag_correlation(delta_2h, 84),
        },
        {
            "diagnostic": "delta_6h_hour_eta_squared",
            "value": eta_squared(delta_6h, raw["DATETIME"].dt.hour),
        },
        {
            "diagnostic": "delta_12h_hour_eta_squared",
            "value": eta_squared(delta_12h, raw["DATETIME"].dt.hour),
        },
        {
            "diagnostic": "delta_6h_weekday_eta_squared",
            "value": eta_squared(delta_6h, raw["DATETIME"].dt.dayofweek),
        },
        {
            "diagnostic": "delta_12h_weekday_eta_squared",
            "value": eta_squared(delta_12h, raw["DATETIME"].dt.dayofweek),
        },
        {"diagnostic": "daily_spectral_power_vs_median", "value": daily_ratio},
        {"diagnostic": "daily_spectral_power_percentile", "value": daily_percentile},
        {"diagnostic": "weekly_spectral_power_vs_median", "value": weekly_ratio},
        {
            "diagnostic": "weekly_spectral_power_percentile",
            "value": weekly_percentile,
        },
    ]
    return pd.DataFrame(rows)


def variant_dir(variant):
    return BASELINE_DIR if variant == "baseline" else PERIODIC_DIR / variant


def read_model_metrics(variant):
    directory = variant_dir(variant)
    holdout = pd.read_excel(directory / "metrics_comparison.xlsx")
    holdout = holdout.loc[
        holdout["model"].str.contains("GRU") & holdout["horizon"].eq("overall")
    ].iloc[0]

    blocked = pd.read_excel(
        directory / "blocked_time_series_cv.xlsx",
        sheet_name="aggregate_metrics",
    )
    blocked = blocked.loc[
        blocked["model"].eq("GRU") & blocked["horizon"].eq("overall")
    ].iloc[0]

    return {
        "variant": variant,
        "holdout_MAE": holdout["MAE"],
        "holdout_RMSE": holdout["RMSE"],
        "holdout_R2": holdout["R2"],
        "blocked_MAE": blocked["MAE"],
        "blocked_RMSE": blocked["RMSE"],
        "blocked_R2": blocked["R2"],
    }


def read_horizon_metrics(variant):
    metrics = pd.read_excel(
        variant_dir(variant) / "blocked_time_series_cv.xlsx",
        sheet_name="aggregate_metrics",
    )
    metrics = metrics.loc[
        metrics["model"].eq("GRU") & metrics["horizon"].ne("overall")
    ].copy()
    metrics.insert(0, "variant", variant)
    return metrics


def moving_block_ci(values):
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
        estimates[rep] = sampled.mean()
    return np.quantile(estimates, [0.025, 0.975])


def paired_error_comparison(variant):
    baseline = pd.read_excel(
        BASELINE_DIR / "blocked_time_series_cv.xlsx", sheet_name="predictions"
    )
    periodic = pd.read_excel(
        variant_dir(variant) / "blocked_time_series_cv.xlsx",
        sheet_name="predictions",
    )
    keys = ["fold", "base_time", "target_time", "horizon_hour"]
    paired = baseline.merge(
        periodic,
        on=keys,
        suffixes=("_baseline", "_periodic"),
        validate="one_to_one",
    )
    paired["baseline_abs_error"] = np.abs(
        paired["true_NTU_baseline"] - paired["pred_NTU_GRU_baseline"]
    )
    paired["periodic_abs_error"] = np.abs(
        paired["true_NTU_periodic"] - paired["pred_NTU_GRU_periodic"]
    )
    paired["error_reduction"] = (
        paired["baseline_abs_error"] - paired["periodic_abs_error"]
    )
    ci_low, ci_high = moving_block_ci(paired["error_reduction"])
    return {
        "variant": variant,
        "paired_count": len(paired),
        "mean_error_reduction": paired["error_reduction"].mean(),
        "periodic_win_rate": np.mean(
            paired["periodic_abs_error"] < paired["baseline_abs_error"]
        ),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def pct_change(value, baseline):
    return (value - baseline) / baseline * 100.0


def build_report(summary, horizon, diagnostics, paired):
    baseline = summary.loc[summary["variant"].eq("baseline")].iloc[0]
    lines = [
        "# P3 TW 周期性特征消融实验",
        "",
        "## 实验设置",
        "",
        "- `baseline`：原 8 个 TW-Qin GRU 特征。",
        "- `daily`：增加 24 小时 `hour_sin/hour_cos`。",
        "- `weekly`：增加 168 小时 `week_sin/week_cos`。",
        "- `daily_weekly`：同时增加日周期和周周期编码。",
        "- 所有版本保持相同模型结构、随机种子、裁剪、缺失值掩码、时间切分和 4 折 expanding-window CV。",
        "",
        "## 总体结果",
        "",
        "| 版本 | Holdout MAE | Holdout RMSE | Holdout R2 | Blocked MAE | 相对基线 | Blocked RMSE | 相对基线 | Blocked R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['variant']} | {row['holdout_MAE']:.6f} | "
            f"{row['holdout_RMSE']:.6f} | {row['holdout_R2']:.6f} | "
            f"{row['blocked_MAE']:.6f} | "
            f"{pct_change(row['blocked_MAE'], baseline['blocked_MAE']):+.2f}% | "
            f"{row['blocked_RMSE']:.6f} | "
            f"{pct_change(row['blocked_RMSE'], baseline['blocked_RMSE']):+.2f}% | "
            f"{row['blocked_R2']:.6f} |"
        )

    lines.extend(
        [
            "",
            "MAE/RMSE 的正百分比表示误差增加，即性能变差。",
            "",
            "## 配对误差比较",
            "",
            "| 版本 | 配对预测点 | 平均误差减少量 | 95% 时间块 bootstrap 区间 | 周期版逐点胜率 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in paired.iterrows():
        lines.append(
            f"| {row['variant']} | {int(row['paired_count'])} | "
            f"{row['mean_error_reduction']:.6f} | "
            f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}] | "
            f"{row['periodic_win_rate']:.2%} |"
        )

    diag = diagnostics.set_index("diagnostic")["value"]
    horizon_pivot = horizon.pivot(
        index="horizon_hour", columns="variant", values="MAE"
    )
    best_periodic_horizon = (
        horizon_pivot[list(VARIANTS[1:])].min(axis=1) / horizon_pivot["baseline"]
        - 1.0
    ) * 100.0
    lines.extend(
        [
            "",
            "## 周期信号诊断",
            "",
            f"- NTU 水平序列 24h 自相关：`{diag['level_acf_24h']:.4f}`；168h 自相关：`{diag['level_acf_168h']:.4f}`。",
            f"- 去除水平惯性后，2h 变化量的 24h 自相关：`{diag['delta_2h_acf_24h']:.4f}`；168h 自相关：`{diag['delta_2h_acf_168h']:.4f}`。",
            f"- 时刻对 6h/12h NTU 增量的解释比例 eta²：`{diag['delta_6h_hour_eta_squared']:.4%}` / `{diag['delta_12h_hour_eta_squared']:.4%}`。",
            f"- 星期对 6h/12h NTU 增量的解释比例 eta²：`{diag['delta_6h_weekday_eta_squared']:.4%}` / `{diag['delta_12h_weekday_eta_squared']:.4%}`。",
            "- 水平序列看起来有 24h 相关，但变化量中的周期相关和分组解释力很弱，说明主要是短期持续性，不是稳定可迁移的周期。",
            f"- 逐 horizon 看，周期版本只有 2h MAE 曾小幅改善（最佳为 `{best_periodic_horizon.loc[2]:+.2f}%`），但对应 RMSE 没有同步改善；4–12h 的最佳周期版本 MAE 均不优于基线。",
            "",
            "## 结论",
            "",
            "显式日周期、周周期及二者组合均未提升 P3_TW 的总体表现。blocked CV 的 MAE 和 RMSE 全部高于基线，最终留出集也全部变差；配对时间块区间未显示稳定收益。因此当前最终模型不建议加入这些周期编码。",
            "",
            "若继续探索，更值得测试的是与运行机制相关的状态特征，例如泵组班次、运行日阶段或流量制度切换，而不是仅由日历生成的固定正余弦周期。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([read_model_metrics(variant) for variant in VARIANTS])
    horizon = pd.concat(
        [read_horizon_metrics(variant) for variant in VARIANTS],
        ignore_index=True,
    )
    diagnostics = build_signal_diagnostics()
    paired = pd.DataFrame(
        [paired_error_comparison(variant) for variant in VARIANTS[1:]]
    )

    summary.to_csv(SUMMARY_PATH, index=False)
    horizon.to_csv(HORIZON_PATH, index=False)
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)
    paired.to_csv(PAIRED_PATH, index=False)
    REPORT_PATH.write_text(
        build_report(summary, horizon, diagnostics, paired),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print()
    print(diagnostics.to_string(index=False))
    print()
    print(paired.to_string(index=False))
    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

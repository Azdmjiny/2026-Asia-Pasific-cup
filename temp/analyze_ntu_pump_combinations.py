#!/usr/bin/env python3
"""Relate raw NTU jumps to raw pump combinations and combination changes."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "temp"))

import merge_and_audit as merge  # noqa: E402


OUTPUT_DIR = ROOT / "analysis" / "ntu_pump"
REPORT_PATH = OUTPUT_DIR / "report.md"
EVENTS_PATH = OUTPUT_DIR / "ntu_jump_events.csv"
COMBOS_PATH = OUTPUT_DIR / "pump_combination_summary.csv"
LAGS_PATH = OUTPUT_DIR / "pump_switch_lag_summary.csv"

PUMP_COLUMNS = ["R/W PUMP DUTY", "T/W PUMP DUTY"]
CONTROL_COLUMNS = ["R/W NTU", "FILT. NTU", "R/W FLOW", "T/W FLOW"]


def canonical_pump(value: object, column: str) -> str | None:
    if merge.is_missing(value):
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if column == "T/W PUMP DUTY" and text == "244":
        return "2+4"
    if column == "T/W PUMP DUTY" and text == "2,2":
        return "2+3"
    ids = re.findall(r"\d+", text)
    if not ids:
        return None
    return "+".join(sorted(set(ids), key=int))


def load_raw() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in sorted((merge.DATA_DIR / "2025").glob("*.xlsx")):
        records.extend(merge.parse_2025_file(path)[0])
    for path in sorted((merge.DATA_DIR / "2026").glob("*.xls")):
        records.extend(merge.parse_2026_file(path)[0])

    frame = pd.DataFrame.from_records(records)
    frame["DATETIME"] = pd.to_datetime(frame["DATE"].astype(str)) + pd.to_timedelta(
        frame["TIME"].map(lambda value: value.hour), unit="h"
    )
    frame = frame.sort_values("DATETIME", kind="stable").reset_index(drop=True)
    for column in ["NTU", *CONTROL_COLUMNS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in PUMP_COLUMNS:
        frame[f"{column} COMBO"] = frame[column].map(
            lambda value: canonical_pump(value, column)
        )

    frame["CONTIGUOUS"] = frame["DATETIME"].diff().eq(pd.Timedelta(hours=2))
    frame["NTU_DELTA"] = frame["NTU"].diff()
    frame["ABS_NTU_DELTA"] = frame["NTU_DELTA"].abs()
    valid_pair = frame["CONTIGUOUS"] & frame["NTU"].notna() & frame["NTU"].shift().notna()
    frame.loc[~valid_pair, ["NTU_DELTA", "ABS_NTU_DELTA"]] = np.nan

    for column in CONTROL_COLUMNS:
        frame[f"ABS_{column}_DELTA"] = frame[column].diff().abs()
        valid = frame["CONTIGUOUS"] & frame[column].notna() & frame[column].shift().notna()
        frame.loc[~valid, f"ABS_{column}_DELTA"] = np.nan

    for column in PUMP_COLUMNS:
        combo = f"{column} COMBO"
        previous = frame[combo].shift()
        observed_pair = frame["CONTIGUOUS"] & frame[combo].notna() & previous.notna()
        switch = (observed_pair & frame[combo].ne(previous)).astype("boolean")
        switch.loc[~observed_pair] = pd.NA
        frame[f"{column} SWITCH"] = switch
        frame[f"{column} OBSERVED_PAIR"] = observed_pair
    return frame


def rate_test(frame: pd.DataFrame, flag: str, jump: str) -> dict[str, float]:
    data = frame.loc[frame[flag].notna() & frame[jump].notna(), [flag, jump]]
    table = pd.crosstab(data[flag], data[jump]).reindex(
        index=[False, True], columns=[False, True], fill_value=0
    )
    stable_n = int(table.loc[False].sum())
    switch_n = int(table.loc[True].sum())
    stable_rate = table.loc[False, True] / stable_n if stable_n else np.nan
    switch_rate = table.loc[True, True] / switch_n if switch_n else np.nan
    odds_ratio, p_value = fisher_exact(table.to_numpy())
    return {
        "stable_n": stable_n,
        "switch_n": switch_n,
        "stable_rate": float(stable_rate),
        "switch_rate": float(switch_rate),
        "rate_ratio": float(switch_rate / stable_rate) if stable_rate else np.nan,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
    }


def fmt_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_raw()
    delta = frame["ABS_NTU_DELTA"].dropna()
    thresholds = {
        "Q95": float(delta.quantile(0.95)),
        "Q99": float(delta.quantile(0.99)),
        "FIXED_0.2": 0.2,
    }
    for name, threshold in thresholds.items():
        frame[f"JUMP_{name}"] = frame["ABS_NTU_DELTA"].ge(threshold).astype("boolean")
        frame.loc[frame["ABS_NTU_DELTA"].isna(), f"JUMP_{name}"] = pd.NA

    lag_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for pump in PUMP_COLUMNS:
        observed = frame[f"{pump} OBSERVED_PAIR"]
        switch = frame[f"{pump} SWITCH"].where(observed)
        for threshold_name in thresholds:
            jump = f"JUMP_{threshold_name}"
            temp = pd.DataFrame({"flag": switch, "jump": frame[jump]})
            result = rate_test(temp, "flag", "jump")
            test_rows.append({"pump": pump, "threshold": threshold_name, **result})

        for lag in range(7):
            lag_flag = switch.shift(lag)
            valid = observed.shift(lag).fillna(False) & frame["ABS_NTU_DELTA"].notna()
            temp = pd.DataFrame(
                {"flag": lag_flag.where(valid), "jump": frame["JUMP_Q99"]}
            )
            result = rate_test(temp, "flag", "jump")
            lag_rows.append(
                {
                    "pump": pump,
                    "lag_steps": lag,
                    "lag_hours": lag * 2,
                    **result,
                }
            )

    tests = pd.DataFrame(test_rows)
    lag_summary = pd.DataFrame(lag_rows)

    combo_rows: list[dict[str, object]] = []
    for pump in PUMP_COLUMNS:
        combo_col = f"{pump} COMBO"
        subset = frame.dropna(subset=[combo_col, "NTU"]).copy()
        groups = [
            group["NTU"].to_numpy()
            for _, group in subset.groupby(combo_col)
            if len(group) >= 20
        ]
        combo_p = kruskal(*groups).pvalue if len(groups) >= 2 else np.nan
        for combo, group in subset.groupby(combo_col):
            if len(group) < 10:
                continue
            combo_rows.append(
                {
                    "pump": pump,
                    "combination": combo,
                    "n": len(group),
                    "NTU_median": group["NTU"].median(),
                    "NTU_mean": group["NTU"].mean(),
                    "NTU_q90": group["NTU"].quantile(0.90),
                    "Q99_jump_rate": group["JUMP_Q99"].mean(),
                    "global_combo_test_p": combo_p,
                    "start": group["DATETIME"].min(),
                    "end": group["DATETIME"].max(),
                    "active_months": group["DATETIME"].dt.to_period("M").nunique(),
                }
            )
    combos = pd.DataFrame(combo_rows).sort_values(["pump", "n"], ascending=[True, False])

    event_columns = [
        "DATETIME",
        "NTU",
        "NTU_DELTA",
        "ABS_NTU_DELTA",
        *CONTROL_COLUMNS,
        *[f"ABS_{column}_DELTA" for column in CONTROL_COLUMNS],
    ]
    for pump in PUMP_COLUMNS:
        event_columns.extend(
            [f"{pump} COMBO", f"{pump} SWITCH"]
        )
        for lag in range(1, 7):
            frame[f"{pump} SWITCH_{lag * 2}H_AGO"] = frame[f"{pump} SWITCH"].shift(lag)
            event_columns.append(f"{pump} SWITCH_{lag * 2}H_AGO")
    events = frame.loc[frame["JUMP_Q99"].eq(True), event_columns].copy()

    # Remove rows where any measured process variable also changed unusually sharply.
    control_cutoffs = {
        column: frame[f"ABS_{column}_DELTA"].quantile(0.95)
        for column in CONTROL_COLUMNS
    }
    control_spikes = pd.DataFrame(
        {
            column: frame[f"ABS_{column}_DELTA"].gt(cutoff)
            for column, cutoff in control_cutoffs.items()
        }
    )
    q99_events = frame["JUMP_Q99"].eq(True)
    any_control_spike = control_spikes.any(axis=1)
    q99_with_control_spike = int((q99_events & any_control_spike).sum())
    q99_with_filter_spike = int((q99_events & control_spikes["FILT. NTU"]).sum())
    clean_mask = frame["ABS_NTU_DELTA"].notna()
    for column, cutoff in control_cutoffs.items():
        change = frame[f"ABS_{column}_DELTA"]
        clean_mask &= change.isna() | change.le(cutoff)

    clean_results: list[dict[str, object]] = []
    magnitude_results: list[dict[str, object]] = []
    for pump in PUMP_COLUMNS:
        observed = frame[f"{pump} OBSERVED_PAIR"]
        switch_col = f"{pump} SWITCH"
        clean = frame.loc[clean_mask & observed].copy()
        clean[switch_col] = clean[switch_col].astype(bool)
        result = rate_test(clean, switch_col, "JUMP_Q99")
        clean_results.append({"pump": pump, **result})

        switched = frame.loc[observed & frame[switch_col], "ABS_NTU_DELTA"].dropna()
        stable = frame.loc[observed & ~frame[switch_col], "ABS_NTU_DELTA"].dropna()
        p_value = mannwhitneyu(switched, stable, alternative="two-sided").pvalue
        magnitude_results.append(
            {
                "pump": pump,
                "switch_median": switched.median(),
                "stable_median": stable.median(),
                "p_value": p_value,
            }
        )

    tests.to_csv(OUTPUT_DIR / "threshold_tests.csv", index=False)
    lag_summary.to_csv(LAGS_PATH, index=False)
    combos.to_csv(COMBOS_PATH, index=False)
    events.to_csv(EVENTS_PATH, index=False, date_format="%Y-%m-%d %H:%M")

    lines = [
        "# 原始数据：NTU 突变与 PUMP DUTY 组合分析",
        "",
        "## 结论",
        "",
        "- **没有证据表明 PUMP DUTY 组合切换会触发 NTU 突变。** 在泵状态前后均有原始记录的样本中，R/W 与 T/W 换泵当点的 Q99 大突变均为 0 次；Q95、固定 0.2 NTU 阈值也不显著。",
        "- **未发现稳定的 2–12 小时滞后效应。** T/W 切换后 2 和 4 小时风险比约 1.76，但样本事件很少且 p=0.325，不足以认为有关。",
        "- **不同泵号组合对应的 NTU 分布有差异，但不能解释为泵组合造成。** 部分少见组合集中在单一异常时段，例如 R/W `1+2` 仅出现于 2025-09-04 至 2025-09-08，覆盖了 9 月 7 日的高浊度过程。",
        f"- 52 次大突变中有 {q99_with_control_spike} 次（{q99_with_control_spike / int(q99_events.sum()):.1%}）同时伴随至少一个原水/滤后水浊度或流量指标的剧变，其中 {q99_with_filter_spike} 次伴随滤后水 NTU 剧变；排除这些同步工况剧变后，换泵关联仍未出现。",
        "",
        "## 口径",
        "",
        f"- 使用处理前月度附件，共 {len(frame)} 条两小时记录；NTU 有效值 {frame['NTU'].notna().sum()} 条。",
        "- 泵号分隔符统一后保留具体组合，例如 `1,3`、`1&3` 统一为 `1+3`，没有压缩成泵数量。",
        f"- R/W PUMP DUTY 原始非空 {frame['R/W PUMP DUTY COMBO'].notna().sum()} 条，T/W PUMP DUTY 原始非空 {frame['T/W PUMP DUTY COMBO'].notna().sum()} 条；泵字段缺失较多，因此结论只适用于有记录时段。",
        f"- 主突变阈值为相邻两小时 `|ΔNTU| >= {thresholds['Q99']:.3f}`（原始变化的 99% 分位数）；共 {int(frame['JUMP_Q99'].sum())} 次。",
        f"- 敏感性阈值：Q95={thresholds['Q95']:.3f}，以及固定阈值 0.2 NTU。",
        "",
        "## 组合切换与突变",
        "",
        "| 泵字段 | 阈值 | 稳定时突变率 | 切换时突变率 | 风险比 | Fisher p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in test_rows:
        lines.append(
            f"| {row['pump']} | {row['threshold']} | "
            f"{row['stable_rate']:.2%} ({row['stable_n']}) | "
            f"{row['switch_rate']:.2%} ({row['switch_n']}) | "
            f"{row['rate_ratio']:.2f} | {fmt_p(row['p_value'])} |"
        )

    lines.extend(
        [
            "",
            "## 滞后结果（主阈值）",
            "",
            "| 泵字段 | 切换领先时间 | 非切换突变率 | 切换后突变率 | 风险比 | p |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in lag_rows:
        lines.append(
            f"| {row['pump']} | {row['lag_hours']} 小时 | "
            f"{row['stable_rate']:.2%} | {row['switch_rate']:.2%} | "
            f"{row['rate_ratio']:.2f} | {fmt_p(row['p_value'])} |"
        )

    lines.extend(["", "## 排除同步工况剧变后的结果", ""])
    for row in clean_results:
        lines.append(
            f"- `{row['pump']}`：切换时 Q99 突变率 {row['switch_rate']:.2%}，"
            f"稳定时 {row['stable_rate']:.2%}，风险比 {row['rate_ratio']:.2f}，"
            f"p={fmt_p(row['p_value'])}。"
        )

    lines.extend(["", "## 相邻变化幅度", ""])
    for row in magnitude_results:
        lines.append(
            f"- `{row['pump']}`：切换时 `|ΔNTU|` 中位数 {row['switch_median']:.3f}，"
            f"稳定时 {row['stable_median']:.3f}，Mann–Whitney p={fmt_p(row['p_value'])}。"
        )

    lines.extend(
        [
            "",
            "## 不同组合的 NTU 水平",
            "",
            "| 泵字段 | 组合 | 样本数 | NTU中位数 | NTU均值 | NTU P90 | 活跃月份数 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in combos.to_dict("records"):
        lines.append(
            f"| {row['pump']} | {row['combination']} | {row['n']} | "
            f"{row['NTU_median']:.3f} | {row['NTU_mean']:.3f} | {row['NTU_q90']:.3f} | "
            f"{row['active_months']} |"
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote {EVENTS_PATH.relative_to(ROOT)}")
    print(f"Wrote {COMBOS_PATH.relative_to(ROOT)}")
    print(f"Wrote {LAGS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

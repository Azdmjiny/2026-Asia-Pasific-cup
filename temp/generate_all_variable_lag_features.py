# ============================================================
# Generate all-variable lag feature dataset for Problem 1
# ============================================================
# Purpose:
#   Read merged.xlsx and construct lag features for ALL usable variables.
#   No correlation screening.
#   No best-lag selection.
#
# Output:
#   data/all_variable_lag_features.xlsx
#   outputs/problem1/all_variable_lag_feature_summary.xlsx
#   outputs/problem1/all_variable_lag_missing_audit.xlsx
#
# Main structure:
#   DATETIME
#   OP_DATE
#   target_NTU
#   all variables lag0-lag12
#   optional historical target NTU_lag1-lag12
#
# Notes:
#   1. Current NTU is kept as target_NTU.
#   2. NTU_lag0 is NOT generated as input, because it equals target_NTU and causes leakage.
#   3. F/RIDE missing values are filled with 0.
#   4. PUMP DUTY is converted to PUMP COUNT before lag construction.
#   5. This script does not standardize features. Standardization should be done inside the model-training pipeline using training data only.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.max_columns", 160)
pd.set_option("display.width", 200)


# ============================================================
# 0. Settings
# ============================================================

TARGET_COL = "NTU"

MAX_LAG = 12
# Data frequency is 2 hours:
# lag0 = current time
# lag1 = 2 hours before
# lag12 = 24 hours before

INCLUDE_LAG0_FOR_EXOGENOUS_FEATURES = True
INCLUDE_TARGET_HISTORY_LAGS = True
# If True, generate NTU_lag1 to NTU_lag12.
# NTU_lag0 is never generated as an input feature because it equals target_NTU.

FILL_FRIDE_MISSING_WITH_ZERO = True

# Do not clip NTU by default in this all-lag data file.
# If you need the clipped version, set CLIP_NTU_RELATED_VALUES = True.
CLIP_NTU_RELATED_VALUES = False
CLIP_NTU_UPPER = 2.0


# ============================================================
# 1. Locate merged.xlsx automatically
# ============================================================

def find_merged_excel(filename="merged.xlsx"):
    cwd = Path.cwd().resolve()
    candidate_paths = []

    search_roots = [cwd] + list(cwd.parents)

    for root in search_roots:
        candidate_paths.extend([
            root / "data" / filename,
            root / "codes" / "data" / filename,
            root / "2026-Asia-Pacific-cup" / "data" / filename,
            root / "2026-Asia-Pacific-cup" / "codes" / "data" / filename,
            root / "2026-Asia-Pasific-cup" / "data" / filename,
            root / "2026-Asia-Pasific-cup" / "codes" / "data" / filename,
            root / filename,
        ])

    seen = set()
    unique_candidates = []
    for p in candidate_paths:
        p = p.resolve()
        if p not in seen:
            seen.add(p)
            unique_candidates.append(p)

    for p in unique_candidates:
        if p.exists():
            return p

    # Fallback: limited recursive search under current directory.
    for p in cwd.rglob(filename):
        if p.name == filename:
            return p.resolve()

    raise FileNotFoundError(
        f"Cannot find {filename}. Current working directory: {cwd}"
    )


DATA_PATH = find_merged_excel()
DATA_DIR = DATA_PATH.parent

if DATA_DIR.name == "data":
    PROJECT_DIR = DATA_DIR.parent
else:
    PROJECT_DIR = DATA_DIR

OUTPUT_DIR = PROJECT_DIR / "outputs" / "problem1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Current working directory:", Path.cwd().resolve())
print("Using data file:", DATA_PATH)
print("Data directory:", DATA_DIR)
print("Output directory:", OUTPUT_DIR)


# ============================================================
# 2. Read merged.xlsx
# ============================================================

df = pd.read_excel(DATA_PATH)

print("\nOriginal data shape:", df.shape)
print("Original columns:")
print(df.columns.tolist())

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")


# ============================================================
# 3. Construct DATETIME and OP_DATE
# ============================================================

def construct_datetime(data):
    data = data.copy()

    if "DATETIME" in data.columns:
        data["DATETIME"] = pd.to_datetime(data["DATETIME"], errors="coerce")
        return data

    date_candidates = ["DATE", "Date", "date"]
    time_candidates = ["TIME", "Time", "time"]

    date_col = next((c for c in date_candidates if c in data.columns), None)
    time_col = next((c for c in time_candidates if c in data.columns), None)

    if date_col is None or time_col is None:
        raise ValueError("Cannot construct DATETIME. DATE and TIME columns are required.")

    data["DATETIME"] = pd.to_datetime(
        data[date_col].astype(str) + " " + data[time_col].astype(str),
        errors="coerce"
    )

    return data


df = construct_datetime(df)

missing_datetime = df["DATETIME"].isna().sum()
print("\nMissing DATETIME:", missing_datetime)

df = df.sort_values("DATETIME").reset_index(drop=True)

# Operating day:
# 07:00, 09:00, ..., 23:00, next day 01:00, 03:00, 05:00
df["OP_DATE"] = df["DATETIME"].dt.date
mask_before_7 = df["DATETIME"].dt.hour < 7
df.loc[mask_before_7, "OP_DATE"] = (
    df.loc[mask_before_7, "DATETIME"] - pd.Timedelta(days=1)
).dt.date

print("\nDatetime range:")
print(df["DATETIME"].min(), "to", df["DATETIME"].max())

print("\nOP_DATE check:")
print(df[["DATETIME", "OP_DATE"]].head(15).to_string(index=False))


# ============================================================
# 4. Convert PUMP DUTY to PUMP COUNT
# ============================================================

def pump_duty_to_count(x):
    """
    Convert pump duty records to number of running pumps.

    Examples:
    1     -> 1
    2     -> 1
    3     -> 1
    1+4   -> 2
    2+3   -> 2
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s == "":
        return np.nan

    if s.endswith(".0"):
        s = s.replace(".0", "")

    if "+" in s:
        parts = [p.strip() for p in s.split("+") if p.strip() != ""]
        return len(parts)

    if s.isdigit():
        return 1

    return np.nan


if "R/W PUMP DUTY" in df.columns:
    df["R/W PUMP COUNT"] = df["R/W PUMP DUTY"].apply(pump_duty_to_count)
    print("\nCreated R/W PUMP COUNT from R/W PUMP DUTY.")

if "T/W PUMP DUTY" in df.columns:
    df["T/W PUMP COUNT"] = df["T/W PUMP DUTY"].apply(pump_duty_to_count)
    print("Created T/W PUMP COUNT from T/W PUMP DUTY.")


# ============================================================
# 5. Basic cleaning
# ============================================================

# Fill F/RIDE missing with 0.
if FILL_FRIDE_MISSING_WITH_ZERO and "F/RIDE" in df.columns:
    df["F/RIDE"] = pd.to_numeric(df["F/RIDE"], errors="coerce")
    missing_before = df["F/RIDE"].isna().sum()
    df["F/RIDE"] = df["F/RIDE"].fillna(0)
    missing_after = df["F/RIDE"].isna().sum()

    print("\nF/RIDE missing before filling:", missing_before)
    print("F/RIDE missing after filling:", missing_after)


# Convert known numeric-like columns to numeric.
exclude_raw_cols = {"DATE", "Date", "date", "TIME", "Time", "time", "DATETIME", "OP_DATE"}
raw_pump_duty_cols = {"R/W PUMP DUTY", "T/W PUMP DUTY"}

for col in df.columns:
    if col in exclude_raw_cols or col in raw_pump_duty_cols:
        continue
    df[col] = pd.to_numeric(df[col], errors="coerce")


# Optional NTU clipping.
clip_records = []

if CLIP_NTU_RELATED_VALUES:
    ntu_related_cols = [
        col for col in df.columns
        if "NTU" in col.upper()
    ]

    for col in ntu_related_cols:
        count_above_before = (df[col] > CLIP_NTU_UPPER).sum()
        max_before = df[col].max(skipna=True)

        df[col] = df[col].clip(upper=CLIP_NTU_UPPER)

        count_above_after = (df[col] > CLIP_NTU_UPPER).sum()
        max_after = df[col].max(skipna=True)

        clip_records.append({
            "column": col,
            "count_above_clip_before": int(count_above_before),
            "max_before": max_before,
            "clip_upper": CLIP_NTU_UPPER,
            "count_above_clip_after": int(count_above_after),
            "max_after": max_after,
        })

clip_report_df = pd.DataFrame(clip_records)
if len(clip_report_df) > 0:
    clip_report_path = OUTPUT_DIR / "all_variable_lag_ntu_clip_report.xlsx"
    clip_report_df.to_excel(clip_report_path, index=False)
    print("\nSaved NTU clipping report:", clip_report_path)


# ============================================================
# 6. Define all lag base variables
# ============================================================
# All usable variables except:
# - raw date/time columns
# - raw PUMP DUTY columns
# - target NTU as current target
#
# For NTU itself, only historical lags lag1-lag12 can be added if INCLUDE_TARGET_HISTORY_LAGS=True.

metadata_cols = {"DATE", "Date", "date", "TIME", "Time", "time", "DATETIME", "OP_DATE"}
exclude_from_exogenous = metadata_cols | raw_pump_duty_cols | {TARGET_COL}

exogenous_features = [
    col for col in df.columns
    if col not in exclude_from_exogenous
]

# Keep only columns that have at least some numeric data.
exogenous_features = [
    col for col in exogenous_features
    if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() > 0
]

print("\nExogenous features used for all-lag construction:")
for col in exogenous_features:
    print(" -", col)

if INCLUDE_TARGET_HISTORY_LAGS:
    print(f"\nHistorical target lags will be generated: {TARGET_COL}_lag1 to {TARGET_COL}_lag{MAX_LAG}")
else:
    print("\nHistorical target lags will NOT be generated.")


# ============================================================
# 7. Construct all lag features
# ============================================================

lag_df = df[["DATETIME", "OP_DATE", TARGET_COL]].copy()
lag_df = lag_df.rename(columns={TARGET_COL: "target_NTU"})

lag_feature_records = []

# Exogenous variables:
start_lag = 0 if INCLUDE_LAG0_FOR_EXOGENOUS_FEATURES else 1

for feature in exogenous_features:
    for lag in range(start_lag, MAX_LAG + 1):
        lag_col = f"{feature}_lag{lag}"
        lag_df[lag_col] = df[feature].shift(lag)

        lag_feature_records.append({
            "base_feature": feature,
            "lag": lag,
            "lag_hours": lag * 2,
            "lag_feature": lag_col,
            "feature_type": "exogenous",
        })

# Historical target NTU lags:
if INCLUDE_TARGET_HISTORY_LAGS:
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    for lag in range(1, MAX_LAG + 1):
        lag_col = f"{TARGET_COL}_lag{lag}"
        lag_df[lag_col] = df[TARGET_COL].shift(lag)

        lag_feature_records.append({
            "base_feature": TARGET_COL,
            "lag": lag,
            "lag_hours": lag * 2,
            "lag_feature": lag_col,
            "feature_type": "historical_target",
        })

# Safety check: target leakage.
if "NTU_lag0" in lag_df.columns:
    raise ValueError("NTU_lag0 found. This causes target leakage and should not be generated.")

lag_feature_summary_df = pd.DataFrame(lag_feature_records)

print("\nAll-variable lag data shape:", lag_df.shape)
print("Number of lag features:", len(lag_feature_summary_df))
print("\nFirst rows:")
print(lag_df.head(10).to_string(index=False))


# ============================================================
# 8. Missing audit
# ============================================================

missing_audit_df = pd.DataFrame({
    "column": lag_df.columns,
    "missing_count": lag_df.isna().sum().values,
    "missing_rate": lag_df.isna().mean().values,
    "non_missing_count": lag_df.notna().sum().values,
})

print("\nMissing audit preview:")
print(missing_audit_df.sort_values("missing_rate", ascending=False).head(20).to_string(index=False))


# ============================================================
# 9. Save output files
# ============================================================

all_lag_path = DATA_DIR / "all_variable_lag_features.xlsx"
summary_path = OUTPUT_DIR / "all_variable_lag_feature_summary.xlsx"
missing_audit_path = OUTPUT_DIR / "all_variable_lag_missing_audit.xlsx"

lag_df.to_excel(all_lag_path, index=False)
lag_feature_summary_df.to_excel(summary_path, index=False)
missing_audit_df.to_excel(missing_audit_path, index=False)

print("\nSaved files:")
print("1.", all_lag_path)
print("2.", summary_path)
print("3.", missing_audit_path)

print("\nMain all-lag data file:")
print(all_lag_path)

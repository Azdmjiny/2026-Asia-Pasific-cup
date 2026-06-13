# ============================================================
# Generate selected-one-lag modeling data for Problem 1
# ============================================================
# Purpose:
# For each selected original variable, several candidate lag orders are given.
# This script selects exactly ONE most important lag for each variable
# according to its correlation with current target_NTU in the training period.
#
# Output:
# 1. data/candidate_lag_all.xlsx
# 2. data/selected_one_lag_model_data.xlsx
# 3. outputs/problem1/selected_one_lag_summary.xlsx
#
# Main modeling form:
# NTU_t = f(one selected lag feature for each variable)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)

# ============================================================
# 0. Settings
# ============================================================

TARGET_COL = "NTU"

# Use Spearman because water-treatment variables may have nonlinear monotonic relations.
# You can change it to "pearson" if needed.
CORR_METHOD = "spearman"

# Keep exactly one lag for each feature in CANDIDATE_LAG_FEATURES.
# If True, every existing variable will be kept even if its correlation is weak.
KEEP_EVERY_EXISTING_FEATURE = True

# Only used when KEEP_EVERY_EXISTING_FEATURE = False.
MIN_ABS_CORR = 0.05

# Important:
# These are CANDIDATE lags, not final lags.
# The script will select exactly one best lag from each list.
CANDIDATE_LAG_FEATURES = {
    "FILT. NTU": [0, 1, 2],
    "CLR": [0],
    "R/W FLOW": [0, 1],
    "T/W FLOW": [0, 1, 3],
    "C/W WELL LEVEL": [0, 1],
    "R/W NTU": [0, 1],
    "CL2": [0],
    "ALUM": [1, 2],
    "F/RIDE": [0, 6],
}

# Do not use PUMP DUTY / PUMP COUNT in this model version.


# ============================================================
# 1. Find merged.xlsx automatically
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
# 2. Read data
# ============================================================

df = pd.read_excel(DATA_PATH)

print("\nOriginal data shape:", df.shape)
print("Columns:")
print(df.columns.tolist())

# =========================
# Fill missing F/RIDE with 0
# =========================

if "F/RIDE" in df.columns:
    df["F/RIDE"] = pd.to_numeric(df["F/RIDE"], errors="coerce")
    missing_before = df["F/RIDE"].isna().sum()
    df["F/RIDE"] = df["F/RIDE"].fillna(0)
    missing_after = df["F/RIDE"].isna().sum()

    print("F/RIDE missing before filling:", missing_before)
    print("F/RIDE missing after filling:", missing_after)
else:
    print("F/RIDE column not found.")

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")


# ============================================================
# 3. Construct DATETIME
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

print("Datetime range:")
print(df["DATETIME"].min(), "to", df["DATETIME"].max())


# ============================================================
# 4. Construct OP_DATE
# ============================================================
# Operating day:
# 07:00, 09:00, ..., 23:00, next day 01:00, 03:00, 05:00
# If hour < 7, OP_DATE = calendar date - 1 day.

df["OP_DATE"] = df["DATETIME"].dt.date
mask_before_7 = df["DATETIME"].dt.hour < 7
df.loc[mask_before_7, "OP_DATE"] = (
    df.loc[mask_before_7, "DATETIME"] - pd.Timedelta(days=1)
).dt.date

print("\nOP_DATE check:")
print(df[["DATETIME", "OP_DATE"]].head(15).to_string(index=False))


# ============================================================
# 5. Build candidate lag features
# ============================================================

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

model_df = df[["DATETIME", "OP_DATE", TARGET_COL]].copy()
model_df = model_df.rename(columns={TARGET_COL: "target_NTU"})

candidate_lag_columns_by_feature = {}
missing_features = []

for feature, lag_list in CANDIDATE_LAG_FEATURES.items():
    if feature not in df.columns:
        missing_features.append(feature)
        continue

    df[feature] = pd.to_numeric(df[feature], errors="coerce")

    lag_cols = []
    for lag in lag_list:
        lag_col = f"{feature}_lag{lag}"
        model_df[lag_col] = df[feature].shift(lag)
        lag_cols.append(lag_col)

    candidate_lag_columns_by_feature[feature] = lag_cols

print("\nMissing candidate features skipped:")
print(missing_features)

print("\nCandidate lag columns:")
for feature, cols in candidate_lag_columns_by_feature.items():
    print(f"{feature}: {cols}")

if len(candidate_lag_columns_by_feature) == 0:
    raise ValueError("No candidate lag features were generated. Please check column names.")

candidate_lag_all_df = model_df.copy()


# ============================================================
# 6. Select one best lag per variable using training period only
# ============================================================
# Important:
# Selection is based only on the first 80% rows with observed target_NTU.
# This reduces data leakage compared with selecting lags using all data.

valid_target_df = model_df[model_df["target_NTU"].notna()].copy()

split_idx = int(len(valid_target_df) * 0.8)
train_part = valid_target_df.iloc[:split_idx].copy()

print("\nRows with observed target_NTU:", len(valid_target_df))
print("Rows used for lag selection training part:", len(train_part))

selected_columns = []
selection_records = []

for feature, lag_cols in candidate_lag_columns_by_feature.items():
    best_col = None
    best_lag = None
    best_corr = np.nan
    best_abs_corr = -1
    best_valid_n = 0

    for lag_col in lag_cols:
        temp = train_part[["target_NTU", lag_col]].dropna()

        if len(temp) < 10:
            continue

        corr_value = temp["target_NTU"].corr(temp[lag_col], method=CORR_METHOD)

        if pd.isna(corr_value):
            continue

        abs_corr = abs(corr_value)

        if abs_corr > best_abs_corr:
            best_abs_corr = abs_corr
            best_corr = corr_value
            best_col = lag_col
            best_lag = int(lag_col.split("_lag")[-1])
            best_valid_n = len(temp)

    if best_col is None:
        keep = False
    elif KEEP_EVERY_EXISTING_FEATURE:
        keep = True
    else:
        keep = best_abs_corr >= MIN_ABS_CORR

    if keep:
        selected_columns.append(best_col)

    selection_records.append({
        "base_feature": feature,
        "candidate_lags": str(CANDIDATE_LAG_FEATURES[feature]),
        "selected_lag": best_lag,
        "selected_lag_hours": best_lag * 2 if best_lag is not None else np.nan,
        "selected_feature": best_col,
        "corr_method": CORR_METHOD,
        "corr": best_corr,
        "abs_corr": best_abs_corr if best_abs_corr != -1 else np.nan,
        "valid_n": best_valid_n,
        "keep": keep,
        "keep_every_existing_feature": KEEP_EVERY_EXISTING_FEATURE,
        "min_abs_corr": MIN_ABS_CORR,
    })

selected_lag_summary_df = pd.DataFrame(selection_records)

print("\nSelected lag summary:")
print(selected_lag_summary_df.sort_values("abs_corr", ascending=False).to_string(index=False))


# ============================================================
# 7. Build final selected-one-lag modeling dataset
# ============================================================

selected_one_lag_model_df = model_df[
    ["DATETIME", "OP_DATE", "target_NTU"] + selected_columns
].copy()

print("\nFinal selected feature columns:")
for col in selected_columns:
    print(" -", col)

print("\nSelected-one-lag model data shape:", selected_one_lag_model_df.shape)
print(selected_one_lag_model_df.head(10).to_string(index=False))


# ============================================================
# 8. Missing audit
# ============================================================

missing_audit_df = pd.DataFrame({
    "column": selected_one_lag_model_df.columns,
    "missing_count": selected_one_lag_model_df.isna().sum().values,
    "missing_rate": selected_one_lag_model_df.isna().mean().values,
})

print("\nMissing audit:")
print(missing_audit_df.to_string(index=False))


# ============================================================
# 9. Save output files
# ============================================================

candidate_lag_all_path = DATA_DIR / "candidate_lag_all.xlsx"
selected_model_data_path = DATA_DIR / "selected_one_lag_model_data.xlsx"
summary_path = OUTPUT_DIR / "selected_one_lag_summary.xlsx"
missing_audit_path = OUTPUT_DIR / "selected_one_lag_missing_audit.xlsx"

candidate_lag_all_df.to_excel(candidate_lag_all_path, index=False)
selected_one_lag_model_df.to_excel(selected_model_data_path, index=False)
selected_lag_summary_df.to_excel(summary_path, index=False)
missing_audit_df.to_excel(missing_audit_path, index=False)

print("\nSaved files:")
print("1.", candidate_lag_all_path)
print("2.", selected_model_data_path)
print("3.", summary_path)
print("4.", missing_audit_path)

print("\nMain file for RF/XGBoost modeling:")
print(selected_model_data_path)

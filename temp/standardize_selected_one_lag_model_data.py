# ============================================================
# Standardize selected-one-lag modeling data for Problem 1
# ============================================================
# Input:
#   data/selected_one_lag_model_data.xlsx
#
# Output:
#   data/selected_one_lag_model_data_standardized.xlsx
#   outputs/problem1/selected_one_lag_standardization_info.xlsx
#   outputs/problem1/selected_one_lag_standardized_missing_audit.xlsx
#
# Key rule:
#   Standardize input features only.
#   Do NOT standardize target_NTU, so MAE/RMSE and final predictions remain in original NTU units.
#
# Leakage control:
#   Mean and std are calculated using only the training period
#   defined as the first 80% of rows with observed target_NTU.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)

# ============================================================
# 0. Settings
# ============================================================

INPUT_FILENAME = "selected_one_lag_model_data.xlsx"
OUTPUT_FILENAME = "selected_one_lag_model_data_standardized.xlsx"

TARGET_COL = "target_NTU"
TIME_COLS = ["DATETIME", "OP_DATE"]

TRAIN_RATIO = 0.8


# ============================================================
# 1. Locate selected_one_lag_model_data.xlsx automatically
# ============================================================

def find_input_excel(filename=INPUT_FILENAME):
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


DATA_PATH = find_input_excel()
DATA_DIR = DATA_PATH.parent

if DATA_DIR.name == "data":
    PROJECT_DIR = DATA_DIR.parent
else:
    PROJECT_DIR = DATA_DIR

OUTPUT_DIR = PROJECT_DIR / "outputs" / "problem1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Current working directory:", Path.cwd().resolve())
print("Using input file:", DATA_PATH)
print("Data directory:", DATA_DIR)
print("Output directory:", OUTPUT_DIR)


# ============================================================
# 2. Read selected-one-lag data
# ============================================================

df = pd.read_excel(DATA_PATH)

print("\nOriginal selected-one-lag data shape:", df.shape)
print("Columns:")
print(df.columns.tolist())

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")

for col in TIME_COLS:
    if col not in df.columns:
        raise ValueError(f"Required time column '{col}' not found.")

# Ensure datetime type if possible.
df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")

# Sort by time again for safety.
df = df.sort_values("DATETIME").reset_index(drop=True)


# ============================================================
# 3. Identify feature columns
# ============================================================

feature_cols = [
    col for col in df.columns
    if col not in TIME_COLS + [TARGET_COL]
]

if len(feature_cols) == 0:
    raise ValueError("No feature columns found for standardization.")

print("\nFeature columns to standardize:")
for col in feature_cols:
    print(" -", col)


# ============================================================
# 4. Define training period for standardization
# ============================================================
# Use only rows with observed target_NTU to determine split.
# Then calculate feature mean/std only on the training portion.
# This avoids using future test-period statistics.

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

valid_target_index = df.index[df[TARGET_COL].notna()].tolist()

if len(valid_target_index) == 0:
    raise ValueError("No observed target_NTU rows found.")

split_position = int(len(valid_target_index) * TRAIN_RATIO)
train_indices = valid_target_index[:split_position]

print("\nRows with observed target_NTU:", len(valid_target_index))
print("Training rows used for standardization:", len(train_indices))

train_df = df.loc[train_indices].copy()


# ============================================================
# 5. Standardize feature columns
# ============================================================
# X_z = (X - train_mean) / train_std
# target_NTU remains unchanged.

standardized_df = df[TIME_COLS + [TARGET_COL]].copy()
standardization_records = []

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

    train_mean = train_df[col].mean(skipna=True)
    train_std = train_df[col].std(skipna=True)

    z_col = f"{col}_z"

    if pd.isna(train_std) or train_std == 0:
        standardized_df[z_col] = np.nan
        status = "failed_zero_or_nan_std"
    else:
        standardized_df[z_col] = (df[col] - train_mean) / train_std
        status = "success"

    standardization_records.append({
        "original_feature": col,
        "standardized_feature": z_col,
        "train_mean": train_mean,
        "train_std": train_std,
        "status": status,
        "original_missing_count": df[col].isna().sum(),
        "original_missing_rate": df[col].isna().mean(),
        "standardized_missing_count": standardized_df[z_col].isna().sum(),
        "standardized_missing_rate": standardized_df[z_col].isna().mean(),
        "train_ratio": TRAIN_RATIO,
        "note": "Mean/std calculated using training period only."
    })

standardization_info_df = pd.DataFrame(standardization_records)

print("\nStandardization info:")
print(standardization_info_df.to_string(index=False))


# ============================================================
# 6. Missing audit
# ============================================================

missing_audit_df = pd.DataFrame({
    "column": standardized_df.columns,
    "missing_count": standardized_df.isna().sum().values,
    "missing_rate": standardized_df.isna().mean().values,
})

print("\nMissing audit after standardization:")
print(missing_audit_df.to_string(index=False))


# ============================================================
# 7. Save output files
# ============================================================

standardized_data_path = DATA_DIR / OUTPUT_FILENAME
standardization_info_path = OUTPUT_DIR / "selected_one_lag_standardization_info.xlsx"
missing_audit_path = OUTPUT_DIR / "selected_one_lag_standardized_missing_audit.xlsx"

standardized_df.to_excel(standardized_data_path, index=False)
standardization_info_df.to_excel(standardization_info_path, index=False)
missing_audit_df.to_excel(missing_audit_path, index=False)

print("\nSaved files:")
print("1.", standardized_data_path)
print("2.", standardization_info_path)
print("3.", missing_audit_path)

print("\nMain standardized file for modeling:")
print(standardized_data_path)

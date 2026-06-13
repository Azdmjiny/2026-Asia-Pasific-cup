# Generate selected-lag feature file for Problem 1

# ============================================================
# Generate selected-lag feature file for Problem 1
# ============================================================
# Function:
# 1. Read merged.xlsx
# 2. Construct DATETIME and OP_DATE
# 3. Convert PUMP DUTY to PUMP COUNT
# 4. Standardize candidate variables
# 5. Generate lag1-lag12 features
# 6. Select one best lag for each variable according to correlation
# 7. Save selected_lag.xlsx for later RF/XGBoost modeling
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)

# Safe display function:
# In Jupyter Notebook, display() is available.
# In a normal .py script, display() is not defined, so we fall back to print().
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

# -----------------------------
# User settings
# -----------------------------

MAX_LAG = 12
# Each lag means 2 hours.
# lag1 = 2 hours before, lag12 = 24 hours before.

CORR_METHOD = "spearman"
# Recommended: "spearman"
# Reason: water-treatment variables may have nonlinear monotonic relationships.
# You may also change this to "pearson".

MIN_ABS_CORR = 0.10
# If a variable's strongest absolute correlation is below this threshold,
# it will not be kept in selected_lag.xlsx.
# Suggested values:
# 0.00 = keep every variable
# 0.10 = moderate filtering
# 0.15 or 0.20 = stricter filtering

TARGET_COL = "NTU"

# Candidate variables used to generate lag features.
# Missing columns will be skipped automatically.
LAG_BASE_FEATURES = [
    "R/W NTU",
    "FILT. NTU",
    "R/W FLOW",
    "T/W FLOW",
    "C/W WELL LEVEL",
    "ALUM",
    "R/W PH",
    "PH",
    "CLR",
    "CL2",
    "RIVER LEVEL",
    "R/W CLR",
    "F/RIDE",
    "R/W PUMP COUNT",
    "T/W PUMP COUNT",
]


# ============================================================
# 1. Locate merged.xlsx automatically
# ============================================================

def find_merged_excel(filename="merged.xlsx"):
    """Search common project paths for merged.xlsx."""
    cwd = Path.cwd().resolve()

    candidate_paths = []

    # Search from current directory and its parents.
    search_roots = [cwd] + list(cwd.parents)

    for root in search_roots:
        candidate_paths.extend([
            root / "data" / filename,
            root / "codes" / "data" / filename,
            root / "2026-Asia-Pacific-cup" / "codes" / "data" / filename,
            root / filename,
        ])

    # Remove duplicates while preserving order.
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
    # This avoids relying on one fixed working directory.
    for p in cwd.rglob(filename):
        if p.name == filename:
            return p.resolve()

    raise FileNotFoundError(
        "Cannot find merged.xlsx. Please check whether merged.xlsx exists under data/ "
        "or codes/data/. Current working directory is: " + str(cwd)
    )


DATA_PATH = find_merged_excel()
DATA_DIR = DATA_PATH.parent

# Output directory:
# If merged.xlsx is in .../codes/data, output to .../codes/outputs/problem1
# Otherwise output to parent/outputs/problem1.
if DATA_DIR.name == "data" and DATA_DIR.parent.name == "codes":
    OUTPUT_DIR = DATA_DIR.parent / "outputs" / "problem1"
else:
    OUTPUT_DIR = DATA_DIR.parent / "outputs" / "problem1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Current working directory:", Path.cwd().resolve())
print("Using data file:", DATA_PATH)
print("Data directory:", DATA_DIR)
print("Output directory:", OUTPUT_DIR)


# ============================================================
# 2. Read data
# ============================================================

df = pd.read_excel(DATA_PATH)

print("Original data shape:", df.shape)
print("Columns:")
print(df.columns.tolist())

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found. Please check column names.")


# ============================================================
# 3. Construct DATETIME
# ============================================================

def construct_datetime(data):
    """Construct DATETIME from existing DATETIME or DATE/TIME columns."""
    data = data.copy()

    if "DATETIME" in data.columns:
        data["DATETIME"] = pd.to_datetime(data["DATETIME"], errors="coerce")
        return data

    possible_date_cols = ["DATE", "Date", "date"]
    possible_time_cols = ["TIME", "Time", "time"]

    date_col = None
    time_col = None

    for col in possible_date_cols:
        if col in data.columns:
            date_col = col
            break

    for col in possible_time_cols:
        if col in data.columns:
            time_col = col
            break

    if date_col is None or time_col is None:
        raise ValueError(
            "Cannot construct DATETIME. Please check whether DATE and TIME columns exist."
        )

    data["DATETIME"] = pd.to_datetime(
        data[date_col].astype(str) + " " + data[time_col].astype(str),
        errors="coerce"
    )

    return data


df = construct_datetime(df)

datetime_missing = df["DATETIME"].isna().sum()
print("Missing DATETIME:", datetime_missing)

if datetime_missing > 0:
    print("Warning: rows with missing DATETIME will remain, but sorting may place them at the end.")

df = df.sort_values("DATETIME").reset_index(drop=True)

print("Datetime range:")
print(df["DATETIME"].min(), "to", df["DATETIME"].max())


# ============================================================
# 4. Construct OP_DATE
# ============================================================
# Operating day definition:
# 07:00, 09:00, ..., 23:00, next day 01:00, 03:00, 05:00
# If hour < 7, OP_DATE = calendar date - 1 day.

df["OP_DATE"] = df["DATETIME"].dt.date

mask_before_7 = df["DATETIME"].dt.hour < 7
df.loc[mask_before_7, "OP_DATE"] = (
    df.loc[mask_before_7, "DATETIME"] - pd.Timedelta(days=1)
).dt.date

print("OP_DATE check:")
display(df[["DATETIME", "OP_DATE"]].head(15))


# ============================================================
# 5. Convert PUMP DUTY to PUMP COUNT
# ============================================================

def pump_duty_to_count(x):
    """
    Convert pump duty codes to number of running pumps.

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
    print("Created: R/W PUMP COUNT")
else:
    print("R/W PUMP DUTY not found. Skip R/W PUMP COUNT.")

if "T/W PUMP DUTY" in df.columns:
    df["T/W PUMP COUNT"] = df["T/W PUMP DUTY"].apply(pump_duty_to_count)
    print("Created: T/W PUMP COUNT")
else:
    print("T/W PUMP DUTY not found. Skip T/W PUMP COUNT.")


# ============================================================
# 6. Keep existing candidate variables
# ============================================================

lag_base_features = [col for col in LAG_BASE_FEATURES if col in df.columns]
missing_features = [col for col in LAG_BASE_FEATURES if col not in df.columns]

print("Existing lag base features:")
print(lag_base_features)

print("\nMissing candidate features skipped:")
print(missing_features)

if len(lag_base_features) == 0:
    raise ValueError("No lag base feature exists. Please check column names.")


# ============================================================
# 7. Build all standardized lag features: lag1-lag12
# ============================================================

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

lag_all_df = df[["DATETIME", "OP_DATE", TARGET_COL]].copy()
lag_all_df = lag_all_df.rename(columns={TARGET_COL: "target_NTU"})

standardization_records = []

for base_col in lag_base_features:
    # Convert to numeric before standardization.
    df[base_col] = pd.to_numeric(df[base_col], errors="coerce")

    mean_value = df[base_col].mean(skipna=True)
    std_value = df[base_col].std(skipna=True)

    z_col = f"{base_col}_z"

    if pd.isna(std_value) or std_value == 0:
        df[z_col] = np.nan
        status = "failed_zero_or_nan_std"
    else:
        df[z_col] = (df[base_col] - mean_value) / std_value
        status = "success"

    standardization_records.append({
        "base_feature": base_col,
        "z_feature": z_col,
        "mean": mean_value,
        "std": std_value,
        "status": status,
        "missing_count_original": df[base_col].isna().sum(),
        "missing_rate_original": df[base_col].isna().mean(),
    })

    # Generate lag1-lag12 only. No lag0.
    for lag in range(1, MAX_LAG + 1):
        lag_col = f"{base_col}_z_lag{lag}"
        lag_all_df[lag_col] = df[z_col].shift(lag)

standardization_info_df = pd.DataFrame(standardization_records)

# Safety check: no lag0 columns should exist.
lag0_columns = [col for col in lag_all_df.columns if "_lag0" in col]
if len(lag0_columns) > 0:
    raise ValueError(f"lag0 columns found unexpectedly: {lag0_columns}")

print("All lag dataframe shape:", lag_all_df.shape)
display(lag_all_df.head())


# ============================================================
# 8. Select one best lag for each variable
# ============================================================

selected_records = []
selected_columns = []

for base_col in lag_base_features:
    candidate_lag_cols = [
        f"{base_col}_z_lag{lag}"
        for lag in range(1, MAX_LAG + 1)
        if f"{base_col}_z_lag{lag}" in lag_all_df.columns
    ]

    best_lag = None
    best_lag_col = None
    best_corr = np.nan
    best_abs_corr = -1
    best_valid_n = 0

    for lag_col in candidate_lag_cols:
        temp = lag_all_df[["target_NTU", lag_col]].dropna().copy()
        valid_n = len(temp)

        if valid_n < 5:
            continue

        corr_value = temp["target_NTU"].corr(temp[lag_col], method=CORR_METHOD)

        if pd.isna(corr_value):
            continue

        abs_corr = abs(corr_value)

        if abs_corr > best_abs_corr:
            best_abs_corr = abs_corr
            best_corr = corr_value
            best_lag_col = lag_col
            best_valid_n = valid_n
            best_lag = int(lag_col.split("_lag")[-1])

    keep_feature = (
        best_lag_col is not None
        and pd.notna(best_abs_corr)
        and best_abs_corr >= MIN_ABS_CORR
    )

    if keep_feature:
        selected_columns.append(best_lag_col)

    selected_records.append({
        "base_feature": base_col,
        "selected_lag": best_lag,
        "selected_lag_hours": best_lag * 2 if best_lag is not None else np.nan,
        "selected_feature": best_lag_col,
        "corr_method": CORR_METHOD,
        "corr": best_corr,
        "abs_corr": best_abs_corr if best_abs_corr != -1 else np.nan,
        "valid_n": best_valid_n,
        "keep": keep_feature,
        "filter_threshold_min_abs_corr": MIN_ABS_CORR,
    })

selected_lag_summary_df = pd.DataFrame(selected_records)

print("Selected lag summary:")
display(selected_lag_summary_df.sort_values("abs_corr", ascending=False))


# ============================================================
# 9. Build selected_lag.xlsx
# ============================================================

selected_lag_df = lag_all_df[["DATETIME", "OP_DATE", "target_NTU"] + selected_columns].copy()

print("Selected columns:")
for col in selected_columns:
    print(" -", col)

print("Selected lag dataframe shape:", selected_lag_df.shape)
display(selected_lag_df.head())


# ============================================================
# 10. Missing-rate audit
# ============================================================

missing_audit_df = pd.DataFrame({
    "column": selected_lag_df.columns,
    "missing_count": selected_lag_df.isna().sum().values,
    "missing_rate": selected_lag_df.isna().mean().values,
})

display(missing_audit_df)


# ============================================================
# 11. Save output files
# ============================================================

lag_all_path = DATA_DIR / "lag_all.xlsx"
selected_lag_path = DATA_DIR / "selected_lag.xlsx"

summary_path = OUTPUT_DIR / "selected_lag_summary.xlsx"
standardization_info_path = OUTPUT_DIR / "selected_lag_standardization_info.xlsx"
missing_audit_path = OUTPUT_DIR / "selected_lag_missing_audit.xlsx"

lag_all_df.to_excel(lag_all_path, index=False)
selected_lag_df.to_excel(selected_lag_path, index=False)

selected_lag_summary_df.to_excel(summary_path, index=False)
standardization_info_df.to_excel(standardization_info_path, index=False)
missing_audit_df.to_excel(missing_audit_path, index=False)

print("Saved files:")
print("1.", lag_all_path)
print("2.", selected_lag_path)
print("3.", summary_path)
print("4.", standardization_info_path)
print("5.", missing_audit_path)

print("\nMain file for later modeling:")
print(selected_lag_path)

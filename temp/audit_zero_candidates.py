from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def normalized(value):
    if pd.isna(value):
        return None
    return re.sub(r"\s+", " ", str(value).strip().upper())


counts = {}


def record(column, value):
    if column is None:
        return
    item = counts.setdefault(
        column,
        {"blank": 0, "dash": 0, "numeric_zero": 0, "text_zero": 0, "value": 0},
    )
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        item["blank"] += 1
    elif isinstance(value, str) and value.strip() in {"-", "--", "—"}:
        item["dash"] += 1
    elif isinstance(value, str) and value.strip() in {"0", "0.0", "0.00"}:
        item["text_zero"] += 1
    elif isinstance(value, (int, float)) and value == 0:
        item["numeric_zero"] += 1
    else:
        item["value"] += 1


for path in sorted((DATA / "2025").glob("*.xlsx")):
    raw = pd.read_excel(path, header=None, engine="openpyxl")
    headers = [normalized(value) for value in raw.iloc[0]]
    for _, row in raw.iloc[1:].dropna(how="all").iterrows():
        for index, column in enumerate(headers):
            record(column, row.iloc[index])

for path in sorted((DATA / "2026").glob("*.xls")):
    workbook = pd.ExcelFile(path, engine="xlrd")
    for sheet_name in workbook.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="xlrd")
        headers = [normalized(value) for value in raw.iloc[0]]
        for _, row in raw.iloc[1:].dropna(how="all").iterrows():
            for index, column in enumerate(headers):
                record(column, row.iloc[index])

print(
    pd.DataFrame.from_dict(counts, orient="index")
    .sort_index()
    .to_string()
)

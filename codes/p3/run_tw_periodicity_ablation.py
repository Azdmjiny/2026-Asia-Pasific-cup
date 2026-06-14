from pathlib import Path
import argparse
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE_NOTEBOOK = ROOT / "codes/p3/p3_TW_Qin.ipynb"
OUTPUT_ROOT = ROOT / "codes/p3/outputs/problem3_formula_gru/TW_Qin_periodicity_ablation"
DEFAULT_VARIANTS = ("daily", "weekly", "daily_weekly")

PERIODIC_FEATURE_CODE = """
time_hour = (
    data_pre["DATETIME"].dt.hour
    + data_pre["DATETIME"].dt.minute / 60.0
)
weekly_phase = data_pre["DATETIME"].dt.dayofweek + time_hour / 24.0
data_pre["hour_sin"] = np.sin(2.0 * np.pi * time_hour / 24.0)
data_pre["hour_cos"] = np.cos(2.0 * np.pi * time_hour / 24.0)
data_pre["week_sin"] = np.sin(2.0 * np.pi * weekly_phase / 7.0)
data_pre["week_cos"] = np.cos(2.0 * np.pi * weekly_phase / 7.0)
"""

PERIODIC_VARIANTS = """
    "daily": BASE_FEATURES + ["hour_sin", "hour_cos"],
    "weekly": BASE_FEATURES + ["week_sin", "week_cos"],
    "daily_weekly": BASE_FEATURES + [
        "hour_sin", "hour_cos", "week_sin", "week_cos",
    ],
"""


def patch_notebook(variant):
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    replacements = 0
    feature_injected = False
    variants_injected = False

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))

        path_marker = "POSSIBLE_DATA_PATHS = ["
        if path_marker in source and str(ROOT / "data/merged.xlsx") not in source:
            source = source.replace(
                path_marker,
                (
                    "POSSIBLE_DATA_PATHS = [\n"
                    f'    Path(r"{ROOT / "data/merged.xlsx"}"),'
                ),
            )
            replacements += 1

        if 'FEATURE_VARIANT = "baseline"' in source:
            source = source.replace(
                'FEATURE_VARIANT = "baseline"',
                f'FEATURE_VARIANT = "{variant}"',
            )
            replacements += 1

        old_output = (
            'OUTPUT_DIR = Path(f"outputs/problem3_formula_gru/'
            'TW_Qin_clip2_ablation/{FEATURE_VARIANT}")'
        )
        if old_output in source:
            source = source.replace(
                old_output,
                (
                    f'OUTPUT_DIR = Path(r"{OUTPUT_ROOT}") / FEATURE_VARIANT'
                ),
            )
            replacements += 1

        marker = 'data_pre["WELL_LEVEL_CHANGE"] = well_level.diff().fillna(0.0)'
        if marker in source and "hour_sin" not in source:
            source = source.replace(
                marker,
                marker + "\n\n" + PERIODIC_FEATURE_CODE.strip(),
            )
            feature_injected = True

        dict_marker = 'FEATURE_VARIANTS = {\n'
        if dict_marker in source and '"daily": BASE_FEATURES' not in source:
            source = source.replace(
                dict_marker,
                dict_marker + PERIODIC_VARIANTS,
            )
            variants_injected = True

        cell["source"] = source.splitlines(keepends=True)

    if replacements != 3 or not feature_injected or not variants_injected:
        raise RuntimeError(
            "Notebook structure changed; periodicity patch could not be applied safely."
        )

    cutoff = next(
        (
            index
            for index, cell in enumerate(notebook["cells"])
            if "# 7. Required-date prediction helpers"
            in "".join(cell.get("source", []))
        ),
        None,
    )
    if cutoff is None:
        raise RuntimeError("Could not locate the post-evaluation notebook cutoff.")
    notebook["cells"] = notebook["cells"][:cutoff]

    notebook["metadata"]["kernelspec"] = {
        "display_name": "mcm",
        "language": "python",
        "name": "mcm",
    }
    return notebook


def run_variant(variant, timeout):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    generated_path = OUTPUT_ROOT / f"p3_TW_Qin_{variant}.ipynb"
    generated_path.write_text(
        json.dumps(patch_notebook(variant), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.kernel_name=mcm",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(generated_path),
    ]
    print(f"\n=== Running {variant} ===", flush=True)
    subprocess.run(command, cwd=ROOT / "codes/p3", check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "variants",
        nargs="*",
        default=DEFAULT_VARIANTS,
        choices=("daily", "weekly", "daily_weekly"),
    )
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    for variant in args.variants:
        run_variant(variant, args.timeout)


if __name__ == "__main__":
    main()

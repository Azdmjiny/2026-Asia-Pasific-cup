import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).with_name("p3_TW_Qin.ipynb")

CELL_MARKER = "# 6.0 Full-timeline true vs predicted visualization"

CELL_SOURCE = """# =========================
# 6.0 Full-timeline true vs predicted visualization
# =========================

def predict_split_for_full_plot(
    split_name,
    loader,
    current_values,
    base_times,
    target_times,
):
    pred_scaled, true_scaled, observed_mask = predict_loader(model, loader)
    pred_delta = inverse_delta(pred_scaled)
    true_delta_local = inverse_delta(true_scaled)
    pred_ntu = np.clip(current_values.reshape(-1, 1) + pred_delta, 0, None)
    true_ntu_local = current_values.reshape(-1, 1) + true_delta_local

    rows = []
    for sample_idx in range(len(current_values)):
        for h_idx, hour in enumerate(HORIZON_HOURS):
            rows.append({
                "split": split_name,
                "base_time": pd.Timestamp(base_times[sample_idx]),
                "target_time": pd.Timestamp(target_times[sample_idx, h_idx]),
                "horizon_hour": hour,
                "target_observed": bool(observed_mask[sample_idx, h_idx]),
                "true_NTU": (
                    true_ntu_local[sample_idx, h_idx]
                    if observed_mask[sample_idx, h_idx]
                    else np.nan
                ),
                "pred_NTU": pred_ntu[sample_idx, h_idx],
            })
    return pd.DataFrame(rows)


full_prediction_parts = [
    predict_split_for_full_plot(
        "Train",
        DataLoader(
            SequenceDataset(X_train, y_train, mask_train),
            batch_size=BATCH_SIZE,
            shuffle=False,
        ),
        current_train,
        base_train_times,
        target_train_times,
    ),
    predict_split_for_full_plot(
        "Validation",
        val_loader,
        current_val,
        base_val_times,
        target_val_times,
    ),
    predict_split_for_full_plot(
        "Test",
        test_loader,
        current_test,
        base_test_times,
        target_test_times,
    ),
]
full_predictions_df = pd.concat(full_prediction_parts, ignore_index=True)

full_predictions_path = OUTPUT_DIR / "full_timeline_predictions.xlsx"
full_predictions_df.to_excel(full_predictions_path, index=False)

split_styles = {
    "Train": {"linestyle": "-", "alpha": 0.72},
    "Validation": {"linestyle": "-.", "alpha": 0.82},
    "Test": {"linestyle": "--", "alpha": 0.95},
}
split_order = ["Train", "Validation", "Test"]

fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=True, sharey=True)
axes = axes.ravel()

for h_idx, hour in enumerate(HORIZON_HOURS):
    ax = axes[h_idx]
    horizon_data = full_predictions_df[
        (full_predictions_df["horizon_hour"] == hour)
        & full_predictions_df["target_observed"]
    ].copy()

    for split_name in split_order:
        split_data = (
            horizon_data[horizon_data["split"] == split_name]
            .sort_values("target_time")
        )
        style = split_styles[split_name]
        ax.plot(
            split_data["target_time"],
            split_data["true_NTU"],
            color="#202020",
            linewidth=1.05,
            linestyle=style["linestyle"],
            alpha=style["alpha"],
            label=f"True - {split_name}",
        )
        ax.plot(
            split_data["target_time"],
            split_data["pred_NTU"],
            color="#1976D2",
            linewidth=1.05,
            linestyle=style["linestyle"],
            alpha=style["alpha"],
            label=f"Predicted - {split_name}",
        )

    ax.axvline(
        train_df["DATETIME"].max(),
        color="#F59E0B",
        linewidth=1.0,
        alpha=0.8,
    )
    ax.axvline(
        val_df["DATETIME"].max(),
        color="#DC2626",
        linewidth=1.0,
        alpha=0.8,
    )
    ax.set_title(f"Forecast horizon: {hour} h")
    ax.set_ylabel("Outlet-water NTU")
    ax.grid(alpha=0.22)

for ax in axes[-2:]:
    ax.set_xlabel("Target time")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.965),
)
fig.suptitle(
    "P3 TW full-timeline true vs predicted NTU\\n"
    "Train: solid | Validation: dash-dot | Test: dashed",
    fontsize=16,
    y=0.995,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])

full_timeline_plot_path = OUTPUT_DIR / "full_timeline_true_vs_predicted.png"
fig.savefig(full_timeline_plot_path, dpi=220, bbox_inches="tight")
plt.show()

print("Saved full-timeline predictions to:", full_predictions_path)
print("Saved full-timeline plot to:", full_timeline_plot_path)
"""


def main():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    for cell in cells:
        if CELL_MARKER in "".join(cell.get("source", [])):
            print("Full-timeline plot cell already exists.")
            return

    insert_at = None
    for index, cell in enumerate(cells):
        source = "".join(cell.get("source", []))
        if 'print("Saved test predictions to:", test_pred_path)' in source:
            insert_at = index + 1
            break

    if insert_at is None:
        raise RuntimeError("Could not locate the test-prediction evaluation cell.")

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CELL_SOURCE.splitlines(keepends=True),
    }
    cells.insert(insert_at, new_cell)
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Inserted full-timeline plot cell at notebook cell {insert_at}.")


if __name__ == "__main__":
    main()

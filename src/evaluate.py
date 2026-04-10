"""
Generate evaluation plots: accuracy bar charts and confusion matrices.

Usage:
    python src/evaluate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "face_ratio", "jaw_width_ratio", "forehead_width_ratio", "chin_length_ratio",
    "eye_aspect_ratio_left", "eye_aspect_ratio_right", "eye_width_ratio", "inter_eye_distance_ratio",
    "left_brow_height_ratio", "right_brow_height_ratio", "left_brow_angle", "right_brow_angle",
    "nose_width_ratio", "nose_length_ratio", "mouth_width_ratio", "mouth_open_ratio",
]

LABEL_COLS = ["face_shape", "eye_type", "mouth_type", "brow_type", "nose_type"]


def plot_accuracy_bars(results: dict, output_dir: Path) -> None:
    """Bar chart comparing model accuracies per category."""
    model_names = [k for k in next(iter(results.values())).keys()
                   if k not in ("best_model", "best_accuracy", "classes")]

    fig, axes = plt.subplots(1, len(LABEL_COLS), figsize=(4 * len(LABEL_COLS), 5), sharey=True)
    if len(LABEL_COLS) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for ax, category in zip(axes, LABEL_COLS):
        cat_results = results[category]
        accs = [cat_results[m]["mean_accuracy"] for m in model_names]
        stds = [cat_results[m]["std"] for m in model_names]

        bars = ax.bar(range(len(model_names)), accs, yerr=stds, color=colors,
                      capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_title(category, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.axhline(y=1.0 / len(cat_results["classes"]), color="red", linestyle="--",
                    linewidth=0.8, label="random baseline")
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Accuracy (5-fold CV)")
    fig.suptitle("Model Comparison by Category", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved accuracy_comparison.png")


def plot_confusion_matrices(X: np.ndarray, df: pd.DataFrame, results: dict,
                            output_dir: Path) -> None:
    """Confusion matrix for the best model per category (using CV predictions)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fig, axes = plt.subplots(1, len(LABEL_COLS), figsize=(4.5 * len(LABEL_COLS), 4))
    if len(LABEL_COLS) == 1:
        axes = [axes]

    for ax, category in zip(axes, LABEL_COLS):
        le = LabelEncoder()
        y = le.fit_transform(df[category].values)

        best_name = results[category]["best_model"]
        model_path = PROJECT_ROOT / "models" / "trained" / f"{category}.pkl"
        pipeline = joblib.load(model_path)

        y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{category}\n({best_name})", fontsize=10, fontweight="bold")
        ax.set_xticklabels(le.classes_, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(le.classes_, fontsize=7)

    fig.suptitle("Confusion Matrices (Best Model per Category)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close()
    print(f"  Saved confusion_matrices.png")


def main() -> None:
    results_path = PROJECT_ROOT / "outputs" / "training_results.json"
    data_path = PROJECT_ROOT / "data" / "processed" / "features_labeled.csv"

    with open(results_path) as f:
        results = json.load(f)

    df = pd.read_csv(data_path)
    X = df[FEATURE_COLS].values

    output_dir = PROJECT_ROOT / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots:")
    plot_accuracy_bars(results, output_dir)
    plot_confusion_matrices(X, df, results, output_dir)
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

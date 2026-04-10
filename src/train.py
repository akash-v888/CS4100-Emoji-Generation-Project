"""
Train per-category classifiers and compare KNN, Random Forest, SVM, Decision Tree.

Usage:
    python src/train.py
    python src/train.py --input data/processed/features_labeled.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "face_ratio", "jaw_width_ratio", "forehead_width_ratio", "chin_length_ratio",
    "eye_aspect_ratio_left", "eye_aspect_ratio_right", "eye_width_ratio", "inter_eye_distance_ratio",
    "left_brow_height_ratio", "right_brow_height_ratio", "left_brow_angle", "right_brow_angle",
    "nose_width_ratio", "nose_length_ratio", "mouth_width_ratio", "mouth_open_ratio",
]

LABEL_COLS = ["face_shape", "eye_type", "mouth_type", "brow_type", "nose_type"]

MODELS = {
    "KNN_3": KNeighborsClassifier(n_neighbors=3),
    "KNN_5": KNeighborsClassifier(n_neighbors=5),
    "KNN_7": KNeighborsClassifier(n_neighbors=7),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM_RBF": SVC(kernel="rbf", random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
}


def train_category(X: np.ndarray, y: np.ndarray, category: str, model_dir: Path,
                   cv: StratifiedKFold) -> dict:
    """Train all models for one category, return results and save best."""
    results = {}
    best_score = -1.0
    best_name = ""
    best_pipeline = None

    for name, base_model in MODELS.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", base_model),
        ])

        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        mean_acc = float(scores.mean())
        std_acc = float(scores.std())

        results[name] = {"mean_accuracy": round(mean_acc, 4), "std": round(std_acc, 4)}

        if mean_acc > best_score:
            best_score = mean_acc
            best_name = name
            best_pipeline = pipeline

    # Retrain best model on full data and save
    best_pipeline.fit(X, y)
    model_path = model_dir / f"{category}.pkl"
    joblib.dump(best_pipeline, model_path)

    results["best_model"] = best_name
    results["best_accuracy"] = round(best_score, 4)
    print(f"  {category}: best={best_name} ({best_score:.4f})")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train emoji classifiers.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "processed" / "features_labeled.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples, {len(FEATURE_COLS)} features.\n")

    X = df[FEATURE_COLS].values

    model_dir = PROJECT_ROOT / "models" / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_results = {}
    label_encoders = {}

    print("Training per-category classifiers (5-fold stratified CV):")
    for category in LABEL_COLS:
        le = LabelEncoder()
        y = le.fit_transform(df[category].values)
        label_encoders[category] = le

        n_classes = len(le.classes_)
        print(f"\n  {category}: {n_classes} classes {list(le.classes_)}")

        all_results[category] = train_category(X, y, category, model_dir, cv)
        all_results[category]["classes"] = list(le.classes_)

    # Save label encoders
    joblib.dump(label_encoders, model_dir / "label_encoders.pkl")

    # Save results
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nModels saved to: {model_dir}")
    print(f"Results saved to: {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for category in LABEL_COLS:
        r = all_results[category]
        print(f"  {category:15s} → {r['best_model']:15s}  acc={r['best_accuracy']:.4f}")


if __name__ == "__main__":
    main()

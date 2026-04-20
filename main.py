"""
Customer Churn Prediction Pipeline
===================================
A structured, production-ready ML pipeline for predicting telecom customer churn.

Models:  Logistic Regression  |  Random Forest
Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATA_PATH = "Telco-Customer-Churn.csv"
IMAGES_DIR = "images"
MODELS_DIR = "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns that are truly numeric in the raw CSV
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges"]

# TotalCharges is stored as a string in the CSV; we handle it separately
TOTAL_CHARGES_COL = "TotalCharges"

# Columns to drop before modelling
DROP_COLS = ["customerID"]

TARGET = "Churn"

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")


# ──────────────────────────────────────────────
# 1. Data Loading & Cleaning
# ──────────────────────────────────────────────
def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load the Telco dataset and perform basic cleaning."""
    df = pd.read_csv(path)

    # Drop identifier column
    df.drop(columns=DROP_COLS, inplace=True)

    # TotalCharges has whitespace entries → coerce to numeric (NaN for blanks)
    df[TOTAL_CHARGES_COL] = pd.to_numeric(df[TOTAL_CHARGES_COL], errors="coerce")

    # Encode target: Yes → 1, No → 0
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    print(f"✅ Data loaded — shape: {df.shape}")
    print(f"   Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    print(f"   Churn rate: {df[TARGET].mean():.2%}")
    return df


# ──────────────────────────────────────────────
# 2. Build sklearn Pipelines (prevents leakage)
# ──────────────────────────────────────────────
def build_pipelines(numeric_cols: list, categorical_cols: list) -> dict:
    """
    Return a dict of named sklearn Pipelines.
    Preprocessing is embedded so it is fit ONLY on training data.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
                ]),
                categorical_cols,
            ),
        ]
    )

    pipelines = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    return pipelines


# ──────────────────────────────────────────────
# 3. Evaluate a trained pipeline
# ──────────────────────────────────────────────
def evaluate_model(name: str, pipeline, X_test, y_test) -> dict:
    """Print and return key metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Recall   : {rec:.4f}  ← (% of churners caught)")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    return {"accuracy": acc, "recall": rec, "f1": f1, "roc_auc": auc,
            "y_pred": y_pred, "y_prob": y_prob}


# ──────────────────────────────────────────────
# 4. Plotting helpers (save to disk, no blocking)
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, model_name: str, save_dir: str):
    """Save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"), dpi=150)
    plt.close(fig)
    print(f"  📊 Confusion matrix saved → {save_dir}/")


def plot_roc_curves(results: dict, y_test, save_dir: str):
    """Save ROC curves for all models on a single chart."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "roc_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  📊 ROC curves saved → {save_dir}/")


def plot_feature_importance(pipeline, feature_names: list, save_dir: str):
    """Save feature importance bar chart from the Random Forest pipeline."""
    clf = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Get transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_features = cat_encoder.get_feature_names_out().tolist()
    num_features = preprocessor.transformers_[0][2]  # numeric col names
    all_features = list(num_features) + cat_features

    importance = pd.Series(clf.feature_importances_, index=all_features)
    importance = importance.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    importance.tail(15).plot(kind="barh", ax=ax, color=sns.color_palette("viridis", 15))
    ax.set_title("Top 15 Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"  📊 Feature importance saved → {save_dir}/")


def plot_model_comparison(results: dict, save_dir: str):
    """Save a grouped bar chart comparing model metrics."""
    metrics = ["accuracy", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Recall", "F1-Score", "ROC-AUC"]
    model_names = list(results.keys())

    x = np.arange(len(metrics))
    width = 0.3
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, name in enumerate(model_names):
        values = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=name)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Performance Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  📊 Model comparison chart saved → {save_dir}/")


# ──────────────────────────────────────────────
# 5. Model Persistence
# ──────────────────────────────────────────────
def save_model(pipeline, name: str, save_dir: str):
    """Serialize a fitted pipeline to disk via joblib."""
    filename = f"{name.lower().replace(' ', '_')}_pipeline.pkl"
    filepath = os.path.join(save_dir, filename)
    joblib.dump(pipeline, filepath)
    print(f"  💾 Model saved → {filepath}")


# ──────────────────────────────────────────────
# 6. Main entry-point
# ──────────────────────────────────────────────
def main():
    # Create output directories
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load & clean ──
    df = load_and_clean_data(DATA_PATH)

    # ── Identify feature groups ──
    numeric_cols = NUMERIC_FEATURES + [TOTAL_CHARGES_COL]
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols and c != TARGET
    ]
    print(f"\n  Numeric features   ({len(numeric_cols)}): {numeric_cols}")
    print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # ── Split ──
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"\n  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # ── Build, train & evaluate ──
    pipelines = build_pipelines(numeric_cols, categorical_cols)
    results = {}

    for name, pipe in pipelines.items():
        # Cross-validated score on training set
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
        print(f"\n  {name} — 5-fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full training set, evaluate on held-out test set
        pipe.fit(X_train, y_train)
        res = evaluate_model(name, pipe, X_test, y_test)
        results[name] = res

        # Confusion matrix per model
        plot_confusion_matrix(y_test, res["y_pred"], name, IMAGES_DIR)

        # Save model to disk
        save_model(pipe, name, MODELS_DIR)

    # ── Comparative visualizations ──
    plot_roc_curves(results, y_test, IMAGES_DIR)
    plot_model_comparison(results, IMAGES_DIR)

    # Feature importance (Random Forest only)
    plot_feature_importance(pipelines["Random Forest"], X.columns.tolist(), IMAGES_DIR)

    print("\n" + "=" * 50)
    print("  ✅  Pipeline complete — all outputs saved.")
    print("=" * 50)


if __name__ == "__main__":
    main()

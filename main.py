"""
Customer Churn Prediction Pipeline  (OOP Architecture)
========================================================
A production-grade, Object-Oriented ML pipeline for predicting
telecom customer churn.

Classes:
    Config         – Centralised, immutable configuration dataclass.
    DataLoader     – Reads, cleans, splits, and introspects the dataset.
    ModelTrainer   – Builds sklearn Pipelines, cross-validates, and fits models.
    Evaluator      – Scores models, generates all visualisations, serialises artifacts.

Models:
    Logistic Regression (sklearn)  |  Random Forest (sklearn)
    Custom Logistic Regression (pure NumPy — see custom_model.py)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

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

from custom_model import CustomLogisticRegression

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")


# ╔══════════════════════════════════════════════════════════╗
# ║  CONFIG                                                  ║
# ╚══════════════════════════════════════════════════════════╝
@dataclass(frozen=True)
class Config:
    """Immutable, single-source-of-truth configuration object."""

    data_path: str = "Telco-Customer-Churn.csv"
    images_dir: str = "images"
    models_dir: str = "models"
    test_size: float = 0.2
    random_state: int = 42
    numeric_features: tuple = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
    drop_cols: tuple = ("customerID",)
    target: str = "Churn"


# ╔══════════════════════════════════════════════════════════╗
# ║  DATA LOADER                                             ║
# ╚══════════════════════════════════════════════════════════╝
class DataLoader:
    """
    Responsible for:
      1. Loading the raw CSV
      2. Cleaning and type-casting columns
      3. Identifying numeric vs. categorical feature groups
      4. Generating a stratified train/test split
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.df: pd.DataFrame | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []

    # ── public API ──────────────────────────────

    def load(self) -> DataLoader:
        """Read CSV, clean types, encode target."""
        df = pd.read_csv(self.cfg.data_path)
        df.drop(columns=list(self.cfg.drop_cols), inplace=True)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df[self.cfg.target] = df[self.cfg.target].map({"Yes": 1, "No": 0})
        self.df = df

        print(f"[DataLoader] Loaded {df.shape[0]} rows x {df.shape[1]} cols")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            print(f"[DataLoader] Missing values:\n{missing.to_string()}")
        print(f"[DataLoader] Churn rate: {df[self.cfg.target].mean():.2%}")
        return self

    def identify_features(self) -> DataLoader:
        """Separate numeric and categorical columns."""
        self.numeric_cols = list(self.cfg.numeric_features)
        self.categorical_cols = [
            c for c in self.df.columns
            if c not in self.numeric_cols and c != self.cfg.target
        ]
        print(f"[DataLoader] Numeric  ({len(self.numeric_cols)}): {self.numeric_cols}")
        print(f"[DataLoader] Category ({len(self.categorical_cols)}): {self.categorical_cols}")
        return self

    def split(self) -> DataLoader:
        """Stratified train/test split."""
        X = self.df.drop(columns=[self.cfg.target])
        y = self.df[self.cfg.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y,
        )
        print(f"[DataLoader] Train: {self.X_train.shape[0]} | Test: {self.X_test.shape[0]}")
        return self


# ╔══════════════════════════════════════════════════════════╗
# ║  MODEL TRAINER                                           ║
# ╚══════════════════════════════════════════════════════════╝
class ModelTrainer:
    """
    Responsible for:
      1. Building leakage-proof sklearn Pipelines (with ColumnTransformer)
      2. Cross-validating each pipeline on the training set
      3. Fitting on the full training set
      4. Training the custom NumPy Logistic Regression alongside sklearn models
    """

    def __init__(self, config: Config, data: DataLoader):
        self.cfg = config
        self.data = data
        self.pipelines: dict[str, Pipeline] = {}
        self.custom_model: CustomLogisticRegression | None = None
        self._preprocessor: ColumnTransformer | None = None

    # ── internal helpers ────────────────────────

    def _build_preprocessor(self) -> ColumnTransformer:
        """Shared ColumnTransformer used by all sklearn pipelines."""
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    self.data.numeric_cols,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
                    ]),
                    self.data.categorical_cols,
                ),
            ]
        )

    # ── public API ──────────────────────────────

    def build(self) -> ModelTrainer:
        """Construct all model pipelines."""
        preprocessor = self._build_preprocessor()
        self._preprocessor = preprocessor

        self.pipelines = {
            "Logistic Regression": Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=self.cfg.random_state,
                )),
            ]),
            "Random Forest": Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=self.cfg.random_state,
                )),
            ]),
        }
        print(f"[ModelTrainer] Built {len(self.pipelines)} sklearn pipelines")
        return self

    def cross_validate(self, cv: int = 5) -> ModelTrainer:
        """Run k-fold cross-validation on every sklearn pipeline."""
        for name, pipe in self.pipelines.items():
            scores = cross_val_score(
                pipe, self.data.X_train, self.data.y_train,
                cv=cv, scoring="roc_auc",
            )
            print(f"[ModelTrainer] {name} — {cv}-fold CV ROC-AUC: "
                  f"{scores.mean():.4f} +/- {scores.std():.4f}")
        return self

    def train(self) -> ModelTrainer:
        """Fit all sklearn pipelines on the full training set."""
        for name, pipe in self.pipelines.items():
            pipe.fit(self.data.X_train, self.data.y_train)
            print(f"[ModelTrainer] {name} — trained")
        return self

    def train_custom_model(self) -> ModelTrainer:
        """
        Train the from-scratch NumPy Logistic Regression.
        We manually apply the same preprocessing that the sklearn
        ColumnTransformer would apply, to keep comparison fair.
        """
        # Use the already-fitted preprocessor from the sklearn pipeline
        preprocessor = self.pipelines["Logistic Regression"].named_steps["preprocessor"]
        X_train_transformed = preprocessor.transform(self.data.X_train)
        if hasattr(X_train_transformed, "toarray"):
            X_train_transformed = X_train_transformed.toarray()

        self.custom_model = CustomLogisticRegression(
            learning_rate=0.05,
            n_iterations=3000,
            regularization=0.01,
            verbose=True,
        )
        print("[ModelTrainer] Training Custom NumPy Logistic Regression...")
        self.custom_model.fit(X_train_transformed, self.data.y_train.values)
        print("[ModelTrainer] Custom model — trained")
        return self


# ╔══════════════════════════════════════════════════════════╗
# ║  EVALUATOR                                               ║
# ╚══════════════════════════════════════════════════════════╝
class Evaluator:
    """
    Responsible for:
      1. Scoring each model (Accuracy, Recall, F1, ROC-AUC)
      2. Generating and saving all visualisations
      3. Serialising trained model artifacts to disk
    """

    def __init__(self, config: Config, data: DataLoader, trainer: ModelTrainer):
        self.cfg = config
        self.data = data
        self.trainer = trainer
        self.results: dict[str, dict] = {}

    # ── internal helpers ────────────────────────

    @staticmethod
    def _score(y_true, y_pred, y_prob) -> dict:
        """Compute all metrics for a single model."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    def _print_report(self, name: str, metrics: dict):
        """Pretty-print a single model's evaluation report."""
        print(f"\n{'=' * 55}")
        print(f"  {name}")
        print(f"{'=' * 55}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}  <- (% of churners caught)")
        print(f"  F1-Score : {metrics['f1']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
        y_test = self.data.y_test
        print(classification_report(
            y_test, metrics["y_pred"],
            target_names=["No Churn", "Churn"],
        ))

    # ── public API: scoring ─────────────────────

    def evaluate_sklearn_models(self) -> Evaluator:
        """Score every sklearn pipeline on the test set."""
        for name, pipe in self.trainer.pipelines.items():
            y_pred = pipe.predict(self.data.X_test)
            y_prob = pipe.predict_proba(self.data.X_test)[:, 1]
            metrics = self._score(self.data.y_test, y_pred, y_prob)
            self.results[name] = metrics
            self._print_report(name, metrics)
        return self

    def evaluate_custom_model(self) -> Evaluator:
        """Score the custom NumPy model on the test set."""
        preprocessor = self.trainer.pipelines["Logistic Regression"].named_steps["preprocessor"]
        X_test_transformed = preprocessor.transform(self.data.X_test)
        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()

        model = self.trainer.custom_model
        y_pred = model.predict(X_test_transformed)
        y_prob = model.predict_proba(X_test_transformed)[:, 1]
        metrics = self._score(self.data.y_test, y_pred, y_prob)
        self.results["Custom LR (NumPy)"] = metrics
        self._print_report("Custom LR (NumPy)", metrics)
        return self

    # ── public API: visualisation ───────────────

    def plot_confusion_matrices(self) -> Evaluator:
        """Save a confusion matrix heatmap for every model."""
        os.makedirs(self.cfg.images_dir, exist_ok=True)
        for name, res in self.results.items():
            cm = confusion_matrix(self.data.y_test, res["y_pred"])
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix - {name}")
            fig.tight_layout()
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            fig.savefig(
                os.path.join(self.cfg.images_dir, f"confusion_matrix_{safe_name}.png"),
                dpi=150,
            )
            plt.close(fig)
        print(f"[Evaluator] Confusion matrices saved -> {self.cfg.images_dir}/")
        return self

    def plot_roc_curves(self) -> Evaluator:
        """Save all ROC curves on a single plot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, res in self.results.items():
            fpr, tpr, _ = roc_curve(self.data.y_test, res["y_prob"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(self.cfg.images_dir, "roc_curves.png"), dpi=150)
        plt.close(fig)
        print(f"[Evaluator] ROC curves saved -> {self.cfg.images_dir}/")
        return self

    def plot_feature_importance(self) -> Evaluator:
        """Save the top-15 feature importance chart from the Random Forest."""
        pipe = self.trainer.pipelines["Random Forest"]
        clf = pipe.named_steps["classifier"]
        preprocessor = pipe.named_steps["preprocessor"]

        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_features = cat_encoder.get_feature_names_out().tolist()
        num_features = list(preprocessor.transformers_[0][2])
        all_features = num_features + cat_features

        importance = pd.Series(clf.feature_importances_, index=all_features)
        importance = importance.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        importance.tail(15).plot(kind="barh", ax=ax, color=sns.color_palette("viridis", 15))
        ax.set_title("Top 15 Feature Importances (Random Forest)")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(os.path.join(self.cfg.images_dir, "feature_importance.png"), dpi=150)
        plt.close(fig)
        print(f"[Evaluator] Feature importance saved -> {self.cfg.images_dir}/")
        return self

    def plot_model_comparison(self) -> Evaluator:
        """Save a grouped bar chart comparing all models across all metrics."""
        metrics_keys = ["accuracy", "recall", "f1", "roc_auc"]
        labels = ["Accuracy", "Recall", "F1-Score", "ROC-AUC"]
        model_names = list(self.results.keys())

        x = np.arange(len(metrics_keys))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 5))

        for i, name in enumerate(model_names):
            values = [self.results[name][m] for m in metrics_keys]
            bars = ax.bar(x + i * width, values, width, label=name)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.15)
        ax.set_title("Model Performance Comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.cfg.images_dir, "model_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"[Evaluator] Model comparison chart saved -> {self.cfg.images_dir}/")
        return self

    def plot_custom_loss_curve(self) -> Evaluator:
        """Save the gradient-descent loss curve for the custom model."""
        if self.trainer.custom_model is None:
            return self
        history = self.trainer.custom_model.loss_history
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history, color="#6C5CE7", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Binary Cross-Entropy Loss")
        ax.set_title("Custom LR (NumPy) — Training Loss Curve")
        fig.tight_layout()
        fig.savefig(os.path.join(self.cfg.images_dir, "custom_lr_loss_curve.png"), dpi=150)
        plt.close(fig)
        print(f"[Evaluator] Custom LR loss curve saved -> {self.cfg.images_dir}/")
        return self

    # ── public API: persistence ─────────────────

    def save_models(self) -> Evaluator:
        """Serialise all sklearn pipelines to disk."""
        os.makedirs(self.cfg.models_dir, exist_ok=True)
        for name, pipe in self.trainer.pipelines.items():
            safe_name = name.lower().replace(" ", "_")
            path = os.path.join(self.cfg.models_dir, f"{safe_name}_pipeline.pkl")
            joblib.dump(pipe, path)
            print(f"[Evaluator] Model saved -> {path}")
        return self


# ╔══════════════════════════════════════════════════════════╗
# ║  ORCHESTRATOR                                            ║
# ╚══════════════════════════════════════════════════════════╝
def main():
    """
    Thin orchestrator: instantiate config -> data -> trainer -> evaluator
    and chain their methods in a clean, readable sequence.
    """
    # 1. Configuration
    config = Config()

    # 2. Data
    data = (
        DataLoader(config)
        .load()
        .identify_features()
        .split()
    )

    # 3. Training
    trainer = (
        ModelTrainer(config, data)
        .build()
        .cross_validate(cv=5)
        .train()
        .train_custom_model()
    )

    # 4. Evaluation & visualisation
    evaluator = (
        Evaluator(config, data, trainer)
        .evaluate_sklearn_models()
        .evaluate_custom_model()
        .plot_confusion_matrices()
        .plot_roc_curves()
        .plot_feature_importance()
        .plot_model_comparison()
        .plot_custom_loss_curve()
        .save_models()
    )

    print("\n" + "=" * 55)
    print("  Pipeline complete — all outputs saved.")
    print("=" * 55)


if __name__ == "__main__":
    main()

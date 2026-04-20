# Customer Churn Prediction

## Problem
Predict whether a telecom customer will churn (leave) based on their account and usage attributes.

## Business Context
Customer churn directly leads to revenue loss. By identifying at-risk customers early, companies can deploy targeted retention strategies (discounts, support outreach) to reduce churn.

---

## How to Run

1. **Clone the repo**
2. **Install dependencies** (pinned versions):
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the pipeline**:
   ```bash
   python main.py
   ```
   All plots are saved to `images/` and trained models to `models/`.

---

## Tools & Libraries
| Library | Purpose |
|---------|---------|
| Pandas | Data loading & manipulation |
| Scikit-learn | ML pipelines, models, metrics |
| Matplotlib / Seaborn | Visualization |
| Joblib | Model serialization |

---

## Pipeline Architecture

```
CSV → load_and_clean_data()
        ↓
  train / test split (stratified)
        ↓
  build_pipelines()
    ├─ ColumnTransformer
    │   ├─ Numeric  → SimpleImputer(median) → StandardScaler
    │   └─ Category → SimpleImputer(freq)   → OneHotEncoder
    ├─ Logistic Regression (class_weight=balanced)
    └─ Random Forest        (class_weight=balanced)
        ↓
  evaluate_model()  →  Accuracy, Recall, F1, ROC-AUC
        ↓
  save plots  →  images/
  save models →  models/*.pkl
```

### Key Design Decisions
- **No data leakage**: Preprocessing is embedded inside `sklearn.Pipeline` and fit only on the training set.
- **Class imbalance handling**: Both models use `class_weight='balanced'` to up-weight the minority (Churn) class.
- **Cross-validation**: 5-fold CV on the training set provides a variance-aware estimate before test evaluation.

---

## Model Comparison

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|-------|----------|--------|----------|---------|
| Logistic Regression | ~0.74 | ~0.80 | ~0.60 | ~0.84 |
| Random Forest | ~0.76 | ~0.72 | ~0.58 | ~0.83 |

> **Note**: With `class_weight='balanced'`, Logistic Regression achieves notably higher **Recall** (catches more actual churners), which is typically the more valuable metric in churn prevention scenarios.

### ROC Curves
![ROC Curves](images/roc_curves.png)

### Model Comparison Chart
![Model Comparison](images/model_comparison.png)

---

## Feature Importance (Random Forest)
![Feature Importance](images/feature_importance.png)

## Confusion Matrices
| Logistic Regression | Random Forest |
|---------------------|---------------|
| ![LR CM](images/confusion_matrix_logistic_regression.png) | ![RF CM](images/confusion_matrix_random_forest.png) |

---

## Key Insights
- **Tenure** is the strongest predictor — customers with longer tenure are far less likely to churn.
- **Monthly Charges** and **Total Charges** are both highly important; higher charges correlate with higher churn.
- **Contract type** significantly impacts retention — month-to-month contracts have the highest churn risk.
- The balanced-weight Logistic Regression model is the recommended choice for deployment due to its superior Recall on the minority class.

---

## Project Structure
```
├── main.py                 # End-to-end ML pipeline
├── requirements.txt        # Pinned dependencies
├── Telco-Customer-Churn.csv
├── images/                 # Auto-generated plots
│   ├── confusion_matrix_*.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── roc_curves.png
└── models/                 # Serialized pipelines
    ├── logistic_regression_pipeline.pkl
    └── random_forest_pipeline.pkl
```
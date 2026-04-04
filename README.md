# Customer Churn Prediction

## 📌 Problem
Predict whether a customer will churn (leave) or not.

## 📌 Business Problem
Customer churn leads to revenue loss.  
Goal: Identify customers likely to leave so companies can take action.

## ▶️ How to Run

1. Clone the repo
2. Install requirements:
   pip install -r requirements.txt
3. Run:
   python main.py

## 🛠 Tools Used
- Python
- Pandas
- Scikit-learn

## ⚙️ Process
- Data Cleaning
- Feature Encoding
- Model Training (Logistic Regression, Random Forest)

## 📊 Results

### 🔹 Model Performance
![Results](images/results.png)

## 📈 Model Comparison

| Model | Accuracy |
|------|--------|
| Logistic Regression | 78.7% |
| Random Forest | 78.5% |

👉 Logistic Regression performed slightly better after scaling.

---

### 🔹 Feature Importance
![Feature Importance](images/feature_importance.png)

## 🔎 Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### 📊 Interpretation
- True Negatives (0 → 0): 915
- False Positives (0 → 1): 118
- False Negatives (1 → 0): 181
- True Positives (1 → 1): 193

👉 Model is better at predicting non-churn customers than churn customers.

## 🔍 Key Insights
- Customers with higher total and monthly charges are more likely to churn
- Longer tenure reduces churn probability
- Contract type significantly impacts customer retention
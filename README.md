# 💳 Credit Card Fraud Detection

> Detecting fraudulent transactions in a highly imbalanced real-world dataset using XGBoost and threshold optimization — achieving **ROC-AUC of 0.97**.

---

## 📌 Problem Statement

Credit card fraud is rare but costly. In this dataset, only **0.17% of transactions are fraudulent** — a classic extreme class imbalance problem where a naive model can achieve 99.8% accuracy by predicting "not fraud" every time, yet be completely useless in practice.

The goal: build a model that reliably catches fraud with minimal false negatives, even when fraudulent samples are severely underrepresented.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Imbalance Handling | Imbalanced-learn (SMOTE) |
| Modeling | Scikit-learn, XGBoost |
| Evaluation | Scikit-learn metrics (ROC-AUC, PR curve, confusion matrix) |

---

## 🔬 Methodology

### 1. Data Preprocessing
- Loaded and explored the transaction dataset; confirmed 0.17% fraud rate
- Applied **StandardScaler** to normalize `Amount` and `Time` features (PCA-transformed features `V1`–`V28` were already scaled)
- Train/test split with stratification to preserve class ratio

### 2. Handling Class Imbalance with SMOTE
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** exclusively on the training set to prevent data leakage
- SMOTE generates synthetic fraud samples by interpolating between existing minority class instances, giving the model more signal to learn from

### 3. Model Training
Trained and compared two models:
- **Logistic Regression** — fast baseline, interpretable
- **XGBoost** — gradient-boosted ensemble, handles non-linear patterns well

### 4. Evaluation Strategy
Standard accuracy is misleading here. Instead, evaluation focused on:
- **ROC-AUC** — measures ability to discriminate fraud vs. non-fraud across all thresholds
- **Precision-Recall Curve** — critical for imbalanced datasets; shows the tradeoff between catching fraud (recall) and avoiding false alarms (precision)
- **Confusion Matrix** — visualizes false negatives (missed fraud) vs. false positives (wrongly flagged)

### 5. Decision Threshold Tuning
Default classifiers output probability scores and threshold at 0.5. Since **missing a fraudulent transaction is more costly than a false alarm**, the decision threshold was tuned to improve **fraud recall** — accepting slightly more false positives in exchange for catching more actual fraud.

---

## 📊 Results

| Model | ROC-AUC |
|---|---|
| Logistic Regression | ~0.97 |
| **XGBoost** | **0.97** |

XGBoost matched Logistic Regression on ROC-AUC but showed stronger performance on the Precision-Recall curve, particularly at higher recall thresholds — meaning it could catch more fraud without an explosion of false positives.

---

## 💼 Business Impact

| Metric | Why It Matters |
|---|---|
| High Recall | Fewer fraudulent transactions slip through undetected |
| Tuned Threshold | Ops teams can configure sensitivity based on acceptable false positive rate |
| PR Curve Analysis | Gives business stakeholders a clear tradeoff visualization |

A deployed version of this pipeline would allow financial institutions to flag high-risk transactions in real time, reducing fraud losses while maintaining a manageable review queue for human analysts.

---

## 📁 Project Structure

```
Credit_card_fraud_detection/
│
├── Credit Card Fraud Detection.ipynb   # Full pipeline: EDA → preprocessing → SMOTE → training → evaluation
└── README.md
```

All steps — data exploration, preprocessing, SMOTE oversampling, model training, threshold tuning, and evaluation — are contained in the single notebook for easy end-to-end reproducibility.

---

## 📦 Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (ULB Machine Learning Group).

- **284,807 transactions**, of which **492 are fraud (0.17%)**
- Features `V1`–`V28` are PCA-transformed for confidentiality
- `Amount`, `Time`, and `Class` are the raw features

> ⚠️ The dataset is not included in this repo due to size. Download it from Kaggle and place it in `data/creditcard.csv`.

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/royaniruddha404-debug/Credit_card_fraud_detection.git
cd Credit_card_fraud_detection

# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn xgboost jupyter

# Download the dataset from Kaggle and place it in the repo root as creditcard.csv
# Then launch the notebook
jupyter notebook "Credit Card Fraud Detection.ipynb"
```

---

## 🧠 Key Learnings

- **Never evaluate imbalanced classifiers on accuracy alone** — ROC-AUC and PR-AUC are far more informative
- **SMOTE must only be applied to training data** — applying it before splitting causes data leakage and inflated metrics
- **Threshold tuning is as important as model selection** — the right threshold depends on the business cost of false negatives vs. false positives
- XGBoost's built-in handling of feature interactions makes it a strong baseline for tabular fraud detection tasks

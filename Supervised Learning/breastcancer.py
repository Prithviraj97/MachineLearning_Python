"""
Logistic Regression on Breast Cancer Dataset
--------------------------------------------
1. Load dataset
2. Split train/test
3. Train Logistic Regression model
4. Evaluate accuracy, precision, recall, F1-score, AUC
5. Plot ROC curve
"""

# ====== Imports ======
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

# ====== 1. Load Data ======
data = load_breast_cancer()
X = data.data            # feature matrix
y = data.target          # target: 0 = malignant, 1 = benign

# ====== 2. Train/Test Split ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====== 3. Fit Logistic Regression ======
# Increase max_iter to ensure convergence
model = LogisticRegression(max_iter=500, solver='lbfgs')
model.fit(X_train, y_train)

# ====== 4. Predictions and Probabilities ======
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]   # Probabilities for ROC/AUC

# ====== 5. Metrics ======
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_prob)

print("Model Evaluation Metrics:")
print(f" Accuracy  : {accuracy:.4f}")
print(f" Precision : {precision:.4f}")
print(f" Recall    : {recall:.4f}")
print(f" F1-score  : {f1:.4f}")
print(f" AUC       : {auc:.4f}")

# ====== 6. Plot ROC Curve ======
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})", color="blue", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression (Breast Cancer)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

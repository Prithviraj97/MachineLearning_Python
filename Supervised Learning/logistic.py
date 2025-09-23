"""
Logistic Regression Demo with Visualization
-------------------------------------------
This script:
1. Generates a simple 2-D binary dataset
2. Fits a Logistic Regression model
3. Plots the decision boundary and probability contours
"""

# ========== Imports ==========
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ========== 1. Create a toy dataset ==========
# 2 informative features → easy to visualize
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Split into train/test for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ========== 2. Fit Logistic Regression ==========
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions for accuracy check
y_pred = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# ========== 3. Visualize decision boundary ==========
# Create a dense grid of points that covers the data range
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

# Predict probabilities for each grid point
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))

# Probability contour plot (light shading shows uncertainty)
contour = plt.contourf(
    xx, yy, Z, levels=20, cmap="RdBu", alpha=0.4
)

# Decision boundary line at probability = 0.5
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

# Scatter plot of training data
plt.scatter(
    X_train[:, 0], X_train[:, 1], c=y_train,
    cmap="bwr", edgecolor='k', s=50, label="Train data"
)

plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(contour, label="Predicted Probability (Class 1)")
plt.legend()
plt.show()

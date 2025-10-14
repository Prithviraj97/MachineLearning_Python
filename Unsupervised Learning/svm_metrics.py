import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --- SVC Example ---
# Generate classification data
X_cls, y_cls = datasets.make_classification(n_samples=200, n_features=2, 
                                             n_redundant=0, n_informative=2,
                                             n_clusters_per_class=1, class_sep=1.5, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)

# Train SVC
svc = SVC(kernel='linear', C=1.0)
svc.fit(X_train_cls, y_train_cls)
y_pred_cls = svc.predict(X_test_cls)

# Evaluate SVC
print("=== SVC Metrics ===")
print("Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
print("Confusion Matrix:\n", confusion_matrix(y_test_cls, y_pred_cls))
print("Classification Report:\n", classification_report(y_test_cls, y_pred_cls))

# --- SVR Example ---
# Generate regression data
X_reg = np.sort(5 * np.random.rand(200, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + 0.2 * np.random.randn(200)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Train SVR
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train_reg, y_train_reg)
y_pred_reg = svr.predict(X_test_reg)

# Evaluate SVR
print("\n=== SVR Metrics ===")
print("MSE:", mean_squared_error(y_test_reg, y_pred_reg))
print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg))
print("R² Score:", r2_score(y_test_reg, y_pred_reg))

# Visualize SVR fit
plt.figure(figsize=(10, 5))
plt.scatter(X_test_reg, y_test_reg, color='darkorange', label='Actual')
plt.plot(X_test_reg, y_pred_reg, color='navy', lw=2, label='SVR Prediction')
plt.title("SVR with RBF Kernel")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
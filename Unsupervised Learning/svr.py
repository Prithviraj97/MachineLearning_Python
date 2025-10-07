import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Step 1: Create a non-linear dataset
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + 0.2 * np.random.randn(100)

# Step 2: Fit SVR with RBF kernel
svr_rbf = SVR(kernel='linear', C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X, y)

# Step 3: Predict
X_test = np.linspace(0, 5, 100).reshape(-1, 1)
y_pred = svr_rbf.predict(X_test)

# Step 4: Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='darkorange', label='Data')
plt.plot(X_test, y_pred, color='navy', lw=2, label='SVR with Linear kernel')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Support Vector Regression with Linear Kernel')
plt.legend()
plt.grid(True)
plt.show()
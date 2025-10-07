import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Step 1: Generate multi-class linearly separable data
X, y = datasets.make_classification(n_samples=300, n_features=2, 
                                     n_redundant=0, n_informative=2,
                                     n_clusters_per_class=1, n_classes=3,
                                     class_sep=2.0, random_state=42)

# Step 2: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train SVM with linear kernel and different C values
C_values = [0.1, 1, 10]
models = [SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(X_scaled, y) for C in C_values]

# Step 4: Plot decision boundaries
def plot_decision_boundary(model, X, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolors='k', s=30)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.predict(xy).reshape(XX.shape)

    plt.contourf(XX, YY, Z, alpha=0.3, cmap=plt.cm.Set1)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

# Step 5: Visualize each model
for C, model in zip(C_values, models):
    plot_decision_boundary(model, X_scaled, y, f"Linear SVM (C={C})")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# Step 1: Generate linearly separable data
X, y = datasets.make_classification(n_samples=100, n_features=2, 
                                     n_redundant=0, n_informative=2,
                                     n_clusters_per_class=1, class_sep=2.0, random_state=42)

# Step 2: Train a linear SVM
clf = SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# Step 3: Plot decision boundary
def plot_svm_boundary(clf, X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # Plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], 
               s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.title("Linear SVM Classification")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

plot_svm_boundary(clf, X, y)
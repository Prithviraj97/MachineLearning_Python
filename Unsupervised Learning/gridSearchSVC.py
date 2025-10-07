import numpy as np
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Step 1: Create a dataset
X, y = datasets.make_classification(n_samples=300, n_features=2, 
                                     n_redundant=0, n_informative=2,
                                     n_clusters_per_class=1, class_sep=1.5, random_state=42)

# Step 2: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Define parameter grid
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

# Step 4: Grid search with cross-validation
grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5)
grid_search.fit(X_scaled, y)

# Step 5: Output best C
print(f"Optimal C value: {grid_search.best_params_['C']}")
print(f"Best cross-validation score: {grid_search.best_score_:.2f}")
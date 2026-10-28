
# ============================================================
# KNN REGRESSION: WITH PCA vs WITHOUT PCA
# Includes finding the best K value
# ============================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# 2. LOAD DATASET

df = pd.read_csv("energydata_complete.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
print(df.isnull().sum())

# 3. EDA

sns.histplot(df["Appliances"], kde=True)
plt.title("Distribution of Appliances")
plt.show()

sns.boxplot(x=df["Appliances"])
plt.title("Appliances Outliers")
plt.show()

df.select_dtypes(include="number").hist(figsize=(15, 12))
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# 4. DEFINE FEATURES AND TARGET

X = df.drop(columns=["Appliances", "date", "rv1", "rv2"])
y = df["Appliances"]

numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

# 5. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# 6. PREPROCESSING WITHOUT PCA

num_pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

cat_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, categorical_cols)
])

# 7. K-FOLD CROSS-VALIDATION

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 8. KNN WITHOUT PCA

knn_pipe = Pipeline([
    ("preprocessing", preprocessing),
    ("regressor", KNeighborsRegressor())
])

# Try different K values and other KNN parameters
param_grid = {
    "regressor__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15],
    "regressor__weights": ["uniform", "distance"],
    "regressor__p": [1, 2]
}

grid_search = GridSearchCV(
    knn_pipe,
    param_grid,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    refit=True
)

grid_search.fit(X_train, y_train)

print("\n===== KNN WITHOUT PCA =====")
print("Best parameters:", grid_search.best_params_)
print("Best K value:",
      grid_search.best_params_["regressor__n_neighbors"])
print("Best CV RMSE:", -grid_search.best_score_)

# 9. EVALUATE KNN WITHOUT PCA

best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

rmse_no_pca = np.sqrt(mean_squared_error(y_test, y_pred))
mse_no_pca = mean_squared_error(y_test, y_pred)
mae_no_pca = mean_absolute_error(y_test, y_pred)
r2_no_pca = r2_score(y_test, y_pred)

print("\nTest RMSE:", rmse_no_pca)
print("Test MSE:", mse_no_pca)
print("Test MAE:", mae_no_pca)
print("Test R2:", r2_no_pca)

# 10. PREPROCESSING WITH PCA

num_pipe_pca = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    PCA(svd_solver="full")
)

preprocessing_pca = ColumnTransformer([
    ("num", num_pipe_pca, numeric_cols),
    ("cat", cat_pipe, categorical_cols)
])

# 11. KNN WITH PCA

knn_pipe_pca = Pipeline([
    ("preprocessing", preprocessing_pca),
    ("regressor", KNeighborsRegressor())
])

param_grid_pca = {
    "preprocessing__num__pca__n_components": [0.80, 0.90, 0.95, 0.99],
    "regressor__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15],
    "regressor__weights": ["uniform", "distance"],
    "regressor__p": [1, 2]
}

grid_search_pca = GridSearchCV(
    knn_pipe_pca,
    param_grid_pca,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    refit=True
)

grid_search_pca.fit(X_train, y_train)

print("\n===== KNN WITH PCA =====")
print("Best parameters:", grid_search_pca.best_params_)
print("Best K value:",
      grid_search_pca.best_params_["regressor__n_neighbors"])
print("Best PCA components:",
      grid_search_pca.best_params_[
          "preprocessing__num__pca__n_components"
      ])
print("Best CV RMSE:", -grid_search_pca.best_score_)

# 12. EVALUATE KNN WITH PCA

best_knn_pca = grid_search_pca.best_estimator_
y_pred_pca = best_knn_pca.predict(X_test)

rmse_pca = np.sqrt(mean_squared_error(y_test, y_pred_pca))
mse_pca = mean_squared_error(y_test, y_pred_pca)
mae_pca = mean_absolute_error(y_test, y_pred_pca)
r2_pca = r2_score(y_test, y_pred_pca)

print("\nTest RMSE:", rmse_pca)
print("Test MSE:", mse_pca)
print("Test MAE:", mae_pca)
print("Test R2:", r2_pca)

# 13. COMPARE MODELS

comparison = pd.DataFrame({
    "Model": ["KNN WITHOUT PCA", "KNN WITH PCA"],
    "Best K": [
        grid_search.best_params_["regressor__n_neighbors"],
        grid_search_pca.best_params_["regressor__n_neighbors"]
    ],
    "Best CV RMSE": [
        -grid_search.best_score_,
        -grid_search_pca.best_score_
    ],
    "Test RMSE": [rmse_no_pca, rmse_pca],
    "Test MAE": [mae_no_pca, mae_pca],
    "Test R2": [r2_no_pca, r2_pca]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

if rmse_no_pca < rmse_pca:
    print("\nBetter model by test RMSE: KNN WITHOUT PCA")
else:
    print("\nBetter model by test RMSE: KNN WITH PCA")

# 14. PLOT BEST K RESULTS WITHOUT PCA

results = pd.DataFrame(grid_search.cv_results_)

k_results = results[
    results["param_regressor__weights"] ==
    grid_search.best_params_["regressor__weights"]
]

k_results = k_results[
    k_results["param_regressor__p"] ==
    grid_search.best_params_["regressor__p"]
]

k_results = k_results.sort_values("param_regressor__n_neighbors")

plt.plot(
    k_results["param_regressor__n_neighbors"].astype(int),
    -k_results["mean_test_score"],
    marker="o"
)

plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Mean CV RMSE")
plt.title("Finding the Best K for KNN Without PCA")
plt.show()


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

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

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_rel


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("energydata_complete.csv")

print("Dataset shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())


# ============================================================
# 3. BASIC EDA
# ============================================================

target = "Appliances"

sns.histplot(data=df, x=target, bins=30)
plt.title("Target Distribution")
plt.xlabel("Appliances")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

sns.boxplot(data=df, x=target)
plt.title("Target Outlier Detection")
plt.tight_layout()
plt.show()

numeric_cols_eda = df.select_dtypes(include="number").columns

df[numeric_cols_eda].hist(figsize=(15, 10), bins=30)
plt.suptitle("Numeric Feature Distributions")
plt.tight_layout()
plt.show()

sns.heatmap(
    df[numeric_cols_eda].corr(),
    cmap="coolwarm",
    annot=False
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# 4. DEFINE X AND y
# ============================================================

X = df.drop(columns=["Appliances", "date", "rv1", "rv2"])
y = df["Appliances"]

numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 6. PREPROCESSING
# ============================================================

cat_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

num_pipe_nopca = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

num_pipe_pca = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    PCA(svd_solver="full")
)

transformers_nopca = [
    ("num", num_pipe_nopca, numeric_cols)
]

transformers_pca = [
    ("num", num_pipe_pca, numeric_cols)
]

if len(categorical_cols) > 0:
    transformers_nopca.append(
        ("cat", cat_pipe, categorical_cols)
    )
    transformers_pca.append(
        ("cat", cat_pipe, categorical_cols)
    )

prep_nopca = ColumnTransformer(
    transformers=transformers_nopca
)

prep_pca = ColumnTransformer(
    transformers=transformers_pca
)


# ============================================================
# 7. CREATE SVR PIPELINES
# ============================================================

pipeline_no_pca = Pipeline([
    ("preprocessing", prep_nopca),
    ("regressor", SVR())
])

pipeline_pca = Pipeline([
    ("preprocessing", prep_pca),
    ("regressor", SVR())
])


# ============================================================
# 8. K-FOLD CROSS-VALIDATION
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 9. HYPERPARAMETER GRIDS
# ============================================================

param_grid_nopca = {
    "regressor__kernel": ["linear", "rbf"],
    "regressor__C": [0.1, 1, 10],
    "regressor__epsilon": [0.1, 0.2, 0.5],
    "regressor__gamma": ["scale", "auto"]
}

param_grid_pca = {
    "preprocessing__num__pca__n_components": [
        0.80, 0.90, 0.95, 0.99
    ],
    "regressor__kernel": ["linear", "rbf"],
    "regressor__C": [0.1, 1, 10],
    "regressor__epsilon": [0.1, 0.2, 0.5],
    "regressor__gamma": ["scale", "auto"]
}


# ============================================================
# 10. GRID SEARCH
# ============================================================

grid_search_nopca = GridSearchCV(
    estimator=pipeline_no_pca,
    param_grid=param_grid_nopca,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

grid_search_pca = GridSearchCV(
    estimator=pipeline_pca,
    param_grid=param_grid_pca,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

grid_search_nopca.fit(X_train, y_train)
grid_search_pca.fit(X_train, y_train)


# ============================================================
# 11. BEST HYPERPARAMETERS
# ============================================================

print("\n--- BEST PARAMETERS WITHOUT PCA ---")
print(grid_search_nopca.best_params_)
print("Best CV RMSE:", -grid_search_nopca.best_score_)

print("\n--- BEST PARAMETERS WITH PCA ---")
print(grid_search_pca.best_params_)
print("Best CV RMSE:", -grid_search_pca.best_score_)


# ============================================================
# 12. TEST SET PREDICTIONS
# ============================================================

model_no_pca = grid_search_nopca.best_estimator_
model_pca = grid_search_pca.best_estimator_

y_pred_no_pca = model_no_pca.predict(X_test)
y_pred_pca = model_pca.predict(X_test)


# ============================================================
# 13. PERFORMANCE METRICS
# ============================================================

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {model_name} ---")
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("MAE:", round(mae, 4))
    print("R2:", round(r2, 4))


evaluate_model(y_test, y_pred_no_pca, "SVR WITHOUT PCA")
evaluate_model(y_test, y_pred_pca, "SVR WITH PCA")


# ============================================================
# 14. FIVE-FOLD CV RMSE: MEAN AND STANDARD DEVIATION
# ============================================================

index_no_pca = grid_search_nopca.best_index_
index_pca = grid_search_pca.best_index_

rmse_folds_no_pca = np.array([
    -grid_search_nopca.cv_results_[f"split{i}_test_score"][index_no_pca]
    for i in range(5)
])

rmse_folds_pca = np.array([
    -grid_search_pca.cv_results_[f"split{i}_test_score"][index_pca]
    for i in range(5)
])

print("\n--- CV RESULTS WITHOUT PCA ---")
print("Fold RMSE:", np.round(rmse_folds_no_pca, 4))
print("Mean RMSE:", round(rmse_folds_no_pca.mean(), 4))
print("Std Dev:", round(rmse_folds_no_pca.std(ddof=1), 4))

print("\n--- CV RESULTS WITH PCA ---")
print("Fold RMSE:", np.round(rmse_folds_pca, 4))
print("Mean RMSE:", round(rmse_folds_pca.mean(), 4))
print("Std Dev:", round(rmse_folds_pca.std(ddof=1), 4))


# ============================================================
# 15. PAIRED T-TEST
# ============================================================

t_stat, p_value = ttest_rel(
    rmse_folds_no_pca,
    rmse_folds_pca
)

print("\nPaired t-test statistic:", round(t_stat, 4))
print("P-value:", round(p_value, 6))

if p_value < 0.05:
    print("Difference is statistically significant at 0.05.")
else:
    print("Insufficient evidence of a significant difference.")


# ============================================================
# 16. FINAL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": ["SVR without PCA", "SVR with PCA"],
    "CV Mean RMSE": [
        rmse_folds_no_pca.mean(),
        rmse_folds_pca.mean()
    ],
    "CV RMSE Std": [
        rmse_folds_no_pca.std(ddof=1),
        rmse_folds_pca.std(ddof=1)
    ],
    "Test RMSE": [
        np.sqrt(mean_squared_error(y_test, y_pred_no_pca)),
        np.sqrt(mean_squared_error(y_test, y_pred_pca))
    ],
    "Test MAE": [
        mean_absolute_error(y_test, y_pred_no_pca),
        mean_absolute_error(y_test, y_pred_pca)
    ],
    "Test R2": [
        r2_score(y_test, y_pred_no_pca),
        r2_score(y_test, y_pred_pca)
    ]
})

print("\n--- MODEL COMPARISON ---")
print(comparison.round(4).to_string(index=False))

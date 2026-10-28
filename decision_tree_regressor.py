
# ============================================================
# APPLIANCES ENERGY PREDICTION - REGRESSION
# Decision Tree Regressor | PCA vs No PCA
# Hyperparameter Tuning | 5-Fold CV | Statistical Analysis
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
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_rel


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("energydata_complete.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Identify numeric columns for EDA
numeric_cols_eda = df.select_dtypes(include=["number"]).columns

# Distribution of Appliances energy consumption
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Appliances", kde=True)
plt.title("Distribution of Appliances Energy Consumption")
plt.xlabel("Appliances (Wh)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Boxplot to detect outliers
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["Appliances"])
plt.title("Boxplot of Appliances Energy Consumption")
plt.xlabel("Appliances (Wh)")
plt.tight_layout()
plt.show()

# Histograms of numeric features
df[numeric_cols_eda].hist(figsize=(16, 12), bins=30)
plt.suptitle("Distributions of Numeric Features")
plt.tight_layout()
plt.show()

# Correlation heatmap
corr_matrix = df[numeric_cols_eda].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# 4. DEFINE FEATURES (X) AND TARGET (y)
# ============================================================

# Appliances is the continuous target, so this is REGRESSION.
# Drop date and random variables rv1/rv2 from the features.

X = df.drop(columns=["Appliances", "date", "rv1", "rv2"])
y = df["Appliances"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print("Appliances")


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# ============================================================
# 6. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_cols = X.select_dtypes(include=["number"]).columns

categorical_cols = X.select_dtypes(
    include=["object", "category", "string", "bool"]
).columns

print("\nNumeric Features:")
print(numeric_cols)

print("\nCategorical Features:")
print(categorical_cols)


# ============================================================
# 7. DATA PREPROCESSING
# ============================================================

# Categorical preprocessing
cat_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

# Numeric preprocessing WITHOUT PCA
num_pipe_nopca = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

# Numeric preprocessing WITH PCA
num_pipe_pca = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    PCA(n_components=0.95, svd_solver="full")
)

# Build preprocessing transformers
transformers_nopca = [
    ("num", num_pipe_nopca, numeric_cols)
]

transformers_pca = [
    ("num", num_pipe_pca, numeric_cols)
]

# Add categorical preprocessing only if categorical columns exist
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
# 8. BUILD DECISION TREE REGRESSION PIPELINES
# ============================================================

pipeline_no_pca = Pipeline([
    ("preprocessor", prep_nopca),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

pipeline_pca = Pipeline([
    ("preprocessor", prep_pca),
    ("regressor", DecisionTreeRegressor(random_state=42))
])


# ============================================================
# 9. HYPERPARAMETER TUNING
# ============================================================

# Same cross-validation splits for both models
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Hyperparameters for the Decision Tree
param_grid_no_pca = {
    "regressor__max_depth": [5, 10, 20, None],
    "regressor__min_samples_split": [2, 10],
    "regressor__min_samples_leaf": [1, 5]
}

# PCA parameters are tuned inside the numeric preprocessing pipeline
param_grid_pca = {
    "preprocessor__num__pca__n_components": [0.80, 0.90, 0.95, 0.99],
    "regressor__max_depth": [5, 10, 20, None],
    "regressor__min_samples_split": [2, 10],
    "regressor__min_samples_leaf": [1, 5]
}

# Tune model WITHOUT PCA
grid_no_pca = GridSearchCV(
    estimator=pipeline_no_pca,
    param_grid=param_grid_no_pca,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

grid_no_pca.fit(X_train, y_train)

# Tune model WITH PCA
grid_pca = GridSearchCV(
    estimator=pipeline_pca,
    param_grid=param_grid_pca,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

grid_pca.fit(X_train, y_train)


# ============================================================
# 10. DISPLAY BEST HYPERPARAMETERS
# ============================================================

print("\n========== NO PCA: BEST PARAMETERS ==========")
print(grid_no_pca.best_params_)

print("\nBest CV RMSE Without PCA:")
print(round(-grid_no_pca.best_score_, 4))

print("\n========== PCA: BEST PARAMETERS ==========")
print(grid_pca.best_params_)

print("\nBest CV RMSE With PCA:")
print(round(-grid_pca.best_score_, 4))


# ============================================================
# 11. EVALUATE ON THE TEST SET
# ============================================================

# GridSearchCV refit=True means best estimator is fitted
# on the complete training set.

best_model_no_pca = grid_no_pca.best_estimator_
best_model_pca = grid_pca.best_estimator_

# Predictions
y_pred_no_pca = best_model_no_pca.predict(X_test)
y_pred_pca = best_model_pca.predict(X_test)

# Metrics WITHOUT PCA
rmse_no_pca_test = np.sqrt(
    mean_squared_error(y_test, y_pred_no_pca)
)
mae_no_pca_test = mean_absolute_error(
    y_test, y_pred_no_pca
)
r2_no_pca_test = r2_score(
    y_test, y_pred_no_pca
)

# Metrics WITH PCA
rmse_pca_test = np.sqrt(
    mean_squared_error(y_test, y_pred_pca)
)
mae_pca_test = mean_absolute_error(
    y_test, y_pred_pca
)
r2_pca_test = r2_score(
    y_test, y_pred_pca
)

print("\n========== TEST SET RESULTS ==========")

print("\n--- Decision Tree WITHOUT PCA ---")
print("RMSE:", round(rmse_no_pca_test, 4))
print("MAE :", round(mae_no_pca_test, 4))
print("R²  :", round(r2_no_pca_test, 4))

print("\n--- Decision Tree WITH PCA ---")
print("RMSE:", round(rmse_pca_test, 4))
print("MAE :", round(mae_pca_test, 4))
print("R²  :", round(r2_pca_test, 4))


# ============================================================
# 12. FIVE-FOLD CROSS-VALIDATION: MEAN AND STANDARD DEVIATION
# ============================================================

# Extract the RMSE for each of the 5 validation folds
# at the selected best hyperparameter combination.

best_index_no_pca = grid_no_pca.best_index_
best_index_pca = grid_pca.best_index_

rmse_folds_no_pca = np.array([
    -grid_no_pca.cv_results_[f"split{i}_test_score"][best_index_no_pca]
    for i in range(5)
])

rmse_folds_pca = np.array([
    -grid_pca.cv_results_[f"split{i}_test_score"][best_index_pca]
    for i in range(5)
])

print("\n========== FIVE-FOLD CV RESULTS ==========")

print("\nWITHOUT PCA")
print("Fold RMSE values:", np.round(rmse_folds_no_pca, 4))
print("Mean RMSE:", round(rmse_folds_no_pca.mean(), 4))
print("Standard Deviation:", round(rmse_folds_no_pca.std(ddof=1), 4))

print("\nWITH PCA")
print("Fold RMSE values:", np.round(rmse_folds_pca, 4))
print("Mean RMSE:", round(rmse_folds_pca.mean(), 4))
print("Standard Deviation:", round(rmse_folds_pca.std(ddof=1), 4))


# ============================================================
# 13. PAIRED T-TEST ON FOLD-LEVEL RMSE
# ============================================================

# Lower RMSE is better.
# Pair fold scores because both models use the same CV splits.

t_statistic, p_value = ttest_rel(
    rmse_folds_no_pca,
    rmse_folds_pca
)

print("\n========== PAIRED T-TEST ==========")
print("T-statistic:", round(t_statistic, 4))
print("P-value:", round(p_value, 6))

alpha = 0.05

if p_value < alpha:
    print(
        "Result: The difference in fold-level RMSE is "
        "statistically significant at alpha = 0.05."
    )
else:
    print(
        "Result: There is insufficient evidence of a "
        "statistically significant difference at alpha = 0.05."
    )


# ============================================================
# 14. FINAL COMPARISON AND BEST MODEL
# ============================================================

comparison = pd.DataFrame({
    "Model": ["Decision Tree WITHOUT PCA", "Decision Tree WITH PCA"],
    "Best CV RMSE": [
        -grid_no_pca.best_score_,
        -grid_pca.best_score_
    ],
    "Mean 5-Fold RMSE": [
        rmse_folds_no_pca.mean(),
        rmse_folds_pca.mean()
    ],
    "CV RMSE Std Dev": [
        rmse_folds_no_pca.std(ddof=1),
        rmse_folds_pca.std(ddof=1)
    ],
    "Test RMSE": [
        rmse_no_pca_test,
        rmse_pca_test
    ],
    "Test MAE": [
        mae_no_pca_test,
        mae_pca_test
    ],
    "Test R2": [
        r2_no_pca_test,
        r2_pca_test
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison.round(4).to_string(index=False))

# Select the model with lower test RMSE for this split
if rmse_no_pca_test < rmse_pca_test:
    print("\nBetter model by test RMSE: Decision Tree WITHOUT PCA")
    print("Selected best parameters:", grid_no_pca.best_params_)
else:
    print("\nBetter model by test RMSE: Decision Tree WITH PCA")
    print("Selected best parameters:", grid_pca.best_params_)


# ============================================================
# 15. ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_no_pca, alpha=0.5)
plt.xlabel("Actual Appliances Consumption (Wh)")
plt.ylabel("Predicted Appliances Consumption (Wh)")
plt.title("Actual vs Predicted - Decision Tree Without PCA")

# Ideal predictions lie on this line
lower = min(y_test.min(), y_pred_no_pca.min())
upper = max(y_test.max(), y_pred_no_pca.max())
plt.plot([lower, upper], [lower, upper], linestyle="--")

plt.tight_layout()
plt.show()

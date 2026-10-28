
# ============================================================
# MACHINE LEARNING LAB EXPERIMENT
# LOGISTIC REGRESSION WITH PCA AND WITHOUT PCA
# ============================================================

# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from scipy.stats import ttest_rel


# ============================================================
# 2. LOAD DATASET AND PERFORM EDA
# ============================================================

df = pd.read_csv("energydata_complete.csv")

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nNumerical summary:")
print(df.describe())

# Target distribution before classification
print("\nOriginal Appliances target distribution:")
print(df["Appliances"].describe())

plt.figure(figsize=(8, 5))
plt.hist(df["Appliances"], bins=30, edgecolor="black")
plt.xlabel("Appliances energy consumption (Wh)")
plt.ylabel("Frequency")
plt.title("Distribution of Appliances Energy Consumption")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.boxplot(df["Appliances"].dropna(), vert=False)
plt.xlabel("Appliances energy consumption (Wh)")
plt.title("Boxplot of Appliances")
plt.tight_layout()
plt.show()

# Correlation heatmap for numerical features
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

plt.figure(figsize=(14, 10))
plt.imshow(correlation, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

df["date"] = pd.to_datetime(df["date"])

df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# Remove original date and random variables
df.drop(
    columns=["date", "rv1", "rv2"],
    inplace=True,
    errors="ignore"
)


# ============================================================
# 4. CREATE BINARY CLASSIFICATION TARGET
# ============================================================

# Original Appliances is continuous, so convert it into classes.
# Class 0 = Low usage (at or below median)
# Class 1 = High usage (above median)

threshold = df["Appliances"].median()

df["Usage_Class"] = (
    df["Appliances"] > threshold
).astype(int)

# Remove original target to avoid data leakage
X = df.drop(columns=["Appliances", "Usage_Class"])
y = df["Usage_Class"]

print("\n========== CLASSIFICATION TARGET ==========")
print("Median threshold:", threshold)
print("Class 0 = Low usage (at or below median)")
print("Class 1 = High usage (above median)")
print("\nClass counts:")
print(y.value_counts().sort_index())

print("\nClass proportions:")
print(y.value_counts(normalize=True).sort_index())


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN-TEST SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. PREPROCESSING PIPELINE
# ============================================================

numeric_cols = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "category", "string", "bool"]
).columns.tolist()

print("\nNumerical columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = [
    ("num", numeric_pipeline, numeric_cols)
]

if categorical_cols:
    transformers.append(
        ("cat", categorical_pipeline, categorical_cols)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# ============================================================
# 7. DEFINE 5-FOLD CROSS-VALIDATION
# ============================================================

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 8. MODEL 1: LOGISTIC REGRESSION WITHOUT PCA
# ============================================================

pipeline_no_pca = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

# Hyperparameter tuning grid
param_grid_no_pca = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

grid_no_pca = GridSearchCV(
    estimator=pipeline_no_pca,
    param_grid=param_grid_no_pca,
    scoring="accuracy",
    cv=kfold,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

print("\n========== TUNING MODEL WITHOUT PCA ==========")

grid_no_pca.fit(X_train, y_train)

print("Hyperparameter tuning method: GridSearchCV")
print("Parameters tested:", param_grid_no_pca)
print("Selected best parameters:", grid_no_pca.best_params_)
print("Best mean 5-fold CV accuracy:", grid_no_pca.best_score_)


# ============================================================
# 9. MODEL 2: LOGISTIC REGRESSION WITH PCA
# ============================================================

pipeline_pca = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA(svd_solver="full")),
    ("classifier", LogisticRegression(max_iter=2000))
])

# Tune both PCA retained variance and Logistic Regression C
param_grid_pca = {
    "pca__n_components": [0.80, 0.90, 0.95, 0.99],
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

grid_pca = GridSearchCV(
    estimator=pipeline_pca,
    param_grid=param_grid_pca,
    scoring="accuracy",
    cv=kfold,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

print("\n========== TUNING MODEL WITH PCA ==========")

grid_pca.fit(X_train, y_train)

print("Hyperparameter tuning method: GridSearchCV")
print("Parameters tested:", param_grid_pca)
print("Selected best parameters:", grid_pca.best_params_)
print("Best mean 5-fold CV accuracy:", grid_pca.best_score_)


# ============================================================
# 10. EVALUATE MODELS ON TEST SET
# ============================================================

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print(f"\n========== {name} ==========")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


results_no_pca = evaluate_model(
    "LOGISTIC REGRESSION WITHOUT PCA",
    grid_no_pca.best_estimator_,
    X_test,
    y_test
)

results_pca = evaluate_model(
    "LOGISTIC REGRESSION WITH PCA",
    grid_pca.best_estimator_,
    X_test,
    y_test
)


# ============================================================
# 11. REPORT FOLD-BY-FOLD CV SCORES
# ============================================================

def get_fold_scores(name, grid):

    results = grid.cv_results_
    best_index = grid.best_index_

    fold_scores = np.array([
        results[f"split{i}_test_score"][best_index]
        for i in range(5)
    ])

    print(f"\n========== {name}: FOLD SCORES ==========")

    for i, score in enumerate(fold_scores, start=1):
        print(f"Fold {i} accuracy: {score:.4f}")

    print(f"Mean CV accuracy: {np.mean(fold_scores):.4f}")
    print(
        "Standard deviation:",
        f"{np.std(fold_scores, ddof=1):.4f}"
    )

    return fold_scores


scores_no_pca = get_fold_scores(
    "WITHOUT PCA",
    grid_no_pca
)

scores_pca = get_fold_scores(
    "WITH PCA",
    grid_pca
)


# ============================================================
# 12. PAIRED T-TEST ON MATCHING FOLD ACCURACIES
# ============================================================

t_statistic, p_value = ttest_rel(
    scores_no_pca,
    scores_pca
)

print("\n========== PAIRED T-TEST ==========")
print("T-statistic:", t_statistic)
print("P-value:", p_value)

if p_value < 0.05:
    print(
        "The difference in fold accuracies is statistically "
        "significant at the 5% level."
    )
else:
    print(
        "Insufficient evidence of a statistically significant "
        "difference at the 5% level."
    )


# ============================================================
# 13. COMPARE THE TWO MODELS
# ============================================================

comparison = pd.DataFrame(
    [results_no_pca, results_pca],
    index=["Logistic Regression without PCA",
           "Logistic Regression with PCA"]
)

print("\n========== TEST-SET COMPARISON ==========")
print(comparison.round(4))


# ============================================================
# 14. REQUIRED HYPERPARAMETER TUNING SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL REPORT: HYPERPARAMETER TUNING")
print("=" * 65)

print("\nMODEL 1: LOGISTIC REGRESSION WITHOUT PCA")
print("Search method: GridSearchCV")
print("Cross-validation: 5-fold")
print("Scoring metric: Accuracy")
print("Candidate parameters:", param_grid_no_pca)
print("Selected best parameters:", grid_no_pca.best_params_)
print(
    "Best mean CV accuracy:",
    round(grid_no_pca.best_score_, 4)
)

print("\nMODEL 2: LOGISTIC REGRESSION WITH PCA")
print("Search method: GridSearchCV")
print("Cross-validation: 5-fold")
print("Scoring metric: Accuracy")
print("Candidate parameters:", param_grid_pca)
print("Selected best parameters:", grid_pca.best_params_)
print(
    "Best mean CV accuracy:",
    round(grid_pca.best_score_, 4)
)

print("\n========== EXPERIMENT CHECKLIST ==========")
print("EDA performed: YES")
print("Missing-value preprocessing included: YES")
print("Feature scaling included: YES")
print("Hyperparameter tuning performed: YES")
print("Selected hyperparameters displayed: YES")
print("Five-fold CV mean and standard deviation: YES")
print("PCA vs. no-PCA comparison: YES")
print("Test-set evaluation: YES")
print("Paired t-test performed: YES")
print("=" * 65)

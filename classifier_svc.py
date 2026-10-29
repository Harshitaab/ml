
# ============================================================
# SVM CLASSIFICATION: WITH PCA VS WITHOUT PCA
# Dataset: Iris (built into scikit-learn)
# ============================================================

# STEP 1: IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wilcoxon

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# STEP 2: LOAD SAMPLE DATASET

iris = load_iris(as_frame=True)
df = iris.frame.copy()

# Rename target column and show flower names.
df = df.rename(columns={"target": "species"})
df["species"] = df["species"].map(
    dict(enumerate(iris.target_names))
)

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe(include="all").T)

print("\nClass distribution:")
print(df["species"].value_counts())


# STEP 3: BASIC EDA PLOTS

df[iris.feature_names].hist(figsize=(10, 7))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(
    df[iris.feature_names].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# STEP 4: SEPARATE FEATURES AND TARGET

X = df.drop(columns=["species"])
y = df["species"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)
print("Classes:", y.unique())


# STEP 5: TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# STEP 6: IDENTIFY NUMERIC AND CATEGORICAL FEATURES

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

if not numeric_features:
    raise ValueError("PCA requires at least one numeric feature.")


# STEP 7: BUILD PREPROCESSING PIPELINES

# Numeric features: impute missing values and scale.
# PCA will be added separately for the PCA version.

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical features: impute and one-hot encode.

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Without PCA: numeric and categorical preprocessing only.

transformers_no_pca = []

if numeric_features:
    transformers_no_pca.append(
        ("num", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers_no_pca.append(
        ("cat", categorical_pipeline, categorical_features)
    )

preprocessor_no_pca = ColumnTransformer(
    transformers=transformers_no_pca
)


# STEP 8: BUILD PREPROCESSOR WITH PCA

# PCA is applied to the scaled numeric features.
# Categorical features, if present, are one-hot encoded
# and kept in a separate branch.

numeric_pipeline_pca = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(svd_solver="full"))
])

transformers_pca = [
    ("num", numeric_pipeline_pca, numeric_features)
]

if categorical_features:
    transformers_pca.append(
        ("cat", categorical_pipeline, categorical_features)
    )

preprocessor_pca = ColumnTransformer(
    transformers=transformers_pca
)


# STEP 9: CREATE TWO SVM PIPELINES

# Pipeline A: WITHOUT PCA

pipeline_no_pca = Pipeline([
    ("preprocessor", preprocessor_no_pca),
    ("model", SVC())
])

# Pipeline B: WITH PCA

pipeline_pca = Pipeline([
    ("preprocessor", preprocessor_pca),
    ("model", SVC())
])


# STEP 10: DEFINE PARAMETER GRIDS

# SVM parameters:
# C = regularization strength
# kernel = decision boundary type
# gamma = influence of individual samples for RBF kernel

param_grid_no_pca = [
    {
        "model__kernel": ["linear"],
        "model__C": [0.1, 1, 10]
    },
    {
        "model__kernel": ["rbf"],
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale", 0.1]
    }
]

param_grid_pca = [
    {
        "preprocessor__num__pca__n_components": [0.80, 0.95],
        "model__kernel": ["linear"],
        "model__C": [0.1, 1, 10]
    },
    {
        "preprocessor__num__pca__n_components": [0.80, 0.95],
        "model__kernel": ["rbf"],
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale", 0.1]
    }
]


# STEP 11: CONFIGURE STRATIFIED 5-FOLD CV

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# STEP 12: GRID SEARCH WITHOUT PCA

grid_no_pca = GridSearchCV(
    estimator=pipeline_no_pca,
    param_grid=param_grid_no_pca,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

print("\nTraining SVM WITHOUT PCA...")
grid_no_pca.fit(X_train, y_train)

print("\nBest parameters WITHOUT PCA:")
print(grid_no_pca.best_params_)

print("Best CV macro F1 WITHOUT PCA:",
      round(grid_no_pca.best_score_, 4))


# STEP 13: GRID SEARCH WITH PCA

grid_pca = GridSearchCV(
    estimator=pipeline_pca,
    param_grid=param_grid_pca,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

print("\nTraining SVM WITH PCA...")
grid_pca.fit(X_train, y_train)

print("\nBest parameters WITH PCA:")
print(grid_pca.best_params_)

print("Best CV macro F1 WITH PCA:",
      round(grid_pca.best_score_, 4))


# STEP 14: EVALUATE BOTH MODELS ON THE TEST SET

best_no_pca = grid_no_pca.best_estimator_
best_pca = grid_pca.best_estimator_

pred_no_pca = best_no_pca.predict(X_test)
pred_pca = best_pca.predict(X_test)


def evaluate_model(model_name, y_true, y_pred, class_names):
    """Print metrics and plot a confusion matrix."""

    print(f"\n========== {model_name} ==========")

    print("Accuracy:",
          round(accuracy_score(y_true, y_pred), 4))

    print("Macro precision:",
          round(precision_score(
              y_true, y_pred, average="macro", zero_division=0
          ), 4))

    print("Macro recall:",
          round(recall_score(
              y_true, y_pred, average="macro", zero_division=0
          ), 4))

    print("Macro F1:",
          round(f1_score(
              y_true, y_pred, average="macro", zero_division=0
          ), 4))

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=class_names,
        zero_division=0
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=class_names,
        display_labels=class_names,
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()


class_names = list(iris.target_names)

evaluate_model(
    "SVM WITHOUT PCA",
    y_test,
    pred_no_pca,
    class_names
)

evaluate_model(
    "SVM WITH PCA",
    y_test,
    pred_pca,
    class_names
)


# STEP 15: EXTRACT BEST MODEL'S FOLD SCORES

# Extract the five CV scores for each selected best model.
# GridSearchCV has already calculated these scores.

scores_no_pca = np.array([
    grid_no_pca.cv_results_[f"split{i}_test_score"][
        grid_no_pca.best_index_
    ]
    for i in range(5)
])

scores_pca = np.array([
    grid_pca.cv_results_[f"split{i}_test_score"][
        grid_pca.best_index_
    ]
    for i in range(5)
])

print("\nFold scores WITHOUT PCA:")
print(scores_no_pca)

print("\nFold scores WITH PCA:")
print(scores_pca)

print("\nMean CV macro F1 WITHOUT PCA:",
      round(scores_no_pca.mean(), 4))

print("Mean CV macro F1 WITH PCA:",
      round(scores_pca.mean(), 4))


# STEP 16: WILCOXON SIGNED-RANK TEST

# Compare paired fold scores from the two approaches.
# With only five folds, this test has limited statistical power.

differences = scores_no_pca - scores_pca

if np.allclose(differences, 0):
    print("\nWilcoxon test skipped: all paired scores are equal.")
else:
    try:
        statistic, p_value = wilcoxon(
            scores_no_pca,
            scores_pca,
            alternative="two-sided"
        )

        print("\nWilcoxon statistic:", statistic)
        print("Wilcoxon p-value:", p_value)

        if p_value < 0.05:
            print("Difference is statistically significant at 0.05.")
        else:
            print("No statistically significant difference at 0.05.")

    except ValueError as error:
        print("\nWilcoxon test could not be calculated:", error)


# STEP 17: FINAL COMPARISON TABLE

results = pd.DataFrame({
    "Approach": ["SVM without PCA", "SVM with PCA"],
    "Best CV Macro F1": [
        grid_no_pca.best_score_,
        grid_pca.best_score_
    ],
    "Test Accuracy": [
        accuracy_score(y_test, pred_no_pca),
        accuracy_score(y_test, pred_pca)
    ],
    "Test Macro F1": [
        f1_score(y_test, pred_no_pca, average="macro"),
        f1_score(y_test, pred_pca, average="macro")
    ]
})

print("\n========== FINAL MODEL COMPARISON ==========")
print(results.round(4).to_string(index=False))
print("============================================")

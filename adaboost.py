# ============================================================
# CLASSIFICATION: ADABOOST, BAGGING, GRADIENT BOOSTING
# PCA vs No PCA | Hyperparameter Tuning | 5-Fold CV
# Evaluation | Confusion Matrix | Wilcoxon Test
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from scipy.stats import wilcoxon


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Replace with your actual dataset filename.
df = pd.read_csv("your_dataset.csv")

# Display basic information about the dataset.
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

# Replace "target" with your actual classification target column.
target_col = "target"

# Show the number of samples in each class.
print("\nClass Distribution:")
print(df[target_col].value_counts())

# Plot class distribution.
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=target_col)
plt.title("Distribution of Target Classes")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Histograms of numerical columns.
numeric_cols_eda = df.select_dtypes(include="number").columns

if len(numeric_cols_eda) > 0:
    df[numeric_cols_eda].hist(figsize=(14, 10), bins=30)
    plt.suptitle("Distributions of Numerical Features")
    plt.tight_layout()
    plt.show()

# Correlation heatmap for numerical columns.
if len(numeric_cols_eda) > 1:
    corr_matrix = df[numeric_cols_eda].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. DEFINE FEATURES (X) AND TARGET (y)
# ============================================================

# X contains input features.
# y contains the class label to predict.
# Remove the target from the feature columns to avoid leakage.

X = df.drop(columns=[target_col])
y = df[target_col]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeatures:")
print(X.columns)

print("\nNumber of Classes:", y.nunique())
print("Class Labels:", y.unique())


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

# stratify=y preserves class proportions in both sets.
# 80% of data is used for training and 20% for testing.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_cols = X.select_dtypes(include=["number"]).columns

categorical_cols = X.select_dtypes(
    include=["object", "category", "string", "bool"]
).columns

print("\nNumerical Features:")
print(numeric_cols)

print("\nCategorical Features:")
print(categorical_cols)


# ============================================================
# 7. DATA PREPROCESSING
# ============================================================

# Impute missing numerical values using the median.
# Scaling is included in the PCA branch because PCA is
# sensitive to differences in feature scales.

num_pipe_no_pca = make_pipeline(
    SimpleImputer(strategy="median")
)

num_pipe_pca = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    PCA(svd_solver="full")
)

# Impute missing categorical values and one-hot encode them.
# Dense output is used for compatibility across the ensemble
# algorithms in this example.

cat_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
)

# Build preprocessing transformers.
# Only add a transformer when that type of feature exists.

transformers_no_pca = [
    ("num", num_pipe_no_pca, numeric_cols)
]

transformers_pca = [
    ("num", num_pipe_pca, numeric_cols)
]

if len(categorical_cols) > 0:
    transformers_no_pca.append(
        ("cat", cat_pipe, categorical_cols)
    )

    transformers_pca.append(
        ("cat", cat_pipe, categorical_cols)
    )

# Create the two preprocessing objects.
preprocessing_no_pca = ColumnTransformer(
    transformers=transformers_no_pca
)

preprocessing_pca = ColumnTransformer(
    transformers=transformers_pca
)


# ============================================================
# 8. DEFINE STRATIFIED CROSS-VALIDATION
# ============================================================

# StratifiedKFold keeps class proportions approximately
# consistent across the five folds.
# Use the same random_state and CV settings for comparisons.

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 9. DEFINE CLASSIFIERS AND PARAMETER GRIDS
# ============================================================

# AdaBoost:
# n_estimators = number of boosting rounds.
# learning_rate = contribution of each boosting round.

# Bagging:
# n_estimators = number of base estimators.
# max_samples = fraction of training samples per estimator.
# max_features = fraction of features per estimator.

# Gradient Boosting:
# n_estimators = number of boosting stages.
# learning_rate = contribution of each stage.
# max_depth = depth of the individual base trees.

models = {
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "classifier__n_estimators": [50, 100],
            "classifier__learning_rate": [0.1, 1.0]
        }
    ),

    "Bagging": (
        BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            random_state=42,
            n_jobs=1
        ),
        {
            "classifier__n_estimators": [10, 50, 100],
            "classifier__max_samples": [0.7, 1.0],
            "classifier__max_features": [0.7, 1.0]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "classifier__n_estimators": [50, 100],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [1, 3]
        }
    )
}


# ============================================================
# 10. TRAIN AND TUNE EACH CLASSIFIER
# ============================================================

# Store grids, fitted models, fold scores and test metrics.
results = {}

for name, (classifier, param_grid) in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Pipeline WITHOUT PCA.
    # "classifier" is the step name used in parameter grids.

    pipeline_no_pca = Pipeline([
        ("preprocessor", preprocessing_no_pca),
        ("classifier", classifier)
    ])

    # Pipeline WITH PCA.
    pipeline_pca = Pipeline([
        ("preprocessor", preprocessing_pca),
        ("classifier", classifier)
    ])

    # Add PCA component selection to the PCA grid.
    # Fractions retain the specified amount of variance.

    param_grid_pca = {
        **param_grid,
        "preprocessor__num__pca__n_components": [
            0.80, 0.90, 0.95
        ]
    }

    # Tune the model WITHOUT PCA.
    grid_no_pca = GridSearchCV(
        estimator=pipeline_no_pca,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )

    # Tune the model WITH PCA.
    grid_pca = GridSearchCV(
        estimator=pipeline_pca,
        param_grid=param_grid_pca,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )

    # Fit both grid searches using only the training data.
    grid_no_pca.fit(X_train, y_train)
    grid_pca.fit(X_train, y_train)

    # Save the best fitted pipelines.
    best_model_no_pca = grid_no_pca.best_estimator_
    best_model_pca = grid_pca.best_estimator_

    # Store results for later comparisons.
    results[name] = {
        "grid_no_pca": grid_no_pca,
        "grid_pca": grid_pca,
        "best_model_no_pca": best_model_no_pca,
        "best_model_pca": best_model_pca
    }

    # Display best hyperparameters and CV macro F1.
    print("\nBest Parameters WITHOUT PCA:")
    print(grid_no_pca.best_params_)

    print("Best CV Macro F1 WITHOUT PCA:")
    print(round(grid_no_pca.best_score_, 4))

    print("\nBest Parameters WITH PCA:")
    print(grid_pca.best_params_)

    print("Best CV Macro F1 WITH PCA:")
    print(round(grid_pca.best_score_, 4))


# ============================================================
# 11. EVALUATE BOTH MODELS ON THE TEST SET
# ============================================================

# Define a reusable evaluation function so we don't have
# to repeat the same evaluation code for every classifier.

def evaluate_model(model, X_test, y_test, model_name):

    # Predict class labels for the test data.
    y_pred = model.predict(X_test)

    # Print classification metrics.
    print("\n" + "-" * 50)
    print(model_name)
    print("-" * 50)

    print("Accuracy:")
    print(round(accuracy_score(y_test, y_pred), 4))

    print("Precision (Macro):")
    print(round(
        precision_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        ), 4
    ))

    print("Recall (Macro):")
    print(round(
        recall_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        ), 4
    ))

    print("F1-score (Macro):")
    print(round(
        f1_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        ), 4
    ))

    # Print precision, recall and F1 for every class.
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        zero_division=0
    ))

    # Plot the confusion matrix.
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        xticks_rotation="vertical"
    )

    plt.title("Confusion Matrix - " + model_name)
    plt.tight_layout()
    plt.show()

    # Return predictions and test metrics for comparison.
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision_Macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Recall_Macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "F1_Macro": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )
    }


# Evaluate all six fitted pipelines.
test_results = {}

for name, saved in results.items():

    test_results[name + " without PCA"] = evaluate_model(
        saved["best_model_no_pca"],
        X_test, y_test,
        name + " WITHOUT PCA"
    )

    test_results[name + " with PCA"] = evaluate_model(
        saved["best_model_pca"],
        X_test, y_test,
        name + " WITH PCA"
    )


# ============================================================
# 12. EXTRACT FIVE-FOLD CV SCORES
# ============================================================

# GridSearchCV has already stored each fold's validation score.
# Because scoring="f1_macro", these are macro F1 scores.
# No extra cross_val_score call is needed for this extraction.

for name, saved in results.items():

    grid_no_pca = saved["grid_no_pca"]
    grid_pca = saved["grid_pca"]

    # Find the best parameter combination in each grid.
    best_index_no_pca = grid_no_pca.best_index_
    best_index_pca = grid_pca.best_index_

    # Extract fold scores for the best non-PCA model.
    scores_no_pca = np.array([
        grid_no_pca.cv_results_[f"split{i}_test_score"][
            best_index_no_pca
        ]
        for i in range(5)
    ])

    # Extract fold scores for the best PCA model.
    scores_pca = np.array([
        grid_pca.cv_results_[f"split{i}_test_score"][
            best_index_pca
        ]
        for i in range(5)
    ])

    # Save fold scores for later statistical comparisons.
    saved["scores_no_pca"] = scores_no_pca
    saved["scores_pca"] = scores_pca

    print("\n" + "=" * 50)
    print("Five-Fold CV Scores:", name)
    print("=" * 50)

    print("Without PCA:", np.round(scores_no_pca, 4))
    print("Mean F1 without PCA:", round(scores_no_pca.mean(), 4))
    print("SD without PCA:", round(scores_no_pca.std(ddof=1), 4))

    print("\nWith PCA:", np.round(scores_pca, 4))
    print("Mean F1 with PCA:", round(scores_pca.mean(), 4))
    print("SD with PCA:", round(scores_pca.std(ddof=1), 4))


# ============================================================
# 13. WILCOXON SIGNED-RANK TEST: PCA VS NO PCA
# ============================================================

# Compare the five fold-wise macro F1 scores for each algorithm.
# This is a paired comparison because the same CV splits are used.
# Higher F1 is better.

for name, saved in results.items():

    scores_no_pca = saved["scores_no_pca"]
    scores_pca = saved["scores_pca"]

    print("\nWilcoxon Test:", name)

    try:
        stat, p_value = wilcoxon(
            scores_no_pca,
            scores_pca
        )

        print("Statistic:", stat)
        print("P-value:", round(p_value, 6))

        if p_value < 0.05:
            print(
                "A statistically significant difference was detected."
            )
        else:
            print(
                "Insufficient evidence of a statistically "
                "significant difference."
            )

    except ValueError as error:
        # This can happen, for example, if all paired differences
        # are zero and the test cannot calculate a statistic.
        print("Wilcoxon test could not be calculated:", error)


# ============================================================
# 14. FINAL MODEL COMPARISON
# ============================================================

# Combine the test metrics of all six pipelines into one table.
comparison = pd.DataFrame(test_results).T

print("\n" + "=" * 60)
print("FINAL TEST SET COMPARISON")
print("=" * 60)

print(comparison.round(4).to_string())

# Select the model with the highest test macro F1.
# This is a descriptive comparison on the chosen test split.

best_name = comparison["F1_Macro"].idxmax()

print("\nBest model by test Macro F1:", best_name)
print("Test Macro F1:", round(
    comparison.loc[best_name, "F1_Macro"], 4
))

# Note: The test set is useful for final evaluation.
# Do not repeatedly tune models based on test-set results.

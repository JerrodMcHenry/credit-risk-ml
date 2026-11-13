"""Milestone 2: leakage-safe preprocessing and a logistic regression baseline.

Run from the repository root:

    python src/train.py

The data are split once into a stratified training set (80%) and a held-out test set (20%).
Both models are evaluated with stratified 5-fold cross-validation on the training set only.
The test set is set aside for Milestone 3 and is not used here.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "credit_data.csv"
RANDOM_STATE = 42

ORIGINAL_TARGET = "credit_risk"  # 0 = bad, 1 = good (official coding)
TARGET = "bad_risk"  # 1 = bad, 0 = good (modelling target)
NUMERIC_FEATURES = ["duration", "amount", "age"]
# Every other predictor is an integer code for a category. Ordinal codes are one-hot encoded too,
# because the credit officers' code order is not evidence of equally spaced, linear effects.
CATEGORICAL_FEATURES = [
    "status", "credit_history", "purpose", "savings", "employment_duration", "installment_rate",
    "personal_status_sex", "other_debtors", "present_residence", "property", "other_installment_plans",
    "housing", "number_credits", "job", "people_liable", "telephone", "foreign_worker",
]

# Precision, recall and F1 refer to the positive class bad_risk = 1.
# zero_division=0 scores a model that never predicts "bad" as 0 rather than raising a warning.
# "roc_auc" ranks continuous scores (decision_function or predict_proba), never hard class predictions.
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    "roc_auc": "roc_auc",
}


def load_data(path=DATA_PATH):
    """Return predictors X and target y (bad_risk = 1 - credit_risk). The original target is not kept in X."""
    df = pd.read_csv(path)
    assert set(df[ORIGINAL_TARGET].unique()) == {0, 1}, "credit_risk must be coded 0/1"
    y = (1 - df[ORIGINAL_TARGET]).rename(TARGET)
    X = df.drop(columns=[ORIGINAL_TARGET])
    assert sorted(X.columns) == sorted(NUMERIC_FEATURES + CATEGORICAL_FEATURES), "unexpected predictor columns"
    return X, y


def split_data(X, y):
    """Stratified 80/20 split. The test part is held out and returned untouched."""
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def build_models():
    """The preprocessor sits inside the pipeline, so it is refit on the training folds of each CV split."""
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
    }


def cross_validate_models(models, X_train, y_train):
    """Evaluate every model on the same stratified folds of the training data. Returns one row per model."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = {}
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=folds, scoring=SCORING)
        rows[name] = {
            metric: (scores[f"test_{metric}"].mean(), scores[f"test_{metric}"].std())
            for metric in SCORING
        }
    return rows


def format_results(rows):
    table = pd.DataFrame({
        name: {metric: f"{mean:.3f} ± {std:.3f}" for metric, (mean, std) in metrics.items()}
        for name, metrics in rows.items()
    }).T
    table.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    return table


def main(data_path=DATA_PATH):
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Data: {len(X)} rows, {X.shape[1]} predictors "
          f"({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)")
    for name, part in [("train", y_train), ("test (held out)", y_test)]:
        print(f"  {name:16} n={len(part):4d}  bad_risk=1: {int(part.sum()):3d} ({part.mean():.1%})")

    results = cross_validate_models(build_models(), X_train, y_train)

    print("\nStratified 5-fold cross-validation on the training set (mean ± std across folds)")
    print("Positive class: bad_risk = 1\n")
    print(format_results(results).to_string())
    print(f"\nThe held-out test set ({len(X_test)} rows) was not used. It is reserved for Milestone 3.")
    return results


if __name__ == "__main__":
    main()

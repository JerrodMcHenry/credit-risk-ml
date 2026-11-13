"""Model comparison and final held-out evaluation (Milestones 2 and 3).

Run from the repository root:

    python src/train.py

1. Split the data once: stratified 80% training set and 20% held-out test set.
2. Compare a majority-class baseline, logistic regression and a random forest with stratified
   5-fold cross-validation on the training set only (identical folds for every model).
3. Select the preferred model from the cross-validation results only.
4. Refit logistic regression and random forest on the full training set and evaluate each once
   on the held-out test set at the default 0.5 threshold. Nothing is changed after that.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score, make_scorer, precision_score,
    recall_score, roc_auc_score,
)
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
# "neg_brier_score" uses predict_proba for bad_risk = 1; scikit-learn negates it so that higher is better,
# and it is flipped back to a positive loss (lower is better) when reported.
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    "roc_auc": "roc_auc",
    "brier": "neg_brier_score",
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
        "Random forest": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def cross_validate_models(models, X_train, y_train):
    """Evaluate every model on the same stratified folds of the training data. Returns one row per model."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = {}
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=folds, scoring=SCORING)
        scores["test_brier"] = -scores["test_brier"]  # back to a positive loss
        rows[name] = {
            metric: (scores[f"test_{metric}"].mean(), scores[f"test_{metric}"].std())
            for metric in SCORING
        }
    return rows


def select_model(cv_rows):
    """Pick logistic regression or random forest from training cross-validation results only.

    Primary metric: ROC-AUC. The random forest is preferred only if its mean ROC-AUC beats logistic
    regression by more than the larger fold standard deviation of the two; otherwise the difference is
    treated as within cross-validation noise and the simpler, more interpretable logistic regression is kept.
    This is a practical rule of thumb, not a statistical significance test.
    """
    lr, rf = cv_rows["Logistic regression"], cv_rows["Random forest"]
    auc_gain = rf["roc_auc"][0] - lr["roc_auc"][0]
    noise = max(lr["roc_auc"][1], rf["roc_auc"][1])
    selected = "Random forest" if auc_gain > noise else "Logistic regression"
    rationale = [
        f"ROC-AUC (primary): random forest {rf['roc_auc'][0]:.3f} vs logistic regression {lr['roc_auc'][0]:.3f} "
        f"(difference {auc_gain:+.3f}; larger fold std {noise:.3f}).",
        "The difference is within cross-validation variability." if auc_gain <= noise
        else "The random forest's advantage exceeds cross-validation variability.",
        f"Recall for bad credit: LR {lr['recall'][0]:.3f} ± {lr['recall'][1]:.3f}, RF {rf['recall'][0]:.3f} ± {rf['recall'][1]:.3f}. "
        f"F1: LR {lr['f1'][0]:.3f}, RF {rf['f1'][0]:.3f}. Precision: LR {lr['precision'][0]:.3f}, RF {rf['precision'][0]:.3f}. "
        f"Brier: LR {lr['brier'][0]:.3f}, RF {rf['brier'][0]:.3f}.",
    ]
    if selected == "Logistic regression":
        rationale.append("Logistic regression is kept: it is simpler and easier to interpret, and the "
                         "secondary metrics give no compelling reason to prefer the random forest.")
    return selected, rationale


def fit_final_models(models, X_train, y_train):
    """Fit each model on the full training set (preprocessing included)."""
    return {name: model.fit(X_train, y_train) for name, model in models.items()}


def evaluate_on_test(fitted_models, X_test, y_test):
    """One-time evaluation on the held-out test set at the default 0.5 threshold."""
    results = {}
    for name, model in fitted_models.items():
        predicted = model.predict(X_test)
        probability_bad = model.predict_proba(X_test)[:, 1]  # column 1 = bad_risk = 1
        results[name] = {
            "accuracy": accuracy_score(y_test, predicted),
            "precision": precision_score(y_test, predicted, pos_label=1, zero_division=0),
            "recall": recall_score(y_test, predicted, pos_label=1, zero_division=0),
            "f1": f1_score(y_test, predicted, pos_label=1, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability_bad),
            "brier": brier_score_loss(y_test, probability_bad),
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]),
        }
    return results


def format_confusion_matrix(matrix):
    (tn, fp), (fn, tp) = matrix
    return pd.DataFrame(
        [[f"TN = {tn}", f"FP = {fp}"], [f"FN = {fn}", f"TP = {tp}"]],
        index=["actual good (0)", "actual bad (1)"],
        columns=["predicted good (0)", "predicted bad (1)"],
    )


def format_results(rows):
    table = pd.DataFrame({
        name: {metric: f"{mean:.3f} ± {std:.3f}" for metric, (mean, std) in metrics.items()}
        for name, metrics in rows.items()
    }).T
    table.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Brier"]
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
    print("Brier score is a loss: lower is better.")

    # Model selection is fixed here, before the test set is touched.
    selected, rationale = select_model(results)
    print(f"\nSelected model (from training cross-validation only): {selected}")
    for line in rationale:
        print(f"  - {line}")

    candidates = {name: model for name, model in build_models().items() if name != "Majority class"}
    final_models = fit_final_models(candidates, X_train, y_train)
    test_results = evaluate_on_test(final_models, X_test, y_test)

    print(f"\nHeld-out test set: one-time evaluation (n={len(X_test)}, threshold 0.5)\n")
    test_table = pd.DataFrame({
        name: {metric: f"{value:.3f}" for metric, value in metrics.items() if metric != "confusion_matrix"}
        for name, metrics in test_results.items()
    }).T
    test_table.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Brier"]
    test_table.index = [f"{name} (selected)" if name == selected else name for name in test_table.index]
    print(test_table.to_string())

    print("\nConfusion matrices (positive class: bad_risk = 1)")
    print("TP: bad predicted bad | FP: good predicted bad | TN: good predicted good | FN: bad predicted good")
    for name, metrics in test_results.items():
        print(f"\n{name}")
        print(format_confusion_matrix(metrics["confusion_matrix"]).to_string())

    print(f"\nInterpretation ({selected}: held-out test vs training cross-validation)")
    for metric, label in [("roc_auc", "ROC-AUC"), ("recall", "Recall"), ("precision", "Precision"),
                          ("f1", "F1"), ("brier", "Brier")]:
        mean, std = results[selected][metric]
        test_value = test_results[selected][metric]
        within = "within" if abs(test_value - mean) <= 2 * std else "outside"
        print(f"  {label:9} test {test_value:.3f} vs CV {mean:.3f} ± {std:.3f} ({within} 2 fold std)")
    other = next(name for name in test_results if name != selected)
    if test_results[other]["roc_auc"] > test_results[selected]["roc_auc"]:
        print(f"  {other} scored higher on the test set (ROC-AUC {test_results[other]['roc_auc']:.3f} vs "
              f"{test_results[selected]['roc_auc']:.3f}). The selection is not revised: it was fixed from")
        print("  cross-validation before testing, and changing it now would turn the test set into a selection set.")
    print(f"  The test set has only {len(y_test)} rows ({int(y_test.sum())} bad), so each test metric is itself uncertain.")
    print("  Bad credits are oversampled (30% here, about 5% in the original portfolio). Predicted probabilities")
    print("  and Brier scores therefore reflect this sample, not real-world default rates.")
    print("  Educational model on 1970s data: not suitable for lending decisions.")

    return {"cv": results, "selected": selected, "test": test_results}


if __name__ == "__main__":
    main()

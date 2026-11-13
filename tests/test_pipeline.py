import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train  # noqa: E402


def test_target_is_inverted_and_excluded_from_features():
    X, y = train.load_data()
    raw = pd.read_csv(train.DATA_PATH)

    assert y.name == "bad_risk"
    assert (y == 1 - raw["credit_risk"]).all()
    assert int(y.sum()) == 300  # 300 bad credits in the source data
    assert "credit_risk" not in X.columns and "bad_risk" not in X.columns
    assert X.shape == (1000, 20)


def test_preprocessor_outputs_finite_numbers_with_missing_and_unseen_values():
    X, y = train.load_data()
    X_train, X_test, _, _ = train.split_data(X, y)

    X_train = X_train.astype(float)  # allow NaN in integer columns
    X_train.iloc[:5, X_train.columns.get_loc("age")] = np.nan
    X_train.iloc[:5, X_train.columns.get_loc("purpose")] = np.nan
    X_test = X_test.astype(float)
    X_test.iloc[0, X_test.columns.get_loc("purpose")] = 99  # category never seen in training

    preprocessor = train.build_preprocessor().fit(X_train)
    for transformed, n_rows in [(preprocessor.transform(X_train), len(X_train)),
                                (preprocessor.transform(X_test), len(X_test))]:
        assert transformed.shape[0] == n_rows
        assert np.issubdtype(transformed.dtype, np.floating)
        assert np.isfinite(transformed).all()


def test_workflow_keeps_test_set_separate_and_uses_identical_folds(monkeypatch):
    X, y = train.load_data()
    X_train, X_test, y_train, y_test = train.split_data(X, y)

    assert (len(X_train), len(X_test)) == (800, 200)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - 0.30) < 0.005 and abs(y_test.mean() - 0.30) < 0.005

    cv_rows, cv_folds, fit_rows, eval_rows = [], [], [], []
    real_cross_validate = train.cross_validate
    real_fit_final_models = train.fit_final_models
    real_evaluate_on_test = train.evaluate_on_test

    def recording_cross_validate(model, X_cv, y_cv, **kwargs):
        cv_rows.append(set(X_cv.index))
        cv_folds.append([(tuple(a), tuple(b)) for a, b in kwargs["cv"].split(X_cv, y_cv)])
        return real_cross_validate(model, X_cv, y_cv, **kwargs)

    def recording_fit_final_models(models, X_fit, y_fit):
        fit_rows.append(set(X_fit.index))
        return real_fit_final_models(models, X_fit, y_fit)

    def recording_evaluate_on_test(models, X_eval, y_eval):
        eval_rows.append(set(X_eval.index))
        return real_evaluate_on_test(models, X_eval, y_eval)

    monkeypatch.setattr(train, "cross_validate", recording_cross_validate)
    monkeypatch.setattr(train, "fit_final_models", recording_fit_final_models)
    monkeypatch.setattr(train, "evaluate_on_test", recording_evaluate_on_test)
    output = train.main()

    # Cross-validation: all three models, training rows only, identical folds.
    assert set(output["cv"]) == {"Majority class", "Logistic regression", "Random forest"}
    assert len(cv_rows) == 3 and all(rows == set(X_train.index) for rows in cv_rows)
    assert cv_folds[0] == cv_folds[1] == cv_folds[2]

    # Selection is one of the two predictive models; final fit on training rows; evaluation once, on test rows only.
    assert output["selected"] in {"Logistic regression", "Random forest"}
    assert fit_rows == [set(X_train.index)]
    assert eval_rows == [set(X_test.index)]
    assert set(output["test"]) == {"Logistic regression", "Random forest"}


def test_predictions_and_probabilities_are_valid():
    X, y = train.load_data()
    X_train, _, y_train, _ = train.split_data(X, y)
    X_fit, X_check, y_fit, y_check = X_train.iloc[:600], X_train.iloc[600:], y_train.iloc[:600], y_train.iloc[600:]

    models = {k: v for k, v in train.build_models().items() if k != "Majority class"}
    results = train.evaluate_on_test(train.fit_final_models(models, X_fit, y_fit), X_check, y_check)
    for name, model in models.items():
        probabilities = model.predict_proba(X_check)
        assert list(model.classes_) == [0, 1]  # column 1 is bad_risk = 1
        assert probabilities.shape == (len(X_check), 2)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert np.allclose(probabilities.sum(axis=1), 1)
        assert set(np.unique(model.predict(X_check))) <= {0, 1}
        assert results[name]["confusion_matrix"].sum() == len(X_check)
        assert 0 <= results[name]["brier"] <= 1 and 0 <= results[name]["roc_auc"] <= 1

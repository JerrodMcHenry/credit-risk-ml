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


def test_cross_validation_only_sees_training_rows(monkeypatch):
    X, y = train.load_data()
    X_train, X_test, y_train, y_test = train.split_data(X, y)

    assert (len(X_train), len(X_test)) == (800, 200)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - 0.30) < 0.005 and abs(y_test.mean() - 0.30) < 0.005

    seen = []
    real_cross_validate = train.cross_validate

    def recording_cross_validate(model, X_cv, y_cv, **kwargs):
        seen.append(set(X_cv.index))
        return real_cross_validate(model, X_cv, y_cv, **kwargs)

    monkeypatch.setattr(train, "cross_validate", recording_cross_validate)
    train.main()

    assert len(seen) == len(train.build_models())
    for rows in seen:
        assert rows == set(X_train.index)
        assert rows.isdisjoint(X_test.index)

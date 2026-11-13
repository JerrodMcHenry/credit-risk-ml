# credit-risk-ml

A small machine-learning portfolio project on credit risk, using the
[UCI South German Credit](https://archive.ics.uci.edu/dataset/522/south+german+credit) dataset
(1,000 credits, 1973–1975, CC BY 4.0).

> **Educational use only.** The dataset is historical and deliberately oversamples bad credits. Results are not
> real-world default probabilities, do not establish causal effects, and must not be used for lending decisions.

## Status

| Milestone | Scope | Status |
|---|---|---|
| 1 | Data acquisition, validation and exploratory analysis | ✅ Done |
| 2 | Leakage-safe preprocessing, logistic regression vs majority-class baseline (cross-validation only) | ✅ Done |
| 3 | Random forest comparison, model selection, one-time held-out test evaluation | ✅ Done |

## Repository layout

```text
credit-risk-ml/
├── data/
│   ├── credit_data.csv      # official data, English column names, original integer codes
│   └── README.md            # source, hashes, license, citation, code table, limitations
├── notebooks/
│   └── analysis.ipynb       # Milestone 1 EDA (executed, outputs included)
├── src/
│   └── train.py             # cross-validation comparison, model selection, held-out test evaluation
├── tests/
│   └── test_pipeline.py     # pytest checks: target, preprocessing, folds, test-set separation, outputs
├── requirements.txt
└── README.md
```

## Setup

Developed and tested on Python 3.14.6.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the exploratory analysis (Milestone 1)

```bash
# Re-run the notebook from top to bottom (it checks the CSV's SHA-256 before analysing)
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb

# Or explore it interactively
jupyter notebook notebooks/analysis.ipynb
```

`data/credit_data.csv` is committed, so no download is needed. `data/README.md` contains the exact code that recreates
the CSV from the official UCI archive, with the archive's SHA-256 check.

## Milestone 1 highlights

All rates below are **within this oversampled sample**, so they are only useful for comparing groups:

- Checking-account status shows the widest spread: the bad-risk rate is **11.7%** for balances ≥ 200 DM or a salary account,
  and **49.3%** for applicants with no checking account.
- The bad-risk rate rises with credit duration, from **21.2%** (≤ 12 months) to **51.7%** (37–72 months).
- Bad credits have somewhat larger recorded amounts (median 2,574.5 vs 2,244, on a transformed scale) and a heavier upper tail.
  The two distributions overlap heavily.

See the notebook for the full validation, tables, plots and a discussion of limitations.

## Training and evaluation workflow (Milestones 2–3)

Run from the repository root (with the virtual environment active):

```bash
python src/train.py      # split → cross-validate 3 models → select → one-time test evaluation
python -m pytest         # run the checks in tests/
```

The whole run is deterministic (`random_state=42` everywhere) and takes a few seconds.

### Steps

1. **Target.** `bad_risk = 1 - credit_risk`, so `bad_risk = 1` means a bad credit and is the positive class.
   The original `credit_risk` column is dropped from the predictors, so the target can't leak into them.
2. **Split.** A stratified 80/20 split gives 800 training rows (240 bad, 560 good) and 200 held-out test rows (60 bad, 140 good).
3. **Preprocessing.** All models share one preprocessing pipeline:
   - The 3 numeric features (`duration`, `amount`, `age`) get median imputation, then standard scaling.
   - The 17 integer-coded categorical features get most-frequent imputation, then one-hot encoding (`handle_unknown="ignore"`).
   - Ordinal codes are one-hot encoded as well, because the historical credit officers' code order doesn't show equally spaced, linear effects.
4. **Models** (default settings, no tuning, no class weighting):
   - majority-class baseline (`DummyClassifier(strategy="most_frequent")`), kept for context;
   - logistic regression (`max_iter=1000`);
   - random forest (`n_estimators=200`).
5. **Cross-validation.** Stratified 5-fold cross-validation (`shuffle=True`) on the **800 training rows only**, with identical folds for all three models.
6. **Model selection.** This uses the cross-validation results only, and the rule is fixed in code (`select_model`).
   ROC-AUC is the primary metric. The random forest is preferred only if its mean ROC-AUC beats logistic regression's by more than
   the larger fold standard deviation of the two. Otherwise the difference counts as cross-validation noise and the simpler model is kept.
   This is a rule of thumb, not a significance test.
7. **Held-out test.** Logistic regression and the random forest are refit on all 800 training rows and evaluated **once** on the
   200 test rows at the default 0.5 threshold. Model choice, preprocessing, hyperparameters and threshold were all fixed before this step,
   and none were changed afterwards.

### Why preprocessing lives inside the pipeline

Scaling statistics, imputation values and the set of one-hot categories are all *learned* from the data. If they were fitted on
the full dataset before cross-validation, each validation fold would have influenced its own preprocessing and the scores
would be optimistic. Inside a `Pipeline`, `cross_validate` refits the preprocessing on the training folds of every split, and the
final models learn it from the training set alone.

### Why stratified cross-validation

Only 30% of rows are bad credits. Stratification keeps that ratio the same in every fold, which makes the fold scores comparable and
stops recall and precision from swinging just because a fold happened to contain fewer bad credits. Using five folds also shows how much
the scores vary, which the model-selection rule relies on.

### Metrics (positive class: bad credit)

| Metric | Meaning |
|---|---|
| Accuracy | Share of all credits classified correctly. Misleading here: always predicting "good" already scores 70%. |
| Precision | Of the credits flagged as bad, the share that really were bad. |
| Recall | Of the truly bad credits, the share that were flagged. |
| F1 | Harmonic mean of precision and recall. |
| ROC-AUC | Probability that a randomly chosen bad credit gets a higher risk score than a randomly chosen good one. It measures **ranking** and doesn't depend on a threshold. 0.5 means no ranking ability. |
| Brier score | Mean squared difference between the predicted probability of "bad" and the actual outcome (0/1). It measures the **accuracy of the probabilities themselves**, so it is a loss: lower is better. Always predicting 0 for "bad" gives 0.30 here. |

ROC-AUC and Brier answer different questions. A model can rank credits well (high AUC) while its probabilities are systematically too
high or too low (poor Brier), and the reverse can also happen.

Precision, recall, F1 and accuracy depend on the 0.5 threshold. Lowering the threshold flags more credits as bad, which raises recall
and lowers precision; raising it does the opposite. Every comparison here uses 0.5, and no threshold was tuned.

### Cross-validation results (training set, 5 folds, mean ± std)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier ↓ |
|---|---|---|---|---|---|---|
| Majority class | 0.700 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.300 ± 0.000 |
| Logistic regression | 0.744 ± 0.020 | 0.595 ± 0.050 | 0.487 ± 0.047 | 0.533 ± 0.022 | 0.780 ± 0.027 | 0.170 ± 0.011 |
| Random forest | 0.744 ± 0.022 | 0.641 ± 0.096 | 0.362 ± 0.084 | 0.455 ± 0.061 | 0.788 ± 0.041 | 0.166 ± 0.012 |

The baseline reaches 70% accuracy while identifying **no** bad credits. That is why accuracy alone is not enough.

### Model selection: logistic regression

The decision was made from the cross-validation table above, before the test set was used:

- **ROC-AUC (primary):** the random forest is ahead by 0.008, much less than the fold standard deviations (0.027 and 0.041), so the two models are effectively tied.
- **Recall for bad credit:** logistic regression 0.487 vs random forest 0.362, a gap larger than either model's fold standard deviation.
  At 0.5, the random forest misses noticeably more bad credits.
- **F1:** logistic regression 0.533 ± 0.022 vs 0.455 ± 0.061, so it is both better and more stable.
- **Precision and Brier** slightly favour the random forest (0.641 vs 0.595; 0.166 vs 0.170). Its precision varies about twice as much across folds.
- **Simplicity and interpretability:** a single linear model with coefficients that can be inspected, versus 200 trees.

None of these differences is shown to be statistically significant. The choice reflects that nothing gives a compelling reason to prefer the more complex model.

### Held-out test results (200 rows, one-time evaluation, threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier ↓ |
|---|---|---|---|---|---|---|
| **Logistic regression (selected)** | 0.740 | 0.591 | 0.433 | 0.500 | 0.742 | 0.179 |
| Random forest | 0.780 | 0.711 | 0.450 | 0.551 | 0.777 | 0.166 |

Confusion matrices (rows = actual, columns = predicted; positive class = bad):

| Logistic regression | predicted good | predicted bad |
|---|---|---|
| **actual good (140)** | TN = 122 | FP = 18 |
| **actual bad (60)** | FN = 34 | TP = 26 |

| Random forest | predicted good | predicted bad |
|---|---|---|
| **actual good (140)** | TN = 129 | FP = 11 |
| **actual bad (60)** | FN = 33 | TP = 27 |

- **TP** (true positive): a bad credit correctly predicted as bad.
- **FP** (false positive): a good credit wrongly predicted as bad.
- **TN** (true negative): a good credit correctly predicted as good.
- **FN** (false negative): a bad credit wrongly predicted as good.

Both models miss more than half of the bad credits at the 0.5 threshold: logistic regression has 34 FN and the random forest 33.

### How to read the test results

- **The selected model is consistent with cross-validation.** Every logistic regression test metric is within two fold standard deviations of its
  cross-validation mean (for example ROC-AUC 0.742 vs 0.780 ± 0.027, recall 0.433 vs 0.487 ± 0.047). Its test results are a little lower but not out of line.
- **The random forest scored higher on this test set** on every metric. In counts, it caught one more bad credit (27 vs 26 TP) and
  raised 7 fewer false alarms (11 vs 18 FP). The cross-validation recall advantage of logistic regression did not show up here.
- **The selection is not revised.** It was fixed from cross-validation beforehand. Switching models after looking at the test set would
  turn the test set into a selection set, and its scores would no longer be an unbiased estimate. With only 200 rows (60 bad), a difference of a few
  predictions is also within the uncertainty of a single test split. A fair conclusion is that the two models perform similarly on this
  data, and the random forest may have an edge that this project does not establish.

### Interpreting probabilities and limitations

- **Oversampling.** Bad credits were deliberately oversampled: 30% of this sample is bad, while the sources report about 5% in the original
  portfolio. Predicted probabilities and Brier scores reflect this sample, **not real-world default probabilities**. A lower Brier score here does not
  show that the model is calibrated for any real lending population, and no recalibration was attempted.
- **Granted credits only.** Rejected applicants are missing, so the model has never seen the population a real lender would screen.
- **Historical data.** Credits from 1973–1975 in southern Germany do not represent modern borrowers or credit products.
- **Small sample.** With 1,000 rows, most of the differences between models reported here are within statistical noise.
- **Sensitive attributes.** The predictors include `personal_status_sex`, `age` and `foreign_worker`. No fairness analysis was done.

**This is an educational project. The models are not suitable for lending decisions.**

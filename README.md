# Credit Risk Classification: South German Credit

An end-to-end, reproducible machine-learning case study that classifies historical consumer credits as good or bad risk.
It covers data validation, exploratory analysis, leakage-safe preprocessing, model comparison with cross-validation, a single
held-out test evaluation, and model interpretation.

> **Educational project, not a lending system.** The data are from 1973–1975, bad credits are deliberately oversampled,
> and nothing here is suitable for real credit decisions.

## Project overview

A bank wants to know, at application time, which credits are likely to go bad. This project frames that as binary classification
on a well-documented public dataset. The goal is to show a sound workflow, not to build a production model:

- no information leaks from evaluation data into training;
- models are compared against a trivial baseline;
- a model is chosen before the final test, and the test result is reported honestly even when it is inconvenient;
- the limits of the data are stated plainly.

## Dataset

[UCI South German Credit](https://archive.ics.uci.edu/dataset/522/south+german+credit) (Grömping 2019; CC BY 4.0), a corrected
version of the widely used Statlog German Credit data. Full documentation, hashes, code table and label verification are in
[`data/README.md`](data/README.md).

| | |
|---|---|
| Observations | 1,000 credits from a southern German bank, 1973–1975 |
| Predictors | 20: 3 numeric (`duration`, `amount`, `age`) and 17 integer-coded categorical |
| Original target | `credit_risk`: 0 = bad (300), 1 = good (700) |
| Modelling target | `bad_risk = 1 - credit_risk` (1 = bad, the positive class) |
| Missing values / duplicates | None / none |

Known data limitations:

- **Oversampling.** Bad credits make up 30% of the sample but about 5% of the original portfolio.
- **Granted credits only.** Rejected applicants are absent.
- **Transformed amounts.** `amount` is a monotonic transformation of the real amount, and the transformation is unknown.
- **Sex can't be recovered** from `personal_status_sex`.
- **Coding.** Categorical codes are expert scores, not measured quantities.

## Technical stack

Python 3.14 · pandas · NumPy · Matplotlib · scikit-learn · Jupyter · pytest

## Methodology

1. **Data inspection.** The official UCI archive is checked by SHA-256. Every category code is checked against the official code table, and the labels were
   cross-checked against the original German code book.
2. **Exploratory analysis** ([`notebooks/analysis.ipynb`](notebooks/analysis.ipynb)). Class balance, descriptive statistics, bad-risk rates by
   group with sample sizes, and three plots.
3. **Train/test separation.** A single stratified 80/20 split (`random_state=42`) gives 800 training rows (240 bad) and 200 held-out test rows (60 bad).
4. **Leakage-safe preprocessing.** A `ColumnTransformer` handles both feature types inside each model's `Pipeline`:
   - numeric features get median imputation, then scaling;
   - categorical features get most-frequent imputation, then one-hot encoding with `handle_unknown="ignore"`.

   Ordinal codes are one-hot encoded too, because the exploratory analysis showed that their code order doesn't follow observed risk.
   The exploratory analysis used all 1,000 rows, including the later test rows, so it informed this encoding choice; nothing else in the pipeline is based on it.
5. **Majority-class baseline.** `DummyClassifier(strategy="most_frequent")` shows what "always predict good" achieves.
6. **Logistic regression** (`max_iter=1000`), using default settings.
7. **Random forest** (`n_estimators=200`), using default settings. Neither model was tuned or class-weighted.
8. **Cross-validation.** Stratified 5-fold cross-validation (`shuffle=True, random_state=42`) on the training set only, with identical folds for all models.
   Metrics are accuracy, precision, recall, F1, ROC-AUC and Brier score, with bad credit as the positive class.
9. **Model selection.** A rule fixed in code before testing: ROC-AUC is the primary metric, and the random forest is chosen only if its advantage exceeds
   cross-validation variability. Otherwise the simpler model wins.
10. **Final held-out evaluation.** Both models are refit on all 800 training rows and evaluated **once** on the 200 test rows at a 0.5 threshold.
    Nothing was changed afterwards.

## Architecture

```text
UCI South German Credit (zip, SHA-256 checked)
         |
         v
data/credit_data.csv  (validated, original codes)
         |
         +--> notebooks/analysis.ipynb --> EDA tables and 3 plots
         |
         v
src/train.py
         |
         v
Stratified 80/20 split
         |
         +--> Training data (800)
         |       |
         |       v
         |   Pipeline: ColumnTransformer -> model   (refit inside every CV fold)
         |       |
         |       v
         |   5-fold stratified CV: baseline, logistic regression, random forest
         |       |
         |       v
         |   Model selection (CV results only)
         |       |
         |       v
         |   Final fit on all 800 rows --> interpretation (coefficients, importances)
         |
         +--> Held-out test data (200)
                 |
                 v
         One-time final evaluation of the already fitted models
```

The test rows play no part in fitting the preprocessing, cross-validation, model selection or interpretation. They are used only to score the final models.
`tests/test_pipeline.py` enforces this by recording which rows reach cross-validation, the final fit and the evaluation.

## Results

### Cross-validation (training set, 5 folds, mean ± std)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier ↓ |
|---|---|---|---|---|---|---|
| Majority class | 0.700 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.300 ± 0.000 |
| Logistic regression | 0.744 ± 0.020 | 0.595 ± 0.050 | 0.487 ± 0.047 | 0.533 ± 0.022 | 0.780 ± 0.027 | 0.170 ± 0.011 |
| Random forest | 0.744 ± 0.022 | 0.641 ± 0.096 | 0.362 ± 0.084 | 0.455 ± 0.061 | 0.788 ± 0.041 | 0.166 ± 0.012 |

The baseline's 70% accuracy with zero bad credits found shows why accuracy alone is misleading here.

### Held-out test set (200 rows, evaluated once, threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier ↓ | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| **Logistic regression (selected)** | 0.740 | 0.591 | 0.433 | 0.500 | 0.742 | 0.179 | 26 | 18 | 122 | 34 |
| Random forest | 0.780 | 0.711 | 0.450 | 0.551 | 0.777 | 0.166 | 27 | 11 | 129 | 33 |

TP = bad credit predicted bad, FP = good credit predicted bad, TN = good credit predicted good, FN = bad credit predicted good.

### Why logistic regression was selected, and why that stands

- **Selection used cross-validation only.** The random forest's ROC-AUC lead (0.788 vs 0.780) is far smaller than the fold standard deviations
  (0.041 / 0.027), so on the primary metric the models are tied. At the 0.5 threshold, logistic regression found clearly more bad credits (recall 0.487 vs 0.362),
  had a better and more stable F1, and is simpler to interpret.
- **On the test set the random forest scored higher on every metric.** In counts, it caught one more bad credit and raised 7 fewer false alarms.
  Logistic regression's own test results are within two fold standard deviations of its cross-validation estimates.
- **The selection is not revised.** Switching after seeing test results would make the test set part of model selection, and the reported
  scores would no longer be an unbiased estimate. With 60 bad credits in the test set, the gap is within the uncertainty of one split. The fair
  conclusion is that both models perform similarly on this data.

**Metrics in brief.**
- **ROC-AUC** measures how well a model *ranks* bad credits above good ones, independent of any threshold.
- **Recall, precision and F1** describe decisions at the 0.5 threshold. Lowering the threshold raises recall and lowers precision.
- **Brier score** is the mean squared error of the predicted probabilities, so it measures the probabilities themselves; lower is better.

## Model interpretation

All values come from `python src/train.py` and use the models fitted on the 800 training rows. The labels come from the code table in `data/README.md`.

### Logistic regression coefficients

**How to read them:**
- Numeric features are standardised, so their coefficients are the change in log-odds per training-set standard deviation.
- Every category has its own 0/1 column, so there is **no reference category**. With the default L2 regularisation, compare categories of the
  same variable with each other rather than reading each coefficient on its own. For example, `foreign_worker` gets +0.621 for "no" and
  −0.620 for "yes": the gap between them is the effect, and neither value is a comparison against an omitted baseline.

| Higher predicted bad-risk log-odds | Coef. | n (train) | Lower predicted bad-risk log-odds | Coef. | n (train) |
|---|---|---|---|---|---|
| credit history: critical account / credits elsewhere | +0.903 | 42 | purpose: new car | −1.052 | 85 |
| checking account: none | +0.781 | 217 | checking account: ≥ 200 DM / salary account | −0.944 | 316 |
| purpose: other | +0.757 | 189 | credit history: all credits at this bank paid back | −0.915 | 226 |
| savings: unknown / none | +0.697 | 475 | savings: 500–1000 DM | −0.688 | 37 |
| foreign worker: no | +0.621 | 769 | foreign worker: yes | −0.620 | 31 |

Numeric coefficients: `amount` +0.368, `duration` +0.340, `age` −0.038, all per standard deviation.

The strongest signals are checking-account status and credit history. That agrees with the exploratory analysis, where these variables
had the widest spread in bad-risk rates. Some large coefficients rest on small groups: for example, foreign worker "yes" has n=31, and
savings 500–1000 DM has n=37. Those estimates are unstable, and `foreign_worker` is also a sensitive attribute.

### Random forest importances

These are impurity-based importances. They show how much each feature was used for splits, **not** whether it raises or lowers risk.

| Top transformed features | Importance | Summed per original variable | Importance |
|---|---|---|---|
| amount | 0.095 | status | 0.113 |
| duration | 0.075 | amount | 0.095 |
| age | 0.074 | purpose | 0.079 |
| checking account: ≥ 200 DM / salary account | 0.044 | duration | 0.075 |
| checking account: none | 0.042 | age | 0.074 |
| credit history: all credits at this bank paid back | 0.021 | credit_history | 0.061 |

Two things distort this ranking:
- **Numeric features look stronger than they are.** The three numeric features lead the per-column ranking partly because continuous features offer many split points, which impurity importance rewards.
- **One-hot encoding splits each categorical variable** across several columns. Summed per variable, checking-account status ranks first.

Correlated predictors also share importance, for example `duration` and `amount` (r ≈ 0.63).

**Both models are describing associations in an oversampled 1970s sample, not causes.** A large coefficient or importance doesn't make a
feature a real-world risk factor.

## Limitations

- **Historical data.** Credits from 1973–1975 don't represent modern borrowers, products or regulation.
- **Small sample.** 1,000 rows overall, and only 60 bad credits in the test set. Most differences between the models are within noise.
- **Oversampling.** 30% bad here vs about 5% in the original portfolio. **Predicted probabilities and Brier scores are not real-world default
  probabilities**, and no recalibration was done.
- **Selection bias.** Only granted credits were observed, so the model never sees the applicants a lender would actually reject.
- **Unknown amount transformation.** `amount` values aren't real DM amounts.
- **Sensitive attributes.** The predictors include `personal_status_sex`, `age` and `foreign_worker`, and **no fairness evaluation was done**.
- **Correlation, not causation.** Neither the coefficients nor the importances show that changing a feature would change risk.
- **No deployment.** There's no monitoring, no threshold chosen for a business cost, and no suitability for lending decisions.

## Reproducibility

Tested on Python 3.14.6 with the exact versions pinned in `requirements.txt`. All randomness is seeded (`random_state=42`), and repeated runs
in that pinned environment produce identical output. Other library versions can change the random forest results slightly.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb   # EDA (checks the CSV hash first)
python src/train.py                                                            # CV comparison, selection, test evaluation, interpretation
python -m pytest                                                               # 4 checks: target, preprocessing, separation, outputs
```

`data/credit_data.csv` is committed. `data/README.md` contains the code that rebuilds it from the official archive.

## Repository structure

```text
credit-risk-ml/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── README.md            # source, hashes, license, citation, code table, label verification, limitations
│   └── credit_data.csv      # 1,000 × 21, English column names, original integer codes
├── notebooks/
│   └── analysis.ipynb       # validation and exploratory analysis (executed, outputs included)
├── src/
│   └── train.py             # split, CV comparison, model selection, test evaluation, interpretation
└── tests/
    └── test_pipeline.py     # target, preprocessing, fold/test-set separation, prediction and interpretation checks
```

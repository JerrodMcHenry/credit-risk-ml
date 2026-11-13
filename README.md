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
| 3 | Final evaluation on the held-out test set | Not started |

## Repository layout

```text
credit-risk-ml/
├── data/
│   ├── credit_data.csv      # official data, English column names, original integer codes
│   └── README.md            # source, hashes, license, citation, code table, limitations
├── notebooks/
│   └── analysis.ipynb       # Milestone 1 EDA (executed, outputs included)
├── src/
│   └── train.py             # Milestone 2 training and cross-validation workflow
├── tests/
│   └── test_pipeline.py     # pytest checks: target, preprocessing, test-set separation
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

## Training workflow (Milestone 2)

Run from the repository root (with the virtual environment active):

```bash
python src/train.py      # split, cross-validate both models, print the comparison table
python -m pytest         # run the checks in tests/
```

### What it does

1. **Target.** `bad_risk = 1 - credit_risk`, so `bad_risk = 1` means a bad credit and is the positive class.
   The original `credit_risk` column is dropped from the predictors, so the target can't leak into them.
2. **Split.** A stratified 80/20 split (`random_state=42`) gives 800 training rows and 200 held-out test rows, each 30% bad.
   **The test set is reserved for Milestone 3.** It is not used for preprocessing, model choice, thresholds or evaluation in this milestone.
3. **Preprocessing.**
   - The 3 numeric features (`duration`, `amount`, `age`) get median imputation, then standard scaling.
   - The 17 integer-coded categorical features get most-frequent imputation, then one-hot encoding (`handle_unknown="ignore"`).
   - Ordinal codes are one-hot encoded as well, because the historical credit officers' code order doesn't show equally spaced, linear effects.
   - The data have no missing values; the imputers keep the pipeline safe if future data do.
4. **Models.**
   - A majority-class baseline (`DummyClassifier(strategy="most_frequent")`).
   - Logistic regression (`max_iter=1000`), using default settings with no tuning and no class weighting.
5. **Evaluation.** Stratified 5-fold cross-validation (`shuffle=True, random_state=42`) on the 800 training rows only,
   with the same folds for both models. Results are reported as mean ± standard deviation across folds.

### Why preprocessing lives inside the pipeline

Scaling statistics, imputation values and the set of one-hot categories are all *learned* from the data. If they were fitted on
the full dataset before cross-validation, each validation fold would have influenced its own preprocessing and the scores
would be optimistic. Inside a `Pipeline`, `cross_validate` refits the preprocessing on the training folds of every split, so each
validation fold is treated like unseen data.

### Why stratified cross-validation

Only 30% of rows are bad credits. Stratification keeps that ratio the same in every fold, which makes the fold scores comparable and
stops recall and precision from swinging just because a fold happened to contain fewer bad credits. Using five folds instead of
one validation split also shows how much the scores vary.

### Metrics (positive class: bad credit)

| Metric | Meaning |
|---|---|
| Accuracy | Share of all credits classified correctly. Misleading here: always predicting "good" already scores about 70%. |
| Precision | Of the credits flagged as bad, the share that really were bad. |
| Recall | Of the truly bad credits, the share that were flagged. |
| F1 | Harmonic mean of precision and recall. |
| ROC-AUC | Probability that a randomly chosen bad credit gets a higher risk score than a randomly chosen good one. It is based on scores rather than a 0.5 cutoff, so it doesn't depend on a threshold. 0.5 means no ranking ability. |

### Cross-validation results

Output of `python src/train.py` (training set only, 5 folds, mean ± std):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority class | 0.700 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Logistic regression | 0.744 ± 0.020 | 0.595 ± 0.050 | 0.487 ± 0.047 | 0.533 ± 0.022 | 0.780 ± 0.027 |

- **The baseline** reaches 70% accuracy while identifying **no** bad credits (recall 0). This is why accuracy alone is not enough.
- **Logistic regression** ranks credits far better than chance (ROC-AUC 0.78 vs 0.50). At the default 0.5 threshold, it catches
  about half of the bad credits (recall 0.49), and about 6 in 10 of its "bad" flags are correct (precision 0.60).
  A lower threshold would raise recall and lower precision. No threshold is chosen in this milestone.
- **These are cross-validation estimates on the training set.** The unbiased estimate comes from the held-out test set in Milestone 3.

### Interpreting predicted probabilities

Bad credits were deliberately oversampled: 30% of this sample is bad, while the sources report about 5% in the original portfolio.
The model's predicted probabilities are therefore calibrated to the sample, not to any real population, and **must not be
read as real-world default probabilities**. The model is an educational baseline on 1970s data and is **not suitable for lending decisions**.

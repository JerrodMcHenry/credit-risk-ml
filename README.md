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
| 2+ | Modelling | Not started |

## Repository layout

```text
credit-risk-ml/
├── data/
│   ├── credit_data.csv      # official data, English column names, original integer codes
│   └── README.md            # source, hashes, license, citation, code table, limitations
├── notebooks/
│   └── analysis.ipynb       # Milestone 1 EDA (executed, outputs included)
├── requirements.txt
└── README.md
```

## Reproduce the analysis

Developed and tested on Python 3.14.6.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

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

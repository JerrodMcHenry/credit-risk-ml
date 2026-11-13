# Data: South German Credit (UCI ID 522)

`credit_data.csv` is a lossless re-export of the official **UCI South German Credit** dataset:
1,000 consumer credits from a southern German bank, 1973–1975, with 20 predictors and a binary target.

## Source and license

| Item | Value |
|---|---|
| Dataset page | <https://archive.ics.uci.edu/dataset/522/south+german+credit> |
| Archive URL | <https://archive.ics.uci.edu/static/public/522/south+german+credit.zip> |
| Archive SHA-256 | `0b40d40eb7321693d559e247a556f88a6cc8df8489c3cb2ae084db7592584551` (13,130 bytes) |
| Archive contents | `SouthGermanCredit.asc` (data), `codetable.txt` (code meanings), `read_SouthGermanCredit.R` (official R reader) |
| `credit_data.csv` SHA-256 | `a4e28e9ca9e250ca619b052f73bf63c1f4ab9c8e883a2b944f68251ed55a02f3` (checked in the notebook) |
| DOI | [10.24432/C5X89F](https://doi.org/10.24432/C5X89F) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Redistributed here unmodified apart from the column names. |

### Citation

- South German Credit [Dataset]. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5X89F
- Grömping, U. (2019). *South German Credit Data: Correcting a Widely Used Data Set.* Report 4/2019,
  Reports in Mathematics, Physics and Chemistry, Department II, Beuth University of Applied Sciences Berlin.

The data were originally published in Fahrmeir, L. & Hamerle, A. (1984), *Multivariate statistische Verfahren*, de Gruyter.
This version corrects severe coding errors in the older, widely used **Statlog German Credit** dataset, so the two
datasets' code meanings are **not** interchangeable.

## How the CSV was produced

The archive is bzip2-compressed inside the zip, which macOS `unzip` cannot extract, so Python's `zipfile` is used instead.

```python
import hashlib, io, urllib.request, zipfile
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/522/south+german+credit.zip"
raw = urllib.request.urlopen(URL).read()
assert hashlib.sha256(raw).hexdigest() == "0b40d40eb7321693d559e247a556f88a6cc8df8489c3cb2ae084db7592584551"

df = pd.read_csv(zipfile.ZipFile(io.BytesIO(raw)).open("SouthGermanCredit.asc"), sep=r"\s+")
english = {  # German column names -> English names from the official R reader
    "laufkont": "status", "laufzeit": "duration", "moral": "credit_history", "verw": "purpose",
    "hoehe": "amount", "sparkont": "savings", "beszeit": "employment_duration", "rate": "installment_rate",
    "famges": "personal_status_sex", "buerge": "other_debtors", "wohnzeit": "present_residence",
    "verm": "property", "alter": "age", "weitkred": "other_installment_plans", "wohn": "housing",
    "bishkred": "number_credits", "beruf": "job", "pers": "people_liable", "telef": "telephone",
    "gastarb": "foreign_worker", "kredit": "credit_risk",
}
assert list(df.columns) == list(english) and df.shape == (1000, 21)
df.rename(columns=english).to_csv("data/credit_data.csv", index=False)
```

Only the column names change. Row order, column order and every value are kept as in `SouthGermanCredit.asc`.

## Target

| Column | Codes | Counts |
|---|---|---|
| `credit_risk` | 0 = bad (contract not complied with), 1 = good | 300 bad / 700 good |

The analysis derives `bad_risk = 1 - credit_risk` (1 = bad). The original column is not changed.

## Feature definitions

Every column is stored as an integer, but **only three are truly numerical**. All the others are integer codes for categories.
Category codes are Häußler's "P2" scores. They reflect **experienced credit officers' prior judgement** that a larger code means a better
credit risk, which is why some scales look reversed (for example, `installment_rate = 1` is the highest burden). That order is an expert opinion,
**not an empirical ranking**. In this sample, the top code has the highest bad-risk rate for `installment_rate` (33.4%), `property` (43.5%)
and `housing` (41.1%). Treat the codes as labels, not as measured risk levels.

### Numerical (3)

| Column | German | Meaning | Notes |
|---|---|---|---|
| `duration` | laufzeit | Credit duration in months | Range 4–72 |
| `amount` | hoehe | Credit amount | **A monotonic transformation of the original DM amount; the transformation is unknown.** Treat it as an ordered scale, not as true DM values. Range 250–18,424. |
| `age` | alter | Age in years | Range 19–75 |

### Integer-coded categorical (17)

| Column | German | Type | Codes |
|---|---|---|---|
| `status` | laufkont | nominal | 1 no checking account · 2 < 0 DM · 3 0 ≤ … < 200 DM · 4 ≥ 200 DM / salary for ≥ 1 yr |
| `credit_history` | moral | nominal | 0 delay in paying off in the past · 1 critical account/other credits elsewhere · 2 no credits taken/all paid back duly · 3 existing credits paid back duly till now · 4 all credits at this bank paid back duly |
| `purpose` | verw | nominal | 0 others · 1 car (new) · 2 car (used) · 3 furniture/equipment · 4 radio/television · 5 domestic appliances · 6 repairs · 7 education (**no observations**) · 8 vacation · 9 retraining · 10 business |
| `savings` | sparkont | ordinal | 1 unknown/no savings account · 2 < 100 DM · 3 100 ≤ … < 500 DM · 4 500 ≤ … < 1000 DM · 5 ≥ 1000 DM |
| `employment_duration` | beszeit | ordinal | 1 unemployed · 2 < 1 yr · 3 1 ≤ … < 4 yrs · 4 4 ≤ … < 7 yrs · 5 ≥ 7 yrs |
| `installment_rate` | rate | ordinal | Installment as % of disposable income: 1 ≥ 35 · 2 25 ≤ … < 35 · 3 20 ≤ … < 25 · 4 < 20 |
| `personal_status_sex` | famges | nominal | 1 male: divorced/separated · 2 female: non-single **or** male: single · 3 male: married/widowed · 4 female: single |
| `other_debtors` | buerge | nominal | 1 none · 2 co-applicant · 3 guarantor |
| `present_residence` | wohnzeit | ordinal | 1 < 1 yr · 2 1 ≤ … < 4 yrs · 3 4 ≤ … < 7 yrs · 4 ≥ 7 yrs |
| `property` | verm | ordinal | 1 unknown / no property · 2 car or other · 3 building society savings agreement / life insurance · 4 real estate |
| `other_installment_plans` | weitkred | nominal | 1 bank · 2 stores · 3 none |
| `housing` | wohn | nominal | 1 for free · 2 rent · 3 own. German source: 1 *kostenlose Wohnung* (rent-free), 2 *Mietwohnung* (rented flat), 3 *Eigentumswohnung* (owner-occupied flat). **This differs from the Statlog German Credit documentation**; see "Label verification" below. |
| `number_credits` | bishkred | ordinal | Credits at this bank: 1 one · 2 2–3 · 3 4–5 · 4 ≥ 6 |
| `job` | beruf | ordinal | 1 unemployed/unskilled, non-resident · 2 unskilled, resident · 3 skilled employee/official · 4 manager/self-employed/highly qualified |
| `people_liable` | pers | binary | 1 three or more · 2 zero to two |
| `telephone` | telef | binary | 1 no · 2 yes (under customer name) |
| `foreign_worker` | gastarb | binary | 1 yes · 2 no |

Binary features are coded **1/2, not 0/1**. The official R reader notes that `pers` and `gastarb` are coded the opposite way
to the older Statlog/Open Data LMU versions.

## Label verification (audit, 2026-11-13)

The code→label mapping was checked against three independent sources:

1. **UCI `codetable.txt`** (dataset 522). The notebook's labels match it for all 18 coded variables. The UCI "South German Credit (UPDATE)"
   entry (ID 573, DOI 10.24432/C5QG88) ships a byte-identical archive (same SHA-256), so it doesn't change any codes.
2. **Grömping (2019), Table 1**, which gives the percentage of bad and good credits in each category. Assuming `credit_risk = 0` is bad,
   this CSV reproduces all 69 category rows to within ±0.004 percentage points (rounding only). With the opposite assumption, 68 of 69 rows fail.
3. **Open Data LMU (2010) German code book** ([doi:10.5282/ubm/data.23](https://doi.org/10.5282/ubm/data.23)), based on Fahrmeir/Hamerle/Tutz.
   This is the original source, and every code and label matches it. Applying Grömping's documented recode (Appendix A: `pers` and `gastarb` → 3 − x,
   `kredit` moved last) to the LMU `kredit.asc` reproduces this CSV exactly. The code book has two minor issues, neither affecting the coding:
   - it lists `famges` code 2 as two subgroups, whose shares add up exactly to the merged code;
   - it gives 9.14% instead of 9.00% for `wohn = 3` among good credits, but its own housing column then adds to 100.14%, so 9.14 is a typo.

**Housing.** The order 1 = for free, 2 = rent, 3 = own is correct for this dataset. It differs from the Statlog German Credit documentation,
which Grömping (2019) explicitly lists as having mixed-up levels for attribute A15 (housing).
Renters have the lowest bad-risk rate (26.1%, n=714) and owners the highest (41.1%, n=107). This is counterintuitive, but it is a real feature of the data,
not a labelling error, and no explanation is given in the sources.

**Target.** Grömping's report says in one sentence that the response is "coded as 0 for good risks and 1 for bad risks". The UCI code table, the official R reader,
the LMU code book ("0 : Kredit wurde nicht ordnungsgemäß zurückgezahlt") and the report's own Table 1 all show **0 = bad, 1 = good**,
so that sentence is an error in the report. This project uses 0 = bad.

## Data quality (validated in `notebooks/analysis.ipynb`)

- 1,000 rows × 21 columns, all integers.
- No missing values and no exact duplicate rows.
- Every value is a valid code from the official code table. `purpose = 7` is allowed but never occurs.

## Known limitations

- **Oversampling of bad credits.** The sample was stratified so that bad credits are heavily overrepresented. The 30% bad share is
  a design choice, **not a real-world default probability**. The UCI documentation suggests using a cost matrix to account for this.
- **Granted credits only.** Rejected applicants are not in the data.
- **`amount` is transformed** with an unknown monotonic function.
- **`number_credits`** is binned, and the original values are not available.
- **Sex can't be recovered.** Code 2 of `personal_status_sex` combines female non-single and male single applicants, and female widows can't be identified reliably.
- **"Unknown" is merged with real answers** in `savings` (code 1) and `property` (code 1).
- **Historical data (1973–1975).** It does not represent modern borrowers, and some variables, such as `telephone`, are obsolete.
- **Sensitive attributes.** `personal_status_sex`, `age` and `foreign_worker` would raise legal and fairness concerns in any real lending use.

This dataset is for education and research only. It is not suitable for real credit decisions.

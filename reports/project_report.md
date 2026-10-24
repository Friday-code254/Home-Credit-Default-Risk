# Home Credit Default Risk: Project Report

## Cleaning results

This project focuses on three data problems with a clear treatment.

### Encoded missing dates

Home Credit uses 365243 as a placeholder in DAYS_EMPLOYED and five previous-application day-offset fields. It is not a real elapsed-day value. Both branches keep the same sentinel indicators and applicant-level counts. They also share the same history tables and bureau-balance linkage rules. The only difference is that the cleaned branch changes 365243 to missing before aggregation; the raw branch keeps it numeric.

This follows a known Home Credit data convention used in participant feature engineering ([example](https://github.com/pklauke/Kaggle-HomeCreditDefaultRisk/blob/master/feature_engineering.py)). The source CSV files are not modified.

| Application field | Training rows with placeholder | Test rows with placeholder |
|---|---:|---:|
| DAYS_EMPLOYED | 55,374 | 9,274 |

Previous-application placeholders found: **1,506,087** across **997,144** rows out of 1,670,214.

| Previous-application field | Placeholder values found | Cleaned aggregate MAX still equal to 365243 |
|---|---:|---:|
| DAYS_FIRST_DRAWING | 934,444 | 0 |
| DAYS_FIRST_DUE | 40,645 | 0 |
| DAYS_LAST_DUE_1ST_VERSION | 93,864 | 0 |
| DAYS_LAST_DUE | 211,221 | 0 |
| DAYS_TERMINATION | 225,913 | 0 |

### Unmatched bureau-balance rows

Bureau-balance rows are joined through the bureau loan ID. Rows without a matching bureau record cannot be assigned to an applicant, so they are excluded from applicant features.

- Rows read: 27,299,925
- Unmatched rows excluded: 3,120,184 (11.43%)

The count is reported here; no separate quarantine file is written.

## Raw versus cleaned model

Both branches use the same 5 stratified folds (split seed 42). Category levels are learned from each training fold only. Early stopping uses each fold's validation data, and that same fold produces the reported score, so scores are slightly optimistic. The raw and cleaned runs are affected equally. The ± values below are standard deviations across folds.

| Metric | Raw (mean ± standard deviation) | Cleaned (mean ± standard deviation) |
|---|---:|---:|
| ROC-AUC | 0.782369 ± 0.004705 | 0.783033 ± 0.003965 |
| Average precision | 0.275911 ± 0.008137 | 0.276634 ± 0.006765 |

| Fold | Raw ROC-AUC | Cleaned ROC-AUC | Difference |
|---:|---:|---:|---:|
| 1 | 0.779823 | 0.780717 | +0.000894 |
| 2 | 0.789092 | 0.788679 | -0.000413 |
| 3 | 0.779734 | 0.779984 | +0.000250 |
| 4 | 0.785390 | 0.785747 | +0.000357 |
| 5 | 0.777807 | 0.780038 | +0.002231 |

Cleaned scored higher on 4 of 5 folds. This is a descriptive comparison from one seed and five folds, not evidence of a reliable performance gain. The main result is the corrected treatment of known placeholder values.

## Model setup

LightGBM binary classifier; 5 stratified folds with split seed 42; fold model seeds 43–47; learning rate 0.03; maximum 2,000 trees with 100-round early stopping. The final cleaned model uses 681 trees, the median best iteration across cleaned folds. LightGBM tree parameters otherwise use library defaults; all CPU threads are enabled and verbosity is suppressed.

- Python used for this run: 3.14.7
- Runtime: 26.5 minutes
- Peak process working set: 2.8 GiB
- Training applicants: 307,511; positive target rate: 8.07%
- Test applicants scored: 48,744

## Final model feature importance

Gain importance shows how much each feature reduced loss in this fitted model. It describes this model; it does not show that a feature causes default.

| Feature | Gain |
|---|---:|
| EXT_SOURCE_3 | 123,352.4 |
| EXT_SOURCE_2 | 107,194.0 |
| ORGANIZATION_TYPE | 41,972.6 |
| EXT_SOURCE_1 | 36,186.6 |
| DAYS_EMPLOYED | 14,085.4 |
| AMT_ANNUITY | 11,591.1 |
| INST_AMT_PAYMENT__MIN | 10,813.6 |
| AMT_CREDIT | 10,742.9 |
| AMT_GOODS_PRICE | 10,213.1 |
| INST_AMT_PAYMENT__SUM | 9,565.8 |

## Data terms and limitations

The shareable project copy excludes the Kaggle CSVs. Obtain the source files from Kaggle and follow the competition rules. Cleaning counts and excluded-row counts apply to the source files used for this run. Metrics use one seed and five folds, with early stopping on each validation fold.

## Reproduce the run

1. Download the Home Credit Default Risk CSV files from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place them in data/raw/.
2. Install packages with: pip install -r requirements.txt
3. Run: python scripts/run_project.py

The script rebuilds temporary aggregates on each run and removes them afterward. The model is saved in artifacts/, predictions in data/processed/, and this report in reports/project_report.md.

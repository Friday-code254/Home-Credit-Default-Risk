# Home Credit Default Risk

This project looks at how a known missing-value code in the Home Credit data affects loan-default features and model results. The main work is data cleaning; the model comparison checks what happened after the correction.

## Cleaning

The value 365243 appears in DAYS_EMPLOYED and five previous-application day fields as a placeholder, not a real number of elapsed days. The cleaned branch converts it to missing before aggregation. Both branches keep the same sentinel indicators and applicant-level counts, use the same feature tables, and exclude bureau-balance rows that cannot be linked to a bureau loan. The only difference between the branches is the treatment of 365243.

On the supplied data, 55,374 training and 9,274 test applicants had the DAYS_EMPLOYED placeholder. The previous-application tables contained 1,506,087 placeholder values across 997,144 rows. Of 27,299,925 bureau-balance records, 3,120,184 (11.43%) had no matching bureau loan and were excluded. Field-level counts are in the report.

## Model results

Both branches use the same five stratified folds, split seed 42, and LightGBM setup. The report shows ROC-AUC for each fold and the fold mean with standard deviation.

| ROC-AUC, mean plus/minus standard deviation across folds | Score |
| --- | ---: |
| Raw values | 0.782369 +/- 0.004705 |
| Cleaned values | 0.783033 +/- 0.003965 |

The cleaned run scored higher on 4 of 5 folds. This is a descriptive result from one seed and does not establish a reliable model-performance gain. The reported fold means and standard deviations describe the folds; they are not a significance test.

Category levels are learned from each training fold. Early stopping uses each fold's validation data, and that same fold produces the reported score, so the scores are slightly optimistic. The raw and cleaned runs are affected equally.

## Model and run details

LightGBM binary classifier; five stratified folds with split seed 42 and fold model seeds 43-47; learning rate 0.03; maximum 2,000 trees with 100-round early stopping. The final model uses 681 trees, the median best iteration across cleaned folds. Other tree settings use LightGBM defaults; the run uses all CPU threads and suppresses model verbosity.

The project supports Python 3.10 or later. The reported run used Python 3.14.7. It took about 26.5 minutes and reached a peak process working set of about 2.8 GiB on a machine with 8 GB of RAM. Runtime and memory use will vary by computer.

## Data and limitations

The shareable project copy does not include the Kaggle CSV files. Download them from the Home Credit Default Risk competition page and place them in data/raw/. The data is governed by Kaggle's competition rules; follow those terms when using or sharing it.

The metrics use one seed and five folds. Cleaning counts and excluded-row counts apply to this copy of the dataset and may differ for another version.

## Run it

From this project folder, install the dependencies:

    python -m pip install -r requirements.txt

Then run the full pipeline:

    python scripts/run_project.py

The script rebuilds temporary history aggregates, compares raw and cleaned data, trains a final cleaned-data model, and writes the report, saved model, and test predictions.

## Project files

- scripts/run_project.py: end-to-end run
- src/home_credit/features.py: sentinel treatment, history aggregation, and joins
- src/home_credit/modeling.py: fold validation and final model fitting
- reports/project_report.md: cleaning counts, fold results, and feature importance
- artifacts/home_credit_default_model.joblib: final fitted model from this run
- data/processed/submission.csv: predictions for the Kaggle test applicants
- requirements.txt: Python dependencies

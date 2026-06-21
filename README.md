# Credit Risk Default Prediction

Predicts the probability that a consumer-finance applicant will default within
12 months, using application, demographic, employment and credit-bureau
information available at the time of application.

## Approach

- **Feature engineering:** debt burden, affordability and declaration-mismatch
  ratios (`unsecured_to_income`, `rent_burden_ratio`, `limit_to_income`,
  `declared_amt_diff`), zero/missing flags, a company-size proxy, channel
  flags, and a PSI-safe grouping of the high-cardinality JIS address field.
- **Target encoding:** smoothed, out-of-fold encoding of categorical fields
  (`Major Media Code`, `Internet Details`, `Reception Type Category`,
  `Industry Type`, `Company Size Category`, grouped JIS address), with a
  Population Stability Index (PSI) check between train and test used to drop
  any encoding unstable enough to risk leaking train-only signal.
- **Model:** LightGBM, trained on a time-based 80/20 split so validation
  reflects performance on applications arriving after the training window.
- **Validation:** ROC-AUC, SHAP explainability, a calibration curve, and a
  decile lift table to check the predicted probabilities are usable directly
  for risk-based decisioning, not just for ranking.

## Results

| Metric | Value |
|---|---|
| Validation AUC | **0.6485** |
| Top-decile lift | **~2x** the portfolio average default rate |

Top SHAP drivers:

| Feature | Mean \|SHAP\| |
|---|---|
| unsecured_to_income | 0.252 |
| age | 0.198 |
| rent_burden_ratio | 0.083 |
| limit_to_income | 0.080 |
| Internet Details | 0.073 |

## Business insights

1. Higher unsecured debt burden relative to income increases default risk.
2. Younger applicants exhibit higher default probability.
3. A large mismatch between declared and actual liabilities is a strong
   risk/fraud indicator.
4. Internet acquisition channels show different risk behaviour than offline
   channels.
5. Rent burden and desired credit limit relative to income are significant
   affordability indicators.

## Repository contents

- `Credit_Risk_LightGBM_Final.ipynb` — the primary, single-model pipeline:
  data loading → feature engineering → target encoding → LightGBM → SHAP →
  calibration → decile lift → PSI check → `submission.csv`.
- `Credit_Risk_Ensemble_Experiment.ipynb` — a secondary notebook exploring a
  CatBoost + XGBoost stacking ensemble on top of the same features, kept
  separate so the primary notebook stays easy to read end to end.

## CV / portfolio blurb

**Consumer Credit Default Prediction | LightGBM, SHAP, Feature Engineering**

- Developed an end-to-end credit risk prediction model on 80,000 consumer finance applications.
- Engineered debt burden, affordability, and declaration-mismatch features
  from application and bureau data.
- Applied target encoding and trained a LightGBM model achieving **0.6485
  ROC-AUC** on out-of-time validation.
- Used SHAP explainability, calibration analysis, and decile lift
  segmentation to identify key drivers of customer default risk.
- Generated actionable lending insights supporting risk-based credit
  decisioning.

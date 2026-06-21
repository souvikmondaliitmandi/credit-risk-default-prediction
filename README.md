# Credit Risk Default Prediction

## Problem Statement
Predict the probability of default within 12 months for consumer finance applicants using application and credit-related metadata.

## Dataset
- Train set: 80,000 rows
- Test set: 20,000 rows
- Target: `Default 12 Flag`

## Approach
- Parsed application date and time
- Engineered age from date of birth
- Created financial risk features:
  - rent burden ratio
  - unsecured loans to income
  - desired limit to income
  - declared vs actual loan mismatch
- Applied capped and log-transformed ratio features
- Used fold-wise target encoding for categorical variables
- Trained a LightGBM model with time-based validation

## Model
- LightGBM binary classifier

## Validation Performance
- Validation AUC: 0.6485

## Key Drivers
- `log1p_unsecured_to_income`
- `age`
- `log1p_rent_burden_ratio`
- `log1p_limit_to_income`
- `declared_amt_diff`
- Channel and industry target-encoded features

## Explainability
- SHAP summary analysis
- Calibration curve
- Decile lift analysis

## Business Insight
Higher unsecured debt burden, younger age, higher rent burden, and declared mismatches were associated with higher default risk.

## Files Included
- `Credit_Risk_Model.ipynb`
- `shap_summary.png`
- `calibration_curve.png`
- `decile_lift_val.csv`
- `requirements.txt`

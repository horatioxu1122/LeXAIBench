# Additivity of native contributions

Gap = |sum of contributions + bias - logit(predicted probability)|, in log odds.

| model | source | explanations | max gap | mean gap |
|---|---|---:|---:|---:|
| ebm | ebm_eval_terms_pair_split | 2640 | 2.04e-14 | 4.98e-16 |
| logistic_regression | lr_coef_x_standardized | 2640 | 1.65e-14 | 8.48e-16 |
| xgboost | xgb_pred_contribs | 2640 | 1.04e-06 | 1.86e-07 |

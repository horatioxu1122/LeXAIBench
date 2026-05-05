# LegalBench-XAI Limitations Memo

## Current Limitations

The current outputs come from a fully real benchmark run on local real dataset files. Model backends in use: `ebm_classifier` x 48, `sklearn_logistic_regression` x 48, `xgboost_classifier` x 48. Explainer backends in use: `leave_one_out_feature_ablation` x 24, `lime_library` x 24, `search_anchor_rule` x 24, `search_counterfactual_dice_style` x 24, `search_counterfactual_wachter_style` x 24, `shap_library` x 24.
The current run already uses local real credit data rather than fixture rows. That is a meaningful empirical step forward, and 4 real datasets are now a credible benchmark base. The main remaining burden is no longer only dataset count, but sensitivity analysis, uncertainty calibration, and larger denied-instance budgets.
The stability metric is now backed by perturbation-based evidence in this run. The remaining limitation is not placeholder logic but evaluation scale and cross-dataset validation.
The uncertainty estimates reported in the current bundle are hierarchical-bootstrap intervals over the benchmark structure. They are stronger than flat group resampling, but they are still not full hierarchical uncertainty estimates over applicants, datasets, models, and random variation jointly.
The actionability and legality layers are normative approximations derived from the project taxonomy. They are useful for benchmarking, but they do not substitute for legal validation or domain-specific compliance review.
The benchmark now runs fully real models and explainers on local real datasets with perturbation-based stability evidence. The main remaining gap is breadth: the current evidence spans 4 real datasets, but still uses a limited denied-instance budget and a narrow domain slice.

## Environment Gaps

Missing empirical dependencies in this environment: none.

## What Must Change Before Submission

- Increase the denied-instance budget so the current real-data findings are supported by materially larger adverse subsets.
- A fifth dataset would strengthen breadth, but the higher-priority next step is sensitivity analysis plus larger denied-instance budgets on the existing four-dataset benchmark.

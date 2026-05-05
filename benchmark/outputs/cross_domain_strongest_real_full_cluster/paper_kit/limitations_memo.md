# LegalBench-XAI Limitations Memo

## Current Limitations

The current outputs come from a fully real benchmark run on local real dataset files. Model backends in use: `ebm_classifier` x 30, `sklearn_logistic_regression` x 30, `xgboost_classifier` x 30. Explainer backends in use: `lime_library` x 18, `search_anchor_rule` x 18, `search_counterfactual_dice_style` x 18, `search_counterfactual_wachter_style` x 18, `shap_library` x 18.
The current run already uses local real credit data rather than fixture rows. That is a meaningful empirical step forward, but 3 real datasets are still not enough to anchor top-tier claims about robustness across credit settings.
The stability metric is now backed by perturbation-based evidence in this run. The remaining limitation is not placeholder logic but evaluation scale and cross-dataset validation.
The uncertainty estimates reported in the current bundle are hierarchical-bootstrap intervals over the benchmark structure. They are stronger than flat group resampling, but they are still not full hierarchical uncertainty estimates over applicants, datasets, models, and random variation jointly.
The actionability and legality layers are normative approximations derived from the project taxonomy. They are useful for benchmarking, but they do not substitute for legal validation or domain-specific compliance review.
The benchmark now runs fully real models and explainers on local real datasets with perturbation-based stability evidence. The main remaining gap is breadth: the current evidence spans 3 real datasets, but still uses a limited denied-instance budget and a narrow domain slice.

## Environment Gaps

Missing empirical dependencies in this environment: none.

## What Must Change Before Submission

- Increase the denied-instance budget so the current real-data findings are supported by materially larger adverse subsets.
- Add at least one more distinct real credit dataset before treating the current tradeoff pattern as robust across credit settings.

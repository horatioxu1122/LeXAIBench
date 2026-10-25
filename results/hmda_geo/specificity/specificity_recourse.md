# Notice specificity (E3) and valid recourse (E6)

## E3: mean pairwise Jaccard overlap of the first 4 reason features across applicants

1.0 means every applicant in the configuration receives the same reasons. `regb_reference` is the comment 9(b)(2)-5 marginal applicant method on the same applicants, and `regb_average` its average applicant method.

| bundle | method | overlap | configurations |
|---|---|---:|---:|
| hmda_geo | anchors | 0.298 [0.272, 0.320] | 6 |
| hmda_geo | lime | 0.845 [0.783, 0.906] | 6 |
| hmda_geo | regb_average | 0.391 [0.343, 0.439] | 6 |
| hmda_geo | regb_reference | 0.311 [0.279, 0.340] | 6 |
| hmda_geo | shap | 0.381 [0.337, 0.425] | 6 |

| bundle | track | method | overlap | configurations |
|---|---|---|---:|---:|
| hmda_geo | audit | anchors | 0.312 [0.286, 0.341] | 3 |
| hmda_geo | audit | lime | 0.842 [0.801, 0.907] | 3 |
| hmda_geo | audit | regb_average | 0.347 [0.299, 0.406] | 3 |
| hmda_geo | audit | regb_reference | 0.278 [0.251, 0.314] | 3 |
| hmda_geo | audit | shap | 0.344 [0.288, 0.389] | 3 |
| hmda_geo | primary | anchors | 0.284 [0.246, 0.319] | 3 |
| hmda_geo | primary | lime | 0.847 [0.729, 0.967] | 3 |
| hmda_geo | primary | regb_average | 0.435 [0.397, 0.489] | 3 |
| hmda_geo | primary | regb_reference | 0.344 [0.339, 0.354] | 3 |
| hmda_geo | primary | shap | 0.417 [0.367, 0.462] | 3 |

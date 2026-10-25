# Notice specificity (E3) and valid recourse (E6)

## E3: mean pairwise Jaccard overlap of the first 4 reason features across applicants

1.0 means every applicant in the configuration receives the same reasons. `regb_reference` is the comment 9(b)(2)-5 marginal applicant method on the same applicants, and `regb_average` its average applicant method.

| bundle | method | overlap | configurations |
|---|---|---:|---:|
| credit | anchors | 0.343 [0.303, 0.386] | 24 |
| credit | dice | 0.438 [0.414, 0.463] | 24 |
| credit | lime | 0.849 [0.812, 0.882] | 24 |
| credit | regb_average | 0.454 [0.422, 0.485] | 24 |
| credit | regb_reference | 0.362 [0.335, 0.390] | 24 |
| credit | shap | 0.430 [0.391, 0.469] | 24 |
| credit | wachter | 0.395 [0.359, 0.436] | 24 |
| lime_default | lime | 0.475 [0.427, 0.522] | 24 |
| lime_default | regb_average | 0.454 [0.422, 0.485] | 24 |
| lime_default | regb_reference | 0.362 [0.335, 0.390] | 24 |
| lime_local | lime | 0.722 [0.652, 0.792] | 24 |
| lime_local | regb_average | 0.454 [0.422, 0.485] | 24 |
| lime_local | regb_reference | 0.362 [0.335, 0.390] | 24 |
| noncredit | anchors | 0.345 [0.311, 0.387] | 18 |
| noncredit | dice | 0.385 [0.369, 0.401] | 6 |
| noncredit | lime | 0.663 [0.574, 0.746] | 18 |
| noncredit | regb_average | 0.395 [0.375, 0.417] | 18 |
| noncredit | regb_reference | 0.369 [0.346, 0.394] | 18 |
| noncredit | shap | 0.400 [0.381, 0.420] | 18 |
| noncredit | wachter | 0.644 [0.455, 0.834] | 6 |
| nullbaselines | global_ranking | 1.000 [1.000, 1.000] | 24 |
| nullbaselines | random_reasons | 0.117 [0.110, 0.123] | 24 |
| nullbaselines | regb_average | 0.454 [0.422, 0.485] | 24 |
| nullbaselines | regb_reference | 0.362 [0.335, 0.390] | 24 |

| bundle | track | method | overlap | configurations |
|---|---|---|---:|---:|
| credit | audit | anchors | 0.363 [0.291, 0.434] | 12 |
| credit | audit | dice | 0.443 [0.409, 0.476] | 12 |
| credit | audit | lime | 0.857 [0.816, 0.892] | 12 |
| credit | audit | regb_average | 0.459 [0.422, 0.499] | 12 |
| credit | audit | regb_reference | 0.355 [0.321, 0.392] | 12 |
| credit | audit | shap | 0.430 [0.387, 0.475] | 12 |
| credit | audit | wachter | 0.387 [0.349, 0.427] | 12 |
| credit | primary | anchors | 0.323 [0.284, 0.363] | 12 |
| credit | primary | dice | 0.434 [0.398, 0.468] | 12 |
| credit | primary | lime | 0.840 [0.780, 0.891] | 12 |
| credit | primary | regb_average | 0.449 [0.407, 0.492] | 12 |
| credit | primary | regb_reference | 0.368 [0.333, 0.408] | 12 |
| credit | primary | shap | 0.430 [0.374, 0.491] | 12 |
| credit | primary | wachter | 0.404 [0.345, 0.470] | 12 |
| lime_default | audit | lime | 0.481 [0.416, 0.544] | 12 |
| lime_default | audit | regb_average | 0.459 [0.422, 0.499] | 12 |
| lime_default | audit | regb_reference | 0.355 [0.321, 0.392] | 12 |
| lime_default | primary | lime | 0.470 [0.408, 0.533] | 12 |
| lime_default | primary | regb_average | 0.449 [0.407, 0.492] | 12 |
| lime_default | primary | regb_reference | 0.368 [0.333, 0.408] | 12 |
| lime_local | audit | lime | 0.728 [0.622, 0.827] | 12 |
| lime_local | audit | regb_average | 0.459 [0.422, 0.499] | 12 |
| lime_local | audit | regb_reference | 0.355 [0.321, 0.392] | 12 |
| lime_local | primary | lime | 0.716 [0.614, 0.820] | 12 |
| lime_local | primary | regb_average | 0.449 [0.407, 0.492] | 12 |
| lime_local | primary | regb_reference | 0.368 [0.333, 0.408] | 12 |
| noncredit | audit | anchors | 0.319 [0.298, 0.343] | 9 |
| noncredit | audit | dice | 0.393 [0.357, 0.412] | 3 |
| noncredit | audit | lime | 0.651 [0.515, 0.778] | 9 |
| noncredit | audit | regb_average | 0.383 [0.359, 0.410] | 9 |
| noncredit | audit | regb_reference | 0.363 [0.329, 0.397] | 9 |
| noncredit | audit | shap | 0.390 [0.371, 0.411] | 9 |
| noncredit | audit | wachter | 0.642 [0.411, 1.000] | 3 |
| noncredit | primary | anchors | 0.371 [0.311, 0.439] | 9 |
| noncredit | primary | dice | 0.377 [0.369, 0.390] | 3 |
| noncredit | primary | lime | 0.675 [0.581, 0.765] | 9 |
| noncredit | primary | regb_average | 0.408 [0.380, 0.443] | 9 |
| noncredit | primary | regb_reference | 0.376 [0.349, 0.404] | 9 |
| noncredit | primary | shap | 0.410 [0.379, 0.442] | 9 |
| noncredit | primary | wachter | 0.645 [0.448, 1.000] | 3 |
| nullbaselines | audit | global_ranking | 1.000 [1.000, 1.000] | 12 |
| nullbaselines | audit | random_reasons | 0.111 [0.099, 0.120] | 12 |
| nullbaselines | audit | regb_average | 0.459 [0.422, 0.499] | 12 |
| nullbaselines | audit | regb_reference | 0.355 [0.321, 0.392] | 12 |
| nullbaselines | primary | global_ranking | 1.000 [1.000, 1.000] | 12 |
| nullbaselines | primary | random_reasons | 0.123 [0.117, 0.130] | 12 |
| nullbaselines | primary | regb_average | 0.449 [0.407, 0.492] | 12 |
| nullbaselines | primary | regb_reference | 0.368 [0.333, 0.408] | 12 |

## E6: share of denied applicants with at least one valid counterfactual

| bundle | method | share | applicants |
|---|---|---:|---:|
| credit | dice | 0.976 [0.964, 0.988] | 576 |
| credit | wachter | 0.882 [0.858, 0.906] | 576 |
| noncredit | dice | 0.444 [0.419, 0.470] | 432 |
| noncredit | wachter | 0.299 [0.282, 0.315] | 432 |

| bundle | dataset | method | share | applicants |
|---|---|---|---:|---:|
| credit | default_of_credit_card_clients | dice | 1.000 [1.000, 1.000] | 144 |
| credit | default_of_credit_card_clients | wachter | 1.000 [1.000, 1.000] | 144 |
| credit | give_me_some_credit | dice | 1.000 [1.000, 1.000] | 144 |
| credit | give_me_some_credit | wachter | 0.993 [0.979, 1.000] | 144 |
| credit | heloc | dice | 0.903 [0.854, 0.951] | 144 |
| credit | heloc | wachter | 0.771 [0.701, 0.840] | 144 |
| credit | south_german_credit | dice | 1.000 [1.000, 1.000] | 144 |
| credit | south_german_credit | wachter | 0.764 [0.688, 0.826] | 144 |
| noncredit | fda_inspections | dice | 0.000 [0.000, 0.000] | 144 |
| noncredit | fda_inspections | wachter | 0.000 [0.000, 0.000] | 144 |
| noncredit | hmda | dice | 1.000 [1.000, 1.000] | 144 |
| noncredit | hmda | wachter | 0.896 [0.847, 0.944] | 144 |
| noncredit | ny_courts_pretrial | dice | 0.333 [0.257, 0.417] | 144 |
| noncredit | ny_courts_pretrial | wachter | 0.000 [0.000, 0.000] | 144 |

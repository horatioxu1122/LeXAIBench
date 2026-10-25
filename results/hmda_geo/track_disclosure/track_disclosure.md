# Disclosure of protected characteristics and proxies (track populations)

Excluded = features the primary track drops (protected characteristics). P = taxonomy label for a possible proxy. Shares of denied applicants' notices (first four stated reasons); the two tracks explain different applicants, so the change is a population comparison.

| bundle | dataset | method | audit: names excluded | audit: first reason excluded | names P (audit) | names P (primary) | change |
|---|---|---|---:|---:|---:|---:|---:|
| hmda_geo | hmda | anchors | 6% [1, 11] | 0% [0, 0] | 47% [36, 58] | 51% [40, 62] | 4% [-11, 21] |
| hmda_geo | hmda | lime | 42% [36, 49] | 1% [0, 4] | 60% [54, 65] | 78% [72, 85] | 18% [10, 26] |
| hmda_geo | hmda | regb_m1 | 51% [40, 62] | 0% [0, 0] | 57% [46, 68] | 58% [47, 69] | 1% [-15, 17] |
| hmda_geo | hmda | regb_m2 | 43% [33, 54] | 0% [0, 0] | 56% [44, 67] | 61% [50, 71] | 6% [-10, 21] |
| hmda_geo | hmda | shap | 38% [28, 47] | 0% [0, 0] | 51% [40, 62] | 61% [50, 72] | 10% [-7, 25] |

## Baselines for the change between tracks (commentary methods)

Audit = the audit track's notices. Refill = the audit model's own ranking with the excluded features left out, so its next-ranked factors fill the freed places (no refit; same applicants). Placebo = the same with as many features labeled neither protected nor possible proxy left out instead. Chance = four reasons drawn at random from the track's reason units. Refill and placebo differences with the audit track are paired.

| bundle | dataset | method | metric | audit | refill | placebo | primary | primary - refill | refill - audit | chance (audit) | chance (primary) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hmda_geo | hmda | regb_m1 | names_P | 57% [46, 68] | 64% [53, 75] | 74% [64, 83] | 58% [47, 69] | -6% [-22, 10] | 7% [1, 12] | 67% | 82% |
| hmda_geo | hmda | regb_m1 | names_census_tract | 25% [15, 35] | 26% [17, 36] | 34% [24, 44] | 33% [22, 44] | 7% [-8, 21] | 1% [0, 4] | 22% | 31% |
| hmda_geo | hmda | regb_m1 | names_county_code | 8% [3, 15] | 18% [10, 26] | 27% [18, 36] | 18% [10, 26] | 0% [-12, 12] | 10% [3, 18] | 22% | 31% |
| hmda_geo | hmda | regb_m1 | names_income | 26% [18, 36] | 26% [18, 36] | 31% [21, 40] | 24% [15, 33] | -3% [-15, 10] | 0% [0, 0] | 22% | 31% |
| hmda_geo | hmda | regb_m1 | names_state_code | 14% [7, 22] | 18% [10, 26] | 24% [16, 32] | 12% [6, 19] | -6% [-17, 6] | 4% [0, 10] | 22% | 31% |
| hmda_geo | hmda | regb_m2 | names_P | 56% [44, 67] | 65% [54, 76] | 79% [69, 87] | 61% [50, 71] | -4% [-21, 11] | 10% [4, 17] | 67% | 82% |
| hmda_geo | hmda | regb_m2 | names_census_tract | 26% [18, 36] | 35% [25, 44] | 42% [32, 52] | 31% [21, 42] | -4% [-19, 10] | 8% [3, 15] | 22% | 31% |
| hmda_geo | hmda | regb_m2 | names_county_code | 4% [0, 10] | 11% [4, 18] | 23% [15, 30] | 10% [4, 17] | -1% [-11, 8] | 7% [1, 12] | 22% | 31% |
| hmda_geo | hmda | regb_m2 | names_income | 28% [19, 38] | 31% [21, 40] | 37% [28, 46] | 31% [22, 40] | 0% [-12, 12] | 3% [0, 7] | 22% | 31% |
| hmda_geo | hmda | regb_m2 | names_state_code | 8% [3, 15] | 11% [4, 18] | 20% [13, 27] | 10% [4, 17] | -1% [-11, 7] | 3% [0, 7] | 22% | 31% |

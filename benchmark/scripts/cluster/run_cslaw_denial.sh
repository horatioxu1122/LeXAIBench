#!/usr/bin/env bash
# Denial-direction correction (benchmark/cslaw/denial_reasons.py) of the v2 bundles,
# then every CS&Law analysis on the corrected bundles. Offline: no model fitting.
#
# Usage:
#   bash benchmark/scripts/cluster/run_cslaw_denial.sh [REPO_ROOT] [--tag TAG] [--force]
#
# Inputs (must exist): benchmark/outputs/cslaw_v2_credit_<TAG>, cslaw_v2_credit_nullbaselines_<TAG>,
#   cslaw_v2_credit_shared_<TAG>, cslaw_v2_noncredit_<TAG>; optional cslaw_v2_credit_lime_local_<TAG>.
# Output: benchmark/outputs/cslaw_denial_<TAG>/{credit,nullbaselines,shared,noncredit,lime_local,analysis}
set -euo pipefail

REPO_ROOT="${HOME}/explainableAI-CS"
if [[ $# -gt 0 && "$1" != --* ]]; then
  REPO_ROOT="$1"
  shift
fi
TAG="r1"
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h | --help) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

cd "${REPO_ROOT}"
PY="${REPO_ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || { echo "no ${PY}; run 'uv sync --extra benchmark-empirical' first" >&2; exit 1; }
export PYTHONPATH="${REPO_ROOT}/benchmark/src"

OUT="benchmark/outputs"
DEST="${OUT}/cslaw_denial_${TAG}"
if [[ -e "${DEST}" && ${FORCE} -ne 1 ]]; then
  echo "${DEST} exists; pass --force to overwrite" >&2
  exit 1
fi
mkdir -p "${DEST}/analysis"
FORCE_ARG=()
[[ ${FORCE} -eq 1 ]] && FORCE_ARG=(--force)

correct() {  # name bundle config
  local name="$1" bundle="$2" config="$3"
  if [[ ! -f "${bundle}/results.json" ]]; then
    echo "[denial] skip ${name}: ${bundle}/results.json not found"
    return 0
  fi
  local config_args=()
  [[ -n "${config}" ]] && config_args=(--config "${config}")
  echo "[denial] ${name}: ${bundle} -> ${DEST}/${name}"
  "${PY}" benchmark/cslaw/denial_reasons.py --bundle "${bundle}" "${config_args[@]}" \
    --out-dir "${DEST}/${name}" "${FORCE_ARG[@]}"
}

correct credit "${OUT}/cslaw_v2_credit_${TAG}" benchmark/configs/experiment_cslaw_v2_credit.json
correct nullbaselines "${OUT}/cslaw_v2_credit_nullbaselines_${TAG}" benchmark/configs/experiment_cslaw_v2_credit_nullbaselines.json
correct shared "${OUT}/cslaw_v2_credit_shared_${TAG}" benchmark/configs/experiment_cslaw_v2_credit_shared.json
correct noncredit "${OUT}/cslaw_v2_noncredit_${TAG}" ""
correct lime_local "${OUT}/cslaw_v2_credit_lime_local_${TAG}" benchmark/configs/experiment_cslaw_v2_credit_lime_local.json

A="${DEST}/analysis"
run() {  # label, then command
  local label="$1"; shift
  echo "[denial] analysis: ${label}"
  "$@" > "${A}/log_${label}.txt" 2>&1 || { echo "[denial] FAILED: ${label} (see ${A}/log_${label}.txt)" >&2; exit 1; }
}

TABLE_ARGS=(--bundle "credit=${DEST}/credit" --bundle "nullbaselines=${DEST}/nullbaselines")
[[ -f "${DEST}/noncredit/results.json" ]] && TABLE_ARGS+=(--bundle "noncredit=${DEST}/noncredit")
run paper_tables "${PY}" benchmark/cslaw/paper_tables.py "${TABLE_ARGS[@]}" --out-dir "${A}/paper_tables"
run regb_credit "${PY}" benchmark/cslaw/regb_reference.py --bundle "${DEST}/credit" --out-dir "${A}/regb_credit"
[[ -f "${DEST}/noncredit/results.json" ]] && run regb_noncredit "${PY}" benchmark/cslaw/regb_reference.py --bundle "${DEST}/noncredit" --out-dir "${A}/regb_noncredit"
[[ -f "${DEST}/lime_local/results.json" ]] && run regb_lime_local "${PY}" benchmark/cslaw/regb_reference.py --bundle "${DEST}/lime_local" --out-dir "${A}/regb_lime_local"
run multiplicity "${PY}" benchmark/cslaw/multiplicity.py --bundle "${DEST}/shared" --out-dir "${A}/multiplicity"
run proxy "${PY}" benchmark/cslaw/proxy_substitution.py --bundle "${DEST}/shared" --out-dir "${A}/proxy_substitution"
DISC_ARGS=(--bundle "${DEST}/credit")
[[ -f "${DEST}/noncredit/results.json" ]] && DISC_ARGS+=(--bundle "${DEST}/noncredit")
run track_disclosure "${PY}" benchmark/cslaw/track_disclosure.py "${DISC_ARGS[@]}" --out-dir "${A}/track_disclosure"
run common_faithfulness "${PY}" benchmark/cslaw/common_faithfulness.py --bundle "${DEST}/credit" --bundle "${DEST}/nullbaselines" --out-dir "${A}/common_faithfulness"
run notices "${PY}" benchmark/cslaw/notices.py --bundle "${DEST}/credit" --n 48 --seed 7 --out-dir "${A}/notices" "${FORCE_ARG[@]}"
run regb_nullbaselines "${PY}" benchmark/cslaw/regb_reference.py --bundle "${DEST}/nullbaselines" --out-dir "${A}/regb_nullbaselines"
[[ -f "${DEST}/lime_local/results.json" ]] && run common_faithfulness_lime_local "${PY}" benchmark/cslaw/common_faithfulness.py --bundle "${DEST}/lime_local" --out-dir "${A}/common_faithfulness_lime_local"

# Specificity across applicants (E3) and valid recourse (E6).
SPEC_ARGS=(--bundle "${DEST}/credit" --bundle "${DEST}/nullbaselines")
for name in noncredit lime_local; do
  [[ -f "${DEST}/${name}/results.json" ]] && SPEC_ARGS+=(--bundle "${DEST}/${name}")
done
run specificity "${PY}" benchmark/cslaw/notice_specificity.py "${SPEC_ARGS[@]}" --out-dir "${A}/specificity"

# Additivity of the persisted native contributions (reads bundles only).
ADD_ARGS=()
for b in "${OUT}/cslaw_v2_credit_${TAG}" "${OUT}/cslaw_v2_credit_shared_${TAG}" "${OUT}/cslaw_v2_noncredit_${TAG}"; do
  [[ -f "${b}/results.json" ]] && ADD_ARGS+=(--bundle "${b}")
done
run additivity "${PY}" benchmark/cslaw/check_additivity.py "${ADD_ARGS[@]}" --out-dir "${A}/additivity"

# Number of reasons k (offline).
for k in 2 3 5; do
  run "regb_k${k}" "${PY}" benchmark/cslaw/regb_reference.py --bundle "${DEST}/credit" --k "${k}" --n-boot 500 --out-dir "${A}/sweep_k/regb_k${k}"
  run "mult_k${k}" "${PY}" benchmark/cslaw/multiplicity.py --bundle "${DEST}/shared" --k "${k}" --n-boot 500 --out-dir "${A}/sweep_k/mult_k${k}"
done

# Marginal applicant reference band (refits the credit models; a few minutes).
run band_sweep "${PY}" benchmark/cslaw/regb_band_sweep.py \
  --config benchmark/configs/experiment_cslaw_v2_credit.json \
  --source-bundle "${OUT}/cslaw_v2_credit_${TAG}" \
  --notice-bundle "${DEST}/credit" \
  --out-dir "${A}/band_sweep"

echo "[denial] done: ${DEST}"

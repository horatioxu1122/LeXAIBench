#!/usr/bin/env bash
# Adds the LIME default-mode robustness run (cslaw_v2_credit_lime_default_<TAG>, from run_cslaw_credit.sh
# --steps limedefault) to an existing cslaw_denial_<TAG> without rerunning the rest of run_cslaw_denial.sh.
# Offline; fits no model. A few minutes.
#
# Usage:
#   bash benchmark/scripts/cluster/run_cslaw_lime_default.sh [REPO_ROOT] [--tag TAG]
#
# Writes cslaw_denial_<TAG>/lime_default/ (direction rule) and, under cslaw_denial_<TAG>/analysis/,
# regb_lime_default/ and common_faithfulness_lime_default/ (new), and reruns specificity/,
# appendix_generated/ and figures/ with the new bundle included (every other table and figure is unchanged).
set -euo pipefail

REPO_ROOT="${HOME}/explainableAI-CS"
if [[ $# -gt 0 && "$1" != --* ]]; then
  REPO_ROOT="$1"
  shift
fi
TAG="r1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    -h | --help) sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

cd "${REPO_ROOT}"
PY="${REPO_ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || { echo "no ${PY}; run 'uv sync --extra benchmark-empirical' first" >&2; exit 1; }
export PYTHONPATH="${REPO_ROOT}/benchmark/src"

OUT="benchmark/outputs"
DEST="${OUT}/cslaw_denial_${TAG}"
A="${DEST}/analysis"
SOURCE="${OUT}/cslaw_v2_credit_lime_default_${TAG}"
[[ -f "${SOURCE}/results.json" ]] || { echo "missing ${SOURCE}/results.json; run run_cslaw_credit.sh --steps limedefault" >&2; exit 1; }
[[ -f "${DEST}/credit/results.json" ]] || { echo "missing ${DEST}; run run_cslaw_denial.sh first" >&2; exit 1; }
for path in "${DEST}/lime_default" "${A}/regb_lime_default" "${A}/common_faithfulness_lime_default"; do
  if [[ -e "${path}" ]]; then echo "${path} exists; remove it to redo this step" >&2; exit 1; fi
done

run() {  # label, then command
  local label="$1"; shift
  echo "[lime_default] ${label}"
  "$@" > "${A}/log_${label}.txt" 2>&1 || { echo "[lime_default] FAILED: ${label} (see ${A}/log_${label}.txt)" >&2; exit 1; }
}

run denial_lime_default "${PY}" benchmark/cslaw/denial_reasons.py --bundle "${SOURCE}" \
  --config benchmark/configs/experiment_cslaw_v2_credit_lime_default.json --out-dir "${DEST}/lime_default"
run regb_lime_default "${PY}" benchmark/cslaw/regb_reference.py --bundle "${DEST}/lime_default" --out-dir "${A}/regb_lime_default"
run common_faithfulness_lime_default "${PY}" benchmark/cslaw/common_faithfulness.py --bundle "${DEST}/lime_default" \
  --out-dir "${A}/common_faithfulness_lime_default"

# Same specificity call as run_cslaw_denial.sh, now with lime_default.
SPEC_ARGS=(--bundle "${DEST}/credit" --bundle "${DEST}/nullbaselines")
for name in noncredit lime_local lime_default; do
  if [[ -f "${DEST}/${name}/results.json" ]]; then SPEC_ARGS+=(--bundle "${DEST}/${name}"); fi
done
run specificity "${PY}" benchmark/cslaw/notice_specificity.py "${SPEC_ARGS[@]}" --out-dir "${A}/specificity"

run appendix_tables "${PY}" benchmark/cslaw/appendix_tables.py --denial-dir "${DEST}" \
  --source-dir "${OUT}" --tag "${TAG}" --out-dir "${A}/appendix_generated"
run paper_figures "${PY}" benchmark/cslaw/paper_figures.py --denial-dir "${DEST}" \
  --source-dir "${OUT}" --tag "${TAG}" --out-dir "${A}/figures"

echo "[lime_default] done: ${DEST}/lime_default, ${A}/regb_lime_default, ${A}/common_faithfulness_lime_default"

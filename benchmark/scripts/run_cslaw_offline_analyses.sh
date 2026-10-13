#!/usr/bin/env bash
# CS&Law offline re-analysis of the EXISTING published bundles. Local-safe: no
# model fitting and no explainer calls. It only rescores stored instance scores,
# rebuilds report artifacts from the rescored results.json, and builds paper tables.
#
# Usage:
#   bash benchmark/scripts/run_cslaw_offline_analyses.sh [REPO_ROOT] [--tag TAG] [--force]
#        [--credit-bundle DIR] [--noncredit-bundle DIR]
#
# Inputs (read only):
#   benchmark/outputs/four_dataset_real_highbudget_heldout_refresh   (credit)
#   benchmark/outputs/cross_domain_strongest_real_full_cluster       (non-credit)
# Outputs (new directory only):
#   benchmark/outputs/cslaw_offline_<TAG>/
#     rescored/<credit|noncredit>/<variant>/   rescored results.json + rebuilt artifacts
#     dsv_sweep/<credit|noncredit>/            decision-sensitivity-visibility weight sweep
#     paper_tables/                            paper_tables.py output
# rebuild-artifacts runs with LEGALBENCH_METRIC_WEIGHT_RANDOM_DRAWS pinned to the
# published bundle's num_draws (default 1000), so the rebuilt random-weight
# artifacts are comparable with the published report.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  REPO_ROOT="$(cd "$1" && pwd)"
  shift
fi

TAG="r1"
CSLAW_FORCE=0
CREDIT_BUNDLE=""
NONCREDIT_BUNDLE=""

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="$2"
      shift 2
      ;;
    --force)
      CSLAW_FORCE=1
      shift
      ;;
    --credit-bundle)
      CREDIT_BUNDLE="$2"
      shift 2
      ;;
    --noncredit-bundle)
      NONCREDIT_BUNDLE="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# shellcheck source=cluster/cslaw_lib.sh
source "${REPO_ROOT}/benchmark/scripts/cluster/cslaw_lib.sh"
export CSLAW_FORCE
cslaw_validate_tag "${TAG}"

cd "${REPO_ROOT}"
cslaw_export_thread_env "${REPO_ROOT}"
cslaw_resolve_python "${REPO_ROOT}"

OUTPUTS="${REPO_ROOT}/benchmark/outputs"
CREDIT_BUNDLE="${CREDIT_BUNDLE:-${OUTPUTS}/${CSLAW_PUBLISHED_CREDIT_BUNDLE_NAME}}"
NONCREDIT_BUNDLE="${NONCREDIT_BUNDLE:-${OUTPUTS}/${CSLAW_PUBLISHED_NONCREDIT_BUNDLE_NAME}}"
CREDIT_CONFIG="${REPO_ROOT}/${CSLAW_PUBLISHED_CREDIT_CONFIG}"
NONCREDIT_CONFIG="${NONCREDIT_BUNDLE}/runtime_config.json"
OFFLINE_DIR="${OUTPUTS}/cslaw_offline_${TAG}"

cslaw_bundle_ready "${CREDIT_BUNDLE}" || cslaw_die "credit bundle not found: ${CREDIT_BUNDLE}/results.json"
cslaw_bundle_ready "${NONCREDIT_BUNDLE}" || cslaw_die "non-credit bundle not found: ${NONCREDIT_BUNDLE}/results.json"
[[ -f "${CREDIT_CONFIG}" ]] || cslaw_die "credit config not found: ${CREDIT_CONFIG}"
[[ -f "${NONCREDIT_CONFIG}" ]] || cslaw_die "non-credit runtime config not found: ${NONCREDIT_CONFIG}"
for script in rescore.py paper_tables.py; do
  [[ -f "${REPO_ROOT}/benchmark/cslaw/${script}" ]] || cslaw_die "missing benchmark/cslaw/${script}"
done

cslaw_guard_target "${OFFLINE_DIR}"
mkdir -p "${OFFLINE_DIR}"

run_cslaw_py() {
  PYTHONPATH="${REPO_ROOT}/benchmark/src${PYTHONPATH:+:${PYTHONPATH}}" "${CSLAW_PY[@]}" "$@"
}

# Number of random metric-weight draws the published bundle was built with
# (num_draws column of its metric_weight_random_summary.csv; reporting.py default
# 1000). rebuild-artifacts reads LEGALBENCH_METRIC_WEIGHT_RANDOM_DRAWS, so it is
# pinned to this value; otherwise the rebuilt random-weight artifacts would not be
# comparable with the published report.md.
published_random_draws() {
  local csv="$1/metric_weight_random_summary.csv"
  local draws=""
  if [[ -s "${csv}" ]]; then
    draws="$(awk -F, '{ sub(/\r$/, "") }
      NR == 1 { for (i = 1; i <= NF; i++) if ($i == "num_draws") col = i; next }
      col && $col ~ /^[0-9]+$/ { print $col; exit }' "${csv}")"
  fi
  echo "${draws:-1000}"
}

rescore_bundle() {
  local label="$1"
  local bundle="$2"
  local config="$3"
  local variant out draws
  draws="$(published_random_draws "${bundle}")"
  if [[ -n "${LEGALBENCH_METRIC_WEIGHT_RANDOM_DRAWS:-}" && "${LEGALBENCH_METRIC_WEIGHT_RANDOM_DRAWS}" != "${draws}" ]]; then
    cslaw_log "WARNING: overriding LEGALBENCH_METRIC_WEIGHT_RANDOM_DRAWS=${LEGALBENCH_METRIC_WEIGHT_RANDOM_DRAWS} with ${draws} (published ${label} bundle)"
  fi
  export LEGALBENCH_METRIC_WEIGHT_RANDOM_DRAWS="${draws}"
  cslaw_log "rebuild-artifacts ${label}: LEGALBENCH_METRIC_WEIGHT_RANDOM_DRAWS=${draws}"
  for variant in "${CSLAW_RESCORE_VARIANTS[@]}"; do
    out="${OFFLINE_DIR}/rescored/${label}/${variant}"
    cslaw_log "rescore ${label} variant=${variant} -> ${out}"
    run_cslaw_py benchmark/cslaw/rescore.py \
      --results "${bundle}/results.json" \
      --variant "${variant}" \
      --out-dir "${out}"
    [[ -f "${out}/results.json" ]] || cslaw_die "rescore.py did not write ${out}/results.json"
    cslaw_log "rebuild-artifacts ${label} variant=${variant}"
    run_cslaw_py -m legalbench_xai rebuild-artifacts \
      --config "${config}" \
      --results-json "${out}/results.json" \
      --output-dir "${out}" >"${out}/rebuild_artifacts.json"
  done
  out="${OFFLINE_DIR}/dsv_sweep/${label}"
  cslaw_log "rescore ${label} DSV sweep ${CSLAW_DSV_SWEEP} -> ${out}"
  run_cslaw_py benchmark/cslaw/rescore.py \
    --results "${bundle}/results.json" \
    --sweep-dsv "${CSLAW_DSV_SWEEP}" \
    --out-dir "${out}"
}

rescore_bundle credit "${CREDIT_BUNDLE}" "${CREDIT_CONFIG}"
rescore_bundle noncredit "${NONCREDIT_BUNDLE}" "${NONCREDIT_CONFIG}"

cslaw_log "paper tables -> ${OFFLINE_DIR}/paper_tables"
# paper_tables.py finds the rescored variants under --rescored-root, but the
# sweeps live in dsv_sweep/, so they are passed explicitly.
run_cslaw_py benchmark/cslaw/paper_tables.py \
  --bundle "credit=${CREDIT_BUNDLE}" \
  --bundle "noncredit=${NONCREDIT_BUNDLE}" \
  --robustness-root "${OUTPUTS}" \
  --rescored-root "${OFFLINE_DIR}/rescored" \
  --sweep-csv "${OFFLINE_DIR}/dsv_sweep/credit/css_dsv_sweep.csv" \
  --sweep-csv "${OFFLINE_DIR}/dsv_sweep/noncredit/css_dsv_sweep.csv" \
  --out-dir "${OFFLINE_DIR}/paper_tables"

echo
echo "CS&Law offline outputs (inputs untouched):"
echo "  ${OFFLINE_DIR}/rescored/credit/<variant>/      (variants: ${CSLAW_RESCORE_VARIANTS[*]})"
echo "  ${OFFLINE_DIR}/rescored/noncredit/<variant>/"
echo "  ${OFFLINE_DIR}/dsv_sweep/credit/"
echo "  ${OFFLINE_DIR}/dsv_sweep/noncredit/"
echo "  ${OFFLINE_DIR}/paper_tables/"

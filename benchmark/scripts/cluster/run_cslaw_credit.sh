#!/usr/bin/env bash
# LeXAIBench v2 consumer credit runs: the main run, the reference notices, the shared
# applicant run, and the two LIME robustness runs. run_cslaw_denial.sh then applies the
# direction rule to these bundles and runs every analysis.
#
# Usage:
#   bash benchmark/scripts/cluster/run_cslaw_credit.sh [REPO_ROOT] [--steps LIST] [--tag TAG] [--force]
#
#   REPO_ROOT   repository checkout (default: $HOME/explainableAI-CS)
#   --steps     comma list, run in this fixed order (default: all except the LIME robustness runs):
#                 selftest       benchmark/cslaw/selftest.py
#                 regression     A/B of experiment_cslaw_regression_check.json (no cslaw block):
#                                baseline commit fb5dadc (git worktree) vs current code, then
#                                compare_bundles.py; both bundles must also carry no
#                                payload["cslaw"], no per-instance cslaw field and no v2-only
#                                key (cslaw_check_bundle ... 0)
#                 taxonomy       validate_taxonomy.py on the v2 taxonomy and registry, compared
#                                with the v1 taxonomy; writes taxonomy_validation.md and the
#                                regenerated manifest into cslaw_analysis_<TAG>/
#                 credit         experiment_cslaw_v2_credit.json (corrections v2, 5 explainers)
#                 nullbaselines  experiment_cslaw_v2_credit_nullbaselines.json (canned and random
#                                reference notices)
#                 shared         experiment_cslaw_v2_credit_shared.json (shared sampling)
#                 limelocal      experiment_cslaw_v2_credit_lime_local.json (LIME only, perturbations
#                                sampled around the instance: corrections v2 + lime_sample_around_instance;
#                                same applicants as credit). Opt-in robustness run, not in the default list
#                 limedefaultpilot  experiment_cslaw_v2_credit_lime_default_pilot.json (the limedefault
#                                settings on South German Credit only; a few minutes, to time the full run)
#                 limedefault    experiment_cslaw_v2_credit_lime_default.json (LIME only, the library's
#                                default discretize_continuous=True: corrections v2 + lime_discretize_continuous;
#                                same applicants as credit). Opt-in robustness run, not in the default list
#   --tag       suffix for every output directory (default: r1)
#   --force     allow writing into existing cslaw_* output directories
#
# Outputs (all new directories under benchmark/outputs):
#   cslaw_regression_baseline_<TAG>/ cslaw_regression_new_<TAG>/ cslaw_regression_compare_<TAG>/
#   cslaw_v2_credit_<TAG>/ cslaw_v2_credit_nullbaselines_<TAG>/ cslaw_v2_credit_shared_<TAG>/
#   cslaw_v2_credit_lime_local_<TAG>/ cslaw_v2_credit_lime_default_pilot_<TAG>/
#   cslaw_v2_credit_lime_default_<TAG>/ cslaw_analysis_<TAG>/ (taxonomy step)
set -euo pipefail

REPO_ROOT="${HOME}/explainableAI-CS"
if [[ $# -gt 0 && "$1" != -* ]]; then
  REPO_ROOT="$1"
  shift
fi

# Execution order of every step; the LIME robustness runs (opt-in) are not in the default list.
ALL_STEPS="selftest,regression,taxonomy,credit,nullbaselines,shared,limelocal,limedefaultpilot,limedefault"
DEFAULT_STEPS="selftest,regression,taxonomy,credit,nullbaselines,shared"
STEPS="${DEFAULT_STEPS}"
TAG="r1"
CSLAW_FORCE=0

usage() {
  sed -n '2,39p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --force)
      CSLAW_FORCE=1
      shift
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

IFS=',' read -r -a REQUESTED_STEPS <<<"${STEPS}"
if [[ ${#REQUESTED_STEPS[@]} -eq 0 ]]; then
  echo "--steps is empty. Allowed: ${ALL_STEPS}" >&2
  exit 1
fi
for step in "${REQUESTED_STEPS[@]}"; do
  case ",${ALL_STEPS}," in
    *",${step},"*) ;;
    *)
      echo "Unknown step '${step}'. Allowed: ${ALL_STEPS}" >&2
      exit 1
      ;;
  esac
done

has_step() {
  [[ ",${STEPS}," == *",$1,"* ]]
}

if command -v module >/dev/null 2>&1; then
  module load python3
else
  echo "[cslaw] 'module' is not available; using the python3 already on PATH" >&2
fi

cd "${REPO_ROOT}"
REPO_ROOT="$(pwd)"

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ "${CSLAW_SKIP_SYNC:-0}" != "1" ]]; then
  uv sync --extra benchmark-empirical
fi

# shellcheck source=cslaw_lib.sh
source "${REPO_ROOT}/benchmark/scripts/cluster/cslaw_lib.sh"
export CSLAW_FORCE
cslaw_validate_tag "${TAG}"
cslaw_export_thread_env "${REPO_ROOT}"
cslaw_resolve_python "${REPO_ROOT}"

OUTPUTS="${REPO_ROOT}/benchmark/outputs"
CONFIGS="${REPO_ROOT}/benchmark/configs"
METADATA="${REPO_ROOT}/benchmark/metadata"
CREDIT_CONFIG="${CONFIGS}/experiment_cslaw_v2_credit.json"
NULLB_CONFIG="${CONFIGS}/experiment_cslaw_v2_credit_nullbaselines.json"
SHARED_CONFIG="${CONFIGS}/experiment_cslaw_v2_credit_shared.json"
LIMELOCAL_CONFIG="${CONFIGS}/experiment_cslaw_v2_credit_lime_local.json"
LIMEDEFAULT_PILOT_CONFIG="${CONFIGS}/experiment_cslaw_v2_credit_lime_default_pilot.json"
LIMEDEFAULT_CONFIG="${CONFIGS}/experiment_cslaw_v2_credit_lime_default.json"
TAXONOMY_V1="${METADATA}/feature_taxonomy.json"
TAXONOMY_V2="${METADATA}/feature_taxonomy_v2.json"
REGISTRY_V2="${METADATA}/dataset_registry_v2.json"
CREDIT_DIR="${OUTPUTS}/cslaw_v2_credit_${TAG}"
NULLB_DIR="${OUTPUTS}/cslaw_v2_credit_nullbaselines_${TAG}"
SHARED_DIR="${OUTPUTS}/cslaw_v2_credit_shared_${TAG}"
LIMELOCAL_DIR="${OUTPUTS}/cslaw_v2_credit_lime_local_${TAG}"
LIMEDEFAULT_PILOT_DIR="${OUTPUTS}/cslaw_v2_credit_lime_default_pilot_${TAG}"
LIMEDEFAULT_DIR="${OUTPUTS}/cslaw_v2_credit_lime_default_${TAG}"
REG_BASE_DIR="${OUTPUTS}/cslaw_regression_baseline_${TAG}"
REG_NEW_DIR="${OUTPUTS}/cslaw_regression_new_${TAG}"
REG_CMP_DIR="${OUTPUTS}/cslaw_regression_compare_${TAG}"
ANALYSIS_DIR="${OUTPUTS}/cslaw_analysis_${TAG}"

# Entries the taxonomy step owns inside the analysis directory.
TAXONOMY_OUTPUTS=(
  "${ANALYSIS_DIR}/taxonomy_validation.md"
  "${ANALYSIS_DIR}/feature_taxonomy_manifest_v2.yaml"
)

# Preflight: refuse existing targets before any long-running step starts.
if has_step regression; then
  cslaw_guard_target "${REG_BASE_DIR}"
  cslaw_guard_target "${REG_NEW_DIR}"
  cslaw_guard_target "${REG_CMP_DIR}"
fi
if has_step taxonomy; then
  cslaw_guard_name "${ANALYSIS_DIR}"
  cslaw_guard_paths_absent "${TAXONOMY_OUTPUTS[@]}"
  for path in "${TAXONOMY_V1}" "${TAXONOMY_V2}" "${REGISTRY_V2}" \
    "${REPO_ROOT}/benchmark/cslaw/validate_taxonomy.py"; do
    [[ -f "${path}" ]] || cslaw_die "taxonomy step: missing ${path}"
  done
fi
for step_config in credit:"${CREDIT_CONFIG}" nullbaselines:"${NULLB_CONFIG}" shared:"${SHARED_CONFIG}" \
  limelocal:"${LIMELOCAL_CONFIG}" limedefaultpilot:"${LIMEDEFAULT_PILOT_CONFIG}" \
  limedefault:"${LIMEDEFAULT_CONFIG}"; do
  if has_step "${step_config%%:*}"; then
    [[ -f "${step_config#*:}" ]] || cslaw_die "missing config ${step_config#*:}"
    for path in "${TAXONOMY_V2}" "${REGISTRY_V2}"; do
      [[ -f "${path}" ]] || cslaw_die "the v2 configs need ${path}"
    done
  fi
done
if has_step credit; then cslaw_guard_target "${CREDIT_DIR}"; fi
if has_step nullbaselines; then cslaw_guard_target "${NULLB_DIR}"; fi
if has_step shared; then cslaw_guard_target "${SHARED_DIR}"; fi
if has_step limelocal; then cslaw_guard_target "${LIMELOCAL_DIR}"; fi
if has_step limedefaultpilot; then cslaw_guard_target "${LIMEDEFAULT_PILOT_DIR}"; fi
if has_step limedefault; then cslaw_guard_target "${LIMEDEFAULT_DIR}"; fi

run_cslaw_py() {
  PYTHONPATH="${REPO_ROOT}/benchmark/src${PYTHONPATH:+:${PYTHONPATH}}" "${CSLAW_PY[@]}" "$@"
}

WORKTREE_PARENT=""
WORKTREE_DIR=""
cleanup_worktree() {
  if [[ -n "${WORKTREE_DIR}" ]]; then
    if [[ -d "${WORKTREE_DIR}" ]]; then
      git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE_DIR}" || true
    fi
    git -C "${REPO_ROOT}" worktree prune || true
  fi
  if [[ -n "${WORKTREE_PARENT}" && -d "${WORKTREE_PARENT}" ]]; then
    rmdir "${WORKTREE_PARENT}" 2>/dev/null || true
  fi
  WORKTREE_DIR=""
  WORKTREE_PARENT=""
}
trap cleanup_worktree EXIT

step_selftest() {
  cslaw_log "step selftest"
  run_cslaw_py benchmark/cslaw/selftest.py
}

step_regression() {
  cslaw_log "step regression (baseline ${CSLAW_BASELINE_COMMIT} vs current code)"
  local config="${CONFIGS}/experiment_cslaw_regression_check.json"
  git -C "${REPO_ROOT}" cat-file -e "${CSLAW_BASELINE_COMMIT}^{commit}" \
    || cslaw_die "baseline commit ${CSLAW_BASELINE_COMMIT} not found in ${REPO_ROOT}"

  WORKTREE_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/cslaw_baseline.XXXXXX")"
  WORKTREE_DIR="${WORKTREE_PARENT}/baseline"
  git -C "${REPO_ROOT}" worktree add --detach "${WORKTREE_DIR}" "${CSLAW_BASELINE_COMMIT}"

  # Both runs read the same data files (the baseline worktree uses its own code
  # and metadata). config.dataset_source_overrides honors these variables.
  export LEGALBENCH_XAI_DATASET_SOUTH_GERMAN_CREDIT_PATH="${REPO_ROOT}/benchmark/data/real/south_german_credit.csv"
  export LEGALBENCH_XAI_DATASET_HELOC_PATH="${REPO_ROOT}/benchmark/data/real/heloc.csv"

  cslaw_run_benchmark "${WORKTREE_DIR}/benchmark/src" "${config}" "${REG_BASE_DIR}"
  cslaw_check_bundle "${REG_BASE_DIR}" 0
  cslaw_run_benchmark "${REPO_ROOT}/benchmark/src" "${config}" "${REG_NEW_DIR}"
  cslaw_check_bundle "${REG_NEW_DIR}" 0

  unset LEGALBENCH_XAI_DATASET_SOUTH_GERMAN_CREDIT_PATH LEGALBENCH_XAI_DATASET_HELOC_PATH
  cleanup_worktree

  mkdir -p "${REG_CMP_DIR}"
  cslaw_log "compare_bundles baseline vs new -> ${REG_CMP_DIR}"
  # compare_bundles.py LEFT RIGHT (positional): exit 1 on any FAILURE; (ebm, shap)
  # score differences are EXPECTED_NONDETERMINISTIC and do not fail the step.
  if ! run_cslaw_py benchmark/cslaw/compare_bundles.py \
    "${REG_BASE_DIR}" \
    "${REG_NEW_DIR}" \
    --out "${REG_CMP_DIR}/comparison.md" | tee "${REG_CMP_DIR}/compare_stdout.txt"; then
    cslaw_die "regression check reported differences between ${REG_BASE_DIR} and ${REG_NEW_DIR}.
See ${REG_CMP_DIR}. If the only differences are the known cross-process EBM SHAP
nondeterminism, continue with: --steps taxonomy,credit,nullbaselines,shared --tag ${TAG}"
  fi
}

step_taxonomy() {
  cslaw_log "step taxonomy -> ${ANALYSIS_DIR}"
  mkdir -p "${ANALYSIS_DIR}"
  # validate_taxonomy.py exits non-zero on a validation error, which stops the job
  # here, before the long benchmark runs use the v2 files.
  run_cslaw_py benchmark/cslaw/validate_taxonomy.py \
    --taxonomy "${TAXONOMY_V2}" \
    --registry "${REGISTRY_V2}" \
    --compare-v1-taxonomy "${TAXONOMY_V1}" \
    --write-yaml "${ANALYSIS_DIR}/feature_taxonomy_manifest_v2.yaml" \
    --report "${ANALYSIS_DIR}/taxonomy_validation.md"
}

step_run() {
  local name="$1"
  local config="$2"
  local out_dir="$3"
  # Expected corrections for cslaw_check_bundle: v2, v2_lime_local (v2 plus
  # lime_sample_around_instance) or v2_lime_default (v2 plus lime_discretize_continuous),
  # the last two recorded as corrections_version "custom".
  local expect="${4:-v2}"
  cslaw_log "step ${name}"
  cslaw_run_benchmark "${REPO_ROOT}/benchmark/src" "${config}" "${out_dir}"
  cslaw_check_bundle "${out_dir}" 1 "${expect}"
}

if has_step selftest; then step_selftest; fi
if has_step regression; then step_regression; fi
if has_step taxonomy; then step_taxonomy; fi
if has_step credit; then step_run credit "${CREDIT_CONFIG}" "${CREDIT_DIR}"; fi
if has_step nullbaselines; then step_run nullbaselines "${NULLB_CONFIG}" "${NULLB_DIR}"; fi
if has_step shared; then step_run shared "${SHARED_CONFIG}" "${SHARED_DIR}"; fi
if has_step limelocal; then step_run limelocal "${LIMELOCAL_CONFIG}" "${LIMELOCAL_DIR}" v2_lime_local; fi
if has_step limedefaultpilot; then
  step_run limedefaultpilot "${LIMEDEFAULT_PILOT_CONFIG}" "${LIMEDEFAULT_PILOT_DIR}" v2_lime_default
fi
if has_step limedefault; then step_run limedefault "${LIMEDEFAULT_CONFIG}" "${LIMEDEFAULT_DIR}" v2_lime_default; fi

echo
echo "CS&Law v2 credit steps finished (${STEPS}); tag=${TAG}"
if has_step regression; then
  echo "  regression:    ${REG_BASE_DIR}"
  echo "                 ${REG_NEW_DIR}"
  echo "                 ${REG_CMP_DIR}"
fi
if has_step taxonomy; then echo "  taxonomy:      ${ANALYSIS_DIR}/taxonomy_validation.md"; fi
if has_step credit; then echo "  credit:        ${CREDIT_DIR}"; fi
if has_step nullbaselines; then echo "  nullbaselines: ${NULLB_DIR}"; fi
if has_step shared; then echo "  shared:        ${SHARED_DIR}"; fi
if has_step limelocal; then echo "  limelocal:     ${LIMELOCAL_DIR}"; fi
if has_step limedefaultpilot; then echo "  limedefaultpilot: ${LIMEDEFAULT_PILOT_DIR}"; fi
if has_step limedefault; then echo "  limedefault:   ${LIMEDEFAULT_DIR}"; fi
exit 0

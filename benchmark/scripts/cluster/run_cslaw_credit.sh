#!/usr/bin/env bash
# LeXAIBench v2 (CS&Law revision) credit runs and analyses, modeled on
# run_cross_domain_full_node001.sh.
#
# Usage:
#   bash benchmark/scripts/cluster/run_cslaw_credit.sh [REPO_ROOT] [--steps LIST] [--tag TAG]
#        [--force] [--noncredit-bundle DIR]
#
#   REPO_ROOT   repository checkout (default: $HOME/explainableAI-CS)
#   --steps     comma list, run in this fixed order (default: all):
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
#                 nullbaselines  experiment_cslaw_v2_credit_nullbaselines.json
#                 shared         experiment_cslaw_v2_credit_shared.json (shared sampling)
#                 offline        rescore/rebuild/paper tables on the published v1 bundles
#                                (benchmark/scripts/run_cslaw_offline_analyses.sh)
#                 analysis       paper_tables, version_diff (published v1 vs v2), regb_reference,
#                                common_faithfulness, multiplicity, proxy_substitution, notices,
#                                v2_legal_functions rescoring of the v2 bundles
#   --tag       suffix for every output directory (default: r1)
#   --force     allow writing into existing cslaw_* output directories
#   --noncredit-bundle  non-credit bundle for the analysis step. Default: the merged v2 bundle
#                       benchmark/outputs/cslaw_v2_noncredit_<TAG> (run_cslaw_noncredit_v2.sh)
#                       when it exists, else the published cross_domain_strongest_real_full_cluster.
#                       If it carries payload["cslaw"], the analysis step also runs
#                       regb_reference, common_faithfulness and notices (HMDA only, by
#                       notices.py's credit default) on it; if it is a v2 bundle, also
#                       version_diff against the published non-credit bundle and the
#                       v2_legal_functions rescoring.
#
# Outputs (all new directories under benchmark/outputs; published bundles are never written):
#   cslaw_regression_baseline_<TAG>/ cslaw_regression_new_<TAG>/ cslaw_regression_compare_<TAG>/
#   cslaw_v2_credit_<TAG>/ cslaw_v2_credit_nullbaselines_<TAG>/ cslaw_v2_credit_shared_<TAG>/
#   cslaw_offline_<TAG>/ cslaw_analysis_<TAG>/
# The taxonomy and analysis steps share cslaw_analysis_<TAG>/; each refuses only the
# entries it writes, so they can run in separate jobs.
set -euo pipefail

REPO_ROOT="${HOME}/explainableAI-CS"
if [[ $# -gt 0 && "$1" != -* ]]; then
  REPO_ROOT="$1"
  shift
fi

ALL_STEPS="selftest,regression,taxonomy,credit,nullbaselines,shared,offline,analysis"
STEPS="${ALL_STEPS}"
TAG="r1"
CSLAW_FORCE=0
NONCREDIT_BUNDLE=""

usage() {
  sed -n '2,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
TAXONOMY_V1="${METADATA}/feature_taxonomy.json"
TAXONOMY_V2="${METADATA}/feature_taxonomy_v2.json"
REGISTRY_V2="${METADATA}/dataset_registry_v2.json"
CREDIT_DIR="${OUTPUTS}/cslaw_v2_credit_${TAG}"
NULLB_DIR="${OUTPUTS}/cslaw_v2_credit_nullbaselines_${TAG}"
SHARED_DIR="${OUTPUTS}/cslaw_v2_credit_shared_${TAG}"
NONCREDIT_V2_DIR="${OUTPUTS}/cslaw_v2_noncredit_${TAG}"
REG_BASE_DIR="${OUTPUTS}/cslaw_regression_baseline_${TAG}"
REG_NEW_DIR="${OUTPUTS}/cslaw_regression_new_${TAG}"
REG_CMP_DIR="${OUTPUTS}/cslaw_regression_compare_${TAG}"
OFFLINE_DIR="${OUTPUTS}/cslaw_offline_${TAG}"
ANALYSIS_DIR="${OUTPUTS}/cslaw_analysis_${TAG}"
PUBLISHED_CREDIT_DIR="${OUTPUTS}/${CSLAW_PUBLISHED_CREDIT_BUNDLE_NAME}"
PUBLISHED_NONCREDIT_DIR="${OUTPUTS}/${CSLAW_PUBLISHED_NONCREDIT_BUNDLE_NAME}"

# Entries each step owns inside the shared analysis directory.
TAXONOMY_OUTPUTS=(
  "${ANALYSIS_DIR}/taxonomy_validation.md"
  "${ANALYSIS_DIR}/feature_taxonomy_manifest_v2.yaml"
)
ANALYSIS_OUTPUTS=(
  "${ANALYSIS_DIR}/paper_tables"
  "${ANALYSIS_DIR}/version_diff"
  "${ANALYSIS_DIR}/version_diff_noncredit"
  "${ANALYSIS_DIR}/regb_reference_credit"
  "${ANALYSIS_DIR}/regb_reference_shared"
  "${ANALYSIS_DIR}/common_faithfulness"
  "${ANALYSIS_DIR}/common_faithfulness_shared"
  "${ANALYSIS_DIR}/multiplicity"
  "${ANALYSIS_DIR}/proxy_substitution"
  "${ANALYSIS_DIR}/notices"
  "${ANALYSIS_DIR}/rescored_v2"
  "${ANALYSIS_DIR}/regb_reference_noncredit"
  "${ANALYSIS_DIR}/common_faithfulness_noncredit"
  "${ANALYSIS_DIR}/notices_noncredit"
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
for step_config in credit:"${CREDIT_CONFIG}" nullbaselines:"${NULLB_CONFIG}" shared:"${SHARED_CONFIG}"; do
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
if has_step offline; then cslaw_guard_target "${OFFLINE_DIR}"; fi
if has_step analysis; then
  cslaw_guard_name "${ANALYSIS_DIR}"
  cslaw_guard_paths_absent "${ANALYSIS_OUTPUTS[@]}"
fi

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
nondeterminism, continue with: --steps taxonomy,credit,nullbaselines,shared,offline,analysis --tag ${TAG}"
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
  cslaw_log "step ${name}"
  cslaw_run_benchmark "${REPO_ROOT}/benchmark/src" "${config}" "${out_dir}"
  cslaw_check_bundle "${out_dir}" 1 v2
}

step_offline() {
  cslaw_log "step offline"
  local args=("${REPO_ROOT}" --tag "${TAG}")
  if [[ "${CSLAW_FORCE}" -eq 1 ]]; then
    args+=(--force)
  fi
  bash "${REPO_ROOT}/benchmark/scripts/run_cslaw_offline_analyses.sh" "${args[@]}"
}

SKIPPED=()
FAILED=()

# True when results.json carries payload["cslaw"], i.e. the bundle was produced
# with a cslaw block and has the persisted fields the analyses need.
bundle_has_cslaw() {
  "${CSLAW_PY[@]}" - "$1/results.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
sys.exit(0 if isinstance(payload.get("cslaw"), dict) else 1)
PY
}

# True when results.json is a LeXAIBench v2 bundle (payload["cslaw"]["corrections_version"]).
bundle_is_v2() {
  "${CSLAW_PY[@]}" - "$1/results.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
block = payload.get("cslaw")
sys.exit(0 if isinstance(block, dict) and block.get("corrections_version") == "v2" else 1)
PY
}

# Runs one optional analysis. A failure is reported at the end (exit 1) instead of
# aborting after the other analyses were written.
run_soft() {
  local label="$1"
  shift
  cslaw_log "${label}"
  if ! run_cslaw_py "$@"; then
    cslaw_log "WARNING: ${label} failed"
    FAILED+=("${label}")
    return 1
  fi
}

require_bundle() {
  local label="$1"
  local bundle="$2"
  local analysis="$3"
  if cslaw_bundle_ready "${bundle}"; then
    return 0
  fi
  cslaw_log "WARNING: skipping ${analysis}: ${label} bundle ${bundle} has no results.json"
  SKIPPED+=("${analysis} (missing ${label} bundle ${bundle})")
  return 1
}

# rescore_v2_legal_functions LABEL BUNDLE CONFIG: rescoring of a v2 bundle with the
# v2_legal_functions variant of rescore.py, then rebuild-artifacts with the config
# the bundle was produced with.
rescore_v2_legal_functions() {
  local label="$1"
  local bundle="$2"
  local config="$3"
  local out="${ANALYSIS_DIR}/rescored_v2/${label}/v2_legal_functions"
  if [[ ! -f "${config}" ]]; then
    cslaw_log "WARNING: skipping v2_legal_functions (${label}): config ${config} not found"
    SKIPPED+=("v2_legal_functions ${label} (missing config ${config})")
    return 0
  fi
  if ! run_soft "rescore v2_legal_functions (${label})" benchmark/cslaw/rescore.py \
    --results "${bundle}/results.json" \
    --variant v2_legal_functions \
    --out-dir "${out}"; then
    return 0
  fi
  if [[ ! -f "${out}/results.json" ]]; then
    cslaw_log "WARNING: rescore.py did not write ${out}/results.json"
    FAILED+=("rescore v2_legal_functions (${label}): no results.json")
    return 0
  fi
  if ! run_cslaw_py -m legalbench_xai rebuild-artifacts \
    --config "${config}" \
    --results-json "${out}/results.json" \
    --output-dir "${out}" >"${out}/rebuild_artifacts.json"; then
    cslaw_log "WARNING: rebuild-artifacts for v2_legal_functions (${label}) failed"
    FAILED+=("rebuild-artifacts v2_legal_functions (${label})")
  fi
}

step_analysis() {
  cslaw_log "step analysis -> ${ANALYSIS_DIR}"
  mkdir -p "${ANALYSIS_DIR}"

  local noncredit="${NONCREDIT_BUNDLE}"
  if [[ -z "${noncredit}" ]]; then
    if cslaw_bundle_ready "${NONCREDIT_V2_DIR}"; then
      noncredit="${NONCREDIT_V2_DIR}"
    else
      noncredit="${PUBLISHED_NONCREDIT_DIR}"
      cslaw_log "WARNING: ${NONCREDIT_V2_DIR} not found; the non-credit tables use the published v1 bundle"
      SKIPPED+=("v2 non-credit analyses (missing ${NONCREDIT_V2_DIR}; run run_cslaw_noncredit_v2.sh --merge first)")
    fi
  fi
  cslaw_log "credit bundle: ${CREDIT_DIR}; non-credit bundle: ${noncredit}"

  if require_bundle credit "${CREDIT_DIR}" paper_tables \
    && require_bundle noncredit "${noncredit}" paper_tables; then
    local tables_args=(
      --bundle "credit=${CREDIT_DIR}"
      --bundle "noncredit=${noncredit}"
      --robustness-root "${OUTPUTS}"
    )
    if cslaw_bundle_ready "${NULLB_DIR}"; then
      tables_args+=(--bundle "nullbaselines=${NULLB_DIR}")
    fi
    if [[ -d "${OFFLINE_DIR}/rescored" ]]; then
      tables_args+=(--rescored-root "${OFFLINE_DIR}/rescored")
    fi
    # The offline step writes the S-weight sweeps outside the rescored root.
    local sweep_csv
    for sweep_csv in "${OFFLINE_DIR}"/dsv_sweep/*/css_dsv_sweep.csv; do
      if [[ -f "${sweep_csv}" ]]; then
        tables_args+=(--sweep-csv "${sweep_csv}")
      fi
    done
    run_cslaw_py benchmark/cslaw/paper_tables.py "${tables_args[@]}" \
      --out-dir "${ANALYSIS_DIR}/paper_tables"
  fi

  # What changed between the published v1 credit results and v2.
  if require_bundle "published credit" "${PUBLISHED_CREDIT_DIR}" version_diff \
    && require_bundle credit "${CREDIT_DIR}" version_diff; then
    run_soft "version_diff (credit, published v1 vs v2)" benchmark/cslaw/version_diff.py \
      --v1 "${PUBLISHED_CREDIT_DIR}" \
      --v2 "${CREDIT_DIR}" \
      --out-dir "${ANALYSIS_DIR}/version_diff" || true
  fi

  if require_bundle credit "${CREDIT_DIR}" "regb_reference (credit)"; then
    run_cslaw_py benchmark/cslaw/regb_reference.py \
      --bundle "${CREDIT_DIR}" \
      --out-dir "${ANALYSIS_DIR}/regb_reference_credit"
  fi
  if require_bundle shared "${SHARED_DIR}" "regb_reference (shared)"; then
    run_cslaw_py benchmark/cslaw/regb_reference.py \
      --bundle "${SHARED_DIR}" \
      --out-dir "${ANALYSIS_DIR}/regb_reference_shared"
  fi

  if require_bundle credit "${CREDIT_DIR}" "common_faithfulness (credit + null baselines)"; then
    local faith_args=(--bundle "${CREDIT_DIR}")
    if cslaw_bundle_ready "${NULLB_DIR}"; then
      faith_args+=(--bundle "${NULLB_DIR}")
    else
      cslaw_log "WARNING: common_faithfulness runs without null baselines (${NULLB_DIR} missing)"
      SKIPPED+=("common_faithfulness null-baseline comparison (missing ${NULLB_DIR})")
    fi
    run_cslaw_py benchmark/cslaw/common_faithfulness.py "${faith_args[@]}" \
      --out-dir "${ANALYSIS_DIR}/common_faithfulness"
  fi
  if require_bundle shared "${SHARED_DIR}" "common_faithfulness (shared)"; then
    run_cslaw_py benchmark/cslaw/common_faithfulness.py \
      --bundle "${SHARED_DIR}" \
      --out-dir "${ANALYSIS_DIR}/common_faithfulness_shared"
  fi

  if require_bundle shared "${SHARED_DIR}" multiplicity; then
    run_cslaw_py benchmark/cslaw/multiplicity.py \
      --bundle "${SHARED_DIR}" \
      --out-dir "${ANALYSIS_DIR}/multiplicity"
  fi
  if require_bundle shared "${SHARED_DIR}" proxy_substitution; then
    run_cslaw_py benchmark/cslaw/proxy_substitution.py \
      --bundle "${SHARED_DIR}" \
      --out-dir "${ANALYSIS_DIR}/proxy_substitution"
  fi

  if require_bundle credit "${CREDIT_DIR}" notices; then
    local notices_args=(--bundle "${CREDIT_DIR}" --n 48 --seed 7 --out-dir "${ANALYSIS_DIR}/notices")
    if [[ "${CSLAW_FORCE}" -eq 1 ]]; then
      # notices.py refuses to replace an existing notices_key.csv on its own;
      # the packet is deterministic for the same bundle and seed.
      notices_args+=(--force)
    fi
    run_cslaw_py benchmark/cslaw/notices.py "${notices_args[@]}"
  fi

  if cslaw_bundle_ready "${CREDIT_DIR}"; then
    rescore_v2_legal_functions credit "${CREDIT_DIR}" "${CREDIT_CONFIG}"
  fi

  # Persisted-field analyses on a cslaw-instrumented non-credit bundle
  # (run_cslaw_noncredit_v2.sh --merge). The published non-credit bundle stores
  # scores only, so it is used by paper_tables alone.
  if cslaw_bundle_ready "${noncredit}"; then
    if bundle_has_cslaw "${noncredit}"; then
      run_soft "regb_reference (noncredit)" benchmark/cslaw/regb_reference.py \
        --bundle "${noncredit}" \
        --out-dir "${ANALYSIS_DIR}/regb_reference_noncredit" || true
      run_soft "common_faithfulness (noncredit)" benchmark/cslaw/common_faithfulness.py \
        --bundle "${noncredit}" \
        --out-dir "${ANALYSIS_DIR}/common_faithfulness_noncredit" || true
      # notices.py renders only the credit datasets by default, i.e. HMDA here. This
      # packet is separate from the credit rating packet in notices/.
      local noncredit_notices_args=(
        --bundle "${noncredit}" --n 48 --seed 7 --out-dir "${ANALYSIS_DIR}/notices_noncredit"
      )
      if [[ "${CSLAW_FORCE}" -eq 1 ]]; then
        noncredit_notices_args+=(--force)
      fi
      run_soft "notices (noncredit, HMDA)" benchmark/cslaw/notices.py \
        "${noncredit_notices_args[@]}" || true
      if bundle_is_v2 "${noncredit}"; then
        if require_bundle "published noncredit" "${PUBLISHED_NONCREDIT_DIR}" "version_diff (noncredit)"; then
          run_soft "version_diff (noncredit, published v1 vs v2)" benchmark/cslaw/version_diff.py \
            --v1 "${PUBLISHED_NONCREDIT_DIR}" \
            --v2 "${noncredit}" \
            --out-dir "${ANALYSIS_DIR}/version_diff_noncredit" || true
        fi
        rescore_v2_legal_functions noncredit "${noncredit}" "${noncredit}/runtime_config.json"
      fi
    else
      cslaw_log "non-credit bundle ${noncredit} has no payload[\"cslaw\"]" \
        "(published bundle); regb_reference, common_faithfulness and notices use the credit bundles only"
    fi
  fi
}

if has_step selftest; then step_selftest; fi
if has_step regression; then step_regression; fi
if has_step taxonomy; then step_taxonomy; fi
if has_step credit; then step_run credit "${CREDIT_CONFIG}" "${CREDIT_DIR}"; fi
if has_step nullbaselines; then step_run nullbaselines "${NULLB_CONFIG}" "${NULLB_DIR}"; fi
if has_step shared; then step_run shared "${SHARED_CONFIG}" "${SHARED_DIR}"; fi
if has_step offline; then step_offline; fi
if has_step analysis; then step_analysis; fi

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
if has_step offline; then echo "  offline:       ${OFFLINE_DIR}"; fi
if has_step analysis; then echo "  analysis:      ${ANALYSIS_DIR}"; fi
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  echo
  echo "WARNING: skipped analyses:"
  for item in "${SKIPPED[@]}"; do
    echo "  - ${item}"
  done
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo
  echo "ERROR: failed analyses (see the messages above):"
  for item in "${FAILED[@]}"; do
    echo "  - ${item}"
  done
  exit 1
fi
exit 0

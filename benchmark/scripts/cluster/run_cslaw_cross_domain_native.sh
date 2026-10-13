#!/usr/bin/env bash
# SUPERSEDED by run_cslaw_noncredit_v2.sh (LeXAIBench v2: all corrections, v2
# taxonomy, 24 cases, runs sharded by dataset:track and merged). Kept only to
# reproduce the round-1 bundle cslaw_cross_domain_native_<TAG>.
#
# Optional CS&Law non-credit re-run (HMDA, NY Courts pretrial, FDA inspections)
# with the round-1 cslaw native block (model-native ranking contributions, persisted
# canonical explanations and native contributions, Reg B reference means; every v1
# scoring and sampling rule unchanged). One monolithic job.
# Modeled on run_cross_domain_full_node001.sh; wraps
# benchmark/scripts/run-cross-domain-strongest-real.sh --cslaw-native.
#
# Usage:
#   bash benchmark/scripts/cluster/run_cslaw_cross_domain_native.sh [REPO_ROOT] [--tag TAG]
#        [--light-ebm] [--force]
#
#   REPO_ROOT    repository checkout (default: $HOME/explainableAI-CS)
#   --tag        output suffix (default: r1) -> benchmark/outputs/cslaw_cross_domain_native_<TAG>
#   --light-ebm  pass --light-ebm through (fewer EBM interactions/bins/rounds). Faster, but
#                the EBM rows are then NOT comparable with the published cross-domain bundle.
#   --force      allow writing into an existing output directory
set -euo pipefail

REPO_ROOT="${HOME}/explainableAI-CS"
if [[ $# -gt 0 && "$1" != -* ]]; then
  REPO_ROOT="$1"
  shift
fi

TAG="r1"
LIGHT_EBM=0
CSLAW_FORCE=0

usage() {
  sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="$2"
      shift 2
      ;;
    --light-ebm)
      LIGHT_EBM=1
      shift
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

cat >&2 <<'WARN'
[cslaw] ######################################################################
[cslaw] NOTE: this round-1 script is superseded by run_cslaw_noncredit_v2.sh
[cslaw] (LeXAIBench v2 corrections, sharded by dataset:track).
[cslaw] WARNING: long run. In the previous full non-credit cluster run
[cslaw] (cross_domain_strongest_real_full_cluster, ~30 h wall clock) the full
[cslaw] EBM fit on NY Courts pretrial alone took ~22.6 h. Budget the job time
[cslaw] accordingly, or pass --light-ebm (EBM results then differ from the
[cslaw] published bundle and must not be compared with it).
[cslaw] ######################################################################
WARN

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

uv sync --extra benchmark-empirical

# shellcheck source=cslaw_lib.sh
source "${REPO_ROOT}/benchmark/scripts/cluster/cslaw_lib.sh"
export CSLAW_FORCE
cslaw_validate_tag "${TAG}"
cslaw_resolve_python "${REPO_ROOT}"

if [[ ! -x benchmark/scripts/run-legalbench.sh ]]; then
  # run-cross-domain-strongest-real.sh executes run-legalbench.sh directly, and
  # git tracks it as 100644, so a fresh clone cannot execute it.
  cslaw_die "benchmark/scripts/run-legalbench.sh is not executable; run: chmod +x benchmark/scripts/run-legalbench.sh"
fi

OUTPUT_NAME="cslaw_cross_domain_native_${TAG}"
OUT_DIR="${REPO_ROOT}/benchmark/outputs/${OUTPUT_NAME}"
cslaw_guard_target "${OUT_DIR}"
if [[ -e "${OUT_DIR}/progress.jsonl" ]]; then
  # run-cross-domain-strongest-real.sh appends to this log; keep runs separate.
  mv "${OUT_DIR}/progress.jsonl" "${OUT_DIR}/progress.jsonl.$(date -u +%Y%m%dT%H%M%SZ).bak"
fi

RUN_ARGS=(
  --output-name "${OUTPUT_NAME}"
  --cslaw-native
  --full-validate
  --summary-only
)
if [[ ${LIGHT_EBM} -eq 1 ]]; then
  RUN_ARGS+=(--light-ebm)
  cslaw_log "WARNING: --light-ebm enabled; EBM rows are not comparable with ${CSLAW_PUBLISHED_NONCREDIT_BUNDLE_NAME}"
fi

cslaw_log "run-cross-domain-strongest-real.sh ${RUN_ARGS[*]}"
bash benchmark/scripts/run-cross-domain-strongest-real.sh "${RUN_ARGS[@]}"

cslaw_check_bundle "${OUT_DIR}" 1

echo
echo "CS&Law non-credit native bundle: ${OUT_DIR}"
echo "To use it in the credit analysis step (add --force if cslaw_analysis_${TAG} already exists):"
echo "  bash benchmark/scripts/cluster/run_cslaw_credit.sh ${REPO_ROOT} --steps analysis --tag ${TAG} --noncredit-bundle ${OUT_DIR}"
echo "That step uses it as paper_tables 'noncredit' and also runs regb_reference, common_faithfulness"
echo "and notices (HMDA) on it, into cslaw_analysis_${TAG}/{regb_reference,common_faithfulness,notices}_noncredit/."

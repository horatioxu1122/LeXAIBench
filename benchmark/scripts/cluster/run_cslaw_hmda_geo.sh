#!/usr/bin/env bash
# HMDA re-run for E5 with geography target-encoded (corrections option geo_target_encoding): county and census
# tract enter the models as the smoothed training denial rate of each code instead of the numeric code. Runs the
# primary and audit shards in parallel (SHAP, LIME, Anchors; about 3-4 h, set by the EBM fit on 6.2M rows), merges
# them, applies the direction rule, and writes the E5 analyses next to an existing cslaw_denial_<TAG>.
#
# Usage:
#   bash benchmark/scripts/cluster/run_cslaw_hmda_geo.sh [REPO_ROOT] [--tag TAG]
#
# Outputs: benchmark/outputs/cslaw_v2_hmda_geo_{primary,audit}_<TAG>/ (shards), cslaw_v2_hmda_geo_<TAG>/ (merged),
#   cslaw_denial_<TAG>/hmda_geo/ (direction rule) and cslaw_denial_<TAG>/analysis/hmda_geo/{regb_reference,
#   track_disclosure,specificity}/. Resumable: a step whose output already exists (results.json, or the analysis
#   step's main file) is reused, so after a failure the same command continues from the failed step.
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
    -h | --help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if command -v module >/dev/null 2>&1; then
  module load python3
fi
cd "${REPO_ROOT}"
REPO_ROOT="$(pwd)"
if [[ "${CSLAW_SKIP_SYNC:-0}" != "1" ]]; then
  uv sync --extra benchmark-empirical
fi
# shellcheck source=cslaw_lib.sh
source "${REPO_ROOT}/benchmark/scripts/cluster/cslaw_lib.sh"
cslaw_validate_tag "${TAG}"
cslaw_export_thread_env "${REPO_ROOT}"
cslaw_resolve_python "${REPO_ROOT}"
run_py() {
  PYTHONPATH="${REPO_ROOT}/benchmark/src${PYTHONPATH:+:${PYTHONPATH}}" "${CSLAW_PY[@]}" "$@"
}

OUTPUTS="${REPO_ROOT}/benchmark/outputs"
CONFIGS="${REPO_ROOT}/benchmark/configs"
MERGED="${OUTPUTS}/cslaw_v2_hmda_geo_${TAG}"
DEST="${OUTPUTS}/cslaw_denial_${TAG}"
A="${DEST}/analysis/hmda_geo"
[[ -f "${REPO_ROOT}/benchmark/data/real/hmda.csv" ]] || cslaw_die "missing benchmark/data/real/hmda.csv"
[[ -d "${DEST}/analysis" ]] || cslaw_die "missing ${DEST}/analysis; run run_cslaw_denial.sh first"
for track in primary audit; do
  [[ -f "${CONFIGS}/experiment_cslaw_v2_hmda_geo_${track}.json" ]] || cslaw_die "missing config for ${track}"
done

# Pilot (a few minutes): the audit settings on benchmark/data/real/hmda_pilot_fast.csv, to catch a failure before the
# long fits. It must load the geographic codes as text and produce notices for every method.
PILOT="${OUTPUTS}/cslaw_v2_hmda_geo_pilot_${TAG}"
PILOT_DATA="benchmark/data/real/hmda_pilot_fast.csv"
if [[ ! -f "${PILOT}/results.json" ]]; then
  [[ -f "${REPO_ROOT}/${PILOT_DATA}" ]] || cslaw_die "missing ${PILOT_DATA} (needed for the pilot)"
  mkdir -p "${PILOT}"
  run_py - "${CONFIGS}/experiment_cslaw_v2_hmda_geo_audit.json" "${PILOT}/runtime_config.json" "${PILOT_DATA}" <<'PY'
import json
import sys

source, target, data = sys.argv[1:4]
config = json.load(open(source, encoding="utf-8"))
config["experiment"]["name"] = "legalbench_xai_cslaw_v2_hmda_geo_pilot"
config["dataset_sources"]["local_paths"]["hmda"] = data
json.dump(config, open(target, "w", encoding="utf-8"), indent=2)
PY
  cslaw_log "pilot -> ${PILOT}"
  cslaw_run_benchmark "${REPO_ROOT}/benchmark/src" "${PILOT}/runtime_config.json" "${PILOT}" > "${PILOT}.log" 2>&1 \
    || cslaw_die "pilot failed; see ${PILOT}.log"
  cslaw_check_bundle "${PILOT}" 1
  run_py - "${PILOT}/results.json" <<'PY' || cslaw_die "pilot check failed"
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
corrections = (payload.get("cslaw") or {}).get("corrections") or {}
assert corrections.get("geo_target_encoding") is True, "geo_target_encoding not recorded"
explainers, kinds = set(), {}
for row in payload.get("summary", []):
    for instance in row.get("instance_results") or []:
        canonical = instance.get("canonical_explanation") or {}
        if canonical.get("referenced_features"):
            explainers.add(row.get("explainer_id"))
        for feature in canonical.get("referenced_features") or []:
            if feature.get("feature_name") in ("county_code", "census_tract") and feature.get("raw_value") is not None:
                kinds.setdefault(feature["feature_name"], set()).add(type(feature["raw_value"]).__name__)
assert explainers >= {"shap", "lime", "anchors"}, f"notices only from {sorted(explainers)}"
assert all(k == {"str"} for k in kinds.values()), f"geographic codes not loaded as text: {kinds}"
print(f"[cslaw] pilot ok: notices from {sorted(explainers)}; geographic raw values {kinds or 'not cited'}", file=sys.stderr)
PY
fi

# Both shards at once; each is one process with pinned threads. A shard with a results.json is reused. Each shard
# folder gets its config as runtime_config.json, so that a new merge also writes the merged runtime_config.json.
pids=()
for track in primary audit; do
  out="${OUTPUTS}/cslaw_v2_hmda_geo_${track}_${TAG}"
  if cslaw_bundle_ready "${out}"; then
    cslaw_log "shard hmda:${track}: reusing ${out}"
    continue
  fi
  mkdir -p "${out}"
  cp "${CONFIGS}/experiment_cslaw_v2_hmda_geo_${track}.json" "${out}/runtime_config.json"
  cslaw_log "shard hmda:${track} -> ${out} (log: ${out}.log)"
  cslaw_run_benchmark "${REPO_ROOT}/benchmark/src" "${out}/runtime_config.json" "${out}" > "${out}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ ${status} -eq 0 ]] || cslaw_die "a shard failed; see ${OUTPUTS}/cslaw_v2_hmda_geo_*_${TAG}.log"
for track in primary audit; do
  cslaw_check_bundle "${OUTPUTS}/cslaw_v2_hmda_geo_${track}_${TAG}" 1
done

if cslaw_bundle_ready "${MERGED}"; then
  cslaw_log "merge: reusing ${MERGED}"
else
  cslaw_guard_target "${MERGED}"
  cslaw_log "merge -> ${MERGED}"
  run_py benchmark/cslaw/merge_bundles.py \
    --inputs "${OUTPUTS}/cslaw_v2_hmda_geo_primary_${TAG}" "${OUTPUTS}/cslaw_v2_hmda_geo_audit_${TAG}" \
    --out-dir "${MERGED}" --experiment-name legalbench_xai_cslaw_v2_hmda_geo
  [[ -f "${MERGED}/results.json" ]] || cslaw_die "merge_bundles.py did not write ${MERGED}/results.json"
fi

mkdir -p "${A}"
step() {  # label, done-file, then command; skipped when the done-file exists
  local label="$1" done_file="$2"; shift 2
  if [[ -f "${done_file}" ]]; then
    cslaw_log "hmda_geo: ${label}: reusing $(dirname "${done_file}")"
    return 0
  fi
  cslaw_log "hmda_geo: ${label}"
  "$@" > "${A}/log_${label}.txt" 2>&1 || cslaw_die "${label} failed (see ${A}/log_${label}.txt)"
}
# The direction rule rebuilds the HMDA dataset definitions from a config; both geo configs name the same taxonomy,
# registry and data, so the audit config serves for both versions. --force writes into a folder a failed attempt left.
step denial "${DEST}/hmda_geo/results.json" run_py benchmark/cslaw/denial_reasons.py --bundle "${MERGED}" \
  --config "${CONFIGS}/experiment_cslaw_v2_hmda_geo_audit.json" --out-dir "${DEST}/hmda_geo" --force
step regb_reference "${A}/regb_reference/regb_reference_instances.csv" \
  run_py benchmark/cslaw/regb_reference.py --bundle "${DEST}/hmda_geo" --out-dir "${A}/regb_reference"
step track_disclosure "${A}/track_disclosure/track_disclosure.csv" \
  run_py benchmark/cslaw/track_disclosure.py --bundle "${DEST}/hmda_geo" --out-dir "${A}/track_disclosure"
step specificity "${A}/specificity/specificity_pooled.csv" \
  run_py benchmark/cslaw/notice_specificity.py --bundle "${DEST}/hmda_geo" --out-dir "${A}/specificity"
cslaw_log "done: ${DEST}/hmda_geo and ${A}"

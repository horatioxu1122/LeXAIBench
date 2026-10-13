#!/usr/bin/env bash
# LeXAIBench v2 (CS&Law revision) non-credit runs (HMDA, NY Courts pretrial, FDA
# inspections), sharded by dataset:track so that each shard can run as its own
# cluster job, plus the merge of the shard bundles into one non-credit bundle.
#
# Usage:
#   bash benchmark/scripts/cluster/run_cslaw_noncredit_v2.sh [REPO_ROOT]
#        (--shard DATASET:TRACK | --all-sequential | --merge) [--tag TAG] [--light-ebm] [--force]
#
#   REPO_ROOT              repository checkout (default: $HOME/explainableAI-CS)
#   --shard DATASET:TRACK  run one shard. DATASET: hmda | ny_courts_pretrial | fda_inspections;
#                          TRACK: primary | audit. The runtime config comes from
#                          make_cross_domain_runtime_config.py --cslaw-v2 --datasets DATASET
#                          --tracks TRACK (corrections v2, v2 taxonomy and registry, 24 cases).
#                          Output: benchmark/outputs/cslaw_v2_noncredit_<DATASET>_<TRACK>_<TAG>/
#   --all-sequential       run all six shards one after another in this job, then --merge.
#   --merge                merge the shard bundles of this tag that exist with
#                          benchmark/cslaw/merge_bundles.py into
#                          benchmark/outputs/cslaw_v2_noncredit_<TAG>/, then rebuild-artifacts
#                          there with a merged runtime config (both tracks). Missing shards are
#                          listed in a loud warning and in shard_manifest.json.
#   --tag TAG              output suffix (default: r1). Use one tag for all shards and the merge.
#   --light-ebm            fewer EBM interactions/bins/rounds. Faster, but it CHANGES THE EBM
#                          MODEL: its rows are then not comparable with the published non-credit
#                          bundle or with full-EBM shards. --merge refuses shards that mix it.
#   --force                allow writing into existing output directories.
#
# Every shard gets the same checks as the credit runs (cslaw_check_bundle ... 1 v2):
# exit 3 on any stub-backend fallback, cslaw error or missing v2 correction.
set -euo pipefail

REPO_ROOT="${HOME}/explainableAI-CS"
if [[ $# -gt 0 && "$1" != -* ]]; then
  REPO_ROOT="$1"
  shift
fi

DATASETS=(hmda ny_courts_pretrial fda_inspections)
TRACKS=(primary audit)
MODE=""
SHARD=""
TAG="r1"
LIGHT_EBM=0
CSLAW_FORCE=0

usage() {
  sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

set_mode() {
  if [[ -n "${MODE}" && "${MODE}" != "$1" ]]; then
    echo "Choose exactly one of --shard, --all-sequential, --merge" >&2
    exit 1
  fi
  MODE="$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shard)
      set_mode shard
      if [[ $# -lt 2 ]]; then
        echo "--shard needs DATASET:TRACK" >&2
        exit 1
      fi
      SHARD="$2"
      shift 2
      ;;
    --all-sequential)
      set_mode all
      shift
      ;;
    --merge)
      set_mode merge
      shift
      ;;
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

if [[ -z "${MODE}" ]]; then
  echo "Choose one of --shard DATASET:TRACK, --all-sequential, --merge (see --help)" >&2
  exit 1
fi

contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

SHARD_DATASET=""
SHARD_TRACK=""
if [[ "${MODE}" == "shard" ]]; then
  if [[ "${SHARD}" != *:* ]]; then
    echo "--shard expects DATASET:TRACK, e.g. ny_courts_pretrial:primary (got '${SHARD}')" >&2
    exit 1
  fi
  SHARD_DATASET="${SHARD%%:*}"
  SHARD_TRACK="${SHARD#*:}"
  if ! contains "${SHARD_DATASET}" "${DATASETS[@]}"; then
    echo "unknown dataset '${SHARD_DATASET}' (allowed: ${DATASETS[*]})" >&2
    exit 1
  fi
  if ! contains "${SHARD_TRACK}" "${TRACKS[@]}"; then
    echo "unknown track '${SHARD_TRACK}' (allowed: ${TRACKS[*]})" >&2
    exit 1
  fi
fi

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
CONFIG_BUILDER="${REPO_ROOT}/benchmark/scripts/make_cross_domain_runtime_config.py"
MERGED_DIR="${OUTPUTS}/cslaw_v2_noncredit_${TAG}"
EXPERIMENT_NAME="legalbench_xai_cslaw_v2_noncredit"

for path in "${CONFIG_BUILDER}" \
  "${REPO_ROOT}/benchmark/metadata/feature_taxonomy_v2.json" \
  "${REPO_ROOT}/benchmark/metadata/dataset_registry_v2.json"; do
  [[ -f "${path}" ]] || cslaw_die "missing ${path}"
done
if [[ "${MODE}" != "shard" ]]; then
  [[ -f "${REPO_ROOT}/benchmark/cslaw/merge_bundles.py" ]] \
    || cslaw_die "missing benchmark/cslaw/merge_bundles.py"
fi

run_cslaw_py() {
  PYTHONPATH="${REPO_ROOT}/benchmark/src${PYTHONPATH:+:${PYTHONPATH}}" "${CSLAW_PY[@]}" "$@"
}

shard_dir() {
  printf '%s' "${OUTPUTS}/cslaw_v2_noncredit_$1_$2_${TAG}"
}

dataset_path() {
  # The make_cross_domain_runtime_config.py defaults (repo-relative).
  case "$1" in
    hmda) printf '%s' "benchmark/data/real/hmda.csv" ;;
    ny_courts_pretrial) printf '%s' "benchmark/data/real/ny_courts_pretrial.csv" ;;
    fda_inspections) printf '%s' "benchmark/data/real/fda_inspections.csv" ;;
    *) cslaw_die "no data path for dataset '$1'" ;;
  esac
}

if [[ ${LIGHT_EBM} -eq 1 ]]; then
  # Same light settings as run-cross-domain-strongest-real.sh --light-ebm.
  export LEGALBENCH_EBM_INTERACTIONS="${LEGALBENCH_EBM_INTERACTIONS:-0}"
  export LEGALBENCH_EBM_MAX_BINS="${LEGALBENCH_EBM_MAX_BINS:-32}"
  export LEGALBENCH_EBM_MAX_ROUNDS="${LEGALBENCH_EBM_MAX_ROUNDS:-80}"
fi

warn_ebm() {
  local dataset="$1"
  if [[ "${dataset}" == "ny_courts_pretrial" || "${dataset}" == "all" ]]; then
    cat >&2 <<'WARN'
[cslaw] ######################################################################
[cslaw] RUNTIME WARNING: in the published non-credit run (~30 h wall clock) the
[cslaw] full EBM fit on NY Courts pretrial took ~22.6 h for both tracks, i.e.
[cslaw] roughly 11 h or more per ny_courts_pretrial shard for the EBM fit alone,
[cslaw] before any explainer runs (v2 explains 24 cases per group, twice v1).
[cslaw] Budget the job time limit accordingly. The other shards are much shorter.
[cslaw] ######################################################################
WARN
  fi
  if [[ ${LIGHT_EBM} -eq 1 ]]; then
    cat >&2 <<'WARN'
[cslaw] ######################################################################
[cslaw] WARNING: --light-ebm CHANGES THE EBM MODEL (fewer interactions, bins and
[cslaw] boosting rounds). Its EBM rows are not comparable with the published
[cslaw] non-credit bundle or with full-EBM shards, and the paper must report the
[cslaw] EBM settings. Use it for every shard of this tag or for none.
[cslaw] ######################################################################
WARN
  else
    local name
    for name in LEGALBENCH_EBM_INTERACTIONS LEGALBENCH_EBM_MAX_BINS LEGALBENCH_EBM_MAX_ROUNDS; do
      if [[ -n "${!name:-}" ]]; then
        cslaw_log "WARNING: ${name}=${!name} is set in the environment; the EBM differs from the published settings"
      fi
    done
  fi
}

# write_shard_info OUT_DIR DATASET TRACK PHASE(started|finished)
# Records how the shard was produced; --merge checks that the model settings agree.
write_shard_info() {
  local out_dir="$1" dataset="$2" track="$3" phase="$4"
  GIT_COMMIT_VALUE="$(cslaw_git_commit "${REPO_ROOT}/benchmark/src")" \
    LIGHT_EBM_VALUE="${LIGHT_EBM}" \
    "${CSLAW_PY[@]}" - "${out_dir}/shard_info.json" "${dataset}" "${track}" "${TAG}" "${phase}" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

path, dataset, track, tag, phase = sys.argv[1:6]
target = Path(path)
info = json.loads(target.read_text(encoding="utf-8")) if target.exists() and phase == "finished" else {}
info.update(
    {
        "dataset_id": dataset,
        "track": track,
        "tag": tag,
        "light_ebm": os.environ.get("LIGHT_EBM_VALUE") == "1",
        "model_env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("LEGALBENCH_EBM_") or key.startswith("LEGALBENCH_LIME_")
        },
        "git_commit": os.environ.get("GIT_COMMIT_VALUE") or None,
        "host": socket.gethostname(),
    }
)
info[f"{phase}_utc"] = datetime.now(timezone.utc).isoformat()
target.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

# check_dataset_file CONFIG DATASET TRACK PATH: header check against the v2 taxonomy.
check_dataset_file() {
  local config="$1" dataset="$2" track="$3" path="$4"
  run_cslaw_py -m legalbench_xai check-dataset-file \
    --config "${config}" --dataset "${dataset}" --track "${track}" --path "${path}" \
    | "${CSLAW_PY[@]}" -c '
import json
import sys

report = json.load(sys.stdin)
keys = ("dataset_id", "track", "path", "status", "target_present", "missing_required_columns")
summary = json.dumps({key: report.get(key) for key in keys})
if report.get("status") != "ok" or not report.get("target_present", False):
    print("[cslaw] dataset check failed: " + summary, file=sys.stderr)
    sys.exit(1)
print("[cslaw] dataset check ok: " + summary, file=sys.stderr)
' || cslaw_die "data file ${path} does not match the v2 taxonomy for ${dataset}:${track}"
}

run_shard() {
  local dataset="$1" track="$2"
  local out_dir data_path
  out_dir="$(shard_dir "${dataset}" "${track}")"
  data_path="$(dataset_path "${dataset}")"
  [[ -f "${REPO_ROOT}/${data_path}" ]] || cslaw_die "missing data file ${REPO_ROOT}/${data_path}"
  cslaw_guard_target "${out_dir}"
  mkdir -p "${out_dir}"
  local runtime_config="${out_dir}/runtime_config.json"
  cslaw_log "shard ${dataset}:${track} -> ${out_dir}"
  "${CSLAW_PY[@]}" "${CONFIG_BUILDER}" \
    --cslaw-v2 \
    --datasets "${dataset}" \
    --tracks "${track}" \
    --output "${runtime_config}" >/dev/null
  check_dataset_file "${runtime_config}" "${dataset}" "${track}" "${data_path}"
  warn_ebm "${dataset}"
  write_shard_info "${out_dir}" "${dataset}" "${track}" started
  cslaw_run_benchmark "${REPO_ROOT}/benchmark/src" "${runtime_config}" "${out_dir}"
  cslaw_check_bundle "${out_dir}" 1 v2
  write_shard_info "${out_dir}" "${dataset}" "${track}" finished
}

merge_shards() {
  local inputs=()
  local missing=()
  local present_datasets=()
  local dataset track dir dataset_present
  for dataset in "${DATASETS[@]}"; do
    dataset_present=0
    for track in "${TRACKS[@]}"; do
      dir="$(shard_dir "${dataset}" "${track}")"
      if cslaw_bundle_ready "${dir}"; then
        inputs+=("${dir}")
        dataset_present=1
      else
        missing+=("${dataset}:${track}")
      fi
    done
    if [[ ${dataset_present} -eq 1 ]]; then
      present_datasets+=("${dataset}")
    fi
  done
  if [[ ${#inputs[@]} -eq 0 ]]; then
    cslaw_die "no shard bundles for tag ${TAG} (expected ${OUTPUTS}/cslaw_v2_noncredit_<dataset>_<track>_${TAG})"
  fi
  if [[ ${#missing[@]} -gt 0 ]]; then
    cat >&2 <<WARN
[cslaw] ######################################################################
[cslaw] WARNING: merging ${#inputs[@]} of 6 shards. Missing: ${missing[*]}
[cslaw] The merged bundle covers only the shards present. Run the missing ones
[cslaw] (--shard DATASET:TRACK --tag ${TAG}), then merge again with --force.
[cslaw] ######################################################################
WARN
  fi

  # Every shard of one tag must have finished its checks and use the same EBM/LIME
  # settings; different code versions only warn.
  "${CSLAW_PY[@]}" - "${inputs[@]}" <<'PY' || cslaw_die "the shards differ in settings or did not finish; do not merge them"
import json
import sys
from pathlib import Path

settings = {}
commits = {}
for directory in sys.argv[1:]:
    path = Path(directory) / "shard_info.json"
    if not path.exists():
        print(f"[cslaw] WARNING: {path} is missing; its model settings cannot be checked", file=sys.stderr)
        continue
    info = json.loads(path.read_text(encoding="utf-8"))
    if "finished_utc" not in info:
        print(f"[cslaw] {directory}: the shard did not pass its bundle check (no finished_utc)", file=sys.stderr)
        sys.exit(1)
    key = json.dumps({"light_ebm": info.get("light_ebm"), "model_env": info.get("model_env")}, sort_keys=True)
    settings.setdefault(key, []).append(directory)
    commits.setdefault(str(info.get("git_commit")), []).append(directory)
if len(settings) > 1:
    for key, directories in settings.items():
        print(f"[cslaw] model settings {key}: {directories}", file=sys.stderr)
    sys.exit(1)
if len(commits) > 1:
    for commit, directories in commits.items():
        print(f"[cslaw] WARNING: shards from code version {commit}: {directories}", file=sys.stderr)
PY

  cslaw_guard_target "${MERGED_DIR}"
  cslaw_log "merge_bundles.py: ${#inputs[@]} shards -> ${MERGED_DIR}"
  run_cslaw_py benchmark/cslaw/merge_bundles.py \
    --inputs "${inputs[@]}" \
    --out-dir "${MERGED_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}"
  [[ -f "${MERGED_DIR}/results.json" ]] || cslaw_die "merge_bundles.py did not write ${MERGED_DIR}/results.json"

  # Runtime config of the merged bundle: the shard settings with every merged dataset
  # and both tracks. rebuild-artifacts reads its reporting settings, and the credit
  # analysis step uses it for the v2_legal_functions rebuild.
  "${CSLAW_PY[@]}" "${CONFIG_BUILDER}" \
    --cslaw-v2 \
    --datasets "${present_datasets[@]}" \
    --tracks both \
    --output "${MERGED_DIR}/runtime_config.json" >/dev/null
  "${CSLAW_PY[@]}" - "${MERGED_DIR}/shard_manifest.json" "${TAG}" "${#missing[@]}" \
    ${missing[@]+"${missing[@]}"} -- "${inputs[@]}" <<'PY'
import json
import sys
from pathlib import Path

target, tag, missing_count = sys.argv[1], sys.argv[2], int(sys.argv[3])
rest = sys.argv[4:]
missing = rest[:missing_count]
inputs = rest[missing_count + 1 :]
shards = []
for directory in inputs:
    info_path = Path(directory) / "shard_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else None
    shards.append({"bundle": directory, "shard_info": info})
Path(target).write_text(
    json.dumps({"tag": tag, "shards": shards, "missing_shards": missing}, indent=2) + "\n",
    encoding="utf-8",
)
PY
  cslaw_log "rebuild-artifacts ${MERGED_DIR}"
  run_cslaw_py -m legalbench_xai rebuild-artifacts \
    --config "${MERGED_DIR}/runtime_config.json" \
    --results-json "${MERGED_DIR}/results.json" \
    --output-dir "${MERGED_DIR}" >"${MERGED_DIR}/rebuild_artifacts.json"
  cslaw_check_bundle "${MERGED_DIR}" 1 v2
}

case "${MODE}" in
  shard)
    run_shard "${SHARD_DATASET}" "${SHARD_TRACK}"
    echo
    echo "CS&Law v2 non-credit shard ${SHARD_DATASET}:${SHARD_TRACK}: $(shard_dir "${SHARD_DATASET}" "${SHARD_TRACK}")"
    echo "When all six shards of tag ${TAG} are done, merge them:"
    echo "  bash benchmark/scripts/cluster/run_cslaw_noncredit_v2.sh ${REPO_ROOT} --merge --tag ${TAG}"
    ;;
  all)
    # Preflight: refuse every existing target before the long runs start.
    for dataset in "${DATASETS[@]}"; do
      for track in "${TRACKS[@]}"; do
        cslaw_guard_target "$(shard_dir "${dataset}" "${track}")"
      done
    done
    cslaw_guard_target "${MERGED_DIR}"
    warn_ebm all
    for dataset in "${DATASETS[@]}"; do
      for track in "${TRACKS[@]}"; do
        run_shard "${dataset}" "${track}"
      done
    done
    merge_shards
    echo
    echo "CS&Law v2 non-credit bundle: ${MERGED_DIR}"
    ;;
  merge)
    merge_shards
    echo
    echo "CS&Law v2 non-credit bundle: ${MERGED_DIR}"
    ;;
esac

if [[ "${MODE}" != "shard" ]]; then
  echo "The credit analysis step uses it automatically for the same tag"
  echo "(add --force if cslaw_analysis_${TAG} already has analysis outputs):"
  echo "  bash benchmark/scripts/cluster/run_cslaw_credit.sh ${REPO_ROOT} --steps analysis --tag ${TAG}"
fi

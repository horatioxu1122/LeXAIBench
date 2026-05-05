#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run-legalbench.sh"
CONFIG_BUILDER="${SCRIPT_DIR}/make_cross_domain_runtime_config.py"

PREPARE_FIRST=0
LIGHT_EBM=0
FAST_BUDGET=0
SUMMARY_ONLY=0
FULL_VALIDATE=0
OUTPUT_NAME="cross_domain_strongest_real_full"
HMDA_PATH="benchmark/data/real/hmda.csv"
NY_PATH="benchmark/data/real/ny_courts_pretrial.csv"
FDA_PATH="benchmark/data/real/fda_inspections.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-first)
      PREPARE_FIRST=1
      shift
      ;;
    --light-ebm)
      LIGHT_EBM=1
      shift
      ;;
    --fast-budget)
      FAST_BUDGET=1
      shift
      ;;
    --summary-only)
      SUMMARY_ONLY=1
      shift
      ;;
    --full-validate)
      FULL_VALIDATE=1
      shift
      ;;
    --output-name)
      OUTPUT_NAME="$2"
      shift 2
      ;;
    --hmda-path)
      HMDA_PATH="$2"
      shift 2
      ;;
    --ny-path)
      NY_PATH="$2"
      shift 2
      ;;
    --fda-path)
      FDA_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

if [[ ${PREPARE_FIRST} -eq 1 ]]; then
  echo "Preparing cross-domain datasets..."
  "${RUNNER}" prepare-hmda --input-csv benchmark/data/raw/hmda/hmda_snapshot.csv --output-csv "${HMDA_PATH}"
  "${RUNNER}" prepare-ny-courts-pretrial --input-csv benchmark/data/raw/ny_courts_pretrial/pretrial_release_rows.csv --output-csv "${NY_PATH}"
  "${RUNNER}" prepare-fda-inspections --input-csv benchmark/data/raw/fda_inspections/fda_inspections_extract.csv --output-csv "${FDA_PATH}"
fi

for pair in \
  "hmda ${HMDA_PATH}" \
  "ny_courts_pretrial ${NY_PATH}" \
  "fda_inspections ${FDA_PATH}"
do
  dataset="${pair%% *}"
  path="${pair#* }"
  echo "Checking ${dataset} at ${path}..."
  "${RUNNER}" check-dataset-file --dataset "${dataset}" --track primary --path "${path}" >/dev/null
  "${RUNNER}" check-dataset-file --dataset "${dataset}" --track audit --path "${path}" >/dev/null
done

MAX_INSTANCES=12
BOOTSTRAP_RESAMPLES=1000
if [[ ${FAST_BUDGET} -eq 1 ]]; then
  MAX_INSTANCES=4
  BOOTSTRAP_RESAMPLES=20
  export LEGALBENCH_LIME_NUM_SAMPLES="${LEGALBENCH_LIME_NUM_SAMPLES:-1500}"
fi

if [[ ${LIGHT_EBM} -eq 1 ]]; then
  export LEGALBENCH_EBM_INTERACTIONS="${LEGALBENCH_EBM_INTERACTIONS:-0}"
  export LEGALBENCH_EBM_MAX_BINS="${LEGALBENCH_EBM_MAX_BINS:-32}"
  export LEGALBENCH_EBM_MAX_ROUNDS="${LEGALBENCH_EBM_MAX_ROUNDS:-80}"
fi

RUNTIME_CONFIG="benchmark/outputs/${OUTPUT_NAME}/runtime_config.json"
mkdir -p "benchmark/outputs/${OUTPUT_NAME}"
python "${CONFIG_BUILDER}" \
  --output "${RUNTIME_CONFIG}" \
  --hmda-path "${HMDA_PATH}" \
  --ny-path "${NY_PATH}" \
  --fda-path "${FDA_PATH}" \
  --max-instances "${MAX_INSTANCES}" \
  --bootstrap-resamples "${BOOTSTRAP_RESAMPLES}"

if [[ ${FULL_VALIDATE} -eq 1 ]]; then
  "${RUNNER}" validate --config "${RUNTIME_CONFIG}"
fi

export LEGALBENCH_PROGRESS=1
export LEGALBENCH_PROGRESS_LOG="${REPO_ROOT}/benchmark/outputs/${OUTPUT_NAME}/progress.jsonl"

RUN_ARGS=(
  run-demo-benchmark
  --config "${RUNTIME_CONFIG}"
  --output-dir "benchmark/outputs/${OUTPUT_NAME}"
  --write
)
if [[ ${SUMMARY_ONLY} -eq 1 ]]; then
  RUN_ARGS+=(--summary-only)
fi

echo "Launching ${OUTPUT_NAME}..."
"${RUNNER}" "${RUN_ARGS[@]}"

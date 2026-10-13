#!/usr/bin/env bash
# Shared helpers for the CS&Law re-run scripts. Source this file; do not execute it.
#
#   source "${REPO_ROOT}/benchmark/scripts/cluster/cslaw_lib.sh"
#
# Used by:
#   benchmark/scripts/cluster/run_cslaw_credit.sh
#   benchmark/scripts/cluster/run_cslaw_noncredit_v2.sh
#   benchmark/scripts/cluster/run_cslaw_cross_domain_native.sh   (round 1, superseded)
#   benchmark/scripts/run_cslaw_offline_analyses.sh
#
# Safety rules enforced here:
#   * every output directory these scripts create is named cslaw_* under
#     benchmark/outputs, so a published bundle can never be a write target;
#   * an existing target directory is refused unless CSLAW_FORCE=1 (--force);
#   * after each benchmark run, cslaw_check_bundle fails loudly if any model or
#     explainer silently fell back to a stub backend ("heuristic*" model or
#     "demo_generator" explainer), or if the cslaw instrumentation recorded errors;
#     for v2 runs it also checks that the v2 corrections actually took effect, and
#     for runs without a cslaw block that no v2 key leaked into the v1 output.

CSLAW_BASELINE_COMMIT="fb5dadc"
CSLAW_PUBLISHED_CREDIT_BUNDLE_NAME="four_dataset_real_highbudget_heldout_refresh"
CSLAW_PUBLISHED_NONCREDIT_BUNDLE_NAME="cross_domain_strongest_real_full_cluster"
CSLAW_PUBLISHED_CREDIT_CONFIG="benchmark/configs/experiment_four_dataset_real_highbudget.json"
CSLAW_RESCORE_VARIANTS=(css_no_dsv comp_no_recourse_simplicity fid_no_quality all_neutral)
CSLAW_DSV_SWEEP="0,0.05,0.10,0.15,0.20,0.25,0.30,0.35"

cslaw_log() {
  printf '[cslaw %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

cslaw_die() {
  printf '\n[cslaw] ERROR: %s\n\n' "$*" >&2
  exit 1
}

cslaw_validate_tag() {
  local tag="$1"
  if [[ ! "${tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    cslaw_die "invalid --tag '${tag}' (allowed: letters, digits, '.', '_', '-')"
  fi
}

# Same thread pinning as benchmark/scripts/run-legalbench.sh, so the CS&Law runs
# use the same numerical environment as the published runs.
cslaw_export_thread_env() {
  local repo_root="$1"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-${repo_root}/.uv-cache}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
}

# Sets the CSLAW_PY array to the project interpreter. Prefers the uv-managed
# .venv directly; falls back to "uv run --no-sync python" when the environment
# lives elsewhere (e.g. UV_PROJECT_ENVIRONMENT).
cslaw_resolve_python() {
  local repo_root="$1"
  if [[ -x "${repo_root}/.venv/bin/python" ]]; then
    CSLAW_PY=("${repo_root}/.venv/bin/python")
  elif command -v uv >/dev/null 2>&1; then
    CSLAW_PY=(uv run --project "${repo_root}" --no-sync python)
  else
    cslaw_die "no project Python found; run 'uv sync --extra benchmark-empirical' in ${repo_root} first"
  fi
  cslaw_log "python: ${CSLAW_PY[*]}"
}

# Refuse to write anywhere except a cslaw_* directory (name check only).
cslaw_guard_name() {
  local target="$1"
  local base
  base="$(basename "${target}")"
  case "${base}" in
    cslaw_*) ;;
    *) cslaw_die "refusing to write to ${target}: CS&Law outputs must be cslaw_* directories" ;;
  esac
  case "${base}" in
    "${CSLAW_PUBLISHED_CREDIT_BUNDLE_NAME}" | "${CSLAW_PUBLISHED_NONCREDIT_BUNDLE_NAME}")
      cslaw_die "refusing to write into the published bundle ${target}"
      ;;
  esac
}

# cslaw_guard_paths_absent PATH...: refuse if any PATH exists, unless CSLAW_FORCE=1.
# For steps that share one cslaw_* directory and each own some entries in it.
cslaw_guard_paths_absent() {
  local path
  for path in "$@"; do
    if [[ -e "${path}" ]]; then
      if [[ "${CSLAW_FORCE:-0}" -eq 1 ]]; then
        cslaw_log "WARNING: ${path} already exists; --force given, it will be overwritten"
      else
        cslaw_die "${path} already exists. Use a new --tag, or pass --force to write into it."
      fi
    fi
  done
}

# Refuse to write anywhere except a cslaw_* directory, and refuse an existing
# directory unless CSLAW_FORCE=1.
cslaw_guard_target() {
  local target="$1"
  cslaw_guard_name "${target}"
  if [[ -e "${target}" ]]; then
    if [[ "${CSLAW_FORCE:-0}" -eq 1 ]]; then
      cslaw_log "WARNING: ${target} already exists; --force given, files in it will be overwritten"
    else
      cslaw_die "${target} already exists. Use a new --tag, or pass --force to write into it."
    fi
  fi
}

cslaw_bundle_ready() {
  [[ -f "$1/results.json" ]]
}

# Fail unless "import legalbench_xai" resolves inside the given src directory,
# so an A/B run can never silently test the wrong code.
cslaw_assert_code_location() {
  local src_dir="$1"
  local resolved
  resolved="$(PYTHONPATH="${src_dir}" "${CSLAW_PY[@]}" -c 'import legalbench_xai; print(legalbench_xai.__file__)')"
  case "${resolved}" in
    "${src_dir}"/*) cslaw_log "code under test: ${resolved}" ;;
    *) cslaw_die "legalbench_xai resolved to ${resolved}, expected a file under ${src_dir}" ;;
  esac
}

# cslaw_git_commit SRC_DIR: commit of the checkout that holds SRC_DIR, with a
# "-dirty" suffix when benchmark/src, benchmark/configs or benchmark/metadata differ
# from it (untracked files included). A GIT_COMMIT already set in the environment
# wins (e.g. an rsync'd tree without .git). Prints nothing outside a git checkout.
cslaw_git_commit() {
  local src_dir="$1"
  if [[ -n "${GIT_COMMIT:-}" ]]; then
    printf '%s' "${GIT_COMMIT}"
    return 0
  fi
  local top commit
  top="$(git -C "${src_dir}" rev-parse --show-toplevel 2>/dev/null)" || return 0
  commit="$(git -C "${top}" rev-parse HEAD 2>/dev/null)" || return 0
  if [[ -n "$(git -C "${top}" status --porcelain -- benchmark/src benchmark/configs benchmark/metadata 2>/dev/null)" ]]; then
    commit="${commit}-dirty"
  fi
  printf '%s' "${commit}"
}

# cslaw_run_benchmark SRC_DIR CONFIG OUT_DIR
# Runs run-demo-benchmark with the code in SRC_DIR. CONFIG and OUT_DIR must be
# absolute. Progress goes to OUT_DIR/progress.jsonl, compact stdout to
# OUT_DIR/run_summary.json. GIT_COMMIT is set for the run, so a v2 bundle records
# the code version in payload["cslaw"]["provenance"]["git_commit"].
cslaw_run_benchmark() {
  local src_dir="$1"
  local config="$2"
  local out_dir="$3"
  [[ -f "${config}" ]] || cslaw_die "config not found: ${config}"
  mkdir -p "${out_dir}"
  local progress_log="${out_dir}/progress.jsonl"
  if [[ -e "${progress_log}" ]]; then
    mv "${progress_log}" "${progress_log}.$(date -u +%Y%m%dT%H%M%SZ).bak"
  fi
  cslaw_assert_code_location "${src_dir}"
  local git_commit
  git_commit="$(cslaw_git_commit "${src_dir}")"
  cslaw_log "run-demo-benchmark --config ${config} --output-dir ${out_dir} (git ${git_commit:-unknown})"
  local start_epoch
  start_epoch="$(date +%s)"
  GIT_COMMIT="${git_commit}" \
    LEGALBENCH_PROGRESS=1 \
    LEGALBENCH_PROGRESS_LOG="${progress_log}" \
    PYTHONPATH="${src_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${CSLAW_PY[@]}" -m legalbench_xai run-demo-benchmark \
    --config "${config}" \
    --output-dir "${out_dir}" \
    --write \
    --summary-only | tee "${out_dir}/run_summary.json"
  cslaw_log "finished in $(( $(date +%s) - start_epoch )) s: ${out_dir}"
}

# cslaw_check_bundle BUNDLE_DIR EXPECT_CSLAW(0|1) [EXPECT_CORRECTIONS(v2)]
# Fails loudly (exit 3) on any stub backend at summary or instance level, and,
# when EXPECT_CSLAW=1, on a missing payload["cslaw"], a non-native contribution
# source, or recorded cslaw errors. When EXPECT_CSLAW=0 (no cslaw block), it also
# fails if payload["cslaw"], any per-instance eval_row_index /
# canonical_explanation / native_contributions field, or any v2-only key (sampling
# selection/seed, CSS dsv_* sub-components, faithfulness_quality_weight) is present.
# With EXPECT_CORRECTIONS=v2 it also fails unless payload["cslaw"] records
# corrections_version "v2", data provenance with SHA-256 hashes and the v2 taxonomy
# and registry, stratified_random sampling in every group, the persisted v2 score
# keys in every instance, faithfulness scored on the common native reference
# (evidence_source "model_native_common_reference") in every instance, and a
# cf_reason_source in every persisted DiCE/Wachter canonical explanation.
cslaw_check_bundle() {
  local bundle="$1"
  local expect_cslaw="${2:-0}"
  local expect_corrections="${3:-}"
  local results="${bundle}/results.json"
  [[ -f "${results}" ]] || cslaw_die "missing ${results}"
  # Exit 3 = invalid bundle (the check failed); any other non-zero status = the
  # check itself could not run (infrastructure), reported through cslaw_die (exit 1).
  local check_status=0
  "${CSLAW_PY[@]}" - "${results}" "${expect_cslaw}" "${expect_corrections}" <<'PY' || check_status=$?
import json
import sys

path, expect_cslaw = sys.argv[1], sys.argv[2] == "1"
expect_corrections = sys.argv[3] if len(sys.argv) > 3 else ""
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)

# Keys only a v2 run may write (corrections.py: persist_score_subcomponents and
# sampling_selection); a run without a cslaw block must not have any of them.
V2_CSS_KEYS = ("dsv_directionality", "dsv_threshold_visibility", "dsv_flip_validity")
V2_FAITHFULNESS_KEY = "faithfulness_quality_weight"
V2_SAMPLING_KEYS = ("selection", "seed")
# faithfulness evidence source under common_faithfulness_reference
# (explainers/registry.py COMMON_REFERENCE_SOURCE).
COMMON_REFERENCE_SOURCE = "model_native_common_reference"


def is_stub(implementation, backend_kind=None):
    # Stub selections carry backend "stub" with implementation "heuristic_model"
    # (models) or "demo_generator" (explainers).
    text = str(implementation or "")
    return backend_kind == "stub" or text == "demo_generator" or text.startswith("heuristic")


problems = []
usage = (payload.get("overview") or {}).get("backend_usage") or {}
for kind in ("models", "explainers"):
    for implementation, count in sorted((usage.get(kind) or {}).items()):
        if is_stub(implementation):
            problems.append(f"overview.backend_usage.{kind}: '{implementation}' in {count} groups")

stub_instances = 0
for row in payload.get("summary", []):
    group = f"{row.get('dataset_id')}|{row.get('track')}|{row.get('model_id')}|{row.get('explainer_id')}"
    for kind in ("model_backend", "explainer_backend"):
        backend = row.get(kind) or {}
        if is_stub(backend.get("implementation"), backend.get("backend")):
            problems.append(f"{group}: {kind}={backend.get('implementation')} ({backend.get('reason')})")
    for instance in row.get("instance_results", []) or []:
        for kind in ("model_backend", "explainer_backend"):
            backend = instance.get(kind) or {}
            if is_stub(backend.get("implementation"), backend.get("backend")):
                stub_instances += 1
                if stub_instances <= 20:
                    problems.append(
                        f"{instance.get('instance_id')} [{row.get('explainer_id')}]: "
                        f"{kind}={backend.get('implementation')} ({backend.get('reason')})"
                    )
if stub_instances > 20:
    problems.append(f"... {stub_instances} stub instance backends in total")

if expect_cslaw:
    block = payload.get("cslaw")
    if not isinstance(block, dict):
        problems.append("payload['cslaw'] is missing although the config enables cslaw features")
    else:
        groups = block.get("groups") or {}
        if not groups:
            problems.append("payload['cslaw']['groups'] is empty")
        for key, group in sorted(groups.items()):
            source = group.get("native_contribution_source")
            if source in (None, "unavailable", "heuristic_stub"):
                problems.append(f"cslaw group {key}: native_contribution_source={source}")
            for message in group.get("errors") or []:
                problems.append(f"cslaw group {key}: {message}")
            shared = group.get("shared_sampling")
            if isinstance(shared, dict) and shared.get("shared_fallback"):
                print(
                    f"[cslaw] WARNING: {key} fell back to per-model sampling "
                    f"({shared.get('fallback_reason')})",
                    file=sys.stderr,
                )
else:
    # EXPECT_CSLAW=0 is the no-cslaw-block contract (A): no new payload keys.
    # compare_bundles.py ignores these keys by design, so assert their absence here.
    if "cslaw" in payload:
        problems.append("payload['cslaw'] present although the config has no cslaw block")
    leaked_instances = 0
    for row in payload.get("summary", []):
        for instance in row.get("instance_results", []) or []:
            leaked = [
                key
                for key in ("eval_row_index", "canonical_explanation", "native_contributions")
                if key in instance
            ]
            if leaked:
                leaked_instances += 1
                if leaked_instances <= 20:
                    problems.append(
                        f"{instance.get('instance_id')} [{row.get('explainer_id')}]: "
                        f"cslaw instance fields {leaked} present although the config has no cslaw block"
                    )
    if leaked_instances > 20:
        problems.append(f"... {leaked_instances} instances with cslaw instance fields in total")
    leaked_v2 = 0
    for row in payload.get("summary", []):
        group = f"{row.get('dataset_id')}|{row.get('track')}|{row.get('model_id')}|{row.get('explainer_id')}"
        found = [key for key in V2_SAMPLING_KEYS if key in (row.get("sampling") or {})]
        for instance in row.get("instance_results", []) or []:
            scores = instance.get("scores") or {}
            found += [key for key in V2_CSS_KEYS if key in (scores.get("contestability_support_score") or {})]
            if V2_FAITHFULNESS_KEY in (scores.get("faithfulness_score") or {}):
                found.append(V2_FAITHFULNESS_KEY)
        if found:
            leaked_v2 += 1
            if leaked_v2 <= 20:
                problems.append(f"{group}: v2-only keys {sorted(set(found))} in a run without a cslaw block")

if expect_corrections == "v2":
    block = payload.get("cslaw") if isinstance(payload.get("cslaw"), dict) else {}
    if block.get("corrections_version") != "v2":
        problems.append(
            f"payload['cslaw']['corrections_version']={block.get('corrections_version')!r}, expected 'v2'"
        )
    provenance = block.get("provenance")
    if not isinstance(provenance, dict):
        problems.append("payload['cslaw']['provenance'] is missing (record_data_provenance)")
    else:
        datasets = provenance.get("datasets") or {}
        if not datasets:
            problems.append("provenance.datasets is empty")
        for dataset_id, entry in sorted(datasets.items()):
            if not (entry or {}).get("sha256"):
                problems.append(f"provenance.datasets.{dataset_id}: no sha256 ({(entry or {}).get('error')})")
        metadata = provenance.get("metadata") or {}
        for key, suffix in (("feature_taxonomy", "feature_taxonomy_v2.json"), ("dataset_registry", "dataset_registry_v2.json")):
            entry = metadata.get(key) or {}
            if not entry.get("sha256"):
                problems.append(f"provenance.metadata.{key}: no sha256 ({entry.get('error')})")
            if not str(entry.get("path", "")).endswith(suffix):
                problems.append(f"provenance.metadata.{key}.path={entry.get('path')!r}, expected the v2 file {suffix}")
        if not provenance.get("git_commit"):
            print("[cslaw] WARNING: provenance.git_commit is empty (not a git checkout?)", file=sys.stderr)
        elif str(provenance.get("git_commit")).endswith("-dirty"):
            print(
                f"[cslaw] WARNING: run used uncommitted code/configs/metadata ({provenance.get('git_commit')})",
                file=sys.stderr,
            )
    missing_v2 = 0
    not_common_reference = 0
    cf_without_source = 0
    cf_not_persisted = 0
    for row in payload.get("summary", []):
        group = f"{row.get('dataset_id')}|{row.get('track')}|{row.get('model_id')}|{row.get('explainer_id')}"
        selection = (row.get("sampling") or {}).get("selection")
        if selection != "stratified_random":
            problems.append(f"{group}: sampling.selection={selection!r}, expected 'stratified_random'")
        for instance in row.get("instance_results", []) or []:
            scores = instance.get("scores") or {}
            css = scores.get("contestability_support_score") or {}
            faithfulness = scores.get("faithfulness_score") or {}
            missing = [key for key in V2_CSS_KEYS if key not in css]
            if faithfulness.get(V2_FAITHFULNESS_KEY) != 0.0:
                missing.append(f"{V2_FAITHFULNESS_KEY}=0.0")
            if missing:
                missing_v2 += 1
                if missing_v2 <= 20:
                    problems.append(f"{instance.get('instance_id')} [{row.get('explainer_id')}]: missing v2 score keys {missing}")
            # common_faithfulness_reference: every explainer (null baselines and stub
            # fallbacks included) is scored against the model's native contributions.
            # Another source means model.corrections never reached the explainer registry.
            if faithfulness.get("evidence_source") != COMMON_REFERENCE_SOURCE:
                not_common_reference += 1
                if not_common_reference <= 20:
                    problems.append(
                        f"{instance.get('instance_id')} [{row.get('explainer_id')}]: faithfulness "
                        f"evidence_source={faithfulness.get('evidence_source')!r}, expected "
                        f"{COMMON_REFERENCE_SOURCE!r}"
                    )
            if row.get("explainer_id") in ("dice", "wachter"):
                canonical = instance.get("canonical_explanation")
                if not isinstance(canonical, dict):
                    cf_not_persisted += 1
                elif "cf_reason_source" not in (canonical.get("metadata") or {}):
                    # counterfactual_reasons_from_changes always records the source (real
                    # and stub generators), so its absence is a wiring failure.
                    cf_without_source += 1
                    if cf_without_source <= 20:
                        problems.append(
                            f"{instance.get('instance_id')} [{row.get('explainer_id')}]: "
                            "canonical_explanation.metadata.cf_reason_source is missing"
                        )
    if missing_v2 > 20:
        problems.append(f"... {missing_v2} instances without the v2 score keys in total")
    if not_common_reference > 20:
        problems.append(f"... {not_common_reference} instances not scored on the common reference in total")
    if cf_without_source > 20:
        problems.append(f"... {cf_without_source} dice/wachter instances without cf_reason_source in total")
    if cf_not_persisted:
        print(
            f"[cslaw] WARNING: {cf_not_persisted} dice/wachter instances have no persisted "
            "canonical_explanation (persist_canonical_explanations is off), so "
            "cf_reason_source cannot be checked",
            file=sys.stderr,
        )

print(f"[cslaw] backend usage for {path}: {json.dumps(usage, sort_keys=True)}", file=sys.stderr)
if problems:
    print("", file=sys.stderr)
    print("[cslaw] ################################################################", file=sys.stderr)
    print(f"[cslaw] FATAL: stub fallback or cslaw errors in {path}", file=sys.stderr)
    for problem in problems:
        print(f"[cslaw]   - {problem}", file=sys.stderr)
    print("[cslaw] ################################################################", file=sys.stderr)
    sys.exit(3)
print(f"[cslaw] backend check passed: {path}", file=sys.stderr)
PY
  if (( check_status == 3 )); then
    printf '\n[cslaw] ERROR: %s\n\n' \
      "backend check failed for ${bundle} (see messages above); do not use this bundle" >&2
    exit 3
  elif (( check_status != 0 )); then
    cslaw_die "backend check could not run for ${bundle} (status ${check_status})"
  fi
}

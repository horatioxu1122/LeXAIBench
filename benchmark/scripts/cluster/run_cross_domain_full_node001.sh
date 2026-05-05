#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$HOME/explainableAI-CS}"

module load python3

cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --extra benchmark-empirical

bash benchmark/scripts/run-cross-domain-strongest-real.sh \
  --output-name cross_domain_strongest_real_full_cluster \
  --full-validate

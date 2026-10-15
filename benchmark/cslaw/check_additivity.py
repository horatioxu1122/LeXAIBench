"""Check that persisted native contributions add up to each model's score.

For every explained applicant, a bundle produced with
``cslaw.persist_native_contributions`` stores the model's per-feature
contributions in log odds of the adverse outcome, the constant (bias), and the
model's predicted probability of the adverse outcome. Additivity requires

    sum_j c_j(x) + bias == logit(prediction_score).

This script reports the largest and mean absolute gap per dataset, track and
model. It reads bundles only; it fits no model.

Usage:
    python benchmark/cslaw/check_additivity.py --bundle DIR [--bundle DIR ...] --out-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PROBABILITY_CLIP = 1e-12


def logit(p: float) -> float:
    p = min(max(float(p), PROBABILITY_CLIP), 1.0 - PROBABILITY_CLIP)
    return math.log(p / (1.0 - p))


def additivity_rows(payload: dict) -> list[dict]:
    gaps: dict[tuple, list[float]] = defaultdict(list)
    sources: dict[tuple, set] = defaultdict(set)
    probability_extreme: dict[tuple, int] = defaultdict(int)
    for row, instance in common.iter_instances(payload):
        native = instance.get("native_contributions")
        if not native or instance.get("prediction_score") is None:
            continue
        key = (row["dataset_id"], row["track"], row["model_id"])
        values = native.get("values") or {}
        total = sum(float(v) for v in values.values()) + float(native.get("bias") or 0.0)
        score = float(instance["prediction_score"])
        if score <= 1e-9 or score >= 1 - 1e-9:
            probability_extreme[key] += 1
        gaps[key].append(abs(total - logit(score)))
        sources[key].add(native.get("source"))
    rows = []
    for key in sorted(gaps):
        values = gaps[key]
        rows.append(
            {
                "dataset_id": key[0],
                "track": key[1],
                "model_id": key[2],
                "source": "|".join(sorted(s for s in sources[key] if s)),
                "n_explanations": len(values),
                "max_abs_gap": max(values),
                "mean_abs_gap": sum(values) / len(values),
                "n_probability_at_clip": probability_extreme[key],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for bundle in args.bundle:
        path = Path(bundle)
        payload = common.load_results(path / "results.json" if path.is_dir() else path)
        for row in additivity_rows(payload):
            all_rows.append({"bundle": path.name, **row})
    if not all_rows:
        sys.exit("no persisted native contributions found in the given bundles")

    csv_path = out_dir / "additivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_model[row["model_id"]].append(row)
    lines = [
        "# Additivity of native contributions",
        "",
        "Gap = |sum of contributions + bias - logit(predicted probability)|, in log odds.",
        "",
        "| model | source | explanations | max gap | mean gap |",
        "|---|---|---:|---:|---:|",
    ]
    for model_id in sorted(by_model):
        rows = by_model[model_id]
        n = sum(r["n_explanations"] for r in rows)
        max_gap = max(r["max_abs_gap"] for r in rows)
        mean_gap = sum(r["mean_abs_gap"] * r["n_explanations"] for r in rows) / n
        source = "|".join(sorted({r["source"] for r in rows}))
        lines.append(f"| {model_id} | {source} | {n} | {max_gap:.2e} | {mean_gap:.2e} |")
    (out_dir / "additivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {csv_path} and {out_dir / 'additivity.md'}")


if __name__ == "__main__":
    main()

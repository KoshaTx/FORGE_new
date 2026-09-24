"""Separate untreated correspondence failures from failures of attempted graph completion."""

import json
from collections import Counter
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    inputs = [
        {"path": str(Path(__file__).relative_to(ROOT)), "sha256": str(sha256_file(Path(__file__)))}
    ]
    populations = {}
    for label in ("discovery", "confirmation_1", "confirmation_2"):
        path = ROOT / f"results/phase1/combinatorial_graph_reuse_{label}_v1/result.json"
        parent = json.loads(path.read_text())
        inputs.append({"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))})
        values = []
        for name in ("layouts.json", "reuse_plans.json"):
            pin = parent["artifacts"][name]
            values.append(json.loads(resolve_pin(pin, ROOT, label=name).read_text()))
            inputs.append(pin)
        counts = Counter(
            (row["family"], row["depth"], plan["status"]) for row, plan in zip(*values, strict=True)
        )
        populations[label] = [
            {"family": family, "depth": depth, "qualification_status": status, "attempts": count}
            for (family, depth, status), count in sorted(counts.items())
        ]
    result = {
        "schema_version": "forge.combinatorial_graph_reuse_scope.v1",
        "inputs": inputs,
        "populations": populations,
        "interpretation": "Unqualified layouts receive no graph-copy proposals. Lack of improvement in an entirely unqualified stratum does not test the effect of correctly enforcing reuse in that stratum.",
    }
    (OUT / "qualification_scope.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

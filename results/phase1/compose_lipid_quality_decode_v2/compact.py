"""Lightweight selector ledgers with immutable full construction provenance."""

import json
from pathlib import Path

from forge.core.io import write_json
from forge.corpus.compose_lipid_source_view import pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def compact_row(row, source):
    branches = {}
    for branch, value in row["branches"].items():
        branches[branch] = dict(
            raw=value["raw"],
            proposals=[{k: p[k] for k in ("kind", "smiles", "check")} for p in value["proposals"]],
            costs=value["costs"],
            construction_provenance=dict(source=source, index=row["index"], branch=branch),
        )
    return {**{k: row[k] for k in ("index", "family", "draw")}, "branches": branches}


def main():
    folder = HERE / "first_draw"
    result = json.loads((folder / "result.json").read_text())
    rows = []
    for shard in result["shards"]:
        source = shard["assessed"]
        for row in json.loads((ROOT / source["path"]).read_text()):
            rows.append(compact_row(row, source))
    write_json(folder / "compact_attempts.json", rows)
    write_json(
        folder / "compact_receipt.json",
        dict(
            inputs=dict(result=pin(ROOT, folder / "result.json")),
            output=pin(ROOT, folder / "compact_attempts.json"),
            implementation=pin(ROOT, Path(__file__)),
            omitted_fields="Dense proposal graph copies and repeated construction graphs remain in pinned full shards",
            request_count=len(rows),
            selection_inputs_unchanged=True,
        ),
    )


if __name__ == "__main__":
    main()

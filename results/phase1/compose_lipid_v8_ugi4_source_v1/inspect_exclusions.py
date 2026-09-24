"""Retain the mechanical failure stage for every excluded TRAIN Ugi-4 product."""

import gzip
import json
import sqlite3
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rdkit import Chem, rdBase

from forge.assembly.families import RegistryAssemblyAdapter
from forge.assembly.program import repair_template_hydrogens
from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def pin(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(path))}


def main():
    receipt_path = ROOT / "results/phase1/compose_lipid_v8_ugi4_program_v1/result.json"
    receipt = json.loads(receipt_path.read_text())
    config = json.loads(resolve_pin(receipt["config"], ROOT, label="config").read_text())
    registry = resolve_pin(config["inputs"]["registry"], ROOT, label="registry")
    adapter = RegistryAssemblyAdapter.from_registry(
        registry,
        reaction_id=config["reaction_id"],
        expected_sha256=config["inputs"]["registry"]["sha256"],
    )
    ledger = resolve_pin(receipt["artifacts"]["programs.jsonl.gz"], ROOT, label="programs")
    imported_path = ROOT / "results/phase1/compose_lipid_v8_import_v1/result.json"
    imported = json.loads(imported_path.read_text())
    database = resolve_pin(imported["artifacts"]["corpus.sqlite"], ROOT, label="corpus")
    rows = []
    with (
        gzip.open(ledger, "rt") as stream,
        sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as db,
    ):
        for row in map(json.loads, stream):
            if row["computed_consistency_pass"]:
                continue
            # This query cannot access provider evaluation molecular graphs.
            (target,) = db.execute(
                "SELECT t.constitution FROM targets t JOIN assignments a USING(target_id) WHERE t.target_id=? AND a.forge_split='train' AND a.family=?",
                (row["target_id"], config["family"]),
            ).fetchone()
            with rdBase.BlockLogs():
                raw = adapter.reaction.reverse.RunReactants(
                    (Chem.MolFromSmiles(target),), maxProducts=config["maximum_outcomes"]
                )
            candidates = []
            for fragments in raw:
                repaired = [repair_template_hydrogens(fragment) for fragment in fragments]
                if any(fragment is None for fragment in repaired):
                    candidates.append({"repair_failed": True})
                    continue
                components = dict(
                    zip(adapter.roles, [fragment[0] for fragment in repaired], strict=True)
                )
                candidates.append(
                    {
                        "components": components,
                        "role_assessments": [asdict(a) for a in adapter.assess_roles(components)],
                    }
                )
            rows.append(
                {
                    "target_id": row["target_id"],
                    "status": row["status"],
                    "raw_inverse_count": len(raw),
                    "saturated": len(raw) >= config["maximum_outcomes"],
                    "candidates": candidates,
                }
            )
    counts = Counter()
    for row in rows:
        reasons = set()
        if not row["raw_inverse_count"]:
            reasons.add("no_raw_inverse")
        for candidate in row["candidates"]:
            if candidate.get("repair_failed"):
                reasons.add("fragment_repair_failed")
            else:
                reasons.update(
                    f"role_rejected:{a['role']}:count={a['handle_count']}"
                    for a in candidate["role_assessments"]
                    if not a["qualified"]
                )
        counts.update(reasons or {"forward_replay_rejected"})
    output = HERE / "excluded_programs.jsonl.gz"
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
    result = {
        "schema_version": "forge.ugi4_exclusion_diagnostic.v1",
        "inputs": {
            "program_result": pin(receipt_path),
            "import_result": pin(imported_path),
            "corpus": pin(database),
            "programs": pin(ledger),
            "registry": pin(registry),
        },
        "implementation": {
            **receipt["implementation"],
            "diagnostic": pin(Path(__file__).resolve()),
        },
        "artifacts": {"rows": pin(output)},
        "seed": 0,
        "rows": len(rows),
        "reasons_per_target": dict(sorted(counts.items())),
        "training_calls": 0,
        "source_flags_changed": False,
        "provider_evaluation_graphs_parsed": False,
        "exclusions_relaxed": False,
    }
    (HERE / "exclusion_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

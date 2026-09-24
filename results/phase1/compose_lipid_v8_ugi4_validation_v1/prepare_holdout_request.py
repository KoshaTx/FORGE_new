"""Export an unsent, metadata-only request for unresolved precursor-holdout identities."""

import gzip
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from forge.assembly.compose_lipid import ComposeLipidError, role_metadata  # noqa: E402
from forge.core.hashing import resolve_pin, sha256_file  # noqa: E402


def pin(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(path))}


def main():
    audit_path = ROOT / "results/phase1/compose_lipid_v8_precursor_audit_v2/result.json"
    audit = json.loads(audit_path.read_text())
    protected = json.loads(
        resolve_pin(audit["inputs"]["protection_result"], ROOT, label="protection").read_text()
    )
    import_path = resolve_pin(protected["inputs"]["import_result"], ROOT, label="import")
    imported = json.loads(import_path.read_text())
    db_path = resolve_pin(imported["artifacts"]["corpus.sqlite"], ROOT, label="corpus")
    catalogue_path = resolve_pin(
        imported["artifacts"]["program_catalogue.json"], ROOT, label="catalogue"
    )
    catalogue = json.loads(catalogue_path.read_text())
    panel_path = resolve_pin(audit["artifacts"]["panels.jsonl.gz"], ROOT, label="panels")
    with gzip.open(panel_path, "rt") as stream:
        panels = [json.loads(line) for line in stream]
    selected = {
        row["target_id"]: row
        for row in panels
        if "unseen_precursor_identity" in row["test_panels"]
        and not row["all_roles_resolved_from_preparation"]
    }
    expected = audit["summary"]["panels"]["unseen_precursor_identity"]
    assert len(selected) == expected["rows"] - expected["all_roles_resolved_from_preparation"]
    rows = []
    labels = set()
    counts = Counter()
    with sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True) as db:
        # The SQL projection deliberately excludes molecular graphs and experimental outcomes.
        records = db.execute(
            "SELECT a.target_id,a.family,a.forge_split,"
            "json_extract(t.payload,'$.primary_metadata') "
            "FROM assignments a JOIN targets t USING(target_id) "
            "WHERE a.forge_split IN ('calibration','heldout') ORDER BY a.target_id"
        )
        for target, family, split, raw in records:
            if target not in selected:
                continue
            audited = selected[target]
            assert (family, split) == (audited["family"], audited["split"])
            source = {"target_id": target, "family": family, "primary_metadata": json.loads(raw)}
            try:
                metadata = role_metadata(source, catalogue[family])
            except ComposeLipidError as exc:
                assert str(exc) == audited["malformed_metadata"]
                metadata = {
                    "roles": {},
                    "missing_role_metadata": sorted(
                        catalogue[family]["source_definition"]["roles"]
                    ),
                }
            malformed = bool(metadata["missing_role_metadata"])
            assert malformed == (audited["malformed_metadata"] is not None)
            requested = {}
            for role in audited["unresolved_roles"]:
                value = metadata["roles"][role]["metadata"]
                requested[role] = value
                labels.add((family, role, json.dumps(value, sort_keys=True)))
            if not malformed:
                assert requested
            rows.append(
                {
                    "target_id": target,
                    "family": family,
                    "source_split": split,
                    "unresolved_role_labels": requested,
                    "missing_role_metadata": metadata["missing_role_metadata"],
                    "metadata_error": audited["malformed_metadata"],
                    "request": (
                        "Supply complete role labels and exact constitutional component structures."
                        if malformed
                        else "Supply exact constitutional component structures for these role labels."
                    ),
                    "holdout_qualified": False,
                }
            )
            counts[family] += 1
    assert len(rows) == len(selected)
    assert len({row["target_id"] for row in rows}) == len(rows)
    assert sum(bool(r["missing_role_metadata"]) for r in rows) == expected["malformed_metadata"]
    output = HERE / "unresolved-holdout-request.jsonl.gz"
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    output.write_bytes(gzip.compress(raw, mtime=0))
    receipt = {
        "schema_version": "forge.unresolved_precursor_request.v1",
        "status": "prepared_not_sent",
        "inputs": {
            "audit": pin(audit_path),
            "import": pin(import_path),
            "corpus": pin(db_path),
            "catalogue": pin(catalogue_path),
            "panels": pin(panel_path),
        },
        "implementation": {
            "export": pin(Path(__file__).resolve()),
            "role_metadata": pin(ROOT / "forge/assembly/compose_lipid.py"),
        },
        "artifacts": {"request": pin(output)},
        "rows": len(rows),
        "malformed_metadata_rows": expected["malformed_metadata"],
        "distinct_unresolved_family_role_labels": len(labels),
        "rows_by_family": dict(sorted(counts.items())),
        "provider_evaluation_graphs_parsed": False,
        "training_calls": 0,
        "source_splits_changed": False,
        "holdout_qualified": False,
        "seed": 0,
        "random_sampling_used": False,
        "additional_required_evidence": "The source split-audit artifact and label-identity scope.",
    }
    (HERE / "holdout-request.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        {
            k: receipt[k]
            for k in ("rows", "malformed_metadata_rows", "distinct_unresolved_family_role_labels")
        }
    )


if __name__ == "__main__":
    main()

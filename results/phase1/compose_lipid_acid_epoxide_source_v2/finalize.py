"""Publish the independently reconciled acid/epoxide readiness increment."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/phase1/compose_lipid_training_readiness_v13"


def verify():
    result = json.loads((OUTPUT / "readiness.json").read_text())
    for name, value in result["inputs"].items():
        resolve_pin(value, ROOT, label=name)
    resolve_pin(result["implementation"], ROOT, label="closeout")
    index = json.loads((HERE / "reconciled/result.json").read_text())
    resolve_pin(index["implementation"], ROOT, label="reconciliation")
    resolve_pin(index["artifact"], ROOT, label="evidence index")
    replay = json.loads((HERE / "replay/result.json").read_text())
    request = json.loads(resolve_pin(replay["request"], ROOT, label="replay request").read_text())
    for name, value in {**request["inputs"], **request["implementation"]}.items():
        resolve_pin(value, ROOT, label=name)
    for value in replay["shards"]:
        shard = json.loads(resolve_pin(value, ROOT, label="shard").read_text())
        resolve_pin(shard["artifact"], ROOT, label="shard ledger")
    if (
        result["exact_chemistry_rows"] != index["summary"]["exact"]
        or result["new_exact_chemistry_rows"] != replay["counts"]["exact_computed_reconstruction"]
    ):
        raise ValueError("Readiness, index and replay totals differ")
    print(
        json.dumps(
            {
                "verified": True,
                "exact_rows": result["exact_chemistry_rows"],
                "training_ready": False,
            }
        )
    )


def main():
    if sys.argv[1:] == ["--verify"]:
        verify()
        return
    if sys.argv[1:]:
        raise ValueError("Usage: finalize.py [--verify]")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    previous = ROOT / "results/phase1/compose_lipid_training_readiness_v12/readiness.json"
    old = json.loads(previous.read_text())
    index = json.loads((HERE / "reconciled/result.json").read_text())
    replay = json.loads((HERE / "replay/result.json").read_text())
    audit = json.loads((HERE / "reconciled/combined-replay-audit.json").read_text())
    tests = ET.parse(HERE / "targeted-tests.xml").getroot().find("testsuite")
    if any(int(tests.attrib[k]) for k in ("errors", "failures", "skipped")):
        raise ValueError("Affected source-program tests did not all pass")
    if "all 30 present vendored assets verified" not in (HERE / "vendor-verify.log").read_text():
        raise ValueError("Vendor verification did not pass")
    added = replay["counts"]["exact_computed_reconstruction"]
    summary = index["summary"]
    if (
        summary["exact"] != old["exact_chemistry_rows"] + added
        or summary["pending"] != old["pending_chemistry_rows"] - added
        or summary["eligible"] != old["eligible_preparation_rows"]
    ):
        raise ValueError("Readiness accounting changed unexpectedly")
    result = {
        **old,
        "schema_version": "forge.compose_lipid_training_readiness.v13",
        "implementation": pin(ROOT, Path(__file__).resolve()),
        "inputs": {
            "previous": pin(ROOT, previous),
            "replay": pin(ROOT, HERE / "replay/result.json"),
            "source_adjudication": pin(ROOT, HERE / "adjudication.json"),
            "registry": pin(
                ROOT, ROOT / "data/vendor/qualified_acid_epoxide_staged_source_program_v1.json"
            ),
            "evidence_index": pin(ROOT, HERE / "reconciled/result.json"),
            "evidence_checkpoint": pin(ROOT, HERE / "reconciled/combined-replay-audit.json"),
            "focused_tests": pin(ROOT, HERE / "targeted-tests.xml"),
            "test_code": pin(ROOT, ROOT / "tests/test_compose_lipid_acid_epoxide_source.py"),
            "vendor_check": pin(ROOT, HERE / "vendor-verify.log"),
        },
        "exact_chemistry_rows": summary["exact"],
        "new_exact_chemistry_rows": added,
        "pending_chemistry_rows": summary["pending"],
        "families_with_exact_evidence": sum(v["exact"] > 0 for v in index["by_family"].values()),
        "by_family": index["by_family"],
        "pending_by_family": {
            f: v["pending"] for f, v in index["by_family"].items() if v["pending"]
        },
        "acid_epoxide": {
            "eligible_rows": replay["counts"]["rows"],
            "exact": added,
            "pending": replay["counts"]["rows"] - added,
            "maximum_exact_heavy_atoms": audit["maximum_exact_heavy_atoms"],
            "exact_above_96_atoms": audit["exact_above_96_atoms"],
            "evidence_basis": "computed_transform_consistency",
            "replay_seconds": replay["seconds"],
            "source_control": "Xu SI pp. 2-3 Fig. S1 and drawn E12CA1A3; p. 21 step f corroboration",
            "source_discrepancy_retained": "Printed shorter-head name conflicts with drawing and reported formula/mass; no exact-execution label admitted.",
        },
        "exact_records_pending_model_preparation": summary["exact"]
        - old["jointly_prepared_records"],
        "fresh_focused_tests": {
            "passed": int(tests.attrib["tests"]),
            "failed": 0,
            "errors": 0,
            "seconds": float(tests.attrib["time"]),
        },
        "full_suite_rerun": False,
        "remaining_work": [
            "Prepare source-aware atom origins and model inputs for newly exact AEMA and acid/epoxide rows, retaining existing prepared records.",
            "Qualify the remaining source domains and source-specific controls; Ren, Love and Zhou SI remain unavailable.",
            "Preserve all frozen protections, including B5 having no eligible training rows.",
            "Restore required historical validation artifacts and obtain a passing final acceptance report.",
            "Complete final deduplicated, source-balanced training cache and full-size representation admission.",
        ],
        "goal_status": "active",
        "training_ready": False,
        "training_admitted": False,
        "training_calls": 0,
        "paid_compute_calls": 0,
    }
    result.pop("new_exact_records_pending_model_preparation", None)
    OUTPUT.mkdir()
    dump(OUTPUT / "readiness.json", result)
    verify()


if __name__ == "__main__":
    main()

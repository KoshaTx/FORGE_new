"""Bind all changed E identities to their original saved, exact source assessments."""

import copy
import difflib
import resource
import time
from pathlib import Path

from audit_new_leaves import MAIN, OUT, pin, read, write
from candidate_routes import BASE, prior
from joint_population import content_sha

E = OUT.parent / "e_diversity/concentration_v1"
DEST = OUT / "e_routes_v1"


def prepare():
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    start = time.process_time()
    protocol = read(BASE / "protocol.json")
    source = prior.read_pin(protocol["population"])
    changed = read(E / "changed_products.json")["records"]
    witnesses = read(E / "source_witnesses.json")
    for ref in witnesses["inputs"].values():
        assert pin(Path(ref["path"]))["sha256"] == ref["sha256"]
    original_candidates = read(Path(witnesses["inputs"]["compact_attempts"]["path"]))
    original_candidates = {r["index"]: r for r in original_candidates}
    by_request = {r["index"]: r for r in witnesses["records"]}
    assert len(changed) == len(by_request) == 245
    rows = copy.deepcopy(source["requests"])
    bridges, changed_indices, old_strict = [], [], []
    for ordinal, record in enumerate(changed):
        index, candidate = record["request"], record["candidate"]
        original, witness = rows[index], by_request[index]
        assert (
            original["index"] == index
            and original["family"] == record["family"] == witness["family"]
        )
        assert original["selected_smiles"] == record["previous"]["smiles"]
        assert original["exact_L1"] and candidate["exact"]
        assert record["prior_route_verdict"] is None and record["prior_likelihood"] is None
        assert (
            candidate["smiles"] == witness["canonical_smiles"]
            and candidate["ordinal"] == witness["ordinal"]
        )
        branch = original_candidates[index]["branches"]["d1"]
        source_proposal = [branch["raw"], *branch.get("proposals", [])][candidate["ordinal"]]
        assert source_proposal == witness["proposal"]
        check = witness["proposal"]["check"]
        assert check["exact"]
        values = {c["role"]: c["smiles"] for c in candidate["components"]}
        assert len(values) == len(candidate["components"])
        assert values == witness["canonical_components"]
        assert set(values) == {item["role"] for item in original["requirements"]}
        for requirement in original["requirements"]:
            requirement["identity"] = values[requirement["role"]]
        if record["family"] == "a3_amine_aldehyde_alkyne":
            counts = {r["role"]: r["quantity"] for r in original["requirements"]}
            assert counts["amine_head"] == 1
            assert all(
                c["inverse"]["declared_events"] == counts["aldehyde"] == counts["alkyne"]
                for c in check["checks"]
            )
        if original["strict_complete"]:
            old_strict.append(index)
        source_ref = {
            "location": str((E / "source_witnesses.json").relative_to(MAIN))
            + f"#/records/{next(i for i, r in enumerate(witnesses['records']) if r['index'] == index)}/proposal/check",
            "sha256": pin(E / "source_witnesses.json")["sha256"],
        }
        original.update(
            {
                "canonical_product": candidate["smiles"],
                "selected_smiles": candidate["smiles"],
                "selected_ordinal": candidate["ordinal"],
                "strict_complete": False,
                "strict_secondary_status": "unestablished_for_changed_molecule",
                "family_receipt": {
                    "path": str((E / "source_witnesses.json").relative_to(MAIN)),
                    "sha256": source_ref["sha256"],
                },
                "l1_assessment_sha256": content_sha(check),
                "selected_L1_check_sha256": content_sha(check),
                "new_source_assessment": source_ref,
                "changed_molecule_no_inherited_route_verdict": True,
            }
        )
        bridges.append(
            {
                "index": index,
                "old_product": record["previous"]["smiles"],
                "new_product": candidate["smiles"],
                "source_assessment": source_ref,
                "roles_quantities_stages_preserved": True,
                "old_strict_positive_not_transferred": index in old_strict,
            }
        )
        changed_indices.append(index)
    assert len(set(changed_indices)) == 245
    assert old_strict == [20, 143, 145, 146, 153, 159, 182, 595, 672, 804, 1112]
    assert len(rows) == 1408 and sum(len(r["requirements"]) for r in rows) == 3723
    assert sum(r["strict_complete"] for r in rows) == 16
    assert sum(r["exact_L1"] for r in rows) == 1387
    assert all(
        a == b
        for a, b in zip(source["requests"], rows, strict=True)
        if a["index"] not in changed_indices
    )
    DEST.mkdir(exist_ok=False)
    write(
        DEST / "assembly_receipt_bridge.json",
        {
            "schema": "forge.current_and_changed_E_L1_witness_bridge.v1",
            "original_assemblies": source["inputs"]["assemblies"],
            "original_population": protocol["population"],
            "changed_products": pin(E / "changed_products.json"),
            "source_witnesses": pin(E / "source_witnesses.json"),
            "changes": bridges,
            "full_denominator": 1408,
        },
    )
    result = copy.deepcopy(source)
    result["schema"] = "forge.changed_E_molecule_computational_population.v1"
    result["inputs"].update(
        {
            "assemblies": pin(DEST / "assembly_receipt_bridge.json"),
            "producer": pin(Path(__file__)),
            "selected": pin(E / "candidate_cohort_1408.json"),
            "old_population": protocol["population"],
            "changed_source_witnesses": pin(E / "source_witnesses.json"),
        }
    )
    result.update(
        {
            "requests": rows,
            "changed_indices": changed_indices,
            "old_strict_positives_not_transferred": old_strict,
            "new_strict_secondary_count": 16,
            "official_cohort_replaced": False,
            "scope": "Separate E candidate cohort, no B repairs merged;245 changed identities require fresh routing. Remaining1163 request objects retained exactly.11 old dossier positives are unestablished for changed molecules, not negative experimental evidence.",
        }
    )
    write(DEST / "requests.json", result)
    original_evaluator = prior.EVAL / "run_v10.py"
    original_text = original_evaluator.read_text()
    assert original_text.count('assert totals["strict_secondary_products"] == 27') == 1
    candidate_text = original_text.replace(
        'assert totals["strict_secondary_products"] == 27',
        'assert totals["strict_secondary_products"] == 16',
    )
    evaluator = Path(__file__).with_name("evaluator_e_v1.py")
    with evaluator.open("x") as stream:
        stream.write(candidate_text)
    with (DEST / "evaluator.patch").open("x") as stream:
        stream.writelines(
            difflib.unified_diff(
                original_text.splitlines(keepends=True),
                candidate_text.splitlines(keepends=True),
                fromfile=str(original_evaluator),
                tofile=str(evaluator),
            )
        )
    assert (
        candidate_text.replace(
            'assert totals["strict_secondary_products"] == 16',
            'assert totals["strict_secondary_products"] == 27',
        )
        == original_text
    )
    write(
        DEST / "preparation.json",
        {
            "passed": True,
            "inputs": [
                pin(Path(__file__)),
                protocol["population"],
                pin(E / "changed_products.json"),
                pin(E / "source_witnesses.json"),
                pin(E / "candidate_cohort_1408.json"),
            ],
            "population": pin(DEST / "requests.json"),
            "assembly_bridge": pin(DEST / "assembly_receipt_bridge.json"),
            "changed_indices": changed_indices,
            "unchanged_request_count": 1163,
            "new_dossiers_unestablished": old_strict,
            "roles_quantities_stages_preserved": True,
            "strict_count_before": 27,
            "strict_count_remaining_admitted": 16,
            "evaluator_original": pin(original_evaluator),
            "evaluator_candidate": pin(evaluator),
            "evaluator_exact_single_assertion_diff": pin(DEST / "evaluator.patch"),
            "classification_and_chemistry_code_byte_identical": True,
            "CPU_seconds": time.process_time() - start,
            "independent_population_and_diff_review_required": True,
        },
    )
    print(pin(DEST / "preparation.json"))


if __name__ == "__main__":
    prepare()

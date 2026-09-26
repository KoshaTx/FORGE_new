"""Independent saved-data route binding review of the frozen joint selector."""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from rdkit import Chem

MAIN = Path(__file__).resolve().parents[5]
N = MAIN / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1"
OUT = N / "d_routes/joint_binding_review_v1"
PAIR = N / "integration/joint_selection_v1/run_v2"
FIELDS = ("exact_L1", "L2_ready", "combined_primary", "L3_direct_only", "strict_secondary")


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {
        "path": str(path.relative_to(MAIN)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def canon(value):
    mol = Chem.MolFromSmiles(value)
    assert mol is not None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def run():
    started = time.process_time()
    protocol = read(PAIR / "protocol.json")
    data = read(PAIR / "result.json")
    assert (
        pin(PAIR / "result.json")["sha256"]
        == "f8504d0fe1edfbd25736cbf36426db619f8e29a6f2c08786f6b12afb1ed6f040"
    )
    assert (
        pin(PAIR / "protocol.json")["sha256"]
        == "64e2d8648a98f9ce7a410f24b9f95ff8a52adfffcd15469b5f0cbe19df21df10"
    )
    for ref in protocol["inputs"].values():
        assert hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    requests, verdicts = {}, {}
    for arm in ("D", "B", "E"):
        requests[arm] = {
            row["index"]: row
            for row in read(Path(protocol["inputs"][f"{arm}_requests"]["path"]))["requests"]
        }
        verdicts[arm] = {
            row["index"]: row
            for row in read(Path(protocol["inputs"][f"{arm}_products"]["path"]))["products"]
        }
        assert set(requests[arm]) == set(verdicts[arm]) == set(range(1408))
        assert read(Path(protocol["inputs"][f"{arm}_validation"]["path"]))["passed"]
        assert (
            read(Path(protocol["inputs"][f"{arm}_protocol"]["path"]))["as_of_utc"]
            == protocol["route_clock"]
        )
    assert len(data["selection"]) == len(data["selected_evidence"]) == 1408
    totals, sources = Counter(), Counter()
    transitions = {field: {"gains": [], "losses": []} for field in FIELDS}
    changed = []
    for index, (selected, evidence) in enumerate(
        zip(data["selection"], data["selected_evidence"], strict=True)
    ):
        assert selected["request"] == evidence["request"] == index
        arm = evidence["source_arm"]
        request, verdict = requests[arm][index], verdicts[arm][index]
        assert evidence["route_request"] == request
        assert evidence["route_verdict"] == verdict
        assert canon(selected["smiles"]) == canon(request["selected_smiles"])
        if request["exact_L1"]:
            assert canon(selected["smiles"]) == request["canonical_product"]
        else:
            assert arm == "D" and not request["requirements"]
        assert selected["ordinal"] == evidence["ordinal"] == request["selected_ordinal"]
        assert selected["exact"] == request["exact_L1"] == verdict["exact_L1"]
        assert {(r["role"], canon(r["smiles"])) for r in selected["components"]} == {
            (r["role"], r["identity"]) for r in request["requirements"]
        }
        assert [(r["branch_id"], r["identity"]) for r in request["requirements"]] == [
            (r["branch_id"], r["identity"]) for r in verdict["branches"]
        ]
        assert verdict["L2_ready"] == (
            verdict["exact_L1"]
            and all(b["direct_listed"] or b["computational_path"] for b in verdict["branches"])
        )
        assert verdict["L3_direct_only"] == (
            verdict["exact_L1"] and all(b["direct_listed"] for b in verdict["branches"])
        )
        assert verdict["combined_primary"] == (
            verdict["exact_L1"] and all(b["makeable"] for b in verdict["branches"])
        )
        assert verdict["strict_secondary"] == request["strict_complete"]
        assert request.get("assembly_context_sha256") == requests["D"][index].get(
            "assembly_context_sha256"
        )
        assert [(r["role"], r["quantity"], r["stages"]) for r in request["requirements"]] == [
            (r["role"], r["quantity"], r["stages"]) for r in requests["D"][index]["requirements"]
        ]
        for field in FIELDS:
            totals[field] += verdict[field]
            if verdict[field] != verdicts["D"][index][field]:
                transitions[field]["gains" if verdict[field] else "losses"].append(index)
            assert not verdicts["D"][index][field] or verdict[field]
        sources[arm] += 1
        if arm != "D":
            assert request["canonical_product"] != requests["D"][index]["canonical_product"]
            assert request["changed_molecule_no_inherited_route_verdict"]
            changed.append(
                {
                    "request": index,
                    "source_arm": arm,
                    "canonical_product": request["canonical_product"],
                    "route_verdict": verdict,
                }
            )
    assert sources == data["chosen_source_counts"] == {"D": 1372, "B": 7, "E": 29}
    assert totals == {
        "exact_L1": 1387,
        "L2_ready": 1189,
        "combined_primary": 623,
        "L3_direct_only": 121,
        "strict_secondary": 27,
    }
    for field in FIELDS[1:]:
        assert totals[field] == data["routing_final"][field]
    result = {
        "schema": "forge.joint_selection.independent_route_binding_review.v1",
        "passed": True,
        "reviewer": "routes_continue",
        "scope": "Independent exact source-to-verdict binding and complete1408 branch-derived metric recount; no route/model/HTTP calls. Quality/diversity independently reviewed by separate agent.",
        "inputs": [pin(PAIR / "protocol.json"), pin(PAIR / "result.json"), pin(Path(__file__))],
        "authenticated_protocol_input_count": len(protocol["inputs"]),
        "common_as_of": protocol["route_clock"],
        "complete_population": 1408,
        "unchanged_original_requests": 1372,
        "changed_requests": 36,
        "source_counts": dict(sources),
        "totals": dict(totals),
        "transitions_against_D": transitions,
        "changed_identity_verdicts": changed,
        "no_old_positive_route_metric_lost": True,
        "new_conditional_listing_or_search_results_used": False,
        "CPU_seconds": time.process_time() - started,
    }
    OUT.mkdir(exist_ok=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "result": pin(OUT / "result.json"),
                "totals": dict(totals),
                "CPU_seconds": result["CPU_seconds"],
            }
        )
    )


if __name__ == "__main__":
    run()

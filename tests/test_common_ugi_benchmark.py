from __future__ import annotations

import json
from pathlib import Path

from forge.assembly import Ugi3AssemblyAdapter
from forge.core.io import iter_csv
from forge.model.common_ugi_benchmark import (
    CommonUgiAttempt,
    assess_common_ugi_attempts,
    load_ugi_identity_references,
)
from forge.synthesis.assessment.common_route_evidence import (
    assess_common_route_evidence,
    load_frozen_component_evidence,
)

REPO = Path(__file__).resolve().parents[1]
ASSIGNMENTS = REPO / "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
REGISTRY = REPO / "data/vendor/qualified_reactions_v1.json"
ROUTES = REPO / "results/phase1/ugi_bounded_hybrid_route_cascade_v1/component_route_ledger.jsonl.gz"


def test_learned_inventory_uses_separate_cpu_smoke_and_exact_h100_preflight() -> None:
    production = json.loads(
        (REPO / "experiments/phase1/multireaction/learned_inventory_selector.json").read_text()
    )
    preflight = json.loads(
        (
            REPO
            / "experiments/phase1/multireaction/learned_inventory_selector_h100_preflight.json"
        ).read_text()
    )
    production_config = json.loads(
        (REPO / "configs/multireaction/learned_inventory_selector_v1.json").read_text()
    )
    preflight_config = json.loads(
        (
            REPO
            / "configs/multireaction/learned_inventory_selector_h100_preflight_v1.json"
        ).read_text()
    )
    assert production_config["smoke"]["device"] == "cpu"
    assert production_config["full"]["device"] == "cuda"
    assert production["replicates"]["full"] == 3
    assert production["stages"][0]["resources"]["gpu_type"] == "H100!"
    assert preflight["profiles"] == ["smoke"]
    assert preflight_config["smoke"]["device"] == "cuda"
    assert preflight["stages"][0]["resources"]["gpu_type"] == "H100!"


def _unique_product(adapter: Ugi3AssemblyAdapter) -> tuple[str, tuple[str, ...]]:
    for row in iter_csv(ASSIGNMENTS):
        if row["primary_product_fold"] != "train":
            continue
        product = row["canonical_product_smiles"]
        if len(adapter.decompose(product)) == 1:
            visible = tuple(f"{role}:{row[f'{role}_smiles']}" for role in adapter.roles)
            return product, visible
    raise AssertionError("test corpus contains no unique exact-L1 product")


def test_hash_addressed_ugi_identity_reference_is_reused_in_process() -> None:
    roles = Ugi3AssemblyAdapter.from_registry(REGISTRY).roles

    first = load_ugi_identity_references(ASSIGNMENTS, roles=roles)
    second = load_ugi_identity_references(ASSIGNMENTS, roles=roles)

    assert first[0] is second[0]
    assert first[1] is second[1]
    assert first[2] is second[2]


def test_common_assessor_uses_method_visible_inventory_for_open_endedness() -> None:
    adapter = Ugi3AssemblyAdapter.from_registry(REGISTRY)
    product, visible = _unique_product(adapter)
    finite = CommonUgiAttempt(
        method_id="finite",
        seed=7,
        attempt_index=0,
        status="generated",
        product_smiles=product,
        method_visible_component_ids=visible,
        generator_calls=1,
        reaction_calls=1,
        route_calls=0,
        oracle_calls=0,
        wall_seconds=0.1,
    )
    _, finite_result = assess_common_ugi_attempts(
        [finite], adapter=adapter, assignments_path=ASSIGNMENTS
    )
    open_attempt = CommonUgiAttempt(
        **{**finite.__dict__, "method_id": "open", "method_visible_component_ids": ()}
    )
    _, open_result = assess_common_ugi_attempts(
        [open_attempt], adapter=adapter, assignments_path=ASSIGNMENTS
    )
    assert (
        finite_result["metrics"][
            "unique_method_visible_open_ended_exact_l1_products_per_1000_attempts"
        ]
        == 0.0
    )
    assert (
        open_result["metrics"][
            "unique_method_visible_open_ended_exact_l1_products_per_1000_attempts"
        ]
        == 1000.0
    )


def test_frozen_route_index_accepts_null_component_value_without_promotion() -> None:
    evidence = load_frozen_component_evidence(ROUTES)
    assert evidence
    assert all("strict_complete" in value for value in evidence.values())


def test_route_assessment_discloses_bounded_noncomparative_evidence_scope() -> None:
    adapter = Ugi3AssemblyAdapter.from_registry(REGISTRY)
    product, _ = _unique_product(adapter)
    attempt = CommonUgiAttempt(
        method_id="open",
        seed=7,
        attempt_index=0,
        status="generated",
        product_smiles=product,
        method_visible_component_ids=(),
        generator_calls=1,
        reaction_calls=0,
        route_calls=0,
        oracle_calls=0,
        wall_seconds=0.1,
    )
    rows, _ = assess_common_ugi_attempts(
        [attempt], adapter=adapter, assignments_path=ASSIGNMENTS
    )
    _, result = assess_common_route_evidence(
        rows, component_evidence=load_frozen_component_evidence(ROUTES)
    )
    assert result["evidence_scope"] == {
        "population": "bounded_forge_ugi_candidate_cascade",
        "method_blind_cross_method_union": False,
        "synthesis_success_comparison_authorized": False,
    }


def test_route_assessment_keeps_upstream_and_terminal_closure_separate() -> None:
    adapter = Ugi3AssemblyAdapter.from_registry(REGISTRY)
    product, _ = _unique_product(adapter)
    attempt = CommonUgiAttempt(
        method_id="open",
        seed=7,
        attempt_index=0,
        status="generated",
        product_smiles=product,
        method_visible_component_ids=(),
        generator_calls=1,
        reaction_calls=0,
        route_calls=0,
        oracle_calls=0,
        wall_seconds=0.1,
    )
    rows, _ = assess_common_ugi_attempts(
        [attempt], adapter=adapter, assignments_path=ASSIGNMENTS
    )
    by_role = rows[0]["exact_l1_traces"][0]["components_by_role"]
    component_evidence = {}
    for index, (role, smiles) in enumerate(sorted(by_role.items())):
        terminal = index != 0
        component_evidence[(role, smiles)] = {
            "strict_complete": terminal,
            "verified_upstream": True,
            "terminal_evidence": terminal,
            "explicit_abstention": not terminal,
            "assessment_outcome": "complete" if terminal else "terminal_evidence_missing",
            "final_component_state": "complete" if terminal else "unresolved",
        }
    assessed, result = assess_common_route_evidence(
        rows,
        component_evidence=component_evidence,
        evidence_scope={"method_blind_cross_method_union": True},
    )
    route = assessed[0]["common_route_evidence"]
    assert route["verified_upstream"] is True
    assert route["terminal_evidence"] is False
    assert route["complete_dossier"] is False
    assert route["abstained"] is True
    assert result["verified_upstream"] == 1
    assert result["terminal_evidence"] == 0

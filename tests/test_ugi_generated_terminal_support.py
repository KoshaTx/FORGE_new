from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem

from forge.potency.audit.ugi_semantic_annotations import ROLE_NAMES
from forge.design.corpus.ugi_generated_terminal_support import (
    DeclaredGraphSupportContext,
    UgiGeneratedTerminalSupportError,
    qualify_locked_generated_ugi_terminal_support,
)
from forge.design.corpus.ugi_held_component_gate import load_ugi_reaction_contract
from forge.design.schedule.ugi_matched_budget_orchestration import LockedMatchedTerminal
from forge.route.engine.planner import KnowledgeDisposition, KnowledgeResult, RouteTarget
from forge.route.terminals.terminal_assessment import (
    QualifiedUgiL1Reverifier,
    ValidatedUgiTerminalPayload,
)
from forge.route.assessment.ugi3_support_boundary import (
    AuthenticatedInternalRoleRegistry,
    MolecularSupportState,
)

REPO = Path(__file__).resolve().parents[1]
COMPONENTS = {
    "amine_head": "CN",
    "oxoester_aldehyde_body_tail": "CC=O",
    "isocyanide_tail": "[C-]#[N+]C",
}
PRODUCT = "CNC(=O)C(C)NC"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _reaction():
    return load_ugi_reaction_contract(REPO / "data/vendor/qualified_reactions_v1.json")


def _payload(
    *,
    components: dict[str, str] | None = None,
    product: str = PRODUCT,
) -> ValidatedUgiTerminalPayload:
    return ValidatedUgiTerminalPayload.from_recovered_components(
        product_smiles=product,
        components_by_role=components or COMPONENTS,
        l1_reaction=_reaction(),
        l1_reaction_sha256=_hash("l1-reaction"),
        component_recovery_contract_sha256=_hash("component-recovery"),
    )


def _reverifier(*, reaction_hash: str | None = None) -> QualifiedUgiL1Reverifier:
    return QualifiedUgiL1Reverifier(
        reaction_contract=_reaction(),
        l1_reaction_sha256=reaction_hash or _hash("l1-reaction"),
    )


def _terminal(
    payload: ValidatedUgiTerminalPayload,
    **overrides: object,
) -> LockedMatchedTerminal:
    values: dict[str, object] = {
        "unit_id": "unit-1",
        "morphology_program_sha256": _hash("program"),
        "checkpoint_index": 0,
        "generator_checkpoint_sha256": _hash("generator"),
        "closure_checkpoint_sha256": _hash("closure"),
        "terminal_id": "terminal-1",
        "terminal_locked": True,
        "terminal_valid": True,
        "exact_l1": True,
        "terminal_bytes": payload.canonical_bytes,
        "generation_trace_bytes": b"sealed-generation-trace\n",
        "payload": payload,
    }
    values.update(overrides)
    return LockedMatchedTerminal(**values)  # type: ignore[arg-type]


def _candidate_record(
    payload: ValidatedUgiTerminalPayload,
) -> dict[str, object]:
    return {
        "valid": True,
        "component_reconstruction_valid": True,
        "smiles": payload.product_smiles,
        "component_smiles_by_role": {
            role: payload.by_role()[role].canonical_smiles for role in ROLE_NAMES
        },
        "program": {
            "node_counts": [1, 1, 1],
            "junction_budgets": [0, 0, 0],
            "cycle_ranks": [0, 0, 0],
            "attachment_counts": [1, 1, 1],
        },
        "offspring_by_role": {role: [0] for role in ROLE_NAMES},
    }


def _atom_vocabulary(product: str) -> frozenset[tuple[str, int, bool, int]]:
    molecule = Chem.MolFromSmiles(product)
    assert molecule is not None
    return frozenset(
        (
            atom.GetSymbol(),
            atom.GetFormalCharge(),
            atom.GetIsAromatic(),
            atom.GetNumExplicitHs(),
        )
        for atom in molecule.GetAtoms()
    )


def _graph_support(
    product: str,
    **config_overrides: int,
) -> DeclaredGraphSupportContext:
    config = {
        "maximum_total_atoms": 96,
        "maximum_component_atoms": 48,
        "maximum_junction_budget": 4,
        "maximum_cycle_rank": 4,
        "maximum_attachment_count": 4,
        "maximum_children": 4,
    }
    config.update(config_overrides)
    return DeclaredGraphSupportContext(
        generator_checkpoint_sha256=_hash("generator"),
        model_config=config,
        atom_vocabulary=_atom_vocabulary(product),
    )


class _OutsideSource:
    def __init__(self) -> None:
        self.calls: list[RouteTarget] = []

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        self.calls.append(target)
        return KnowledgeResult(
            disposition=KnowledgeDisposition.OUTSIDE_SUPPORT,
            evidence=(),
            detail="exact evidence index miss",
        )


def test_locked_terminal_is_rechecked_and_yields_three_exact_root_qualifications() -> None:
    payload = _payload()
    terminal = _terminal(payload)

    support = qualify_locked_generated_ugi_terminal_support(
        terminal,
        candidate_record=_candidate_record(payload),
        graph_support=_graph_support(payload.product_smiles),
        l1_reverifier=_reverifier(),
    )

    assert support.terminal_sha256 == terminal.terminal_sha256
    assert support.product_smiles == payload.product_smiles
    assert support.l1_reverification.exact_product_reconstructed
    assert tuple(item.role for item in support.handle_rechecks) == ROLE_NAMES
    assert all(item.passes_registry_handle_policy for item in support.handle_rechecks)
    assert not any(item.forbidden_substructure_match for item in support.handle_rechecks)
    assert tuple(target.role for target in support.root_targets) == ROLE_NAMES
    assert len(support.qualification_mapping) == 3
    for qualification in support.root_qualifications:
        assert qualification.exact_l1_eligible
        assert qualification.supported_ugi_role
        assert qualification.role_handle_qualified
        assert (
            qualification.molecular_support_state is MolecularSupportState.WITHIN_DECLARED_SUPPORT
        )


def test_qualified_terminal_wraps_existing_boundary_without_promoting_evidence() -> None:
    payload = _payload()
    support = qualify_locked_generated_ugi_terminal_support(
        _terminal(payload),
        candidate_record=_candidate_record(payload),
        graph_support=_graph_support(payload.product_smiles),
        l1_reverifier=_reverifier(),
    )
    delegate = _OutsideSource()
    source = support.wrap_source(
        delegate,
        authenticated_internal_roles=AuthenticatedInternalRoleRegistry(frozenset()),
    )

    decision = source.lookup(support.root_targets[0])

    assert decision.disposition is KnowledgeDisposition.MISSING_KNOWLEDGE
    assert decision.evidence == ()
    assert delegate.calls == [support.root_targets[0]]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"terminal_locked": False}, "sealed terminal"),
        ({"terminal_valid": False, "exact_l1": False}, "valid terminal"),
        ({"exact_l1": False}, "exact L1"),
        ({"terminal_bytes": b"not-the-payload\n"}, "bytes do not match"),
    ],
)
def test_unqualified_or_unsealed_terminal_fails_closed(
    overrides: dict[str, object],
    message: str,
) -> None:
    payload = _payload()

    with pytest.raises(UgiGeneratedTerminalSupportError, match=message):
        qualify_locked_generated_ugi_terminal_support(
            _terminal(payload, **overrides),
            candidate_record=_candidate_record(payload),
            graph_support=_graph_support(payload.product_smiles),
            l1_reverifier=_reverifier(),
        )


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("valid", None, "valid state"),
        ("component_reconstruction_valid", None, "component reconstruction"),
        ("program", None, "program"),
        ("offspring_by_role", None, "offspring_by_role"),
        ("component_smiles_by_role", None, "components"),
        ("smiles", None, "candidate product"),
    ],
)
def test_missing_or_unassessed_candidate_fields_fail_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _payload()
    record = _candidate_record(payload)
    record[field] = value

    with pytest.raises(UgiGeneratedTerminalSupportError, match=message):
        qualify_locked_generated_ugi_terminal_support(
            _terminal(payload),
            candidate_record=record,
            graph_support=_graph_support(payload.product_smiles),
            l1_reverifier=_reverifier(),
        )


def test_candidate_identity_must_match_sealed_product_and_components() -> None:
    payload = _payload()
    product_mismatch = _candidate_record(payload)
    product_mismatch["smiles"] = "CCO"
    component_mismatch = _candidate_record(payload)
    component_mismatch["component_smiles_by_role"] = {
        **component_mismatch["component_smiles_by_role"],  # type: ignore[dict-item]
        "amine_head": "CCN",
    }

    with pytest.raises(UgiGeneratedTerminalSupportError, match="product does not match"):
        qualify_locked_generated_ugi_terminal_support(
            _terminal(payload),
            candidate_record=product_mismatch,
            graph_support=_graph_support(payload.product_smiles),
            l1_reverifier=_reverifier(),
        )
    with pytest.raises(UgiGeneratedTerminalSupportError, match="amine_head component"):
        qualify_locked_generated_ugi_terminal_support(
            _terminal(payload),
            candidate_record=component_mismatch,
            graph_support=_graph_support(payload.product_smiles),
            l1_reverifier=_reverifier(),
        )


def test_declared_graph_support_is_reapplied_and_cannot_be_unassessed() -> None:
    payload = _payload()
    molecule = Chem.MolFromSmiles(payload.product_smiles)
    assert molecule is not None
    with pytest.raises(UgiGeneratedTerminalSupportError, match="maximum_total_atoms"):
        qualify_locked_generated_ugi_terminal_support(
            _terminal(payload),
            candidate_record=_candidate_record(payload),
            graph_support=_graph_support(
                payload.product_smiles,
                maximum_total_atoms=molecule.GetNumHeavyAtoms() - 1,
            ),
            l1_reverifier=_reverifier(),
        )
    with pytest.raises(UgiGeneratedTerminalSupportError, match="fields are missing"):
        DeclaredGraphSupportContext(
            generator_checkpoint_sha256=_hash("generator"),
            model_config={},
            atom_vocabulary=_atom_vocabulary(payload.product_smiles),
        )
    with pytest.raises(UgiGeneratedTerminalSupportError, match="nonempty frozenset"):
        DeclaredGraphSupportContext(
            generator_checkpoint_sha256=_hash("generator"),
            model_config=_graph_support(payload.product_smiles).model_config,
            atom_vocabulary=frozenset(),
        )


def test_generator_checkpoint_and_independent_l1_hash_are_bound() -> None:
    payload = _payload()
    support = _graph_support(payload.product_smiles)

    with pytest.raises(UgiGeneratedTerminalSupportError, match="different generator"):
        qualify_locked_generated_ugi_terminal_support(
            replace(_terminal(payload), generator_checkpoint_sha256=_hash("other-generator")),
            candidate_record=_candidate_record(payload),
            graph_support=support,
            l1_reverifier=_reverifier(),
        )
    with pytest.raises(UgiGeneratedTerminalSupportError, match="reverification failed"):
        qualify_locked_generated_ugi_terminal_support(
            _terminal(payload),
            candidate_record=_candidate_record(payload),
            graph_support=support,
            l1_reverifier=_reverifier(reaction_hash=_hash("other-reaction")),
        )


def test_exact_l1_component_with_disallowed_handle_multiplicity_is_rejected() -> None:
    components = {
        "amine_head": "NCC(N)CNC",
        "oxoester_aldehyde_body_tail": "CC=O",
        "isocyanide_tail": "[C-]#[N+]C",
    }
    reaction = _reaction()
    reactants = tuple(
        Chem.MolFromSmiles(components[role.name]) for role in reaction.definition.reactant_roles
    )
    outcomes = reaction.forward.RunReactants(reactants, maxProducts=64)
    products = []
    for outcome in outcomes:
        molecule = Chem.Mol(outcome[0])
        Chem.SanitizeMol(molecule)
        products.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))
    payload = _payload(components=components, product=sorted(set(products))[0])

    with pytest.raises(UgiGeneratedTerminalSupportError, match="exact frozen handle policy"):
        qualify_locked_generated_ugi_terminal_support(
            _terminal(payload),
            candidate_record=_candidate_record(payload),
            graph_support=_graph_support(payload.product_smiles),
            l1_reverifier=_reverifier(),
        )

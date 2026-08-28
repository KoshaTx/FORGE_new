"""Identity guards for the reuse added to the production-evaluation hot path.

Each test re-derives the rewritten quantity a second, naive way -- the pre-change one-step
assembly without a reuse table, and a fresh canonicalization for every occurrence -- and requires
exact equality.  A speed change to this path is only admissible if it changes nothing.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from forge.assembly import ReactionProgramSpec, RegistryRepeatedReactionProgram
from forge.assembly.program import ReactionProgramError
from forge.core.hashing import sha256_file
from forge.model.reaction_program_evaluation import (
    ReactionProgramEvaluationError,
    _canonical_component,
    load_reaction_program_training_references,
)

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data/vendor/qualified_reaction_families_v1.json"

# One three-nitrogen head and one acrylate, deep enough that the reverse search revisits the same
# one-step assemblies: the product decomposes at two distinct depths.
HEAD = "NCCNCCN"
REPEAT = "C=CC(=O)OCCCC"
PRODUCT = "CCCCOC(=O)CCN(CCNCCN)CCC(=O)OCCCC"

pytestmark = pytest.mark.skipif(
    not REGISTRY.is_file(), reason="qualified reaction family registry is not vendored here"
)


def _adapter() -> RegistryRepeatedReactionProgram:
    return RegistryRepeatedReactionProgram.from_registry(
        REGISTRY,
        ReactionProgramSpec(
            program_id="bl_2023_repeated_aza_michael",
            reaction_id="aza_michael_amine_acrylate",
            accumulator_role="amine_head",
            repeat_role="alkyl_acrylate_or_acrylamide_tail",
            minimum_steps=1,
            maximum_steps=4,
        ),
        expected_sha256=str(sha256_file(REGISTRY)),
    )


def _count_single_layers(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count executed one-step assemblies, which is what the reuse table is supposed to reduce."""

    calls = [0]
    original = RegistryRepeatedReactionProgram._single_forward_layer

    def counted(self, accumulator_smiles, repeated, *, maximum_outcomes):
        calls[0] += 1
        return original(self, accumulator_smiles, repeated, maximum_outcomes=maximum_outcomes)

    monkeypatch.setattr(RegistryRepeatedReactionProgram, "_single_forward_layer", counted)
    return calls


def _disable_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the pre-change behaviour: every one-step assembly is executed on every request."""

    original = RegistryRepeatedReactionProgram._forward_layer

    def without_memo(self, accumulator_smiles, repeated_smiles, *, maximum_outcomes, memo=None):
        return original(
            self, accumulator_smiles, repeated_smiles, maximum_outcomes=maximum_outcomes, memo=None
        )

    monkeypatch.setattr(RegistryRepeatedReactionProgram, "_forward_layer", without_memo)


def test_forward_layer_reuse_leaves_every_repeated_program_result_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()

    with monkeypatch.context() as patched:
        naive_calls = _count_single_layers(patched)
        _disable_memo(patched)
        naive_traces = adapter.decompose(PRODUCT)
        naive_products = adapter.forward_products(HEAD, (REPEAT, REPEAT))
        naive_forward = adapter.forward_traces(HEAD, (REPEAT, REPEAT))
        naive_check = adapter.check_forward(HEAD, (REPEAT, REPEAT), PRODUCT)
        naive_layers = naive_calls[0]

    with monkeypatch.context() as patched:
        reused_calls = _count_single_layers(patched)
        reused_traces = adapter.decompose(PRODUCT)
        reused_products = adapter.forward_products(HEAD, (REPEAT, REPEAT))
        reused_forward = adapter.forward_traces(HEAD, (REPEAT, REPEAT))
        reused_check = adapter.check_forward(HEAD, (REPEAT, REPEAT), PRODUCT)
        reused_layers = reused_calls[0]

    assert reused_traces == naive_traces
    assert reused_products == naive_products
    assert reused_forward == naive_forward
    assert reused_check == naive_check
    # The fixture must actually exercise the reuse, or the equality above proves nothing.
    assert reused_layers < naive_layers


def test_forward_layer_reuse_keeps_invalid_input_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    with pytest.raises(ReactionProgramError):
        adapter.decompose("not a molecule")
    with pytest.raises(ReactionProgramError):
        adapter.forward_products(HEAD, ("not a molecule",))
    with pytest.raises(ReactionProgramError):
        adapter.forward_products("not a molecule", (REPEAT,))
    # A reuse table must never turn a second invalid request into a cache hit.
    with pytest.raises(ReactionProgramError):
        adapter.forward_products(HEAD, ("not a molecule", REPEAT))


def _write_csv_gz(path: Path, rows: list[dict[str, str]]) -> None:
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _reference_fixture(tmp_path: Path) -> dict[str, Path]:
    roles = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
    # Deliberately non-canonical and repeated source strings, so memoized and per-row
    # canonicalization can only agree if the memo is faithful.
    assignments = [
        {
            "canonical_product_smiles": "OCCN(CC)CC",
            "amine_head_smiles": "NCC",
            "amine_head_family_fold": "train",
            "oxoester_aldehyde_body_tail_smiles": "C(=O)CC",
            "oxoester_aldehyde_body_tail_family_fold": "train",
            "isocyanide_tail_smiles": "[C-]#[N+]CC",
            "isocyanide_tail_family_fold": "train",
            "primary_product_fold": "train",
        },
        {
            "canonical_product_smiles": "OCCN(CCC)CC",
            "amine_head_smiles": "C(C)N",
            "amine_head_family_fold": "train",
            "oxoester_aldehyde_body_tail_smiles": "CCC=O",
            "oxoester_aldehyde_body_tail_family_fold": "train",
            "isocyanide_tail_smiles": "CC[N+]#[C-]",
            "isocyanide_tail_family_fold": "train",
            "primary_product_fold": "train",
        },
        {
            "canonical_product_smiles": "OCCCN(CC)CC",
            "amine_head_smiles": "NCC",
            "amine_head_family_fold": "heldout",
            "oxoester_aldehyde_body_tail_smiles": "C(=O)CC",
            "oxoester_aldehyde_body_tail_family_fold": "heldout",
            "isocyanide_tail_smiles": "[C-]#[N+]CC",
            "isocyanide_tail_family_fold": "heldout",
            "primary_product_fold": "heldout",
        },
    ]
    splits = [
        {"record_id": "r1", "program_id": "prog", "product_fold": "train"},
        {"record_id": "r2", "program_id": "prog", "product_fold": "train"},
        {"record_id": "r3", "program_id": "prog", "product_fold": "heldout"},
    ]
    atlas = [
        {
            "record_id": "r1",
            "program_id": "prog",
            "canonical_product_smiles": "CCOC(=O)CCNCC",
            "terminal_head_smiles": "NCC",
            "step_count": "2",
            "repeat_component_smiles": "C(=C)C(=O)OCC",
        },
        {
            "record_id": "r2",
            "program_id": "prog",
            "canonical_product_smiles": "CCOC(=O)CCNCCC",
            "terminal_head_smiles": "C(C)CN",
            "step_count": "1",
            "repeat_component_smiles": "C=CC(=O)OCC",
        },
        {
            "record_id": "r3",
            "program_id": "prog",
            "canonical_product_smiles": "CCOC(=O)CCNCCCC",
            "terminal_head_smiles": "NCCCC",
            "step_count": "1",
            "repeat_component_smiles": "C=CC(=O)OCC",
        },
    ]
    paths = {
        "ugi_assignments": tmp_path / "assignments.csv.gz",
        "multireaction_splits": tmp_path / "splits.csv.gz",
        "multireaction_atlas": tmp_path / "atlas.csv.gz",
    }
    _write_csv_gz(paths["ugi_assignments"], assignments)
    _write_csv_gz(paths["multireaction_splits"], splits)
    _write_csv_gz(paths["multireaction_atlas"], atlas)
    return {"roles": roles, **paths}


def _naive_training_references(fixture: dict[str, Path], spec: ReactionProgramSpec) -> tuple:
    """Re-derive the references with one fresh canonicalization per occurrence."""

    from forge.core.io import iter_csv
    from forge.corpus.reaction_program_records import repeat_component_smiles

    products: dict[str, set[str]] = {"ugi_3cr_agile": set(), "prog": set()}
    components: dict[str, dict[str, set[str]]] = {"ugi_3cr_agile": {}, "prog": {}}
    for row in iter_csv(fixture["ugi_assignments"]):
        if row.get("primary_product_fold") != "train":
            continue
        products["ugi_3cr_agile"].add(_canonical_component(row["canonical_product_smiles"]))
        for role in fixture["roles"]:
            components["ugi_3cr_agile"].setdefault(role, set()).add(
                _canonical_component(row[f"{role}_smiles"])
            )
    train = {
        row["record_id"]
        for row in iter_csv(fixture["multireaction_splits"])
        if row.get("product_fold") == "train"
    }
    for row in iter_csv(fixture["multireaction_atlas"]):
        if row["record_id"] not in train:
            continue
        products["prog"].add(_canonical_component(row["canonical_product_smiles"]))
        components["prog"].setdefault(spec.accumulator_role, set()).add(
            _canonical_component(row["terminal_head_smiles"])
        )
        components["prog"].setdefault(spec.repeat_role, set()).update(
            _canonical_component(value) for value in repeat_component_smiles(row)
        )
    return products, components


def test_training_reference_canonicalization_reuse_matches_a_fresh_parse_per_row(
    tmp_path: Path,
) -> None:
    fixture = _reference_fixture(tmp_path)
    spec = ReactionProgramSpec(
        program_id="prog",
        reaction_id="aza_michael_amine_acrylate",
        accumulator_role="amine_head",
        repeat_role="alkyl_acrylate_or_acrylamide_tail",
        minimum_steps=1,
        maximum_steps=4,
    )
    observed = load_reaction_program_training_references(
        ugi_assignments=fixture["ugi_assignments"],
        multireaction_atlas=fixture["multireaction_atlas"],
        multireaction_splits=fixture["multireaction_splits"],
        repeated_program_specs={"prog": spec},
        ugi_program_id="ugi_3cr_agile",
        ugi_roles=fixture["roles"],
    )
    assert observed == _naive_training_references(fixture, spec)
    # The fixture must contain repeated source strings, or the equality proves nothing.
    assert len(observed[1]["ugi_3cr_agile"]["amine_head"]) == 1


def test_training_reference_reuse_still_rejects_an_invalid_component(tmp_path: Path) -> None:
    fixture = _reference_fixture(tmp_path)
    rows = [
        {
            "canonical_product_smiles": "OCCN(CC)CC",
            "amine_head_smiles": "not a molecule",
            "amine_head_family_fold": "train",
            "oxoester_aldehyde_body_tail_smiles": "C(=O)CC",
            "oxoester_aldehyde_body_tail_family_fold": "train",
            "isocyanide_tail_smiles": "[C-]#[N+]CC",
            "isocyanide_tail_family_fold": "train",
            "primary_product_fold": "train",
        }
    ]
    _write_csv_gz(fixture["ugi_assignments"], rows)
    with pytest.raises(ReactionProgramEvaluationError):
        load_reaction_program_training_references(
            ugi_assignments=fixture["ugi_assignments"],
            multireaction_atlas=fixture["multireaction_atlas"],
            multireaction_splits=fixture["multireaction_splits"],
            repeated_program_specs={
                "prog": ReactionProgramSpec(
                    program_id="prog",
                    reaction_id="aza_michael_amine_acrylate",
                    accumulator_role="amine_head",
                    repeat_role="alkyl_acrylate_or_acrylamide_tail",
                    minimum_steps=1,
                    maximum_steps=4,
                )
            },
            ugi_program_id="ugi_3cr_agile",
            ugi_roles=fixture["roles"],
        )

"""Adversarial checks and explicit scope gaps in the draft B5 source programs."""

import copy
import json
from dataclasses import replace
from pathlib import Path

from rdkit import Chem

from forge.assembly.families import constitutional_molecule
from forge.assembly.repeated_components import RepeatBounds
from forge.assembly.staged_program import RegistryStagedProgram
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    registry = HERE / "draft-registry.json"
    source = HERE / "control-transcriptions.json"
    controls = {c["label"]: c for c in json.loads(source.read_text())["controls"]}
    programs = {
        k: RegistryStagedProgram.from_registry(
            registry,
            program_id="source_b5_" + c["profile"],
            expected_sha256=pin(ROOT, registry)["sha256"],
        )
        for k, c in controls.items()
    }
    checked = {}

    def run(label, control, parts, target, expected, program=None):
        result = (program or programs[control]).replay(parts, target)
        if result["computed_consistency_pass"] is not expected:
            raise ValueError(f"Unexpected adversarial disposition: {label}")
        checked[label] = {"expected_computed_consistency_pass": expected, "replay": result}

    for k, c in controls.items():

        def reorder(s):
            m = Chem.MolFromSmiles(s)
            return Chem.MolToSmiles(Chem.RenumberAtoms(m, list(reversed(range(m.GetNumAtoms())))))

        parts = {r: reorder(s) for r, s in c["components"].items()}
        run(k + "_atom_order", k, parts, reorder(c["expected_product"]), True)
        run(
            k + "_outcome_bound",
            k,
            c["components"],
            c["expected_product"],
            False,
            replace(programs[k], bounds=RepeatBounds(maximum_outcomes=1)),
        )
    c = controls["I91"]
    parts = copy.deepcopy(c["components"])
    parts["tail_acid_1"], parts["tail_acid_2"] = parts["tail_acid_2"], parts["tail_acid_1"]
    run("heterotail_source_sites_swapped", "I91", parts, c["expected_product"], False)
    # A homo-pair program must fail even when the supplied unequal pair reconstructs
    # the exact target of another source architecture and all mass checks pass.
    run(
        "homo_program_cannot_accept_heteropair",
        "I95",
        c["components"],
        c["expected_product"],
        False,
    )
    for key, replacement in [
        ("vitamin_b5_core", "CC(C)(CO)CC(O)C(=O)NCCC(=O)O"),
        ("head_nucleophile", "CN(C)CCO"),
    ]:
        run(
            "wrong_source_" + key,
            "I91",
            {**c["components"], key: replacement},
            c["expected_product"],
            False,
        )
    # Mechanical reconstruction alone does not prove the complete precursor is
    # inside the qualified source design. These constructed counterexamples are
    # deliberately retained as failing readiness evidence for this draft.
    gaps = {}
    for name, key, role, value in [
        (
            "I7_unequal_internal_arms",
            "I71",
            "tail6_alcohol",
            "OCC(OC(=O)C(CCCCCC)CCCCCCCC)COC(=O)C(CCCCCC)CCCCCCCCC",
        ),
        ("I8_secondary_tail_alcohol", "I81", "r2_tail_alcohol", "CCCCCCCCCC(O)C"),
        ("I8_unreported_head_scaffold", "I81", "head_acid", "O=C(O)CCN1CCCCC1"),
    ]:
        parts = {**controls[key]["components"], role: value}
        p = programs[key]
        state = constitutional_molecule(parts[p.specification["initial_role"]])[0]
        for stage, adapter in zip(p.specification["stages"], p.adapters, strict=True):
            products = adapter.forward_products(
                {stage["accumulator_role"]: state, **{r: parts[r] for r in stage["added_roles"]}},
                maximum_outcomes=256,
            )
            if products.saturated or len(products.products) != 1:
                raise ValueError("Scope diagnostic lacks unique mechanical reconstruction")
            state = products.products[0]
        result = p.replay(parts, state)
        if not result["computed_consistency_pass"]:
            raise ValueError("Expected draft scope gap disappeared; inspect before qualification")
        gaps[name] = {
            "components": parts,
            "target_for_synthetic_regression_only": state,
            "draft_mechanical_consistency_pass": True,
            "required_final_scope_disposition": "abstain_outside_qualified_complete_precursor_design",
            "training_admitted": False,
            "replay": result,
        }
    dump(
        HERE / "draft-adversarial-audit.json",
        {
            "schema_version": "forge.b5_draft_adversarial_audit.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {"registry": pin(ROOT, registry), "source_controls": pin(ROOT, source)},
            "mechanical_invariant_checks": checked,
            "remaining_complete_terminal_scope_counterexamples": gaps,
            "corpus_replay_qualified": False,
            "training_admitted": False,
            "next_required_work": [
                "Registry-backed complete head-scaffold/spacer grammar separated by I7/I8/I9",
                "Primary-only source alcohols and complete paired-tail equality",
                "Explicit original metadata-to-program binding with original roles, global IDs and quantities preserved",
                "Adversarial and full eligible-family replay after these gates",
            ],
        },
    )
    print(
        f"{len(checked)} mechanical checks passed; {len(gaps)} draft scope gaps retained; no corpus admission"
    )


if __name__ == "__main__":
    main()

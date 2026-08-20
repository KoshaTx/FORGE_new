#!/usr/bin/env python3
"""Verify planner proposals against the engine's OWN template library.

The repository's independent forward resolver knows eleven hand-curated
transforms.  Adjudicating ~48,000-template engine output against eleven local
transforms measures our bookkeeping, not chemistry: 1,094 of 1,187 planner
proposals came back "new family hypothesis retained", which means unverifiable
by us rather than invalid.

This runs the same *idea* — mechanical forward reconstruction — using the
engine's own retro template, reversed and applied forward to the proposed
reactants.  It owes nothing to the local transform set.

Three declared signals per proposal, none of which is "a model said so":

  A  self_consistent      the engine's own template, run forward on its own
                          proposed reactants, reproduces the exact target
  B  library_occurence    how many documented USPTO reactions instantiate this
                          template (a count, not an opinion)
  E  disconnection_class  skeletal construction vs functional-group
                          interconversion, from carbon-count change

None of these establishes chemoselectivity, substrate scope, evidence tier or
procurement.  Self-consistency filters hallucination; it does not predict that
a reaction works.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_DEFAULT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DEFAULT / "src"))

from experiments.phase1.synthesis_guidance.route_cascade import (  # noqa: E402
    NOT_ASSESSED_OUTCOME,
    UgiBoundedHybridRouteCascadeError,
    atomic_write,
    canonical_json_bytes,
    jsonl_gzip_bytes,
    load_json,
    read_jsonl_gzip,
    sha256_file,
    sha256_payload,
)

RESULT_SCHEMA_VERSION = "phase1_ugi_engine_template_verification.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_engine_template_verification_ledger.v1"

AUTHORITY = {
    "self_consistency_is_a_mechanical_check": True,
    "self_consistency_is_chemoselectivity_proof": False,
    "self_consistency_is_substrate_scope_proof": False,
    "self_consistency_is_route_evidence": False,
    "library_occurence_is_substrate_scope": False,
    "may_close_route": False,
    "verification_uses_engine_templates_not_local_transforms": True,
}

RETAINED_METADATA = (
    "policy_name",
    "policy_probability",
    "policy_probability_rank",
    "template_code",
    "template_hash",
    "classification",
    "library_occurence",
    "mapped_reaction_smiles",
)


def load_template_library(repo: Path) -> dict[str, dict[str, str]]:
    """Index the engine's own retro templates by template code."""

    library: dict[str, dict[str, str]] = {}
    for name in ("uspto_templates.csv.gz", "uspto_ringbreaker_templates.csv.gz"):
        path = repo / "data/source_cache/aizynthfinder_public_v4_4_1" / name
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                key = f"{name}:{row['template_code']}"
                library[key] = row
                library.setdefault(str(row["template_hash"]), row)
    return library


def reverse_retro_template(retro: str) -> str | None:
    """Turn ``product>>reactants`` into ``reactants>>product``."""

    if not isinstance(retro, str) or retro.count(">>") != 1:
        return None
    product_side, reactant_side = retro.split(">>")
    if not product_side.strip() or not reactant_side.strip():
        return None
    return f"{reactant_side}>>{product_side}"


def carbon_count(smiles: str) -> int:
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return 0
    return sum(1 for atom in molecule.GetAtoms() if atom.GetSymbol() == "C")


def disconnection_class(target: str, reactants: list[str]) -> str:
    """Separate skeletal construction from functional-group interconversion.

    A disconnection whose largest precursor keeps essentially the whole carbon
    skeleton has not built anything; it has moved the problem one step back.
    """

    target_carbons = carbon_count(target)
    if target_carbons == 0 or not reactants:
        return "undetermined"
    largest = max(carbon_count(value) for value in reactants)
    ratio = largest / target_carbons
    if ratio >= 0.9:
        return "functional_group_interconversion"
    if ratio >= 0.5:
        return "partial_skeletal_construction"
    return "skeletal_construction"


def run(repo: Path, output_dir: Path, maximum_proposals: int) -> dict[str, Any]:
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdChemReactions

    cascade_ledger = repo / (
        "results/phase1/ugi_bounded_hybrid_route_cascade_v1/component_route_ledger.jsonl.gz"
    )
    component_rows = read_jsonl_gzip(cascade_ledger, label="component route ledger")
    targets = [
        row
        for row in component_rows
        if row["final_component_state"] != "complete"
        and row["exact_evidence"]["assessment_outcome"] != NOT_ASSESSED_OUTCOME
    ]
    targets.sort(key=lambda row: (str(row["role"]), str(row["canonical_smiles"])))

    runtime_manifest_path = repo / (
        "configs/route/aizynthfinder_public_v4_4_1_diagnostic_macos_arm64_v1.json"
    )
    runtime_manifest = load_json(runtime_manifest_path, label="AiZynthFinder runtime manifest")
    engine_config = repo / runtime_manifest["config"]["path"]
    if sha256_file(engine_config) != runtime_manifest["config"]["sha256"]:
        raise UgiBoundedHybridRouteCascadeError("AiZynthFinder engine config changed")

    library = load_template_library(repo)
    print(f"template library entries indexed: {len(library)}", flush=True)

    from aizynthfinder.aizynthfinder import AiZynthExpander

    expander = AiZynthExpander(configfile=str(engine_config))
    expander.expansion_policy.select(["uspto", "ringbreaker"])
    expander.filter_policy.select(["uspto"])

    def canonical(value: str) -> str | None:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(value)
        return None if molecule is None else Chem.MolToSmiles(molecule, canonical=True)

    rows: list[dict[str, Any]] = []
    per_component: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        smiles = str(target["canonical_smiles"])
        target_canonical = canonical(smiles)
        groups = expander.do_expansion(smiles, return_n=maximum_proposals)
        component_rows_out: list[dict[str, Any]] = []
        for rank, group in enumerate(groups, start=1):
            if not group:
                continue
            reaction = group[0]
            metadata = dict(getattr(reaction, "metadata", {}))
            retained = {key: metadata.get(key) for key in RETAINED_METADATA}
            outcomes = getattr(reaction, "reactants", ())
            reactants = []
            if outcomes:
                for molecule in outcomes[0]:
                    value = canonical(molecule.smiles)
                    if value:
                        reactants.append(value)
            reactants = sorted(reactants)

            # Rung A: forward-reconstruct with the engine's own template.
            template_row = None
            for key in (
                f"uspto_templates.csv.gz:{retained.get('template_code')}",
                f"uspto_ringbreaker_templates.csv.gz:{retained.get('template_code')}",
                str(retained.get("template_hash")),
            ):
                if key in library:
                    template_row = library[key]
                    break
            self_consistent = False
            verification_note = "template_not_found_in_library"
            occurence = None
            if template_row is not None:
                occurence = template_row.get("library_occurence")
                forward = reverse_retro_template(template_row.get("retro_template", ""))
                if forward is None:
                    verification_note = "retro_template_unparseable"
                else:
                    try:
                        with rdBase.BlockLogs():
                            rxn = rdChemReactions.ReactionFromSmarts(forward)
                            mols = tuple(Chem.MolFromSmiles(value) for value in reactants)
                            if rxn is None or any(m is None for m in mols):
                                verification_note = "reaction_or_reactants_unparseable"
                            else:
                                produced: set[str] = set()
                                for ordering in (mols, tuple(reversed(mols))):
                                    try:
                                        for outcome in rxn.RunReactants(ordering):
                                            for product in outcome:
                                                try:
                                                    Chem.SanitizeMol(product)
                                                except Exception:  # noqa: BLE001
                                                    continue
                                                produced.add(
                                                    Chem.MolToSmiles(product, canonical=True)
                                                )
                                    except Exception:  # noqa: BLE001
                                        continue
                                self_consistent = target_canonical in produced
                                verification_note = (
                                    "target_reconstructed"
                                    if self_consistent
                                    else f"target_absent_from_{len(produced)}_forward_products"
                                )
                    except Exception as error:  # noqa: BLE001 - recorded, never dropped
                        verification_note = f"forward_error:{type(error).__name__}"
            row = {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "component_sha256": target["component_sha256"],
                "role": target["role"],
                "target_smiles": smiles,
                "rank": rank,
                "reactants": reactants,
                "self_consistent_under_engine_template": self_consistent,
                "verification_note": verification_note,
                "library_occurence": None if occurence is None else int(float(occurence)),
                "classification": retained.get("classification"),
                "template_code": retained.get("template_code"),
                "template_hash": retained.get("template_hash"),
                "policy_probability": retained.get("policy_probability"),
                "disconnection_class": disconnection_class(smiles, reactants),
                "authority": dict(AUTHORITY),
            }
            rows.append(row)
            component_rows_out.append(row)
        verified = [r for r in component_rows_out if r["self_consistent_under_engine_template"]]
        per_component.append(
            {
                "component_sha256": target["component_sha256"],
                "role": target["role"],
                "canonical_smiles": smiles,
                "occurrence_count": int(target["occurrence_count"]),
                "proposals": len(component_rows_out),
                "self_consistent_proposals": len(verified),
                "any_self_consistent": bool(verified),
                "best_library_occurence": max(
                    (r["library_occurence"] or 0 for r in verified), default=0
                ),
                "any_skeletal_construction": any(
                    r["disconnection_class"] == "skeletal_construction" for r in verified
                ),
            }
        )
        if index % 20 == 0 or index == len(targets):
            print(f"template-verify {index}/{len(targets)}", flush=True)

    rows.sort(key=lambda r: (str(r["role"]), str(r["target_smiles"]), int(r["rank"])))
    per_component.sort(key=lambda r: (str(r["role"]), str(r["canonical_smiles"])))
    ledger_path = output_dir / "engine_template_verification_ledger.jsonl.gz"
    atomic_write(ledger_path, jsonl_gzip_bytes(rows))

    verified_rows = [r for r in rows if r["self_consistent_under_engine_template"]]
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "engine_template_verification_complete",
        "inputs": {
            "component_route_ledger": {
                "path": str(cascade_ledger.relative_to(repo)),
                "sha256": sha256_file(cascade_ledger),
            },
            "runtime_manifest": {
                "path": str(runtime_manifest_path.relative_to(repo)),
                "sha256": sha256_file(runtime_manifest_path),
            },
        },
        "settings": {"maximum_proposals": maximum_proposals},
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "summary": {
            "components": len(per_component),
            "proposals": len(rows),
            "self_consistent_proposals": len(verified_rows),
            "self_consistent_fraction": (round(len(verified_rows) / len(rows), 4) if rows else 0.0),
            "components_with_any_self_consistent_proposal": sum(
                1 for r in per_component if r["any_self_consistent"]
            ),
            "components_by_role_with_self_consistent": dict(
                sorted(
                    Counter(r["role"] for r in per_component if r["any_self_consistent"]).items()
                )
            ),
            "verification_notes": dict(
                sorted(Counter(r["verification_note"] for r in rows).items())
            ),
            "disconnection_classes_all": dict(
                sorted(Counter(r["disconnection_class"] for r in rows).items())
            ),
            "disconnection_classes_self_consistent": dict(
                sorted(Counter(r["disconnection_class"] for r in verified_rows).items())
            ),
            "components_with_self_consistent_skeletal_construction": sum(
                1 for r in per_component if r["any_skeletal_construction"]
            ),
            "candidate_occurrences_behind_self_consistent_components": sum(
                r["occurrence_count"] for r in per_component if r["any_self_consistent"]
            ),
        },
        "per_component": per_component,
        "artifacts": {
            "engine_template_verification_ledger.jsonl.gz": {
                "path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
                "rows": len(rows),
                "schema_version": LEDGER_SCHEMA_VERSION,
            }
        },
        "authority": dict(AUTHORITY),
        "interpretation": (
            "Self-consistency uses the engine's own retro template, reversed and applied forward "
            "to the engine's own proposed reactants. It owes nothing to the eleven local "
            "transforms. It filters hallucinated disconnections; it does not establish "
            "chemoselectivity, substrate scope, evidence tier or procurement."
        ),
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / "result.json", canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_engine_template_verification_v1"),
    )
    parser.add_argument("--maximum-proposals", type=int, default=10)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    result = run(repo, output_dir.resolve(), args.maximum_proposals)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

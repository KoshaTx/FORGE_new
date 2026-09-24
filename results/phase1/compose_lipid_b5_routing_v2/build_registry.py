"""Derive narrow I9 source-pair routing without changing original generator labels."""

import json
from pathlib import Path

from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]


def main():
    parent = ROOT / "data/vendor/qualified_b5_staged_source_program_v1.json"
    registry = json.loads(parent.read_text())
    routes = []
    for kind in ("amine", "alcohol"):
        original = "one_tail_acid_knob_" + kind
        source = "matched_identical_tail_acids_" + kind
        profile = registry["original_task_contract"]["profiles"][source]
        routes.append(
            {
                "route_id": "source_I9_compound_11_pair_" + kind,
                "original_profile": original,
                "source_profile": source,
                "task_fields": {"tail_axis": "reported_tail_acid__reported_tail_acid"},
                "required_components": {
                    role: profile["tail_rules"][role]["reported_tail_acid"]["smiles"][0]
                    for role in profile["tail_roles"]
                },
                "minimum_reported_tails": 2,
            }
        )
    result = {
        "schema_version": "forge.b5_source_pair_routing.v2",
        "parent_registry": pin(ROOT, parent),
        "derivation": pin(ROOT, Path(__file__).resolve()),
        "source_assets": registry["source_assets"],
        "routes": routes,
        "source_adjudication": {
            "doi": "10.1002/adhm.202403366",
            "locators": [
                "Supplement PDF page 25, Figure S3: I9 homo and hetero branches",
                "Supplement PDF page 32, compound 26: two source acid 11 residues",
                "Supplement PDF page 33, compound 27: Fm removal preserves both glutarates",
                "Supplement PDF pages 34-35, I95 and I97 head amide/ester products",
            ],
            "finding": "The original one_tail_acid_knob lane includes the exact 11/11 source pair. A generation lane does not identify the hetero 18/11 chemistry. Route only the complete 11/11 pair to the existing homo source program; preserve the original task, metadata, head grammar, component IDs, quantities and stage gates.",
            "evidence_basis": "computed_transform_consistency",
            "experimental_execution_admitted": False,
            "remaining_scope": "No new tail identity, mixed pair, site order, source series, reaction primitive or precursor scope is admitted. Other original reported-tail disagreements remain pending.",
            "source_conflicts": "Retain all analytical and procedure discrepancies from the parent source registry and transcription ledger.",
        },
        "training_admitted": False,
    }
    dump(ROOT / "data/vendor/qualified_b5_source_pair_routing_v2.json", result)


if __name__ == "__main__":
    main()

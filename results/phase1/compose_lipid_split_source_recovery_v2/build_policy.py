"""Transcribe source split descriptors as data; preserve upstream source bytes."""

import ast
import json
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    receipt = HERE / "result.json"
    report = json.loads(receipt.read_text())
    paths = {
        a["upstream_path"]: resolve_pin(
            {k: a[k] for k in ("path", "sha256")}, ROOT, label=a["upstream_path"]
        )
        for a in report["assets"]
    }
    source = paths["scripts/build_post_instruction_generator_splits_v8_1.py"]
    tree = ast.parse(source.read_text())
    fields = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "MORPHOLOGY_FIELDS" for t in node.targets)
    )
    policy = {
        "schema_version": "forge.compose_lipid_frozen_split_policy.v1",
        "source_recovery": pin(ROOT, receipt),
        "derivation": pin(ROOT, Path(__file__).resolve()),
        "source_config": pin(
            ROOT, paths["configs/corpus/post_instruction_generator_splits_v8_1.json"]
        ),
        "digest": {
            "algorithm": "sha256",
            "sort_keys": True,
            "ensure_ascii": True,
            "separators": [",", ":"],
        },
        "signature_labels": {
            "core": "post_instruction_family_core_v8_1",
            "regional": "post_instruction_coarse_regional_morphology_v8_1",
            "morphology": "reaction_core_regional_morphology_v2",
            "combination": "complete_component_role_multiset_v2",
        },
        "metadata_fields": sorted(fields),
        "metadata_suffix": "_axis",
        "source_anchor_context": {"source_anchor": True},
        "elements": {"carbon": 6, "nitrogen": 7, "oxygen": 8, "phosphorus": 15, "sulfur": 16},
        "profile_fields": {
            "heavy_atom_band": {"feature": "heavy_atoms", "divisor": 8},
            "carbon_band": {"feature": "carbon", "divisor": 6},
            "formal_charge": {"feature": "formal_charge"},
            "ring_count": {"feature": "rings", "maximum": 4},
            "carbon_branch_points": {"feature": "carbon_branch_points", "maximum": 4},
            "cc_double_bonds": {"feature": "cc_double_bonds", "maximum": 4},
            "nitrogen_band": {"feature": "nitrogen", "divisor": 2},
            "oxygen_band": {"feature": "oxygen", "divisor": 2},
            "phosphorus_atoms": {"feature": "phosphorus"},
            "sulfur_band": {"feature": "sulfur", "divisor": 2},
        },
        "carbon_branch_minimum_carbon_neighbors": 3,
        "role_signature": "one_role_and_complete_component_id_per_incorporated_copy",
        "scope": "Exact source descriptor and frozen-group projection only; no new group selection, chemical decomposition, training admission or size cap.",
        "training_admitted": False,
    }
    dump(ROOT / "data/vendor/compose_lipid_frozen_split_policy_v1.json", policy)


if __name__ == "__main__":
    main()

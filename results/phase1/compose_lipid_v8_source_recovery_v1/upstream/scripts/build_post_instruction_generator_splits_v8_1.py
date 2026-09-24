"""Freeze structure-resolved grouped splits for the broad v8 lipid release."""

from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from compose_lipid.data.assets import sha256_file
from compose_lipid.data.generator_splits_v2 import make_generator_splits_v2
from compose_lipid.data.source_inventory import publish_json
from compose_lipid.data.training_corpus import digest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/corpus/post_instruction_generator_splits_v8_1.json"


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        yield from map(json.loads, handle)


def output_sha(receipt: dict, name: str) -> str:
    value = receipt["outputs"][name]
    return value["sha256"] if isinstance(value, dict) else value


def publish_gzip_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as raw:
        temporary = Path(raw.name)
        with gzip.GzipFile(
            fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6
        ) as compressed, io.TextIOWrapper(compressed, encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
        raw.flush()
        os.fsync(raw.fileno())
    if path.exists():
        if sha256_file(path) != sha256_file(temporary):
            temporary.unlink()
            raise ValueError("refusing to overwrite changed split output: " + path.name)
        temporary.unlink()
    else:
        os.replace(temporary, path)


def molecule_profile(smiles: str) -> dict:
    """Coarse whole-molecule morphology used only for grouped evaluation."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("split target does not parse")
    carbons = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6]
    return {
        "heavy_atom_band": molecule.GetNumHeavyAtoms() // 8,
        "carbon_band": len(carbons) // 6,
        "formal_charge": Chem.GetFormalCharge(molecule),
        "ring_count": min(4, rdMolDescriptors.CalcNumRings(molecule)),
        "carbon_branch_points": min(
            4,
            sum(
                sum(
                    neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()
                )
                >= 3
                for atom in carbons
            ),
        ),
        "cc_double_bonds": min(
            4,
            sum(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetBeginAtom().GetAtomicNum() == 6
                and bond.GetEndAtom().GetAtomicNum() == 6
                for bond in molecule.GetBonds()
            ),
        ),
        "nitrogen_band": sum(
            atom.GetAtomicNum() == 7 for atom in molecule.GetAtoms()
        )
        // 2,
        "oxygen_band": sum(
            atom.GetAtomicNum() == 8 for atom in molecule.GetAtoms()
        )
        // 2,
        "phosphorus_atoms": sum(
            atom.GetAtomicNum() == 15 for atom in molecule.GetAtoms()
        ),
        "sulfur_band": sum(
            atom.GetAtomicNum() == 16 for atom in molecule.GetAtoms()
        )
        // 2,
    }


MORPHOLOGY_FIELDS = {
    "architecture_stratum",
    "coupled_ketone_split",
    "design_lane",
    "distance",
    "event_count",
    "events",
    "head_arm_count",
    "interface_stratum",
    "mechanism",
    "occupancy",
    "pair_lane",
    "reported_released_tuple",
    "series",
    "site_disposition",
}


def morphology_metadata(metadata: dict) -> dict:
    """Retain declared design axes without treating IDs as morphology."""

    return {
        key: value
        for key, value in sorted(metadata.items())
        if key in MORPHOLOGY_FIELDS or key.endswith("_axis")
    }


def _bound_receipt(config: dict, key: str, bindings: dict) -> tuple[Path, dict]:
    path = ROOT / config[f"{key}_receipt"]
    expected = config[f"{key}_receipt_sha256"]
    if sha256_file(path) != expected:
        raise ValueError("changed parent receipt: " + key)
    receipt = json.loads(path.read_text())
    if receipt.get("complete") is not True:
        raise ValueError("incomplete parent receipt: " + key)
    bindings[str(path.relative_to(ROOT))] = expected
    return path.parent, receipt


def build() -> dict:
    config = json.loads(CONFIG.read_text())
    if any(config[key] for key in ("biology_used", "beae_outcomes_used", "gpu_used")):
        raise ValueError("split construction must remain chemistry-only and CPU-only")
    if config["training_admissible"]:
        raise ValueError("split assignment cannot itself admit training")

    bindings = {str(CONFIG.relative_to(ROOT)): sha256_file(CONFIG)}
    release_dir, release_receipt = _bound_receipt(config, "release", bindings)
    component_dir, component_receipt = _bound_receipt(
        config, "component_manifest", bindings
    )
    _bound_receipt(config, "superseded_split", bindings)
    release_path = release_dir / "accepted_targets.jsonl.gz"
    component_path = component_dir / "components.jsonl.gz"
    for path, receipt in (
        (release_path, release_receipt),
        (component_path, component_receipt),
    ):
        expected = output_sha(receipt, path.name)
        if sha256_file(path) != expected:
            raise ValueError("changed bound input: " + path.name)
        bindings[str(path.relative_to(ROOT))] = expected

    records = []
    family_counts = Counter()
    basis_counts = Counter()
    source_counts = Counter()
    for release, components in zip_longest(
        rows(release_path), rows(component_path), fillvalue=None
    ):
        if release is None or components is None:
            raise ValueError("release/component manifest row counts differ")
        target_id = release["target_id"]
        family = release["primary_family"]
        if components["target_id"] != target_id or components["family"] != family:
            raise ValueError("release/component manifest order or identity changed")
        instances = [
            {"role": item["role"], "component_id": item["component_id"]}
            for item in components["component_instances"]
        ]
        component_ids = sorted({item["component_id"] for item in instances})
        if not component_ids:
            raise ValueError("target lacks structure-resolved components")
        core = digest(["post_instruction_family_core_v8_1", family])
        profile = molecule_profile(release["constitution"])
        context = (
            {"source_anchor": True}
            if release["source_anchor"]
            else morphology_metadata(release["primary_metadata"])
        )
        regional_morphology = digest(
            [
                "post_instruction_coarse_regional_morphology_v8_1",
                family,
                context,
                profile,
                Counter(item["role"] for item in instances),
            ]
        )
        records.append(
            {
                "target_id": target_id,
                "family": family,
                "component_ids": component_ids,
                "component_instances": instances,
                "core_scaffold_signature": core,
                "regional_morphology_signature": regional_morphology,
                "regional_profile": profile,
                "morphology_context": context,
                "component_binding_basis": components["binding_basis"],
                "source_anchor": release["source_anchor"],
                "source_pmids": components.get("source_pmids", []),
                "source_names": components.get("source_names", []),
                "evidence_lane": release["admission_lane"],
            }
        )
        family_counts[family] += 1
        basis_counts[components["binding_basis"]] += 1
        source_counts["source" if release["source_anchor"] else "virtual"] += 1

    if len(records) != config["expected_targets"]:
        raise ValueError("split target membership changed")
    if len({row["target_id"] for row in records}) != len(records):
        raise ValueError("duplicate split target")
    trainable = {
        family
        for family, count in family_counts.items()
        if count >= config["minimum_training_class"]
    }
    if len(trainable) != config["expected_trainable_families"]:
        raise ValueError("formal family count changed")

    result = make_generator_splits_v2(
        records,
        seed=config["seed"],
        test_fraction=config["test_fraction"],
        calibration_fraction=config["calibration_fraction"],
        minimum_training_class=config["minimum_training_class"],
    )
    assignments = result.pop("assignments")
    split_counts = Counter(row["split"] for row in assignments)
    panel_counts = Counter(
        panel for row in assignments for panel in row["test_panels"]
    )
    test_without_panel = sum(
        row["split"] == "test" and not row["test_panels"] for row in assignments
    )
    summary = {
        "schema": "post_instruction_generator_splits_v8_1_summary",
        "targets": len(assignments),
        "source_targets": source_counts["source"],
        "virtual_targets": source_counts["virtual"],
        "formal_evaluation_families": result["formal_evaluation_families"],
        "reference_only_families": result["reference_only_families"],
        "split_counts": dict(sorted(split_counts.items())),
        "panel_counts": dict(sorted(panel_counts.items())),
        "family_split_counts": result["family_split_counts"],
        "test_targets_without_named_panel": test_without_panel,
        "selected_unseen_component_structures": len(result["selected_components"]),
        "selected_regional_morphology_groups": len(
            result["selected_morphology_groups"]
        ),
        "selected_exact_component_combination_groups": len(
            result["selected_combination_groups"]
        ),
        "selected_source_studies": len(result["selected_source_studies"]),
        "component_binding_basis_counts": dict(sorted(basis_counts.items())),
        "component_identity": "global canonical stereo-free complete-component constitution",
        "combination_contract": "all component structures observed in the same family training set",
        "study_contract": "selected PMID absent globally from all formal-family training rows",
        "morphology_contract": "coarse regional morphology grouping, not exact graph topology",
        "supersedes": "post_instruction_generator_splits_v8 for model-facing evaluation",
        "grouped_splits_frozen": True,
        "target_measure_frozen": False,
        "biology_used": False,
        "beae_outcomes_used": False,
        "gpu_used": False,
        "training_ready": False,
        "training_admissible": False,
    }
    if test_without_panel:
        raise ValueError("formal test rows lack a named evaluation panel")

    out = ROOT / config["output_dir"]
    publish_gzip_jsonl(out / "split_groups.jsonl.gz", records)
    publish_gzip_jsonl(out / "assignments.jsonl.gz", assignments)
    publish_json(out / "selected_groups.json", result)
    publish_json(out / "summary.json", summary)
    producers = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (
            Path(__file__),
            ROOT / "src/compose_lipid/data/generator_splits_v2.py",
        )
    }
    outputs = {
        name: sha256_file(out / name)
        for name in (
            "split_groups.jsonl.gz",
            "assignments.jsonl.gz",
            "selected_groups.json",
            "summary.json",
        )
    }
    publish_json(
        out / "receipt.json",
        {
            "schema": "post_instruction_generator_splits_v8_1_receipt",
            "complete": True,
            "inputs": dict(sorted(bindings.items())),
            "producers": producers,
            "outputs": outputs,
            "assignment_digest": digest(assignments),
            "grouped_splits_frozen": True,
            "training_ready": False,
            "training_admissible": False,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()

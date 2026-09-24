"""Freeze component-aware grouped splits for the balanced v8 lipid release."""

from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from compose_lipid.data.assets import sha256_file
from compose_lipid.data.generator_splits import make_generator_splits
from compose_lipid.data.source_inventory import publish_json
from compose_lipid.data.training_corpus import digest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/corpus/post_instruction_generator_splits_v8.json"


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        yield from map(json.loads, handle)


def output_sha(receipt: dict, name: str) -> str:
    value = receipt["outputs"][name]
    return value["sha256"] if isinstance(value, dict) else value


def row_at(path: Path, line_number: int) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for index, line in enumerate(handle, 1):
            if index == line_number:
                return json.loads(line)
    raise ValueError(f"missing line {line_number} in {path}")


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
                sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) >= 3
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
        ) // 2,
        "oxygen_band": sum(
            atom.GetAtomicNum() == 8 for atom in molecule.GetAtoms()
        ) // 2,
        "phosphorus_atoms": sum(
            atom.GetAtomicNum() == 15 for atom in molecule.GetAtoms()
        ),
        "sulfur_band": sum(
            atom.GetAtomicNum() == 16 for atom in molecule.GetAtoms()
        ) // 2,
    }


def instance(role: str, identifier: str, multiplicity: int = 1) -> list[dict]:
    if not identifier or multiplicity < 1:
        raise ValueError("invalid component instance")
    return [
        {"role": role, "precursor_id": identifier} for _ in range(multiplicity)
    ]


def virtual_instances(family: str, metadata: dict) -> list[dict]:
    """Normalize complete family components already bound by exact enumeration."""

    p = lambda role, value: f"{family}:{role}:{value}"
    if family == "preassembled_thiol_yne_tail_amidation":
        return (
            instance("amine_head", metadata["head_id"])
            + instance("alkynoic_linker", p("linker", metadata["linker_code"]))
            + instance("thiol_tail", p("tail", metadata["tail_code"]), 2)
        )
    if family == "o_esterification":
        return instance("aminoalcohol_head", metadata["head_id"]) + instance(
            "acid_tail", p("acid", metadata["acid_code"]), metadata["occupancy"]
        )
    if family == "aldehyde_ugi4":
        roles = ("amine", "aldehyde", "carboxylic_acid", "isocyanide")
        return [
            {"role": role, "precursor_id": identifier}
            for role, identifier in zip(roles, metadata["precursor_ids"], strict=True)
        ]
    if family == "ketone_ugi4":
        return (
            instance("amine_head", metadata["head_id"])
            + instance("coupled_ketone", p("ketone", metadata["ketone_code"]))
            + instance("isocyanide", p("isocyanide", metadata["isocyanide_code"]))
            + instance("carboxylic_acid", p("acid", metadata["acid_code"]))
        )
    if family == "aldehyde_ugi3":
        roles = ("amine", "aldehyde", "isocyanide")
        return [
            {"role": role, "precursor_id": identifier}
            for role, identifier in zip(roles, metadata["precursor_ids"], strict=True)
        ]
    if family == "a3_amine_aldehyde_alkyne":
        events = metadata["events"]
        return (
            instance("amine_head", metadata["head_id"])
            + instance("aldehyde", metadata["aldehyde_id"], events)
            + instance("alkyne", metadata["alkyne_id"], events)
        )
    if family == "aema_aza_thiol_addition":
        return instance("amine_core", metadata["core_id"]) + instance(
            "thiol_periphery", metadata["thiol_id"]
        )
    if family in {"aza_michael_acrylate", "aza_michael_acrylamide"}:
        return instance("amine_head", metadata["head_id"]) + instance(
            "acceptor_tail", metadata["tail_id"], metadata["occupancy"]
        )
    if family == "amine_epoxide_opening":
        return instance("amine_head", metadata["head_id"]) + instance(
            "epoxide_tail", metadata["tail_id"], metadata["occupancy"]
        )
    if family == "epoxide_opening_o_acylation":
        return (
            instance("amine_head", metadata["head_id"])
            + instance("epoxide_tail", metadata["epoxide_tail_id"], 2)
            + instance("acyl_tail", metadata["acyl_tail_id"], 2)
        )
    if family in {
        "alpha_isocyanoester_dihydroimidazole",
        "ketone_isocyanide_amide",
    }:
        return (
            instance("amine_head", metadata["head_id"])
            + instance("coupled_ketone", metadata["ketone_id"])
            + instance("isocyanide", metadata["isocyanide_id"])
        )
    if family == "amine_alkylation":
        return instance("amine_head", metadata["head_id"]) + instance(
            "bromoester_arm", metadata["arm_id"], 2
        )
    if family in {"reductive_amination", "aryl_reductive_amination"}:
        return instance("amine_head", metadata["head_id"]) + instance(
            "coupled_aldehyde", metadata["aldehyde_id"]
        )
    if family == "disulfide_michael":
        return instance("amine_head", metadata["head_id"]) + instance(
            "disulfide_acceptor", metadata["arm_id"], metadata["occupancy"]
        )
    if family == "thiolactone_aminolysis_michael":
        return (
            instance("amine_head", metadata["head_id"])
            + instance("thiolactone_region", metadata["thiolactone_id"])
            + instance("acrylate_tail", metadata["acrylate_id"])
        )
    if family == "iphos_ring_opening":
        return instance("amine_head", metadata["head_id"]) + instance(
            "phosphate_tail", metadata["phosphate_id"], metadata["event_count"]
        )
    if family == "maleate_addition":
        return instance("amine_head", metadata["head_id"]) + instance(
            "maleate", metadata["arm_id"], metadata["occupancy"]
        )
    if family == "acid_epoxide_diester_multistep":
        return (
            instance("amine_acid_head", metadata["head_id"])
            + instance("epoxide", metadata["epoxide_id"])
            + instance("hydrophobic_acid", metadata["hydrophobic_acid_id"])
        )
    if family == "passerini_3cr":
        return (
            instance("amine_acid_head", metadata["head_id"])
            + instance("aldehyde_tail", metadata["aldehyde_id"])
            + instance("isocyanide_tail", metadata["isocyanide_id"])
        )
    if family == "vitamin_b5_multistep":
        return instance("series_head", metadata["head_id"]) + instance(
            "series_tail_design",
            p("tail", metadata["series"] + ":" + metadata["tail_axis"]),
        )
    raise ValueError("unregistered virtual family components: " + family)


TOPOLOGY_FIELDS = {
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


def topology_metadata(metadata: dict) -> dict:
    return {
        key: value
        for key, value in sorted(metadata.items())
        if key in TOPOLOGY_FIELDS or key.endswith("_axis")
    }


def source_instances(rows_: list[dict]) -> list[dict]:
    normalized = []
    for row in rows_:
        roles = row.get("roles") or [row.get("role")]
        role = "+".join(sorted(value for value in roles if value))
        identifier = row.get("precursor_id")
        if not role or not identifier:
            raise ValueError("source component instance lacks role or identity")
        normalized.append({"role": role, "precursor_id": identifier})
    return normalized


def build() -> dict:
    config = json.loads(CONFIG.read_text())
    if any(config[key] for key in ("biology_used", "beae_outcomes_used", "gpu_used")):
        raise ValueError("split construction must remain chemistry-only and CPU-only")
    if config["training_admissible"]:
        raise ValueError("split assignment cannot admit training")
    bindings = {str(CONFIG.relative_to(ROOT)): sha256_file(CONFIG)}
    receipts = {}
    for key in ("release", "supervision", "source_corpus", "source_supervision"):
        path = ROOT / config[f"{key}_receipt"]
        expected = config[f"{key}_receipt_sha256"]
        if sha256_file(path) != expected:
            raise ValueError("changed parent receipt: " + key)
        receipt = json.loads(path.read_text())
        if receipt.get("complete") is not True:
            raise ValueError("incomplete parent receipt: " + key)
        receipts[key] = (path.parent, receipt)
        bindings[str(path.relative_to(ROOT))] = expected

    release_dir, release_receipt = receipts["release"]
    supervision_dir, supervision_receipt = receipts["supervision"]
    source_dir, source_receipt = receipts["source_corpus"]
    source_supervision_dir, source_supervision_receipt = receipts["source_supervision"]
    release_path = release_dir / "accepted_targets.jsonl.gz"
    supervision_path = supervision_dir / "manifest.jsonl.gz"
    source_path = source_dir / "accepted_targets.jsonl.gz"
    source_precursor_path = source_supervision_dir / "precursors.jsonl"
    for path, receipt in (
        (release_path, release_receipt),
        (supervision_path, supervision_receipt),
        (source_path, source_receipt),
        (source_precursor_path, source_supervision_receipt),
    ):
        expected = output_sha(receipt, path.name)
        if sha256_file(path) != expected:
            raise ValueError("changed bound input: " + path.name)
        bindings[str(path.relative_to(ROOT))] = expected

    source_evidence = {
        row["target_id"]: row for row in rows(source_path) if row["source_anchor"]
    }
    source_supervision = {
        row["target_id"]: row
        for row in rows(supervision_path)
        if row["source_anchor"]
    }
    if len(source_evidence) != 11_267 or set(source_evidence) != set(source_supervision):
        raise ValueError("source anchor evidence/supervision membership changed")
    muscle_overrides = {}
    for supervision in rows(supervision_path):
        if supervision.get("source_anchor") or supervision.get("family") != "aldehyde_ugi3":
            continue
        reference = supervision.get("program_reference", {})
        if "muscle_bounded_contexts" not in reference.get("path", ""):
            continue
        payload = row_at(ROOT / reference["path"], reference["line"])
        components = []
        for precursor in payload["precursors"]:
            role, smiles = precursor["role"], precursor["smiles"]
            components.append(
                {
                    "role": role,
                    "precursor_id": digest(
                        ["post_instruction_complete_component_v8", role, smiles]
                    ),
                }
            )
        muscle_overrides[supervision["target_id"]] = components
    if len(muscle_overrides) != 12:
        raise ValueError("MUSCLE bounded-context component binding changed")

    records = []
    source_groups = {}
    basis_counts = Counter()
    family_counts = Counter()
    for row in rows(release_path):
        target_id = row["target_id"]
        family = row["primary_family"]
        if row["source_anchor"]:
            supervision = source_supervision[target_id]
            instances = source_instances(supervision["precursor_instances"])
            evidence = source_evidence[target_id]
            studies = sorted(
                {f"pmid:{value}" for value in evidence.get("source_pmids", []) if str(value).isdigit()}
            )
            source_groups[target_id] = studies
            topology_context = {"source_anchor": True}
            basis = "reported_source_bound_components"
        else:
            instances = muscle_overrides.get(target_id) or virtual_instances(
                family, row["primary_metadata"]
            )
            topology_context = topology_metadata(row["primary_metadata"])
            studies = []
            basis = "v8_exact_virtual_bound_components"
        precursor_ids = sorted({item["precursor_id"] for item in instances})
        if not precursor_ids:
            raise ValueError("target lacks complete-component identities")
        core = digest(["post_instruction_family_core_v8", family])
        molecular_profile = molecule_profile(row["constitution"])
        topology = digest(
            [
                "post_instruction_regional_topology_v8",
                family,
                topology_context,
                molecular_profile,
                Counter(item["role"] for item in instances),
            ]
        )
        records.append(
            {
                "target_id": target_id,
                "family": family,
                "precursor_ids": precursor_ids,
                "precursor_instances": instances,
                "reaction_program_signature": core,
                "core_scaffold_signature": core,
                "regional_topology_signature": topology,
                "regional_profile": molecular_profile,
                "topology_context": topology_context,
                "profile_basis": basis,
                "reaction_core_profile_available": family.startswith("reported_") is False,
                "source_anchor": row["source_anchor"],
                "source_pmids": [value.removeprefix("pmid:") for value in studies],
                "source_names": source_evidence.get(target_id, {}).get("source_names", []),
                "evidence_lane": row["admission_lane"],
            }
        )
        basis_counts[basis] += 1
        family_counts[family] += 1

    if len(records) != config["expected_targets"] or len({r["target_id"] for r in records}) != len(records):
        raise ValueError("split target membership changed")
    trainable = {family for family, count in family_counts.items() if count >= config["minimum_training_class"]}
    if len(trainable) != config["expected_trainable_families"]:
        raise ValueError("formal family count changed")

    result = make_generator_splits(
        records,
        source_groups,
        seed=config["seed"],
        test_fraction=config["test_fraction"],
        calibration_fraction=config["calibration_fraction"],
        minimum_training_class=config["minimum_training_class"],
    )
    assignments = result.pop("assignments")
    split_counts = Counter(row["split"] for row in assignments)
    family_split = Counter((row["family"], row["split"]) for row in assignments)
    panel_counts = Counter(
        panel for row in assignments for panel in row["test_panels"]
    )
    summary = {
        "schema": "post_instruction_generator_splits_v8_summary",
        "targets": len(assignments),
        "formal_evaluation_families": result["formal_evaluation_families"],
        "reference_only_families": result["reference_only_families"],
        "split_counts": dict(sorted(split_counts.items())),
        "panel_counts": dict(sorted(panel_counts.items())),
        "family_split_counts": [
            {"family": family, "split": split, "targets": count}
            for (family, split), count in sorted(family_split.items())
        ],
        "selected_unseen_components": len(result["selected_precursors"]),
        "selected_regional_topology_groups": len(result["selected_structural_groups"]),
        "selected_exact_combination_groups": len(result["selected_combination_groups"]),
        "selected_source_studies": len(result["selected_source_studies"]),
        "profile_basis_counts": dict(sorted(basis_counts.items())),
        "component_identity_grouping_used": True,
        "grouped_splits_frozen": True,
        "target_measure_frozen": False,
        "biology_used": False,
        "beae_outcomes_used": False,
        "gpu_used": False,
        "training_ready": False,
        "training_admissible": False,
    }
    out = ROOT / config["output_dir"]
    publish_gzip_jsonl(out / "split_groups.jsonl.gz", records)
    publish_gzip_jsonl(out / "assignments.jsonl.gz", assignments)
    publish_json(out / "selected_groups.json", result)
    publish_json(out / "summary.json", summary)
    producers = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (Path(__file__), ROOT / "src/compose_lipid/data/generator_splits.py")
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
            "schema": "post_instruction_generator_splits_v8_receipt",
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

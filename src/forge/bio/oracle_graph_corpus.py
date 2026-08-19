"""Freeze the leakage-safe graph corpus for the M0-07 oracle matrix."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.hashing import sha256_file as _sha256_file
from forge.core.io import atomic_write as _atomic_write

CONFIG_SCHEMA_VERSION = "m0_07_oracle_graph_corpus_config.v2"
RESULT_SCHEMA_VERSION = "m0_07_oracle_graph_corpus.v2"

RETAINED_FIELDS = (
    "graph_id",
    "constitutional_smiles",
    "atom_count",
    "undirected_bond_count",
    "directed_edge_count",
)
EXCLUDED_FIELDS = (
    "r0_id",
    "canonical_isomeric_smiles",
    "constitutional_smiles",
    "observed_source_ids",
    "matched_oracle_labels_json",
)
COLLAPSE_FIELDS = (
    "constitutional_smiles",
    "group_scope",
    "group_size",
    "r0_ids_json",
    "canonical_isomeric_smiles_json",
    "observed_source_ids_json",
)


class OracleGraphCorpusError(ValueError):
    """Raised when the graph-corpus audit violates its frozen contract."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise OracleGraphCorpusError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OracleGraphCorpusError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise OracleGraphCorpusError(f"{label} must contain an object")
    return value


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _verify_input(
    repo_root: Path,
    specification: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    relative = specification.get("path")
    expected = specification.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise OracleGraphCorpusError(f"{label} input specification is incomplete")
    path = repo_root / relative
    if not path.is_file():
        raise OracleGraphCorpusError(f"{label} not found: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise OracleGraphCorpusError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {"path": relative, "sha256": observed, "bytes": path.stat().st_size}


def _read_csv(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    try:
        with opener(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise OracleGraphCorpusError(f"{label} is missing fields: {sorted(missing)}")
            return [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise OracleGraphCorpusError(f"{label} not found: {path}") from exc


def _molecule(smiles: str, label: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles:
        raise OracleGraphCorpusError(f"{label} lacks SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise OracleGraphCorpusError(f"{label} contains invalid SMILES: {smiles!r}")
    return molecule


def _constitution(smiles: str, label: str) -> tuple[str, Chem.Mol]:
    molecule = _molecule(smiles, label)
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise OracleGraphCorpusError(f"{label} is disconnected")
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )
    return canonical, molecule


def _gzip_csv(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as archive:
        archive.write(text.getvalue().encode())
    return output.getvalue()


def _artifact_metadata(payload: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise OracleGraphCorpusError("cannot summarize an empty distribution")
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "minimum": min(values),
        "q1": quartiles[0],
        "median": statistics.median(values),
        "q3": quartiles[2],
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def _graph_profile(smiles: Sequence[str], label: str) -> dict[str, Any]:
    atom_counts: list[int] = []
    bond_counts: list[int] = []
    elements: Counter[str] = Counter()
    formal_charges: Counter[str] = Counter()
    total_degrees: Counter[str] = Counter()
    total_hydrogens: Counter[str] = Counter()
    hybridizations: Counter[str] = Counter()
    bond_types: Counter[str] = Counter()
    molecular_net_charges: Counter[str] = Counter()
    aromatic_atoms = 0
    ring_atoms = 0
    conjugated_bonds = 0
    ring_bonds = 0
    disconnected_records = 0

    for index, value in enumerate(smiles):
        molecule = _molecule(value, f"{label} row {index}")
        atom_counts.append(molecule.GetNumAtoms())
        bond_counts.append(molecule.GetNumBonds())
        molecular_net_charges[str(Chem.GetFormalCharge(molecule))] += 1
        disconnected_records += int(len(Chem.GetMolFrags(molecule)) != 1)
        for atom in molecule.GetAtoms():
            elements[atom.GetSymbol()] += 1
            formal_charges[str(atom.GetFormalCharge())] += 1
            total_degrees[str(atom.GetTotalDegree())] += 1
            total_hydrogens[str(atom.GetTotalNumHs())] += 1
            hybridizations[str(atom.GetHybridization())] += 1
            aromatic_atoms += int(atom.GetIsAromatic())
            ring_atoms += int(atom.IsInRing())
        for bond in molecule.GetBonds():
            bond_types[str(bond.GetBondType())] += 1
            conjugated_bonds += int(bond.GetIsConjugated())
            ring_bonds += int(bond.IsInRing())

    if not atom_counts:
        raise OracleGraphCorpusError(f"{label} graph profile is empty")
    return {
        "records": len(atom_counts),
        "atoms": sum(atom_counts),
        "undirected_bonds": sum(bond_counts),
        "directed_edges": 2 * sum(bond_counts),
        "atom_count_distribution": _distribution(atom_counts),
        "undirected_bond_count_distribution": _distribution(bond_counts),
        "records_at_most_64_atoms": sum(value <= 64 for value in atom_counts),
        "records_at_most_96_atoms": sum(value <= 96 for value in atom_counts),
        "records_above_96_atoms": sum(value > 96 for value in atom_counts),
        "disconnected_records": disconnected_records,
        "feature_vocabulary": {
            "elements": dict(sorted(elements.items())),
            "formal_charges": dict(sorted(formal_charges.items())),
            "total_degrees": dict(sorted(total_degrees.items())),
            "total_hydrogens": dict(sorted(total_hydrogens.items())),
            "hybridizations": dict(sorted(hybridizations.items())),
            "bond_types": dict(sorted(bond_types.items())),
            "molecular_net_charges": dict(sorted(molecular_net_charges.items())),
        },
        "binary_feature_counts": {
            "aromatic_atoms": aromatic_atoms,
            "ring_atoms": ring_atoms,
            "conjugated_bonds": conjugated_bonds,
            "ring_bonds": ring_bonds,
        },
    }


def _stereo_profile(smiles: Sequence[str], label: str) -> dict[str, int | float]:
    assigned_legacy = 0
    assigned_modern = 0
    explicit_bond_stereo = 0
    assigned_union_legacy = 0
    assigned_union_modern = 0
    any_potential_stereo = 0
    potential_tetrahedral = 0
    potential_double_bond = 0
    for index, value in enumerate(smiles):
        molecule = _molecule(value, f"{label} stereo row {index}")
        legacy = bool(
            Chem.FindMolChiralCenters(
                molecule,
                includeUnassigned=False,
                useLegacyImplementation=True,
            )
        )
        modern = bool(
            Chem.FindMolChiralCenters(
                molecule,
                includeUnassigned=False,
                useLegacyImplementation=False,
            )
        )
        bond = any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in molecule.GetBonds())
        potential_stereo = Chem.FindPotentialStereo(molecule)
        potential_atom = any(
            item.type == Chem.StereoType.Atom_Tetrahedral for item in potential_stereo
        )
        potential_bond = any(item.type == Chem.StereoType.Bond_Double for item in potential_stereo)
        potential = bool(potential_stereo)
        assigned_legacy += int(legacy)
        assigned_modern += int(modern)
        explicit_bond_stereo += int(bond)
        assigned_union_legacy += int(legacy or bond)
        assigned_union_modern += int(modern or bond)
        potential_tetrahedral += int(potential_atom)
        potential_double_bond += int(potential_bond)
        any_potential_stereo += int(potential)
    count = len(smiles)
    return {
        "records": count,
        "assigned_tetrahedral_legacy_records": assigned_legacy,
        "assigned_tetrahedral_modern_records": assigned_modern,
        "explicit_bond_stereo_records": explicit_bond_stereo,
        "assigned_atom_or_bond_legacy_records": assigned_union_legacy,
        "assigned_atom_or_bond_modern_records": assigned_union_modern,
        "potential_tetrahedral_records": potential_tetrahedral,
        "potential_double_bond_records": potential_double_bond,
        "potential_atom_or_bond_stereo_records": any_potential_stereo,
        "assigned_atom_or_bond_modern_fraction": assigned_union_modern / count,
        "potential_tetrahedral_fraction": potential_tetrahedral / count,
        "potential_double_bond_fraction": potential_double_bond / count,
        "potential_atom_or_bond_stereo_fraction": any_potential_stereo / count,
    }


def run_oracle_graph_corpus_audit(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Build the deterministic graph and pretraining corpus audit."""

    config = _load_json(config_path, "graph-corpus config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise OracleGraphCorpusError("unsupported graph-corpus config schema")
    if config.get("seed") != 1729:
        raise OracleGraphCorpusError("graph-corpus seed must remain 1729")
    pretraining_policy = config.get("pretraining_policy")
    if (
        not isinstance(pretraining_policy, dict)
        or {
            "exclude_every_exact_oracle_constitution_globally": True,
            "deduplicate_retained_r0_by_constitution": True,
            "biological_labels_used": False,
            "virtual_candidates_used": False,
            "virtual_candidate_role": "applicability_only",
        }.items()
        - pretraining_policy.items()
    ):
        raise OracleGraphCorpusError("graph-corpus pretraining policy changed")
    representation = config.get("representation")
    if (
        not isinstance(representation, dict)
        or representation.get("stereochemistry") != "excluded"
        or representation.get("truncation_allowed") is not False
    ):
        raise OracleGraphCorpusError("graph representation contract changed")
    if tuple(config.get("pretraining_output_fields", ())) != RETAINED_FIELDS:
        raise OracleGraphCorpusError("pretraining output-field whitelist changed")
    identity_policy = config.get("identity_policy")
    if identity_policy != {
        "sanitize": True,
        "canonical_isomeric_smiles": False,
        "preserve_formal_charge": True,
        "preserve_protonation": True,
        "preserve_tautomers": True,
        "strip_fragments": False,
        "require_connected": True,
    }:
        raise OracleGraphCorpusError("graph identity policy changed")

    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise OracleGraphCorpusError("graph-corpus config lacks inputs")
    verified = {
        name: _verify_input(repo_root, specification, name)
        for name, specification in sorted(inputs.items())
        if isinstance(specification, dict)
    }
    if set(verified) != {
        "agile_label_reconciliation",
        "curated_oracle_data",
        "r0_observed_structures",
        "training_corpus_manifest",
        "virtual_candidate_library",
    }:
        raise OracleGraphCorpusError("graph-corpus inputs are incomplete")

    reconciliation = _load_json(
        repo_root / verified["agile_label_reconciliation"]["path"],
        "AGILE label reconciliation",
    )
    manifest = _load_json(
        repo_root / verified["training_corpus_manifest"]["path"],
        "training corpus manifest",
    )

    curated = _read_csv(
        repo_root / verified["curated_oracle_data"]["path"],
        {"label", "model_smiles"},
        "curated oracle data",
    )
    r0 = _read_csv(
        repo_root / verified["r0_observed_structures"]["path"],
        {
            "r0_structure_id",
            "canonical_isomeric_smiles",
            "formal_charge",
            "heavy_atoms",
            "observed_source_ids",
            "r0_pretraining_eligible",
            "stereogenic_atom_count",
        },
        "R0",
    )
    virtual = _read_csv(
        repo_root / verified["virtual_candidate_library"]["path"],
        {"source_row_index", "canonical_isomeric_smiles"},
        "virtual candidate library",
    )
    expected = config.get("expected")
    if not isinstance(expected, dict):
        raise OracleGraphCorpusError("graph-corpus config lacks expected counts")
    for label, observed in (
        ("curated_oracle_records", len(curated)),
        ("r0_input_rows", len(r0)),
        ("virtual_candidate_records", len(virtual)),
    ):
        if observed != expected.get(label):
            raise OracleGraphCorpusError(
                f"{label} changed: expected {expected.get(label)}, observed {observed}"
            )
    if reconciliation.get("summary", {}).get("curated_single_structure_records") != len(curated):
        raise OracleGraphCorpusError("AGILE reconciliation does not match the curated graph corpus")
    manifest_r0 = manifest.get("layers", {}).get("r0_observed_real", {})
    if manifest_r0.get("sha256") != verified["r0_observed_structures"]["sha256"] or manifest_r0.get(
        "row_count"
    ) != len(r0):
        raise OracleGraphCorpusError("training manifest does not identify the pinned R0")

    oracle_by_constitution: dict[str, list[str]] = defaultdict(list)
    curated_smiles = []
    oracle_labels: set[str] = set()
    for row in curated:
        if row["label"] in oracle_labels:
            raise OracleGraphCorpusError(f"duplicate oracle label: {row['label']}")
        oracle_labels.add(row["label"])
        canonical, _ = _constitution(row["model_smiles"], f"oracle {row['label']}")
        oracle_by_constitution[canonical].append(row["label"])
        curated_smiles.append(canonical)
    if len(oracle_by_constitution) != len(curated):
        raise OracleGraphCorpusError("curated oracle constitutional graphs are not unique")

    r0_ids = [row["r0_structure_id"] for row in r0]
    if len(set(r0_ids)) != len(r0_ids):
        raise OracleGraphCorpusError("R0 structure IDs are not unique")
    virtual_ids = [row["source_row_index"] for row in virtual]
    if len(set(virtual_ids)) != len(virtual_ids):
        raise OracleGraphCorpusError("virtual source-row indices are not unique")

    eligible_rows = [row for row in r0 if row["r0_pretraining_eligible"].strip().lower() == "true"]
    if len(eligible_rows) != expected["r0_pretraining_eligible_rows"]:
        raise OracleGraphCorpusError("R0 eligibility count changed")

    all_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    row_constitutions: dict[str, str] = {}
    for row in eligible_rows:
        canonical, molecule = _constitution(
            row["canonical_isomeric_smiles"],
            f"R0 {row['r0_structure_id']}",
        )
        potential_tetrahedral_count = sum(
            item.type == Chem.StereoType.Atom_Tetrahedral
            for item in Chem.FindPotentialStereo(molecule)
        )
        if int(row["heavy_atoms"]) != molecule.GetNumAtoms():
            raise OracleGraphCorpusError(
                f"R0 {row['r0_structure_id']} stored heavy-atom count changed"
            )
        if int(row["formal_charge"]) != Chem.GetFormalCharge(molecule):
            raise OracleGraphCorpusError(
                f"R0 {row['r0_structure_id']} stored formal charge changed"
            )
        if int(row["stereogenic_atom_count"]) != potential_tetrahedral_count:
            raise OracleGraphCorpusError(
                f"R0 {row['r0_structure_id']} stored stereogenic-atom count changed"
            )
        all_groups[canonical].append(row)
        row_constitutions[row["r0_structure_id"]] = canonical

    duplicate_groups = {canonical: rows for canonical, rows in all_groups.items() if len(rows) > 1}
    rows_collapsed_before_exclusion = len(eligible_rows) - len(all_groups)
    for label, observed in (
        ("r0_unique_constitutions", len(all_groups)),
        ("r0_constitutional_duplicate_groups", len(duplicate_groups)),
        ("r0_rows_collapsed_before_oracle_exclusion", rows_collapsed_before_exclusion),
    ):
        if observed != expected[label]:
            raise OracleGraphCorpusError(
                f"{label} changed: expected {expected[label]}, observed {observed}"
            )

    excluded: list[dict[str, Any]] = []
    retained_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible_rows:
        canonical = row_constitutions[row["r0_structure_id"]]
        if canonical in oracle_by_constitution:
            excluded.append(
                {
                    "r0_id": row["r0_structure_id"],
                    "canonical_isomeric_smiles": row["canonical_isomeric_smiles"],
                    "constitutional_smiles": canonical,
                    "observed_source_ids": row["observed_source_ids"],
                    "matched_oracle_labels_json": json.dumps(
                        sorted(oracle_by_constitution[canonical]),
                        separators=(",", ":"),
                    ),
                }
            )
        else:
            retained_groups[canonical].append(row)
    if len(excluded) != expected["r0_rows_matching_any_oracle_constitution"]:
        raise OracleGraphCorpusError("R0 oracle-overlap exclusion count changed")
    if len(retained_groups) != expected["r0_retained_unique_constitutions"]:
        raise OracleGraphCorpusError("R0 retained constitutional count changed")
    unique_excluded = len({row["constitutional_smiles"] for row in excluded})
    nonoracle_before_dedup = len(eligible_rows) - len(excluded)
    additional_retained_collapse = nonoracle_before_dedup - len(retained_groups)
    for label, observed in (
        ("r0_unique_oracle_constitutions_matched", unique_excluded),
        ("r0_nonoracle_rows_before_constitutional_deduplication", nonoracle_before_dedup),
        (
            "r0_additional_rows_collapsed_after_oracle_exclusion",
            additional_retained_collapse,
        ),
    ):
        if observed != expected[label]:
            raise OracleGraphCorpusError(
                f"{label} changed: expected {expected[label]}, observed {observed}"
            )

    provenance_only_leaks = [
        row
        for row in excluded
        if "agile_measured1200"
        not in {
            source.strip() for source in row["observed_source_ids"].split("|") if source.strip()
        }
    ]
    provenance_only_leak_constitutions = {
        row["constitutional_smiles"] for row in provenance_only_leaks
    }
    if (
        len(provenance_only_leaks) != expected["provenance_only_filter_remaining_oracle_rows"]
        or len(provenance_only_leak_constitutions)
        != expected["provenance_only_filter_remaining_oracle_constitutions"]
    ):
        raise OracleGraphCorpusError("provenance-only leakage regression changed")

    retained = []
    for canonical, rows in sorted(retained_groups.items()):
        molecule = _molecule(canonical, "retained R0 constitution")
        retained.append(
            {
                "graph_id": hashlib.sha256(canonical.encode()).hexdigest(),
                "constitutional_smiles": canonical,
                "atom_count": molecule.GetNumAtoms(),
                "undirected_bond_count": molecule.GetNumBonds(),
                "directed_edge_count": 2 * molecule.GetNumBonds(),
            }
        )
    excluded.sort(key=lambda row: row["r0_id"])

    collapse_ledger = []
    for canonical, rows in sorted(duplicate_groups.items()):
        ordered = sorted(rows, key=lambda row: row["r0_structure_id"])
        collapse_ledger.append(
            {
                "constitutional_smiles": canonical,
                "group_scope": (
                    "oracle_excluded" if canonical in oracle_by_constitution else "retained"
                ),
                "group_size": len(ordered),
                "r0_ids_json": json.dumps(
                    [row["r0_structure_id"] for row in ordered],
                    separators=(",", ":"),
                ),
                "canonical_isomeric_smiles_json": json.dumps(
                    [row["canonical_isomeric_smiles"] for row in ordered],
                    separators=(",", ":"),
                ),
                "observed_source_ids_json": json.dumps(
                    [row["observed_source_ids"] for row in ordered],
                    separators=(",", ":"),
                ),
            }
        )

    retained_inchi_groups: dict[str, list[str]] = defaultdict(list)
    for row in retained:
        molecule = _molecule(row["constitutional_smiles"], "retained InChI audit")
        connectivity_key = Chem.MolToInchiKey(molecule).split("-")[0]
        retained_inchi_groups[connectivity_key].append(row["graph_id"])
    retained_inchi_collapses = {
        key: sorted(graph_ids)
        for key, graph_ids in retained_inchi_groups.items()
        if len(graph_ids) > 1
    }
    if len(retained_inchi_collapses) != expected["retained_inchi_connectivity_collapse_groups"]:
        raise OracleGraphCorpusError("retained InChI connectivity audit changed")
    oracle_inchi_connectivity = {
        Chem.MolToInchiKey(_molecule(smiles, "oracle InChI audit")).split("-")[0]
        for smiles in curated_smiles
    }
    retained_oracle_inchi_overlap = set(retained_inchi_groups) & oracle_inchi_connectivity
    if retained_oracle_inchi_overlap:
        raise OracleGraphCorpusError(
            "retained R0 still overlaps AGILE by standard InChI connectivity"
        )

    virtual_smiles = [
        _constitution(
            row["canonical_isomeric_smiles"],
            f"virtual {row['source_row_index']}",
        )[0]
        for row in virtual
    ]
    r0_input_smiles = [row["canonical_isomeric_smiles"] for row in eligible_rows]
    graph_profiles = {
        "curated_oracle": _graph_profile(curated_smiles, "curated oracle"),
        "r0_input": _graph_profile(r0_input_smiles, "R0 input"),
        "r0_pretraining": _graph_profile(
            [row["constitutional_smiles"] for row in retained],
            "R0 pretraining",
        ),
        "virtual_applicability": _graph_profile(
            virtual_smiles,
            "virtual applicability",
        ),
    }
    r0_stereo_profile = _stereo_profile(r0_input_smiles, "R0 input")
    manifest_profile = manifest.get("kernel_readiness_profile")
    if not isinstance(manifest_profile, dict):
        raise OracleGraphCorpusError("training manifest lacks kernel-readiness profile")
    actual_r0_profile = graph_profiles["r0_input"]
    actual_atom_distribution = actual_r0_profile["atom_count_distribution"]
    actual_elements = sorted(actual_r0_profile["feature_vocabulary"]["elements"])
    net_charge_counts = actual_r0_profile["feature_vocabulary"]["molecular_net_charges"]
    charged_records = sum(count for charge, count in net_charge_counts.items() if charge != "0")
    actual_manifest_comparison = {
        "heavy_atoms": {
            "minimum": actual_atom_distribution["minimum"],
            "median": actual_atom_distribution["median"],
            "maximum": actual_atom_distribution["maximum"],
            "records_at_most_64": actual_r0_profile["records_at_most_64_atoms"],
            "fraction_at_most_64": (
                actual_r0_profile["records_at_most_64_atoms"] / len(eligible_rows)
            ),
            "records_at_most_96": actual_r0_profile["records_at_most_96_atoms"],
            "fraction_at_most_96": (
                actual_r0_profile["records_at_most_96_atoms"] / len(eligible_rows)
            ),
        },
        "elements_present": actual_elements,
        "net_charged_records": charged_records,
        "fraction_net_charged": charged_records / len(eligible_rows),
        "stereo": r0_stereo_profile,
    }
    manifest_discrepancies = {
        "profile_matches_pinned_r0": False,
        "manifest": {
            "heavy_atoms": manifest_profile.get("heavy_atoms"),
            "elements_present": manifest_profile.get("elements_present"),
            "fraction_charged": manifest_profile.get("fraction_charged"),
            "fraction_with_stereocenter": manifest_profile.get("fraction_with_stereocenter"),
        },
        "actual_pinned_r0": actual_manifest_comparison,
        "stereo_definition_note": (
            "The manifest's fraction_with_stereocenter is ambiguous and matches "
            "neither audited potential nor explicitly specified stereo definitions."
        ),
        "encoder_support_policy": (
            "Derive feature vocabulary and graph-size support from the pinned data; "
            "do not inherit the stale manifest profile."
        ),
    }

    retained_payload = _gzip_csv(retained, RETAINED_FIELDS)
    excluded_payload = _gzip_csv(excluded, EXCLUDED_FIELDS)
    collapse_payload = _gzip_csv(collapse_ledger, COLLAPSE_FIELDS)
    try:
        config_relative = _portable(config_path, repo_root)
    except OSError:
        config_relative = str(config_path.resolve())
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed_graph_corpus_leakage_gate",
        "seed": config["seed"],
        "inputs": {
            "config": {
                "path": config_relative,
                "sha256": _sha256_file(config_path),
                "bytes": config_path.stat().st_size,
            },
            **verified,
        },
        "summary": {
            "curated_oracle_records": len(curated),
            "curated_oracle_unique_constitutions": len(oracle_by_constitution),
            "r0_input_rows": len(r0),
            "r0_pretraining_eligible_rows": len(eligible_rows),
            "r0_unique_constitutions": len(all_groups),
            "r0_constitutional_duplicate_groups": len(duplicate_groups),
            "r0_rows_collapsed_before_oracle_exclusion": (rows_collapsed_before_exclusion),
            "r0_duplicate_group_size_distribution": dict(
                sorted(Counter(str(len(rows)) for rows in duplicate_groups.values()).items())
            ),
            "r0_rows_matching_any_oracle_constitution": len(excluded),
            "r0_unique_oracle_constitutions_matched": unique_excluded,
            "r0_oracle_overlap_group_size_distribution": dict(
                sorted(
                    Counter(
                        str(len(all_groups[canonical])) for canonical in oracle_by_constitution
                    ).items()
                )
            ),
            "r0_nonoracle_rows_before_constitutional_deduplication": (nonoracle_before_dedup),
            "r0_retained_unique_constitutions": len(retained),
            "r0_additional_rows_collapsed_by_constitutional_deduplication": (
                additional_retained_collapse
            ),
            "provenance_only_filter_remaining_oracle_rows": len(provenance_only_leaks),
            "provenance_only_filter_remaining_oracle_constitutions": len(
                provenance_only_leak_constitutions
            ),
            "post_exclusion_oracle_constitution_overlap": 0,
            "post_exclusion_oracle_inchi_connectivity_overlap": 0,
            "retained_inchi_connectivity_collapse_groups": len(retained_inchi_collapses),
            "virtual_candidate_records": len(virtual),
        },
        "representation": representation,
        "identity_policy": identity_policy,
        "pretraining_policy": pretraining_policy,
        "pretraining_output_fields": list(RETAINED_FIELDS),
        "graph_profiles": graph_profiles,
        "r0_stereo_profile": r0_stereo_profile,
        "training_manifest_audit": manifest_discrepancies,
        "retained_inchi_connectivity_collapses": retained_inchi_collapses,
        "software": {"rdkit": rdBase.rdkitVersion},
        "decision": {
            "graph_tensorization_authorized": True,
            "r0_pretraining_authorized_only_from_retained_ledger": True,
            "virtual_candidates_authorized_for_pretraining": False,
            "manifest_kernel_profile_authorized_for_encoder_support": False,
            "oracle_model_frozen": False,
        },
        "artifacts": {
            "oracle_graph_r0_pretraining.csv.gz": _artifact_metadata(retained_payload),
            "oracle_graph_r0_exclusions.csv.gz": _artifact_metadata(excluded_payload),
            "oracle_graph_isomer_collapse.csv.gz": _artifact_metadata(collapse_payload),
        },
    }
    result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(output_dir / "oracle_graph_r0_pretraining.csv.gz", retained_payload)
    _atomic_write(output_dir / "oracle_graph_r0_exclusions.csv.gz", excluded_payload)
    _atomic_write(output_dir / "oracle_graph_isomer_collapse.csv.gz", collapse_payload)
    _atomic_write(output_dir / "oracle_graph_corpus_result.json", result_payload)
    return result

"""Reproducible L2 supervision inventory for M0-09.

This module inventories observed reaction supervision. It deliberately keeps molecule-only
corpora separate from reaction records so product structures cannot be mistaken for routes.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
import platform
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

SCHEMA_VERSION = "m0_09_l2_supervision_inventory.v1"
LEDGER_SCHEMA_VERSION = "m0_09_supervision_sources.v1"
CHEMISTRY_CLASSES = (
    "ester",
    "isocyanide",
    "aldehyde",
    "carbonate",
    "acrylate",
    "heterocycle_formation",
)


class InventoryError(ValueError):
    """Raised when an inventory input violates its declared contract."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def load_source_ledger(path: Path) -> dict[str, Any]:
    """Load and structurally validate the curated source ledger."""

    try:
        ledger = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise InventoryError(f"source ledger not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InventoryError(f"source ledger is not valid JSON: {path}: {exc}") from exc

    if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise InventoryError(
            f"unsupported ledger schema {ledger.get('schema_version')!r}; "
            f"expected {LEDGER_SCHEMA_VERSION!r}"
        )
    if tuple(ledger.get("chemistry_classes", ())) != CHEMISTRY_CLASSES:
        raise InventoryError("ledger chemistry_classes are missing, reordered, or unsupported")

    agile_groups = ledger.get("agile_si", {}).get("route_groups", [])
    if not agile_groups:
        raise InventoryError("agile_si.route_groups must not be empty")
    agile_targets = [target for group in agile_groups for target in group.get("targets", [])]
    if not all(group.get("steps") for group in agile_groups):
        raise InventoryError("every AGILE route group must contain at least one step")
    if len(agile_targets) != len(set(agile_targets)):
        raise InventoryError("AGILE terminal target names must be unique")

    protocols = ledger.get("internal_rm_protocols", {}).get("protocols", [])
    if len(protocols) != 7:
        raise InventoryError(f"expected seven RM protocols, found {len(protocols)}")
    for field in ("protocol", "asset", "expected_sha256", "target", "steps", "coverage"):
        if any(not protocol.get(field) for protocol in protocols):
            raise InventoryError(f"every RM protocol must define non-empty {field!r}")
    if len({protocol["protocol"] for protocol in protocols}) != len(protocols):
        raise InventoryError("RM protocol identifiers must be unique")
    if len({protocol["target"] for protocol in protocols}) != len(protocols):
        raise InventoryError("RM terminal target names must be unique")

    excluded = {entry.get("source") for entry in ledger.get("excluded_sources", [])}
    if "r1_reaction_enumerated_support_v1.csv" not in excluded:
        raise InventoryError("R1 must be explicitly excluded from L2 supervision")
    return ledger


def _declared_assets(ledger: dict[str, Any]) -> dict[str, str]:
    declared: dict[str, str] = {}
    sections: list[dict[str, Any]] = [
        ledger["uspto_pretraining"],
        ledger["uspto_pretraining"]["large_corpus"],
        ledger["agile_si"],
        *ledger["internal_rm_protocols"]["protocols"],
        *[
            corpus
            for corpus in ledger.get("related_structure_corpora_not_counted_as_l2", [])
            if "asset" in corpus
        ],
    ]
    for section in sections:
        asset = section["asset"]
        expected = section["expected_sha256"]
        previous = declared.setdefault(asset, expected)
        if previous != expected:
            raise InventoryError(f"conflicting expected hashes declared for {asset}")
    if "r1_reaction_enumerated_support_v1.csv" in declared:
        raise InventoryError("R1 cannot be declared as an M0-09 input asset")
    return declared


def validate_vendored_assets(ledger: dict[str, Any], vendor_dir: Path) -> list[dict[str, Any]]:
    """Verify every declared input asset and return provenance records."""

    records = []
    for name, expected in sorted(_declared_assets(ledger).items()):
        path = vendor_dir / name
        if not path.exists():
            raise InventoryError(f"required vendored asset not found: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise InventoryError(
                f"hash mismatch for {name}: expected {expected}, observed {actual}"
            )
        records.append({"asset": name, "bytes": path.stat().st_size, "sha256": actual})
    return records


def _parse_reaction_smiles(reaction: str, row_number: int) -> tuple[list[Chem.Mol], Chem.Mol]:
    parts = reaction.split(">>")
    if len(parts) != 2 or not all(parts):
        raise InventoryError(f"USPTO row {row_number} does not contain one reactants>>product pair")
    reactants_text, product_text = parts
    with rdBase.BlockLogs():
        reactants = [Chem.MolFromSmiles(smiles) for smiles in reactants_text.split(".")]
        product = Chem.MolFromSmiles(product_text)
    if product is None or any(reactant is None for reactant in reactants):
        raise InventoryError(f"USPTO row {row_number} contains an RDKit-invalid SMILES")
    return reactants, product


def analyze_uspto(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Count USPTO classes and transparent lipid-relevant formation candidates."""

    class_names = config["reaction_class_names"]
    patterns = {
        name: Chem.MolFromSmarts(smarts) for name, smarts in config["product_motif_smarts"].items()
    }
    invalid_patterns = [name for name, pattern in patterns.items() if pattern is None]
    if invalid_patterns:
        raise InventoryError(f"invalid USPTO product motif SMARTS: {invalid_patterns}")

    class_counts: Counter[str] = Counter()
    product_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    candidate_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    ids: set[str] = set()

    try:
        handle = path.open(newline="")
    except FileNotFoundError as exc:
        raise InventoryError(f"USPTO asset not found: {path}") from exc

    with handle:
        reader = csv.DictReader(handle)
        required = {"id", "class", "reactions"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise InventoryError(f"USPTO CSV must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            reaction_id = row["id"]
            if not reaction_id:
                raise InventoryError(f"USPTO row {row_number} has an empty id")
            ids.add(reaction_id)

            reaction_class = row["class"]
            if reaction_class not in class_names:
                raise InventoryError(f"USPTO row {row_number} has unknown class {reaction_class!r}")
            class_counts[reaction_class] += 1
            reactants, product = _parse_reaction_smiles(row["reactions"], row_number)

            for chemistry, pattern in patterns.items():
                product_has_motif = product.HasSubstructMatch(pattern)
                if product_has_motif:
                    product_counts[chemistry] += 1
                if product_has_motif and not any(
                    reactant.HasSubstructMatch(pattern) for reactant in reactants
                ):
                    candidate_counts[chemistry] += 1
                    candidate_by_class[chemistry][reaction_class] += 1

    ordered_classes = {
        reaction_class: {
            "name": class_names[reaction_class],
            "reactions": class_counts[reaction_class],
        }
        for reaction_class in sorted(class_names, key=int)
    }
    coverage: dict[str, dict[str, Any]] = {}
    for chemistry in CHEMISTRY_CLASSES:
        if chemistry == "heterocycle_formation":
            coverage[chemistry] = {
                "candidate_formation_reactions": class_counts["4"],
                "method": "USPTO reaction class 4",
                "product_motif_reactions": None,
                "candidate_by_uspto_class": {"4": class_counts["4"]},
            }
            continue
        coverage[chemistry] = {
            "candidate_formation_reactions": candidate_counts[chemistry],
            "method": "product motif absent from every reactant and present in product",
            "product_motif_reactions": product_counts[chemistry],
            "candidate_by_uspto_class": {
                key: candidate_by_class[chemistry][key]
                for key in sorted(candidate_by_class[chemistry], key=int)
            },
        }

    return {
        "dataset": config["dataset"],
        "reaction_records": sum(class_counts.values()),
        "unique_source_document_ids": len(ids),
        "reaction_classes": len([count for count in class_counts.values() if count]),
        "class_counts": ordered_classes,
        "lipid_relevant_coverage": coverage,
        "coverage_limitation": config["coverage_method"],
    }


def analyze_uspto_mit(
    path: Path, config: dict[str, Any], motif_config: dict[str, Any]
) -> dict[str, Any]:
    """Count the larger USPTO-MIT pretraining corpus and product-motif candidates."""

    patterns = {
        name: Chem.MolFromSmarts(smarts)
        for name, smarts in motif_config["product_motif_smarts"].items()
    }
    invalid_patterns = [name for name, pattern in patterns.items() if pattern is None]
    if invalid_patterns:
        raise InventoryError(f"invalid USPTO product motif SMARTS: {invalid_patterns}")

    split_counts: dict[str, int] = {}
    product_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    invalid_records = 0
    invalid_examples: list[str] = []
    try:
        archive = zipfile.ZipFile(path)
    except FileNotFoundError as exc:
        raise InventoryError(f"USPTO-MIT archive not found: {path}") from exc
    except zipfile.BadZipFile as exc:
        raise InventoryError(f"USPTO-MIT asset is not a valid zip archive: {path}") from exc

    with archive:
        for split, member in config["archive_members"].items():
            try:
                raw_handle = archive.open(member)
            except KeyError as exc:
                raise InventoryError(f"USPTO-MIT archive is missing {member!r}") from exc
            count = 0
            with raw_handle, io.TextIOWrapper(raw_handle, encoding="utf-8") as text_handle:
                for line_number, line in enumerate(text_handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        raise InventoryError(
                            f"USPTO-MIT {split} line {line_number} is unexpectedly blank"
                        )
                    reaction = stripped.split(maxsplit=1)[0]
                    try:
                        reactants, product = _parse_reaction_smiles(reaction, line_number)
                    except InventoryError:
                        invalid_records += 1
                        if len(invalid_examples) < 10:
                            invalid_examples.append(f"{split}:{line_number}")
                        count += 1
                        continue
                    for chemistry, pattern in patterns.items():
                        product_has_motif = product.HasSubstructMatch(pattern)
                        if product_has_motif:
                            product_counts[chemistry] += 1
                        if product_has_motif and not any(
                            reactant.HasSubstructMatch(pattern) for reactant in reactants
                        ):
                            candidate_counts[chemistry] += 1
                    count += 1
            split_counts[split] = count

    coverage = {}
    for chemistry in CHEMISTRY_CLASSES:
        if chemistry == "heterocycle_formation":
            coverage[chemistry] = {
                "candidate_formation_reactions": None,
                "method": "not quantified because this distribution has no reaction-class labels",
                "product_motif_reactions": None,
            }
            continue
        coverage[chemistry] = {
            "candidate_formation_reactions": candidate_counts[chemistry],
            "method": "product motif absent from every reactant and present in product",
            "product_motif_reactions": product_counts[chemistry],
        }
    return {
        "dataset": config["dataset"],
        "reaction_records": sum(split_counts.values()),
        "rdkit_parseable_records": sum(split_counts.values()) - invalid_records,
        "rdkit_invalid_records": invalid_records,
        "rdkit_invalid_record_examples": invalid_examples,
        "split_counts": split_counts,
        "reaction_classes": None,
        "class_labels_available": config["class_labels_available"],
        "lipid_relevant_coverage": coverage,
        "coverage_limitation": motif_config["coverage_method"],
        "intended_role": config["intended_role"],
    }


def analyze_r0_structure_context(path: Path) -> dict[str, Any]:
    """Count existing region annotations that can prioritize, but not supervise, L2."""

    source_rows: Counter[str] = Counter()
    field_rows: Counter[tuple[str, str]] = Counter()
    unique_values: dict[tuple[str, str], set[str]] = defaultdict(set)
    row_count = 0
    try:
        handle = path.open(newline="")
    except FileNotFoundError as exc:
        raise InventoryError(f"R0 asset not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "region_annotations_json" not in reader.fieldnames:
            raise InventoryError("R0 CSV must contain region_annotations_json")
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            try:
                annotations = json.loads(row["region_annotations_json"])
            except json.JSONDecodeError as exc:
                raise InventoryError(
                    f"R0 row {row_number} has invalid region_annotations_json"
                ) from exc
            for source, fields in annotations.items():
                source_rows[source] += 1
                for field, values in fields.items():
                    field_rows[(source, field)] += 1
                    if not isinstance(values, list):
                        values = [values]
                    unique_values[(source, field)].update(value for value in values if value)

    selected_fields = {
        "agile_measured1200": ("A_smiles", "B_smiles", "C_smiles"),
        "lnpdb_v1": (
            "IL_head_SMILES",
            "IL_linker_SMILES",
            "IL_tail1_SMILES",
            "IL_tail2_SMILES",
        ),
        "lion_repository_all": ("Amine_SMILES",),
    }
    sources = {}
    for source, fields in selected_fields.items():
        sources[source] = {
            "annotated_r0_rows": source_rows[source],
            "region_fields": {
                field: {
                    "annotated_rows": field_rows[(source, field)],
                    "unique_raw_values": len(unique_values[(source, field)]),
                }
                for field in fields
            },
        }
    return {
        "r0_records": row_count,
        "sources": sources,
        "interpretation": (
            "These region annotations can prioritize head, linker, and tail blocks after "
            "train-only extraction and chemistry QA; they are not observed L2 routes."
        ),
    }


def _summarize_agile(agile: dict[str, Any]) -> dict[str, Any]:
    reaction_instances = 0
    terminal_targets = 0
    intermediate_products = 0
    depth_counts: Counter[int] = Counter()
    transformation_counts: Counter[str] = Counter()
    coverage_counts: Counter[str] = Counter()
    groups = []

    for group in agile["route_groups"]:
        targets = group["targets"]
        steps = group["steps"]
        target_count = len(targets)
        depth = len(steps)
        reaction_instances += target_count * depth
        terminal_targets += target_count
        intermediate_products += target_count * (depth - 1)
        depth_counts[depth] += target_count
        for step in steps:
            transformation_counts[step] += target_count
        for chemistry in group["coverage"]:
            coverage_counts[chemistry] += target_count
        groups.append(
            {
                "route_group": group["route_group"],
                "terminal_targets": target_count,
                "route_depth": depth,
                "reaction_instances": target_count * depth,
                "steps": steps,
                "coverage": group["coverage"],
            }
        )

    return {
        "sources": {"supplementary_information": 1, "patents": 0},
        "distinct_upstream_reaction_instances": reaction_instances,
        "unique_upstream_products": reaction_instances,
        "unique_intermediate_products": intermediate_products,
        "unique_terminal_precursor_scaffolds": terminal_targets,
        "terminal_route_depth_distribution": {
            str(depth): count for depth, count in sorted(depth_counts.items())
        },
        "transformation_step_counts": dict(sorted(transformation_counts.items())),
        "lipid_relevant_coverage": {
            chemistry: coverage_counts[chemistry] for chemistry in CHEMISTRY_CLASSES
        },
        "route_groups": groups,
        "outcome_evidence": agile["outcome_evidence"],
        "qa_flags": agile["qa_flags"],
    }


def _summarize_rm(internal: dict[str, Any]) -> dict[str, Any]:
    protocols = internal["protocols"]
    depth_counts: Counter[int] = Counter()
    coverage_counts: Counter[str] = Counter()
    transformation_counts: Counter[str] = Counter()
    records = []
    for protocol in protocols:
        depth = len(protocol["steps"])
        depth_counts[depth] += 1
        for chemistry in protocol["coverage"]:
            coverage_counts[chemistry] += 1
        for step in protocol["steps"]:
            transformation_counts[step] += 1
        records.append(
            {
                "protocol": protocol["protocol"],
                "target": protocol["target"],
                "route_depth": depth,
                "coverage": protocol["coverage"],
                "documented_conditions": protocol["documented_conditions"],
                "outcome_status": protocol["outcome_status"],
                "qa_flags": protocol.get("qa_flags", []),
            }
        )
    return {
        "protocol_documents": len(protocols),
        "distinct_upstream_reaction_instances": sum(len(item["steps"]) for item in protocols),
        "unique_upstream_products": len(protocols),
        "unique_terminal_precursor_scaffolds": len(protocols),
        "terminal_route_depth_distribution": {
            str(depth): count for depth, count in sorted(depth_counts.items())
        },
        "transformation_step_counts": dict(sorted(transformation_counts.items())),
        "lipid_relevant_coverage": {
            chemistry: coverage_counts[chemistry] for chemistry in CHEMISTRY_CLASSES
        },
        "outcome_evidence": internal["outcome_evidence"],
        "protocols": records,
    }


def _combined_lipid_specific(agile: dict[str, Any], rm: dict[str, Any]) -> dict[str, Any]:
    depth_counts: Counter[str] = Counter(agile["terminal_route_depth_distribution"])
    depth_counts.update(rm["terminal_route_depth_distribution"])
    transformations: Counter[str] = Counter(agile["transformation_step_counts"])
    transformations.update(rm["transformation_step_counts"])
    return {
        "distinct_upstream_reaction_instances": agile["distinct_upstream_reaction_instances"]
        + rm["distinct_upstream_reaction_instances"],
        "unique_upstream_products": agile["unique_upstream_products"]
        + rm["unique_upstream_products"],
        "unique_terminal_precursor_scaffolds": agile["unique_terminal_precursor_scaffolds"]
        + rm["unique_terminal_precursor_scaffolds"],
        "terminal_route_depth_distribution": {
            depth: depth_counts[depth] for depth in sorted(depth_counts, key=int)
        },
        "broad_transformation_classes": sorted(transformations),
        "transformation_step_counts": dict(sorted(transformations.items())),
        "lipid_relevant_coverage": {
            chemistry: agile["lipid_relevant_coverage"][chemistry]
            + rm["lipid_relevant_coverage"][chemistry]
            for chemistry in CHEMISTRY_CLASSES
        },
        "explicit_failed_syntheses": agile["outcome_evidence"]["explicit_failed_syntheses"]
        + rm["outcome_evidence"]["explicit_failed_syntheses"],
        "explicit_success_yields": agile["outcome_evidence"]["explicit_success_yields"]
        + rm["outcome_evidence"]["explicit_success_yields"],
        "rm_protocols_with_missing_outcomes": rm["outcome_evidence"]["blank_yield_fields"],
    }


def build_inventory(
    ledger_path: Path,
    vendor_dir: Path,
    *,
    generated_utc: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Build the complete M0-09 result from hash-verified inputs."""

    ledger = load_source_ledger(ledger_path)
    asset_records = validate_vendored_assets(ledger, vendor_dir)
    uspto_benchmark = analyze_uspto(
        vendor_dir / ledger["uspto_pretraining"]["asset"], ledger["uspto_pretraining"]
    )
    uspto_large = analyze_uspto_mit(
        vendor_dir / ledger["uspto_pretraining"]["large_corpus"]["asset"],
        ledger["uspto_pretraining"]["large_corpus"],
        ledger["uspto_pretraining"],
    )
    uspto = {
        "large_corpus": uspto_large,
        "class_labeled_benchmark": uspto_benchmark,
        "recommended_use": (
            "Pretrain generic reaction proposals on USPTO-MIT; use USPTO-50K for controlled "
            "class-aware development and holdouts. Neither corpus supplies lipid-specific outcomes."
        ),
    }
    r0_context = analyze_r0_structure_context(vendor_dir / "r0_observed_real_structures.csv")
    agile = _summarize_agile(ledger["agile_si"])
    rm = _summarize_rm(ledger["internal_rm_protocols"])
    combined = _combined_lipid_specific(agile, rm)

    if generated_utc is None:
        generated_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    try:
        parsed_timestamp = dt.datetime.fromisoformat(generated_utc)
    except ValueError as exc:
        raise InventoryError(f"generated_utc is not ISO-8601: {generated_utc!r}") from exc
    if parsed_timestamp.tzinfo is None:
        raise InventoryError("generated_utc must include a timezone")

    repository = vendor_dir.resolve().parents[1]
    try:
        ledger_label = str(ledger_path.resolve().relative_to(repository))
    except ValueError:
        ledger_label = str(ledger_path)
    inputs = [
        {
            "asset": ledger_label,
            "bytes": ledger_path.stat().st_size,
            "sha256": sha256_file(ledger_path),
        },
        *asset_records,
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": "M0-09",
        "generated_utc": generated_utc,
        "randomness": {"seed": seed, "used": False},
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
        "inputs": inputs,
        "counting_definitions": {
            "distinct_upstream_reaction_instance": "One documented transformation applied to one substrate or target series member.",
            "unique_upstream_product": "A distinct intermediate or terminal product named by a documented upstream reaction.",
            "unique_terminal_precursor_scaffold": "A distinct terminal tail block supplied to L1 assembly; intermediates are excluded.",
            "route_depth": "Number of documented L2 transformations from the stated starting material to a terminal L1 precursor.",
            "coverage_counts": "Non-exclusive reaction-instance or target counts; an acrylate ester can count as both ester and acrylate.",
        },
        "inventory": {
            "uspto_pretraining": uspto,
            "patent_si_lipid_routes": {
                **agile,
                "patent_availability": ledger["patent_routes"],
            },
            "agile_tail_routes": agile,
            "internal_rm_protocols": rm,
            "negative_outcomes": {
                "explicit_failed_syntheses": len(
                    ledger["negative_outcomes"]["explicit_failed_syntheses"]
                ),
                "note": ledger["negative_outcomes"]["note"],
            },
        },
        "lipid_specific_combined": combined,
        "r0_head_tail_structure_context": r0_context,
        "related_structure_corpora_not_counted_as_l2": ledger[
            "related_structure_corpora_not_counted_as_l2"
        ],
        "excluded_sources": ledger["excluded_sources"],
        "viability_decision": {
            "standalone_learned_lipid_specific_l2_viable": False,
            "recommended_architecture": "generic neural proposal plus deterministic lipid-specific template search",
            "basis": [
                f"Only {combined['distinct_upstream_reaction_instances']} lipid-specific reaction instances and "
                f"{combined['unique_terminal_precursor_scaffolds']} terminal precursor scaffolds were documented.",
                "No explicit failed L2 syntheses were found, so the local corpus has no observed negative supervision.",
                "The lipid-specific sources contain no carbonate or heterocycle-formation examples.",
                f"USPTO-50K contains only {uspto_benchmark['lipid_relevant_coverage']['isocyanide']['candidate_formation_reactions']} "
                "candidate isocyanide-forming reactions under the declared motif heuristic.",
                "AGILE candidate structures and R0/LNPDB-like real lipids describe product space, not observed tail-making routes.",
            ],
            "data_action_plan": [
                "Use USPTO-MIT for broad proposal pretraining and USPTO-50K for class-aware benchmarking, with reaction-class and scaffold holdouts.",
                "Encode the four observed lipid-specific transformation types as deterministic, source-linked templates.",
                "Use R0 real-lipid structures and AGILE candidates to prioritize tail/linker coverage gaps only after split-before-extraction.",
                "Curate carbonate, acrylate, heterocycle, and failed-route records from patents, supplementary information, and lab notebooks.",
                "Capture every future internal attempt with substrate, conditions, conversion or yield, analytical evidence, and failure status.",
            ],
        },
    }
    return result


def write_inventory(result: dict[str, Any], output_path: Path) -> None:
    """Atomically serialize a result so exceptions cannot leave a partial artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

"""Evidence-bounded source-paper prioritization for the M0-09 review queue.

This module enriches the immutable acquisition queue with a separate priority
overlay. It never changes source acquisition state, route evidence, procurement
state, or biological labels. Missing evidence receives zero priority credit and
remains explicit.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TextIO

from forge.core.io import atomic_write as _atomic_write
from forge.core.io import read_json_object
from forge.route.sources.source_ledger import SOURCE_COLUMNS
from forge.route.sources.supervision_inventory import sha256_file

CONFIG_SCHEMA_VERSION = "m0_09_source_priority_config.v1"
RESULT_SCHEMA_VERSION = "m0_09_source_priority_result.v1"

PRIORITY_COLUMNS = (
    "priority_rank",
    "next_review_rank",
    "priority_score",
    "priority_max_score",
    "assessed_axis_count",
    "review_completion_state",
    "missing_priority_axes_json",
    "motif_distinctiveness_score",
    "motif_distinctiveness_state",
    "curated_motif_count",
    "curated_motif_class_count",
    "biological_provenance_score",
    "biological_provenance_state",
    "numeric_measurement_count",
    "in_vivo_measurement_count",
    "assay_context_count",
    "agile_exact_novelty_score",
    "agile_exact_novelty_state",
    "parsed_hydrophobic_component_count",
    "exact_agile_hydrophobic_component_count",
    "non_agile_hydrophobic_component_count",
    "source_exclusive_non_agile_component_count",
    "method_accessibility_score",
    "method_accessibility_state",
    "ugi_handle_conversion_score",
    "ugi_handle_conversion_state",
    "ugi_forward_verified_component_count",
    "ugi_route_complete_component_count",
    "exact_component_redundancy_score",
    "exact_component_redundancy_state",
    "shared_hydrophobic_component_count",
    "shared_hydrophobic_component_fraction",
    "priority_evidence_ids_json",
)

AXES = (
    "motif_distinctiveness",
    "biological_provenance",
    "agile_exact_novelty",
    "method_accessibility",
    "ugi_handle_conversion",
    "exact_component_redundancy",
)

REQUIRED_LNPDB_COLUMNS = (
    "Publication_PMID",
    "Publication_link",
    "Model",
    "Model_type",
    "Model_target",
    "Route_of_administration",
    "Cargo",
    "Cargo_type",
    "Experiment_method",
    "Experiment_value",
)


class SourcePriorityError(ValueError):
    """Raised when source-priority evidence violates the frozen contract."""


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return read_json_object(path, error=SourcePriorityError, label=label)


def _safe_relative_path(repo_root: Path, value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SourcePriorityError(f"{label} must be a safe repository-relative path")
    resolved = repo_root / path
    try:
        resolved.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise SourcePriorityError(f"{label} escapes the repository root") from exc
    return resolved


def _verified_input(
    repo_root: Path,
    record: dict[str, Any],
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, dict):
        raise SourcePriorityError(f"inputs.{label} must be an object")
    path = _safe_relative_path(repo_root, str(record.get("path", "")), f"inputs.{label}.path")
    expected = str(record.get("expected_sha256", ""))
    if len(expected) != 64:
        raise SourcePriorityError(f"inputs.{label}.expected_sha256 must be a SHA-256 digest")
    if not path.exists():
        raise SourcePriorityError(f"priority input {label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise SourcePriorityError(
            f"hash mismatch for priority input {label}: expected {expected}, observed {observed}"
        )
    return path, {
        "path": str(path.resolve().relative_to(repo_root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": observed,
    }


def load_config(path: Path, repo_root: Path) -> tuple[dict[str, Any], dict[str, Path], list[dict]]:
    """Load the frozen scoring policy and verify every evidence input."""

    config = _load_json(path, "source-priority config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SourcePriorityError(
            f"unsupported config schema {config.get('schema_version')!r}; "
            f"expected {CONFIG_SCHEMA_VERSION!r}"
        )
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise SourcePriorityError("source-priority config must define policy")
    if policy.get("missing_evidence_score") != 0:
        raise SourcePriorityError("missing evidence must receive zero priority credit")
    if tuple(policy.get("hydrophobic_roles", ())) != ("linker", "tail1", "tail2"):
        raise SourcePriorityError("policy hydrophobic_roles must be linker, tail1, and tail2")
    weights = policy.get("axis_weights")
    if not isinstance(weights, dict) or set(weights) != set(AXES):
        raise SourcePriorityError(f"policy axis_weights must define exactly {list(AXES)}")
    if any(value != 1 for value in weights.values()):
        raise SourcePriorityError("M0-09 source-priority axis weights must all remain one")
    thresholds = policy.get("thresholds", {})
    required_thresholds = {
        "motif_high_class_count",
        "novelty_high_fraction",
        "redundancy_low_shared_fraction",
        "redundancy_mid_shared_fraction",
    }
    if set(thresholds) != required_thresholds:
        raise SourcePriorityError(
            f"policy thresholds must define exactly {sorted(required_thresholds)}"
        )
    if not (
        0
        <= thresholds["redundancy_low_shared_fraction"]
        < thresholds["redundancy_mid_shared_fraction"]
        <= 1
    ):
        raise SourcePriorityError("redundancy thresholds must be ordered within [0, 1]")
    if not 0 < thresholds["novelty_high_fraction"] <= 1:
        raise SourcePriorityError("novelty_high_fraction must be in (0, 1]")
    if thresholds["motif_high_class_count"] < 1:
        raise SourcePriorityError("motif_high_class_count must be positive")

    paths: dict[str, Path] = {}
    verified: list[dict[str, Any]] = [
        {
            "path": str(path.resolve().relative_to(repo_root.resolve())),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": "priority_policy",
        }
    ]
    required_inputs = {
        "source_queue",
        "component_source_ledger",
        "lnpdb",
        "agile_component_ledger",
        "paper_route_reviews",
        "motif_transfer_result",
        "motif_transfer_ledger",
    }
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != required_inputs:
        raise SourcePriorityError(f"config inputs must define exactly {sorted(required_inputs)}")
    for label in sorted(required_inputs):
        input_path, metadata = _verified_input(repo_root, inputs[label], label)
        paths[label] = input_path
        verified.append({"role": label, **metadata})
    return config, paths, verified


def _read_csv(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise SourcePriorityError(f"{label} is missing required columns: {missing}")
        return list(reader)


def _source_queue(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows = _read_csv(path, set(SOURCE_COLUMNS), "source queue")
    if not rows:
        raise SourcePriorityError("source queue must contain at least one row")
    if tuple(rows[0].keys()) != SOURCE_COLUMNS:
        raise SourcePriorityError("source queue columns do not match the frozen acquisition schema")
    source_ids = [row["source_id"] for row in rows]
    if any(not source_id for source_id in source_ids) or len(source_ids) != len(set(source_ids)):
        raise SourcePriorityError("source queue source_id values must be non-empty and unique")
    ranks = sorted(int(row["review_rank"]) for row in rows)
    if ranks != list(range(1, len(rows) + 1)):
        raise SourcePriorityError("source queue review_rank must be contiguous")
    return rows, {row["source_id"]: row for row in rows}


def _match_lnpdb_sources(
    path: Path,
    source_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    pair_to_source: dict[tuple[str, str], str] = {}
    for row in source_rows:
        try:
            pmids = json.loads(row["reported_pmids_json"])
            links = json.loads(row["publication_links_json"])
        except json.JSONDecodeError as exc:
            raise SourcePriorityError(
                f"source queue row {row['source_id']} has invalid source identity JSON"
            ) from exc
        for pmid in pmids:
            for link in links:
                pair = (str(pmid), str(link))
                prior = pair_to_source.setdefault(pair, row["source_id"])
                if prior != row["source_id"]:
                    raise SourcePriorityError(f"ambiguous source identity pair: {pair}")

    evidence: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "numeric_measurements": 0,
            "in_vivo_measurements": 0,
            "assay_contexts": set(),
        }
    )
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(REQUIRED_LNPDB_COLUMNS).issubset(reader.fieldnames):
            missing = sorted(set(REQUIRED_LNPDB_COLUMNS) - set(reader.fieldnames or ()))
            raise SourcePriorityError(f"raw LNPDB is missing priority fields: {missing}")
        for row_number, row in enumerate(reader, start=2):
            pair = (row["Publication_PMID"].strip(), row["Publication_link"].strip())
            source_id = pair_to_source.get(pair)
            if source_id is None:
                raise SourcePriorityError(
                    f"raw LNPDB row {row_number} does not map to one source queue row: {pair}"
                )
            value = row["Experiment_value"].strip()
            if value and value.upper() != "NA":
                try:
                    float(value)
                except ValueError as exc:
                    raise SourcePriorityError(
                        f"raw LNPDB row {row_number} has nonnumeric Experiment_value {value!r}"
                    ) from exc
                evidence[source_id]["numeric_measurements"] += 1
                if row["Model"].strip() == "in_vivo":
                    evidence[source_id]["in_vivo_measurements"] += 1
                evidence[source_id]["assay_contexts"].add(
                    tuple(row[field].strip() for field in REQUIRED_LNPDB_COLUMNS[2:-1])
                )
    return evidence


def _component_evidence(
    component_path: Path,
    agile_path: Path,
    source_ids: set[str],
    hydrophobic_roles: set[str],
) -> dict[str, dict[str, Any]]:
    component_rows = _read_csv(
        component_path,
        {
            "role",
            "parse_status",
            "canonical_smiles",
            "source_count",
            "source_ids_json",
        },
        "component source ledger",
    )
    agile_rows = _read_csv(
        agile_path,
        {"canonical_smiles"},
        "AGILE component ledger",
    )
    agile_components = {row["canonical_smiles"] for row in agile_rows}
    evidence: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "parsed": set(),
            "agile": set(),
            "novel": set(),
            "exclusive_novel": set(),
            "shared": set(),
        }
    )
    for row in component_rows:
        if row["role"] not in hydrophobic_roles or row["parse_status"] != "parsed":
            continue
        try:
            row_source_ids = {str(value) for value in json.loads(row["source_ids_json"])}
        except json.JSONDecodeError as exc:
            raise SourcePriorityError(
                f"component row has invalid source_ids_json: {row.get('canonical_smiles')}"
            ) from exc
        if int(row["source_count"]) != len(row_source_ids):
            raise SourcePriorityError(
                f"component row source_count disagrees with source_ids_json: "
                f"{row.get('canonical_smiles')}"
            )
        if not row_source_ids.issubset(source_ids):
            raise SourcePriorityError(
                f"component row references unknown sources: {sorted(row_source_ids - source_ids)}"
            )
        smiles = row["canonical_smiles"]
        for source_id in row_source_ids:
            evidence[source_id]["parsed"].add(smiles)
            if len(row_source_ids) > 1:
                evidence[source_id]["shared"].add(smiles)
            if smiles in agile_components:
                evidence[source_id]["agile"].add(smiles)
            else:
                evidence[source_id]["novel"].add(smiles)
                if len(row_source_ids) == 1:
                    evidence[source_id]["exclusive_novel"].add(smiles)
    return evidence


def _paper_review_evidence(
    path: Path,
    source_ids: set[str],
) -> dict[str, dict[str, Any]]:
    payload = _load_json(path, "paper-route reviews")
    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        raise SourcePriorityError("paper-route reviews must contain a reviews list")
    evidence: dict[str, dict[str, Any]] = {}
    for review in reviews:
        source_id = str(review.get("pmid", ""))
        if source_id not in source_ids:
            raise SourcePriorityError(f"reviewed PMID is absent from source queue: {source_id}")
        if source_id in evidence:
            raise SourcePriorityError(f"duplicate paper review PMID: {source_id}")
        asset = review.get("source_asset", {})
        review_complete = asset.get("review_status") == "chemistry_pages_visually_reviewed"
        direct_ugi = False
        for family in review.get("l1_assembly_families", []):
            roles = {str(role).lower() for role in family.get("reactant_roles", [])}
            if roles == {"amine head", "aldehyde tail", "isocyanide tail"}:
                direct_ugi = True
        ugi_forward_verified = sum(
            int(
                family.get("frozen_ugi_forward_verification", {}).get(
                    "members_with_exactly_one_sanitized_product", 0
                )
            )
            for family in review.get("l2_route_families", [])
        )
        if any(
            family.get("frozen_ugi_forward_verification", {}).get(
                "experimental_ugi_success_inferred"
            )
            is not False
            for family in review.get("l2_route_families", [])
            if "frozen_ugi_forward_verification" in family
        ):
            raise SourcePriorityError(
                f"paper review {source_id} improperly infers experimental Ugi success"
            )
        conclusion = review.get("review_conclusion", {})
        evidence[source_id] = {
            "review_id": str(review.get("review_id", "")),
            "review_complete": review_complete,
            "direct_ugi": direct_ugi,
            "upstream_component_routes": int(conclusion.get("upstream_component_routes", 0)),
            "ugi_forward_verified": ugi_forward_verified,
        }
    return evidence


def _motif_evidence(
    result_path: Path,
    ledger_path: Path,
    source_ids: set[str],
) -> dict[str, dict[str, Any]]:
    result = _load_json(result_path, "hydrophobic-motif transfer result")
    source_metadata = result.get("sources")
    if not isinstance(source_metadata, dict):
        raise SourcePriorityError("motif-transfer result must define source metadata")
    source_key_to_queue_id: dict[str, str] = {}
    for source_key, metadata in source_metadata.items():
        source_id = str(metadata.get("pmid") or metadata.get("source_id") or "")
        if source_id not in source_ids:
            raise SourcePriorityError(
                f"motif-transfer source {source_key} is absent from source queue: {source_id}"
            )
        source_key_to_queue_id[source_key] = source_id

    rows = _read_csv(
        ledger_path,
        {
            "record_id",
            "source_id",
            "motif_classes_json",
            "source_attachment_mapping_status",
            "proposed_ugi_component_smiles",
            "frozen_ugi_forward_products",
            "computationally_route_complete",
            "biological_label_inherited",
        },
        "hydrophobic-motif transfer ledger",
    )
    evidence: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "record_ids": [],
            "motif_classes": set(),
            "forward_verified": 0,
            "route_complete": 0,
        }
    )
    for row in rows:
        source_key = row["source_id"]
        if source_key not in source_key_to_queue_id:
            raise SourcePriorityError(f"motif ledger references unknown source key: {source_key}")
        if row["biological_label_inherited"].lower() != "false":
            raise SourcePriorityError(
                f"motif record {row['record_id']} improperly inherits a biological label"
            )
        if row["source_attachment_mapping_status"] != "exact_mapped":
            raise SourcePriorityError(
                f"motif record {row['record_id']} lacks an exact attachment mapping"
            )
        source_id = source_key_to_queue_id[source_key]
        record = evidence[source_id]
        record["record_ids"].append(row["record_id"])
        try:
            record["motif_classes"].update(json.loads(row["motif_classes_json"]))
        except json.JSONDecodeError as exc:
            raise SourcePriorityError(
                f"motif record {row['record_id']} has invalid motif_classes_json"
            ) from exc
        if row["proposed_ugi_component_smiles"] and int(row["frozen_ugi_forward_products"]) > 0:
            record["forward_verified"] += 1
        if row["computationally_route_complete"].lower() == "true":
            record["route_complete"] += 1
    return evidence


def _score_source(
    source: dict[str, str],
    *,
    policy: dict[str, Any],
    biological: dict[str, Any],
    components: dict[str, Any],
    paper_review: dict[str, Any] | None,
    motifs: dict[str, Any] | None,
) -> dict[str, Any]:
    thresholds = policy["thresholds"]
    missing: list[str] = []
    evidence_ids: list[str] = []

    if motifs:
        motif_class_count = len(motifs["motif_classes"])
        motif_score = 2 if motif_class_count >= thresholds["motif_high_class_count"] else 1
        motif_state = "curated_exactly_mapped_transfer_motifs"
        evidence_ids.extend(f"motif:{record_id}" for record_id in motifs["record_ids"])
    else:
        motif_class_count = 0
        motif_score = 0
        motif_state = "not_assessed"
        missing.append("motif_distinctiveness")

    numeric = int(biological.get("numeric_measurements", 0))
    in_vivo = int(biological.get("in_vivo_measurements", 0))
    assay_contexts = len(biological.get("assay_contexts", set()))
    if in_vivo:
        biological_score = 2
        biological_state = "lnpdb_numeric_in_vivo_measurements_present"
    elif numeric:
        biological_score = 1
        biological_state = "lnpdb_numeric_in_vitro_measurements_only"
    else:
        biological_score = 0
        biological_state = "no_numeric_lnpdb_measurement"

    parsed = len(components.get("parsed", set()))
    agile = len(components.get("agile", set()))
    novel = len(components.get("novel", set()))
    exclusive_novel = len(components.get("exclusive_novel", set()))
    if not parsed:
        novelty_score = 0
        novelty_state = "no_parsed_hydrophobic_component"
    else:
        novel_fraction = novel / parsed
        if novel_fraction >= thresholds["novelty_high_fraction"]:
            novelty_score = 2
            novelty_state = "high_exact_identity_novelty_vs_agile"
        elif novel:
            novelty_score = 1
            novelty_state = "partial_exact_identity_novelty_vs_agile"
        else:
            novelty_score = 0
            novelty_state = "all_exact_hydrophobic_identities_present_in_agile"

    if paper_review and paper_review["review_complete"]:
        method_score = 3
        method_state = "chemistry_pages_visually_reviewed"
        evidence_ids.append(f"review:{paper_review['review_id']}")
    elif motifs:
        method_score = 3
        method_state = "targeted_source_reaction_visually_reviewed"
    elif source["acquisition_status"] in {"local_source_available", "pmc_identifier_available"}:
        method_score = 2
        method_state = "hash_pinned_or_public_full_text_available_unreviewed"
    else:
        method_score = 0
        method_state = "source_methods_not_acquired_or_not_snapshotted"

    direct_ugi_routes = bool(
        paper_review
        and paper_review["direct_ugi"]
        and paper_review["upstream_component_routes"] > 0
    )
    paper_forward = int(paper_review["ugi_forward_verified"]) if paper_review else 0
    motif_forward = int(motifs["forward_verified"]) if motifs else 0
    route_complete = int(motifs["route_complete"]) if motifs else 0
    if direct_ugi_routes:
        ugi_score = 3
        ugi_state = "native_ugi_l1_with_reviewed_upstream_component_routes"
    elif route_complete:
        ugi_score = 3
        ugi_state = "transferred_component_forward_verified_and_route_complete"
    elif paper_forward or motif_forward:
        ugi_score = 2
        ugi_state = "transferred_component_forward_verified_route_incomplete"
    else:
        ugi_score = 0
        ugi_state = "not_assessed"
        missing.append("ugi_handle_conversion")

    shared = len(components.get("shared", set()))
    if not parsed:
        redundancy_score = 0
        redundancy_state = "no_parsed_hydrophobic_component"
        shared_fraction = 0.0
    else:
        shared_fraction = shared / parsed
        if shared_fraction <= thresholds["redundancy_low_shared_fraction"]:
            redundancy_score = 2
            redundancy_state = "low_exact_component_redundancy"
        elif shared_fraction <= thresholds["redundancy_mid_shared_fraction"]:
            redundancy_score = 1
            redundancy_state = "moderate_exact_component_redundancy"
        else:
            redundancy_score = 0
            redundancy_state = "high_exact_component_redundancy"

    scores = {
        "motif_distinctiveness": motif_score,
        "biological_provenance": biological_score,
        "agile_exact_novelty": novelty_score,
        "method_accessibility": method_score,
        "ugi_handle_conversion": ugi_score,
        "exact_component_redundancy": redundancy_score,
    }
    priority_score = sum(scores[axis] * policy["axis_weights"][axis] for axis in AXES)
    priority_max_score = 2 + 2 + 2 + 3 + 3 + 2
    if priority_score > priority_max_score:
        raise SourcePriorityError(
            f"priority score exceeds frozen maximum for {source['source_id']}"
        )

    review_completion = (
        "chemistry_review_complete"
        if paper_review and paper_review["review_complete"]
        else "pending_or_targeted_partial_review"
    )
    return {
        **source,
        "priority_rank": 0,
        "next_review_rank": "",
        "priority_score": priority_score,
        "priority_max_score": priority_max_score,
        "assessed_axis_count": len(AXES) - len(missing),
        "review_completion_state": review_completion,
        "missing_priority_axes_json": json.dumps(sorted(missing)),
        "motif_distinctiveness_score": motif_score,
        "motif_distinctiveness_state": motif_state,
        "curated_motif_count": len(motifs["record_ids"]) if motifs else 0,
        "curated_motif_class_count": motif_class_count,
        "biological_provenance_score": biological_score,
        "biological_provenance_state": biological_state,
        "numeric_measurement_count": numeric,
        "in_vivo_measurement_count": in_vivo,
        "assay_context_count": assay_contexts,
        "agile_exact_novelty_score": novelty_score,
        "agile_exact_novelty_state": novelty_state,
        "parsed_hydrophobic_component_count": parsed,
        "exact_agile_hydrophobic_component_count": agile,
        "non_agile_hydrophobic_component_count": novel,
        "source_exclusive_non_agile_component_count": exclusive_novel,
        "method_accessibility_score": method_score,
        "method_accessibility_state": method_state,
        "ugi_handle_conversion_score": ugi_score,
        "ugi_handle_conversion_state": ugi_state,
        "ugi_forward_verified_component_count": paper_forward + motif_forward,
        "ugi_route_complete_component_count": route_complete,
        "exact_component_redundancy_score": redundancy_score,
        "exact_component_redundancy_state": redundancy_state,
        "shared_hydrophobic_component_count": shared,
        "shared_hydrophobic_component_fraction": f"{shared_fraction:.6f}",
        "priority_evidence_ids_json": json.dumps(sorted(evidence_ids)),
    }


def _csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(*SOURCE_COLUMNS, *PRIORITY_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def build_source_priority(
    config_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the deterministic evidence-bounded priority overlay."""

    config, paths, verified_inputs = load_config(config_path, repo_root)
    policy = config["policy"]
    source_rows, source_index = _source_queue(paths["source_queue"])
    source_ids = set(source_index)
    biological = _match_lnpdb_sources(paths["lnpdb"], source_rows)
    components = _component_evidence(
        paths["component_source_ledger"],
        paths["agile_component_ledger"],
        source_ids,
        set(policy["hydrophobic_roles"]),
    )
    paper_reviews = _paper_review_evidence(paths["paper_route_reviews"], source_ids)
    motifs = _motif_evidence(
        paths["motif_transfer_result"],
        paths["motif_transfer_ledger"],
        source_ids,
    )

    priority_rows = [
        _score_source(
            source,
            policy=policy,
            biological=biological.get(source["source_id"], {}),
            components=components.get(source["source_id"], {}),
            paper_review=paper_reviews.get(source["source_id"]),
            motifs=motifs.get(source["source_id"]),
        )
        for source in source_rows
    ]
    priority_rows.sort(
        key=lambda row: (
            -int(row["priority_score"]),
            -int(row["assessed_axis_count"]),
            int(row["review_rank"]),
            row["source_id"],
        )
    )
    for rank, row in enumerate(priority_rows, start=1):
        row["priority_rank"] = rank
    active_rows = [
        row
        for row in priority_rows
        if row["review_completion_state"] != "chemistry_review_complete"
    ]
    for rank, row in enumerate(active_rows, start=1):
        row["next_review_rank"] = rank

    output_text = _csv_text(priority_rows)
    axis_coverage = {
        axis: sum(
            axis not in json.loads(row["missing_priority_axes_json"]) for row in priority_rows
        )
        for axis in AXES
    }
    timestamp = str(config["generated_utc"])
    try:
        parsed_timestamp = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SourcePriorityError("generated_utc must be valid ISO-8601") from exc
    if parsed_timestamp.tzinfo is None:
        raise SourcePriorityError("generated_utc must be timezone-aware")

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-09 source-paper marginal-value priority overlay",
        "status": "completed_evidence_bounded_source_priority_queue",
        "generated_utc": timestamp,
        "randomness": {"used": False, "seed": int(config["seed"])},
        "inputs": verified_inputs,
        "policy": policy,
        "summary": {
            "source_records": len(priority_rows),
            "acquisition_publications_preserved": sum(
                row["source_kind"] == "publication" for row in priority_rows
            ),
            "acquisition_commercial_sources_preserved": sum(
                row["source_kind"] == "commercial_catalog" for row in priority_rows
            ),
            "acquisition_bucket_counts_preserved": dict(
                sorted(Counter(row["review_bucket"] for row in priority_rows).items())
            ),
            "chemistry_review_complete_sources": sum(
                review["review_complete"] for review in paper_reviews.values()
            ),
            "active_next_review_sources": len(active_rows),
            "axis_evidence_coverage": axis_coverage,
            "sources_with_missing_priority_axes": sum(
                bool(json.loads(row["missing_priority_axes_json"])) for row in priority_rows
            ),
            "priority_score_distribution": dict(
                sorted(
                    Counter(str(row["priority_score"]) for row in priority_rows).items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "top_active_sources": [
                {
                    "next_review_rank": row["next_review_rank"],
                    "source_id": row["source_id"],
                    "priority_score": row["priority_score"],
                    "missing_priority_axes": json.loads(row["missing_priority_axes_json"]),
                }
                for row in active_rows[:10]
            ],
        },
        "evidence_contract": {
            "score_interpretation": (
                "The score prioritizes evidence review. It is not a synthesis-success, "
                "biological-potency, route-closure, or procurement score."
            ),
            "missing_evidence": (
                "Unassessed motif or Ugi-conversion evidence receives zero credit and is listed "
                "explicitly; missing evidence is never inferred as favorable."
            ),
            "biological_provenance": (
                "Biological provenance records the presence and depth of source-linked LNPDB "
                "measurements, not their magnitude or transferability to a generated Ugi lipid."
            ),
            "novelty": (
                "Novelty and redundancy are exact component-identity diagnostics only. They do "
                "not establish motif transfer, Ugi compatibility, or biological value."
            ),
            "acquisition_stability": (
                "The original source_queue.csv remains the acquisition authority. This overlay "
                "does not change its rows, review ranks, buckets, or downstream hashes."
            ),
        },
        "decision": {
            "marginal_value_queue_implemented": True,
            "next_action": (
                "Use next_review_rank for targeted source review, then let generator-induced "
                "missing-route failures and the frozen saturation rule control further curation."
            ),
            "model_training_authorized": False,
        },
        "artifacts": {
            "source_priority_queue.csv": {
                "records": len(priority_rows),
                "sha256": hashlib.sha256(output_text.encode()).hexdigest(),
            }
        },
    }
    return result, priority_rows


def write_source_priority(
    result: dict[str, Any],
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write the priority result and CSV without partial outputs."""

    csv_bytes = _csv_text(rows).encode()
    expected = result["artifacts"]["source_priority_queue.csv"]["sha256"]
    if hashlib.sha256(csv_bytes).hexdigest() != expected:
        raise SourcePriorityError("source-priority queue changed after result construction")
    result_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(output_dir / "source_priority_queue.csv", csv_bytes)
    _atomic_write(output_dir / "source_priority_result.json", result_bytes)

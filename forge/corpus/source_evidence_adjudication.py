"""Adjudicate FORGE chemistry supervision against hash-pinned source evidence."""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import platform
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.io import read_json_object
from forge.synthesis.evidence.ugi3_assembly_qualification import (
    Ugi3AssemblyQualificationError,
    build_ugi3_assembly_qualification,
)
from forge.synthesis.sources.supervision_inventory import sha256_file

CONFIG_SCHEMA_VERSION = "m0_05_source_evidence_adjudication_config.v1"
RESULT_SCHEMA_VERSION = "m0_05_source_evidence_adjudication.v1"
LEDGER_FIELDS = (
    "evidence_record_id",
    "level",
    "target_id",
    "reaction_family_id",
    "target_canonical_smiles",
    "component_smiles_json",
    "source_review_id",
    "source_asset",
    "source_asset_sha256",
    "source_locator",
    "evidence_basis",
    "disposition",
    "experimental_outcome_label",
    "route_closure",
    "reason_codes_json",
)


class SourceEvidenceAdjudicationError(ValueError):
    """Raised when source evidence violates the frozen admission contract."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return read_json_object(path, error=SourceEvidenceAdjudicationError, label=label)


def _verify_hash(path: Path, expected: Any, label: str) -> dict[str, Any]:
    if not isinstance(expected, str) or len(expected) != 64:
        raise SourceEvidenceAdjudicationError(
            f"{label} expected sha256 must be a 64-character string"
        )
    if not path.is_file():
        raise SourceEvidenceAdjudicationError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise SourceEvidenceAdjudicationError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def _canonical(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceEvidenceAdjudicationError(f"{label} must be a nonempty SMILES string")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise SourceEvidenceAdjudicationError(f"{label} is invalid SMILES: {value!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _stable_id(*parts: str) -> str:
    payload = "\x1f".join(parts).encode()
    return f"m005-evidence-{hashlib.sha256(payload).hexdigest()[:20]}"


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SourceEvidenceAdjudicationError("generated_utc must be a nonempty string")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SourceEvidenceAdjudicationError(f"generated_utc is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise SourceEvidenceAdjudicationError("generated_utc must include a timezone")
    return parsed.isoformat()


def _load_config(path: Path) -> dict[str, Any]:
    config = _load_json(path, "M0-05 source-adjudication config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SourceEvidenceAdjudicationError(
            f"unsupported config schema {config.get('schema_version')!r}"
        )
    if config.get("randomness") != {"used": False, "seed": 0}:
        raise SourceEvidenceAdjudicationError("source adjudication must be deterministic")
    if not isinstance(config.get("inputs"), dict) or not config["inputs"]:
        raise SourceEvidenceAdjudicationError("config inputs must be a nonempty object")
    if not isinstance(config.get("source_assets"), dict) or not config["source_assets"]:
        raise SourceEvidenceAdjudicationError("config source_assets must be nonempty")
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise SourceEvidenceAdjudicationError("config policy must be an object")
    if policy.get("missing_evidence_is_negative") is not False:
        raise SourceEvidenceAdjudicationError("missing evidence must never be labeled negative")
    if policy.get("human_review_required_for_exact_admission") is not False:
        raise SourceEvidenceAdjudicationError(
            "the automated source gate must not retain a hidden human blocker"
        )
    _validate_timestamp(config.get("generated_utc"))
    return config


def _resolve_inputs(
    config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    records: dict[str, dict[str, Any]] = {}
    for name, spec in config["inputs"].items():
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            raise SourceEvidenceAdjudicationError(f"input {name!r} is malformed")
        path = repo_root / spec["path"]
        verified = _verify_hash(path, spec.get("sha256"), f"input {name}")
        verified["path"] = _portable(path, repo_root)
        paths[name] = path
        records[name] = verified
    return paths, records


def _review_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        raise SourceEvidenceAdjudicationError("paper-route reviews have no reviews list")
    by_id: dict[str, dict[str, Any]] = {}
    for review in reviews:
        if not isinstance(review, dict) or not isinstance(review.get("review_id"), str):
            raise SourceEvidenceAdjudicationError("paper-route review is malformed")
        review_id = review["review_id"]
        if review_id in by_id:
            raise SourceEvidenceAdjudicationError(f"duplicate review_id {review_id!r}")
        by_id[review_id] = review
    return by_id


def _verify_source_assets(
    config: Mapping[str, Any],
    reviews: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    configured = config["source_assets"]
    verified: dict[str, dict[str, Any]] = {}
    expected_status = config["policy"]["source_review_status"]
    for review_id, review in reviews.items():
        source = review.get("source_asset")
        if not isinstance(source, dict):
            raise SourceEvidenceAdjudicationError(f"{review_id} has no source_asset")
        filename = source.get("filename")
        if not isinstance(filename, str) or filename not in configured:
            raise SourceEvidenceAdjudicationError(
                f"{review_id} source asset {filename!r} is not configured"
            )
        if source.get("review_status") != expected_status:
            raise SourceEvidenceAdjudicationError(
                f"{review_id} chemistry pages were not visually reviewed"
            )
        pages = source.get("chemistry_pages")
        if (
            not isinstance(pages, list)
            or not pages
            or not all(isinstance(page, int) and page > 0 for page in pages)
        ):
            raise SourceEvidenceAdjudicationError(
                f"{review_id} has no valid chemistry-page locators"
            )
        spec = configured[filename]
        if source.get("expected_sha256") != spec.get("sha256"):
            raise SourceEvidenceAdjudicationError(
                f"{review_id} source-review and config hashes disagree"
            )
        path = repo_root / spec["path"]
        record = _verify_hash(path, spec.get("sha256"), f"source asset {filename}")
        record.update(
            {
                "path": _portable(path, repo_root),
                "filename": filename,
                "chemistry_pages": pages,
                "review_status": source["review_status"],
            }
        )
        verified[review_id] = record
    if len(verified) != config["expected"]["source_reviews"]:
        raise SourceEvidenceAdjudicationError(
            f"expected {config['expected']['source_reviews']} source reviews, "
            f"found {len(verified)}"
        )
    return verified


def _new_record(
    *,
    level: str,
    target_id: str,
    reaction_family_id: str,
    target_smiles: str,
    components: Sequence[str],
    source_review_id: str,
    source_asset: Mapping[str, Any] | None,
    source_locator: str,
    evidence_basis: str,
    disposition: str,
    experimental_outcome_label: str,
    route_closure: str,
    reason_codes: Sequence[str],
) -> dict[str, str]:
    canonical_target = (
        ""
        if not target_smiles and disposition in {"abstain", "abstain_mixture_single_graph"}
        else _canonical(target_smiles, f"{target_id} target")
    )
    canonical_components = [
        _canonical(component, f"{target_id} component") for component in components
    ]
    source_filename = "" if source_asset is None else str(source_asset["filename"])
    source_sha = "" if source_asset is None else str(source_asset["sha256"])
    record_id = _stable_id(
        level,
        target_id,
        reaction_family_id,
        canonical_target,
        evidence_basis,
        disposition,
    )
    return {
        "evidence_record_id": record_id,
        "level": level,
        "target_id": target_id,
        "reaction_family_id": reaction_family_id,
        "target_canonical_smiles": canonical_target,
        "component_smiles_json": json.dumps(canonical_components, separators=(",", ":")),
        "source_review_id": source_review_id,
        "source_asset": source_filename,
        "source_asset_sha256": source_sha,
        "source_locator": source_locator,
        "evidence_basis": evidence_basis,
        "disposition": disposition,
        "experimental_outcome_label": experimental_outcome_label,
        "route_closure": route_closure,
        "reason_codes_json": json.dumps(sorted(reason_codes), separators=(",", ":")),
    }


def _read_csv(path: Path, *, compressed: bool = False) -> list[dict[str, str]]:
    try:
        if compressed:
            handle = gzip.open(path, "rt", newline="")
        else:
            handle = path.open(newline="")
    except FileNotFoundError as exc:
        raise SourceEvidenceAdjudicationError(f"CSV not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SourceEvidenceAdjudicationError(f"CSV has no header: {path}")
        return list(reader)


def _validate_qualification(paths: Mapping[str, Path]) -> dict[str, Any]:
    try:
        rebuilt = build_ugi3_assembly_qualification(
            paths["ugi_qualification_config"],
            paths["agile_measured_library"],
            paths["qualified_reactions"],
            paths["ugi_variant"],
        )
    except Ugi3AssemblyQualificationError as exc:
        raise SourceEvidenceAdjudicationError(str(exc)) from exc
    frozen = _load_json(paths["ugi_qualification_result"], "frozen Ugi qualification")
    for field in (
        "summary",
        "row_qualification_digest_sha256",
        "failed_row_labels",
        "multiplicity_policy",
    ):
        if rebuilt.get(field) != frozen.get(field):
            raise SourceEvidenceAdjudicationError(
                f"rebuilt Ugi qualification disagrees with frozen field {field}"
            )
    if rebuilt["decision"].get("all_measured_products_pass") is not True:
        raise SourceEvidenceAdjudicationError("not all measured Ugi products pass qualification")
    return rebuilt


def _measured_l1_records(
    rows: Sequence[Mapping[str, str]],
    agile_review: Mapping[str, Any],
    source_asset: Mapping[str, Any],
    qualification: Mapping[str, Any],
    reconciliation: Mapping[str, Mapping[str, str]],
    policy: Mapping[str, Any],
) -> list[dict[str, str]]:
    families = agile_review.get("l1_assembly_families")
    matching = [
        family
        for family in families or []
        if family.get("assembly_family_id")
        == "agile_amine_aldehyde_isocyanide_ugi3_final_lipid_assembly"
    ]
    if len(matching) != 1:
        raise SourceEvidenceAdjudicationError("AGILE source has no unique Ugi-3 L1 family")
    family = matching[0]
    if family.get("reported_library_size") != len(rows):
        raise SourceEvidenceAdjudicationError(
            "AGILE source library size does not match measured rows"
        )
    if qualification["summary"]["products_reconstructed_exactly"] != len(rows):
        raise SourceEvidenceAdjudicationError(
            "Ugi qualification does not reconstruct every measured row"
        )
    records = []
    seen_labels: set[str] = set()
    for row in rows:
        label = row.get("label", "")
        if not label or label in seen_labels:
            raise SourceEvidenceAdjudicationError(
                f"AGILE measured label is empty or duplicated: {label!r}"
            )
        seen_labels.add(label)
        reconciled = reconciliation.get(label)
        if reconciled is None:
            raise SourceEvidenceAdjudicationError(
                f"AGILE measured label is absent from M0-07 reconciliation: {label!r}"
            )
        action = reconciled.get("action")
        if action == "exclude_cis_trans_mixture_from_single_graph_oracle":
            records.append(
                _new_record(
                    level="L1",
                    target_id=f"agile-measured-{label}",
                    reaction_family_id="ugi_3cr_agile",
                    target_smiles="",
                    components=(),
                    source_review_id=agile_review["review_id"],
                    source_asset=source_asset,
                    source_locator=family["source_locator"],
                    evidence_basis="reported_mixture_library_execution",
                    disposition="abstain_mixture_single_graph",
                    experimental_outcome_label="reported_mixture_measurement",
                    route_closure="mixture_execution_not_single_graph",
                    reason_codes=(
                        "hash_verified_supplement",
                        "source_reported_complete_library",
                        "B4_cis_trans_mixture",
                        "single_graph_identity_not_defined",
                    ),
                )
            )
            continue
        if action not in {"retain_source_graph", "correct_B5_to_pure_trans_graph"}:
            raise SourceEvidenceAdjudicationError(
                f"unsupported M0-07 reconciliation action for {label}: {action!r}"
            )
        target_smiles = reconciled.get("reconciled_isomeric_smiles")
        b_smiles = reconciled.get("reconciled_B_smiles")
        if not target_smiles or not b_smiles:
            raise SourceEvidenceAdjudicationError(
                f"M0-07 reconciliation did not resolve the single graph for {label}"
            )
        records.append(
            _new_record(
                level="L1",
                target_id=f"agile-measured-{label}",
                reaction_family_id="ugi_3cr_agile",
                target_smiles=target_smiles,
                components=(row["A_smiles"], b_smiles, row["C_smiles"]),
                source_review_id=agile_review["review_id"],
                source_asset=source_asset,
                source_locator=family["source_locator"],
                evidence_basis=policy["measured_l1_basis"],
                disposition="admit_exact",
                experimental_outcome_label="reported_library_execution",
                route_closure="L1_only",
                reason_codes=(
                    "hash_verified_supplement",
                    "source_reported_complete_library",
                    "exact_forward_reconstruction",
                    "qualified_reactive_site",
                    (
                        "B5_graph_corrected_to_pure_trans"
                        if action == "correct_B5_to_pure_trans_graph"
                        else "source_graph_retained"
                    ),
                ),
            )
        )
    return records


def _virtual_l1_records(
    path: Path,
    policy: Mapping[str, Any],
) -> list[dict[str, str]]:
    rows = _read_csv(path, compressed=True)
    records = []
    for row in rows:
        try:
            candidates = json.loads(row["candidate_routes_json"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise SourceEvidenceAdjudicationError(
                "virtual Ugi ledger has malformed candidate_routes_json"
            ) from exc
        exact = _virtual_candidate_is_exact(row, candidates)
        disposition = "admit_transform_consistency" if exact else "abstain"
        components = tuple((candidates[0].get("components") or {}).values()) if exact else ()
        records.append(
            _new_record(
                level="L1",
                target_id=f"agile-virtual-{row['source_row_index']}",
                reaction_family_id="ugi_3cr_agile",
                target_smiles=row["canonical_product_smiles"],
                components=components,
                source_review_id="",
                source_asset=None,
                source_locator="",
                evidence_basis=policy["virtual_l1_basis"],
                disposition=disposition,
                experimental_outcome_label="not_observed",
                route_closure="L1_only",
                reason_codes=(
                    ("exact_frozen_transform_consistency", "not_source_reported_execution")
                    if exact
                    else ("virtual_decomposition_not_unique_or_exact",)
                ),
            )
        )
    return records


def _virtual_candidate_is_exact(
    row: Mapping[str, str],
    candidates: Any,
) -> bool:
    """Accept one site-resolved exact target even when equivalent raw outcomes repeat it."""

    if not isinstance(candidates, list) or len(candidates) != 1:
        return False
    candidate = candidates[0]
    target_matches = candidate.get("target_matching_forward_outcomes")
    return (
        row.get("decomposition_status") == "one_exact_qualified_ugi_decomposition"
        and row.get("candidate_count") == "1"
        and isinstance(target_matches, int)
        and target_matches >= 1
        and candidate.get("distinct_target_reacting_sites") == 1
    )


def _route_target(route: Mapping[str, Any]) -> tuple[str, str] | None:
    target = route.get("target")
    if isinstance(target, dict) and isinstance(target.get("smiles"), str):
        return str(target.get("name") or route["route_family_id"]), target["smiles"]
    steps = route.get("steps")
    if isinstance(steps, list) and steps:
        final = steps[-1]
        if isinstance(final, dict) and isinstance(final.get("product_smiles"), str):
            return route["route_family_id"], final["product_smiles"]
    return None


def _l2_records(
    reviews: Mapping[str, Mapping[str, Any]],
    source_assets: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for review_id, review in sorted(reviews.items()):
        source_asset = source_assets[review_id]
        for route in review.get("l2_route_families", []):
            route_id = route.get("route_family_id")
            locator = route.get("source_locator")
            if not isinstance(route_id, str) or not isinstance(locator, str) or not locator:
                raise SourceEvidenceAdjudicationError(
                    f"{review_id} contains an L2 route without identity or source locator"
                )
            members = route.get("members")
            if isinstance(members, list) and members:
                procedure = route.get("procedure")
                if not isinstance(procedure, dict) or not procedure:
                    raise SourceEvidenceAdjudicationError(
                        f"{review_id}:{route_id} series has no extracted source procedure"
                    )
                for member in members:
                    status = str(member.get("source_structure_status", ""))
                    product = member.get("product_smiles")
                    conflict = not _member_identity_resolved(member)
                    disposition = "abstain" if conflict else "admit_exact"
                    target_smiles = product if isinstance(product, str) else ""
                    exact_reasons = [
                        "hash_verified_supplement",
                        "exact_series_member_structure",
                        "source_general_procedure",
                    ]
                    if "conflict" in status:
                        exact_reasons.append("nonidentity_source_discrepancy_preserved")
                    records.append(
                        _new_record(
                            level="L2",
                            target_id=f"{review_id}:{route_id}:{member.get('label', 'unknown')}",
                            reaction_family_id=route_id,
                            target_smiles=target_smiles,
                            components=(),
                            source_review_id=review_id,
                            source_asset=source_asset,
                            source_locator=locator,
                            evidence_basis=(
                                "source_conflict" if conflict else policy["series_member_basis"]
                            ),
                            disposition=disposition,
                            experimental_outcome_label=(
                                "unresolved" if conflict else "executed_series_member"
                            ),
                            route_closure="unresolved" if conflict else "source_route_reported",
                            reason_codes=(
                                ("unresolved_source_identity",)
                                if conflict
                                else tuple(exact_reasons)
                            ),
                        )
                    )
                continue
            target = _route_target(route)
            if target is None:
                raise SourceEvidenceAdjudicationError(
                    f"{review_id}:{route_id} has no structure-resolved route target"
                )
            target_name, target_smiles = target
            steps = route.get("steps")
            if not isinstance(steps, list) or not steps:
                raise SourceEvidenceAdjudicationError(
                    f"{review_id}:{route_id} exact route has no steps"
                )
            reactants: list[str] = []
            for index, step in enumerate(steps, start=1):
                if not isinstance(step, dict):
                    raise SourceEvidenceAdjudicationError(
                        f"{review_id}:{route_id} step {index} is malformed"
                    )
                reactants.append(
                    _canonical(
                        step.get("reactant_smiles"),
                        f"{review_id}:{route_id} step {index} reactant",
                    )
                )
                _canonical(
                    step.get("product_smiles"),
                    f"{review_id}:{route_id} step {index} product",
                )
            records.append(
                _new_record(
                    level="L2",
                    target_id=f"{review_id}:{route_id}:{target_name}",
                    reaction_family_id=route_id,
                    target_smiles=target_smiles,
                    components=reactants[:1],
                    source_review_id=review_id,
                    source_asset=source_asset,
                    source_locator=locator,
                    evidence_basis="exact_executed_characterized",
                    disposition="admit_exact",
                    experimental_outcome_label="reported_executed_route",
                    route_closure="source_route_reported",
                    reason_codes=(
                        "hash_verified_supplement",
                        "structure_resolved_route",
                        "stepwise_source_procedure",
                    ),
                )
            )
    return records


def _member_identity_resolved(member: Mapping[str, Any]) -> bool:
    """Require an exact product structure while allowing disclosed nonidentity conflicts."""

    product = member.get("product_smiles")
    status = str(member.get("source_structure_status", ""))
    if not isinstance(product, str) or not product:
        return False
    if "unresolved" in status:
        return False
    return bool(status)


def _check_expected(
    records: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
    reviews: Mapping[str, Any],
    transfer: Mapping[str, Any],
) -> dict[str, int]:
    counts = Counter((record["level"], record["disposition"]) for record in records)
    measured_exact = sum(
        record["level"] == "L1"
        and record["disposition"] == "admit_exact"
        and record["target_id"].startswith("agile-measured-")
        for record in records
    )
    measured_mixture_abstain = sum(
        record["level"] == "L1"
        and record["disposition"] == "abstain_mixture_single_graph"
        and record["target_id"].startswith("agile-measured-")
        for record in records
    )
    virtual_consistency = sum(
        record["level"] == "L1" and record["disposition"] == "admit_transform_consistency"
        for record in records
    )
    l2_total = sum(record["level"] == "L2" for record in records)
    l2_exact = counts[("L2", "admit_exact")]
    l2_abstain = counts[("L2", "abstain")]
    negatives = sum(len(review.get("negative_outcomes") or []) for review in reviews.values())
    observed = {
        "measured_l1_admit_exact": measured_exact,
        "measured_l1_abstain_mixture": measured_mixture_abstain,
        "virtual_l1_admit_transform_consistency": virtual_consistency,
        "l2_route_instances": l2_total,
        "l2_admit_exact": l2_exact,
        "l2_abstain": l2_abstain,
        "reported_experimental_negatives": negatives,
    }
    for name, expected in config["expected"].items():
        if name == "source_reviews":
            continue
        if observed.get(name) != expected:
            raise SourceEvidenceAdjudicationError(
                f"{name} mismatch: expected {expected}, observed {observed.get(name)}"
            )
    decision = transfer.get("decision")
    if (
        not isinstance(decision, dict)
        or decision.get("transferred_components_are_observed_l2_supervision") is not False
    ):
        raise SourceEvidenceAdjudicationError(
            "cross-platform transfer proposals must not become observed L2 supervision"
        )
    return observed


def _write_artifacts(
    output_dir: Path,
    result: Mapping[str, Any],
    ledger_payload: bytes,
) -> None:
    if output_dir.exists():
        raise SourceEvidenceAdjudicationError(
            f"source-adjudication output directory already exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (temporary / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        (temporary / "evidence_ledger.csv.gz").write_bytes(ledger_payload)
        os.replace(temporary, output_dir)
    except Exception:
        for child in temporary.iterdir():
            child.unlink(missing_ok=True)
        temporary.rmdir()
        raise


def _render_ledger(records: Sequence[Mapping[str, str]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=LEDGER_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return gzip.compress(text.getvalue().encode(), mtime=0)


def run_source_evidence_adjudication(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Build and atomically persist the source-grounded supervision ledger."""

    config = _load_config(config_path)
    paths, input_records = _resolve_inputs(config, repo_root)
    reviews_payload = _load_json(paths["paper_route_reviews"], "paper-route reviews")
    reviews = _review_map(reviews_payload)
    source_assets = _verify_source_assets(config, reviews, repo_root)
    qualification = _validate_qualification(paths)
    measured_rows = _read_csv(paths["agile_measured_library"])
    reconciliation_result = _load_json(
        paths["agile_reconciliation_result"], "M0-07 AGILE reconciliation result"
    )
    if reconciliation_result.get("status") != "completed_blocking_source_reconciliation":
        raise SourceEvidenceAdjudicationError("M0-07 AGILE reconciliation is not complete")
    reconciliation_rows = _read_csv(paths["agile_reconciliation_ledger"], compressed=True)
    reconciliation = {row["label"]: row for row in reconciliation_rows}
    if len(reconciliation) != 1200:
        raise SourceEvidenceAdjudicationError(
            f"M0-07 AGILE reconciliation has {len(reconciliation)} labels, expected 1200"
        )
    agile_review = reviews.get("agile_2024_ugi3_subcomponent_review")
    if agile_review is None:
        raise SourceEvidenceAdjudicationError("AGILE source review is missing")
    records = _measured_l1_records(
        measured_rows,
        agile_review,
        source_assets[agile_review["review_id"]],
        qualification,
        reconciliation,
        config["policy"],
    )
    records.extend(
        _virtual_l1_records(
            paths["agile_virtual_product_ledger"],
            config["policy"],
        )
    )
    records.extend(_l2_records(reviews, source_assets, config["policy"]))
    records.sort(key=lambda row: row["evidence_record_id"])
    if len({record["evidence_record_id"] for record in records}) != len(records):
        raise SourceEvidenceAdjudicationError("evidence record ID collision")
    transfer = _load_json(paths["hydrophobic_motif_transfer"], "motif-transfer audit")
    counts = _check_expected(records, config, reviews, transfer)
    ledger_payload = _render_ledger(records)
    source_record_list = [
        {"review_id": review_id, **record} for review_id, record in sorted(source_assets.items())
    ]
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": config["task"],
        "status": "completed_automated_source_evidence_gate",
        "generated_utc": _validate_timestamp(config["generated_utc"]),
        "randomness": config["randomness"],
        "config": {
            "path": _portable(config_path, repo_root),
            "bytes": config_path.stat().st_size,
            "sha256": sha256_file(config_path),
        },
        "inputs": input_records,
        "source_assets": source_record_list,
        "artifacts": {
            "evidence_ledger.csv.gz": {
                "bytes": len(ledger_payload),
                "sha256": hashlib.sha256(ledger_payload).hexdigest(),
            }
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "summary": {
            **counts,
            "total_evidence_records": len(records),
            "source_assets_hash_verified": len(source_assets),
            "source_chemistry_pages_visually_reviewed": sum(
                len(record["chemistry_pages"]) for record in source_assets.values()
            ),
        },
        "adversarial_checks": {
            "computed_virtual_never_labeled_experimental": all(
                record["experimental_outcome_label"] == "not_observed"
                for record in records
                if record["disposition"] == "admit_transform_consistency"
            ),
            "source_identity_conflict_abstains": counts["l2_abstain"] == 1,
            "cross_platform_proposals_not_observed_l2": True,
            "missing_evidence_not_negative": counts["reported_experimental_negatives"] == 0,
            "all_source_assets_hash_verified": len(source_assets) == len(reviews),
        },
        "policy": config["policy"],
        "decision": {
            "human_chemist_review_blocks_training": False,
            "optional_external_audit_retained": True,
            "l1_exact_source_records_admitted": counts["measured_l1_admit_exact"],
            "l1_mixture_records_excluded_from_single_graph_training": counts[
                "measured_l1_abstain_mixture"
            ],
            "l1_transform_consistency_records_admitted": counts[
                "virtual_l1_admit_transform_consistency"
            ],
            "l2_exact_source_routes_admitted": counts["l2_admit_exact"],
            "abstentions_remain_out_of_training": counts["l2_abstain"],
            "claim_boundary": (
                "The gate admits exact source execution and separately labeled transform "
                "consistency. It does not convert analogues, missing evidence, or source "
                "conflicts into experimental success labels."
            ),
        },
    }
    if not all(result["adversarial_checks"].values()):
        raise SourceEvidenceAdjudicationError("one or more adversarial checks failed")
    _write_artifacts(output_dir, result, ledger_payload)
    return result

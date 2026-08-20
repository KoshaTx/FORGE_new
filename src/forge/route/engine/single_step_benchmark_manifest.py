"""Freeze development-only targets for the single-step proposal-lane benchmark.

The builder is deliberately nonexecuting: it never imports or calls a proposal
backend.  It reads only hash-pinned development artifacts through an allowlisted
reader, separates lane-facing identities from scoring truth, and emits explicit
target-specific visibility masks for known-route and held-family cases.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.data.r1_prime_audit import sha256_file
from forge.design.ugi_tail_chemotype_audit import component_chemotype_metrics

CONFIG_SCHEMA_VERSION = "forge.single_step_benchmark_manifest_builder.v1"
LANE_MANIFEST_SCHEMA_VERSION = "forge.single_step_benchmark_lane_targets.v1"
SCORING_TRUTH_SCHEMA_VERSION = "forge.single_step_benchmark_scoring_truth.v1"
VISIBILITY_MASK_SCHEMA_VERSION = "forge.single_step_benchmark_visibility_masks.v1"
RESULT_SCHEMA_VERSION = "forge.single_step_benchmark_manifest_result.v1"
FROZEN_STATUS = "frozen_development_manifest_benchmark_not_executed"

OUTPUT_FILENAMES = {
    "lane_targets": "lane_targets.json.gz",
    "scoring_truth": "scoring_truth.json.gz",
    "visibility_masks": "visibility_masks.json.gz",
    "result": "result.json",
}
EXPECTED_INPUTS = {
    "benchmark_policy",
    "exact_source_step_verification",
    "exact_source_verification_config",
    "source_component_routes",
    "component_priority_ledger",
    "component_program_ledger",
    "expanded_component_exemplars",
    "ugi_l1_registry",
}
EXPECTED_IMPLEMENTATION = {"builder_module", "builder_cli"}
ROUTE_STRATA = {"known_exact_l2_routes", "held_reaction_families"}
ALDEHYDE_ROLE = "oxoester_aldehyde_body_tail"
ISOCYANIDE_ROLE = "isocyanide_tail"
HEAD_ROLE = "amine_head"


class SingleStepBenchmarkManifestError(ValueError):
    """Raised when an honest, immutable development manifest cannot be built."""


def _stable_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _gzip_json_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(_stable_json_bytes(value))
    return buffer.getvalue()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(value: Any) -> str:
    return _sha256_bytes(_stable_json_bytes(value))


def _path(repo: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SingleStepBenchmarkManifestError(f"{label} path is not explicitly bound")
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else repo / candidate


def _portable(path: Path, *, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


@dataclass
class _PinnedReader:
    repo: Path
    specifications: Mapping[str, Any]
    forbidden_inputs: tuple[str, ...]
    accessed: dict[str, dict[str, str]]

    def __init__(
        self,
        repo: Path,
        specifications: Mapping[str, Any],
        forbidden_inputs: Sequence[str],
    ) -> None:
        self.repo = repo.resolve()
        self.specifications = specifications
        self.forbidden_inputs = tuple(str(value) for value in forbidden_inputs)
        self.accessed = {}

    def _forbidden(self, path: Path) -> str | None:
        portable = _portable(path, repo=self.repo)
        absolute = str(path.resolve())
        for raw in self.forbidden_inputs:
            candidate = raw.strip()
            if not candidate:
                continue
            if "/" in candidate or " " not in candidate:
                if portable == candidate or portable.startswith(candidate.rstrip("/") + "/"):
                    return raw
                forbidden_absolute = str((self.repo / candidate).resolve())
                if absolute == forbidden_absolute or absolute.startswith(forbidden_absolute + "/"):
                    return raw
        return None

    def _verified_path(self, label: str) -> Path:
        specification = self.specifications.get(label)
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise SingleStepBenchmarkManifestError(f"input {label} specification is malformed")
        path = _path(self.repo, specification.get("path"), label=f"input {label}")
        forbidden = self._forbidden(path)
        if forbidden is not None:
            raise SingleStepBenchmarkManifestError(
                f"input {label} resolves inside forbidden source {forbidden!r}"
            )
        expected = specification.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise SingleStepBenchmarkManifestError(f"input {label} hash is not bound")
        try:
            observed = sha256_file(path)
        except OSError as exc:
            raise SingleStepBenchmarkManifestError(f"input {label} is missing: {path}") from exc
        if observed != expected:
            raise SingleStepBenchmarkManifestError(f"input {label} hash changed")
        self.accessed[label] = {
            "path": _portable(path, repo=self.repo),
            "sha256": observed,
        }
        return path

    def json(self, label: str) -> dict[str, Any]:
        path = self._verified_path(label)
        try:
            value = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise SingleStepBenchmarkManifestError(f"input {label} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise SingleStepBenchmarkManifestError(f"input {label} must be a JSON object")
        return value

    def csv_gzip(self, label: str) -> list[dict[str, str]]:
        path = self._verified_path(label)
        try:
            with gzip.open(path, "rt", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (gzip.BadGzipFile, OSError, csv.Error) as exc:
            raise SingleStepBenchmarkManifestError(f"input {label} is not valid gzip CSV") from exc
        if not rows:
            raise SingleStepBenchmarkManifestError(f"input {label} is empty")
        return rows


def _canonical_connected(smiles: Any, *, label: str) -> tuple[str, Chem.Mol]:
    if not isinstance(smiles, str) or not smiles:
        raise SingleStepBenchmarkManifestError(f"{label} has empty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise SingleStepBenchmarkManifestError(f"{label} must be one valid connected molecule")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False), molecule


def _stable_rank(seed: int, namespace: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}\t{namespace}\t{identity}".encode()).hexdigest()


def _row_identity(row: Mapping[str, Any]) -> str:
    return _content_sha256(dict(sorted(row.items())))


def _load_role_queries(registry: Mapping[str, Any]) -> dict[str, tuple[Chem.Mol, set[int]]]:
    reactions = registry.get("reactions")
    if not isinstance(reactions, list) or len(reactions) != 1:
        raise SingleStepBenchmarkManifestError("Ugi L1 registry must contain one reaction")
    role_rows = reactions[0].get("reactant_roles")
    if not isinstance(role_rows, list):
        raise SingleStepBenchmarkManifestError("Ugi L1 registry has no role definitions")
    result: dict[str, tuple[Chem.Mol, set[int]]] = {}
    for row in role_rows:
        role = row.get("name")
        smarts = row.get("required_handle_smarts")
        multiplicities = row.get("allowed_site_multiplicity")
        query = Chem.MolFromSmarts(smarts) if isinstance(smarts, str) else None
        if (
            role not in {HEAD_ROLE, ALDEHYDE_ROLE, ISOCYANIDE_ROLE}
            or query is None
            or not isinstance(multiplicities, list)
        ):
            raise SingleStepBenchmarkManifestError("Ugi L1 role definition is malformed")
        result[role] = (query, {int(value) for value in multiplicities})
    if set(result) != {HEAD_ROLE, ALDEHYDE_ROLE, ISOCYANIDE_ROLE}:
        raise SingleStepBenchmarkManifestError("Ugi L1 role definitions are incomplete")
    return result


def _qualifies_role(molecule: Chem.Mol, role: str, queries: Mapping[str, Any]) -> bool:
    query, multiplicities = queries[role]
    return len(molecule.GetSubstructMatches(query, uniquify=True)) in multiplicities


def _heterocyclic(molecule: Chem.Mol) -> bool:
    return any(
        any(molecule.GetAtomWithIdx(index).GetAtomicNum() != 6 for index in ring)
        for ring in molecule.GetRingInfo().AtomRings()
    )


def _feature_record(smiles: str, molecule: Chem.Mol) -> dict[str, Any]:
    metrics = component_chemotype_metrics(smiles)
    return {
        "carbon_branch_points": metrics["carbon_branch_atoms"],
        "noncarbonyl_carbon_double_bonds": metrics["carbon_carbon_double_bonds"],
        "noncarbonyl_carbon_triple_bonds": metrics["carbon_carbon_triple_bonds"],
        "ester_like_carbonyl_count": metrics["ester_like_carbonyl_count"],
        "ring_count": metrics["ring_count"],
        "heterocyclic": _heterocyclic(molecule),
    }


def _secondary_tags(role: str, features: Mapping[str, Any]) -> list[str]:
    tags = [f"declared_role:{role}"]
    tags.append("cyclic" if features["ring_count"] else "acyclic")
    if features["heterocyclic"]:
        tags.append("heterocyclic")
    if features["carbon_branch_points"]:
        tags.append("branched_carbon_skeleton")
    else:
        tags.append("unbranched_carbon_skeleton")
    if features["noncarbonyl_carbon_double_bonds"]:
        tags.append("carbon_carbon_double_bond")
    if features["noncarbonyl_carbon_triple_bonds"]:
        tags.append("carbon_carbon_triple_bond")
    if features["ester_like_carbonyl_count"]:
        tags.append("ester_containing")
    return sorted(tags)


def _source_route_steps(routes: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    raw_routes = routes.get("routes")
    if not isinstance(raw_routes, list):
        raise SingleStepBenchmarkManifestError("source component route artifact has no routes")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for route in raw_routes:
        route_id = route.get("route_id")
        steps = route.get("steps")
        if not isinstance(route_id, str) or not isinstance(steps, list):
            raise SingleStepBenchmarkManifestError("source component route is malformed")
        for step in steps:
            step_index = step.get("step_index")
            if not isinstance(step_index, int):
                raise SingleStepBenchmarkManifestError("source route step lacks integer index")
            key = (route_id, step_index)
            if key in result:
                raise SingleStepBenchmarkManifestError("source route step identity is duplicated")
            result[key] = {"route": route, "step": step}
    return result


def _route_truth_records(
    rows: Sequence[Mapping[str, str]],
    source_routes: Mapping[str, Any],
    verifier_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    steps = _source_route_steps(source_routes)
    bindings = verifier_config.get("qualified_transform_bindings")
    if not isinstance(bindings, dict):
        raise SingleStepBenchmarkManifestError("exact-source verifier bindings are missing")
    records: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("verification_status") != "verified_exact_product_unique":
            raise SingleStepBenchmarkManifestError(f"exact-source row {index} is not verified")
        if row.get("expected_product_in_outputs", "").lower() != "true":
            raise SingleStepBenchmarkManifestError(
                f"exact-source row {index} did not reproduce its target"
            )
        if row.get("forward_product_count") != "1":
            raise SingleStepBenchmarkManifestError(
                f"exact-source row {index} is not forward unique"
            )
        transformation = row.get("transformation")
        binding = bindings.get(transformation) if isinstance(transformation, str) else None
        if not isinstance(binding, dict) or binding.get("reaction_id") != row.get(
            "qualified_reaction_id"
        ):
            raise SingleStepBenchmarkManifestError(
                f"exact-source row {index} has no independently bound verifier"
            )
        target, target_molecule = _canonical_connected(
            row.get("expected_product"), label=f"exact-source target {index}"
        )
        if target in seen_targets:
            raise SingleStepBenchmarkManifestError(
                "exact-source targets are not constitutional uniques"
            )
        seen_targets.add(target)
        try:
            raw_reactants = json.loads(row.get("reactants_json", ""))
        except json.JSONDecodeError as exc:
            raise SingleStepBenchmarkManifestError(
                f"exact-source row {index} reactants are malformed"
            ) from exc
        if not isinstance(raw_reactants, list) or not raw_reactants:
            raise SingleStepBenchmarkManifestError(
                f"exact-source row {index} has no reactant multiset"
            )
        reactants = sorted(
            _canonical_connected(value, label=f"exact-source reactant {index}")[0]
            for value in raw_reactants
        )
        route_id = row.get("source_route_id")
        try:
            step_index = int(row.get("step_index", ""))
        except ValueError as exc:
            raise SingleStepBenchmarkManifestError(
                f"exact-source row {index} step index is malformed"
            ) from exc
        source = steps.get((route_id, step_index))
        if source is None:
            raise SingleStepBenchmarkManifestError(
                f"exact-source row {index} lacks a documented source step"
            )
        source_product, _ = _canonical_connected(
            source["step"].get("product", {}).get("canonical_smiles"),
            label=f"source product {index}",
        )
        source_reactants = sorted(
            _canonical_connected(value.get("canonical_smiles"), label=f"source reactant {index}")[0]
            for value in source["step"].get("reactants", [])
        )
        if source_product != target or source_reactants != reactants:
            raise SingleStepBenchmarkManifestError(
                f"exact-source row {index} disagrees with the documented route"
            )
        features = _feature_record(target, target_molecule)
        local_record_key = _content_sha256(
            {
                "target": target,
                "reactants": reactants,
                "transformation": transformation,
                "qualified_reaction_id": row["qualified_reaction_id"],
            }
        )
        records.append(
            {
                "canonical_smiles": target,
                "component_context_role": row.get("component_role"),
                "features": features,
                "qualified_reaction_id": row["qualified_reaction_id"],
                "reactants": reactants,
                "source_locator": source["route"].get("source", {}).get("locator"),
                "source_route_id": route_id,
                "source_step_index": step_index,
                "transformation": transformation,
                "local_record_key_sha256": local_record_key,
                "row_identity_sha256": _row_identity(row),
            }
        )
    return records


def _round_robin_select(
    records: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
    namespace: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["transformation"]].append(row)
    for family, rows in groups.items():
        rows.sort(
            key=lambda row: _stable_rank(seed, f"{namespace}:{family}", row["canonical_smiles"])
        )
    family_order = sorted(groups, key=lambda value: _stable_rank(seed, namespace, value))
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for family in family_order:
            if groups[family]:
                selected.append(groups[family].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise SingleStepBenchmarkManifestError(
                f"stratum {namespace} has only {len(selected)} honest route targets; requires {count}"
            )
    remaining = [row for rows in groups.values() for row in rows]
    return selected, remaining


def _component_candidates(
    priority_rows: Sequence[Mapping[str, str]],
    exemplar_rows: Sequence[Mapping[str, str]],
    program_rows: Sequence[Mapping[str, str]],
    queries: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    programs: dict[tuple[str, str], Mapping[str, str]] = {}
    program_unqualified = 0
    for index, row in enumerate(program_rows):
        role = row.get("role")
        smiles, molecule = _canonical_connected(
            row.get("canonical_smiles"), label=f"component program row {index}"
        )
        if role not in queries:
            raise SingleStepBenchmarkManifestError(
                f"component program row {index} has an unsupported declared role"
            )
        program_unqualified += int(not _qualifies_role(molecule, role, queries))
        programs[(role, smiles)] = row

    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    source_rows = Counter()
    role_unqualified = Counter()

    def add(
        row: Mapping[str, str],
        *,
        smiles_field: str,
        source_label: str,
        source_priority: int,
    ) -> None:
        source_rows[source_label] += 1
        role = row.get("role")
        if role not in queries:
            raise SingleStepBenchmarkManifestError(
                f"{source_label} contains unsupported role {role!r}"
            )
        smiles, molecule = _canonical_connected(
            row.get(smiles_field), label=f"{source_label} component"
        )
        if not _qualifies_role(molecule, role, queries):
            role_unqualified[source_label] += 1
            return
        key = (role, smiles)
        record = {
            "canonical_smiles": smiles,
            "role": role,
            "features": _feature_record(smiles, molecule),
            "program_ledger_match": key in programs,
            "source_artifact": source_label,
            "source_row_identity_sha256": _row_identity(row),
            "source_priority": source_priority,
        }
        existing = candidates.get(key)
        if existing is None or source_priority < existing["source_priority"]:
            candidates[key] = record

    for row in priority_rows:
        add(
            row,
            smiles_field="canonical_smiles",
            source_label="component_priority_ledger",
            source_priority=0,
        )
    for row in exemplar_rows:
        add(
            row,
            smiles_field="component_smiles",
            source_label="expanded_component_exemplars",
            source_priority=1,
        )
    records = sorted(candidates.values(), key=lambda row: (row["role"], row["canonical_smiles"]))
    return records, {
        "source_rows": dict(sorted(source_rows.items())),
        "role_unqualified_source_rows": dict(sorted(role_unqualified.items())),
        "component_program_rows": len(program_rows),
        "component_program_rows_role_unqualified": program_unqualified,
        "qualified_constitutional_candidates": len(records),
    }


def _select_component_stratum(
    candidates: Sequence[dict[str, Any]],
    *,
    stratum: str,
    count: int,
    seed: int,
    used: set[str],
    predicate: Any,
) -> list[dict[str, Any]]:
    eligible = [
        row for row in candidates if row["canonical_smiles"] not in used and bool(predicate(row))
    ]
    eligible.sort(
        key=lambda row: _stable_rank(seed, stratum, f"{row['role']}\t{row['canonical_smiles']}")
    )
    if len(eligible) < count:
        raise SingleStepBenchmarkManifestError(
            f"stratum {stratum} has only {len(eligible)} honest targets; requires {count}"
        )
    selected = eligible[:count]
    used.update(row["canonical_smiles"] for row in selected)
    return selected


def _target_id(stratum: str, role: str, canonical_smiles: str) -> str:
    digest = hashlib.sha256(f"{stratum}\t{role}\t{canonical_smiles}".encode()).hexdigest()[:20]
    return f"sst-{digest}"


def _lane_target(
    *,
    stratum: str,
    role: str,
    canonical_smiles: str,
    features: Mapping[str, Any],
    source_artifact: str,
    source_row_identity_sha256: str,
    visibility_mask_id: str | None,
    target_context_kind: str,
) -> dict[str, Any]:
    return {
        "target_id": _target_id(stratum, role, canonical_smiles),
        "canonical_smiles": canonical_smiles,
        "declared_role": role,
        "primary_stratum": stratum,
        "secondary_chemotype_tags": _secondary_tags(role, features),
        "selection_source": {
            "artifact_label": source_artifact,
            "row_identity_sha256": source_row_identity_sha256,
        },
        "target_context_kind": target_context_kind,
        "visibility_mask_id": visibility_mask_id,
    }


def _build_records(
    *,
    policy: Mapping[str, Any],
    exact_rows: Sequence[Mapping[str, str]],
    verifier_config: Mapping[str, Any],
    source_routes: Mapping[str, Any],
    priority_rows: Sequence[Mapping[str, str]],
    program_rows: Sequence[Mapping[str, str]],
    exemplar_rows: Sequence[Mapping[str, str]],
    ugi_registry: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    seed = int(policy["target_manifest_policy"]["selection_seed"])
    counts = {row["stratum_id"]: int(row["count"]) for row in policy["target_strata"]}
    queries = _load_role_queries(ugi_registry)
    route_records = _route_truth_records(exact_rows, source_routes, verifier_config)

    held, remaining = _round_robin_select(
        route_records,
        count=counts["held_reaction_families"],
        seed=seed,
        namespace="held_reaction_families",
    )
    known, _ = _round_robin_select(
        remaining,
        count=counts["known_exact_l2_routes"],
        seed=seed,
        namespace="known_exact_l2_routes",
    )

    lane_targets: list[dict[str, Any]] = []
    scoring_truth: list[dict[str, Any]] = []
    masks: list[dict[str, Any]] = []
    used: set[str] = set()

    all_family_record_keys: dict[str, list[str]] = defaultdict(list)
    all_family_reactions: dict[str, set[str]] = defaultdict(set)
    for row in route_records:
        all_family_record_keys[row["transformation"]].append(row["local_record_key_sha256"])
        all_family_reactions[row["transformation"]].add(row["qualified_reaction_id"])

    def add_route(row: dict[str, Any], stratum: str) -> None:
        canonical = row["canonical_smiles"]
        if canonical in used:
            raise SingleStepBenchmarkManifestError("route strata are not constitutionally disjoint")
        used.add(canonical)
        role = row["component_context_role"]
        target_id = _target_id(stratum, role, canonical)
        mask_id = f"mask-{target_id}"
        lane_targets.append(
            _lane_target(
                stratum=stratum,
                role=role,
                canonical_smiles=canonical,
                features=row["features"],
                source_artifact="exact_source_step_verification",
                source_row_identity_sha256=row["row_identity_sha256"],
                visibility_mask_id=mask_id,
                target_context_kind="authenticated_l2_step_product",
            )
        )
        scoring_truth.append(
            {
                "target_id": target_id,
                "truth_kind": "documented_exact_forward_unique_reactant_multiset",
                "canonical_reactant_multiset": row["reactants"],
                "transformation": row["transformation"],
                "qualified_reaction_id": row["qualified_reaction_id"],
                "source_route_id": row["source_route_id"],
                "source_step_index": row["source_step_index"],
                "source_locator": row["source_locator"],
                "local_record_key_sha256": row["local_record_key_sha256"],
            }
        )
        if stratum == "held_reaction_families":
            hidden_families = [row["transformation"]]
            hidden_reactions = sorted(all_family_reactions[row["transformation"]])
            hidden_records = sorted(all_family_record_keys[row["transformation"]])
            mask_scope = "all_local_records_and_templates_in_target_transformation_family"
        else:
            hidden_families = []
            hidden_reactions = []
            hidden_records = [row["local_record_key_sha256"]]
            mask_scope = "exact_scoring_truth_record_only"
        masks.append(
            {
                "mask_id": mask_id,
                "target_id": target_id,
                "mask_scope": mask_scope,
                "hidden_local_transformation_families": hidden_families,
                "hidden_local_qualified_reaction_ids": hidden_reactions,
                "hidden_local_record_key_sha256s": hidden_records,
                "learned_checkpoint_training_is_not_claimed_family_disjoint": True,
                "exact_reactants_exposed_to_lane": False,
            }
        )

    for row in known:
        add_route(row, "known_exact_l2_routes")
    for row in held:
        add_route(row, "held_reaction_families")

    components, component_source_audit = _component_candidates(
        priority_rows, exemplar_rows, program_rows, queries
    )
    selections: list[tuple[str, list[dict[str, Any]]]] = []
    selections.append(
        (
            "ester_containing_aldehydes",
            _select_component_stratum(
                components,
                stratum="ester_containing_aldehydes",
                count=counts["ester_containing_aldehydes"],
                seed=seed,
                used=used,
                predicate=lambda row: row["role"] == ALDEHYDE_ROLE
                and row["features"]["ester_like_carbonyl_count"] > 0,
            ),
        )
    )
    selections.append(
        (
            "branched_aldehydes",
            _select_component_stratum(
                components,
                stratum="branched_aldehydes",
                count=counts["branched_aldehydes"],
                seed=seed,
                used=used,
                predicate=lambda row: row["role"] == ALDEHYDE_ROLE
                and row["features"]["carbon_branch_points"] > 0,
            ),
        )
    )
    selections.append(
        (
            "unsaturated_aldehydes",
            _select_component_stratum(
                components,
                stratum="unsaturated_aldehydes",
                count=counts["unsaturated_aldehydes"],
                seed=seed,
                used=used,
                predicate=lambda row: row["role"] == ALDEHYDE_ROLE
                and (
                    row["features"]["noncarbonyl_carbon_double_bonds"]
                    + row["features"]["noncarbonyl_carbon_triple_bonds"]
                    > 0
                ),
            ),
        )
    )
    selections.append(
        (
            "linear_aldehydes",
            _select_component_stratum(
                components,
                stratum="linear_aldehydes",
                count=counts["linear_aldehydes"],
                seed=seed,
                used=used,
                predicate=lambda row: row["role"] == ALDEHYDE_ROLE
                and row["features"]["carbon_branch_points"] == 0
                and row["features"]["noncarbonyl_carbon_double_bonds"] == 0
                and row["features"]["noncarbonyl_carbon_triple_bonds"] == 0
                and row["features"]["ring_count"] == 0,
            ),
        )
    )
    selections.append(
        (
            "isocyanide_formamide_precursors",
            _select_component_stratum(
                components,
                stratum="isocyanide_formamide_precursors",
                count=counts["isocyanide_formamide_precursors"],
                seed=seed,
                used=used,
                predicate=lambda row: row["role"] == ISOCYANIDE_ROLE,
            ),
        )
    )
    selections.append(
        (
            "heterocyclic_amine_heads",
            _select_component_stratum(
                components,
                stratum="heterocyclic_amine_heads",
                count=counts["heterocyclic_amine_heads"],
                seed=seed,
                used=used,
                predicate=lambda row: row["role"] == HEAD_ROLE and row["features"]["heterocyclic"],
            ),
        )
    )
    for stratum, rows in selections:
        for row in rows:
            lane_targets.append(
                _lane_target(
                    stratum=stratum,
                    role=row["role"],
                    canonical_smiles=row["canonical_smiles"],
                    features=row["features"],
                    source_artifact=row["source_artifact"],
                    source_row_identity_sha256=row["source_row_identity_sha256"],
                    visibility_mask_id=None,
                    target_context_kind="ugi_component_root",
                )
            )

    role_swaps = (
        (ALDEHYDE_ROLE, ISOCYANIDE_ROLE),
        (ISOCYANIDE_ROLE, HEAD_ROLE),
        (HEAD_ROLE, ALDEHYDE_ROLE),
    )
    adversarial_count = counts["adversarial_incompatibles"]
    if adversarial_count % len(role_swaps):
        raise SingleStepBenchmarkManifestError("adversarial count must balance role-swap controls")
    per_swap = adversarial_count // len(role_swaps)
    for source_role, assigned_role in role_swaps:
        adversarial = _select_component_stratum(
            components,
            stratum=f"adversarial:{source_role}:as:{assigned_role}",
            count=per_swap,
            seed=seed,
            used=used,
            predicate=lambda row, source_role=source_role, assigned_role=assigned_role: (
                row["role"] == source_role
                and _qualifies_role(
                    _canonical_connected(row["canonical_smiles"], label="adversarial target")[1],
                    source_role,
                    queries,
                )
                and not _qualifies_role(
                    _canonical_connected(row["canonical_smiles"], label="adversarial target")[1],
                    assigned_role,
                    queries,
                )
            ),
        )
        for row in adversarial:
            target_id = _target_id(
                "adversarial_incompatibles", assigned_role, row["canonical_smiles"]
            )
            lane_targets.append(
                _lane_target(
                    stratum="adversarial_incompatibles",
                    role=assigned_role,
                    canonical_smiles=row["canonical_smiles"],
                    features=row["features"],
                    source_artifact=row["source_artifact"],
                    source_row_identity_sha256=row["source_row_identity_sha256"],
                    visibility_mask_id=None,
                    target_context_kind="valid_connected_wrong_role_control",
                )
            )
            scoring_truth.append(
                {
                    "target_id": target_id,
                    "truth_kind": "valid_connected_wrong_handle_role_swap_control",
                    "source_qualified_role": source_role,
                    "adversarial_assigned_role": assigned_role,
                    "expected_disposition": "reject_role_or_handle_incompatible",
                }
            )

    lane_targets.sort(key=lambda row: row["target_id"])
    scoring_truth.sort(key=lambda row: row["target_id"])
    masks.sort(key=lambda row: row["target_id"])
    observed = Counter(row["primary_stratum"] for row in lane_targets)
    if observed != Counter(counts):
        raise SingleStepBenchmarkManifestError(
            f"materialized stratum counts differ from policy: {dict(observed)}"
        )
    identities = [row["canonical_smiles"] for row in lane_targets]
    if len(identities) != len(set(identities)):
        raise SingleStepBenchmarkManifestError("lane targets are not constitutionally deduplicated")
    target_ids = [row["target_id"] for row in lane_targets]
    if len(target_ids) != len(set(target_ids)):
        raise SingleStepBenchmarkManifestError("target IDs are not unique")
    route_truth_targets = {
        row["target_id"]
        for row in scoring_truth
        if row["truth_kind"] == "documented_exact_forward_unique_reactant_multiset"
    }
    masked_targets = {row["target_id"] for row in masks}
    if route_truth_targets != masked_targets:
        raise SingleStepBenchmarkManifestError("known route truth and visibility masks differ")
    return lane_targets, scoring_truth, masks, component_source_audit


def _load_config(repo: Path, config_path: Path) -> tuple[dict[str, Any], str]:
    path = config_path if config_path.is_absolute() else repo / config_path
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise SingleStepBenchmarkManifestError(f"invalid builder config: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SingleStepBenchmarkManifestError("builder config schema changed")
    return value, sha256_file(path)


def configured_output_directory(repo: Path, config_path: Path) -> Path:
    config, _ = _load_config(repo, config_path)
    return _path(repo, config.get("output_directory"), label="output directory")


def build_manifest_payloads(repo: Path, config_path: Path) -> dict[str, bytes]:
    """Return deterministic output bytes without executing any proposal lane."""

    repo = repo.resolve()
    config, config_sha256 = _load_config(repo, config_path)
    if config.get("status") != "authorized_manifest_materialization_benchmark_not_executed":
        raise SingleStepBenchmarkManifestError(
            "builder config is not authorized for materialization"
        )
    if config.get("graph2edits_execution_authorized") is not False:
        raise SingleStepBenchmarkManifestError(
            "builder config must keep Graph2Edits execution disabled"
        )
    inputs = config.get("inputs")
    implementation = config.get("implementation")
    if not isinstance(inputs, dict) or set(inputs) != EXPECTED_INPUTS:
        raise SingleStepBenchmarkManifestError("builder input contract changed")
    if not isinstance(implementation, dict) or set(implementation) != EXPECTED_IMPLEMENTATION:
        raise SingleStepBenchmarkManifestError("builder implementation contract changed")

    policy_spec = inputs["benchmark_policy"]
    if not isinstance(policy_spec, dict) or not isinstance(policy_spec.get("path"), str):
        raise SingleStepBenchmarkManifestError("benchmark policy input is malformed")
    policy_path = _path(repo, policy_spec["path"], label="benchmark policy")
    try:
        policy_preview = json.loads(policy_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleStepBenchmarkManifestError("benchmark policy could not be previewed") from exc
    forbidden = policy_preview.get("forbidden_inputs")
    if not isinstance(forbidden, list) or not forbidden:
        raise SingleStepBenchmarkManifestError("benchmark policy has no forbidden-input contract")

    reader = _PinnedReader(repo, inputs, forbidden)
    policy = reader.json("benchmark_policy")
    if policy.get("status") != "frozen_specification_not_executed":
        raise SingleStepBenchmarkManifestError("benchmark policy is not frozen and inactive")
    activation = policy.get("activation", {})
    if (
        activation.get("benchmark_execution_authorized") is not False
        or activation.get("graph2edits_backend_active") is not False
    ):
        raise SingleStepBenchmarkManifestError("benchmark policy unexpectedly authorizes execution")
    verifier = policy.get("forward_verifier_resolution", {})
    if (
        verifier.get("proposal_output_may_verify_itself") is not False
        or verifier.get("ugi_l1_registry_may_verify_l2_proposals") is not False
    ):
        raise SingleStepBenchmarkManifestError("benchmark L2 verifier is not independent")

    exact_rows = reader.csv_gzip("exact_source_step_verification")
    verifier_config = reader.json("exact_source_verification_config")
    source_routes = reader.json("source_component_routes")
    priority_rows = reader.csv_gzip("component_priority_ledger")
    program_rows = reader.csv_gzip("component_program_ledger")
    exemplar_rows = reader.csv_gzip("expanded_component_exemplars")
    ugi_registry = reader.json("ugi_l1_registry")

    lane_targets, scoring_truth, masks, component_source_audit = _build_records(
        policy=policy,
        exact_rows=exact_rows,
        verifier_config=verifier_config,
        source_routes=source_routes,
        priority_rows=priority_rows,
        program_rows=program_rows,
        exemplar_rows=exemplar_rows,
        ugi_registry=ugi_registry,
    )

    implementation_hashes: dict[str, dict[str, str]] = {}
    for label in sorted(implementation):
        specification = implementation[label]
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise SingleStepBenchmarkManifestError(f"implementation {label} is malformed")
        path = _path(repo, specification["path"], label=f"implementation {label}")
        observed = sha256_file(path)
        if observed != specification["sha256"]:
            raise SingleStepBenchmarkManifestError(f"implementation {label} hash changed")
        implementation_hashes[label] = {
            "path": _portable(path, repo=repo),
            "sha256": observed,
        }

    policy_sha256 = reader.accessed["benchmark_policy"]["sha256"]
    common = {
        "status": FROZEN_STATUS,
        "benchmark_policy_sha256": policy_sha256,
        "selection_seed": policy["target_manifest_policy"]["selection_seed"],
    }
    lane_payload = {
        "schema_version": LANE_MANIFEST_SCHEMA_VERSION,
        **common,
        "truth_fields_present": False,
        "known_routes_exposed_to_lanes": False,
        "targets": lane_targets,
    }
    truth_payload = {
        "schema_version": SCORING_TRUTH_SCHEMA_VERSION,
        **common,
        "lane_input": False,
        "records": scoring_truth,
    }
    mask_payload = {
        "schema_version": VISIBILITY_MASK_SCHEMA_VERSION,
        **common,
        "purpose": "orchestrator_only_local_source_visibility_masks",
        "graph2edits_pretraining_family_disjointness_claimed": False,
        "masks": masks,
    }
    payloads = {
        "lane_targets": _gzip_json_bytes(lane_payload),
        "scoring_truth": _gzip_json_bytes(truth_payload),
        "visibility_masks": _gzip_json_bytes(mask_payload),
    }
    accessed = {label: reader.accessed[label] for label in sorted(reader.accessed)}
    accessed_paths = {record["path"] for record in accessed.values()}
    forbidden_accessed = [
        value
        for value in forbidden
        if any(
            path == value or path.startswith(str(value).rstrip("/") + "/")
            for path in accessed_paths
        )
    ]
    if forbidden_accessed:
        raise SingleStepBenchmarkManifestError("forbidden input appeared in access ledger")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        **common,
        "config_sha256": config_sha256,
        "implementation": implementation_hashes,
        "inputs_accessed": accessed,
        "input_access_log_sha256": _content_sha256(accessed),
        "policy_bound_artifacts": policy.get("artifacts"),
        "outputs": {
            label: {"filename": OUTPUT_FILENAMES[label], "sha256": _sha256_bytes(payload)}
            for label, payload in sorted(payloads.items())
        },
        "summary": {
            "target_count": len(lane_targets),
            "primary_strata": dict(
                sorted(Counter(row["primary_stratum"] for row in lane_targets).items())
            ),
            "constitutional_unique_targets": len({row["canonical_smiles"] for row in lane_targets}),
            "known_route_truth_records": sum(
                row["truth_kind"] == "documented_exact_forward_unique_reactant_multiset"
                for row in scoring_truth
            ),
            "adversarial_valid_connected_controls": sum(
                row["truth_kind"] == "valid_connected_wrong_handle_role_swap_control"
                for row in scoring_truth
            ),
            "held_family_masks": sum(
                row["mask_scope"]
                == "all_local_records_and_templates_in_target_transformation_family"
                for row in masks
            ),
            "exact_truth_masks": sum(
                row["mask_scope"] == "exact_scoring_truth_record_only" for row in masks
            ),
            "component_source_audit": component_source_audit,
        },
        "safety_receipt": {
            "development_sources_only": True,
            "known_route_truth_absent_from_lane_payload": all(
                "reactant" not in json.dumps(row).lower() for row in lane_targets
            ),
            "held_family_local_mask_metadata_materialized": True,
            "adversarial_targets_are_valid_connected_molecules": True,
            "sealed_holdout_accessed": False,
            "forbidden_inputs_accessed": forbidden_accessed,
            "graph2edits_executed": False,
            "proposal_lane_executed": False,
            "graph2edits_backend_active": False,
            "hybrid_production_lane_active": False,
            "strict_lane_remains_production_default": True,
            "candidate_selected": False,
            "route_evidence_created": False,
            "v_syn_modified": False,
        },
    }
    payloads["result"] = _stable_json_bytes(result, pretty=True)
    return payloads

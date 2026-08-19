"""One identical, arm-blind, bounded hybrid L1/L2/L3 route cascade.

The cascade is downstream of molecular generation and of biological
applicability/ranking, and upstream of any prospective panel lock.  Every
candidate in the route-blinded v2 shortlist receives the same frozen exact
evidence assessment, the same bounded Graph2Edits proposal budget, the same
residual AiZynthFinder budget and the same independent adjudication.

Generation arm, cohort, authority tier and predicted potency never reach the
route search or the planner cache key.  They are joined back only for
reporting.  This module never treats a learned proposal or a public-stock
planner solution as route evidence, and never calls an unresolved route
unsynthesizable: a route is unresolved *under the frozen bounded cascade,
evidence and 3 August procurement snapshot*.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.bio.ugi_semantic_annotations import ROLE_NAMES

CONFIG_SCHEMA_VERSION = "phase1_ugi_bounded_hybrid_route_cascade_config.v1"
PREFLIGHT_SCHEMA_VERSION = "phase1_ugi_bounded_hybrid_route_cascade_preflight.v1"
EXACT_EVIDENCE_SCHEMA_VERSION = "phase1_ugi_bounded_hybrid_route_cascade_exact_evidence.v1"
CANDIDATE_EXACT_LEDGER_SCHEMA_VERSION = (
    "phase1_ugi_bounded_hybrid_route_cascade_candidate_exact_ledger.v1"
)
COMPONENT_EXACT_LEDGER_SCHEMA_VERSION = (
    "phase1_ugi_bounded_hybrid_route_cascade_component_exact_ledger.v1"
)
COMPONENT_LEDGER_SCHEMA_VERSION = "phase1_ugi_bounded_hybrid_route_cascade_component_ledger.v1"
CANDIDATE_LEDGER_SCHEMA_VERSION = "phase1_ugi_bounded_hybrid_route_cascade_candidate_ledger.v1"
LEAF_LEDGER_SCHEMA_VERSION = "phase1_ugi_bounded_hybrid_route_cascade_leaf_ledger.v1"
ADJUDICATION_SCHEMA_VERSION = "phase1_ugi_bounded_hybrid_route_cascade_adjudication.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_bounded_hybrid_route_cascade.v1"

CACHE_CLONE_ID = "bounded-hybrid-route-cascade-v1"

COMPLETE = "complete"
UNRESOLVED = "unresolved"
REJECTED = "rejected"
SEARCH_CENSORED = "search_censored"
COMPONENT_STATES = (COMPLETE, UNRESOLVED, REJECTED, SEARCH_CENSORED)

NOT_ASSESSED_OUTCOME = "not_assessed_no_admitted_parent_product"
_CENSORED_OUTCOMES = frozenset(
    {"budget_exhausted", "invalid_input", "execution_error", NOT_ASSESSED_OUTCOME}
)
_REJECTING_OUTCOMES = frozenset({"incompatible"})

EXPECTED_SCOPE = {
    "post_generation_route_assessment": True,
    "assessment_before_candidate_lock": True,
    "arm_blind_single_lane": True,
    "guided_post_hoc_cache_split_inherited": False,
    "identical_budget_all_arms_and_cohorts": True,
    "proposal_engines_are_proposal_only": True,
    "public_stock_solution_is_evidence": False,
    "biological_model_calls": 0,
    "potency_tilting": False,
    "synthesis_tilting": False,
    "generator_retraining": False,
    "applicability_retuning": False,
    "candidate_selection": False,
    "prospective_panel_lock": False,
    "preregistration_created": False,
    "sealed_holdout_access": False,
}

EXPECTED_INPUTS = frozenset(
    {
        "shortlist_result",
        "shortlist_ledger",
        "main_generation_result",
        "main_generation_ledger",
        "branch_generation_result",
        "branch_generation_ledger",
        "production_refit_result",
        "joint_checkpoint",
        "closure_checkpoint",
        "atom_vocabulary",
        "qualified_reaction_registry",
        "current_route_source",
        "route_completion_utility",
        "terminal_rescoring_ledger",
        "terminal_rescoring_result",
        "graded_evidence_ledger",
        "graph2edits_runtime_qualification",
        "graph2edits_training_corpus_manifest",
        "graph2edits_forward_resolver",
        "aizynthfinder_runtime_manifest",
    }
)

MAIN_ARMS = ("broad_prior", "support_enriched")
BRANCH_ARM = "branch_exploration"
ARM_ORDER = ("broad_prior", "support_enriched", BRANCH_ARM)


class UgiBoundedHybridRouteCascadeError(RuntimeError):
    """Raised when the frozen bounded hybrid route cascade cannot proceed exactly."""


# ---------------------------------------------------------------------------
# serialization helpers
# ---------------------------------------------------------------------------


def stable_json(value: Any) -> str:
    """Return canonical JSON with sorted keys and no incidental whitespace."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_payload(value: Any) -> str:
    """Return a deterministic SHA-256 over canonical JSON."""

    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 of one file without loading it entirely."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Return newline-terminated canonical JSON bytes."""

    return (stable_json(value) + "\n").encode()


def jsonl_gzip_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Return deterministic gzip JSONL bytes for a row sequence."""

    raw = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as handle:
        handle.write(raw)
    return output.getvalue()


def csv_gzip_bytes(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Return deterministic gzip CSV bytes for a row sequence."""

    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.getvalue().encode())
    return output.getvalue()


def atomic_write(path: Path, payload: bytes) -> None:
    """Write complete bytes or leave the destination untouched."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    """Load one JSON object with an actionable failure message."""

    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiBoundedHybridRouteCascadeError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiBoundedHybridRouteCascadeError(f"{label} must contain one JSON object")
    return value


def read_jsonl_gzip(path: Path, *, label: str) -> list[dict[str, Any]]:
    """Read gzip JSONL into a list of objects."""

    try:
        with gzip.open(path, "rt") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiBoundedHybridRouteCascadeError(f"{label} is not valid gzip JSONL") from error
    if any(not isinstance(row, dict) for row in rows):
        raise UgiBoundedHybridRouteCascadeError(f"{label} contains a non-object row")
    return rows


def read_csv_gzip(path: Path) -> list[dict[str, str]]:
    """Read gzip CSV into a list of string dictionaries."""

    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def component_sha256(role: str, canonical_smiles: str) -> str:
    """Return the content address of one canonical Ugi component target."""

    return sha256_payload(
        {
            "schema": "forge.ugi_bounded_hybrid_route_component_identity.v1",
            "role": role,
            "canonical_smiles": canonical_smiles,
        }
    )


def component_target_id(role: str, canonical_smiles: str) -> str:
    """Return a short, arm-independent identifier for one component target."""

    return f"component:{component_sha256(role, canonical_smiles)[:20]}"


# ---------------------------------------------------------------------------
# frozen contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CascadeContract:
    """Validated configuration plus resolved, hash-pinned input paths."""

    repo: Path
    config_path: Path
    config: Mapping[str, Any]
    paths: Mapping[str, Path]

    @property
    def output_dir(self) -> Path:
        return (self.repo / str(self.config["output_directory"])).resolve()

    @property
    def assessment_as_of_utc(self) -> str:
        return str(self.config["assessment_as_of_utc"])

    @property
    def cumulative_source_inputs_sha256(self) -> str:
        return str(self.config["cumulative_source_inputs_sha256"])

    @property
    def config_sha256(self) -> str:
        return sha256_file(self.config_path)

    def stage(self, name: str) -> Mapping[str, Any]:
        stages = self.config["stages"]
        if name not in stages:
            raise UgiBoundedHybridRouteCascadeError(f"missing frozen stage settings: {name}")
        return stages[name]


def _pin(repo: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise UgiBoundedHybridRouteCascadeError(f"malformed input pin: {label}")
    path = (repo / str(value["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiBoundedHybridRouteCascadeError(f"input pin escapes repository: {label}") from error
    if path.is_symlink() or not path.is_file():
        raise UgiBoundedHybridRouteCascadeError(f"input pin is not a regular file: {label}")
    if sha256_file(path) != value["sha256"]:
        raise UgiBoundedHybridRouteCascadeError(f"input pin changed: {label}")
    return path


def load_cascade_contract(repo: Path, config_path: Path) -> CascadeContract:
    """Validate the frozen config, its scope guards and every input hash."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = load_json(config_path, label="bounded hybrid route cascade config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiBoundedHybridRouteCascadeError("unsupported bounded hybrid cascade config")
    if config.get("status") != "frozen_before_bounded_hybrid_route_cascade":
        raise UgiBoundedHybridRouteCascadeError("bounded hybrid cascade config is not frozen")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiBoundedHybridRouteCascadeError("bounded hybrid cascade scope changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != EXPECTED_INPUTS:
        raise UgiBoundedHybridRouteCascadeError("bounded hybrid cascade input set changed")
    snapshot = config.get("procurement_snapshot")
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("is_live_availability") is not False
        or snapshot.get("refresh_required_before_panel_lock") is not True
    ):
        raise UgiBoundedHybridRouteCascadeError("procurement snapshot declaration changed")
    paths = {label: _pin(repo, record, label=label) for label, record in inputs.items()}

    source = load_json(paths["current_route_source"], label="current route source")
    current = source.get("current_cumulative_source")
    if (
        source.get("status") != "current_source_l3_and_zero_guidance_route_values_requalified"
        or not isinstance(current, Mapping)
        or current.get("assessment_at_utc") != config.get("assessment_as_of_utc")
        or current.get("inputs_sha256") != config.get("cumulative_source_inputs_sha256")
        or current.get("all_l3_windows_cover_decision_horizon") is not True
    ):
        raise UgiBoundedHybridRouteCascadeError("route source is not current at the frozen time")
    horizon = current.get("decision_horizon")
    declared = snapshot.get("declared_comparison_horizon")
    if not isinstance(horizon, Mapping) or not isinstance(declared, Mapping):
        raise UgiBoundedHybridRouteCascadeError("decision horizon declaration is malformed")
    if (
        horizon.get("start_utc") != declared.get("start_utc")
        or horizon.get("end_utc") != declared.get("end_utc")
        or horizon.get("interval_semantics") != declared.get("interval_semantics")
    ):
        raise UgiBoundedHybridRouteCascadeError(
            "config comparison horizon differs from the frozen procurement snapshot horizon"
        )
    return CascadeContract(repo=repo, config_path=config_path, config=config, paths=paths)


# ---------------------------------------------------------------------------
# candidate population
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateRecord:
    """One shortlist product joined to its native generation record."""

    arm_id: str
    draw_index: int
    canonical_product: str
    components: Mapping[str, str]
    shortlist_row: Mapping[str, Any]
    generation_row: Mapping[str, Any]

    @property
    def unit_id(self) -> str:
        return f"bounded-hybrid-route:{self.arm_id}:{self.draw_index:05d}"

    @property
    def native_terminal(self) -> Mapping[str, Any]:
        native = self.generation_row.get("native_terminal")
        if not isinstance(native, Mapping):
            raise UgiBoundedHybridRouteCascadeError(
                f"generation row has no native terminal: {self.unit_id}"
            )
        return native


def build_candidate_population(contract: CascadeContract) -> list[CandidateRecord]:
    """Join every shortlist row to its exact native generation record."""

    shortlist = read_jsonl_gzip(contract.paths["shortlist_ledger"], label="shortlist ledger")
    main_rows = read_jsonl_gzip(
        contract.paths["main_generation_ledger"], label="main generation ledger"
    )
    branch_rows = read_jsonl_gzip(
        contract.paths["branch_generation_ledger"], label="branch generation ledger"
    )
    main_by_key = {(str(row["arm_id"]), int(row["draw_index"])): row for row in main_rows}
    branch_by_key = {int(row["draw_index"]): row for row in branch_rows}
    if len(main_by_key) != len(main_rows) or len(branch_by_key) != len(branch_rows):
        raise UgiBoundedHybridRouteCascadeError("generation draw keys are not unique")

    records: list[CandidateRecord] = []
    for row in shortlist:
        arm = str(row["arm_id"])
        draw_index = int(row["draw_index"])
        if arm in MAIN_ARMS:
            generation = main_by_key.get((arm, draw_index))
        elif arm == BRANCH_ARM:
            generation = branch_by_key.get(draw_index)
        else:
            raise UgiBoundedHybridRouteCascadeError(f"unsupported shortlist arm: {arm}")
        if generation is None:
            raise UgiBoundedHybridRouteCascadeError(
                f"shortlist row has no native generation record: {arm}:{draw_index}"
            )
        native = generation.get("native_terminal")
        if not isinstance(native, Mapping):
            raise UgiBoundedHybridRouteCascadeError("generation row has no native terminal")
        components = native.get("component_smiles_by_role")
        if not isinstance(components, Mapping) or set(components) != set(ROLE_NAMES):
            raise UgiBoundedHybridRouteCascadeError(
                "native terminal does not carry exactly the three frozen Ugi roles"
            )
        if native.get("smiles") != row.get("canonical_product"):
            raise UgiBoundedHybridRouteCascadeError(
                "shortlist product differs from its native generated terminal"
            )
        for role in ROLE_NAMES:
            if components[role] != row["components"][role]:
                raise UgiBoundedHybridRouteCascadeError(
                    f"shortlist and native {role} components disagree"
                )
        if native.get("l1_forward_verification", {}).get("exact_product_reconstructed") is not True:
            raise UgiBoundedHybridRouteCascadeError(
                "shortlist row does not carry exact constitutional L1 reconstruction"
            )
        records.append(
            CandidateRecord(
                arm_id=arm,
                draw_index=draw_index,
                canonical_product=str(row["canonical_product"]),
                components={role: str(components[role]) for role in ROLE_NAMES},
                shortlist_row=row,
                generation_row=generation,
            )
        )
    records.sort(key=lambda record: (ARM_ORDER.index(record.arm_id), record.draw_index))
    return records


def unique_component_targets(
    records: Sequence[CandidateRecord],
) -> list[tuple[str, str]]:
    """Return the deduplicated (role, canonical component) targets, arm-blind."""

    seen: set[tuple[str, str]] = set()
    for record in records:
        for role in ROLE_NAMES:
            seen.add((role, record.components[role]))
    return sorted(seen)


def assert_expected_population(
    contract: CascadeContract, records: Sequence[CandidateRecord]
) -> dict[str, Any]:
    """Fail closed unless the frozen population assertions hold exactly."""

    expected = contract.config.get("expected_population")
    if not isinstance(expected, Mapping):
        raise UgiBoundedHybridRouteCascadeError("expected population declaration is missing")
    observed_arms = dict(sorted(Counter(record.arm_id for record in records).items()))
    observed_cohorts = dict(
        sorted(Counter(str(record.shortlist_row["cohort"]) for record in records).items())
    )
    targets = unique_component_targets(records)
    observed_components = dict(sorted(Counter(role for role, _ in targets).items()))
    unique_products = {record.canonical_product for record in records}
    observed = {
        "rows": len(records),
        "unique_products": len(unique_products),
        "arms": observed_arms,
        "cohorts": observed_cohorts,
        "unique_components": observed_components,
        "unique_components_total": len(targets),
    }
    declared = {
        "rows": int(expected["rows"]),
        "unique_products": int(expected["unique_products"]),
        "arms": dict(sorted(expected["arms"].items())),
        "cohorts": dict(sorted(expected["cohorts"].items())),
        "unique_components": dict(sorted(expected["unique_components"].items())),
        "unique_components_total": int(expected["unique_components_total"]),
    }
    if observed != declared:
        raise UgiBoundedHybridRouteCascadeError(
            f"population changed: expected {declared}, observed {observed}"
        )
    return observed


def blindness_receipt(
    contract: CascadeContract, records: Sequence[CandidateRecord]
) -> dict[str, Any]:
    """Return a receipt proving reporting-only fields never enter route search.

    The receipt digest is computed from the arm-blind route inputs alone.  If a
    reporting-only field ever leaked into the route population, the digest would
    change while the declared field list did not.
    """

    reporting = contract.config.get("reporting_only_fields")
    if not isinstance(reporting, list) or not reporting:
        raise UgiBoundedHybridRouteCascadeError("reporting-only field list is missing")
    route_inputs = [
        {
            "canonical_product": record.canonical_product,
            "components": {role: record.components[role] for role in ROLE_NAMES},
        }
        for record in sorted(records, key=lambda item: item.canonical_product)
    ]
    targets = unique_component_targets(records)
    return {
        "schema": "forge.ugi_bounded_hybrid_route_blindness_receipt.v1",
        "reporting_only_fields": sorted(str(name) for name in reporting),
        "route_input_fields": ["canonical_product", "components"],
        "route_population_sha256": sha256_payload(route_inputs),
        "component_target_sha256": sha256_payload([[role, smiles] for role, smiles in targets]),
        "arm_influences_route_search": False,
        "cohort_influences_route_search": False,
        "authority_tier_influences_route_search": False,
        "potency_influences_route_search": False,
    }


# ---------------------------------------------------------------------------
# reporting-only potency resolution
# ---------------------------------------------------------------------------


def resolve_potency_reporting(
    record: CandidateRecord,
    rescoring_index: Mapping[tuple[str, int], Mapping[str, str]],
) -> dict[str, Any]:
    """Return the authoritative conservative-high flag with its evidence source.

    The v2 shortlist merges two cohorts written under different ledger schemas:
    the preserved v1 linear-potency rows omit ``conservative_high_potency`` while
    the branch rows carry it explicitly.  Coercing the missing field to ``False``
    would report zero conservative-high candidates in precisely the cohort that
    was selected for being conservative-high.  This resolves the flag from the
    frozen rescoring ledger instead, and fails closed if neither source has it.
    """

    short = record.shortlist_row
    if "conservative_high_potency" in short:
        return {
            "conservative_high_potency": bool(short["conservative_high_potency"]),
            "conservative_high_source": "shortlist_ledger",
            "oracle_scored": bool(short.get("oracle_scored", False)),
        }
    row = rescoring_index.get((record.arm_id, record.draw_index))
    if row is None:
        raise UgiBoundedHybridRouteCascadeError(
            f"no authoritative potency record for {record.arm_id}:{record.draw_index}"
        )
    if row.get("canonical_product") != record.canonical_product:
        raise UgiBoundedHybridRouteCascadeError(
            "rescoring ledger product identity differs from the shortlist product"
        )
    flag = row.get("conservative_high_potency")
    scored = row.get("oracle_scored")
    if flag not in {"True", "False"} or scored not in {"True", "False"}:
        raise UgiBoundedHybridRouteCascadeError(
            "rescoring ledger potency flags are not serialized booleans"
        )
    return {
        "conservative_high_potency": flag == "True",
        "conservative_high_source": "terminal_rescoring_v3",
        "oracle_scored": scored == "True",
    }


def rescoring_index(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, int], Mapping[str, str]]:
    """Index the frozen rescoring ledger by its arm and draw identity."""

    index: dict[tuple[str, int], Mapping[str, str]] = {}
    for row in rows:
        index[(str(row["arm_id"]), int(row["draw_index"]))] = row
    if len(index) != len(rows):
        raise UgiBoundedHybridRouteCascadeError("rescoring arm/draw keys are not unique")
    return index


# ---------------------------------------------------------------------------
# component state vocabulary
# ---------------------------------------------------------------------------


def derive_final_component_state(
    *,
    strict_complete: bool,
    assessment_outcome: str,
    proposal_search_censored: bool,
) -> tuple[str, str]:
    """Map independent evidence to the frozen four-state vocabulary.

    Only the independent exact assessment can return ``complete``.  A learned
    proposal or a public-stock planner solution can never set this state.
    """

    if strict_complete:
        if assessment_outcome != "complete":
            raise UgiBoundedHybridRouteCascadeError(
                "strict closure disagrees with its assessment outcome"
            )
        return COMPLETE, "independent_exact_recursive_closure_to_current_terminal_materials"
    if assessment_outcome in _CENSORED_OUTCOMES:
        return SEARCH_CENSORED, f"exact_assessment_censored:{assessment_outcome}"
    if assessment_outcome in _REJECTING_OUTCOMES:
        return REJECTED, "independently_evidenced_chemical_incompatibility"
    if proposal_search_censored:
        return SEARCH_CENSORED, "every_proposal_engine_attempt_failed_to_execute"
    return UNRESOLVED, f"unresolved_under_frozen_bounded_cascade:{assessment_outcome}"


def product_route_state(component_states: Sequence[str]) -> str:
    """Require all three Ugi roles to close before a product is complete."""

    if len(component_states) != len(ROLE_NAMES) or any(
        state not in COMPONENT_STATES for state in component_states
    ):
        raise UgiBoundedHybridRouteCascadeError(
            "product roll-up requires three typed component states"
        )
    if all(state == COMPLETE for state in component_states):
        return COMPLETE
    if any(state == REJECTED for state in component_states):
        return REJECTED
    if any(state == SEARCH_CENSORED for state in component_states):
        return SEARCH_CENSORED
    return UNRESOLVED


# ---------------------------------------------------------------------------
# chemical descriptors for the balance audit
# ---------------------------------------------------------------------------


def component_descriptors(smiles: str) -> dict[str, Any]:
    """Return bounded chemical descriptors used only for the balance audit."""

    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise UgiBoundedHybridRouteCascadeError(f"descriptor target is invalid SMILES: {smiles}")
    # An ester oxygen also satisfies the generic dialkyl-ether pattern, so the
    # ether count subtracts esters.  Ester and amide patterns are disjoint.
    ester = len(molecule.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]")))
    amide = len(molecule.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]")))
    ether = len(molecule.GetSubstructMatches(Chem.MolFromSmarts("[OD2]([#6])[#6]")))
    double_bonds = len(molecule.GetSubstructMatches(Chem.MolFromSmarts("[#6]=[#6]")))
    carbon_atoms = sum(1 for atom in molecule.GetAtoms() if atom.GetSymbol() == "C")
    branch_points = sum(
        1
        for atom in molecule.GetAtoms()
        if atom.GetSymbol() == "C"
        and sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "C") >= 3
    )
    return {
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "carbon_atoms": carbon_atoms,
        "carbon_carbon_double_bonds": double_bonds,
        "unsaturated": double_bonds > 0,
        "ester_groups": ester,
        "amide_groups": amide,
        "ether_oxygens": max(ether - ester, 0),
        "carbon_branch_points": branch_points,
    }


def tail_length_bucket(carbon_atoms: int) -> str:
    """Return a coarse tail-length stratum for reporting."""

    if carbon_atoms <= 8:
        return "c00_c08"
    if carbon_atoms <= 12:
        return "c09_c12"
    if carbon_atoms <= 16:
        return "c13_c16"
    if carbon_atoms <= 20:
        return "c17_c20"
    return "c21_plus"


# ---------------------------------------------------------------------------
# attrition and balance audit
# ---------------------------------------------------------------------------


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def summarize_attrition(
    candidate_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report exact denominators by arm and cohort without any selection."""

    def _group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        unique_products = {str(row["canonical_product"]) for row in rows}
        complete_rows = [row for row in rows if row["product_route_state"] == COMPLETE]
        role_components: dict[str, set[str]] = {role: set() for role in ROLE_NAMES}
        retained_components: dict[str, set[str]] = {role: set() for role in ROLE_NAMES}
        for row in rows:
            for role in ROLE_NAMES:
                smiles = str(row["components"][role])
                role_components[role].add(smiles)
                if row["product_route_state"] == COMPLETE:
                    retained_components[role].add(smiles)
        unsaturated = sum(1 for row in rows if row["descriptors"]["product_unsaturated"])
        return {
            "rows": len(rows),
            "unique_products": len(unique_products),
            "exact_l1_pass_rows": sum(1 for row in rows if row["exact_l1_reverified"]),
            "route_assessment_admitted_rows": sum(
                1 for row in rows if row.get("route_assessment_admitted", True)
            ),
            "route_assessment_not_admitted_rows": sum(
                1 for row in rows if not row.get("route_assessment_admitted", True)
            ),
            "product_route_states": _distribution(row["product_route_state"] for row in rows),
            "route_complete_rows": len(complete_rows),
            "route_complete_unique_products": len(
                {str(row["canonical_product"]) for row in complete_rows}
            ),
            "exact_evidence_current_route_rows": sum(
                1 for row in rows if row["exact_evidence_complete_roles"] == len(ROLE_NAMES)
            ),
            "graph2edits_proposal_coverage_rows": sum(
                1 for row in rows if row["proposal_coverage"]["graph2edits_any_role"]
            ),
            "aizynthfinder_incremental_coverage_rows": sum(
                1 for row in rows if row["proposal_coverage"]["aizynthfinder_incremental_any_role"]
            ),
            "union_proposal_coverage_rows": sum(
                1 for row in rows if row["proposal_coverage"]["union_any_role"]
            ),
            "unique_components_present": {
                role: len(values) for role, values in sorted(role_components.items())
            },
            "unique_components_retained_route_complete": {
                role: len(values) for role, values in sorted(retained_components.items())
            },
            "authority_tiers": _distribution(row["reporting"]["authority_tier"] for row in rows),
            "authority_tiers_route_complete": _distribution(
                row["reporting"]["authority_tier"] for row in complete_rows
            ),
            "conservative_high_rows": sum(
                1 for row in rows if row["reporting"]["conservative_high_potency"]
            ),
            "conservative_high_sources": _distribution(
                row["reporting"]["conservative_high_source"] for row in rows
            ),
            "oracle_scored_rows": sum(1 for row in rows if row["reporting"]["oracle_scored"]),
            "conservative_high_route_complete_rows": sum(
                1 for row in complete_rows if row["reporting"]["conservative_high_potency"]
            ),
            "branched_rows": sum(1 for row in rows if row["descriptors"]["product_branched"]),
            "branched_route_complete_rows": sum(
                1 for row in complete_rows if row["descriptors"]["product_branched"]
            ),
            "linear_rows": sum(1 for row in rows if not row["descriptors"]["product_branched"]),
            "linear_route_complete_rows": sum(
                1 for row in complete_rows if not row["descriptors"]["product_branched"]
            ),
            "unsaturated_rows": unsaturated,
            "unsaturated_route_complete_rows": sum(
                1 for row in complete_rows if row["descriptors"]["product_unsaturated"]
            ),
            "functional_group_rows": {
                "ester": sum(1 for row in rows if row["descriptors"]["product_ester_groups"]),
                "ether": sum(1 for row in rows if row["descriptors"]["product_ether_oxygens"]),
                "amide": sum(1 for row in rows if row["descriptors"]["product_amide_groups"]),
            },
            "aldehyde_tail_length_buckets": _distribution(
                row["descriptors"]["aldehyde_tail_length_bucket"] for row in rows
            ),
            "aldehyde_tail_length_buckets_route_complete": _distribution(
                row["descriptors"]["aldehyde_tail_length_bucket"] for row in complete_rows
            ),
            "corpus_present_rows": sum(
                1 for row in rows if row["reporting"]["exact_refit_corpus_product"]
            ),
            "corpus_present_route_complete_rows": sum(
                1 for row in complete_rows if row["reporting"]["exact_refit_corpus_product"]
            ),
            "corpus_absent_rows": sum(
                1 for row in rows if not row["reporting"]["exact_refit_corpus_product"]
            ),
            "corpus_absent_route_complete_rows": sum(
                1 for row in complete_rows if not row["reporting"]["exact_refit_corpus_product"]
            ),
        }

    by_arm = {
        arm: _group([row for row in candidate_rows if row["reporting"]["arm_id"] == arm])
        for arm in ARM_ORDER
    }
    cohorts = sorted({str(row["reporting"]["cohort"]) for row in candidate_rows})
    by_cohort = {
        cohort: _group([row for row in candidate_rows if row["reporting"]["cohort"] == cohort])
        for cohort in cohorts
    }

    role_states: dict[str, dict[str, int]] = {}
    unresolved_leaf_classes: Counter[str] = Counter()
    for role in ROLE_NAMES:
        subset = [row for row in component_rows if row["role"] == role]
        role_states[role] = {
            "unique_components": len(subset),
            "states": _distribution(row["final_component_state"] for row in subset),
            "exact_assessment_outcomes": _distribution(
                row["exact_evidence"]["assessment_outcome"] for row in subset
            ),
            "graph2edits_attempted": sum(1 for row in subset if row["graph2edits"]["attempted"]),
            "graph2edits_with_proposals": sum(
                1 for row in subset if row["graph2edits"]["proposal_count"] > 0
            ),
            "aizynthfinder_attempted": sum(
                1 for row in subset if row["aizynthfinder"]["attempted"]
            ),
            "aizynthfinder_with_proposals": sum(
                1 for row in subset if row["aizynthfinder"]["single_step_proposal_count"] > 0
            ),
            "aizynthfinder_solved_to_public_stock": sum(
                1 for row in subset if row["aizynthfinder"]["full_search_solved_to_public_stock"]
            ),
        }
    for row in component_rows:
        for leaf_class, count in row["exact_evidence"]["unresolved_leaf_classes"].items():
            unresolved_leaf_classes[leaf_class] += int(count)

    broad = by_arm["broad_prior"]
    support = by_arm["support_enriched"]
    return {
        "by_arm": by_arm,
        "by_cohort": by_cohort,
        "by_role": role_states,
        "unresolved_leaf_classes": dict(sorted(unresolved_leaf_classes.items())),
        "matched_causal_arms": {
            "broad_prior_route_complete_unique_products": broad["route_complete_unique_products"],
            "support_enriched_route_complete_unique_products": support[
                "route_complete_unique_products"
            ],
            "balanced_fillable_unique_products": min(
                broad["route_complete_unique_products"],
                support["route_complete_unique_products"],
            ),
            "routing_induced_imbalance_unique_products": (
                support["route_complete_unique_products"] - broad["route_complete_unique_products"]
            ),
            "arms_were_not_rebalanced_by_dropping_candidates": True,
        },
    }


def attrition_csv_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten the attrition audit into reviewable CSV rows."""

    rows: list[dict[str, Any]] = []
    for grouping in ("by_arm", "by_cohort"):
        for name, values in sorted(summary[grouping].items()):
            rows.append(
                {
                    "grouping": grouping.removeprefix("by_"),
                    "group": name,
                    "rows": values["rows"],
                    "unique_products": values["unique_products"],
                    "exact_l1_pass_rows": values["exact_l1_pass_rows"],
                    "route_complete_rows": values["route_complete_rows"],
                    "route_complete_unique_products": values["route_complete_unique_products"],
                    "exact_evidence_current_route_rows": values[
                        "exact_evidence_current_route_rows"
                    ],
                    "graph2edits_proposal_coverage_rows": values[
                        "graph2edits_proposal_coverage_rows"
                    ],
                    "aizynthfinder_incremental_coverage_rows": values[
                        "aizynthfinder_incremental_coverage_rows"
                    ],
                    "union_proposal_coverage_rows": values["union_proposal_coverage_rows"],
                    "conservative_high_rows": values["conservative_high_rows"],
                    "conservative_high_route_complete_rows": values[
                        "conservative_high_route_complete_rows"
                    ],
                    "branched_rows": values["branched_rows"],
                    "branched_route_complete_rows": values["branched_route_complete_rows"],
                    "unsaturated_rows": values["unsaturated_rows"],
                    "corpus_absent_rows": values["corpus_absent_rows"],
                    "corpus_absent_route_complete_rows": values[
                        "corpus_absent_route_complete_rows"
                    ],
                    "unique_amines": values["unique_components_present"]["amine_head"],
                    "unique_aldehydes": values["unique_components_present"][
                        "oxoester_aldehyde_body_tail"
                    ],
                    "unique_isocyanides": values["unique_components_present"]["isocyanide_tail"],
                }
            )
    return rows


ATTRITION_CSV_FIELDS = (
    "grouping",
    "group",
    "rows",
    "unique_products",
    "exact_l1_pass_rows",
    "route_complete_rows",
    "route_complete_unique_products",
    "exact_evidence_current_route_rows",
    "graph2edits_proposal_coverage_rows",
    "aizynthfinder_incremental_coverage_rows",
    "union_proposal_coverage_rows",
    "conservative_high_rows",
    "conservative_high_route_complete_rows",
    "branched_rows",
    "branched_route_complete_rows",
    "unsaturated_rows",
    "corpus_absent_rows",
    "corpus_absent_route_complete_rows",
    "unique_amines",
    "unique_aldehydes",
    "unique_isocyanides",
)


def branch_sufficiency_recommendation(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report branch route readiness and recommend, never perform, a new tranche."""

    branch_rows = [row for row in candidate_rows if row["reporting"]["arm_id"] == BRANCH_ARM]
    complete = [row for row in branch_rows if row["product_route_state"] == COMPLETE]
    distinct_aldehydes = {str(row["components"]["oxoester_aldehyde_body_tail"]) for row in complete}
    conservative_high_complete = [
        row for row in complete if row["reporting"]["conservative_high_potency"]
    ]
    return {
        "branch_rows": len(branch_rows),
        "branch_route_complete_rows": len(complete),
        "branch_route_complete_distinct_aldehydes": len(distinct_aldehydes),
        "branch_route_complete_conservative_high_rows": len(conservative_high_complete),
        "second_tranche_recommended": len(distinct_aldehydes) < 3,
        "recommendation": (
            "Launch one additional identically configured, independently seeded "
            "branch-conditioned tranche; do not change the branch definition, the "
            "admission contract or the applicability boundary."
            if len(distinct_aldehydes) < 3
            else "Current branch route readiness is sufficient for prospective selection."
        ),
        "recommendation_is_not_an_action": True,
    }


# ---------------------------------------------------------------------------
# route-tree leaves
# ---------------------------------------------------------------------------


def collect_route_leaves(route_tree: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return every leaf of one recursive assessment tree with its outcome."""

    leaves: list[dict[str, Any]] = []

    def _walk(node: Mapping[str, Any], depth: int) -> None:
        children = node.get("children")
        if not isinstance(children, list):
            raise UgiBoundedHybridRouteCascadeError("route node children are malformed")
        if not children:
            target = node.get("target")
            if not isinstance(target, Mapping):
                raise UgiBoundedHybridRouteCascadeError("route leaf has no target")
            evidence = node.get("evidence")
            leaves.append(
                {
                    "leaf_smiles": str(target.get("canonical_smiles")),
                    "leaf_role": str(target.get("role")),
                    "depth": depth,
                    "outcome": str(node.get("outcome")),
                    "detail": str(node.get("detail")),
                    "evidence_tiers": sorted(
                        {
                            str(record.get("tier"))
                            for record in (evidence if isinstance(evidence, list) else [])
                            if isinstance(record, Mapping)
                        }
                    ),
                }
            )
            return
        for child in children:
            if not isinstance(child, Mapping):
                raise UgiBoundedHybridRouteCascadeError("route child node is malformed")
            _walk(child, depth + 1)

    _walk(route_tree, 0)
    return leaves


NEVER_EXPANDED = "never_expanded_outside_evidence_index"
EXPANDED_THEN_UNRESOLVED = "expanded_then_unresolved"


def unresolved_disposition(
    route_tree: Mapping[str, Any] | None, *, strict_complete: bool
) -> str | None:
    """Distinguish an unattempted lookup miss from a search that ran and failed.

    This is the load-bearing diagnostic for reading the closure rate honestly.
    The frozen exact-evidence source is an index, not a retrosynthetic search:
    a target absent from the index returns ``missing_knowledge`` at depth zero
    having attempted no disconnection at all.  Reporting closure without this
    split invites reading an index-miss rate as a search result.
    """

    if strict_complete:
        return None
    if not isinstance(route_tree, Mapping):
        return NEVER_EXPANDED
    children = route_tree.get("children")
    if not isinstance(children, list) or not children:
        return NEVER_EXPANDED
    return EXPANDED_THEN_UNRESOLVED


def unresolved_leaf_classes(leaves: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count leaves that did not close to a current terminal material."""

    counts: Counter[str] = Counter()
    for leaf in leaves:
        outcome = str(leaf["outcome"])
        if outcome != "complete":
            counts[outcome] += 1
    return dict(sorted(counts.items()))


__all__ = [
    "ADJUDICATION_SCHEMA_VERSION",
    "ARM_ORDER",
    "ATTRITION_CSV_FIELDS",
    "BRANCH_ARM",
    "CACHE_CLONE_ID",
    "CANDIDATE_EXACT_LEDGER_SCHEMA_VERSION",
    "CANDIDATE_LEDGER_SCHEMA_VERSION",
    "COMPLETE",
    "COMPONENT_EXACT_LEDGER_SCHEMA_VERSION",
    "COMPONENT_LEDGER_SCHEMA_VERSION",
    "COMPONENT_STATES",
    "CONFIG_SCHEMA_VERSION",
    "EXPANDED_THEN_UNRESOLVED",
    "NEVER_EXPANDED",
    "EXACT_EVIDENCE_SCHEMA_VERSION",
    "EXPECTED_INPUTS",
    "EXPECTED_SCOPE",
    "LEAF_LEDGER_SCHEMA_VERSION",
    "MAIN_ARMS",
    "NOT_ASSESSED_OUTCOME",
    "PREFLIGHT_SCHEMA_VERSION",
    "REJECTED",
    "RESULT_SCHEMA_VERSION",
    "SEARCH_CENSORED",
    "UNRESOLVED",
    "CandidateRecord",
    "CascadeContract",
    "UgiBoundedHybridRouteCascadeError",
    "assert_expected_population",
    "atomic_write",
    "attrition_csv_rows",
    "blindness_receipt",
    "branch_sufficiency_recommendation",
    "build_candidate_population",
    "canonical_json_bytes",
    "collect_route_leaves",
    "component_descriptors",
    "component_sha256",
    "component_target_id",
    "csv_gzip_bytes",
    "derive_final_component_state",
    "jsonl_gzip_bytes",
    "load_cascade_contract",
    "load_json",
    "product_route_state",
    "read_csv_gzip",
    "read_jsonl_gzip",
    "rescoring_index",
    "resolve_potency_reporting",
    "sha256_file",
    "sha256_payload",
    "stable_json",
    "summarize_attrition",
    "tail_length_bucket",
    "unique_component_targets",
    "unresolved_disposition",
    "unresolved_leaf_classes",
]

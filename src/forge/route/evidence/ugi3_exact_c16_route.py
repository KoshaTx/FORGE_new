"""Qualify one exact C16 alkynyl-aldehyde route without family promotion.

The admitted route is deliberately narrow.  Exact protected propargyl alcohol
and exact 1-bromotridecane form the C16 chain, which is deprotected, subjected
to the reported C16 alkyne zipper, and oxidized to hexadec-15-ynal.  Every
reaction is bound to one exact substrate/product pair.  A contradictory patent
example is retained as rejected ``source_conflict`` evidence and cannot close
or expand the route.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from forge.core.io import stable_json as _stable_json
from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.route.engine.planner import (
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteKnowledgeSource,
    RouteStepProposal,
    RouteTarget,
)
from forge.route.engine.qualified_forward import (
    QualifiedForwardError,
    load_qualified_forward_reaction,
    unique_forward_products,
)
from forge.value.synthesis.synthesis import component_synthesis_value_from_assessment

CONFIG_SCHEMA_VERSION = "phase1_ugi3_exact_c16_route_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_exact_c16_route_audit.v1"
STEP_LEDGER_SCHEMA_VERSION = "phase1_ugi3_exact_c16_route_steps.v1"
ASSESSMENT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_exact_c16_assessment.v1"


class Ugi3ExactC16RouteError(ValueError):
    """Raised when the exact C16 evidence chain violates its frozen contract."""


def _gzip_json_bytes(value: Any) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write((_stable_json(value) + "\n").encode())
    return output.getvalue()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise Ugi3ExactC16RouteError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3ExactC16RouteError(f"{label} must be a JSON object")
    return value


def _canonical(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise Ugi3ExactC16RouteError(f"{label} must be nonempty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3ExactC16RouteError(f"{label} contains invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _validate_inputs(config: Mapping[str, Any], input_paths: Mapping[str, Path]) -> dict[str, str]:
    declared = config.get("inputs")
    if not isinstance(declared, dict) or set(declared) != set(input_paths):
        raise Ugi3ExactC16RouteError("configured and supplied C16 inputs differ")
    hashes: dict[str, str] = {}
    for label, path in input_paths.items():
        record = declared[label]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise Ugi3ExactC16RouteError(f"input {label} is malformed")
        observed = sha256_file(path)
        if observed != record["sha256"]:
            raise Ugi3ExactC16RouteError(f"input hash changed: {label}")
        hashes[label] = observed
    return hashes


def _validate_expected(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise Ugi3ExactC16RouteError(f"{label} fields changed")
        for key, value in expected.items():
            _validate_expected(observed[key], value, label=f"{label}.{key}")
        return
    if observed != expected:
        raise Ugi3ExactC16RouteError(
            f"{label} changed: expected {expected!r}, observed {observed!r}"
        )


def _carbon_count(molecule: Chem.Mol) -> int:
    return sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())


def _parse_iso_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise Ugi3ExactC16RouteError(f"{label} must be strict ISO UTC")
    format_string = "%Y-%m-%dT%H:%M:%SZ"
    try:
        parsed = datetime.strptime(value, format_string)
    except ValueError as exc:
        raise Ugi3ExactC16RouteError(f"{label} must be strict ISO UTC") from exc
    if parsed.strftime(format_string) != value:
        raise Ugi3ExactC16RouteError(f"{label} must be strict ISO UTC")
    return parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class _ExactStep:
    step_id: str
    reaction_id: str
    reactants: tuple[RouteTarget, ...]
    product: RouteTarget
    evidence: EvidenceRecord


class _MissingSource:
    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        return KnowledgeResult(
            disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
            evidence=(),
            detail="base source contains no exact route for this target",
        )


class ExactC16RouteOverlay:
    """Override one exact route tree and delegate every other identity."""

    def __init__(
        self,
        *,
        base_source: RouteKnowledgeSource,
        decisions: Mapping[tuple[str, str], KnowledgeResult],
        target: RouteTarget,
    ):
        self._base_source = base_source
        self._decisions = dict(decisions)
        self.target = target

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        canonical = _canonical(target.canonical_smiles, label="C16 overlay target")
        selected = self._decisions.get((target.role, canonical))
        if selected is not None:
            return selected
        return self._base_source.lookup(target)


def _validate_scope_limitations(config: Mapping[str, Any]) -> None:
    if config.get("scope") != "one_exact_c16_route_only":
        raise Ugi3ExactC16RouteError("C16 config has broader-than-authorized scope")
    limitations = config.get("limitations")
    required = {
        "exact_substrate_product_pairs_only": True,
        "reaction_family_promotion_authorized": False,
        "substrate_scope_extrapolation_authorized": False,
        "analogue_promotion_authorized": False,
        "homologue_promotion_authorized": False,
        "general_enumeration_authorized": False,
        "terminal_closure_transfers_to_analogues": False,
        "rejected_source_conflict_can_support_route": False,
    }
    if not isinstance(limitations, dict) or any(
        limitations.get(key) is not value for key, value in required.items()
    ):
        raise Ugi3ExactC16RouteError("C16 config lacks exact-pair-only limitations")


def _validate_source_manifest(
    *,
    input_paths: Mapping[str, Path],
) -> None:
    manifest = _load_json(input_paths["source_manifest"], label="C16 source manifest")
    if (
        manifest.get("schema_version") != "forge.source_cache_manifest.v1"
        or manifest.get("scope") != "one_exact_c16_alkynyl_aldehyde_route_only"
    ):
        raise Ugi3ExactC16RouteError("C16 source manifest scope changed")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise Ugi3ExactC16RouteError("C16 source manifest must contain three sources")
    required = {
        "wo2021035214a1_family_primary": (
            "route_primary",
            "accepted_exact_source",
        ),
        "zheng_ja311416v_si_primary": (
            "oxidation_primary",
            "accepted_exact_source",
        ),
        "wo2024073486a2_conflicted": (
            "conflicted_source",
            "rejected_source_conflict",
        ),
    }
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise Ugi3ExactC16RouteError("C16 source manifest entry is malformed")
        source_id = source.get("source_id")
        if source_id not in required or source_id in seen:
            raise Ugi3ExactC16RouteError("C16 source manifest identifiers changed")
        input_label, disposition = required[str(source_id)]
        if (
            source.get("sha256") != sha256_file(input_paths[input_label])
            or source.get("disposition") != disposition
        ):
            raise Ugi3ExactC16RouteError("C16 source manifest hash or disposition changed")
        seen.add(str(source_id))
    claims = manifest.get("claims_boundary")
    if not isinstance(claims, dict) or any(
        claims.get(key) is not False
        for key in (
            "blocked_or_access_denied_html_preserved",
            "family_scope_promoted",
            "homologue_scope_promoted",
            "rejected_source_used_in_route",
        )
    ):
        raise Ugi3ExactC16RouteError("C16 source manifest lacks fail-closed boundaries")


def _validate_rejected_source(
    *,
    config: Mapping[str, Any],
    input_paths: Mapping[str, Path],
) -> dict[str, Any]:
    records = config.get("rejected_evidence")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise Ugi3ExactC16RouteError("C16 route must retain one rejected source conflict")
    record = records[0]
    expected = {
        "status": "rejected_source_conflict",
        "source_input": "conflicted_source",
        "use_in_route": False,
        "family_promotion_authorized": False,
        "homologue_promotion_authorized": False,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise Ugi3ExactC16RouteError("C16 rejected source conflict was promoted or changed")
    carbon = record.get("carbon_audit")
    if (
        not isinstance(carbon, dict)
        or carbon.get("first_reactant_carbons") != 8
        or carbon.get("second_reactant_carbons") != 7
        or carbon.get("reported_product_carbons") != 16
        or carbon.get("reactant_carbon_sum") != 15
        or carbon.get("is_carbon_coherent") is not False
    ):
        raise Ugi3ExactC16RouteError("C16 rejected source conflict lacks exact carbon audit")
    source_sha = sha256_file(input_paths["conflicted_source"])
    if record.get("source_sha256") != source_sha:
        raise Ugi3ExactC16RouteError("C16 rejected source hash changed")
    return {
        "evidence_id": record.get("evidence_id"),
        "status": record.get("status"),
        "source_sha256": source_sha,
        "source_locator": record.get("source_locator"),
        "reason": record.get("reason"),
        "carbon_audit": carbon,
        "used_in_route": False,
    }


def _validate_registry(
    *,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    molecule_smiles: Mapping[str, str],
) -> None:
    if registry.get("scope") != "one_exact_c16_substrate_route_only":
        raise Ugi3ExactC16RouteError("C16 registry has broader-than-authorized scope")
    claims = registry.get("claims_boundary")
    required_false = {
        "qualified_for_general_enumeration",
        "qualified_for_substrate_scope_extrapolation",
        "qualified_as_a_reaction_family",
        "conditions_encoded_in_graph_transform",
        "exact_graph_reconstruction_is_experimental_success",
        "exact_l2_evidence_implies_current_l3_procurement",
        "rejected_source_conflict_used",
    }
    if not isinstance(claims, dict) or any(claims.get(key) is not False for key in required_false):
        raise Ugi3ExactC16RouteError("C16 registry lacks fail-closed claim boundaries")
    reactions = registry.get("reactions")
    steps = config.get("steps")
    if not isinstance(reactions, list) or not isinstance(steps, list):
        raise Ugi3ExactC16RouteError("C16 registry or route steps are malformed")
    by_id = {
        reaction.get("reaction_id"): reaction
        for reaction in reactions
        if isinstance(reaction, dict) and isinstance(reaction.get("reaction_id"), str)
    }
    if len(by_id) != len(reactions) or len(reactions) != len(steps):
        raise Ugi3ExactC16RouteError("C16 reaction identifiers are not one-to-one")
    for step in steps:
        if not isinstance(step, dict):
            raise Ugi3ExactC16RouteError("C16 route step is malformed")
        definition = by_id.get(step.get("reaction_id"))
        reactant_ids = step.get("reactant_ids")
        if not isinstance(definition, dict) or not isinstance(reactant_ids, list):
            raise Ugi3ExactC16RouteError("C16 reaction definition is malformed")
        declared_reactants = definition.get("exact_reactant_smiles")
        if not isinstance(declared_reactants, list):
            raise Ugi3ExactC16RouteError("C16 exact reaction reactants must be a list")
        configured_reactants = [molecule_smiles.get(str(item)) for item in reactant_ids]
        if (
            definition.get("status") != "qualified_for_one_exact_substrate_product_pair_only"
            or definition.get("qualification_pair_count") != 1
            or len(declared_reactants) != len(configured_reactants)
            or [_canonical(value, label="registry exact reactant") for value in declared_reactants]
            != configured_reactants
            or _canonical(definition.get("exact_product_smiles"), label="registry product")
            != molecule_smiles.get(str(step.get("product_id")))
        ):
            raise Ugi3ExactC16RouteError(
                f"reaction {step.get('reaction_id')!r} is not exact-pair qualified"
            )


def _molecule_records(config: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, int]]:
    records = config.get("molecules")
    if not isinstance(records, dict):
        raise Ugi3ExactC16RouteError("C16 molecule records are malformed")
    smiles_by_id: dict[str, str] = {}
    chain_carbons_by_id: dict[str, int] = {}
    for molecule_id, record in records.items():
        if not isinstance(molecule_id, str) or not isinstance(record, dict):
            raise Ugi3ExactC16RouteError("C16 molecule identity is malformed")
        canonical = _canonical(record.get("canonical_smiles"), label=molecule_id)
        molecule = Chem.MolFromSmiles(canonical)
        assert molecule is not None
        observed_carbon_count = _carbon_count(molecule)
        chain_count = record.get("route_chain_carbon_count")
        if (
            Chem.MolToInchiKey(molecule) != record.get("inchi_key")
            or rdMolDescriptors.CalcMolFormula(molecule) != record.get("molecular_formula")
            or observed_carbon_count != record.get("carbon_count")
            or isinstance(chain_count, bool)
            or not isinstance(chain_count, int)
            or chain_count < 1
            or chain_count > observed_carbon_count
        ):
            raise Ugi3ExactC16RouteError(f"C16 molecule identity changed: {molecule_id}")
        smiles_by_id[molecule_id] = canonical
        chain_carbons_by_id[molecule_id] = chain_count
    return smiles_by_id, chain_carbons_by_id


def _validate_route_tree_and_carbon_identity(
    *,
    config: Mapping[str, Any],
    molecule_smiles: Mapping[str, str],
    chain_carbons: Mapping[str, int],
) -> None:
    steps = config.get("steps")
    if not isinstance(steps, list) or len(steps) != 4:
        raise Ugi3ExactC16RouteError("C16 route must contain exactly four steps")
    expected_reactants = [
        ["tetrahydro_2_propynyloxy_2h_pyran", "1_bromotridecane"],
        ["thp_protected_hexadec_2_yn_1_ol"],
        ["hexadec_2_yn_1_ol"],
        ["hexadec_15_yn_1_ol"],
    ]
    expected_products = [
        "thp_protected_hexadec_2_yn_1_ol",
        "hexadec_2_yn_1_ol",
        "hexadec_15_yn_1_ol",
        "hexadec_15_ynal",
    ]
    expected_total_carbon = [(21, 21, 0), (21, 16, 5), (16, 16, 0), (16, 16, 0)]
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise Ugi3ExactC16RouteError("C16 route step is malformed")
        reactant_ids = step.get("reactant_ids")
        product_id = step.get("product_id")
        if reactant_ids != expected_reactants[index] or product_id != expected_products[index]:
            raise Ugi3ExactC16RouteError("C16 route topology or chain continuity changed")
        if (
            any(item not in molecule_smiles for item in reactant_ids)
            or product_id not in molecule_smiles
        ):
            raise Ugi3ExactC16RouteError("C16 route references an unknown molecule")
        total_reactant_carbons = sum(
            _carbon_count(Chem.MolFromSmiles(molecule_smiles[item])) for item in reactant_ids
        )
        product_carbons = _carbon_count(Chem.MolFromSmiles(molecule_smiles[product_id]))
        discarded_carbons = total_reactant_carbons - product_carbons
        if (total_reactant_carbons, product_carbons, discarded_carbons) != expected_total_carbon[
            index
        ]:
            raise Ugi3ExactC16RouteError("C16 total-carbon identity changed")
        if index == 0:
            expected_chain = sum(chain_carbons[item] for item in reactant_ids)
        else:
            expected_chain = chain_carbons[reactant_ids[0]]
        if expected_chain != 16 or chain_carbons[product_id] != expected_chain:
            raise Ugi3ExactC16RouteError("C16 route-chain carbon continuity changed")
        carbon = step.get("carbon_identity")
        expected_carbon_record = {
            "reactant_total_carbons": total_reactant_carbons,
            "product_total_carbons": product_carbons,
            "discarded_protecting_group_carbons": discarded_carbons,
            "route_chain_carbons": 16,
            "exact_chain_identity_preserved": True,
        }
        if carbon != expected_carbon_record:
            raise Ugi3ExactC16RouteError("C16 configured carbon-identity record changed")


def _terminal_records(
    *,
    config: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    molecule_smiles: Mapping[str, str],
) -> tuple[dict[tuple[str, str], KnowledgeResult], AvailabilityState, int]:
    terminals = config.get("terminal_procurement")
    if not isinstance(terminals, list) or len(terminals) != 2:
        raise Ugi3ExactC16RouteError("C16 route requires exactly two terminal leaves")
    observation = _load_json(
        input_paths["terminal_observations"], label="C16 terminal observations"
    )
    items = observation.get("items")
    if (
        observation.get("schema_version") != "forge.vendor_product_observations.v1"
        or not isinstance(items, list)
        or len(items) != 2
        or observation.get("raw_html_cached") is not False
    ):
        raise Ugi3ExactC16RouteError("C16 terminal observation record is malformed")
    assessment_as_of = _parse_iso_utc(
        config.get("assessment_as_of_utc"), label="C16 assessment_as_of_utc"
    )
    recorded_at = _parse_iso_utc(
        observation.get("recorded_at_utc"), label="C16 terminal recorded_at_utc"
    )
    refresh = observation.get("refresh_policy")
    if not isinstance(refresh, dict) or set(refresh) != {
        "validity_days",
        "expires_at_utc",
        "refresh_required_before_candidate_lock",
    }:
        raise Ugi3ExactC16RouteError("C16 terminal refresh policy is malformed")
    validity_days = refresh.get("validity_days")
    if (
        isinstance(validity_days, bool)
        or not isinstance(validity_days, int)
        or validity_days <= 0
        or refresh.get("refresh_required_before_candidate_lock") is not True
    ):
        raise Ugi3ExactC16RouteError("C16 terminal refresh policy is malformed")
    expires_at = _parse_iso_utc(refresh.get("expires_at_utc"), label="C16 terminal refresh expiry")
    if expires_at != recorded_at + timedelta(days=validity_days):
        raise Ugi3ExactC16RouteError("C16 terminal refresh policy is inconsistent")
    claims = observation.get("claims_boundary")
    if (
        not isinstance(claims, dict)
        or claims.get("availability_remains_current_after_expiry") is not False
    ):
        raise Ugi3ExactC16RouteError("C16 terminal expiry claim boundary is missing")
    if assessment_as_of < recorded_at:
        raise Ugi3ExactC16RouteError("C16 terminal observation is future-dated")
    if assessment_as_of > expires_at:
        raise Ugi3ExactC16RouteError("C16 terminal availability evidence is expired")
    observation_by_id = {item.get("molecule_id"): item for item in items if isinstance(item, dict)}
    if len(observation_by_id) != 2:
        raise Ugi3ExactC16RouteError("C16 terminal observation identities are not unique")
    decisions: dict[tuple[str, str], KnowledgeResult] = {}
    for terminal in terminals:
        if not isinstance(terminal, dict):
            raise Ugi3ExactC16RouteError("C16 terminal record is malformed")
        molecule_id = terminal.get("molecule_id")
        item = observation_by_id.get(molecule_id)
        canonical = molecule_smiles.get(str(molecule_id))
        if not isinstance(item, dict) or canonical is None:
            raise Ugi3ExactC16RouteError("C16 terminal observation is missing")
        availability = item.get("availability_observation")
        item_observed_at = _parse_iso_utc(
            item.get("observed_at_utc"), label=f"C16 terminal {molecule_id} observed_at_utc"
        )
        item_expires_at = _parse_iso_utc(
            item.get("expires_at_utc"), label=f"C16 terminal {molecule_id} expires_at_utc"
        )
        molecule = Chem.MolFromSmiles(canonical)
        assert molecule is not None
        if (
            terminal.get("observation_source_input") != "terminal_observations"
            or terminal.get("vendor") != item.get("vendor")
            or terminal.get("product_url") != item.get("official_product_url")
            or terminal.get("item_id") != item.get("catalog_number")
            or terminal.get("cas_number") != item.get("cas_number")
            or _canonical(terminal.get("canonical_smiles"), label="C16 terminal") != canonical
            or _canonical(item.get("canonical_smiles"), label="C16 observed terminal") != canonical
            or terminal.get("inchi_key") != Chem.MolToInchiKey(molecule)
            or item.get("inchi_key") != Chem.MolToInchiKey(molecule)
            or terminal.get("identity_exact") is not True
            or item.get("identity_exact") is not True
            or terminal.get("item_specific_purity") != item.get("item_specific_purity")
            or terminal.get("observed_at_utc") != item.get("observed_at_utc")
            or terminal.get("expires_at_utc") != item.get("expires_at_utc")
            or item_observed_at != recorded_at
            or item_expires_at != expires_at
            or not (item_observed_at <= assessment_as_of <= item_expires_at)
            or not isinstance(availability, dict)
            or availability.get("explicit_current_stock_or_shipping_observed") is not True
            or availability.get("current_item_level_procurement_closed") is not True
        ):
            raise Ugi3ExactC16RouteError(
                "C16 terminal config disagrees with exact vendor observation"
            )
        evidence = EvidenceRecord(
            evidence_id=f"exact-c16:terminal:{molecule_id}",
            tier=EvidenceTier.ACCEPTED_TERMINAL,
            source_sha256=sha256_file(input_paths["terminal_observations"]),
            source_locator=str(item.get("official_product_url")),
            exact_substrate=True,
            forward_verification=ForwardVerificationState.NOT_APPLICABLE,
            availability=AvailabilityState.CURRENT_CLOSED,
        )
        target = RouteTarget(role=str(terminal.get("role")), canonical_smiles=canonical)
        decisions[(target.role, target.canonical_smiles)] = KnowledgeResult(
            disposition=KnowledgeDisposition.TERMINAL,
            evidence=(evidence,),
            detail="exact C16 route leaf has item-level identity, purity, and current availability",
        )
    return decisions, AvailabilityState.CURRENT_CLOSED, len(decisions)


def _randomized_reactants(smiles: Sequence[str]) -> list[str]:
    randomized: list[str] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        assert molecule is not None
        randomized.append(
            Chem.MolToSmiles(
                molecule,
                canonical=False,
                doRandom=True,
                isomericSmiles=False,
            )
        )
    return randomized


def _build_overlay_and_rows(
    *,
    config: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    base_source: RouteKnowledgeSource,
) -> tuple[ExactC16RouteOverlay, list[dict[str, Any]], AvailabilityState, int]:
    molecule_smiles, chain_carbons = _molecule_records(config)
    _validate_route_tree_and_carbon_identity(
        config=config,
        molecule_smiles=molecule_smiles,
        chain_carbons=chain_carbons,
    )
    registry = _load_json(input_paths["reaction_registry"], label="C16 reaction registry")
    _validate_registry(
        config=config,
        registry=registry,
        molecule_smiles=molecule_smiles,
    )
    steps = config["steps"]
    trials = config.get("verification_policy", {}).get("randomized_smiles_trials")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise Ugi3ExactC16RouteError("randomized SMILES trial count must be positive")
    exact_steps: list[_ExactStep] = []
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(steps):
        reactant_ids = raw["reactant_ids"]
        product_id = raw["product_id"]
        reactant_smiles = [molecule_smiles[item] for item in reactant_ids]
        expected = molecule_smiles[product_id]
        variant_label = raw.get("variant_input")
        reaction_id = raw.get("reaction_id")
        if (
            not isinstance(variant_label, str)
            or variant_label not in input_paths
            or not isinstance(reaction_id, str)
        ):
            raise Ugi3ExactC16RouteError("C16 step lacks a pinned exact reaction variant")
        try:
            compiled = load_qualified_forward_reaction(
                input_paths["reaction_registry"],
                input_paths[variant_label],
                reaction_id=reaction_id,
            )
            observed = unique_forward_products(
                compiled,
                reactant_smiles,
                max_products=32,
                isomeric_smiles=False,
            )
            if observed != (expected,):
                raise Ugi3ExactC16RouteError(
                    f"C16 step {reaction_id} did not uniquely reconstruct its target"
                )
            successful_trials = 0
            for _ in range(trials):
                randomized_outputs = unique_forward_products(
                    compiled,
                    _randomized_reactants(reactant_smiles),
                    max_products=32,
                    isomeric_smiles=False,
                )
                successful_trials += int(randomized_outputs == (expected,))
        except QualifiedForwardError as exc:
            raise Ugi3ExactC16RouteError(
                f"C16 forward verification failed for {reaction_id}: {exc}"
            ) from exc
        if successful_trials != trials:
            raise Ugi3ExactC16RouteError(
                f"C16 step {reaction_id} is not invariant to randomized SMILES"
            )
        source_input = raw.get("primary_source_input")
        if not isinstance(source_input, str) or source_input not in input_paths:
            raise Ugi3ExactC16RouteError("C16 step lacks a pinned primary source")
        if source_input == "conflicted_source":
            raise Ugi3ExactC16RouteError("rejected C16 source conflict cannot support a route step")
        source_sha = sha256_file(input_paths[source_input])
        conditions = raw.get("conditions")
        if not isinstance(conditions, dict) or not conditions:
            raise Ugi3ExactC16RouteError("C16 step lacks structured conditions")
        if raw.get("conditions_source_sha256") != source_sha:
            raise Ugi3ExactC16RouteError("C16 step conditions source hash changed")
        source_locator = raw.get("source_locator")
        conditions_locator = raw.get("conditions_source_locator")
        if not isinstance(source_locator, str) or not isinstance(conditions_locator, str):
            raise Ugi3ExactC16RouteError("C16 step lacks exact source locators")
        evidence = EvidenceRecord(
            evidence_id=f"exact-c16:{raw.get('step_id')}",
            tier=EvidenceTier.EXACT_SOURCE,
            source_sha256=source_sha,
            source_locator=source_locator,
            exact_substrate=True,
            forward_verification=ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE,
            availability=AvailabilityState.UNASSESSED,
        )
        reactant_targets = tuple(
            RouteTarget(role=str(role), canonical_smiles=smiles)
            for role, smiles in zip(raw.get("reactant_roles", []), reactant_smiles, strict=True)
        )
        if len(reactant_targets) != len(reactant_smiles):
            raise Ugi3ExactC16RouteError("C16 reactant roles do not align")
        product_target = RouteTarget(role=str(raw.get("product_role")), canonical_smiles=expected)
        exact_steps.append(
            _ExactStep(
                step_id=str(raw.get("step_id")),
                reaction_id=reaction_id,
                reactants=reactant_targets,
                product=product_target,
                evidence=evidence,
            )
        )
        rows.append(
            {
                "step_index": index + 1,
                "step_id": raw.get("step_id"),
                "reaction_id": reaction_id,
                "reaction_class": raw.get("reaction_class"),
                "reactants": [
                    {
                        "molecule_id": molecule_id,
                        "role": target.role,
                        "canonical_smiles": target.canonical_smiles,
                        "carbon_count": _carbon_count(Chem.MolFromSmiles(target.canonical_smiles)),
                    }
                    for molecule_id, target in zip(reactant_ids, reactant_targets, strict=True)
                ],
                "product_id": product_id,
                "product_role": product_target.role,
                "product_smiles": product_target.canonical_smiles,
                "carbon_identity": raw.get("carbon_identity"),
                "source_input": source_input,
                "source_sha256": source_sha,
                "source_locator": source_locator,
                "conditions": conditions,
                "conditions_source_sha256": source_sha,
                "conditions_source_locator": conditions_locator,
                "isolated_yield_percent": raw.get("isolated_yield_percent"),
                "characterization": raw.get("characterization"),
                "forward_products": list(observed),
                "forward_product_count": len(observed),
                "verified_exact_product_unique": observed == (expected,),
                "randomized_smiles_trials": trials,
                "randomized_smiles_successes": successful_trials,
                "family_scope_admitted": False,
            }
        )

    decisions, terminal_state, terminal_count = _terminal_records(
        config=config,
        input_paths=input_paths,
        molecule_smiles=molecule_smiles,
    )
    for step in exact_steps:
        proposal = RouteStepProposal(
            reaction_id=step.reaction_id,
            reactants=step.reactants,
            evidence=(step.evidence,),
            forward_product_count=1,
            verifier_calls_required=1,
            product_candidates_considered=1,
        )
        decisions[(step.product.role, step.product.canonical_smiles)] = KnowledgeResult(
            disposition=KnowledgeDisposition.EXPAND,
            evidence=(step.evidence,),
            proposal=proposal,
            detail="exact C16 substrate/product pair retrieved and uniquely verified",
        )
    overlay = ExactC16RouteOverlay(
        base_source=base_source,
        decisions=decisions,
        target=exact_steps[-1].product,
    )
    return overlay, rows, terminal_state, terminal_count


def build_exact_c16_route_audit(
    *,
    config_path: Path,
    input_paths: Mapping[str, Path],
    base_source: RouteKnowledgeSource | None = None,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Verify the exact C16 route and emit an evidence-preserving assessment."""

    config = _load_json(config_path, label="exact C16 route config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3ExactC16RouteError("unsupported exact C16 route config")
    _validate_scope_limitations(config)
    input_hashes = _validate_inputs(config, input_paths)
    _validate_source_manifest(input_paths=input_paths)
    rejected = _validate_rejected_source(config=config, input_paths=input_paths)
    overlay, rows, terminal_state, terminal_count = _build_overlay_and_rows(
        config=config,
        input_paths=input_paths,
        base_source=base_source or _MissingSource(),
    )
    limits = PlannerBudgetLimits.from_dict(config.get("planner_budget"))
    budget = PlannerBudgetLedger(limits)
    assessment = RecursiveRouteAssessor(overlay).assess(overlay.target, budget)
    value = component_synthesis_value_from_assessment(assessment)
    summary = {
        "target_role": overlay.target.role,
        "target_smiles": overlay.target.canonical_smiles,
        "target_carbon_count": 16,
        "assessment_as_of_utc": config.get("assessment_as_of_utc"),
        "exact_steps": len(rows),
        "uniquely_forward_verified_steps": sum(
            row["verified_exact_product_unique"] for row in rows
        ),
        "randomized_smiles_trials_per_step": rows[0]["randomized_smiles_trials"],
        "randomized_smiles_total_successes": sum(
            row["randomized_smiles_successes"] for row in rows
        ),
        "terminal_leaf_count": terminal_count,
        "terminal_availability": terminal_state.value,
        "assessment_outcome": assessment.outcome.value,
        "route_complete": value.route_complete,
        "route_step_count": value.route_step_count,
        "maximum_route_depth": value.maximum_route_depth,
        "unassessed_terminal_leaf_count": value.unassessed_terminal_leaf_count,
        "forward_consistency": value.forward_consistency.value,
        "evidence_support": value.evidence_support.value,
        "reaction_family_admitted": False,
        "rejected_source_conflict_count": 1,
        "rejected_source_conflict_used": False,
        "development_product_coverage_changed": False,
    }
    _validate_expected(summary, config.get("expected_summary"), label="C16 summary")
    step_ledger = _gzip_json_bytes({"schema_version": STEP_LEDGER_SCHEMA_VERSION, "rows": rows})
    assessment_ledger = _gzip_json_bytes(
        {
            "schema_version": ASSESSMENT_LEDGER_SCHEMA_VERSION,
            "assessment_as_of_utc": config.get("assessment_as_of_utc"),
            "assessment": assessment.to_dict(),
            "synthesis_value": value.to_dict(),
            "budget": budget.to_dict(),
        }
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": (
            "complete_exact_l2_l3_closed_audit"
            if value.route_complete
            else "complete_exact_l2_l3_open_audit"
        ),
        "config_sha256": sha256_file(config_path),
        "inputs": {
            label: {"path": str(path), "sha256": input_hashes[label]}
            for label, path in sorted(input_paths.items())
        },
        "summary": summary,
        "rejected_evidence": [rejected],
        "artifacts": {
            "step_verification_ledger.json.gz": {
                "schema_version": STEP_LEDGER_SCHEMA_VERSION,
                "sha256": sha256_bytes(step_ledger),
            },
            "assessment.json.gz": {
                "schema_version": ASSESSMENT_LEDGER_SCHEMA_VERSION,
                "sha256": sha256_bytes(assessment_ledger),
            },
        },
        "adjudication": {
            "independent_exact_l2_route_admitted": True,
            "current_l3_procurement_closed": (terminal_state is AvailabilityState.CURRENT_CLOSED),
            "route_complete": value.route_complete,
            "family_template_admitted": False,
            "homologue_template_admitted": False,
            "conflicted_source_admitted": False,
            "coverage_update_authorized": False,
            "synthesis_guidance_authorized": False,
            "holdout_reveal_authorized": False,
        },
        "safe_claim": (
            "An independent, exact four-step route to hexadec-15-ynal is uniquely "
            "forward consistent for the declared substrate/product pairs and closes "
            "to two current exact-item terminal-material observations."
        ),
        "nonclaims": [
            "Exact graph reconstruction is not experimental synthesis success.",
            "The four exact pairs do not establish family-wide or homologue-wide scope.",
            "Current terminal procurement closure does not transfer to analogues.",
            "WO 2024/073486 A2 Example 2 Step 1 is rejected because its carbon accounting is internally inconsistent.",
            "This audit does not update route coverage or authorize synthesis-guided selection.",
        ],
    }
    return result, step_ledger, assessment_ledger


def load_exact_c16_route_overlay(
    *,
    base_source: RouteKnowledgeSource,
    config_path: Path,
    input_paths: Mapping[str, Path],
    stored_result_path: Path,
    stored_step_ledger_path: Path,
    stored_assessment_path: Path,
) -> ExactC16RouteOverlay:
    """Reproduce the frozen audit before exposing its exact route overlay."""

    fresh, steps, assessment = build_exact_c16_route_audit(
        config_path=config_path,
        input_paths=input_paths,
        base_source=base_source,
    )
    stored = _load_json(stored_result_path, label="stored exact C16 route result")
    if fresh != stored:
        raise Ugi3ExactC16RouteError("stored exact C16 route result is not reproducible")
    if stored_step_ledger_path.read_bytes() != steps:
        raise Ugi3ExactC16RouteError("stored exact C16 step ledger is not reproducible")
    if stored_assessment_path.read_bytes() != assessment:
        raise Ugi3ExactC16RouteError("stored exact C16 assessment is not reproducible")
    config = _load_json(config_path, label="exact C16 route config")
    overlay, _, _, _ = _build_overlay_and_rows(
        config=config,
        input_paths=input_paths,
        base_source=base_source,
    )
    return overlay

"""Qualify one exact C18 aldehyde route without promoting family scope.

The evidence chain is deliberately narrow: stearolic acid is reduced to the
internal alkynol, the exact C18 chain undergoes a reported alkyne zipper, and
the terminal alkynol is oxidized to octadec-17-ynal.  Every graph transform is
bound to that one reported substrate/product pair.  Current terminal-material
closure remains an independent, timestamped L3 decision.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

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

CONFIG_SCHEMA_VERSION = "phase1_ugi3_exact_c18_route_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_exact_c18_route_audit.v1"
STEP_LEDGER_SCHEMA_VERSION = "phase1_ugi3_exact_c18_route_steps.v1"
ASSESSMENT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_exact_c18_assessment.v1"


class Ugi3ExactC18RouteError(ValueError):
    """Raised when the exact C18 evidence chain violates its frozen contract."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _gzip_json_bytes(value: Any) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write((_stable_json(value) + "\n").encode())
    return output.getvalue()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise Ugi3ExactC18RouteError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3ExactC18RouteError(f"{label} must be a JSON object")
    return value


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Ugi3ExactC18RouteError(f"{label} must be ISO-8601 UTC ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Ugi3ExactC18RouteError(f"{label} is not valid ISO-8601 UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise Ugi3ExactC18RouteError(f"{label} must be UTC")
    return parsed.astimezone(timezone.utc)


def _canonical(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise Ugi3ExactC18RouteError(f"{label} must be nonempty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3ExactC18RouteError(f"{label} contains invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _validate_inputs(config: Mapping[str, Any], input_paths: Mapping[str, Path]) -> dict[str, str]:
    declared = config.get("inputs")
    if not isinstance(declared, dict) or set(declared) != set(input_paths):
        raise Ugi3ExactC18RouteError("configured and supplied C18 inputs differ")
    hashes: dict[str, str] = {}
    for label, path in input_paths.items():
        record = declared[label]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise Ugi3ExactC18RouteError(f"input {label} is malformed")
        observed = sha256_file(path)
        if observed != record["sha256"]:
            raise Ugi3ExactC18RouteError(f"input hash changed: {label}")
        hashes[label] = observed
    return hashes


def _validate_expected(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise Ugi3ExactC18RouteError(f"{label} fields changed")
        for key, value in expected.items():
            _validate_expected(observed[key], value, label=f"{label}.{key}")
        return
    if observed != expected:
        raise Ugi3ExactC18RouteError(
            f"{label} changed: expected {expected!r}, observed {observed!r}"
        )


@dataclass(frozen=True)
class _ExactStep:
    step_id: str
    reaction_id: str
    reactant: RouteTarget
    product: RouteTarget
    evidence: EvidenceRecord


class _MissingSource:
    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        return KnowledgeResult(
            disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
            evidence=(),
            detail="base source contains no exact route for this target",
        )


class ExactC18RouteOverlay:
    """Override one exact route chain and delegate every other identity."""

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
        canonical = _canonical(target.canonical_smiles, label="C18 overlay target")
        selected = self._decisions.get((target.role, canonical))
        if selected is not None:
            return selected
        return self._base_source.lookup(target)


def _validate_registry(config: Mapping[str, Any], registry: Mapping[str, Any]) -> None:
    if registry.get("scope") != "one_exact_c18_substrate_chain_only":
        raise Ugi3ExactC18RouteError("C18 registry has broader-than-authorized scope")
    claims = registry.get("claims_boundary")
    required_false = {
        "qualified_for_general_enumeration",
        "qualified_for_substrate_scope_extrapolation",
        "qualified_as_a_reaction_family",
        "conditions_encoded_in_graph_transform",
        "exact_graph_reconstruction_is_experimental_success",
        "exact_l2_evidence_implies_current_l3_procurement",
    }
    if not isinstance(claims, dict) or any(claims.get(key) is not False for key in required_false):
        raise Ugi3ExactC18RouteError("C18 registry lacks fail-closed claim boundaries")
    reactions = registry.get("reactions")
    steps = config.get("steps")
    if not isinstance(reactions, list) or not isinstance(steps, list):
        raise Ugi3ExactC18RouteError("C18 registry or step list is malformed")
    by_id = {
        reaction.get("reaction_id"): reaction
        for reaction in reactions
        if isinstance(reaction, dict) and isinstance(reaction.get("reaction_id"), str)
    }
    if len(by_id) != len(reactions) or len(reactions) != len(steps):
        raise Ugi3ExactC18RouteError("C18 reaction identifiers are not one-to-one")
    for step in steps:
        if not isinstance(step, dict):
            raise Ugi3ExactC18RouteError("C18 route step is malformed")
        definition = by_id.get(step.get("reaction_id"))
        if (
            not isinstance(definition, dict)
            or definition.get("status") != "qualified_for_one_exact_substrate_product_pair_only"
            or definition.get("qualification_pair_count") != 1
            or _canonical(definition.get("exact_reactant_smiles"), label="registry reactant")
            != _canonical(step.get("reactant_smiles"), label="step reactant")
            or _canonical(definition.get("exact_product_smiles"), label="registry product")
            != _canonical(step.get("product_smiles"), label="step product")
        ):
            raise Ugi3ExactC18RouteError(
                f"reaction {step.get('reaction_id')!r} is not exact-pair qualified"
            )


def _validate_scope_limitations(config: Mapping[str, Any]) -> None:
    if config.get("scope") != "one_exact_c18_route_only":
        raise Ugi3ExactC18RouteError("C18 config has broader-than-authorized scope")
    limitations = config.get("limitations")
    required = {
        "exact_substrate_product_pairs_only": True,
        "reaction_family_promotion_authorized": False,
        "substrate_scope_extrapolation_authorized": False,
        "analogue_promotion_authorized": False,
        "homologue_promotion_authorized": False,
        "general_enumeration_authorized": False,
        "terminal_closure_transfers_to_analogues": False,
    }
    if limitations != required:
        raise Ugi3ExactC18RouteError("C18 config lacks exact-pair-only limitations")


def _validate_procurement_policy(config: Mapping[str, Any]) -> tuple[str, int]:
    expected = {
        "expiry_days": 30,
        "status_semantics": "current_as_of_frozen_assessment_time",
        "refresh_required_before_candidate_lock": True,
    }
    if config.get("procurement_policy") != expected:
        raise Ugi3ExactC18RouteError("C18 procurement policy changed")
    assessment_as_of_utc = config.get("assessment_as_of_utc")
    _parse_utc(assessment_as_of_utc, label="assessment_as_of_utc")
    return str(assessment_as_of_utc), expected["expiry_days"]


def _validate_contiguous_chain(steps: list[Any], molecule_smiles: Mapping[str, str]) -> None:
    for index, (left, right) in enumerate(zip(steps, steps[1:], strict=False), start=1):
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise Ugi3ExactC18RouteError("C18 route step is malformed")
        left_product = left.get("product_id")
        right_reactant = right.get("reactant_id")
        if left_product != right_reactant:
            raise Ugi3ExactC18RouteError(
                f"C18 route is not contiguous between steps {index} and {index + 1}"
            )
        if molecule_smiles.get(str(left_product)) != molecule_smiles.get(str(right_reactant)):
            raise Ugi3ExactC18RouteError(
                f"C18 route identity changed between steps {index} and {index + 1}"
            )


def _terminal_state(terminal: Mapping[str, Any]) -> AvailabilityState:
    if terminal.get("identity_exact") is not True:
        raise Ugi3ExactC18RouteError("C18 terminal identity is not exact")
    purity = terminal.get("item_specific_purity")
    if not isinstance(purity, str) or not purity:
        raise Ugi3ExactC18RouteError("C18 terminal lacks item-specific purity")
    stock = terminal.get("explicit_current_stock_or_shipping_observed")
    closed = terminal.get("current_item_level_procurement_closed")
    if stock is True and closed is True:
        return AvailabilityState.CURRENT_CLOSED
    if stock is False and closed is False:
        return AvailabilityState.UNASSESSED
    raise Ugi3ExactC18RouteError(
        "C18 terminal closure is inconsistent with explicit stock/shipping evidence"
    )


def _terminal_evidence(
    *,
    terminal: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    terminal_state: AvailabilityState,
    terminal_smiles: str,
    terminal_inchi_key: str,
    assessment_as_of_utc: str,
    expiry_days: int,
) -> tuple[EvidenceRecord, ...]:
    observation_input = terminal.get("observation_source_input")
    if not isinstance(observation_input, str) or observation_input not in input_paths:
        raise Ugi3ExactC18RouteError("C18 terminal lacks a pinned observation source")
    observation = _load_json(
        input_paths[observation_input], label="C18 terminal vendor observation"
    )
    item = observation.get("item")
    if (
        observation.get("schema_version") != "forge.vendor_product_observation.v1"
        or not isinstance(item, dict)
        or observation.get("vendor") != terminal.get("vendor")
        or observation.get("official_product_url") != terminal.get("product_url")
        or item.get("catalog_number") != terminal.get("item_id")
        or item.get("cas_number") != terminal.get("cas_number")
        or _canonical(item.get("canonical_smiles"), label="vendor-observation terminal")
        != terminal_smiles
        or item.get("inchi_key") != terminal_inchi_key
        or observation.get("identity_exact") is not terminal.get("identity_exact")
        or observation.get("item_specific_purity") != terminal.get("item_specific_purity")
        or observation.get("explicit_current_stock_or_shipping_observed")
        is not terminal.get("explicit_current_stock_or_shipping_observed")
        or observation.get("current_item_level_procurement_closed")
        is not terminal.get("current_item_level_procurement_closed")
        or observation.get("observed_at_utc") != terminal.get("observed_at_utc")
        or observation.get("expires_at_utc") != terminal.get("expires_at_utc")
    ):
        raise Ugi3ExactC18RouteError(
            "C18 terminal config disagrees with the pinned vendor observation"
        )
    observed_at = _parse_utc(observation.get("observed_at_utc"), label="terminal observed_at_utc")
    expires_at = _parse_utc(observation.get("expires_at_utc"), label="terminal expires_at_utc")
    assessment_as_of = _parse_utc(assessment_as_of_utc, label="assessment_as_of_utc")
    if expires_at != observed_at + timedelta(days=expiry_days):
        raise Ugi3ExactC18RouteError(
            "C18 terminal expiry does not match the frozen procurement policy"
        )
    if not observed_at <= assessment_as_of <= expires_at:
        raise Ugi3ExactC18RouteError("C18 assessment time is outside the vendor observation window")

    raw_evidence = terminal.get("evidence_records")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise Ugi3ExactC18RouteError("C18 terminal lacks evidence records")
    records: list[EvidenceRecord] = []
    evidence_ids: set[str] = set()
    observed_current_evidence = False
    for raw in raw_evidence:
        if not isinstance(raw, dict):
            raise Ugi3ExactC18RouteError("C18 terminal evidence record is malformed")
        evidence_id = raw.get("evidence_id")
        source_input = raw.get("source_input")
        source_locator = raw.get("source_locator")
        supports_current_availability = raw.get("supports_current_availability")
        if (
            not isinstance(evidence_id, str)
            or not evidence_id
            or evidence_id in evidence_ids
            or not isinstance(source_input, str)
            or source_input not in input_paths
            or not isinstance(source_locator, str)
            or not source_locator
            or not isinstance(supports_current_availability, bool)
        ):
            raise Ugi3ExactC18RouteError("C18 terminal evidence record is malformed")
        if supports_current_availability and source_input != observation_input:
            raise Ugi3ExactC18RouteError(
                "C18 terminal availability must point to the pinned observation"
            )
        availability = (
            terminal_state if supports_current_availability else AvailabilityState.UNASSESSED
        )
        observed_current_evidence |= supports_current_availability
        evidence_ids.add(evidence_id)
        records.append(
            EvidenceRecord(
                evidence_id=evidence_id,
                tier=EvidenceTier.ACCEPTED_TERMINAL,
                source_sha256=sha256_file(input_paths[source_input]),
                source_locator=source_locator,
                exact_substrate=True,
                forward_verification=ForwardVerificationState.NOT_APPLICABLE,
                availability=availability,
            )
        )
    if terminal_state is AvailabilityState.CURRENT_CLOSED and not observed_current_evidence:
        raise Ugi3ExactC18RouteError("C18 terminal closure lacks current availability evidence")
    return tuple(records)


def _build_overlay_and_rows(
    *,
    config: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    base_source: RouteKnowledgeSource,
) -> tuple[ExactC18RouteOverlay, list[dict[str, Any]], AvailabilityState]:
    registry = _load_json(input_paths["reaction_registry"], label="C18 reaction registry")
    _validate_registry(config, registry)
    assessment_as_of_utc, expiry_days = _validate_procurement_policy(config)
    molecules = config.get("molecules")
    steps = config.get("steps")
    if not isinstance(molecules, dict) or not isinstance(steps, list) or len(steps) != 3:
        raise Ugi3ExactC18RouteError("C18 molecule or step records are malformed")

    molecule_smiles: dict[str, str] = {}
    for molecule_id, record in molecules.items():
        if not isinstance(molecule_id, str) or not isinstance(record, dict):
            raise Ugi3ExactC18RouteError("C18 molecule identity is malformed")
        canonical = _canonical(record.get("canonical_smiles"), label=molecule_id)
        molecule = Chem.MolFromSmiles(canonical)
        assert molecule is not None
        if Chem.MolToInchiKey(molecule) != record.get("inchi_key"):
            raise Ugi3ExactC18RouteError(f"C18 molecule InChIKey changed: {molecule_id}")
        molecule_smiles[molecule_id] = canonical

    _validate_contiguous_chain(steps, molecule_smiles)

    exact_steps: list[_ExactStep] = []
    rows: list[dict[str, Any]] = []
    trials = config.get("verification_policy", {}).get("randomized_smiles_trials")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise Ugi3ExactC18RouteError("randomized SMILES trial count must be positive")
    for index, raw in enumerate(steps):
        if not isinstance(raw, dict):
            raise Ugi3ExactC18RouteError("C18 route step is malformed")
        reactant_id = raw.get("reactant_id")
        product_id = raw.get("product_id")
        if molecule_smiles.get(str(reactant_id)) != _canonical(
            raw.get("reactant_smiles"), label="configured reactant"
        ) or molecule_smiles.get(str(product_id)) != _canonical(
            raw.get("product_smiles"), label="configured product"
        ):
            raise Ugi3ExactC18RouteError("C18 step and molecule identities disagree")
        variant_label = raw.get("variant_input")
        if not isinstance(variant_label, str) or variant_label not in input_paths:
            raise Ugi3ExactC18RouteError("C18 step lacks a pinned reaction variant")
        reaction_id = raw.get("reaction_id")
        if not isinstance(reaction_id, str):
            raise Ugi3ExactC18RouteError("C18 step reaction_id is malformed")
        try:
            compiled = load_qualified_forward_reaction(
                input_paths["reaction_registry"],
                input_paths[variant_label],
                reaction_id=reaction_id,
            )
            expected = molecule_smiles[str(product_id)]
            observed = unique_forward_products(
                compiled,
                [molecule_smiles[str(reactant_id)]],
                max_products=32,
                isomeric_smiles=False,
            )
            if observed != (expected,):
                raise Ugi3ExactC18RouteError(
                    f"C18 step {reaction_id} did not uniquely reconstruct its target"
                )
            successful_trials = 0
            reactant_molecule = Chem.MolFromSmiles(molecule_smiles[str(reactant_id)])
            assert reactant_molecule is not None
            for _ in range(trials):
                randomized = Chem.MolToSmiles(
                    reactant_molecule,
                    canonical=False,
                    doRandom=True,
                    isomericSmiles=False,
                )
                randomized_outputs = unique_forward_products(
                    compiled,
                    [randomized],
                    max_products=32,
                    isomeric_smiles=False,
                )
                successful_trials += int(randomized_outputs == (expected,))
        except QualifiedForwardError as exc:
            raise Ugi3ExactC18RouteError(
                f"C18 forward verification failed for {reaction_id}: {exc}"
            ) from exc
        if successful_trials != trials:
            raise Ugi3ExactC18RouteError(
                f"C18 step {reaction_id} is not invariant to randomized SMILES"
            )
        source_input = raw.get("primary_source_input")
        if not isinstance(source_input, str) or source_input not in input_paths:
            raise Ugi3ExactC18RouteError("C18 step lacks a pinned primary source")
        source_sha = sha256_file(input_paths[source_input])
        conditions = raw.get("conditions")
        conditions_source_input = raw.get("conditions_source_input")
        conditions_source_sha256 = raw.get("conditions_source_sha256")
        conditions_source_locator = raw.get("conditions_source_locator")
        if not isinstance(conditions, dict) or not conditions:
            raise Ugi3ExactC18RouteError("C18 step lacks structured conditions")
        if (
            not isinstance(conditions_source_input, str)
            or conditions_source_input not in input_paths
        ):
            raise Ugi3ExactC18RouteError("C18 step lacks a pinned conditions source")
        observed_conditions_sha = sha256_file(input_paths[conditions_source_input])
        if conditions_source_sha256 != observed_conditions_sha:
            raise Ugi3ExactC18RouteError("C18 step conditions source hash changed")
        if not isinstance(conditions_source_locator, str) or not conditions_source_locator:
            raise Ugi3ExactC18RouteError("C18 step lacks a conditions source locator")
        evidence = EvidenceRecord(
            evidence_id=f"exact-c18:{raw.get('step_id')}",
            tier=EvidenceTier.EXACT_SOURCE,
            source_sha256=source_sha,
            source_locator=str(raw.get("source_locator")),
            exact_substrate=True,
            forward_verification=ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE,
            availability=AvailabilityState.UNASSESSED,
        )
        reactant_target = RouteTarget(
            role=str(raw.get("reactant_role")),
            canonical_smiles=molecule_smiles[str(reactant_id)],
        )
        product_target = RouteTarget(
            role=str(raw.get("product_role")),
            canonical_smiles=expected,
        )
        exact_steps.append(
            _ExactStep(
                step_id=str(raw.get("step_id")),
                reaction_id=reaction_id,
                reactant=reactant_target,
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
                "reactant_id": reactant_id,
                "reactant_role": reactant_target.role,
                "reactant_smiles": reactant_target.canonical_smiles,
                "product_id": product_id,
                "product_role": product_target.role,
                "product_smiles": product_target.canonical_smiles,
                "source_input": source_input,
                "source_sha256": source_sha,
                "source_locator": raw.get("source_locator"),
                "conditions": conditions,
                "conditions_source_input": conditions_source_input,
                "conditions_source_sha256": observed_conditions_sha,
                "conditions_source_locator": conditions_source_locator,
                "isolated_yield_percent": raw.get("isolated_yield_percent"),
                "characterization": raw.get("characterization"),
                "forward_products": list(observed),
                "forward_product_count": len(observed),
                "expected_product_in_outputs": expected in observed,
                "verified_exact_product_unique": observed == (expected,),
                "randomized_smiles_trials": trials,
                "randomized_smiles_successes": successful_trials,
                "family_scope_admitted": False,
            }
        )

    terminal = config.get("terminal_procurement")
    if not isinstance(terminal, dict):
        raise Ugi3ExactC18RouteError("C18 terminal procurement record is malformed")
    terminal_state = _terminal_state(terminal)
    terminal_id = terminal.get("molecule_id")
    if molecule_smiles.get(str(terminal_id)) != _canonical(
        terminal.get("canonical_smiles"), label="C18 terminal"
    ):
        raise Ugi3ExactC18RouteError("C18 terminal identity does not match route root")
    terminal_molecule = Chem.MolFromSmiles(molecule_smiles[str(terminal_id)])
    assert terminal_molecule is not None
    terminal_evidence = _terminal_evidence(
        terminal=terminal,
        input_paths=input_paths,
        terminal_state=terminal_state,
        terminal_smiles=molecule_smiles[str(terminal_id)],
        terminal_inchi_key=Chem.MolToInchiKey(terminal_molecule),
        assessment_as_of_utc=assessment_as_of_utc,
        expiry_days=expiry_days,
    )
    decisions: dict[tuple[str, str], KnowledgeResult] = {}
    for step in exact_steps:
        proposal = RouteStepProposal(
            reaction_id=step.reaction_id,
            reactants=(step.reactant,),
            evidence=(step.evidence,),
            forward_product_count=1,
            verifier_calls_required=1,
            product_candidates_considered=1,
        )
        decisions[(step.product.role, step.product.canonical_smiles)] = KnowledgeResult(
            disposition=KnowledgeDisposition.EXPAND,
            evidence=(step.evidence,),
            proposal=proposal,
            detail="exact C18 substrate/product pair retrieved and uniquely verified",
        )
    terminal_target = exact_steps[0].reactant
    decisions[(terminal_target.role, terminal_target.canonical_smiles)] = KnowledgeResult(
        disposition=KnowledgeDisposition.TERMINAL,
        evidence=terminal_evidence,
        detail=(
            "exact C18 terminal has item-level identity and purity evidence; "
            "current availability is assessed independently"
        ),
    )
    overlay = ExactC18RouteOverlay(
        base_source=base_source,
        decisions=decisions,
        target=exact_steps[-1].product,
    )
    return overlay, rows, terminal_state


def build_exact_c18_route_audit(
    *,
    config_path: Path,
    input_paths: Mapping[str, Path],
    base_source: RouteKnowledgeSource | None = None,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Verify the exact chain and emit an evidence-preserving assessment."""

    config = _load_json(config_path, label="exact C18 route config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3ExactC18RouteError("unsupported exact C18 route config")
    _validate_scope_limitations(config)
    input_hashes = _validate_inputs(config, input_paths)
    overlay, rows, terminal_state = _build_overlay_and_rows(
        config=config,
        input_paths=input_paths,
        base_source=base_source or _MissingSource(),
    )
    limits = PlannerBudgetLimits.from_dict(config.get("planner_budget"))
    budget = PlannerBudgetLedger(limits)
    assessment = RecursiveRouteAssessor(overlay).assess(overlay.target, budget)
    value = component_synthesis_value_from_assessment(assessment)
    terminal = config["terminal_procurement"]
    summary = {
        "target_role": overlay.target.role,
        "target_smiles": overlay.target.canonical_smiles,
        "exact_steps": len(rows),
        "uniquely_forward_verified_steps": sum(
            row["verified_exact_product_unique"] for row in rows
        ),
        "randomized_smiles_trials_per_step": rows[0]["randomized_smiles_trials"],
        "randomized_smiles_total_successes": sum(
            row["randomized_smiles_successes"] for row in rows
        ),
        "terminal_availability": terminal_state.value,
        "terminal_observed_at_utc": terminal["observed_at_utc"],
        "terminal_expires_at_utc": terminal["expires_at_utc"],
        "assessment_as_of_utc": config["assessment_as_of_utc"],
        "assessment_outcome": assessment.outcome.value,
        "route_complete": value.route_complete,
        "route_step_count": value.route_step_count,
        "maximum_route_depth": value.maximum_route_depth,
        "unassessed_terminal_leaf_count": value.unassessed_terminal_leaf_count,
        "forward_consistency": value.forward_consistency.value,
        "evidence_support": value.evidence_support.value,
        "reaction_family_admitted": False,
        "development_product_coverage_changed": value.route_complete,
    }
    _validate_expected(summary, config.get("expected_summary"), label="C18 summary")
    step_ledger = _gzip_json_bytes({"schema_version": STEP_LEDGER_SCHEMA_VERSION, "rows": rows})
    assessment_ledger = _gzip_json_bytes(
        {
            "schema_version": ASSESSMENT_LEDGER_SCHEMA_VERSION,
            "assessment": assessment.to_dict(),
            "synthesis_value": value.to_dict(),
            "budget": budget.to_dict(),
            "assessment_as_of_utc": config["assessment_as_of_utc"],
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
            label: {
                "path": str(path),
                "sha256": input_hashes[label],
            }
            for label, path in sorted(input_paths.items())
        },
        "summary": summary,
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
            "exact_l2_chain_admitted": True,
            "current_l3_procurement_closed": terminal_state is AvailabilityState.CURRENT_CLOSED,
            "route_complete": value.route_complete,
            "family_template_admitted": False,
            "synthesis_guidance_authorized": False,
            "holdout_reveal_authorized": False,
        },
        "safe_claim": (
            "An exact, three-step, historically executed and product-characterized "
            "route to octadec-17-ynal is uniquely forward consistent for the three "
            "declared substrate/product pairs and, at the frozen assessment time, "
            "closes to a current exact-item stearolic-acid procurement observation."
        ),
        "nonclaims": [
            "Exact graph reconstruction is not experimental synthesis success.",
            "The three exact pairs do not establish family-wide substrate scope.",
            "Current procurement closure for stearolic acid does not transfer to analogues.",
            "Current procurement status is time-pinned and must be refreshed before candidate lock.",
            "L2 evidence does not authorize synthesis-guided selection or holdout reveal.",
        ],
    }
    return result, step_ledger, assessment_ledger


def load_exact_c18_route_overlay(
    *,
    base_source: RouteKnowledgeSource,
    config_path: Path,
    input_paths: Mapping[str, Path],
    stored_result_path: Path,
    stored_step_ledger_path: Path,
    stored_assessment_path: Path,
) -> ExactC18RouteOverlay:
    """Reproduce the frozen audit before exposing its exact route overlay."""

    fresh, steps, assessment = build_exact_c18_route_audit(
        config_path=config_path,
        input_paths=input_paths,
        base_source=base_source,
    )
    stored = _load_json(stored_result_path, label="stored exact C18 route result")
    if fresh != stored:
        raise Ugi3ExactC18RouteError("stored exact C18 route result is not reproducible")
    if stored_step_ledger_path.read_bytes() != steps:
        raise Ugi3ExactC18RouteError("stored exact C18 step ledger is not reproducible")
    if stored_assessment_path.read_bytes() != assessment:
        raise Ugi3ExactC18RouteError("stored exact C18 assessment is not reproducible")
    config = _load_json(config_path, label="exact C18 route config")
    overlay, _, _ = _build_overlay_and_rows(
        config=config,
        input_paths=input_paths,
        base_source=base_source,
    )
    return overlay

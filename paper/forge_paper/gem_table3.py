"""Render GEM Table 3 from a route-blind sample of final Transformer products."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_json
from forge.core.io import atomic_write, iter_jsonl, read_json_object, write_json, write_jsonl
from forge.synthesis.assessment.common_route_evidence import load_frozen_component_evidence

CONFIG_SCHEMA = "forge.gem_table3_route_assessment_config.v1"
RESULT_SCHEMA = "forge.gem_table3_route_assessment.v1"
LEDGER_SCHEMA = "forge.gem_table3_route_assessed_products.v1"
COMPONENT_LEDGER_SCHEMA = "forge.gem_table3_route_assessed_components.v1"
ASSESSMENT_LEDGER_SCHEMA = "forge.common_ugi_assessed_attempts.v1"
EVIDENCE_RESULT_SCHEMA = "forge.common_ugi_method_blind_route_evidence.v1"


class GemTable3Error(ValueError):
    """The final-product population or its frozen route evidence is inadmissible."""


def _load_assessed(path: Path, *, method_id: str, seed: int) -> list[dict[str, Any]]:
    records = list(iter_jsonl(path))
    expected_header = {"schema_version": ASSESSMENT_LEDGER_SCHEMA, "rows": len(records) - 1}
    if not records or records[0] != expected_header:
        raise GemTable3Error(f"common assessed-attempt ledger changed: {path}")
    rows = []
    for raw in records[1:]:
        if (
            not isinstance(raw, Mapping)
            or raw.get("method_id") != method_id
            or raw.get("seed") != seed
            or isinstance(raw.get("attempt_index"), bool)
            or not isinstance(raw.get("attempt_index"), int)
        ):
            raise GemTable3Error(f"common assessed-attempt identity changed: {path}")
        rows.append(dict(raw))
    return rows


def _trace_key(trace: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    components = trace.get("components_by_role")
    if not isinstance(components, Mapping) or not components:
        raise GemTable3Error("exact-L1 trace omits its component mapping")
    pairs = []
    for role, raw_smiles in sorted(components.items()):
        values = raw_smiles if isinstance(raw_smiles, list) else [raw_smiles]
        if not isinstance(role, str) or not role or not values:
            raise GemTable3Error("exact-L1 trace has a malformed role")
        for smiles in values:
            if not isinstance(smiles, str) or not smiles:
                raise GemTable3Error("exact-L1 trace has a malformed component")
            pairs.append((role, smiles))
    return tuple(pairs)


def _eligible(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("status") == "generated"
        and row.get("valid") is True
        and row.get("exact_l1_program") is True
        and row.get("exact_l1_trace_count") == 1
        and row.get("forward_verified_trace_count") == 1
        and isinstance(row.get("canonical_smiles"), str)
        and row.get("canonical_smiles")
    )


def _select_products(
    rows: list[dict[str, Any]], *, sample_size: int, salt: str
) -> list[dict[str, Any]]:
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _eligible(row):
            occurrences[str(row["canonical_smiles"])].append(row)
    if len(occurrences) < sample_size:
        raise GemTable3Error(
            f"only {len(occurrences)} distinct eligible products for sample size {sample_size}"
        )
    products = []
    for canonical_smiles, product_rows in occurrences.items():
        trace_options: dict[tuple[tuple[str, str], ...], Mapping[str, Any]] = {}
        source_occurrences = []
        for row in product_rows:
            traces = row.get("exact_l1_traces")
            if (
                not isinstance(traces, list)
                or len(traces) != 1
                or not isinstance(traces[0], Mapping)
            ):
                raise GemTable3Error("eligible product has a malformed exact-L1 trace")
            trace_options[_trace_key(traces[0])] = traces[0]
            source_occurrences.append(
                {"seed": int(row["seed"]), "attempt_index": int(row["attempt_index"])}
            )
        selected_key = min(trace_options)
        selection_digest = str(sha256_json({"salt": salt, "canonical_smiles": canonical_smiles}))
        products.append(
            {
                "canonical_smiles": canonical_smiles,
                "selection_digest": selection_digest,
                "selected_components": [
                    {"role": role, "canonical_smiles": smiles} for role, smiles in selected_key
                ],
                "exact_trace_variants_across_occurrences": len(trace_options),
                "source_occurrences": sorted(
                    source_occurrences, key=lambda item: (item["seed"], item["attempt_index"])
                ),
            }
        )
    products.sort(key=lambda item: (item["selection_digest"], item["canonical_smiles"]))
    selected = products[:sample_size]
    for rank, product in enumerate(selected, start=1):
        product["assessment_rank"] = rank
    return selected


def _assess_products(
    selected: list[dict[str, Any]],
    *,
    evidence: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Counter[str]]]:
    product_states: Counter[str] = Counter()
    components: dict[tuple[str, str], dict[str, Any]] = {}
    assessed = []
    for product in selected:
        component_rows = []
        for component in product["selected_components"]:
            key = (component["role"], component["canonical_smiles"])
            disposition = evidence.get(key)
            if disposition is None:
                raise GemTable3Error(f"selected component is absent from frozen union: {key}")
            state = str(disposition["final_component_state"])
            component_row = {
                **component,
                "final_component_state": state,
                "strict_complete": bool(disposition["strict_complete"]),
                "verified_upstream": bool(disposition["verified_upstream"]),
                "terminal_evidence": bool(disposition["terminal_evidence"]),
                "assessment_outcome": disposition.get("assessment_outcome"),
            }
            component_rows.append(component_row)
            prior = components.get(key)
            if prior is not None and prior != component_row:
                raise GemTable3Error(f"component disposition changed within assessment: {key}")
            components[key] = component_row
        states = {row["final_component_state"] for row in component_rows}
        if states == {"complete"}:
            product_state = "complete"
        elif "search_censored" in states:
            product_state = "search_censored"
        else:
            product_state = "unresolved"
        product_states[product_state] += 1
        assessed.append(
            {
                **product,
                "component_assessments": component_rows,
                "final_product_state": product_state,
            }
        )
    component_states = Counter(row["final_component_state"] for row in components.values())
    component_rows = [components[key] for key in sorted(components)]
    return assessed, component_rows, {"products": product_states, "components": component_states}


def _macro_text(summary: Mapping[str, Any]) -> str:
    values = {
        "ForgeRouteCandidates": summary["products"],
        "ForgeRouteExactLone": summary["products"],
        "ForgeRouteAdmitted": summary["products"],
        "ForgeRouteCompleteProducts": summary["product_states"].get("complete", 0),
        "ForgeRouteUnresolvedProducts": summary["product_states"].get("unresolved", 0),
        "ForgeRouteCensoredProducts": summary["product_states"].get("search_censored", 0),
        "ForgeRouteComponents": summary["components"],
        "ForgeRouteCompleteComponents": summary["component_states"].get("complete", 0),
        "ForgeRouteUnresolvedComponents": summary["component_states"].get("unresolved", 0),
        "ForgeRouteCensoredComponents": summary["component_states"].get("search_censored", 0),
    }
    eligible = summary["eligible_distinct_products_before_sampling"]
    overrides = "".join(
        f"\\renewcommand{{\\{name}}}{{{value}}}\n" for name, value in values.items()
    )
    return f"\\newcommand{{\\ForgeRouteEligiblePool}}{{{eligible}}}\n{overrides}"


def render_gem_table3_route_assessment(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Assess and render the final Transformer route-disposition table."""

    config = read_json_object(config_path, error=GemTable3Error, label="GEM Table 3 config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "frozen_before_route_blind_assessment"
        or set(config) != {"schema_version", "status", "inputs", "population", "policy"}
    ):
        raise GemTable3Error("GEM Table 3 config changed")
    inputs = config.get("inputs")
    population = config.get("population")
    policy = config.get("policy")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(population, Mapping)
        or not isinstance(policy, Mapping)
    ):
        raise GemTable3Error("GEM Table 3 config sections are malformed")
    expected_policy = {
        "route_or_oracle_calls": 0,
        "family_projection_can_close": False,
        "proposal_only_route_can_close": False,
        "missing_evidence_is_unsynthesizable": False,
        "complete_requires_every_component_strict_complete": True,
        "candidate_selection": False,
    }
    if dict(policy) != expected_policy:
        raise GemTable3Error("GEM Table 3 route policy changed")
    pins = inputs.get("assessed_attempts")
    seeds = population.get("seeds")
    method_id = population.get("method_id")
    if not isinstance(pins, list) or not isinstance(seeds, list) or len(pins) != len(seeds):
        raise GemTable3Error("GEM Table 3 assessed inputs changed")
    rows = []
    source_records = []
    for seed, pin in zip(seeds, pins, strict=True):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise GemTable3Error("GEM Table 3 seed is malformed")
        path = resolve_pin(pin, repo, label=f"GEM Table 3 assessed attempts seed {seed}")
        rows.extend(_load_assessed(path, method_id=str(method_id), seed=seed))
        source_records.append(pin_record(path, repo))
    evidence_result_path = resolve_pin(
        inputs.get("route_evidence_result"), repo, label="GEM Table 3 route evidence result"
    )
    evidence_path = resolve_pin(
        inputs.get("component_evidence"), repo, label="GEM Table 3 component evidence"
    )
    evidence_result = read_json_object(
        evidence_result_path, error=GemTable3Error, label="GEM Table 3 route evidence result"
    )
    evidence_gates = evidence_result.get("gates")
    expected_evidence_gates = {
        "public_worklist_only": True,
        "private_membership_read": False,
        "every_union_component_dispositioned": True,
        "family_projection_cannot_close": True,
        "proposal_only_route_cannot_close": True,
        "unknown_components_explicitly_abstain": True,
    }
    if (
        evidence_result.get("schema_version") != EVIDENCE_RESULT_SCHEMA
        or evidence_result.get("status") != "complete_with_explicit_abstentions"
        or evidence_result.get("candidate_selection") is not False
        or evidence_result.get("component_evidence", {}).get("sha256")
        != str(pin_record(evidence_path, repo)["sha256"])
        or evidence_gates != expected_evidence_gates
    ):
        raise GemTable3Error("GEM Table 3 route evidence is inadmissible")
    evidence = load_frozen_component_evidence(evidence_path)
    sample_size = population.get("sample_size")
    salt = population.get("salt")
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size <= 0
        or not isinstance(salt, str)
        or not salt
    ):
        raise GemTable3Error("GEM Table 3 population contract is malformed")
    selected = _select_products(rows, sample_size=sample_size, salt=salt)
    assessed, components, states = _assess_products(selected, evidence=evidence)
    summary = {
        "products": len(assessed),
        "components": len(components),
        "product_states": dict(sorted(states["products"].items())),
        "component_states": dict(sorted(states["components"].items())),
        "eligible_distinct_products_before_sampling": len(
            {str(row["canonical_smiles"]) for row in rows if _eligible(row)}
        ),
        "trace_ambiguous_selected_products": sum(
            int(row["exact_trace_variants_across_occurrences"] > 1) for row in assessed
        ),
    }
    if sum(summary["product_states"].values()) != sample_size:
        raise GemTable3Error("GEM Table 3 product dispositions do not preserve the denominator")
    if sum(summary["component_states"].values()) != len(components):
        raise GemTable3Error("GEM Table 3 component dispositions do not preserve the denominator")

    result_path.parent.mkdir(parents=True, exist_ok=True)
    product_ledger = result_path.parent / "product_route_ledger.jsonl.gz"
    component_ledger = result_path.parent / "component_route_ledger.jsonl.gz"
    write_jsonl(
        product_ledger,
        [{"schema_version": LEDGER_SCHEMA, "rows": len(assessed)}, *assessed],
    )
    write_jsonl(
        component_ledger,
        [{"schema_version": COMPONENT_LEDGER_SCHEMA, "rows": len(components)}, *components],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    macro_path = output_dir / "gem_table3_route_macros.tex"
    atomic_write(macro_path, _macro_text(summary).encode("utf-8"))
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "config": pin_record(config_path, repo),
        "inputs": {
            "assessed_attempts": source_records,
            "route_evidence_result": pin_record(evidence_result_path, repo),
            "component_evidence": pin_record(evidence_path, repo),
        },
        "population": dict(population),
        "summary": summary,
        "artifacts": {
            "product_route_ledger": artifact_record(product_ledger),
            "component_route_ledger": artifact_record(component_ledger),
            "latex_macros": artifact_record(macro_path),
        },
        "gates": {
            "all_selected_products_valid_unique_exact_l1": True,
            "all_selected_components_in_blind_union": True,
            "route_evidence_did_not_affect_sampling": True,
            "planner_and_oracle_calls_zero": True,
            "family_and_proposal_only_evidence_cannot_close": True,
            "product_and_component_denominators_preserved": True,
        },
        "nonclaims": [
            "An unresolved disposition is missing evidence, not proof of unsynthesizability.",
            "A complete computational dossier is not experimental synthesis success.",
            "The historical frozen terminal-evidence snapshot is not current procurement evidence.",
        ],
        "candidate_selection": False,
    }
    write_json(result_path, result)
    return result


__all__ = ["GemTable3Error", "render_gem_table3_route_assessment"]

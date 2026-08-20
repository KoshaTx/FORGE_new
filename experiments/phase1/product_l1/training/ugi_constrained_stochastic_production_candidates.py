"""Contracts for the fully stochastic matched Ugi production candidate pool.

The molecular flow is sampled once per frozen morphology coordinate.  Raw
outcomes are never repaired or retried.  Exact terminal chemical admission is
recorded as a separate field so raw generator performance and the constrained
candidate distribution remain simultaneously auditable.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.corpus.ugi_component_expansion import reaction_handle_qualification
from forge.potency.annotations import ROLE_NAMES

ARMS = ("broad_prior", "support_enriched")
CONFIG_SCHEMA_VERSION = "phase1_ugi_constrained_stochastic_production_candidates_config.v2"
SCHEDULE_SCHEMA_VERSION = "forge.ugi_production_candidate_schedule.v1"
LEDGER_SCHEMA_VERSION = "forge.ugi_constrained_stochastic_production_candidate_ledger.v2"
RESULT_SCHEMA_VERSION = "phase1_ugi_constrained_stochastic_production_candidates.v2"


class UgiConstrainedStochasticProductionCandidatesError(RuntimeError):
    """Raised when the frozen stochastic production contract changes."""


def stable_sha256(value: Any) -> str:
    """Return a deterministic SHA-256 for a JSON-compatible value."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    """Load one JSON object with an actionable failure."""

    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiConstrainedStochasticProductionCandidatesError(
            f"invalid {label}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise UgiConstrainedStochasticProductionCandidatesError(f"{label} must contain one object")
    return value


def branch_class(program: Mapping[str, Any]) -> str:
    """Classify role-level tail branching without interpreting tree leaves."""

    budgets = tuple(int(value) for value in program["junction_budgets"])
    if len(budgets) != len(ROLE_NAMES):
        raise UgiConstrainedStochasticProductionCandidatesError(
            "morphology program has the wrong number of role budgets"
        )
    aldehyde = budgets[ROLE_NAMES.index("oxoester_aldehyde_body_tail")] > 0
    isocyanide = budgets[ROLE_NAMES.index("isocyanide_tail")] > 0
    if aldehyde and isocyanide:
        return "both_tail_origins_branched"
    if aldehyde:
        return "aldehyde_origin_branched"
    if isocyanide:
        return "isocyanide_origin_branched"
    return "linear_tail_origins"


def program_shard(
    schedule: Mapping[str, Any],
    *,
    arm: str,
    shard_index: int,
    shard_draws: int,
) -> dict[str, Any]:
    """Materialize one sampler-compatible program shard from the frozen schedule."""

    if arm not in ARMS or shard_index < 0 or shard_draws < 1:
        raise UgiConstrainedStochasticProductionCandidatesError(
            "invalid production program-shard request"
        )
    if (
        schedule.get("schema_version") != SCHEDULE_SCHEMA_VERSION
        or schedule.get("status") != "frozen_before_production_candidate_generation"
    ):
        raise UgiConstrainedStochasticProductionCandidatesError("production schedule is not frozen")
    records = schedule.get("records")
    if not isinstance(records, list):
        raise UgiConstrainedStochasticProductionCandidatesError(
            "production schedule records are malformed"
        )
    start = shard_index * shard_draws
    stop = start + shard_draws
    selected = records[start:stop]
    if len(selected) != shard_draws:
        raise UgiConstrainedStochasticProductionCandidatesError(
            "production schedule does not contain the requested shard"
        )
    samples = []
    for record in selected:
        draw_index = int(record["draw_index"])
        arm_record = record["arms"][arm]
        program = dict(arm_record["program"])
        samples.append(
            {
                "product_id": f"production-v2-{arm}-{draw_index:05d}",
                "source_stratum": f"frozen_{arm}_morphology_proposal",
                "branch_class": branch_class(program),
                "component_novelty_class": "generated_complete_components",
                "program": program,
            }
        )
    return {
        "schema_version": "phase1_ugi_program_probe.v1",
        "fold": "frozen_production_candidate_schedule_v2",
        "seed": int(schedule["design"]["common_uniform_seed"]),
        "input_prior": {
            "schedule_sha256": str(schedule["schedule_sha256"]),
            "arm": arm,
            "shard_index": shard_index,
        },
        "samples": samples,
        "stratum_branch_counts": dict(
            sorted(Counter(sample["branch_class"] for sample in samples).items())
        ),
    }


def exact_terminal_admission(terminal: Mapping[str, Any], reaction: Any) -> dict[str, Any]:
    """Evaluate exact chemistry without modifying the generated terminal."""

    verification = terminal.get("l1_forward_verification")
    exact_l1 = bool(
        terminal.get("valid") is True
        and terminal.get("terminal_valid") is True
        and terminal.get("component_reconstruction_valid") is True
        and isinstance(verification, Mapping)
        and verification.get("exact_product_reconstructed") is True
        and verification.get("maximum_outcomes_saturated") is False
    )
    output: dict[str, Any] = {
        "admitted": False,
        "reason": "invalid_or_nonexact_l1",
        "raw_molecule_valid": terminal.get("valid") is True,
        "exact_l1": exact_l1,
        "handles_by_role": {},
    }
    if not exact_l1:
        return output
    components = terminal.get("component_smiles_by_role")
    if not isinstance(components, Mapping) or set(components) != set(ROLE_NAMES):
        output["reason"] = "component_identity_missing"
        return output
    all_handles = True
    failed_roles = []
    for role_index, role in enumerate(ROLE_NAMES):
        from rdkit import Chem, rdBase

        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(str(components[role]))
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            qualification = {
                "passes_registry_handle_policy": False,
                "reason": "invalid_component",
            }
        else:
            qualification = reaction_handle_qualification(
                molecule,
                query=reaction.handles[role_index],
                forbidden=reaction.forbidden[role_index],
                allowed_site_multiplicity=(
                    reaction.definition.reactant_roles[role_index].allowed_site_multiplicity
                ),
            )
        passed = bool(qualification["passes_registry_handle_policy"])
        output["handles_by_role"][role] = {
            "passes_registry_handle_policy": passed,
            "reason": qualification.get("reason"),
            "matching_sites": qualification.get("matching_sites"),
            "symmetry_distinct_matching_sites": qualification.get(
                "symmetry_distinct_matching_sites"
            ),
        }
        if not passed:
            all_handles = False
            failed_roles.append(role)
    output["admitted"] = all_handles
    output["reason"] = (
        "admitted" if all_handles else "handle_policy_failure:" + ",".join(failed_roles)
    )
    return output


def gzip_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize a deterministic compressed candidate ledger."""

    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        for row in rows:
            handle.write(
                (
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode()
            )
    return raw.getvalue()


__all__ = [
    "ARMS",
    "CONFIG_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiConstrainedStochasticProductionCandidatesError",
    "branch_class",
    "exact_terminal_admission",
    "gzip_jsonl",
    "load_json",
    "program_shard",
    "stable_sha256",
]

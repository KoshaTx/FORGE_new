"""Read-only adjudication of proposal-augmented synthesis guidance."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_synthesis_guidance_failure_audit_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_synthesis_guidance_failure_audit.v1"


class UgiSynthesisGuidanceFailureAuditError(RuntimeError):
    """Raised when an input or invariant of the frozen audit changes."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise UgiSynthesisGuidanceFailureAuditError(f"JSON object required: {path}")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiSynthesisGuidanceFailureAuditError(f"invalid pin: {label}")
    path = (repo / str(record["path"])).resolve()
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiSynthesisGuidanceFailureAuditError(f"input changed: {label}")
    return path


def build_synthesis_guidance_failure_audit(repo: Path, config_path: Path) -> dict[str, Any]:
    """Explain why the matched nonzero synthesis tilt is not promoted."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiSynthesisGuidanceFailureAuditError("unsupported config schema")
    paths = {
        label: _pin(repo, record, label=label) for label, record in config.get("inputs", {}).items()
    }
    if set(paths) != {
        "guidance_result",
        "guidance_diagnostics",
        "guidance_support_audits",
        "panel_readiness_result",
        "proposal_readjudication_result",
    }:
        raise UgiSynthesisGuidanceFailureAuditError("input set changed")
    result = _load(paths["guidance_result"])
    diagnostics = _load(paths["guidance_diagnostics"])
    support = _load(paths["guidance_support_audits"])
    panel = _load(paths["panel_readiness_result"])
    proposal = _load(paths["proposal_readjudication_result"])
    typed = result["execution"]["typed_route_readiness_outcomes"]["by_arm"]
    guided_final = typed["guided"]["productive_final"]
    post_final = typed["post_hoc"]["productive_final"]
    ancestry = diagnostics["guided_ancestry_by_checkpoint"]
    total_groups = sum(int(row["groups"]) for row in ancestry.values())
    resampled = sum(int(row["resampled_groups"]) for row in ancestry.values())
    mixed = sum(int(row["mixed_observed_utility_groups"]) for row in ancestry.values())
    changed = int(
        diagnostics["guided_vs_post_hoc_productive"]["canonical_identity"]["different_by_index"]
    )

    proposal_components = []
    for record in support["records"]:
        graded = record.get("graded_route_readiness")
        if not isinstance(graded, dict):
            continue
        for component in graded.get("components", []):
            discovery = component.get("proposal_discovery")
            if discovery is not None:
                proposal_components.append((component["canonical_smiles"], discovery))
    statuses = Counter(
        str(discovery["semantic_resolution_status"]) for _, discovery in proposal_components
    )
    if any(
        discovery.get("may_enter_synthesis_value") is not False
        for _, discovery in proposal_components
    ):
        raise UgiSynthesisGuidanceFailureAuditError("proposal-only hypothesis entered value")

    guided_ready = int(guided_final["route_ready"])
    post_ready = int(post_final["route_ready"])
    if guided_ready > post_ready:
        raise UgiSynthesisGuidanceFailureAuditError("negative-result audit no longer applies")
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "proposal_augmented_synthesis_guidance_failure_audited",
        "config": {"path": str(config_path.relative_to(repo)), "sha256": sha256_file(config_path)},
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "matched_result": {
            "guidance_strength": result["execution"]["guidance_strength"],
            "guided_unique_route_ready_final": guided_ready,
            "post_hoc_unique_route_ready_final": post_ready,
            "absolute_difference": guided_ready - post_ready,
            "guided_canonical_representatives": guided_final["canonical_representatives"],
            "post_hoc_canonical_representatives": post_final["canonical_representatives"],
            "changed_terminal_identities": changed,
            "terminal_particles": 64,
            "selected_constitutional_overlap": diagnostics["guided_vs_post_hoc_productive"][
                "selected_constitutional_overlap"
            ],
        },
        "controller_diagnostics": {
            "checkpoint_groups": total_groups,
            "mixed_utility_groups": mixed,
            "mixed_utility_group_fraction": mixed / total_groups,
            "resampled_groups": resampled,
            "resampled_group_fraction": resampled / total_groups,
            "interpretation": "Binary route readiness was too sparse within four-particle morphology groups to alter most ancestry decisions.",
        },
        "proposal_engine_diagnostics": {
            "proposal_component_occurrences": len(proposal_components),
            "unique_proposed_components": len({smiles for smiles, _ in proposal_components}),
            "semantic_status_occurrences": dict(sorted(statuses.items())),
            "proposal_model_scores_used": False,
            "proposal_only_hypotheses_entered_value": False,
            "global_semantic_readjudication": proposal["summary"],
            "interpretation": "The proposal engine mostly rediscovered a known oxidation family; new-family hypotheses correctly remained evidence-free and could not create route readiness.",
        },
        "coverage_context": {
            "separate_terminal_panel_readiness": panel["summary"],
            "interpretation": "Family evidence improves terminal route triage, but it does not make incomplete graph states informative enough for this mid-trajectory controller.",
        },
        "diagnosis": [
            "The fixed 16-program schedule permits ancestry changes only among four particles sharing one morphology; it cannot reallocate probability across route-favorable morphologies.",
            "Only a small minority of checkpoint groups contained both ready and unready completions, so most resampling probabilities were uniform.",
            "Exact component chemistry, not coarse partial topology, determined most route-readiness outcomes.",
            "Graph2Edits supplies search hypotheses, not experimental evidence; its unverified new-family proposals cannot legitimately be converted into synthesis value.",
        ],
        "decision": {
            "midtrajectory_synthesis_tilting_promoted": False,
            "repeat_lambda_tuning_on_same_schedule": False,
            "production_synthesis_method": "complete generation followed by proposal-augmented L1/L2/L3 routing before prospective panel lock",
            "proposal_engine_retained": True,
            "synthesis_grounded_claim_retained": True,
            "reason": "No matched final route-ready yield gain and sparse, weakly actionable intermediate reward.",
        },
        "paper_interpretation": "Route search remains part of candidate selection before synthesis-panel locking, but the current evidence does not support claiming that synthesis guidance improves the molecular trajectory.",
        "nonclaims": [
            "The negative controller result does not imply that the generated molecules are unsynthesizable.",
            "Failure to close a route is missing knowledge under a declared search system, not proof of chemical impossibility.",
            "Proposal discovery does not constitute route validation or synthesis-success probability.",
        ],
    }
    content["result_sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return content


__all__ = [
    "UgiSynthesisGuidanceFailureAuditError",
    "build_synthesis_guidance_failure_audit",
]

#!/usr/bin/env python3
"""Put AiZynthFinder proposals through the same independent forward resolver.

The forward resolver is engine-agnostic: it consumes a typed
``SingleStepRetrosynthesisProposal`` carrying any backend manifest.  It was only
ever wired to Graph2Edits, which left the planner's proposals unverified and
made the two engines incomparable.  This runs the identical verification over
the planner's single-step proposals so both engines are adjudicated by exactly
the same independent contract.

Verification here is a mechanical forward-reconstruction check, not a knowledge
base: it confirms the proposed reactants reproduce the exact target.  It does
not establish substrate scope, evidence tier or procurement.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_DEFAULT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DEFAULT / "src"))

from forge.product.ugi_bounded_hybrid_route_cascade import (  # noqa: E402
    UgiBoundedHybridRouteCascadeError,
    atomic_write,
    canonical_json_bytes,
    jsonl_gzip_bytes,
    load_json,
    read_jsonl_gzip,
    sha256_file,
    sha256_payload,
)
from forge.route.graph2edits_one_gap_diagnostic import _qualification_from_dict  # noqa: E402
from forge.route.l2_forward_resolver import load_independent_l2_forward_resolver  # noqa: E402
from forge.route.planner import RouteTarget  # noqa: E402
from forge.route.proposal_discovery_status import (  # noqa: E402
    SourceNeutralProposalDiscoveryResolver,
)
from forge.route.proposal_engine import (  # noqa: E402
    ProposalBackendManifest,
    ProposalRequest,
    ProposalTargetKind,
    RootQualificationReceipt,
    SingleStepRetrosynthesisProposal,
)
from forge.route.semantic_family_equivalence import (  # noqa: E402
    audit_semantic_family_equivalence,
)

RESULT_SCHEMA_VERSION = "phase1_ugi_planner_proposal_verification.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_planner_proposal_verification_ledger.v1"
OPERATIONAL_POLICY_ID = "forge.planner_proposal_independent_forward_verification.v1"

AUTHORITY = {
    "forward_reconstruction_is_a_mechanical_check": True,
    "forward_verification_is_substrate_scope_proof": False,
    "forward_verification_is_route_evidence": False,
    "forward_verification_is_procurement_evidence": False,
    "may_close_route": False,
}


def build(repo: Path, output_dir: Path) -> dict[str, Any]:
    reach_result_path = repo / "results/phase1/ugi_planner_reach_sweep_v1/result.json"
    reach_result = load_json(reach_result_path, label="planner reach result")
    if reach_result.get("status") != "planner_reach_sweep_complete":
        raise UgiBoundedHybridRouteCascadeError("planner reach sweep is not complete")
    reach_ledger_path = repo / (
        "results/phase1/ugi_planner_reach_sweep_v1/planner_reach_ledger.jsonl.gz"
    )
    if sha256_file(reach_ledger_path) != (
        reach_result["artifacts"]["planner_reach_ledger.jsonl.gz"]["sha256"]
    ):
        raise UgiBoundedHybridRouteCascadeError("planner reach ledger changed")
    reach_rows = read_jsonl_gzip(reach_ledger_path, label="planner reach ledger")

    exact_path = repo / (
        "results/phase1/ugi_bounded_hybrid_route_cascade_v1/component_exact_evidence.jsonl.gz"
    )
    exact_rows = read_jsonl_gzip(exact_path, label="component exact evidence")
    support_by = {row["component_sha256"]: row for row in exact_rows}

    runtime_manifest_path = repo / (
        "configs/route/aizynthfinder_public_v4_4_1_diagnostic_macos_arm64_v1.json"
    )
    runtime_manifest = load_json(runtime_manifest_path, label="AiZynthFinder runtime manifest")
    manifest = ProposalBackendManifest(
        backend_id="aizynthfinder-public-v4.4.1-uspto-ringbreaker",
        implementation_version=f"aizynthfinder=={runtime_manifest['aizynthfinder_version']}",
        checkpoint_sha256=sha256_file(
            repo / "data/source_cache/aizynthfinder_public_v4_4_1/uspto_model.onnx"
        ),
        checkpoint_license="public-release-zenodo-7797465",
        training_corpus_id="aizynthfinder-public-uspto-templates",
        training_corpus_snapshot_sha256=sha256_file(
            repo / "data/source_cache/aizynthfinder_public_v4_4_1/uspto_templates.csv.gz"
        ),
        source_locator=runtime_manifest["sources"]["code"],
    )
    exact_resolver = load_independent_l2_forward_resolver(
        repo / "configs/route/graph2edits_l2_forward_resolver_v1.json", repo_root=repo
    )
    resolver = SourceNeutralProposalDiscoveryResolver(exact_resolver=exact_resolver)
    policy_sha256 = sha256_payload(
        {
            "policy_id": OPERATIONAL_POLICY_ID,
            "proposal_only": True,
            "operational_screen_run": False,
            "route_closure_authorized": False,
        }
    )

    rows: list[dict[str, Any]] = []
    per_component: list[dict[str, Any]] = []
    for index, reach in enumerate(reach_rows, start=1):
        key = str(reach["component_sha256"])
        role = str(reach["role"])
        smiles = str(reach["canonical_smiles"])
        support_row = support_by.get(key)
        representative = (support_row or {}).get("representative")
        if representative is None:
            per_component.append(
                {
                    "component_sha256": key,
                    "role": role,
                    "canonical_smiles": smiles,
                    "verified_proposals": 0,
                    "proposals": len(reach["single_step_proposals"]),
                    "skipped": "no_qualified_route_root",
                }
            )
            continue
        support = representative["support_audit"]["support"]
        matches = [
            (root, qual)
            for root, qual in zip(
                support["root_targets"], support["root_qualifications"], strict=True
            )
            if str(root.get("role")) == role and str(root.get("canonical_smiles")) == smiles
        ]
        if len(matches) != 1:
            raise UgiBoundedHybridRouteCascadeError(
                f"component does not match exactly one qualified root: {role} {smiles}"
            )
        root_dict, qual_dict = matches[0]
        route_root = RouteTarget(
            role=role,
            canonical_smiles=smiles,
            product_context_smiles=tuple(root_dict.get("product_context_smiles", ())),
        )
        request = ProposalRequest(
            route_root=route_root,
            target=route_root,
            depth=0,
            target_kind=ProposalTargetKind.ROOT,
            root_qualification=RootQualificationReceipt(
                route_root=route_root,
                qualification=_qualification_from_dict(qual_dict),
                terminal_sha256=support["terminal_sha256"],
                generator_checkpoint_sha256=support["generator_checkpoint_sha256"],
                l1_reaction_sha256=representative["l1_reaction_sha256"],
                qualification_artifact_sha256=representative["support_audit"]["support_sha256"],
            ),
            operational_policy_id=OPERATIONAL_POLICY_ID,
            operational_policy_sha256=policy_sha256,
        )
        verified = 0
        for item in reach["single_step_proposals"]:
            reactants = tuple(item["canonical_reactants"])
            if not reactants:
                continue
            try:
                proposal = SingleStepRetrosynthesisProposal(
                    proposal_id=f"aizynth:{key[:16]}:{int(item['rank']):02d}",
                    backend=manifest,
                    request=request,
                    reactant_smiles=reactants,
                    rank=int(item["rank"]),
                )
                resolution = resolver.resolve(proposal)
                semantic = audit_semantic_family_equivalence(resolution, resolver.transforms)
            except Exception as error:  # noqa: BLE001 - recorded, never dropped
                rows.append(
                    {
                        "schema_version": LEDGER_SCHEMA_VERSION,
                        "component_sha256": key,
                        "role": role,
                        "target_smiles": smiles,
                        "rank": int(item["rank"]),
                        "reactants": list(reactants),
                        "resolution_error": f"{type(error).__name__}: {error}",
                        "graph_consistent_discovery_hypothesis": False,
                        "authority": dict(AUTHORITY),
                    }
                )
                continue
            consistent = bool(semantic["graph_consistent_discovery_hypothesis"])
            verified += int(consistent)
            rows.append(
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "component_sha256": key,
                    "role": role,
                    "target_smiles": smiles,
                    "rank": int(item["rank"]),
                    "reactants": list(reactants),
                    "resolution_error": None,
                    "raw_discovery_resolution": resolution.to_dict(),
                    "semantic_equivalence": semantic,
                    "graph_consistent_discovery_hypothesis": consistent,
                    "authority": dict(AUTHORITY),
                }
            )
        per_component.append(
            {
                "component_sha256": key,
                "role": role,
                "canonical_smiles": smiles,
                "proposals": len(reach["single_step_proposals"]),
                "verified_proposals": verified,
                "any_forward_verified": verified > 0,
                "planner_solved_to_public_catalog": bool(reach["planner_solved_to_public_catalog"]),
                "skipped": None,
            }
        )
        if index % 20 == 0 or index == len(reach_rows):
            print(f"verify {index}/{len(reach_rows)}", flush=True)

    rows.sort(key=lambda row: (str(row["role"]), str(row["target_smiles"]), int(row["rank"])))
    per_component.sort(key=lambda row: (str(row["role"]), str(row["canonical_smiles"])))
    ledger_path = output_dir / "planner_proposal_verification_ledger.jsonl.gz"
    atomic_write(ledger_path, jsonl_gzip_bytes(rows))

    verified_components = [row for row in per_component if row.get("any_forward_verified")]
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "planner_proposal_verification_complete",
        "inputs": {
            "planner_reach_result": {
                "path": str(reach_result_path.relative_to(repo)),
                "sha256": sha256_file(reach_result_path),
            },
            "planner_reach_ledger": {
                "path": str(reach_ledger_path.relative_to(repo)),
                "sha256": sha256_file(reach_ledger_path),
            },
            "component_exact_evidence": {
                "path": str(exact_path.relative_to(repo)),
                "sha256": sha256_file(exact_path),
            },
            "forward_resolver_config": {
                "path": "configs/route/graph2edits_l2_forward_resolver_v1.json",
                "sha256": sha256_file(
                    repo / "configs/route/graph2edits_l2_forward_resolver_v1.json"
                ),
            },
        },
        "backend_manifest": manifest.to_dict(),
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "summary": {
            "components": len(per_component),
            "components_skipped_no_qualified_root": sum(
                1 for row in per_component if row.get("skipped")
            ),
            "proposals_examined": len(rows),
            "proposals_forward_verified": sum(
                1 for row in rows if row["graph_consistent_discovery_hypothesis"]
            ),
            "proposals_with_resolution_error": sum(1 for row in rows if row["resolution_error"]),
            "components_with_any_forward_verified_proposal": len(verified_components),
            "components_by_role_with_verified_proposal": dict(
                sorted(Counter(row["role"] for row in verified_components).items())
            ),
            "semantic_status_counts": dict(
                sorted(
                    Counter(
                        str(row["semantic_equivalence"]["semantic_resolution_status"])
                        for row in rows
                        if row["resolution_error"] is None
                    ).items()
                )
            ),
        },
        "per_component": per_component,
        "artifacts": {
            "planner_proposal_verification_ledger.jsonl.gz": {
                "path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
                "rows": len(rows),
                "schema_version": LEDGER_SCHEMA_VERSION,
            }
        },
        "authority": dict(AUTHORITY),
        "interpretation": (
            "Both proposal engines are now adjudicated by one identical independent "
            "forward-reconstruction contract. Verification confirms the proposed reactants "
            "reproduce the exact target; it does not establish substrate scope, evidence tier "
            "or procurement."
        ),
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / "result.json", canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_planner_proposal_verification_v1"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    result = build(repo, output_dir.resolve())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

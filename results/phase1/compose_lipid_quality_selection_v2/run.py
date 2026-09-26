"""Matched complete-component pool selection and independent quality reassessment."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from rdkit import Chem, rdBase

from forge.core.hashing import sha256_file
from forge.model.compose_lipid_component_diversity import accepted_components
from forge.model.compose_lipid_quality import canonical_molecule, source_role_mapping
from forge.model.compose_lipid_quality_selection import (
    ComponentIdentity,
    SelectionCandidate,
    first_exact,
    select_reference_supported,
)
from results.phase1.compose_lipid_component_decoder_v1.contracts import load_all, matching
from results.phase1.compose_lipid_quality_v1.run import assess_panel, support_tables

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
QUALITY = HERE.with_name("compose_lipid_quality_v1")
CACHE = HERE.with_name("compose_lipid_mapped_preparation_v3") / "cache"


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def reference_support(reference, quality_policy, pins):
    """Reuse a task-owned, hash-verified local cache of deterministic support sets."""
    cached = HERE / "reference_support.pkl"
    receipt = HERE / "reference_support.json"
    source = {
        key: pins[key]
        for key in ("reference", "quality_module", "quality_runner", "quality_policy")
    }
    if cached.exists() and receipt.exists():
        saved = read(receipt)
        if saved["inputs"] == source and saved["artifact"] == pin(cached):
            with cached.open("rb") as handle:
                return pickle.load(handle)
        raise ValueError("Reference-support cache pin changed; use a new task output version")
    result = support_tables(reference, quality_policy)
    with cached.open("wb") as handle:
        pickle.dump(result, handle, protocol=5)
    write(
        receipt, {"inputs": source, "artifact": pin(cached), "trust": "task-owned local cache only"}
    )
    return result


def candidate_pools(
    rows, branch, product_support, component_support, reference, layouts, executors, db
):
    all_components = set(reference["component_smiles_by_identity"].values())
    diagnostics, by_family, originals = [], defaultdict(list), {}
    product_cache, component_cache = {}, {}
    for row in rows:
        index, family = row["index"], row["family"]
        branch_row = row["branches"][branch]
        raw = dict(branch_row["raw"], kind="raw")
        proposals = [raw, *branch_row.get("proposals", [])]
        options = matching(executors, layouts[index])
        pool, seen = [], set()
        for ordinal, proposal in enumerate(proposals):
            saved_smiles, assessment = proposal.get("smiles"), proposal["check"]
            parsed = canonical_molecule(saved_smiles)
            canonical = parsed[0] if parsed else None
            connected = parsed is not None and len(Chem.GetMolFrags(parsed[1])) == 1
            if canonical is not None and canonical in seen:
                diagnostics.append(
                    {
                        "index": index,
                        "family": family,
                        "branch": branch,
                        "ordinal": ordinal,
                        "kind": proposal.get("kind"),
                        "status": "duplicate_canonical_product",
                        "smiles": canonical,
                    }
                )
                continue
            if canonical:
                seen.add(canonical)
            exact = assessment.get("exact") is True
            components = []
            support = None
            product_novel = False
            if connected:
                key = family, canonical
                if key not in product_cache:
                    product_cache[key] = {
                        "product_support": product_support[family].assess(parsed[1]),
                        "product_train_novel": db.execute(
                            "select 1 from sources where constitution_id=?",
                            (hashlib.sha256(canonical.encode()).hexdigest(),),
                        ).fetchone()
                        is None,
                    }
                support = dict(product_cache[key])
                product_novel = support["product_train_novel"]
            if exact:
                if not connected:
                    raise ValueError("A full-source exact candidate has no connected molecule")
                parts = accepted_components(assessment)
                role_mapping = source_role_mapping(parts, assessment, options)
                support["components"] = {}
                for role, smiles in sorted(parts.items()):
                    parsed_part = canonical_molecule(smiles)
                    if parsed_part is None:
                        raise ValueError("An accepted source component does not sanitize")
                    smiles = parsed_part[0]
                    source_role = role_mapping[role]
                    key = family, source_role, smiles
                    if key not in component_cache:
                        source = component_support[family][source_role]
                        component_cache[key] = {
                            "source_role": source_role,
                            "smiles": smiles,
                            "global_train_novel": smiles not in all_components,
                            "role_train_novel": smiles not in source.identities,
                            **source.assess(parsed_part[1]),
                        }
                    part = component_cache[key]
                    support["components"][role] = part
                    components.append(
                        ComponentIdentity(
                            role, smiles, part["global_train_novel"], part["role_train_novel"]
                        )
                    )
                observed = support["product_support"][
                    "all_local_features_observed"
                ] is True and all(
                    p["all_local_features_observed"] is True for p in support["components"].values()
                )
            else:
                observed = False
            candidate = SelectionCandidate(
                index,
                ordinal,
                canonical,
                connected,
                exact,
                observed,
                product_novel,
                tuple(components),
            )
            pool.append(candidate)
            originals[index, ordinal] = {
                "index": index,
                "family": family,
                "selected_smiles": saved_smiles,
                "selected_check": assessment,
                "selected_ordinal": ordinal,
                "selected_kind": proposal.get("kind"),
                "source_branch": branch,
            }
            diagnostics.append(
                {
                    "index": index,
                    "family": family,
                    "branch": branch,
                    "ordinal": ordinal,
                    "kind": proposal.get("kind"),
                    "status": "assessed",
                    "candidate": asdict(candidate),
                    "diagnostic": support,
                }
            )
        by_family[family].append(pool)
    return by_family, diagnostics, originals


def novel_component_summary(selection):
    output = defaultdict(lambda: defaultdict(list))
    for family, candidates in selection.items():
        for candidate in candidates:
            for component in candidate.components:
                output[family][component.role].append(component)
    return {
        family: {
            role: {
                "observations": len(parts),
                "full_train_novel_observations": sum(p.global_train_novel for p in parts),
                "full_train_novel_unique": len({p.smiles for p in parts if p.global_train_novel}),
                "source_role_train_novel_observations": sum(p.role_train_novel for p in parts),
                "source_role_train_novel_unique": len(
                    {p.smiles for p in parts if p.role_train_novel}
                ),
            }
            for role, parts in roles.items()
        }
        for family, roles in output.items()
    }


def transitions(before, after):
    left = {row["index"]: row for row in before}
    right = {row["index"]: row for row in after}
    if set(left) != set(right):
        raise ValueError("Paired comparison changed request IDs")
    counts, rows = Counter(), []
    for index in sorted(left):
        a, b = left[index], right[index]
        if a["family"] != b["family"]:
            raise ValueError("Paired family assignment changed")
        sa = (
            a["exact_l1"]
            and a.get("all_component_features_observed", False)
            and a["product_support"]["all_local_features_observed"] is True
        )
        sb = (
            b["exact_l1"]
            and b.get("all_component_features_observed", False)
            and b["product_support"]["all_local_features_observed"] is True
        )
        label = f"exact:{int(a['exact_l1'])}->{int(b['exact_l1'])};support:{int(sa)}->{int(sb)}"
        counts[label] += 1
        rows.append(
            {
                "index": index,
                "family": a["family"],
                "transition": label,
                "product_changed": a.get("canonical_smiles") != b.get("canonical_smiles"),
            }
        )
    return {"counts": dict(counts), "requests": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--producer-protocol", type=Path, required=True)
    parser.add_argument("--label", default="first_draw")
    args = parser.parse_args()
    started = time.monotonic()
    out = HERE / args.label
    out.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    paths = {
        "ledger": args.ledger.resolve(),
        "producer_protocol": args.producer_protocol.resolve(),
        "policy": HERE / "policy.json",
        "prespecification": HERE / "prespecification.json",
        "selector": ROOT / "forge/model/compose_lipid_quality_selection.py",
        "runner": Path(__file__),
        "reference": QUALITY / "reference.json",
        "quality_policy": QUALITY / "policy.json",
        "quality_controls": QUALITY / "positive_controls.json",
        "quality_module": ROOT / "forge/model/compose_lipid_quality.py",
        "quality_runner": QUALITY / "run.py",
        "membership": CACHE / "sources.sqlite",
        "manifest": CACHE / "manifest.json",
        "layout_payload": HERE.with_name("compose_lipid_component_decoder_v1") / "fresh/input.pt",
        "layout_protocol": HERE.with_name("compose_lipid_component_decoder_v1")
        / "fresh/protocol.json",
        "role_contract_loader": HERE.with_name("compose_lipid_component_decoder_v1")
        / "contracts.py",
    }
    pins = {name: pin(path) for name, path in paths.items()}
    policy, quality_policy, reference = (
        read(paths["policy"]),
        read(paths["quality_policy"]),
        read(paths["reference"]),
    )
    assert read(paths["prespecification"])["policy"] == pins["policy"]
    assert reference["inputs"]["quality_code"] == pins["quality_module"]
    assert reference["inputs"]["runner"] == pins["quality_runner"]
    assert pins["membership"] == read(paths["manifest"])["artifacts"]["sources.sqlite"]
    assert pins["layout_payload"] == read(paths["layout_protocol"])["inputs"]["payload"]
    rows = read(paths["ledger"])
    if len({row["index"] for row in rows}) != len(rows):
        raise ValueError("Duplicate requested output IDs")
    if any(set(row["branches"]) != {"d0", "d1"} for row in rows):
        raise ValueError("Expected exactly the frozen D0 and D1 branches")
    layouts = torch.load(paths["layout_payload"], weights_only=False, map_location="cpu")["layouts"]
    if any(layouts[row["index"]].family != row["family"] for row in rows):
        raise ValueError("Candidate requests differ from pinned TRAIN layouts")
    with rdBase.BlockLogs():
        executors, source_pins, _ = load_all()
    pins.update({"source_contract:" + name: value for name, value in source_pins.items()})
    write(
        out / "protocol.json",
        {"inputs": pins, "policy": policy, "seed": policy["seed"], "requests": len(rows)},
    )
    print("Loading pinned TRAIN feature reference", flush=True)
    product_support, component_support, _ = reference_support(reference, quality_policy, pins)
    panels, per_attempt, selection_reports, novelty, costs = {}, {}, {}, {}, {}
    with sqlite3.connect(paths["membership"].as_uri() + "?mode=ro", uri=True) as db:
        for branch in ("d0", "d1"):
            print(f"Assessing complete-component candidates: {branch}", flush=True)
            pools, diagnostics, originals = candidate_pools(
                rows, branch, product_support, component_support, reference, layouts, executors, db
            )
            write(
                out / f"{branch}_candidate_diagnostics.json",
                {"inputs": pins, "candidates": diagnostics},
            )
            baseline, supported, reports = {}, {}, {}
            for family, family_pools in sorted(pools.items()):
                baseline[family] = [first_exact(pool) for pool in family_pools]
                supported[family], reports[family] = select_reference_supported(
                    family_pools,
                    node_limit=policy["support_aware"]["solver"]["node_limit"],
                    time_limit=policy["support_aware"]["solver"]["time_limit_seconds_per_stage"],
                )
            selection_reports[branch] = reports
            costs[branch] = {
                "requests": len(rows),
                "producer_proposals": sum(
                    len(row["branches"][branch].get("proposals", [])) for row in rows
                ),
                "actual_costs_by_request": [
                    {
                        "index": row["index"],
                        "costs": row["branches"][branch].get("costs"),
                        "construction": row["branches"][branch].get("construction"),
                    }
                    for row in rows
                ],
            }
            for kind, selection in (("exact_only", baseline), ("support_aware", supported)):
                label = branch + "_" + kind
                selected = [
                    originals[c.request, c.ordinal]
                    for family in sorted(selection)
                    for c in selection[family]
                ]
                selected.sort(key=lambda row: row["index"])
                write(out / f"{label}_selected.json", {"inputs": pins, "attempts": selected})
                panels[label], detail = assess_panel(
                    selected,
                    "selected",
                    product_support,
                    component_support,
                    reference,
                    quality_policy,
                    db,
                    executors,
                    layouts,
                )
                per_attempt[label] = detail
                novelty[label] = novel_component_summary(selection)
                write(out / f"{label}_assessment.json", {"inputs": pins, "attempts": detail})
    comparisons = {
        name: transitions(per_attempt[a], per_attempt[b])
        for name, a, b in [
            ("d0_selection", "d0_exact_only", "d0_support_aware"),
            ("d1_selection", "d1_exact_only", "d1_support_aware"),
            ("decoder_exact_only", "d0_exact_only", "d1_exact_only"),
            ("decoder_support_aware", "d0_support_aware", "d1_support_aware"),
        ]
    }
    result = {
        "schema_version": "forge.compose_lipid_quality_selection_result.v2",
        "inputs": pins,
        "requests": len(rows),
        "by_panel": panels,
        "component_novelty": novelty,
        "selection_reports": selection_reports,
        "paired_transitions": comparisons,
        "costs": costs,
        "runtime_seconds": time.monotonic() - started,
        "nonclaims": [
            "Observed neighborhoods/rings do not certify chemical quality.",
            "Unknown structures remain in the candidate ledger and attempt denominators.",
            "Broad observed-lipid TRAIN comparison is not family-matched or independent generalization.",
            "No new model calls, training, sealed TEST access or paid compute.",
        ],
    }
    write(out / "result.json", result)
    print(
        json.dumps(
            {
                name: {
                    "exact": sum(r["exact_l1"] for r in families.values()),
                    "observed": sum(
                        r["exact_all_local_features_observed"] for r in families.values()
                    ),
                }
                for name, families in panels.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

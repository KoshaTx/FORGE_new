"""TRAIN whole-component compatibility census; never constructs new molecules."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from forge.corpus.mapped_program_cache import MappedProgramCache
from forge.model.compose_lipid_layout import encoded, summarize_layout

WORKTREE = Path(__file__).resolve().parents[3]
ROOT = WORKTREE.parent.parent
N = ROOT / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1"
OUT = N / "b_quality/component_feasibility_v1/run_v2"
MIRROR = ROOT / ".Codex-scratch/compose-family-rules-v2"
FAMILIES = ("a3_amine_aldehyde_alkyne", "ketone_ugi4")
STAGES = (
    "TRAIN_source_role",
    "source_quantity",
    "origin_block_sizes",
    "fixed_assembly_core",
    "ordered_core_correspondence",
    "role_morphology",
    "requested_ring_signature",
    "retained_head_witness",
)


def read(path):
    return json.loads(path.read_text())


def pin(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest()}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def layout_signature(layout):
    """Reconstruct the anonymous sampled template without reading exterior atom labels."""
    record, graph = layout.record, layout.record.graph
    core = np.flatnonzero(record.core_position_states > 1)
    inverse = {int(node): i for i, node in enumerate(core)}
    if not np.array_equal(record.fixed_atom_mask, record.core_position_states > 1):
        raise ValueError("Unqualified fixed exterior atoms")
    tree = []
    for child in np.flatnonzero(record.fixed_parent_bond_mask):
        parent = int(graph.parents[child])
        tree.append([inverse[parent], inverse[int(child)], int(graph.parent_bonds[child])])
    closures = [
        [
            inverse[int(graph.closure_left[j])],
            inverse[int(graph.closure_right[j])],
            int(graph.closure_bonds[j]),
        ]
        for j in np.flatnonzero(record.fixed_closure_bond_mask)
    ]
    blocks, positions = [], defaultdict(list)
    for block in record.component_blocks:
        local = record.core_position_states[block.start : block.stop].tolist()
        blocks.append([block.role, block.role_state, [p for p in local if p > 1]])
        positions[block.role_state].append(local)
    bundle = {
        "family": layout.family,
        "program": record.program_id,
        "program_state": record.program_state,
        "depth": record.program_depth,
        "quantities": layout.quantities,
        "introduced": list(layout.introduced_roles),
        "blocks": blocks,
        "core_nodes": graph.node_states[core].tolist(),
        "core_units": layout.core_units[core].tolist(),
        "core_tree": tree,
        "core_closures": closures,
    }
    roles = {}
    for block in record.component_blocks:
        if block.role in roles:
            continue
        state = block.role_state
        local = positions[state]
        roles[block.role] = {
            "role_state": state,
            "quantity": layout.quantities.get(block.role),
            "positions": local,
            "block_sizes": list(map(len, local)),
            "morphology": record.role_morphology_states[block.start].tolist(),
            "closures": layout.variable_closures_by_role[state],
            "allowed_ring_sizes": list(layout.ring_sizes_by_role.get(state, ())),
        }
    return bundle, roles


def role_aliases(rows):
    """Source roles must come from saved exact diagnostic correspondences."""
    mappings = defaultdict(set)
    for row in rows:
        if row["family"] not in FAMILIES:
            continue
        for accepted, component in (row.get("diagnostic") or {}).get("components", {}).items():
            mappings[row["family"], accepted].add(component["source_role"])
    if any(len(v) != 1 for v in mappings.values()):
        raise ValueError("Ambiguous accepted-to-source role correspondence")
    return {k: next(iter(v)) for k, v in mappings.items()}


def compatible_stages(profile, template, requirement):
    """Nested necessary conditions; never an admission of a recombined product."""
    checks = [
        True,
        profile["quantity"] == requirement["quantity"],
        profile["block_sizes"] == requirement["block_sizes"],
        profile["template"] == template,
        profile["positions"] == requirement["positions"],
        profile["morphology"] == requirement["morphology"],
        profile["closures"] == requirement["closures"]
        and set(profile["ring_sizes"]) <= set(requirement["allowed_ring_sizes"]),
        profile["head_status"] in {"pass", "not_applicable"},
    ]
    flags, passed = [], True
    for check in checks:
        passed = passed and check
        flags.append(passed)
    return flags


def retained_head_witness(smiles, reaction, contract):
    """Apply only the admitted head query and source reactive-site properties."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid admitted TRAIN component")
    retained = reaction["retained_queries"]
    if len(retained) != 1:
        raise ValueError("Expected the pinned single retained-head query")
    retained = retained[0]
    role = next(r for r in reaction["reactant_roles"] if r["name"] == retained["role"])
    sites = [s for s in contract["site_contract"] if s["role"] == retained["role"]]
    if len(sites) != 1:
        raise ValueError("Ambiguous head source site contract")
    reactive_query = Chem.MolFromSmarts(role["required_handle_smarts"])
    basic_query = Chem.MolFromSmarts(retained["smarts"])
    if reactive_query.GetNumAtoms() != 1 or basic_query.GetNumAtoms() != 1:
        raise ValueError("This census requires the source's single-atom head queries")
    properties = sites[0]["properties"]
    reactive = []
    for (i,) in mol.GetSubstructMatches(reactive_query):
        atom = mol.GetAtomWithIdx(i)
        values = {
            "atomic_number": atom.GetAtomicNum(),
            "formal_charge": atom.GetFormalCharge(),
            "total_hydrogens": atom.GetTotalNumHs(),
        }
        if set(properties) - values.keys():
            raise ValueError("Unknown source site property")
        if all(values[k] == v for k, v in properties.items()):
            reactive.append(i)
    survivors = sorted({m[0] for m in mol.GetSubstructMatches(basic_query)} - set(reactive))
    return {
        "reactive_atoms": reactive,
        "retained_atoms": survivors,
        "status": "pass" if reactive and len(survivors) >= retained["minimum_matches"] else "fail",
        "basis": "Pinned precursor-role structural witness; no pKa or full assembly execution",
    }


def require_source_correspondence(target, family, example):
    if example.family != family or example.record.graph.structure_id != target:
        raise ValueError("TRAIN index/source correspondence mismatch")


def paths():
    return {
        "producer": Path(__file__),
        "tests": WORKTREE / "tests/test_component_feasibility_census_v2.py",
        "test_receipt": N / "b_quality/component_feasibility_tests_v2.xml",
        "cache_code": WORKTREE / "forge/corpus/mapped_program_cache.py",
        "layout_code": WORKTREE / "forge/model/compose_lipid_layout.py",
        "morphology_code": WORKTREE / "forge/model/reaction_program_flow.py",
        "manifest": ROOT / "results/phase1/compose_lipid_mapped_preparation_v3/cache/manifest.json",
        "weights": ROOT / "results/phase1/compose_lipid_training_cohort_v1/weights/weights.sqlite",
        "measure": ROOT / "results/phase1/compose_lipid_training_cohort_v1/weights/result.json",
        "TRAIN_census": N / "e_diversity/isocyanide_diagnosis_v1/run_v4/result.json",
        "reference": ROOT / "results/phase1/compose_lipid_quality_v1/reference.json",
        "layouts": ROOT / "results/phase1/compose_lipid_quality_confirmation_v1/input.pt",
        "current": ROOT
        / "results/phase1/compose_lipid_iclr22_research_v1/quality/all22_context_preserving/result.json",
        "joint": N / "integration/joint_selection_v1/run_v2/result.json",
        "gates": N / "a_attribution/common_gate_ledger_v1/gates.json",
        "candidate_diagnostics": ROOT
        / "results/phase1/compose_lipid_quality_confirmation_v1/assessment/d1_candidate_diagnostics.json",
        "registry": MIRROR / "data/vendor/qualified_ketone_ugi4_source_program_v1.json",
        "adjudication": MIRROR
        / "results/phase1/compose_lipid_ketone_ugi4_source_v1/adjudication.json",
    }


def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    files = paths()
    write(
        OUT / "protocol.json",
        {
            "schema": "forge.complete_component_feasibility.v1",
            "inputs": {k: pin(v) for k, v in files.items()},
            "families": list(FAMILIES),
            "requests": 128,
            "per_family": 64,
            "TRAIN_graphs_expected": {"a3_amine_aldehyde_alkyne": 34104, "ketone_ugi4": 27027},
            "scope": "All TRAIN component identities witnessed in exact admitted mapped source graphs; nested compatibility census only, never new assembly or new chemistry admission.",
            "stages": list(STAGES),
            "CPU_seconds_cap": 570,
            "threads": 1,
            "profile": "First 200 source records, abort if projected total exceeds 520 CPU seconds.",
            "ring_basis": "Actual saved TRAIN ordered tree fundamental cycles; subset of unchanged request allowed sizes, no SSSR substitution.",
            "fixed_core": "Exact anonymous assembly-core signature including quantities, core labels, positions, graph and total incident valence; ordered per-role core correspondence retained.",
            "head_scope": "Ketone Ugi4 pinned retained-head/site predicates. A3 has no borrowed head policy; original admitted repeated-event context preserved through source template.",
            "not_claimed": [
                "full mixed-component assembly compatibility",
                "route closure for new identities",
                "independent lipid realism",
                "sampling or retraining improvement",
            ],
            "new_molecules_model_source_executor_TEST_network_GPU_calls": 0,
        },
    )


def run():
    start = time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (570, 575))
    torch.set_num_threads(1)
    files = paths()
    protocol = read(OUT / "protocol.json")
    for name, value in protocol["inputs"].items():
        if pin(files[name]) != value:
            raise ValueError("Changed input " + name)
    assert Path(
        __import__("forge.corpus.mapped_program_cache", fromlist=["x"]).__file__
    ).is_relative_to(WORKTREE)
    census = read(files["TRAIN_census"])
    reference = read(files["reference"])
    if reference["heldout_structures_used"] is not False:
        raise ValueError("Reference is not TRAIN-only")
    aliases = role_aliases(read(files["candidate_diagnostics"])["candidates"])
    layouts = torch.load(files["layouts"], weights_only=False, map_location="cpu")["layouts"]
    requested = {}
    templates = {}
    for i, layout in enumerate(layouts):
        if layout.family in FAMILIES:
            bundle, requirements = layout_signature(layout)
            key = encoded(bundle)
            templates[key] = bundle
            requested[i] = {"family": layout.family, "template": key, "requirements": requirements}
    assert len(requested) == 128 and Counter(r["family"] for r in requested.values()) == {
        f: 64 for f in FAMILIES
    }
    reaction = read(files["registry"])["reactions"][0]
    adjudication = read(files["adjudication"])
    contract = adjudication["families"]["ketone_ugi4"]
    heads = {}
    for row in census["all22_TRAIN_role_counts"]["ketone_ugi4"]["amine_head"][
        "identities"
    ].values():
        heads[row["smiles"]] = retained_head_witness(row["smiles"], reaction, contract)
    write(OUT / "head_witnesses.json", heads)
    profiles = {}
    by_family = Counter()
    with (
        sqlite3.connect(files["weights"].as_uri() + "?mode=ro", uri=True) as db,
        MappedProgramCache(ROOT, manifest=pin(files["manifest"])) as cache,
    ):
        rows = db.execute(
            "SELECT record_index,target_id,family,probability FROM weights WHERE family IN (?,?) ORDER BY record_index",
            FAMILIES,
        )
        processed = 0
        iteration_start = time.process_time()
        for index, target, family, probability in rows:
            example = cache.record(int(index))
            require_source_correspondence(target, family, example)
            bundle, options, rings = summarize_layout(example)
            template = encoded(bundle)
            templates.setdefault(template, bundle)
            blocks = {b.role: b.role_state for b in example.record.component_blocks}
            for role, identity, quantity in example.component_instances:
                state = blocks[role]
                shape = options[state]
                smiles = reference["component_smiles_by_identity"][identity]
                if identity not in census["all22_TRAIN_role_counts"][family][role]["identities"]:
                    raise ValueError("Component not in exact source TRAIN census")
                profile = {
                    "family": family,
                    "role": role,
                    "identity": identity,
                    "smiles": smiles,
                    "quantity": quantity,
                    "template": template,
                    "positions": shape["positions"],
                    "block_sizes": list(map(len, shape["positions"])),
                    "morphology": shape["morphology"],
                    "closures": shape["closures"],
                    "ring_sizes": rings.get(state, []),
                    "head_status": (
                        heads[smiles]["status"]
                        if family == "ketone_ugi4" and role == "amine_head"
                        else "not_applicable"
                    ),
                }
                key = encoded(profile)
                if key not in profiles:
                    profiles[key] = {
                        **profile,
                        "witness_index": int(index),
                        "witness_target": target,
                        "TRAIN_graphs": 0,
                        "TRAIN_probability_mass": 0.0,
                    }
                profiles[key]["TRAIN_graphs"] += 1
                profiles[key]["TRAIN_probability_mass"] += probability
            by_family[family] += 1
            processed += 1
            if processed == 200:
                elapsed = time.process_time() - iteration_start
                projection = time.process_time() - start + elapsed / processed * (61131 - processed)
                write(
                    OUT / "profile.json",
                    {
                        "first_records": processed,
                        "iteration_CPU": elapsed,
                        "elapsed_CPU": time.process_time() - start,
                        "projected_total_CPU": projection,
                    },
                )
                print(json.dumps({"profile_projected_CPU": projection}), flush=True)
                if projection > 520:
                    raise RuntimeError("Projected census cost exceeds authorized CPU envelope")
            if processed % 10000 == 0:
                print(
                    json.dumps(
                        {"TRAIN_records": processed, "CPU_seconds": time.process_time() - start}
                    ),
                    flush=True,
                )
    assert dict(by_family) == protocol["TRAIN_graphs_expected"]
    write(
        OUT / "TRAIN_profiles.json",
        {"templates": templates, "profiles": list(profiles.values()), "by_family": dict(by_family)},
    )
    by_role = defaultdict(list)
    for j, profile in enumerate(profiles.values()):
        by_role[profile["family"], profile["role"]].append((j, profile))
    original = {
        c["request"]: c
        for f in read(files["current"])["by_family"].values()
        for c in f["selections"]
    }
    joint = read(files["joint"])
    joint_selected = {c["request"]: c for c in joint["selection"]}
    joint_gates = {r["request"]: r["design_assessment"] for r in joint["selected_evidence"]}
    gates = {(r["index"], r["ordinal"]): r["assessment"] for r in read(files["gates"])["attempts"]}
    request_results = []
    for i, request in requested.items():
        family, roles = request["family"], {}
        for role, requirement in request["requirements"].items():
            if requirement["quantity"] is None:
                roles[role] = {
                    "status": "assembly_introduced_not_a_replaceable_precursor",
                    "requirement": requirement,
                }
                continue
            staged = {stage: set() for stage in STAGES}
            witnesses = defaultdict(list)
            for j, profile in by_role[family, role]:
                for stage, passed in zip(
                    STAGES,
                    compatible_stages(profile, request["template"], requirement),
                    strict=True,
                ):
                    if passed:
                        staged[stage].add(profile["identity"])
                if profile["identity"] in staged[STAGES[-1]] and all(
                    compatible_stages(profile, request["template"], requirement)
                ):
                    witnesses[profile["identity"]].append(j)
            matching = staged[STAGES[-1]]
            observed = {}
            for label, chosen in (("original", original[i]), ("joint", joint_selected[i])):
                mapped = {aliases[family, c["role"]]: c for c in chosen["components"]}
                part = mapped.get(role)
                observed[label] = {
                    "smiles": part["smiles"] if part else None,
                    "source_exact": chosen["exact"],
                    "identity_among_necessary_matches": bool(
                        part
                        and any(
                            reference["component_smiles_by_identity"][k] == part["smiles"]
                            for k in matching
                        )
                    ),
                    "TRAIN_novel_by_saved_metric": part["role_train_novel"] if part else None,
                }
            roles[role] = {
                "status": (
                    "necessary_support_present" if matching else "no_match_under_current_contract"
                ),
                "requirement": requirement,
                "staged_unique_counts": {k: len(v) for k, v in staged.items()},
                "matching_component_ids": sorted(matching),
                "profile_witnesses": dict(witnesses),
                "selected": observed,
            }
        local = [r for r in roles.values() if "matching_component_ids" in r]
        oldgate = gates[i, original[i]["ordinal"]]
        request_results.append(
            {
                "request": i,
                "family": family,
                "template": request["template"],
                "roles": roles,
                "all_precursor_roles_have_necessary_support": all(
                    r["matching_component_ids"] for r in local
                ),
                "full_assembly_compatibility": "unassessed_no_component_recombination_or_source_executor_calls",
                "original_design": oldgate["qualified_design_pass"],
                "original_axes": oldgate["status_by_axis"],
                "joint_design": joint_gates[i]["qualified_design_pass"],
                "joint_axes": joint_gates[i]["status_by_axis"],
            }
        )
    summary = {}
    for family in FAMILIES:
        rows = [r for r in request_results if r["family"] == family]
        role_summary = {}
        for role in sorted(
            {k for row in rows for k, v in row["roles"].items() if "matching_component_ids" in v}
        ):
            parts = [row["roles"][role] for row in rows]
            role_summary[role] = {
                "requests": len(parts),
                "requests_with_matches": sum(bool(p["matching_component_ids"]) for p in parts),
                "unique_compatible_components_union": len(
                    {k for p in parts for k in p["matching_component_ids"]}
                ),
                "staged_requests_with_support": {
                    s: sum(p["staged_unique_counts"][s] > 0 for p in parts) for s in STAGES
                },
                "original_identity_is_match": sum(
                    p["selected"]["original"]["identity_among_necessary_matches"] for p in parts
                ),
                "joint_identity_is_match": sum(
                    p["selected"]["joint"]["identity_among_necessary_matches"] for p in parts
                ),
            }
        summary[family] = {
            "requests": len(rows),
            "all_roles_have_necessary_support": sum(
                r["all_precursor_roles_have_necessary_support"] for r in rows
            ),
            "joint_design_failures": sum(not r["joint_design"] for r in rows),
            "joint_design_failures_with_all_role_support": sum(
                not r["joint_design"] and r["all_precursor_roles_have_necessary_support"]
                for r in rows
            ),
            "roles": role_summary,
        }
    write(
        OUT / "result.json",
        {
            "complete": True,
            "protocol": pin(OUT / "protocol.json"),
            "TRAIN_profiles": pin(OUT / "TRAIN_profiles.json"),
            "head_witnesses": pin(OUT / "head_witnesses.json"),
            "requests": request_results,
            "summary": summary,
            "TRAIN_graphs": dict(by_family),
            "CPU_seconds": time.process_time() - start,
            "new_molecules_model_source_executor_TEST_network_GPU_calls": 0,
            "claim_scope": "Necessary compatibility and source-record witnesses, not admitted recombined assemblies, independent realism or decoder success.",
            "production_or_official_cohort_changed": False,
        },
    )
    print(
        json.dumps(
            {"result": pin(OUT / "result.json"), "CPU_seconds": time.process_time() - start}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    else:
        try:
            run()
        except BaseException as exc:
            if not (OUT / "failure.json").exists():
                write(
                    OUT / "failure.json",
                    {"complete": False, "error": type(exc).__name__, "message": str(exc)},
                )
            raise

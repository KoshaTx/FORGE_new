"""Matched complete-component D0/D1 construction on immutable saved logits."""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from rdkit import rdBase

from forge.core.hashing import resolve_pin
from forge.core.io import write_json
from forge.corpus.compose_lipid_source_view import pin
from forge.model.compose_lipid_component_constraints import propose_component_constraints
from forge.model.compose_lipid_component_policy import compile_component_policies
from forge.model.compose_lipid_family_rules import reserve_ordered_core_attachments
from forge.model.compose_lipid_generation import constrained_readout
from forge.model.compose_lipid_layout import collate_generated_layouts
from forge.model.compose_lipid_scaffold_construction import construct_scaffold_proposals
from forge.model.compose_lipid_source_constraints import SourceConstraints, compile_constraints
from forge.model.compose_lipid_symmetric_arms import construct_symmetric_role_arms
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph
from results.phase1.compose_lipid_component_decoder_v1.contracts import (
    MIRROR,
    ROOT,
    assess,
    load_all,
    matching,
)
from results.phase1.compose_lipid_component_decoder_v1.policies import (
    compile_for_executor,
    load_proposal_domains,
)
from results.phase1.compose_lipid_family_rules_v2.run import load

HERE = Path(__file__).resolve().parent
FRESH = HERE.with_name("compose_lipid_component_decoder_v1") / "fresh"
OUT = HERE / "first_draw"
LIMITS = dict(
    maximum_policies=4,
    maximum_edits=4,
    maximum_mutations=128,
    maximum_donors=64,
    maximum_constructor_seeds=4,
    maximum_seconds=600,
)


def serial_state(nodes, edges, layout):
    """Recover an ordered spanning tree without changing any generated bond."""
    record = layout.record
    nodes, edges = np.asarray(nodes), np.asarray(edges)
    if len(nodes) != record.node_count or not fixed_graph_preserved(nodes, edges, record):
        raise ValueError("Constructed graph changed size or fixed core")
    if np.count_nonzero(np.triu(edges)) != len(nodes) - 1 + record.graph.closure_count:
        raise ValueError("Constructed graph changed closure count")
    parents, bonds, tree = [0] * len(nodes), [0] * len(nodes), set()
    for child in range(1, len(nodes)):
        options = np.flatnonzero(edges[child, :child])
        if not len(options):
            return None
        parent = (
            int(record.graph.parents[child])
            if record.fixed_parent_bond_mask[child]
            else int(options[0])
        )
        if not edges[parent, child]:
            raise ValueError("Constructed graph removed a fixed tree bond")
        parents[child], bonds[child] = parent, int(edges[parent, child]) - 1
        tree.add((parent, child))
    closures = [
        (a, b) for a, b in zip(*np.nonzero(np.triu(edges)), strict=True) if (a, b) not in tree
    ]
    return dict(
        nodes=nodes.tolist(),
        parents=parents,
        parent_bonds=bonds,
        closure_left=[int(a) for a, _ in closures],
        closure_right=[int(b) for _, b in closures],
        closure_bonds=[int(edges[a, b]) - 1 for a, b in closures],
    )


def construct_one(
    layout, raw, pred, atoms, found, policies, domains, constraint, reactions, vocabulary, aware
):
    proposals, construction, memo = [], [], {}
    costs = Counter(
        readouts=1, source_checks=0, donor_combinations=0, mutations=0, constructor_seeds=0
    )

    def add(smiles, kind, proposal=None):
        if not smiles or smiles in memo:
            return
        if proposal is not None and "nodes" in proposal:
            nodes, edges = np.asarray(proposal["nodes"]), np.asarray(proposal["edges"])
            serial_state(nodes, edges, layout)
            if graph_smiles(nodes, edges, atoms) != smiles:
                raise ValueError("Proposal graph and identity disagree")
        memo[smiles] = assess(found, layout, smiles)
        costs["source_checks"] += 1
        proposals.append(
            dict(
                kind=kind,
                smiles=smiles,
                check=memo[smiles],
                **(
                    {k: proposal[k] for k in ("nodes", "edges") if k in proposal}
                    if proposal
                    else {}
                ),
            )
        )

    def repair(state, label, policies_to_use):
        for policy_index, policy in enumerate(policies_to_use):
            value = propose_component_constraints(
                layout,
                state,
                pred,
                atoms,
                policy,
                core_aliases=constraint.core_aliases,
                maximum_edits=LIMITS["maximum_edits"],
                maximum_mutations=LIMITS["maximum_mutations"],
                maximum_donors=LIMITS["maximum_donors"],
                allow_branch_edits=True,
            )
            costs["component_construction_calls"] += 1
            costs["donor_combinations"] += value.get("donors_considered", 0)
            costs["mutations"] += value.get("mutations_evaluated", 0)
            construction.append(dict(kind=label, policy_index=policy_index, result=value))
            for proposal in value["proposals"]:
                add(
                    proposal.get("smiles"),
                    f"{label}:policy{policy_index}:{proposal['policy']}",
                    proposal,
                )

    if not raw["reason"]:
        nodes, edges = state_graph(raw["state"])
        smiles = graph_smiles(nodes, edges, atoms)
        add(smiles, "raw", dict(nodes=nodes.tolist(), edges=edges.tolist()))
        raw.update(smiles=smiles, check=memo.get(smiles, dict(exact=False, reason="invalid_graph")))
        if smiles:
            repair(raw["state"], "raw_component", policies)
    else:
        raw.update(smiles=None, check=dict(exact=False, reason=raw["reason"]))
    if layout.family in reactions:
        value = construct_scaffold_proposals(
            layout, pred, atoms, reactions[layout.family], atom_aware_topology=aware
        )
        costs["scaffold_readouts"] += len(value["proposals"])
        construction.append(dict(kind="registered_scaffold", result=value))
        for proposal in value["proposals"]:
            if not proposal.get("smiles"):
                continue
            if costs["constructor_seeds"] >= LIMITS["maximum_constructor_seeds"]:
                construction.append(
                    dict(kind="registered_scaffold", status="constructor_seed_budget_exhausted")
                )
                break
            costs["constructor_seeds"] += 1
            add(proposal["smiles"], "registered_scaffold:" + proposal["scaffold_id"], proposal)
            state = serial_state(proposal["nodes"], proposal["edges"], layout)
            if state is not None:
                repair(state, "scaffold_component:" + proposal["scaffold_id"], policies)
    if layout.family == "maleate_addition":
        for executor_index, executor in enumerate(matching(found, layout)):
            adapter = executor["adapter"]
            role = next(r for r in adapter.roles if executor["mapping"][r] == "maleate")
            value = construct_symmetric_role_arms(
                layout, pred, atoms, adapter, role, vocabulary, atom_aware_topology=aware
            )
            construction.append(
                dict(kind="template_symmetric_arms", executor_index=executor_index, result=value)
            )
            costs["symmetric_readouts"] += 1
            policy = compile_component_policies(executor)
            head = executor["program"]["accumulator_role"]
            definition = next(
                r for r in adapter.reaction.definition.reactant_roles if r.name == head
            )
            policy.append(
                dict(
                    role=executor["mapping"][head],
                    remaining_handles=[
                        dict(
                            smarts=definition.required_handle_smarts,
                            maximum_matches=0,
                            exclude_core=True,
                        )
                    ],
                )
            )
            for proposal in value["proposals"]:
                if costs["constructor_seeds"] >= LIMITS["maximum_constructor_seeds"]:
                    construction.append(
                        dict(
                            kind="template_symmetric_arms",
                            status="constructor_seed_budget_exhausted",
                        )
                    )
                    break
                costs["constructor_seeds"] += 1
                add(proposal.get("smiles"), "template_symmetric_arms", proposal)
                repair(proposal["state"], "symmetric_component", [policy])
    return dict(raw=raw, proposals=proposals, construction=construction, costs=dict(costs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="First8 requests per family; shards are reused by complete run",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    parent = json.loads((FRESH / "protocol.json").read_text())
    receipt = json.loads((FRESH / "generation.json").read_text())
    inputs = {k: parent["inputs"][k] for k in ("payload", "bindings")}
    inputs.update(
        generation=pin(ROOT, FRESH / "generation.json"),
        source_protocol=pin(ROOT, FRESH / "protocol.json"),
        scaffold_registry=pin(
            ROOT, ROOT / "data/vendor/qualified_reductive_source_program_v1.json"
        ),
    )
    protocol = dict(
        schema_version="forge.quality_complete_decode.v2",
        inputs=inputs,
        branches={"d0": {"atom_aware_topology": False}, "d1": {"atom_aware_topology": True}},
        limits=LIMITS,
        seed=parent["seed"],
        layout_seed=parent["layout_seed"],
        scope="First-draw TRAIN-derived development; identical full construction budgets, no model calls",
        core_reservation="ordered for every family, identical D0/D1",
        process_previous_successes=True,
        source_selection=False,
        new_model_calls=0,
        constructor_seed_order="raw repairs; registered scaffolds and repairs; symmetric arms and repairs",
        policy_order="canonical sorted JSON",
        maximum_policies=4,
        implementation=[
            pin(ROOT, Path(module.__file__))
            for name, module in sorted(sys.modules.items())
            if name.startswith("forge.") and getattr(module, "__file__", "").endswith(".py")
        ]
        + [
            pin(ROOT, Path(__file__)),
            pin(ROOT, FRESH.parent / "contracts.py"),
            pin(ROOT, FRESH.parent / "policies.py"),
        ],
    )
    if (OUT / "protocol.json").exists():
        if json.loads((OUT / "protocol.json").read_text()) != protocol:
            raise ValueError(
                "Started protocol or implementation changed; use a new output directory"
            )
    else:
        write_json(OUT / "protocol.json", protocol)
    hashes = load(
        MIRROR / "results/phase1/compose_lipid_posttraining_v1/adjudicate.py",
        "quality_decode_hashes",
    ).VerifiedHashes()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    elapsed = 0.0
    with patch("forge.core.hashing.sha256_file", hashes), rdBase.BlockLogs():
        for key, value in inputs.items():
            resolve_pin(value, ROOT, label=key)
        payload = torch.load(
            resolve_pin(inputs["payload"], ROOT, label="payload"), weights_only=False
        )
        found, sources, controls = load_all()
        controls_path = OUT / "source-controls.json"
        if not controls_path.exists():
            write_json(controls_path, controls)
        domains = load_proposal_domains()
        reactions = {
            r["reaction_id"].removeprefix("source_"): r
            for r in json.loads(
                resolve_pin(inputs["scaffold_registry"], ROOT, label="scaffolds").read_text()
            )["reactions"]
        }
        constraints = {}
        for binding in json.loads(
            resolve_pin(inputs["bindings"], ROOT, label="bindings").read_text()
        )["bindings"]:
            constraints[binding["program_id"]] = compile_constraints(
                json.loads(resolve_pin(binding["registry"], ROOT, label="registry").read_text()),
                program_id=binding["program_id"],
                core_vocabulary=payload["vocabulary"]["core_position_states"],
                **{k: binding[k] for k in ("sequential_program", "reaction_id") if k in binding},
            )
        rows, receipts = [], []
        selected_shards = [s for s in receipt["batches"] if not args.pilot or s["offset"] % 64 == 0]
        for shard in selected_shards:
            offset = shard["offset"]
            output, receipt_path = (
                OUT / f"assessed-{offset:04d}.json",
                OUT / f"receipt-{offset:04d}.json",
            )
            if receipt_path.exists():
                saved = json.loads(receipt_path.read_text())
                resolve_pin(saved["assessed"], ROOT, label="completed shard")
                resolve_pin(saved["predictions"], ROOT, label="completed saved logits")
                local = json.loads(output.read_text())
                elapsed += saved["seconds"]
            else:
                if elapsed >= LIMITS["maximum_seconds"]:
                    raise TimeoutError("Cumulative immutable-shard CPU budget exhausted")
                tick = time.monotonic()
                pred = torch.load(
                    resolve_pin(shard["predictions"], ROOT, label="saved logits"),
                    weights_only=False,
                )
                layouts = payload["layouts"][offset : offset + 8]
                batch = collate_generated_layouts(layouts, maximum_closures=12)
                reserved, failures = [], []
                for j, layout in enumerate(layouts):
                    n = layout.record.node_count
                    value, reason = reserve_ordered_core_attachments(
                        layout, pred["parents"][j, :n, :n].numpy()
                    )
                    reserved.append(value or layout)
                    failures.append(reason)
                readouts = {
                    branch: constrained_readout(pred, batch, reserved, payload["atoms"], **kwargs)
                    for branch, kwargs in protocol["branches"].items()
                }
                local = []
                for j, layout in enumerate(layouts):
                    row = dict(index=offset + j, family=layout.family, draw=0, branches={})
                    policies = {
                        json.dumps(
                            compile_for_executor(e, domains), sort_keys=True
                        ): compile_for_executor(e, domains)
                        for e in matching(found, layout)
                    }
                    if len(policies) > LIMITS["maximum_policies"]:
                        raise ValueError("Source policy count exceeds frozen bound")
                    ordered = [v for _, v in sorted(policies.items())]
                    constraint = constraints.get(
                        layout.record.program_id, SourceConstraints(layout.record.program_id)
                    )
                    for branch, (state, reasons) in readouts.items():
                        raw = dict(
                            reason=failures[j] or reasons[j],
                            state={
                                k: v[
                                    j,
                                    : (
                                        layout.record.graph.closure_count
                                        if k.startswith("closure")
                                        else layout.record.node_count
                                    ),
                                ].tolist()
                                for k, v in state.items()
                            },
                        )
                        row["branches"][branch] = construct_one(
                            layout,
                            raw,
                            {k: v[j].numpy() for k, v in pred.items()},
                            payload["atoms"],
                            found,
                            ordered,
                            domains,
                            constraint,
                            reactions,
                            payload["vocabulary"]["core_position_states"],
                            branch == "d1",
                        )
                    local.append(row)
                write_json(output, local)
                saved = dict(
                    offset=offset,
                    predictions=shard["predictions"],
                    assessed=pin(ROOT, output),
                    seconds=time.monotonic() - tick,
                )
                write_json(receipt_path, saved)
                elapsed += saved["seconds"]
                print(
                    json.dumps(
                        dict(
                            offset=offset,
                            seconds=round(saved["seconds"], 3),
                            cumulative=round(elapsed, 3),
                        )
                    ),
                    flush=True,
                )
            rows.extend(local)
            receipts.append(saved)
            write_json(
                OUT / "progress.json",
                dict(requests_completed=len(rows), seconds=elapsed, pilot=args.pilot),
            )
        hashes.validate()
        totals = defaultdict(Counter)
        for row in rows:
            total = totals[row["family"]]
            total["requests"] += 1
            for branch, value in row["branches"].items():
                total[branch + "_raw_exact"] += int(value["raw"]["check"]["exact"])
                total[branch + "_constructed_exact_available"] += int(
                    any(p["check"]["exact"] for p in value["proposals"])
                )
                for key, v in value["costs"].items():
                    total[branch + "_" + key] += v
        stem = "pilot" if args.pilot else "result"
        ledger = OUT / ("pilot_attempts.json" if args.pilot else "attempts.json")
        write_json(ledger, rows)
        write_json(
            OUT / (stem + ".json"),
            dict(
                totals={k: dict(v) for k, v in totals.items()},
                seconds=elapsed,
                complete=not args.pilot,
                inputs=dict(protocol=pin(ROOT, OUT / "protocol.json"), attempts=pin(ROOT, ledger)),
                shards=receipts,
                source_inputs=sources,
                verified_inputs=[
                    dict(path=str(p.relative_to(ROOT)), sha256=str(d))
                    for p, (_, d) in hashes.entries.items()
                ],
                quality_promoted=False,
                selected=False,
            ),
        )


if __name__ == "__main__":
    main()

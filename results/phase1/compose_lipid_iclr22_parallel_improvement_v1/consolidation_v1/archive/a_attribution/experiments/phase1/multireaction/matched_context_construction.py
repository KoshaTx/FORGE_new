"""Parameterized frozen constructors; outputs stay under the caller's new experiment."""

import copy
import json

from forge.core.hashing import resolve_pin
from forge.model.compose_lipid_family_rules import reserve_ordered_core_attachments
from forge.model.compose_lipid_generation import constrained_readout
from forge.model.compose_lipid_layout import collate_generated_layouts
from forge.model.compose_lipid_quality_generation import with_retained_component_domain
from forge.model.compose_lipid_source_constraints import SourceConstraints, compile_constraints
from results.phase1.compose_lipid_quality_decode_v2 import run


def setup(root, protocol, payload):
    for p in protocol["inputs"].values():
        resolve_pin(p, root, label="frozen construction input")
    found, sources, controls = run.load_all()
    expected = json.loads(
        resolve_pin(
            protocol["inputs"]["source_controls"], root, label="source controls"
        ).read_text()
    )
    if controls != expected or protocol["limits"] != run.LIMITS:
        raise ValueError("Source controls or constructor budgets differ")
    domains = run.load_proposal_domains()
    reactions = {
        r["reaction_id"].removeprefix("source_"): r
        for r in json.loads(
            resolve_pin(
                protocol["inputs"]["scaffold_registry"], root, label="scaffolds"
            ).read_text()
        )["reactions"]
    }
    constraints = {}
    for binding in json.loads(
        resolve_pin(protocol["inputs"]["bindings"], root, label="bindings").read_text()
    )["bindings"]:
        constraints[binding["program_id"]] = compile_constraints(
            json.loads(resolve_pin(binding["registry"], root, label="registry").read_text()),
            program_id=binding["program_id"],
            core_vocabulary=payload["vocabulary"]["core_position_states"],
            **{k: binding[k] for k in ("sequential_program", "reaction_id") if k in binding},
        )
    policies, domain_policies = {}, {}
    for i, layout in enumerate(payload["layouts"]):
        base, domain = {}, {}
        for executor in run.matching(found, layout):
            value = run.compile_for_executor(executor, domains)
            base[json.dumps(value, sort_keys=True)] = value
            if layout.family == "aema_aza_thiol_addition":
                value = with_retained_component_domain(executor, value)
                domain[json.dumps(value, sort_keys=True)] = value
        if (
            len(base) > run.LIMITS["maximum_policies"]
            or len(domain) > run.LIMITS["maximum_policies"]
        ):
            raise ValueError("Constructor policy budget exceeded")
        policies[i] = [v for _, v in sorted(base.items())]
        if domain:
            domain_policies[i] = [v for _, v in sorted(domain.items())]
    return {
        "found": found,
        "sources": sources,
        "controls": controls,
        "domains": domains,
        "reactions": reactions,
        "constraints": constraints,
        "policies": policies,
        "domain_policies": domain_policies,
    }


def construct(context, payload, pred, *, draw, offset, mode="base", base_rows=None):
    """Same base/domain D0/D1 readout, ordering and caps as original confirmation."""
    layouts = payload["layouts"][offset : offset + 8]
    if mode not in ("base", "domain") or (mode == "domain" and base_rows is None):
        raise ValueError("Invalid constructor shard mode")
    if mode == "base":
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
            branch: constrained_readout(
                pred, batch, reserved, payload["atoms"], atom_aware_topology=branch == "d1"
            )
            for branch in ("d0", "d1")
        }
    rows = []
    for j, layout in enumerate(layouts):
        index = offset + j
        row = {"index": index, "family": layout.family, "draw": draw, "branches": {}}
        for branch in ("d0", "d1"):
            if mode == "base":
                states, reasons = readouts[branch]
                raw = {
                    "reason": failures[j] or reasons[j],
                    "state": {
                        k: v[
                            j,
                            : (
                                layout.record.graph.closure_count
                                if k.startswith("closure")
                                else layout.record.node_count
                            ),
                        ].tolist()
                        for k, v in states.items()
                    },
                }
            else:
                raw = copy.deepcopy(base_rows[j]["branches"][branch]["raw"])
            value = run.construct_one(
                layout,
                raw,
                {k: v[j].numpy() for k, v in pred.items()},
                payload["atoms"],
                context["found"],
                context["policies" if mode == "base" else "domain_policies"][index],
                context["domains"],
                context["constraints"].get(
                    layout.record.program_id, SourceConstraints(layout.record.program_id)
                ),
                context["reactions"] if mode == "base" else {},
                payload["vocabulary"]["core_position_states"],
                branch == "d1",
            )
            if mode == "domain":
                value["costs"]["readouts"] = 0
                value["costs"]["reused_raw_readouts"] = 1
            row["branches"][branch] = value
        rows.append(row)
    return rows

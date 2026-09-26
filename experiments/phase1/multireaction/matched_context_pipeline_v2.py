"""Parameterize the frozen five-draw constructor and arm-own full D1 selector."""

import copy
import gzip
import importlib.util
import json
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from rdkit import rdBase

from forge.core.hashing import resolve_pin, sha256_file
from forge.model.compose_lipid_quality_selection import first_exact, select_reference_supported
from results.phase1.compose_lipid_quality_confirmation_v1.select import reference_support_read_only
from results.phase1.compose_lipid_quality_decode_v2.compact import compact_row
from results.phase1.compose_lipid_quality_selection_v2.run import candidate_pools
from results.phase1.compose_lipid_structure_repair_v1.evaluation import gates as gate_module
from results.phase1.compose_lipid_structure_repair_v1.evaluation.gates import (
    assess_design,
    load_gate_context,
)
from results.phase1.compose_lipid_structure_repair_v1.evaluation.replay import load_candidate_graphs


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def construct_all(root, out, payload, generation, context, constructor, pin, write, commit):
    """Closed receipts are resumable; never alter seeds, caps, order or constructor outputs."""
    constructed = out / "constructed"
    constructed.mkdir(exist_ok=True)
    schedule = [("base", r) for r in generation] + [
        ("domain", r) for r in generation if r["offset"] in context["domain_policies"]
    ]
    combined = {}
    receipts = []
    for mode, entry in schedule:
        draw, offset = entry["draw"], entry["offset"]
        stem = constructed / f"{mode}-{draw}-{offset:04d}"
        receipt_path = stem.with_suffix(".receipt.json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["predictions"] != entry["predictions"]:
                raise ValueError("Constructor resume logits changed")
            for key in ("full", "compact"):
                resolve_pin(receipt[key], root, label="saved constructor shard")
            compact = json.loads((root / receipt["compact"]["path"]).read_text())
        else:
            for suffix in (".json.gz", ".compact.json"):
                if stem.with_suffix(suffix).exists():
                    raise ValueError("Unclosed constructor shard; preserve before manual recovery")
            started = time.monotonic()
            predictions = torch.load(
                resolve_pin(entry["predictions"], root, label="arm terminal logits"),
                weights_only=False,
                map_location="cpu",
            )
            baseline = None
            if mode == "domain":
                baseline = json.loads(
                    gzip.decompress(
                        (constructed / f"base-{draw}-{offset:04d}.json.gz").read_bytes()
                    )
                )
            rows = constructor.construct(
                context,
                payload,
                predictions,
                draw=draw,
                offset=offset,
                mode=mode,
                base_rows=baseline,
            )
            full_path = stem.with_suffix(".json.gz")
            with full_path.open("xb") as stream:
                stream.write(
                    gzip.compress(
                        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode(),
                        compresslevel=1,
                        mtime=0,
                    )
                )
            full = pin(full_path)
            compact = [compact_row(row, full) for row in rows]
            write(stem.with_suffix(".compact.json"), compact)
            receipt = {
                "mode": mode,
                "draw": draw,
                "offset": offset,
                "predictions": entry["predictions"],
                "full": full,
                "compact": pin(stem.with_suffix(".compact.json")),
                "seconds": time.monotonic() - started,
            }
            write(receipt_path, receipt)
            commit()
        receipts.append(receipt)
        for row in compact:
            index = row["index"]
            if mode == "base" and draw == 0:
                combined[index] = copy.deepcopy(row)
                continue
            for branch, value in row["branches"].items():
                dest = combined[index]["branches"][branch]
                followup = {
                    "draw": draw,
                    "costs": value["costs"],
                    "construction_provenance": value["construction_provenance"],
                }
                if mode == "base":
                    followup["raw"] = value["raw"]
                dest.setdefault(
                    "additional_draws" if mode == "base" else "domain_followups", []
                ).append(followup)
                prefix = f"draw{draw}:" if mode == "base" else f"retained_domain:draw{draw}:"
                dest["proposals"].extend(
                    {**p, "kind": prefix + p["kind"]} for p in value["proposals"]
                )
                for k, v in value["costs"].items():
                    dest["costs"][k] = dest["costs"].get(k, 0) + v
    if sorted(combined) != list(range(1408)) or len(generation) != 880 or len(receipts) != 920:
        raise ValueError("Full constructor denominator changed")
    ledger = [combined[i] for i in range(1408)]
    path = out / "compact_attempts.json"
    if path.exists():
        if json.loads(path.read_text()) != ledger:
            raise ValueError("Existing compact pool changed")
    else:
        write(path, ledger)
    return ledger, receipts


def selection(root, out, payload, rows, pin, write, commit):
    """Each model arm uses its own support baseline; no conditioned outcomes are inserted."""
    base = root / "results/phase1"
    frozen = base / "compose_lipid_quality_selection_v2"
    quality = base / "compose_lipid_quality_v1"
    evaluation = base / "compose_lipid_structure_repair_v1/evaluation"
    policy = json.loads((frozen / "policy.json").read_text())
    reference = json.loads((quality / "reference.json").read_text())
    frozen_pins = json.loads((frozen / "aema_domain/protocol.json").read_text())["inputs"]
    product_support, component_support, _ = reference_support_read_only(frozen_pins)
    from results.phase1.compose_lipid_component_decoder_v1.contracts import load_all

    executors, _, _ = load_all()
    membership = base / "compose_lipid_mapped_preparation_v3/cache/sources.sqlite"
    with sqlite3.connect(membership.as_uri() + "?mode=ro", uri=True) as db:
        pools, diagnostics, _ = candidate_pools(
            rows,
            "d1",
            product_support,
            component_support,
            reference,
            payload["layouts"],
            executors,
            db,
        )
    support, first = {}, {}
    support_reports = {}
    for family, groups in sorted(pools.items()):
        first[family] = [first_exact(group) for group in groups]
        support[family], support_reports[family] = select_reference_supported(
            groups,
            node_limit=policy["support_aware"]["solver"]["node_limit"],
            time_limit=policy["support_aware"]["solver"]["time_limit_seconds_per_stage"],
        )
    selector = load(evaluation / "design_selector.py", "matched_full_design_selector")

    # Dataclass types originate from separate frozen modules; convert at this boundary.
    def convert(c):
        value = asdict(c)
        return selector.SelectionCandidate(
            **{
                **value,
                "components": tuple(selector.ComponentIdentity(**p) for p in value["components"]),
            }
        )

    all_candidates = {
        (c.request, c.ordinal): convert(c)
        for groups in pools.values()
        for group in groups
        for c in group
    }
    support = {f: [convert(c) for c in selected] for f, selected in support.items()}
    byrequest = {c.request: c for values in support.values() for c in values}
    required = {key for key, c in all_candidates.items() if c.exact or c == byrequest[c.request]}
    # Model imports come from the immutable qualified source, so code pins must
    # authenticate those actual bytes rather than unrelated active-root edits.
    qualified_source = (
        root / "results/phase1/compose_lipid_iclr22_table_completion_v1/null_runtime/source"
    )

    def source_aware_pin(path):
        relative = path.relative_to(root)
        actual = qualified_source / relative if relative.parts[0] == "forge" else path
        return {"path": str(relative), "sha256": str(sha256_file(actual))}

    with patch.object(gate_module, "pin", source_aware_pin):
        context = load_gate_context(root, evaluation / "gate_policy_v3.json")
    gates = {}
    source_pins = {}
    with rdBase.BlockLogs():
        for i, o, nodes, edges, state, basis, source in load_candidate_graphs(
            rows, "d1", required, payload["layouts"]
        ):
            c = all_candidates[i, o]
            gates[i, o] = assess_design(
                payload["layouts"][i], nodes, edges, state, basis, c.smiles, c.exact, context
            )
            if source:
                source_pins[source["path"]] = source

    # Deliberately retain original tree-basis unknowns here. The new pool needs its own
    # authenticated transport ledger before comparing to current1324 basis-dependent metric.
    def codes(c):
        a = gates[c.request, c.ordinal]
        return {f["code"] for f in a["chemical"]["flags"] if f["tier"] == "context_required"}

    validator = load(
        Path(__file__).with_name("verify_saved_pool_controls.py"),
        "matched_selection_independent_counts",
    )
    result = {}
    for family, groups in sorted(pools.items()):
        baseline = support[family]
        allowed = []
        excluded = []
        for group, old in zip(groups, baseline, strict=True):
            values = []
            for value in group:
                c = all_candidates[value.request, value.ordinal]
                if (c.request, c.ordinal) in gates and not codes(c) <= codes(old):
                    excluded.append([c.request, c.ordinal])
                else:
                    values.append(c)
            allowed.append(values)
        clean = frozenset(
            (c.request, c.ordinal)
            for group in allowed
            for c in group
            if (assessment := gates.get((c.request, c.ordinal)))
            and assessment["qualified_design_pass"]
        )
        chosen, report = selector.select_design_supported(
            allowed, baseline=baseline, design_pass=clean
        )
        if any(
            c.exact != old.exact or not codes(c) <= codes(old)
            for c, old in zip(chosen, baseline, strict=True)
        ):
            raise ValueError("Per-request exact/context invariant failed")
        before_summary = validator.recount([asdict(c) for c in baseline], gates)
        after_summary = validator.recount([asdict(c) for c in chosen], gates)
        validator.floors(before_summary, after_summary)
        result[family] = {
            "summary": after_summary,
            "support_baseline_summary": before_summary,
            "all_floors_verified": True,
            "selected": [asdict(c) for c in chosen],
            "support_baseline": [asdict(c) for c in baseline],
            "solver": report,
            "context_exclusions": excluded,
        }
    destination = out / "selection"
    destination.mkdir(exist_ok=False)
    write(destination / "candidate_diagnostics.json", {"candidates": diagnostics})
    write(
        destination / "gates.json",
        {
            "attempts": [
                {"index": i, "ordinal": o, "assessment": a} for (i, o), a in sorted(gates.items())
            ],
            "source_pins": source_pins,
        },
    )
    result = {
        "by_family": result,
        "support_reports": support_reports,
        "requests": 1408,
        "metric_basis": "original gate_policy_v3 with explicit unknown tree bases; not latest1324-only basis admission",
        "quality_promotion": False,
        "post_selection_source_inputs": {"pool": pin(out / "compact_attempts.json")},
    }
    write(destination / "result.json", result)
    commit()
    return result

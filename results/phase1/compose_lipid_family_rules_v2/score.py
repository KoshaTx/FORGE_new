"""Score bounded family proposals against restored complete source executors."""

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rdkit import rdBase

from forge.core.hashing import resolve_pin
from forge.core.io import write_json
from forge.corpus.compose_lipid_source_view import pin
from forge.model.compose_lipid_family_rules import (
    admit_family_proposals,
    propose_scaffold_completion,
)
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph
from results.phase1.compose_lipid_family_rules_v2.run import BASE, HERE, MIRROR, ROOT, load


def target_executors():
    from forge.assembly.families import RegistryAssemblyAdapter
    from forge.assembly.repeated_components import RepeatBounds
    from forge.corpus import compose_lipid_fixed_replay as fixed
    from forge.corpus import compose_lipid_scaffold_event as scaffold
    from forge.corpus.compose_lipid_family_replay import _scaffold_replay

    path = MIRROR / "configs/multireaction/compose_lipid_v8_reductive_program_v1.json"
    config = json.loads(path.read_text())
    registry_path = resolve_pin(config["inputs"]["registry"], MIRROR, label="reductive registry")
    registry = json.loads(registry_path.read_text())
    adjudication = json.loads(
        resolve_pin(config["inputs"]["adjudication"], MIRROR, label="adjudication").read_text()
    )
    for name, value in adjudication["assets"].items():
        resolve_pin(value, MIRROR, label=name)
    controls_doc = json.loads(
        resolve_pin(adjudication["assets"]["controls.json"], MIRROR, label="controls").read_text()
    )
    bounds = RepeatBounds(**config["search_bounds"])
    adapters = {}
    reactions = {}
    found = {}
    for family, binding in config["families"].items():
        adapter = RegistryAssemblyAdapter.from_registry(
            registry_path,
            reaction_id=binding["reaction_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
        )
        reaction = next(r for r in registry["reactions"] if r["reaction_id"] == adapter.reaction_id)
        adapters[family], reactions[family] = adapter, reaction
        found[family] = [
            dict(
                kind="fixed",
                run=partial(_scaffold_replay, adapter, reaction, bounds),
                bounds=bounds,
            )
        ]
    controls = {"reductive": scaffold._controls(controls_doc, adapters, reactions, bounds)}
    ketone_path = MIRROR / "configs/multireaction/compose_lipid_supplied_ketone_ugi4_v1.json"
    cfg, _, local, checked = fixed.load_contract(MIRROR, ketone_path)
    controls["ketone"] = checked
    found["ketone_ugi4"] = [
        dict(
            **local["ketone_ugi4"],
            bounds=RepeatBounds(maximum_events=1, maximum_outcomes=cfg["maximum_outcomes"]),
        )
    ]
    inputs = {"reductive": pin(ROOT, path), "ketone": pin(ROOT, ketone_path)}
    checker = load(MIRROR / "results/phase1/compose_lipid_posttraining_v1/adjudicate.py", "checker")
    return checker, found, inputs, controls


def summary(rows, baseline, is_train):
    output = {}
    for family in sorted({r["family"] for r in rows}):
        local = [r for r in rows if r["family"] == family]
        prefix = "baseline" if baseline else "selected"
        counts = Counter(r[prefix + "_smiles"] for r in local if r[prefix + "_smiles"])
        exact = [r for r in local if r[prefix + "_check"]["exact"]]
        components = defaultdict(Counter)
        for row in exact:
            witnesses = {
                tuple(sorted(parts.items()))
                for c in row[prefix + "_check"]["checks"]
                for parts in c.get("accepted_components", [])
            }
            if len(witnesses) != 1:
                raise ValueError("Component diversity requires one accepted precursor tuple")
            for role, smiles in next(iter(witnesses)):
                components[role][smiles] += 1
        output[family] = dict(
            attempts=len(local),
            valid=sum(counts.values()),
            exact=len(exact),
            unique_valid=len(counts),
            unique_exact=len({r[prefix + "_smiles"] for r in exact}),
            novel_valid=sum(n for s, n in counts.items() if not is_train(s)),
            novel_exact=sum(not is_train(r[prefix + "_smiles"]) for r in exact),
            inverse_simpson=(
                sum(counts.values()) ** 2 / sum(n * n for n in counts.values()) if counts else 0
            ),
            exact_component_diversity={
                role: dict(
                    unique=len(c),
                    occurrences=sum(c.values()),
                    inverse_simpson=sum(c.values()) ** 2 / sum(n * n for n in c.values()),
                )
                for role, c in components.items()
            },
            proposal_statuses=dict(Counter(r["proposal_status"] for r in local)),
        )
    return output


def main():
    for phase in ("paired", "fresh"):
        if (HERE / phase / "result.json").exists():
            raise FileExistsError("Completed scoring is immutable")
    previous = json.loads((BASE / "fresh/attempts.json").read_text())["constrained"]
    membership = json.loads((BASE / "fresh/result.json").read_text())["inputs"]["membership"]
    dbpath = resolve_pin(membership, ROOT, label="TRAIN membership")
    checker0 = load(MIRROR / "results/phase1/compose_lipid_posttraining_v1/adjudicate.py", "hashes")
    hashes = checker0.VerifiedHashes()
    with (
        patch("forge.core.hashing.sha256_file", hashes),
        rdBase.BlockLogs(),
        sqlite3.connect(dbpath.as_uri() + "?mode=ro", uri=True) as db,
    ):
        checker, found, inputs, controls = target_executors()

        def is_train(smiles):
            return (
                db.execute(
                    "SELECT 1 FROM sources WHERE constitution_id=?",
                    (hashlib.sha256(smiles.encode()).hexdigest(),),
                ).fetchone()
                is not None
            )

        for phase in ("paired", "fresh"):
            out = HERE / phase
            p = json.loads((out / "protocol.json").read_text())
            payload = torch.load(
                resolve_pin(p["input"], ROOT, label="payload"),
                map_location="cpu",
                weights_only=False,
            )
            generation = json.loads((out / "generation.json").read_text())
            states = json.loads(resolve_pin(generation["states"], ROOT, label="states").read_text())
            rows = []
            for item in states:
                i = item["index"]
                layout = payload["layouts"][i]
                example = SimpleNamespace(
                    record=layout.record, family=layout.family, source_quantities=layout.quantities
                )
                state = item["baseline"]["state"]
                old = (
                    None
                    if item["baseline"]["reason"]
                    else graph_smiles(*state_graph(state), payload["atoms"])
                )
                if phase == "paired":
                    prior = previous[i]
                    assert old == prior["selected_smiles"] and prior["family"] == layout.family
                checked = (
                    checker.audit_product(found, old, example)
                    if old
                    else dict(status="invalid_product", exact=False)
                )
                row = dict(
                    index=i,
                    family=layout.family,
                    program=layout.record.program_id,
                    baseline_smiles=old,
                    baseline_check=checked,
                    proposal_smiles=None,
                    proposal_status="baseline_exact_immutable",
                )
                if not checked["exact"]:
                    proposal = None
                    if layout.family == "ketone_ugi4":
                        reserved = item["reserved"]
                        row["proposal_status"] = reserved["reason"] or "reserved_core_proposal"
                        if reserved["reason"] is None:
                            nodes, edges = state_graph(reserved["state"])
                            assert fixed_graph_preserved(nodes, edges, layout.record)
                            proposal = graph_smiles(nodes, edges, payload["atoms"])
                    elif old:
                        executor = found[layout.family][0]
                        adapter, reaction, _ = executor["run"].args
                        result = propose_scaffold_completion(
                            layout, state, payload["atoms"], adapter, reaction
                        )
                        row["proposal_status"] = result["status"]
                        row["proposal_ledger"] = result
                        if result["proposals"]:
                            proposal = result["proposals"][0]["smiles"]
                    else:
                        row["proposal_status"] = item["baseline"]["reason"] or "baseline_invalid"
                    row["proposal_smiles"] = proposal
                    if proposal:
                        row["proposal_check"] = checker.audit_product(found, proposal, example)
                rows.append(row)
            selected = admit_family_proposals(rows, is_train_product=is_train)
            before, after = summary(selected, True, is_train), summary(selected, False, is_train)
            checks = dict(
                all_attempts_retained=len(selected) == 192,
                baseline_exact_preserved=all(
                    not r["baseline_check"]["exact"] or r["selected_smiles"] == r["baseline_smiles"]
                    for r in selected
                ),
                nondecreasing_product_quality=all(
                    after[f][k] >= before[f][k]
                    for f in before
                    for k in (
                        "valid",
                        "exact",
                        "unique_valid",
                        "unique_exact",
                        "novel_valid",
                        "novel_exact",
                        "inverse_simpson",
                    )
                ),
                positive_target_gain=any(after[f]["exact"] > before[f]["exact"] for f in before),
            )
            write_json(out / "attempts.json", selected)
            hashes.validate()
            write_json(
                out / "result.json",
                dict(
                    before=before,
                    after=after,
                    checks=checks,
                    passed=all(checks.values()),
                    inputs=dict(
                        previous=pin(ROOT, BASE / "fresh/attempts.json"),
                        generation=pin(ROOT, out / "generation.json"),
                        attempts=pin(ROOT, out / "attempts.json"),
                        membership=membership,
                    ),
                    implementation=[
                        pin(ROOT, Path(__file__)),
                        pin(ROOT, ROOT / "forge/model/compose_lipid_family_rules.py"),
                    ],
                    source_inputs=inputs,
                    source_controls=controls,
                    seed=p["seed"],
                    limitations=[
                        "TRAIN-derived development requests, one checkpoint, no training-seed replication",
                        "Component diversity is reported among exact products; component novelty not measured",
                        "Incomplete or missing precursor scaffold is an abstention, not rebuilt from a stored fragment",
                    ],
                    quality_promoted=False,
                ),
            )
            print(
                json.dumps(
                    dict(
                        phase=phase,
                        before={f: v["exact"] for f, v in before.items()},
                        after={f: v["exact"] for f, v in after.items()},
                        checks=checks,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

"""One bounded A17 demonstration from saved TRAIN-derived requests/logits; no inference."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from rdkit import rdBase

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import pin
from forge.model.compose_lipid_head_survival import load_head_survival_policy, propose_head_survival
from forge.model.precursor_reuse_projection import graph_smiles, state_graph
from results.phase1.compose_lipid_component_decoder_v1.contracts import assess, load_all
from results.phase1.compose_lipid_structural_audit_v1.layout_audit import recover_graphs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CONFIRM = ROOT / "results/phase1/compose_lipid_quality_confirmation_v1"


def read(path):
    return json.loads(path.read_text())


def policy():
    frozen = read(HERE / "policy_and_controls_v2.json")
    values = frozen["inputs"]
    value = load_head_survival_policy(
        ROOT,
        config_pin=values["config"],
        retained_registry_pin=values["retained_registry"],
        reference_pin=values["reference"],
        motif_families=tuple(frozen["motif_families"]),
    )
    if any(frozen[k] != v for k, v in json.loads(json.dumps(asdict(value))).items()):
        raise ValueError("Frozen head policy/control assessment changed")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    head_policy = policy()
    selected_path = CONFIRM / "assessment/d1_support_aware_selected.json"
    selected = read(selected_path)
    layout_path = resolve_pin(selected["inputs"]["layout_payload"], ROOT, label="A17 layout")
    ledger_path = resolve_pin(selected["inputs"]["ledger"], ROOT, label="A17 graph ledger")
    cases_path = ROOT / "results/phase1/compose_lipid_ring_quality_v1/audit_cases.json"
    case = next(c for c in read(cases_path)["cases"] if c["audit_id"] == "A17")
    chosen = next(r for r in selected["attempts"] if r["index"] == case["source_index"])
    if chosen["selected_smiles"] != case["smiles"] or chosen["selected_kind"] != "draw4:raw":
        raise ValueError("The reviewed A17 source identity changed")
    payload = torch.load(layout_path, map_location="cpu", weights_only=False)
    layout, atoms = payload["layouts"][chosen["index"]], payload["atoms"]
    _, nodes, edges, state, _, graph_source = next(recover_graphs([chosen], read(ledger_path)))
    if state is None or graph_smiles(nodes, edges, atoms) != case["smiles"]:
        raise ValueError("A17 must retain its observed source graph")
    offset = chosen["index"] // 8 * 8
    logit_receipt_path = CONFIRM / f"logits/draw-4-{offset:04d}.json"
    logit_receipt = read(logit_receipt_path)
    logit_path = resolve_pin(logit_receipt["predictions"], ROOT, label="A17 saved logits")
    logits = torch.load(logit_path, map_location="cpu", weights_only=False)
    individual = {k: v[chosen["index"] - offset].numpy() for k, v in logits.items()}
    inputs = dict(
        selected=pin(ROOT, selected_path),
        layout=pin(ROOT, layout_path),
        ledger=pin(ROOT, ledger_path),
        case_review=pin(ROOT, cases_path),
        logits=pin(ROOT, logit_path),
        logits_receipt=pin(ROOT, logit_receipt_path),
        policy=pin(ROOT, HERE / "policy_and_controls_v2.json"),
        module=pin(ROOT, ROOT / "forge/model/compose_lipid_head_survival.py"),
        producer=pin(ROOT, Path(__file__)),
    )
    if graph_source is not None:
        inputs["graph_shard"] = graph_source
    # Retain this exact small fixture for owning tests; it is not reconstructed evidence.
    torch.save(
        dict(layout=layout, atoms=atoms, state=state, predictions=individual), HERE / "a17_input.pt"
    )
    if args.prepare_only:
        (HERE / "a17_fixture.json").write_text(
            json.dumps(
                dict(
                    inputs=inputs,
                    fixture=pin(ROOT, HERE / "a17_input.pt"),
                    cpu_seconds=time.process_time() - started,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print("A17 exact input fixture prepared without scoring repair candidates")
        return
    with rdBase.BlockLogs():
        found, sources, _ = load_all()
        result = propose_head_survival(
            layout,
            state,
            individual,
            atoms,
            head_policy,
            source_assessor=lambda smiles: assess(found, layout, smiles),
        )
    new_nodes, new_edges = state_graph(result["state"])
    if new_edges.tolist() != edges.tolist() or len(new_nodes) != len(nodes):
        raise ValueError("Head repair changed topology or atom budget")
    elapsed = time.process_time() - started
    report = dict(
        schema_version="forge.scoped_head_survival_case.v1",
        inputs=inputs,
        source_contracts=sources,
        fixture=pin(ROOT, HERE / "a17_input.pt"),
        result=result,
        case="A17",
        request_index=chosen["index"],
        cpu_seconds=elapsed,
        maximum_initial_cpu_seconds=90,
        whole_product_quality_promoted=False,
        limitations=[
            "One selected TRAIN-derived diagnostic case; not held-out performance.",
            "New retained-head candidate predicate is not a pKa or biological qualification.",
            "A local edit may yield a previously seen complete head; novelty must be measured separately.",
        ],
    )
    if elapsed > 90:
        raise ValueError("Initial head case exceeded the bounded CPU allowance")
    (HERE / "a17_result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            dict(
                changed=result["changed"],
                before=result["before"]["status"],
                costs=result["costs"],
                cpu_seconds=elapsed,
            )
        )
    )


if __name__ == "__main__":
    main()

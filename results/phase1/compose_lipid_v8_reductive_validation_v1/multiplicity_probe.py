"""Probe the excluded aryl cases at two net events without admitting any rows.

This is deliberately outside the one-event source-key contract. Sequential net
reductive aminations do not establish the source's condensation/reduction staging.
"""

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from forge.assembly.families import RegistryAssemblyAdapter
from forge.assembly.repeated_components import RepeatBounds, replay_repeated_components
from forge.assembly.repeated_inverse import infer_repeated_components
from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
PROGRAM = "results/phase1/compose_lipid_v8_reductive_program_v1/result.json"


def pin(path):
    path = (ROOT / path).resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(path))}


def main():
    receipt = json.loads((ROOT / PROGRAM).read_text())
    for name, value in receipt["implementation"].items():
        resolve_pin(value, ROOT, label=name)
    config_path = resolve_pin(receipt["config"], ROOT, label="config")
    config = json.loads(config_path.read_text())
    registry_path = resolve_pin(receipt["inputs"]["registry"], ROOT, label="registry")
    registry = json.loads(registry_path.read_text())
    ledger = resolve_pin(receipt["artifacts"]["programs.jsonl.gz"], ROOT, label="programs")
    family = "aryl_reductive_amination"
    binding = config["families"][family]
    reaction = next(r for r in registry["reactions"] if r["reaction_id"] == binding["reaction_id"])
    adapter = RegistryAssemblyAdapter.from_registry(
        registry_path,
        reaction_id=binding["reaction_id"],
        expected_sha256=receipt["inputs"]["registry"]["sha256"],
    )
    bounds = RepeatBounds(**{**config["search_bounds"], "maximum_events": 2})
    rows, heads = [], defaultdict(set)
    with gzip.open(ledger, "rt") as stream:
        for line in stream:
            original = json.loads(line)
            if (
                original["family"] != family
                or original["status"] != "excluded_forward_or_stoichiometry"
            ):
                continue
            target = original["replay"]["reverse_layers"][0][0]
            inferred = infer_repeated_components(
                adapter, target, accumulator_role="amine_head", events=2, bounds=bounds
            )
            candidates = []
            for components in inferred["candidate_components"]:
                replay = replay_repeated_components(
                    adapter,
                    components,
                    target,
                    accumulator_role="amine_head",
                    events=2,
                    byproducts_per_event=reaction["precursor_scaffolds"][
                        "net_byproducts_per_event"
                    ],
                    bounds=bounds,
                )
                candidates.append(
                    {
                        "components": components,
                        "replay": replay,
                        "target_in_forward_products": target in replay["forward_layers"][-1],
                    }
                )
                heads[original["source_metadata"]["head_code"]].add(components["amine_head"])
            rows.append(
                {
                    "target_id": original["target_id"],
                    "constitution_id": original["constitution_id"],
                    "source_metadata": original["source_metadata"],
                    "original_status": original["status"],
                    "diagnostic_events": 2,
                    "inferred": inferred,
                    "candidates": candidates,
                    "training_admitted": False,
                    "source_program_admitted": False,
                }
            )
    assert (
        len(rows)
        == receipt["summary"]["by_family"][family]["by_status"]["excluded_forward_or_stoichiometry"]
    )
    ledger_path = OUTPUT / "multiplicity-probe.jsonl.gz"
    with ledger_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
    inputs = {
        PROGRAM,
        config_path,
        registry_path,
        ledger,
        Path(__file__),
        "forge/assembly/repeated_inverse.py",
        "results/phase1/compose_lipid_v8_reductive_source_v1/xue-fig1.jpg",
        "results/phase1/compose_lipid_v8_reductive_source_v1/xue-fig1.acquisition.json",
    }
    inputs.update(ROOT / p["path"] for p in receipt["implementation"].values())
    result = {
        "schema_version": "forge.compose_lipid_reductive_multiplicity_probe.v1",
        "status": "off_contract_diagnostic_no_admission",
        "inputs": {pin(p)["path"]: pin(p) for p in sorted(map(str, inputs))},
        "artifacts": {ledger_path.name: pin(ledger_path)},
        "policy": {"seed": 0, "training_calls": 0, "generation_calls": 0},
        "rows": len(rows),
        "rows_by_source_head_code": dict(
            sorted(Counter(r["source_metadata"]["head_code"] for r in rows).items())
        ),
        "inferred_heads_by_source_code": {k: sorted(v) for k, v in sorted(heads.items())},
        "complete_unique_two_event_inverse": sum(
            r["inferred"]["complete_search"] and len(r["candidates"]) == 1 for r in rows
        ),
        "candidate_replay_counts": {
            "candidates": sum(len(r["candidates"]) for r in rows),
            "complete_search": sum(
                c["replay"]["checks"]["complete_search"] for r in rows for c in r["candidates"]
            ),
            "target_in_forward_products": sum(
                c["target_in_forward_products"] for r in rows for c in r["candidates"]
            ),
            "balanced": sum(
                c["replay"]["checks"]["full_element_hydrogen_charge_balance"]
                for r in rows
                for c in r["candidates"]
            ),
            "computed_consistency_pass": sum(
                c["replay"]["computed_consistency_pass"] for r in rows for c in r["candidates"]
            ),
            "forward_product_count_distribution": dict(
                sorted(
                    Counter(
                        len(c["replay"]["forward_layers"][-1])
                        for r in rows
                        for c in r["candidates"]
                    ).items()
                )
            ),
        },
        "source_figure_visual_review": "Xue Figure 1b depicts condensation then reduction; Figure 1c includes multifunctional primary amine heads. This figure was inspected independently of computed products.",
        "training_rows_admitted": 0,
        "limitations": [
            "Two events are a diagnostic hypothesis outside the pinned one-event key, not an admitted correction.",
            "All forward products are retained. Finding the target does not establish uniqueness.",
            "Sequential net alkylation does not encode the source's imine formation before reduction.",
            "No inference here resolves experimental occupancy, stage-specific sites, selectivity or source-bank membership.",
            "This diagnostic does not replace the original exclusions or qualify any training record.",
        ],
    }
    (OUTPUT / "multiplicity-probe.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"inputs", "limitations"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()

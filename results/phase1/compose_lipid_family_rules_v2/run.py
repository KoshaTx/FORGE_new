"""Fixed-weight family rules with complete source executors and paired ledgers."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rdkit import rdBase

from forge.core.hashing import resolve_pin
from forge.core.io import write_json
from forge.corpus.compose_lipid_source_view import pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.with_name("compose_lipid_source_constraints_v2")
MIRROR = ROOT / ".Codex-scratch/compose-family-rules-v2"
TARGETS = ("aryl_reductive_amination", "reductive_amination", "ketone_ugi4")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def executors():
    checker = load(MIRROR / "results/phase1/compose_lipid_posttraining_v1/adjudicate.py", "checker")
    found, inputs, controls = checker.load_executors()
    from forge.corpus import compose_lipid_family_replay as family
    from forge.corpus import compose_lipid_fixed_profiles as profiles
    from forge.corpus import compose_lipid_fixed_replay as fixed

    configuration = MIRROR / "results/phase1/compose_lipid_fixed_origins_v1/config.json"
    config = json.loads(configuration.read_text())
    full = {}
    for name, item in config["contracts"].items():
        path = resolve_pin(item["config"], MIRROR, label=name)
        if item["loader"] == "family":
            cfg = json.loads(path.read_text())
            paths = {k: resolve_pin(v, MIRROR, label=k) for k, v in cfg["inputs"].items()}
            local, checked = family.load_executors(MIRROR, paths)
        else:
            loader = profiles if item["loader"] == "profiles" else fixed
            _, _, local, checked = loader.load_contract(MIRROR, path)
        controls["restored_" + name] = checked
        full.update({k: v for k, v in local.items() if v["kind"] == "fixed"})
        inputs["restored_" + name] = pin(ROOT, path)
    for family_name, candidates in found.items():
        if family_name in full and any(c["kind"] == "introduced" for c in candidates):
            if family_name not in full or len(candidates) != 1:
                raise ValueError(f"No full source executor for {family_name}")
            found[family_name] = [{**full[family_name], "bounds": candidates[0]["bounds"]}]
    return checker, found, inputs, controls


def audit():
    if (HERE / "audit.json").exists():
        raise FileExistsError("Completed audit is immutable")
    payload = torch.load(BASE / "input.pt", map_location="cpu", weights_only=False)
    previous = json.loads((BASE / "fresh/attempts.json").read_text())["constrained"]
    historical = load(
        MIRROR / "results/phase1/compose_lipid_posttraining_v1/adjudicate.py", "hashes"
    )
    hashes = historical.VerifiedHashes()
    rows = []
    with patch("forge.core.hashing.sha256_file", hashes), rdBase.BlockLogs():
        checker, found, source_inputs, controls = executors()
        for i, (old, layout) in enumerate(zip(previous, payload["layouts"], strict=True)):
            result = dict(status="invalid_product", exact=False)
            if old["selected_smiles"]:
                result = checker.audit_product(
                    found,
                    old["selected_smiles"],
                    SimpleNamespace(
                        record=layout.record,
                        family=layout.family,
                        source_quantities=layout.quantities,
                    ),
                )
            rows.append(
                dict(
                    index=i,
                    family=layout.family,
                    program=layout.record.program_id,
                    smiles=old["selected_smiles"],
                    previous_exact=old["accepted"],
                    check=result,
                )
            )
            if (i + 1) % 64 == 0:
                print(f"Full source audit: {i + 1}/{len(previous)}", flush=True)
        hashes.validate()
    counts = {}
    for f in sorted({r["family"] for r in rows}):
        local = [r for r in rows if r["family"] == f]
        counts[f] = dict(
            attempts=len(local),
            previous_exact=sum(r["previous_exact"] for r in local),
            corrected_exact=sum(r["check"]["exact"] for r in local),
        )
    write_json(HERE / "audit-attempts.json", rows)
    write_json(
        HERE / "audit.json",
        dict(
            schema_version="forge.full_generated_source_audit.v1",
            by_family=counts,
            inputs={
                "payload": pin(ROOT, BASE / "input.pt"),
                "previous_attempts": pin(ROOT, BASE / "fresh/attempts.json"),
            },
            source_inputs=source_inputs,
            controls=controls,
            implementation=pin(ROOT, Path(__file__)),
            attempts=pin(ROOT, HERE / "audit-attempts.json"),
            seed=2026092414,
            authenticated_inputs=[
                dict(path=str(p), sha256=d) for p, (_, d) in hashes.entries.items()
            ],
            historical_metric_superseded=True,
        ),
    )
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["audit"])
    args = parser.parse_args()
    audit()

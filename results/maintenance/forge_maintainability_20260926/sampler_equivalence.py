"""Compare the preserved sampler with current code on existing local unit fixtures."""

import argparse
import hashlib
import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import torch

from experiments._runtime.source import source_fingerprint
from forge.core.io import write_json
from forge.model import synthesis_program_sampling as current
from forge.model import ugi_sampling_trace as observer
from tests import test_ugi_sampling_trace as fixtures
from tests.test_fixed_closure_decoding import ring_case

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
spec = importlib.util.spec_from_loader(
    "preserved_sampler", SourceFileLoader("preserved_sampler", str(args.baseline))
)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def normalize(v):
    if isinstance(v, torch.Tensor):
        return {"dtype": str(v.dtype), "shape": list(v.shape), "values": v.tolist()}
    if isinstance(v, np.ndarray):
        return {"dtype": str(v.dtype), "shape": list(v.shape), "values": v.tolist()}
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): normalize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [normalize(x) for x in v]
    return v


def evaluate(module):
    fixtures.sampling = module
    observer.sampling = module
    with observer.SamplingTrace() as trace:
        sampler, calls = fixtures._sampler_draw()
    results = {
        "sampler": normalize(sampler),
        "calls": calls,
        "trace": trace.events,
        "endpoints": trace.endpoint_state,
        "trace_summary": trace.summary,
    }
    for seed in (0, 17, 77):
        rng = np.random.default_rng(seed)
        with observer.SamplingTrace() as trace:
            tail = fixtures._tail_draw(trace.observe_generator(rng), fixtures._FrequencyPolicy())
        results[f"tail_{seed}"] = {
            "value": normalize(tail),
            "rng": normalize(rng.bit_generator.state),
            "events": trace.events,
            "summary": trace.summary,
        }
        record, layout, logits, atoms = ring_case()
        for reserve in (False, True):
            for stochastic in (False, True):
                rng = np.random.default_rng(seed)
                terminal, reasons = module.decode_synthesis_program_strict_argmax(
                    logits,
                    layout,
                    [record],
                    atoms,
                    terminal_generator=rng if stochastic else None,
                    reserve_fixed_closures=reserve,
                )
                results[f"decode_{seed}_{reserve}_{stochastic}"] = {
                    "states": normalize(terminal),
                    "reasons": reasons,
                    "rng": normalize(rng.bit_generator.state),
                    "fixed": module._fixed_state_exact_records(terminal, [record]),
                    "smiles": module._terminal_smiles(
                        terminal, 0, record.node_count, record.graph.closure_count, atoms
                    ),
                }
    return normalize(results)


try:
    before = evaluate(baseline)
    after = evaluate(current)
finally:
    fixtures.sampling = current
    observer.sampling = current
changed = [key for key in before if before[key] != after[key]]
files = [
    args.baseline,
    Path(__file__),
    Path("tests/test_ugi_sampling_trace.py"),
    Path("tests/test_fixed_closure_decoding.py"),
    Path("forge/model/ugi_sampling_trace.py"),
    Path("forge/model/synthesis_program_sampling.py"),
    *sorted(Path("forge/model/_synthesis_sampling").glob("*.py")),
]
pins = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
document = {
    "schema_version": "forge.sampler_refactor_equivalence.v1",
    "status": "pass" if not changed else "fail",
    "seeds": [0, 17, 77],
    "checks": len(before),
    "changed": changed,
    "inputs": pins,
    "before_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
    "after_sha256": hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
    "training_calls": 0,
    "remote_calls": 0,
    "dependency_context": "baseline sampler and refactored sampler use identical current shared dependencies",
    "current_source_fingerprint": source_fingerprint(Path.cwd()),
}
write_json(args.output, document)
print(json.dumps({k: v for k, v in document.items() if k != "inputs"}, indent=2))
if changed:
    raise SystemExit(1)

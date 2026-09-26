"""Record exact structural differences in the failed domain fixture; no policy edits."""

import gzip
import importlib.util
import json
import resource
import time
from pathlib import Path
from unittest.mock import patch

import torch
from rdkit import rdBase

p = Path(__file__).with_name("matched_context_evaluation_v4.py")
spec = importlib.util.spec_from_file_location("context_fixture_diagnostic", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
resource.setrlimit(resource.RLIMIT_CPU, (60, 61))
start = time.process_time()
protocol, payload = m.inputs()
torch.set_num_threads(4)
entry = next(x for x in protocol["fixture_schedule"] if x["offset"] == 128)
pred = torch.load(m.ROOT / entry["predictions"]["path"], map_location="cpu", weights_only=False)
hashes = m.frozen_construction.load(
    m.frozen_construction.MIRROR / "results/phase1/compose_lipid_posttraining_v1/adjudicate.py",
    "domain_difference_hashes",
).VerifiedHashes()
with patch("forge.core.hashing.sha256_file", hashes), rdBase.BlockLogs():
    context = m.construction.setup(
        m.ROOT, m.read(m.CONFIRM / "construction-protocol.json"), payload
    )
    base = m.construction.construct(context, payload, pred, draw=0, offset=128)
    actual = m.construction.construct(
        context, payload, pred, draw=0, offset=128, mode="domain", base_rows=base
    )
    expected = json.loads(
        gzip.decompress((m.CONFIRM / "constructed/domain-0-0128.json.gz").read_bytes())
    )
    differences = []

    def visit(a, b, path=""):
        if type(a) != type(b):
            differences.append(
                {"path": path, "actual_type": str(type(a)), "expected_type": str(type(b))}
            )
            return
        if isinstance(a, dict):
            if a.keys() != b.keys():
                differences.append({"path": path, "actual_keys": list(a), "expected_keys": list(b)})
            for k in sorted(a.keys() & b.keys()):
                visit(a[k], b[k], path + "/" + k)
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(
                    {"path": path, "actual_length": len(a), "expected_length": len(b)}
                )
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + "/" + str(i))
        elif a != b:
            differences.append({"path": path, "actual": a, "expected": b})

    visit(actual, expected)
    out = m.OUT / "domain_diagnosis.json"
    m.write(
        out,
        {
            "protocol": m.pin(m.OUT / "protocol.json"),
            "producer": m.pin(Path(__file__)),
            "expected": m.pin(m.CONFIRM / "constructed/domain-0-0128.json.gz"),
            "differences": differences,
            "CPU_seconds": time.process_time() - start,
            "new_model_GPU_TEST_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "differences": len(differences),
                "first": differences[:12],
                "CPU_seconds": time.process_time() - start,
            }
        )
    )

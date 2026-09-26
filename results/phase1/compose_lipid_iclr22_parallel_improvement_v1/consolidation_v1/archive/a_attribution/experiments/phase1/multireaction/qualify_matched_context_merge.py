"""Authenticate and merge all saved constructor shards without generating candidates."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_context_evaluation_v5 as fixture  # noqa: E402
import matched_context_pipeline_v2 as pipeline  # noqa: E402

ROOT = fixture.ROOT
SOURCE = ROOT / "results/phase1/compose_lipid_quality_confirmation_v1"
OUT = fixture.OUT / "merge_qualification_v1"


def freeze():
    OUT.mkdir(exist_ok=False)
    fixture.write(
        OUT / "protocol.json",
        {
            "inputs": {
                key: fixture.pin(path)
                for key, path in {
                    "producer": Path(__file__),
                    "pipeline": Path(pipeline.__file__),
                    "construction": SOURCE / "result.json",
                    "generation": SOURCE / "generation.json",
                    "expected_compact_ledger": SOURCE / "compact_attempts.json",
                }.items()
            },
            "maximum_CPU_seconds": 60,
            "full_shards": 920,
            "generation_shards": 880,
            "requests": 1408,
            "expectation": "Every saved compact/full artifact authenticates; merged complete dictionaries equal original; no model, constructor, chemistry or TEST calls",
        },
    )


def run():
    start = time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 61))
    protocol = fixture.read(OUT / "protocol.json")
    for value in protocol["inputs"].values():
        fixture.authenticate(value)
    saved = fixture.read(protocol["inputs"]["construction"]["path"])
    generation = fixture.read(protocol["inputs"]["generation"]["path"])
    directory = OUT / "run_v1"
    directory.mkdir(exist_ok=False)
    shards = directory / "constructed"
    shards.mkdir()
    for receipt in saved["shards"]:
        fixture.write(
            shards / f"{receipt['mode']}-{receipt['draw']}-{receipt['offset']:04d}.receipt.json",
            receipt,
        )

    class ForbiddenConstructor:
        def construct(self, *args, **kwargs):
            raise AssertionError("No new construction permitted in saved-shard merge qualification")

    context = {
        "domain_policies": {r["offset"]: None for r in saved["shards"] if r["mode"] == "domain"}
    }
    try:
        ledger, receipts = pipeline.construct_all(
            ROOT,
            directory,
            None,
            generation["receipts"],
            context,
            ForbiddenConstructor(),
            fixture.pin,
            fixture.write,
            lambda: None,
        )
        assert ledger == fixture.read(protocol["inputs"]["expected_compact_ledger"]["path"])
        assert receipts == saved["shards"]
        result = {
            "passed": True,
            "protocol": fixture.pin(OUT / "protocol.json"),
            "requests": len(ledger),
            "constructor_shards": len(receipts),
            "generation_shards": len(generation["receipts"]),
            "merged_complete_dictionaries_identical": True,
            "all_full_and_compact_hashes_verified": True,
            "new_model_constructor_chemistry_TEST_calls": 0,
            "CPU_seconds": time.process_time() - start,
            "output": fixture.pin(directory / "compact_attempts.json"),
        }
        fixture.write(OUT / "result.json", result)
        print(json.dumps(result))
    except BaseException as error:
        fixture.write(
            OUT / "failure.json",
            {
                "complete": False,
                "error": repr(error),
                "CPU_seconds": time.process_time() - start,
                "protocol": fixture.pin(OUT / "protocol.json"),
            },
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("freeze", "run"))
    a = p.parse_args()
    freeze() if a.action == "freeze" else run()

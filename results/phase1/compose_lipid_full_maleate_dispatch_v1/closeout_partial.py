"""Retain the stopped redundant pass as operational history, never completed evidence."""

import json
from pathlib import Path

from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    partial = HERE / "replay"
    if (partial / "result.json").exists():
        raise ValueError("The stopped pass unexpectedly has a completed result")
    paths = sorted(partial.glob("*/result.json"))
    dump(
        HERE / "partial-pass-closeout.json",
        {
            "schema_version": "forge.local_preparation_operational_closeout.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "request": pin(ROOT, partial / "request.json"),
            "retained_shards": [pin(ROOT, p) for p in paths],
            "retained_rows": sum(json.loads(p.read_text())["rows"] for p in paths),
            "process_exit_code": 143,
            "termination_reason": "Stopped the redundant whole-family chemistry pass after "
            "observing serial throughput; retained its atomic shards and separately evaluated "
            "only the currently unresolved eligible recipes. Existing qualified evidence is "
            "carried unchanged. The incomplete pass supplies no additional evidence.",
            "replacement_completed_result": pin(ROOT, HERE / "incremental/result.json"),
            "complete": False,
            "scientific_failure_claimed": False,
            "training_admitted": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()

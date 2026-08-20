#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.value.synthesis.ugi_synthesis_guidance_failure_audit import (
    build_synthesis_guidance_failure_audit,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/route/phase1_ugi_synthesis_guidance_failure_audit_v1.json"
DEFAULT_OUTPUT = REPO / "results/phase1/ugi_synthesis_guidance_failure_audit_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    result = build_synthesis_guidance_failure_audit(REPO, args.config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp:
        root = Path(temp)
        (root / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(root, output)
    print(
        json.dumps(
            {"matched_result": result["matched_result"], "decision": result["decision"]}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

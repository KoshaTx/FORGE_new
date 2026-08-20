#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.design.audit.ugi_checkpoint_series_audit import audit_ugi_checkpoint_series

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--program-probe", type=Path, required=True)
    parser.add_argument("--sample-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_ugi_checkpoint_series(
        REPO,
        training_result_path=args.training_result,
        program_probe_path=args.program_probe,
        sample_result_paths=args.sample_result,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "checkpoints": result["checkpoints"],
                "decision": result["decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

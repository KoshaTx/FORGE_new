#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from forge.product.ugi_training_cache import prepare_ugi_training_cache

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    result = prepare_ugi_training_cache(
        args.config,
        REPO,
        args.output_dir,
        overwrite=args.overwrite,
        reuse_existing=args.reuse_existing,
    )
    print(args.output_dir / "result.json")
    print(result["artifact"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

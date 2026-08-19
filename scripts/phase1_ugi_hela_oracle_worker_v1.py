#!/usr/bin/env python3
"""Serve the frozen M0-07 HeLa oracle across an authenticated process boundary.

The selected-v3 generator and the frozen production oracle intentionally use
different qualified software environments.  This worker is launched only by
the diagnostic potency seam under ``.venv-oracle-2025``.  It loads the
hash-verified checkpoint once, accepts canonical JSON-lines requests, and
returns classifications and three-member-ensemble predictions.  It performs
no generation, ranking, candidate selection, synthesis assessment or file
writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "forge.ugi_hela_oracle_worker.v1"
REQUEST_SCHEMA_VERSION = "forge.ugi_hela_oracle_worker_request.v1"
RESPONSE_SCHEMA_VERSION = "forge.ugi_hela_oracle_worker_response.v1"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _emit(value: Mapping[str, Any]) -> None:
    sys.stdout.write(_stable_json(dict(value)) + "\n")
    sys.stdout.flush()


def _candidate(value: Any, *, index: int) -> dict[str, str]:
    required = {
        "label",
        "product_smiles",
        "amine_smiles",
        "aldehyde_smiles",
        "isocyanide_smiles",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"candidate {index} has an unsupported field set")
    output = {key: str(value[key]) for key in sorted(required)}
    if any(not item for item in output.values()):
        raise ValueError(f"candidate {index} contains an empty field")
    return output


def _serve(repo: Path, result_path: Path, expected_result_sha256: str) -> None:
    # The worker can be launched directly without relying on an activated venv.
    source = repo / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    import sklearn
    import torch
    from rdkit import rdBase

    from forge.potency.oracle_production import (
        classify_production_ugi_candidate,
        load_production_checkpoint,
        predict_production_ugi_smiles,
        sha256_file,
    )

    if sha256_file(result_path) != expected_result_sha256:
        raise ValueError("production-result hash differs from the launch contract")
    checkpoint = load_production_checkpoint(result_path, repo_root=repo)
    model = checkpoint.get("selected_model")
    ready = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "endpoint": "expt_Hela",
        "production_result_sha256": expected_result_sha256,
        "selected_model": model,
        "software": {
            "rdkit": rdBase.rdkitVersion,
            "scikit_learn": sklearn.__version__,
            "torch": str(torch.__version__),
        },
    }
    ready["receipt_sha256"] = _sha256_payload(ready)
    _emit(ready)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, Mapping):
                raise ValueError("request must be a JSON object")
            if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
                raise ValueError("unsupported worker request schema")
            action = request.get("action")
            if action == "shutdown":
                _emit(
                    {
                        "schema_version": RESPONSE_SCHEMA_VERSION,
                        "status": "shutdown",
                        "request_sha256": _sha256_payload(request),
                    }
                )
                return
            if action != "predict" or request.get("endpoint") != "expt_Hela":
                raise ValueError("worker accepts only HeLa prediction requests")
            raw_candidates = request.get("candidates")
            if not isinstance(raw_candidates, list) or not raw_candidates:
                raise ValueError("prediction request requires candidates")
            candidates = [
                _candidate(candidate, index=index) for index, candidate in enumerate(raw_candidates)
            ]
            classifications = [
                classify_production_ugi_candidate(checkpoint, candidate) for candidate in candidates
            ]
            prediction = predict_production_ugi_smiles(
                checkpoint,
                candidates,
                endpoint="expt_Hela",
            )
            response: dict[str, Any] = {
                "schema_version": RESPONSE_SCHEMA_VERSION,
                "status": "complete",
                "request_sha256": _sha256_payload(request),
                "classifications": classifications,
                "prediction": prediction,
            }
            response["receipt_sha256"] = _sha256_payload(response)
            _emit(response)
        except Exception as error:  # Fail closed across the process boundary.
            _emit(
                {
                    "schema_version": RESPONSE_SCHEMA_VERSION,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--production-result", type=Path, required=True)
    parser.add_argument("--production-result-sha256", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    result_path = args.production_result.resolve()
    result_path.relative_to(repo)
    _serve(repo, result_path, args.production_result_sha256)


if __name__ == "__main__":
    main()

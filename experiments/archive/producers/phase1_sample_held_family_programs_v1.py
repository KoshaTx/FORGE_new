#!/usr/bin/env python3
"""Generate under the sealed held-family-derived morphology programs, once.

Stage 2 of the held-family evaluation. Stage 1 measured teacher-forced denoising, which is not
generative competence; this measures what the model actually produces when asked to execute a
design program associated with chemistry it never saw.

The question, in full: when asked to generate under structural programs associated with
held-out chemistry, does the model remain valid, chemically admitted, morphology-faithful and
stable? It is not evidence that unseen chemical families are regenerated, and no wording here
implies that.

Everything is inherited rather than chosen: sampling settings come from the frozen production
execution config, the joint checkpoint is the development refit trained on the train fold only,
and the closure checkpoint was trained on strict train-fold cyclic components with calibration
used only for selection, so neither leaks heldout structure. The program draw is consumed
through the sealed path by SHA-256, so the programs cannot change between sealing and sampling.

Prespecified in docs/FORGE_EVIDENCE_CONTRACT_v1.md section B2. One draw per stratum at a budget
fixed before sampling; no resampling at a different budget, temperature or seed after seeing
admission rates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from experiments.phase1.product_l1.sampling.ugi_blinded_headless_sampling import (  # noqa: E402
    preflight_headless_runtime,
    sample_headless_runtime,
)

JOINT = "results/phase1/ugi_joint_sparse_balanced_v2_full/checkpoint_step_1000.pt"
CLOSURE = "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt"
REACTIONS = "data/vendor/qualified_reactions_v1.json"
PRODUCTION_SETTINGS = "configs/route/phase1_ugi3_route_saturation_blinded_execution_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw", type=Path, required=True, help="sealed program draw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override only for a plumbing smoke test; never for a result")
    parser.add_argument("--joint-checkpoint", type=str, default=JOINT,
                        help="joint generator checkpoint; defaults to the joint-sparse "
                             "development refit")
    parser.add_argument("--terminal-decoder-mode", type=str, default=None,
                        help="override the inherited decoder mode to match a specific "
                             "production run")
    args = parser.parse_args()
    started = time.time()

    draw = json.loads(args.draw.read_text())
    rows = draw["samples"]
    draw_sha = sha256_file(args.draw)
    print(f"sealed draw {args.draw.name}: {len(rows)} programs, sha256 {draw_sha[:16]}...")

    joint = args.joint_checkpoint
    checkpoint = torch.load(REPO / joint, map_location="cpu", weights_only=False)
    cache = checkpoint["inputs"]["prepared_cache"]
    # The checkpoint records the container path it was written under on the training host.
    # Remap it exactly as the sampler's own resolver does, or the pin comparison fails.
    cache_path = Path(str(cache["path"]))
    if not cache_path.is_file() and str(cache_path).startswith("/root/forge_repo/"):
        cache_path = REPO / cache_path.relative_to("/root/forge_repo")
    elif not cache_path.is_absolute():
        cache_path = REPO / cache_path
    print(f"checkpoint {joint}")
    print(f"  step {checkpoint['step']}, decoration_state_conditioning="
          f"{checkpoint['model_config'].get('decoration_state_conditioning')}, "
          f"cache pinned at {cache['sha256'][:16]}...")

    production = json.loads((REPO / PRODUCTION_SETTINGS).read_text())["sampling"]
    settings = {
        "sample_steps": production["sample_steps"],
        "batch_size": args.batch_size or production["batch_size"],
        "flow_seed": production["flow_seed"],
        "closure_seed": production["closure_seed"],
        "terminal_seed": production["terminal_seed"],
        "terminal_decoder_mode": production["terminal_decoder_mode"],
        "terminal_temperature": production["terminal_temperature"],
        "maximum_adjacent_branch_runs": production["maximum_adjacent_branch_runs"],
        "device": production["device"],
    }
    if args.terminal_decoder_mode:
        settings["terminal_decoder_mode"] = args.terminal_decoder_mode
    print(f"settings inherited from {PRODUCTION_SETTINGS}: "
          f"{settings['sample_steps']} steps, batch {settings['batch_size']}, "
          f"device {settings['device']}, decoder {settings['terminal_decoder_mode']}")

    runtime = preflight_headless_runtime(
        REPO,
        joint_checkpoint_path=REPO / joint,
        closure_checkpoint_path=REPO / CLOSURE,
        prepared_cache_path=cache_path,
        prepared_cache_sha256=str(cache["sha256"]),
        qualified_reactions_path=REPO / REACTIONS,
        program_draw_path=args.draw,
        expected_program_rows=len(rows),
    )
    print(f"preflight complete after {time.time() - started:.1f}s; sampling {len(rows)} programs")

    result = sample_headless_runtime(
        runtime,
        settings=settings,
        pinned_inputs={
            "joint_checkpoint": {"path": joint, "sha256": sha256_file(REPO / joint)},
            "closure_checkpoint": {"path": CLOSURE, "sha256": sha256_file(REPO / CLOSURE)},
            "program_draw": {"path": str(args.draw), "sha256": draw_sha},
            "qualified_reactions": {"path": REACTIONS, "sha256": sha256_file(REPO / REACTIONS)},
        },
    )
    result["stage"] = "stage_2_ood_morphology_conditioned_sampling"
    result["contract"] = "docs/FORGE_EVIDENCE_CONTRACT_v1.md section B2"
    result["program_draw_metadata"] = {
        "per_stratum": draw["per_stratum"], "seed": draw["seed"],
        "composition": draw["composition"], "sha256": draw_sha,
    }
    result["settings_provenance"] = {
        "inherited_from": PRODUCTION_SETTINGS,
        "overridden": {"batch_size": args.batch_size} if args.batch_size else {},
    }
    result["runtime_environment"] = {
        "python_version": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "elapsed_seconds": round(time.time() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"\nsampled {len(result['samples'])} rows in "
          f"{result['runtime_environment']['elapsed_seconds']:.1f}s total")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""How much of Stage 2's near-ceiling admission comes from the learned flow?

Stage 2 showed the complete system producing admitted, distinct products at essentially the same
rate under held-family-derived structural programs as under train-family ones. That is a
system-level result, and the constrained sampler is a legitimate part of the system. But it does
not say how much of the admission rate needs the trained flow at all, and a reviewer will ask.

This runs the identical pipeline on the identical sealed programs with the trained denoiser
replaced by a deliberately non-learned proposal. Everything else is held fixed: the same
programs, the same structural masks, the same closure model, the same transition schedule, the
same source distributions, the same admission predicate and the same seeds.

Two null arms, because they bracket the answer and one alone can be argued with:

  marginal_null   every prediction head emits the role-specific empirical source marginal that
                  the flow was trained against, so the denoiser proposes the training marginal
                  regardless of the current state. The fair null: it knows the per-role atom,
                  bond and offspring frequencies but nothing about context.
  random_init     the same architecture at its random initialization, seed fixed. The weak null.

Reading the outcome, fixed before running:

  both nulls much worse            the learned flow supplies chemistry the constraints do not,
                                   and Stage 2 supports a learned-competence claim
  nulls near ceiling on admission  admission is largely a property of the masks. Stage 2 stands
    but poor on diversity          as a system result; learned competence must be carried by
                                   distributional quality, not by the admission rate
  nulls match on everything        Stage 2 is a constrained-sampler result and must be
                                   attributed as one

No retraining, no architecture change, no new baseline method.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from forge.design.sampling.ugi_blinded_headless_sampling import (  # noqa: E402
    preflight_headless_runtime,
    sample_headless_runtime,
)
from forge.design.flow.ugi_joint_sparse_flow import UgiJointSparseFlow  # noqa: E402

JOINT = "results/phase1/ugi_joint_sparse_balanced_v2_full/checkpoint_step_1000.pt"
CLOSURE = "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt"
REACTIONS = "data/vendor/qualified_reactions_v1.json"
PRODUCTION_SETTINGS = "configs/route/phase1_ugi3_route_saturation_blinded_execution_v1.json"
RANDOM_INIT_SEED = 20260816

# Which marginal each prediction head is replaced by. Keyed by name, never by tensor width:
# offspring and parent_bonds are both four-way, so size alone cannot disambiguate them.
ROLE_MARGINAL = {"offspring": "offspring", "nodes": "atoms", "parent_bonds": "bonds"}
GLOBAL_MARGINAL = {
    "closure_bonds": "closure_bonds",
    "decoration_anchors": "decoration",
    "decoration_atoms": "decoration_atoms",
    "decoration_bonds": "decoration_bonds",
}
LOG_FLOOR = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MarginalNullFlow(UgiJointSparseFlow):
    """Same architecture and interface; every head emits a fixed source marginal.

    Subclassed rather than duck-typed because the sampler reads `maximum_decorations`,
    `conditioning_mode`, `maximum_attachment_count` and `maximum_cycle_rank` off the model. The
    inherited forward is called once per step for its output keys and shapes only; every value
    it produces is discarded before returning, so no learned or randomly initialized weight
    influences a sampling decision.
    """

    def __init__(self, *, source_marginals, **kwargs) -> None:
        super().__init__(**kwargs)
        self._marginals = {k: np.asarray(v, dtype=np.float64)
                           for k, v in source_marginals.items()}
        self.head_treatment: dict[str, str] = {}

    def forward(self, **kwargs):  # type: ignore[override]
        shapes = super().forward(**kwargs)
        roles = kwargs["role_states"]
        out = {}
        for key, tensor in shapes.items():
            classes = tensor.shape[-1]
            role_key = ROLE_MARGINAL.get(key)
            global_key = GLOBAL_MARGINAL.get(key)
            if (role_key is not None and tensor.ndim == 3
                    and tuple(tensor.shape[:2]) == tuple(roles.shape)
                    and self._marginals[role_key].shape[-1] == classes):
                table = torch.as_tensor(
                    np.log(np.clip(self._marginals[role_key], LOG_FLOOR, None)),
                    dtype=tensor.dtype, device=tensor.device)
                out[key] = table[roles]
                self.head_treatment[key] = f"role-specific marginal '{role_key}'"
            elif global_key is not None and self._marginals[global_key].shape[-1] == classes:
                vector = torch.as_tensor(
                    np.log(np.clip(self._marginals[global_key], LOG_FLOOR, None)),
                    dtype=tensor.dtype, device=tensor.device)
                out[key] = vector.expand_as(tensor).clone()
                self.head_treatment[key] = f"global marginal '{global_key}'"
            else:
                # Not a categorical head with a matching marginal. Pass it through untouched.
                # `hidden` in particular is consumed by the terminal decoder, and zeroing it
                # would make this arm a test of a broken decoder input rather than of a
                # non-learned denoiser. `attachment_counts` and `cycle_ranks` are constrained to
                # the requested program under full_morphology conditioning, so their logits do
                # not select anything.
                out[key] = tensor
                self.head_treatment[key] = "passed through from the untrained network"
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw", type=Path, required=True)
    parser.add_argument("--arm", choices=("marginal_null", "random_init"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()

    draw_sha = sha256_file(args.draw)
    rows = json.loads(args.draw.read_text())["samples"]
    checkpoint = torch.load(REPO / JOINT, map_location="cpu", weights_only=False)
    cache = checkpoint["inputs"]["prepared_cache"]
    cache_path = Path(str(cache["path"]))
    if not cache_path.is_file() and str(cache_path).startswith("/root/forge_repo/"):
        cache_path = REPO / cache_path.relative_to("/root/forge_repo")

    production = json.loads((REPO / PRODUCTION_SETTINGS).read_text())["sampling"]
    settings = {key: production[key] for key in (
        "sample_steps", "batch_size", "flow_seed", "closure_seed", "terminal_seed",
        "terminal_decoder_mode", "terminal_temperature", "maximum_adjacent_branch_runs",
        "device")}

    runtime = preflight_headless_runtime(
        REPO,
        joint_checkpoint_path=REPO / JOINT,
        closure_checkpoint_path=REPO / CLOSURE,
        prepared_cache_path=cache_path,
        prepared_cache_sha256=str(cache["sha256"]),
        qualified_reactions_path=REPO / REACTIONS,
        program_draw_path=args.draw,
        expected_program_rows=len(rows),
    )

    architecture = dict(checkpoint["model_config"])
    architecture.pop("source_probability_floor", None)
    torch.manual_seed(RANDOM_INIT_SEED)
    if args.arm == "marginal_null":
        model = MarginalNullFlow(
            atom_classes=len(runtime.corpus.atom_vocabulary),
            source_marginals=checkpoint["source_marginals"], **architecture)
        description = ("every prediction head emits the source marginal the flow was trained "
                       "against; no learned or random weight influences a sampling decision")
    else:
        model = UgiJointSparseFlow(
            atom_classes=len(runtime.corpus.atom_vocabulary), **architecture)
        description = (f"same architecture at random initialization, torch seed "
                       f"{RANDOM_INIT_SEED}")
    model.eval()
    trained_state = checkpoint["model_state"]
    live = dict(model.state_dict())
    identical = sum(1 for k in trained_state
                    if k in live and torch.equal(live[k].to(trained_state[k].dtype),
                                                 trained_state[k]))
    if identical:
        raise SystemExit(f"{identical} parameter tensors match the trained checkpoint; "
                         f"this arm is not a null control")
    print(f"arm {args.arm}: {description}")
    print(f"verified: 0 of {len(trained_state)} parameter tensors match the trained checkpoint")

    # Swap the denoiser, keeping every other runtime resource identical. HeadlessRuntime is a
    # frozen dataclass, so replace() is the only way to substitute one field without rebuilding
    # the corpus, closure model and reaction contract.
    runtime = dataclasses.replace(runtime, model=model)
    result = sample_headless_runtime(
        runtime, settings=settings,
        pinned_inputs={
            "program_draw": {"path": str(args.draw), "sha256": draw_sha},
            "closure_checkpoint": {"path": CLOSURE, "sha256": sha256_file(REPO / CLOSURE)},
            "joint_checkpoint": {"path": JOINT, "sha256": sha256_file(REPO / JOINT),
                                 "note": "architecture and source marginals only; its trained "
                                         "weights are not used by this arm"},
        },
    )
    result["stage"] = "stage_2_constraint_only_control"
    result["arm"] = args.arm
    result["arm_description"] = description
    result["head_treatment"] = getattr(model, "head_treatment", {})
    result["trained_parameters_reused"] = 0
    result["settings_provenance"] = {"inherited_from": PRODUCTION_SETTINGS}
    result["runtime_environment"] = {
        "python_version": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "elapsed_seconds": round(time.time() - started, 1)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"heads: {json.dumps(result['head_treatment'], indent=1)}")
    print(f"\nsampled {len(result['samples'])} rows in "
          f"{result['runtime_environment']['elapsed_seconds']:.1f}s")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

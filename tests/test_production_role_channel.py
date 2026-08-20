"""The production generator's role channel, pinned.

These guard a mistake that was made and nearly cost six GPU training runs. An ablation was
specified against `origin_embedding` in `AdapterNodeConditioning`, on the assumption that it was
the production model's role channel. It is not: that module belongs to `UgiChemistryFlow`, and
the production generator `UgiJointSparseFlow` never instantiates it.

The second test pins the fact that makes the ablation moot. Role is not merely predictable from
the surviving inputs; it is exactly reconstructible from the conditioning program, because the
program declares per-role node counts and the serialization emits contiguous role blocks in that
order. If that ever stops being true, an embedding ablation becomes meaningful again and this
test should fail loudly rather than let the old framing quietly return.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")

from forge.design.flow.ugi_joint_sparse_flow import (  # noqa: E402
    UgiJointSparseFlow,
    collate_ugi_joint_sparse_records,
)
from forge.design.training.ugi_training_cache import load_ugi_training_cache  # noqa: E402

CACHE = REPO / "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"
CHECKPOINT = REPO / "results/phase1/ugi_joint_sparse_balanced_v2_full/checkpoint_step_1000.pt"

pytestmark = pytest.mark.skipif(
    not (CACHE.exists() and CHECKPOINT.exists()),
    reason="frozen training cache or development checkpoint is unavailable",
)


@pytest.fixture(scope="module")
def loaded():
    corpus, records_by_fold = load_ugi_training_cache(CACHE)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    architecture = dict(checkpoint["model_config"])
    architecture.pop("source_probability_floor", None)
    model = UgiJointSparseFlow(atom_classes=len(corpus.atom_vocabulary), **architecture)
    return corpus, records_by_fold, architecture, model


def test_production_model_has_no_origin_embedding(loaded):
    _, _, _, model = loaded
    names = [name for name, _ in model.named_parameters()]
    assert not any("origin" in name.lower() for name in names)
    assert not any("adapter" in name.lower() for name, _ in model.named_modules())


def test_role_channel_is_role_embedding_and_is_tiny(loaded):
    _, _, _, model = loaded
    parameters = dict(model.named_parameters())
    assert tuple(parameters["role_embedding.weight"].shape) == (3, 192)
    total = sum(p.numel() for p in model.parameters())
    role = parameters["role_embedding.weight"].numel()
    assert total == 1_081_388
    assert role == 576
    assert role / total < 0.001


def test_role_states_are_determined_by_the_conditioning_program(loaded):
    """If this fails, an embedding ablation is meaningful again. Read the failure carefully."""
    _, records_by_fold, architecture, _ = loaded
    checked = 0
    for fold in ("train", "calibration", "heldout"):
        records = list(itertools.islice(records_by_fold[fold], 256))
        for start in range(0, len(records), 128):
            chunk = tuple(records[start:start + 128])
            batch = collate_ugi_joint_sparse_records(
                chunk, maximum_nodes=max(r.node_count for r in chunk),
                maximum_children=int(architecture["maximum_children"]),
                maximum_closures=int(architecture["maximum_cycle_rank"]) * 3,
                maximum_decorations=int(architecture["maximum_decorations"]),
            )
            for index, record in enumerate(chunk):
                checked += 1
                mask = batch["node_mask"][index]
                expected_roles: list[int] = []
                expected_positions: list[int] = []
                for role_index, count in enumerate(record.program.node_counts):
                    expected_roles += [role_index] * count
                    expected_positions += list(range(count))
                assert sum(record.program.node_counts) == record.node_count
                assert batch["role_states"][index][mask].tolist() == expected_roles
                assert batch["within_role_positions"][index][mask].tolist() == expected_positions
    assert checked >= 700

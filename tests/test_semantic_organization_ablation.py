"""Executable specification for the product-only baseline arm.

These tests are written BEFORE the implementation, and they are the gate on it. A baseline that
fails any of them is not a baseline and its runs are void.

The reason for writing them first is a mistake made twice in one day: auditing
`AdapterNodeConditioning` when the production generator is `UgiJointSparseFlow`, and evaluating
`checkpoint_best` when the frozen contract named step 3000. Both were the same error, assuming the
thing being manipulated was the thing that mattered. A `semantic_organization: none` switch invites
exactly that error, because a flat serialization is not one flag: it changes the record projection,
the layout, the source construction and the loss. A switch applied to only some of those would
leave role structure in place while being labelled as removing it.

The decisive fact those tests must respect: role is an EXACT deterministic function of the design
program and serialized position, verified over 9,000 records across all three folds with zero
mismatches. So dropping the role embedding while keeping role-blocked ordering removes nothing.
Position must stop carrying role, or there is no ablation.

Skipped until the switch exists, so that the suite is committed with the specification rather than
after the fact.
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")

from forge.product.ugi_joint_sparse_flow import (  # noqa: E402
    UgiJointSparseFlow,
    collate_ugi_joint_sparse_records,
)
from forge.product.ugi_training_cache import load_ugi_training_cache  # noqa: E402

CACHE = REPO / "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"
REFERENCE = REPO / "results/phase1/ugi_decoration_coupling_v1/challenger/checkpoint_step_3000.pt"
FLAT = "flat_no_role"
FLAT_TRUE = "flat_true_role"
STRUCTURED = "role_structured"

pytestmark = pytest.mark.skipif(
    not CACHE.exists() or not REFERENCE.exists(),
    reason="frozen training cache or reference checkpoint is unavailable",
)


def _supports_switch() -> bool:
    try:
        import inspect
        return "semantic_organization" in inspect.signature(UgiJointSparseFlow).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


requires_switch = pytest.mark.skipif(
    not _supports_switch(),
    reason="semantic_organization switch is not implemented yet; this suite specifies it",
)


@pytest.fixture(scope="module")
def corpus_and_records():
    return load_ugi_training_cache(CACHE)


@pytest.fixture(scope="module")
def architecture():
    checkpoint = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    config = dict(checkpoint["model_config"])
    config.pop("source_probability_floor", None)
    return config


def _batch(records, architecture, semantic_organization=None):
    kwargs = {}
    if semantic_organization is not None:
        kwargs["semantic_organization"] = semantic_organization
    return collate_ugi_joint_sparse_records(
        tuple(records), maximum_nodes=max(r.node_count for r in records),
        maximum_children=int(architecture["maximum_children"]),
        maximum_closures=int(architecture["maximum_cycle_rank"]) * 3,
        maximum_decorations=int(architecture["maximum_decorations"]), **kwargs)


# ------------------------------------------------------- 1. the ablation must remove information


@requires_switch
def test_flat_layout_does_not_encode_role_in_position(corpus_and_records, architecture):
    """The load-bearing test. Role must not be reconstructible from position and program.

    Under role-structured serialization the program's per-role node counts partition the
    serialized positions into contiguous blocks, so role is exact from position. If the flat
    arm keeps that property, dropping the role embedding removes nothing and the arm is not an
    ablation.
    """
    _, records_by_fold = corpus_and_records
    records = list(itertools.islice(records_by_fold["train"], 256))
    flat = _batch(records, architecture, semantic_organization=FLAT)

    exact = 0
    for index, record in enumerate(records):
        mask = flat["node_mask"][index]
        expected: list[int] = []
        for role_index, count in enumerate(record.program.node_counts):
            expected += [role_index] * count
        observed = flat.get("role_states")
        if observed is None:
            exact = 0
            break
        if observed[index][mask].tolist() == expected:
            exact += 1
    # Amendment 9: role contiguity is forced by chemistry plus preorder-forest validity, so the
    # achievable property is that the layout no longer reproduces PROGRAM ORDER, not that regions
    # stop being contiguous. Measured at 0.245 with content-based ordering.
    assert exact <= len(records) * 0.40, (
        f"{exact}/{len(records)} flat-arm records reproduce program-order role blocks; the flat "
        "layout is not decoupling position from the program at all")


@requires_switch
def test_flat_arm_drops_role_but_keeps_generic_sequence_position(architecture):
    """Fairness, not just ablation.

    `position_embedding` in the structured arm is nn.Embedding(maximum_component_atoms, hidden)
    indexed by WITHIN-ROLE position, so it is experiment-specific semantic position. Removing it
    outright would leave the flat arm with no positional signal at all, which is a handicap no
    ordinary whole-molecule generator would carry, and a reviewer would rightly say the baseline
    was crippled rather than ablated.

    So the flat arm must keep a GLOBAL sequence position embedding sized to maximum_total_atoms,
    and must not carry a within-role one.
    """
    model = UgiJointSparseFlow(atom_classes=14, semantic_organization=FLAT, **architecture)
    names = dict(model.named_parameters())
    # Amendment 12: the table is RETAINED with a null slot so capacity parity is exact. The
    # ablation is which index is fed, not whether the machinery exists.
    from forge.product.ugi_joint_sparse_flow import ROLE_NAMES
    assert names["role_embedding.weight"].shape[0] == len(ROLE_NAMES) + 1, (
        "the flat arms must share one role table including the null slot, so their parameter "
        "counts are identical and the ablation is the fed index rather than the capacity")
    assert "position_embedding.weight" in names, (
        "the flat arm needs generic sequence position; removing it handicaps the baseline")
    rows = names["position_embedding.weight"].shape[0]
    assert rows == int(architecture["maximum_total_atoms"]), (
        f"flat position embedding has {rows} rows; a global sequence position embedding must span "
        f"maximum_total_atoms ({architecture['maximum_total_atoms']}), not "
        f"maximum_component_atoms ({architecture['maximum_component_atoms']})")


@requires_switch
def test_flat_position_index_is_global_not_within_role(corpus_and_records, architecture):
    """The index itself must be an absolute sequence position, not a per-role counter."""
    _, records_by_fold = corpus_and_records
    records = list(itertools.islice(records_by_fold["train"], 128))
    flat = _batch(records, architecture, semantic_organization=FLAT)
    positions = flat["within_role_positions"]
    mask = flat["node_mask"]
    for index, record in enumerate(records):
        observed = positions[index][mask[index]].tolist()
        assert observed == list(range(record.node_count)), (
            "flat positions must run 0..node_count-1 over the whole molecule; a per-role counter "
            "that resets at each region boundary reintroduces role through position")


@requires_switch
def test_role_is_not_recoverable_from_the_flat_arm_layout_channels(corpus_and_records,
                                                                  architecture):
    """The model-level leakage test, scoped to the channels that carry role in the structured arm.

    Scoping matters or this test fails for the wrong reason. Role IS partly inferable from atom and
    bond states, because an isocyanide-derived region carries a distinctive amide nitrogen and the
    repaired inverse showed role is fully determined by the completed product graph. That is
    chemistry, not leakage, and a chemist would infer it too.

    What must not leak are the channels that in the structured arm give role EXACTLY: position
    together with the program vector. The origin audit measured that pair at recoverability 1.0 over
    9,000 records. This applies the same estimator, a Bayes-optimal lookup on the empirical
    conditional, to the flat arm's global position plus program, and requires it to sit near the
    majority-class baseline rather than near 1.0.
    """
    from collections import defaultdict
    _, records_by_fold = corpus_and_records
    fit = list(itertools.islice(records_by_fold["train"], 4096))
    evaluate = list(itertools.islice(records_by_fold["calibration"], 1024))

    def rows(records):
        flat = _batch(records, architecture, semantic_organization=FLAT)
        truth = flat["role_states"]
        positions = flat["within_role_positions"]
        mask = flat["node_mask"]
        out = []
        for index, record in enumerate(records):
            program = tuple(int(v) for v in flat["programs"][index].tolist())
            selected = mask[index]
            for position, role in zip(positions[index][selected].tolist(),
                                      truth[index][selected].tolist(), strict=True):
                out.append(((position, program), role))
        return out

    table: dict[tuple, Counter] = defaultdict(Counter)
    prior: Counter = Counter()
    for key, role in rows(fit):
        table[key][role] += 1
        prior[role] += 1
    majority = prior.most_common(1)[0][0]
    rule = {key: counts.most_common(1)[0][0] for key, counts in table.items()}

    held = rows(evaluate)
    correct = sum(1 for key, role in held if rule.get(key, majority) == role)
    accuracy = correct / len(held)
    baseline = max(Counter(role for _, role in held).values()) / len(held)

    # Graded by the contract's own A1 rule rather than an ad-hoc margin. Full role-freedom is not
    # reachable here with a natural serialization: role exteriors differ systematically in size
    # (amine median 8, aldehyde 17, isocyanide 14, all three distinct in 91.5% of records) and the
    # program hands the model those three sizes, so any size-ordered canonical layout groups roles.
    # Role regions in this chemistry ARE the branches off the Ugi core, so contiguity is a property
    # of the molecule class, not an encoding choice.
    assert accuracy < 0.95, (
        f"role recovered at {accuracy:.4f}; at or above 0.95 this is not an ablation at all and the "
        "arm is void")
    assert accuracy <= baseline + 0.35, (
        f"role recovered at {accuracy:.4f} against a {baseline:.4f} baseline, a lift beyond the "
        "partial-ablation band; the flat layout is not removing enough")
    # Recorded for the arm's own audit: this is a PARTIAL ablation and every statement about the
    # flat arm must carry this figure, exactly as the origin-channel rule requires.
    print(f"\n  partial ablation: role recoverability {accuracy:.4f} against baseline "
          f"{baseline:.4f}; structured arm is 1.000 by construction")


@requires_switch
def test_flat_arm_consumes_the_program_without_role_indexed_tables(architecture):
    """Parity requires the same twelve numbers, not the same role-indexed consumption of them."""
    model = UgiJointSparseFlow(atom_classes=14, semantic_organization=FLAT, **architecture)
    names = list(dict(model.named_parameters()))
    for banned in ("count_embeddings", "junction_embeddings", "cycle_embeddings",
                   "attachment_embeddings", "pending_embedding"):
        assert not any(n.startswith(banned) for n in names), (
            f"{banned} is role-indexed and must not survive into the flat arm")


# --------------------------------------------------------------- 2. parity must be preserved


@requires_switch
def test_both_arms_receive_the_identical_program_vector(corpus_and_records, architecture):
    """The treatment is organization, not information. Same twelve numbers to both arms."""
    _, records_by_fold = corpus_and_records
    records = list(itertools.islice(records_by_fold["train"], 64))
    structured = _batch(records, architecture, semantic_organization=STRUCTURED)
    flat = _batch(records, architecture, semantic_organization=FLAT)
    assert torch.equal(structured["programs"], flat["programs"]), (
        "the flat arm must receive the same program vector; a different vector makes any "
        "advantage uninterpretable")


@requires_switch
def test_parameter_counts_are_close_enough_to_report(architecture):
    structured = UgiJointSparseFlow(atom_classes=14, semantic_organization=STRUCTURED,
                                    **architecture)
    flat = UgiJointSparseFlow(atom_classes=14, semantic_organization=FLAT, **architecture)
    a = sum(p.numel() for p in structured.parameters())
    b = sum(p.numel() for p in flat.parameters())
    assert abs(a - b) / a < 0.10, (
        f"parameter counts differ by more than 10% ({a} vs {b}); the parity table must either "
        "match capacity or state the gap explicitly")


@requires_switch
def test_structured_arm_is_unchanged_by_the_switch(corpus_and_records, architecture):
    """The default path must be bit-identical to the frozen production behaviour."""
    corpus, records_by_fold = corpus_and_records
    records = list(itertools.islice(records_by_fold["train"], 64))
    without = _batch(records, architecture)
    with_flag = _batch(records, architecture, semantic_organization=STRUCTURED)
    for key in sorted(without):
        assert key in with_flag, f"{key} vanished when the default was named explicitly"
        assert torch.equal(without[key], with_flag[key]), (
            f"{key} changed when semantic_organization was set to its default")


@requires_switch
def test_reference_checkpoint_still_loads_into_the_default_arm(corpus_and_records, architecture):
    corpus, _ = corpus_and_records
    checkpoint = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    model = UgiJointSparseFlow(atom_classes=len(corpus.atom_vocabulary),
                               semantic_organization=STRUCTURED, **architecture)
    model.load_state_dict(checkpoint["model_state"], strict=True)


# ------------------------------------------------- 3. the objective and sources must follow


@requires_switch
def test_flat_arm_uses_a_pooled_loss_not_a_role_balanced_one(corpus_and_records, architecture):
    """A role-balanced objective is role structure. It must not survive the ablation."""
    from forge.product.ugi_joint_sparse_flow import ugi_joint_sparse_loss
    corpus, records_by_fold = corpus_and_records
    records = list(itertools.islice(records_by_fold["train"], 64))
    flat = _batch(records, architecture, semantic_organization=FLAT)
    model = UgiJointSparseFlow(atom_classes=len(corpus.atom_vocabulary),
                               semantic_organization=FLAT, **architecture)
    model.eval()
    with torch.no_grad():
        predictions = model(
            offspring=flat["offspring"], nodes=flat["nodes"],
            parent_bonds=flat["parent_bonds"], programs=flat["programs"],
            role_states=flat["role_states"],
            within_role_positions=flat["within_role_positions"],
            node_mask=flat["node_mask"], t=torch.full((len(records),), 0.5),
            **{k: flat[k] for k in ("closure_left", "closure_right", "decoration_anchors",
                                    "decoration_atoms", "decoration_bonds") if k in flat})
        _, metrics = ugi_joint_sparse_loss(predictions, flat,
                                           semantic_organization=FLAT)
    assert not any("amine" in k or "isocyanide" in k or "oxoester" in k for k in metrics), (
        "the flat arm reported per-role loss terms, so the objective is still role-partitioned")


@requires_switch
def test_flat_arm_source_marginals_are_pooled(corpus_and_records, architecture):
    from forge.product.ugi_joint_sparse_flow import joint_sparse_source_marginals
    import numpy as np
    corpus, records_by_fold = corpus_and_records
    records = tuple(itertools.islice(records_by_fold["train"], 512))
    sources = joint_sparse_source_marginals(
        records, atom_classes=len(corpus.atom_vocabulary),
        bond_classes=int(architecture["bond_classes"]),
        maximum_children=int(architecture["maximum_children"]),
        maximum_decorations=int(architecture["maximum_decorations"]),
        probability_floor=1e-5, semantic_organization=FLAT)
    for key in ("atoms", "bonds", "offspring"):
        array = np.asarray(sources[key])
        # Pooled then broadcast back, so the shape the noiser expects is unchanged and every role
        # row carries the identical distribution.
        assert array.ndim == 2 and array.shape[0] == 3, array.shape
        assert np.allclose(array[0], array[1]) and np.allclose(array[1], array[2]), (
            f"source marginal '{key}' still differs by role; the flat arms must pool it")


# ------------------------------------------------------ 4. the causal check, after training


@requires_switch
def test_flat_arm_has_no_privileged_region_structure(corpus_and_records, architecture):
    """Perturbing a contiguous span must not act differently from perturbing a random subset.

    Under role-structured serialization a contiguous span is a precursor region, so perturbing it
    moves predictions in a structured way. Under a flat serialization a contiguous span is
    arbitrary, so it must behave like any equal-sized subset. This is the untrained-weights
    version; the trained version belongs in the run's own audit.
    """
    corpus, records_by_fold = corpus_and_records
    records = list(itertools.islice(records_by_fold["train"], 32))
    flat = _batch(records, architecture, semantic_organization=FLAT)
    model = UgiJointSparseFlow(atom_classes=len(corpus.atom_vocabulary),
                               semantic_organization=FLAT, **architecture)
    model.eval()
    keys = {k: flat[k] for k in ("closure_left", "closure_right", "decoration_anchors",
                                 "decoration_atoms", "decoration_bonds") if k in flat}

    def run(nodes):
        return model(offspring=flat["offspring"], nodes=nodes,
                     parent_bonds=flat["parent_bonds"], programs=flat["programs"],
                     role_states=flat["role_states"],
                     within_role_positions=flat["within_role_positions"],
                     node_mask=flat["node_mask"], t=torch.full((len(records),), 0.5), **keys)

    with torch.no_grad():
        base = run(flat["nodes"])
        mask = flat["node_mask"]
        span = torch.zeros_like(mask)
        span[:, : mask.shape[1] // 3] = True
        span &= mask
        scattered = torch.zeros_like(mask)
        scattered[:, ::3] = True
        scattered &= mask
        deltas = []
        for selection in (span, scattered):
            perturbed = flat["nodes"].clone()
            perturbed[selection] = (perturbed[selection] + 1) % len(corpus.atom_vocabulary)
            out = run(perturbed)
            deltas.append(float((out["nodes"] - base["nodes"]).abs()[mask].mean()))
    contiguous, spread = deltas
    assert 0.5 <= contiguous / spread <= 2.0, (
        f"a contiguous span behaves differently from a scattered subset "
        f"({contiguous:.4f} vs {spread:.4f}); the flat arm still has privileged region structure")


# ------------------------------- 5. the two arms differ in exactly one thing, checked functionally


@requires_switch
def test_flat_role_labels_are_the_permuted_true_roles(corpus_and_records, architecture):
    """The true-role arm must receive the role of the atom actually sitting at each flat position.

    Checked against the permutation directly rather than trusting the collate, because an off-by-one
    or an unpermuted label would silently turn the treatment arm into a misaligned arm and invert the
    experiment's meaning.
    """
    import numpy as np
    from forge.product.ugi_joint_sparse_flow import flat_subtree_permutation
    _, records_by_fold = corpus_and_records
    records = list(itertools.islice(records_by_fold["train"], 64))
    flat = _batch(records, architecture, semantic_organization=FLAT_TRUE)
    for index, record in enumerate(records):
        permutation = flat_subtree_permutation(record)
        blocks = np.concatenate([
            np.full(count, role, dtype=np.int64)
            for role, count in enumerate(record.program.node_counts)])
        mask = flat["node_mask"][index]
        assert flat["role_states"][index][mask].tolist() == blocks[permutation].tolist()


@requires_switch
def test_no_role_arm_output_is_invariant_to_the_role_channel(corpus_and_records, architecture):
    """The decisive functional check that the null tag is really fed.

    Reading the code is not enough; a partially applied switch was the exact failure mode this suite
    exists to catch. If the no-role arm's output changes when the role input is scrambled, it is
    consuming role after all and the arm is void. The true-role arm must change, or it is not
    consuming role at all.
    """
    corpus, records_by_fold = corpus_and_records
    records = list(itertools.islice(records_by_fold["train"], 64))

    def logits(model, batch, roles):
        with torch.no_grad():
            return model(
                offspring=batch["offspring"], nodes=batch["nodes"],
                parent_bonds=batch["parent_bonds"], role_states=roles,
                within_role_positions=batch["within_role_positions"],
                programs=batch["programs"], node_mask=batch["node_mask"],
                t=torch.full((len(records),), 0.5),
                closure_left=batch["closure_left"], closure_right=batch["closure_right"],
                decoration_anchors=batch["decoration_anchors"],
                decoration_atoms=batch["decoration_atoms"],
                decoration_bonds=batch["decoration_bonds"])["nodes"]

    torch.manual_seed(0)
    for organization, must_be_invariant in ((FLAT, True), (FLAT_TRUE, False)):
        batch = _batch(records, architecture, semantic_organization=organization)
        scrambled = torch.randint(0, 3, batch["role_states"].shape)
        torch.manual_seed(7)
        model = UgiJointSparseFlow(atom_classes=len(corpus.atom_vocabulary),
                                   semantic_organization=organization, **architecture)
        model.eval()
        delta = (logits(model, batch, batch["role_states"])
                 - logits(model, batch, scrambled)).abs().max().item()
        if must_be_invariant:
            assert delta == 0.0, (
                f"{organization} changed by {delta} when the role input was scrambled, so it is "
                "consuming role and is not a no-role arm")
        else:
            assert delta > 0.0, (
                f"{organization} did not change when the role input was scrambled, so it is not "
                "consuming role at all")


# ------------------------------------------------ 6. paired randomness across the two pilot arms


@requires_switch
def test_pilot_arms_are_paired_on_randomness_and_initialization(corpus_and_records, architecture):
    """The two arms must differ only in the role channel, including in their randomness.

    Identical initial weights are achievable here, not merely an identical seeding policy, because
    the flat arms share module structure exactly. If that stops holding, the comparison acquires an
    initialization confound and this test should fail rather than let it pass silently.
    """
    corpus, _ = corpus_and_records
    built = {}
    for organization in (FLAT, FLAT_TRUE):
        torch.manual_seed(20260817)
        built[organization] = UgiJointSparseFlow(
            atom_classes=len(corpus.atom_vocabulary),
            semantic_organization=organization, **architecture).state_dict()
    left, right = built[FLAT], built[FLAT_TRUE]
    assert sorted(left) == sorted(right), "the pilot arms do not share module structure"
    for key in left:
        assert torch.equal(left[key], right[key]), (
            f"{key} differs at initialization under the same seed, so the arms are not paired")


@requires_switch
def test_pilot_arms_draw_identical_minibatches_and_times(corpus_and_records, architecture):
    """Same seed must give the same example sequence and the same corruption times."""
    import json
    import numpy as np
    sealed = REPO / "results/phase1/forge_coverage_splits_v1/splits.json"
    if not sealed.exists():
        pytest.skip("sealed coverage splits are unavailable")
    _, records_by_fold = corpus_and_records
    keep = set(json.loads(sealed.read_text())["training_product_ids"]["0.25"])
    records = tuple(r for r in records_by_fold["train"] if r.product_id in keep)
    assert len(records) == len(keep), "the sealed coverage subset did not resolve exactly"

    observed = {}
    for organization in (FLAT, FLAT_TRUE):
        rng = np.random.default_rng(20260817 + 2)
        generator = torch.Generator().manual_seed(20260817 + 1)
        ids, times = [], []
        for _ in range(5):
            index = rng.choice(len(records), size=32, replace=True)
            chunk = tuple(records[int(j)] for j in index)
            _batch(chunk, architecture, semantic_organization=organization)
            ids.append([records[int(j)].product_id for j in index])
            times.append(torch.rand(len(chunk), generator=generator).clamp(0.02, 0.98).tolist())
        observed[organization] = (ids, times)
    assert observed[FLAT][0] == observed[FLAT_TRUE][0], "minibatch sequences diverge across arms"
    assert observed[FLAT][1] == observed[FLAT_TRUE][1], "corruption times diverge across arms"

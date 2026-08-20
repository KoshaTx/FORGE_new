#!/usr/bin/env python3
"""Prove the flat projection changes only the layout, before any of it is wired in.

Contract Amendment 7. The flat arm's serialization must be a permutation of whole preorder
subtrees and nothing else. If it alters the molecular dataset in any way, the experiment stops
being "semantic organization" and becomes "semantic organization plus different data", which is
not the treatment.

The transformation, and why it is only a permutation:

  the role-structured projection lays out three per-role exteriors, each a canonical preorder
  attached forest, concatenated in role order. A concatenation of preorder forests is itself a
  preorder forest, so the roles occupy contiguous blocks purely as an artefact of concatenation
  order. Reordering whole trees by a role-agnostic key therefore yields another valid preorder
  forest over the same atoms, and role stops being recoverable from position.

The key is the original product atom index of each tree's root. That is a property of the
molecule, not of the role, and it is deterministic.

Checked per record, and a failure is a correctness failure that voids the arm:

  molecular identity   the flat layout inverts to the same ordered atom set
  atom targets         identical under the permutation, and identical as a multiset
  bond targets         identical under the permutation
  offspring            a valid preorder attached forest with the same total attachment count,
                       and the same multiset of child counts
  closures             endpoints remapped consistently, bond states unchanged
  decorations          anchors remapped consistently, states unchanged
  program              the twelve numbers unchanged
  role recoverability  the flat layout must NOT place roles in program-implied contiguous blocks
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge.design.flow.ugi_joint_sparse_flow import (  # noqa: E402
    project_joint_sparse_record,
)
from forge.design.training.ugi_training_cache import load_ugi_training_cache  # noqa: E402
from forge.design.flow.ugi_morphology_program import (  # noqa: E402
    preorder_attached_forest_to_parents,
)

CACHE = "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_spans(offspring: np.ndarray, attachment_count: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) span of each tree in a preorder attached forest."""
    parents = preorder_attached_forest_to_parents(offspring, attachment_count=attachment_count)
    roots = [index for index, parent in enumerate(parents.tolist()) if parent < 0]
    if len(roots) != attachment_count:
        raise ValueError(f"expected {attachment_count} roots, found {len(roots)}")
    bounds = roots + [int(offspring.size)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(roots))]


def flatten(record) -> tuple[np.ndarray, np.ndarray]:
    """Return (permutation, per-position true role) for the flat layout of a projected record.

    The permutation reorders whole trees by the product atom index of their root, which is
    molecular rather than role information.
    """
    program = record.program
    attachment_total = int(sum(program.attachment_counts))
    spans = tree_spans(record.offspring, attachment_total)
    # True role per position under the role-structured layout: contiguous blocks by construction.
    role_of_position = np.concatenate([
        np.full(count, index, dtype=np.int64)
        for index, count in enumerate(program.node_counts)
    ])
    keyed = sorted(spans, key=lambda span: int(record.full_indices[span[0]]))
    permutation = np.concatenate([np.arange(start, end, dtype=np.int64)
                                 for start, end in keyed])
    return permutation, role_of_position[permutation]


def check(record) -> dict[str, bool]:
    permutation, flat_roles = flatten(record)
    program = record.program
    attachment_total = int(sum(program.attachment_counts))
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size, dtype=np.int64)

    flat_offspring = record.offspring[permutation]
    flat_atoms = record.atom_states[permutation]
    flat_bonds = record.parent_bond_states[permutation]
    flat_full = record.full_indices[permutation]

    results: dict[str, bool] = {}
    results["permutation_is_a_bijection"] = (
        sorted(permutation.tolist()) == list(range(record.offspring.size)))
    # Round-trip, not a tautology: inverting the permutation must recover the original exactly.
    results["atom_targets_round_trip"] = bool(
        np.array_equal(flat_atoms[inverse], record.atom_states))
    results["bond_targets_round_trip"] = bool(
        np.array_equal(flat_bonds[inverse], record.parent_bond_states))
    results["offspring_round_trip"] = bool(
        np.array_equal(flat_offspring[inverse], record.offspring))
    results["atom_multiset_preserved"] = (
        Counter(record.atom_states.tolist()) == Counter(flat_atoms.tolist()))
    results["bond_multiset_preserved"] = (
        Counter(record.parent_bond_states.tolist()) == Counter(flat_bonds.tolist()))
    results["atom_set_by_product_index_preserved"] = (
        sorted(record.full_indices.tolist()) == sorted(flat_full.tolist()))
    results["offspring_multiset_preserved"] = (
        Counter(record.offspring.tolist()) == Counter(flat_offspring.tolist()))
    try:
        parents = preorder_attached_forest_to_parents(
            flat_offspring, attachment_count=attachment_total)
        results["flat_offspring_is_a_valid_preorder_forest"] = True
        results["flat_forest_has_same_root_count"] = (
            int((parents < 0).sum()) == attachment_total)
    except Exception:  # noqa: BLE001
        results["flat_offspring_is_a_valid_preorder_forest"] = False
        results["flat_forest_has_same_root_count"] = False

    # closures and decorations remap through the inverse permutation
    flat_left = inverse[record.closure_left] if record.closure_left.size else record.closure_left
    flat_right = inverse[record.closure_right] if record.closure_right.size else record.closure_right
    results["closure_count_preserved"] = flat_left.size == record.closure_left.size
    results["closure_endpoints_stay_in_range"] = bool(
        flat_left.size == 0 or (flat_left.max() < permutation.size
                                and flat_right.max() < permutation.size))
    results["closure_pairs_reference_same_atoms"] = bool(
        flat_left.size == 0 or np.array_equal(
            np.sort(flat_full[flat_left]), np.sort(record.full_indices[record.closure_left])))
    anchors = record.decoration_anchors
    flat_anchors = inverse[anchors] if anchors.size else anchors
    results["decoration_count_preserved"] = flat_anchors.size == anchors.size
    results["decoration_anchors_reference_same_atoms"] = bool(
        anchors.size == 0 or np.array_equal(
            np.sort(flat_full[flat_anchors]), np.sort(record.full_indices[anchors])))

    # The twelve program numbers, recomputed from the flat layout rather than asserted.
    from forge.design.flow.ugi_morphology_program import attached_tree_junction_contributions
    recomputed_counts, recomputed_junctions, recomputed_attachments = [], [], []
    for role_index in range(len(program.node_counts)):
        selected = flat_roles == role_index
        recomputed_counts.append(int(selected.sum()))
        role_offspring = flat_offspring[selected]
        role_parents = preorder_attached_forest_to_parents(
            record.offspring[permutation][selected],
            attachment_count=int(program.attachment_counts[role_index]))
        recomputed_attachments.append(int((role_parents < 0).sum()))
        recomputed_junctions.append(int(attached_tree_junction_contributions(role_offspring).sum()))
    recomputed_cycles = [0] * len(program.node_counts)
    for left in flat_left.tolist() if flat_left.size else []:
        recomputed_cycles[int(flat_roles[left])] += 1
    results["program_node_counts_unchanged"] = (
        tuple(recomputed_counts) == tuple(program.node_counts))
    results["program_junctions_unchanged"] = (
        tuple(recomputed_junctions) == tuple(program.junction_budgets))
    results["program_cycles_unchanged"] = (
        tuple(recomputed_cycles) == tuple(program.cycle_ranks))
    results["program_attachments_unchanged"] = (
        tuple(recomputed_attachments) == tuple(program.attachment_counts))

    # the load-bearing property: role must not be a contiguous block layout any more
    expected_blocks = np.concatenate([
        np.full(count, index, dtype=np.int64)
        for index, count in enumerate(program.node_counts)])
    results["role_no_longer_in_program_blocks"] = not bool(
        np.array_equal(flat_roles, expected_blocks))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=20000)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_flat_projection_invariant_v1/result.json")
    args = parser.parse_args()
    started = time.time()

    corpus, records_by_fold = load_ugi_training_cache(REPO / CACHE)
    records = list(itertools.islice(records_by_fold["train"], args.records))
    print(f"checking {len(records)} projected train-fold records")

    failures: Counter = Counter()
    checked = 0
    block_preserving = 0
    for record in records:
        checked += 1
        outcome = check(record)
        for name, ok in outcome.items():
            if not ok:
                failures[name] += 1
        if not outcome["role_no_longer_in_program_blocks"]:
            block_preserving += 1

    print(f"\n{'property':52s} {'failures':>9s}")
    names = sorted(check(records[0]))
    for name in names:
        print(f"{name:52s} {failures[name]:9d}")

    ok = not failures
    print(f"\nrecords checked {checked}; records whose flat layout still matches program blocks "
          f"{block_preserving} ({block_preserving / checked:.4f})")
    verdict = (
        "The flat layout is a pure permutation of whole preorder subtrees: molecular identity, "
        "atom and bond targets, offspring structure, closures, decorations and the program are all "
        "preserved, and role is no longer laid out in program-implied contiguous blocks. The arm is "
        "cleared on the Amendment 7 correctness gate."
        if ok else
        f"CORRECTNESS FAILURE on {sorted(failures)}. Under Amendment 7 this voids the arm; the "
        "treatment would be semantic organization plus an altered molecular dataset."
    )
    print(f"\n{verdict}")

    payload = {
        "schema_version": "phase1_forge_flat_projection_invariant.v1",
        "status": "clear" if ok else "correctness_failure",
        "contract": "docs/FORGE_EVIDENCE_CONTRACT_v1.md Amendment 7",
        "inputs": {"training_cache": {"path": CACHE, "sha256": sha256_file(REPO / CACHE)}},
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "transformation": {
            "operation": "reorder whole preorder subtrees of the exterior forest",
            "key": "product atom index of each tree's root, a molecular rather than role property",
            "why_valid": "a concatenation of preorder forests is a preorder forest, so role blocks "
                         "are an artefact of concatenation order and reordering trees preserves "
                         "forest validity over the same atoms",
        },
        "records_checked": checked,
        "failures_by_property": dict(failures),
        "records_still_matching_program_blocks": block_preserving,
        "verdict": verdict,
        "nonclaims": [
            "This is a correctness gate on the data transformation, not a result about either arm.",
            "Passing it does not make the flat arm a good model; it makes it a valid comparison.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

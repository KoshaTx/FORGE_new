"""Oracle TRAIN sensitivity of bond predictions to correctly committed atom states.

No graph is generated. Target ester commitments use the existing decoder's motif
routine, and their bonds are excluded from all evaluated coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from forge.model.synthesis_program_sampling import _ugi_ester_motif_constraints
from forge.model.ugi_realism_model_attribution import (
    classification_summary,
    masked_probabilities,
    select_measured_train_rows,
)


class UgiCommittedBondProbeError(ValueError):
    """The prespecified probe or its paired-coordinate invariants failed."""


def select_component_covering_panel(
    rows: Sequence[Mapping[str, str]], *, roles: Sequence[str], records: int, seed: int
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Select admitted TRAIN rows before predictions; fail at the requested coverage bound."""

    selected, receipt = select_measured_train_rows(rows, roles=roles, records=records, seed=seed)
    if receipt["selected_components_by_role"] != receipt["available_components_by_role"]:
        raise UgiCommittedBondProbeError(
            f"fixed panel of {records} does not cover every admitted measured TRAIN component"
        )
    receipt["all_admitted_measured_components_covered"] = True
    return selected, receipt


def oracle_ester_commitment_masks(
    clean: Mapping[str, Any],
    records: Sequence[Any],
    atom_vocabulary: Sequence[Any],
    policy: Any,
    *,
    bond_classes: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Locate the target motif with clean one-hot scores using the unchanged decoder routine.

    This is oracle mask construction, not a model prediction or a sampled terminal decision.
    A returned atom or bond that disagrees with the target causes failure.
    """

    masks = {
        field: torch.zeros_like(clean[field], dtype=torch.bool)
        for field in ("parent_bonds", "closure_bonds")
    }
    if type(bond_classes) is not int or bond_classes < 1:
        raise UgiCommittedBondProbeError("declared model bond classes are invalid")
    predictions = {
        field: torch.nn.functional.one_hot(clean[field], num_classes=count).float().numpy()
        for field, count in (
            ("nodes", len(atom_vocabulary)),
            ("parent_bonds", bond_classes),
            ("closure_bonds", bond_classes),
        )
    }
    receipts = []
    for index, record in enumerate(records):
        parents = record.graph.parents
        left, right = record.graph.closure_left, record.graph.closure_right
        neighbors = [set() for _ in range(record.node_count)]
        coordinates = {}
        for child in range(1, record.node_count):
            pair = tuple(sorted((child, int(parents[child]))))
            coordinates[pair] = ("parent_bonds", child)
        for slot, (a, b) in enumerate(zip(left, right, strict=True)):
            pair = tuple(sorted((int(a), int(b))))
            if pair in coordinates:
                raise UgiCommittedBondProbeError("target topology repeats an edge")
            coordinates[pair] = ("closure_bonds", slot)
        for a, b in coordinates:
            neighbors[a].add(b)
            neighbors[b].add(a)
        motif = _ugi_ester_motif_constraints(
            predictions, index, record, atom_vocabulary, parents, left, right, neighbors, policy
        )
        if motif is None or not motif[1]:
            raise UgiCommittedBondProbeError("admitted target has no supported ester commitment")
        atoms, bonds = motif
        if any(
            atom_vocabulary[int(clean["nodes"][index, node])].symbol != symbol
            for node, symbol in atoms.items()
        ):
            raise UgiCommittedBondProbeError("oracle ester atom assignment disagrees with target")
        entries = []
        for pair, state in sorted(bonds.items()):
            field, coordinate = coordinates[pair]
            variable = clean[
                (
                    "parent_bond_variable_mask"
                    if field == "parent_bonds"
                    else "closure_bond_variable_mask"
                )
            ]
            if not variable[index, coordinate] or int(clean[field][index, coordinate]) != state:
                raise UgiCommittedBondProbeError(
                    "oracle ester bond assignment disagrees with target"
                )
            masks[field][index, coordinate] = True
            entries.append({"field": field, "coordinate": coordinate, "target_state": int(state)})
        receipts.append({"record_index": index, "committed_bonds": entries})
    return masks, receipts


def paired_committed_states(
    clean: Mapping[str, torch.Tensor],
    noisy: Mapping[str, torch.Tensor],
    commitments: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Construct matched inputs without changing any uncommitted bond coordinate."""

    stale = {key: value.clone() for key, value in noisy.items()}
    for field in ("parents", "closure_left", "closure_right"):
        stale[field] = clean[field].clone()
    refreshed = {key: value.clone() for key, value in stale.items()}
    refreshed["nodes"] = clean["nodes"].clone()
    for field, variable_name in (
        ("parent_bonds", "parent_bond_variable_mask"),
        ("closure_bonds", "closure_bond_variable_mask"),
    ):
        mask = commitments[field]
        if mask.dtype != torch.bool or mask.shape != clean[field].shape:
            raise UgiCommittedBondProbeError("commitment mask shape/dtype changed")
        if torch.any(mask & ~clean[variable_name]):
            raise UgiCommittedBondProbeError("commitment includes fixed or padded coordinates")
        refreshed[field] = torch.where(mask, clean[field], stale[field])
        if not torch.equal(refreshed[field][~mask], stale[field][~mask]):
            raise UgiCommittedBondProbeError("evaluated bond inputs changed")
    for field, variable in (
        ("nodes", "atom_variable_mask"),
        ("parents", "parent_variable_mask"),
        ("parent_bonds", "parent_bond_variable_mask"),
        ("closure_left", "closure_endpoint_variable_mask"),
        ("closure_right", "closure_endpoint_variable_mask"),
        ("closure_bonds", "closure_bond_variable_mask"),
    ):
        if torch.any((noisy[field] != clean[field]) & ~clean[variable]):
            raise UgiCommittedBondProbeError(f"fixed-state noising violation: {field}")
    return stale, refreshed


def _bond_arrays(
    clean: Mapping[str, torch.Tensor],
    noisy: Mapping[str, torch.Tensor],
    commitments: Mapping[str, torch.Tensor],
    predictions: Mapping[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    closure_roles = clean["role_states"].gather(1, clean["closure_left"])
    closure_other = clean["role_states"].gather(1, clean["closure_right"])
    exterior_closure = (
        (clean["core_position_states"].gather(1, clean["closure_left"]) == 1)
        & (clean["core_position_states"].gather(1, clean["closure_right"]) == 1)
        & (closure_roles == closure_other)
    )
    parent_active = clean["parent_bond_variable_mask"] & clean["node_mask"]
    if torch.any(parent_active & (clean["core_position_states"] != 1)) or torch.any(
        clean["closure_bond_variable_mask"] & ~exterior_closure
    ):
        raise UgiCommittedBondProbeError("variable bond outside an exterior role")
    masks = (
        parent_active & ~commitments["parent_bonds"],
        clean["closure_bond_variable_mask"] & ~commitments["closure_bonds"],
    )
    fields = ("parent_bonds", "closure_bonds")
    tensors = (
        torch.cat([predictions[field] for field in fields], dim=1),
        torch.cat([clean[field] for field in fields], dim=1),
        torch.cat([noisy[field] for field in fields], dim=1),
        torch.cat(masks, dim=1),
        torch.cat((clean["role_states"], closure_roles), dim=1),
    )
    return tuple(value.detach().cpu().numpy() for value in tensors)


def summarize_bond_response(
    *,
    clean: Mapping[str, torch.Tensor],
    noisy: Mapping[str, torch.Tensor],
    commitments: Mapping[str, torch.Tensor],
    stale: Mapping[str, torch.Tensor],
    repeat: Mapping[str, torch.Tensor],
    refreshed: Mapping[str, torch.Tensor],
    roles: Mapping[str, int],
    repeat_atol: float,
) -> dict[str, Any]:
    """Score uncommitted bonds, with corrupted/unchanged denominators kept separate."""

    if not np.isfinite(repeat_atol) or repeat_atol < 0:
        raise UgiCommittedBondProbeError("invalid repeat tolerance")
    arrays = {
        name: _bond_arrays(clean, noisy, commitments, values)
        for name, values in (("stale", stale), ("repeat", repeat), ("refreshed", refreshed))
    }
    logits, targets, corrupted_input, active, role_states = arrays["stale"]
    if np.any(active & ~np.isin(role_states, list(roles.values()))):
        raise UgiCommittedBondProbeError("active bond has an unreported role")
    probabilities = {}
    for name, values in arrays.items():
        if values[0].shape != logits.shape:
            raise UgiCommittedBondProbeError("paired bond prediction shapes differ")
        probs = np.zeros_like(values[0], dtype=np.float64)
        probs[active] = masked_probabilities(
            values[0][active], np.ones_like(values[0][active], dtype=bool)
        )
        probabilities[name] = probs
    actually_corrupted = active & (targets != corrupted_input)
    repeat_delta = (
        float(np.max(np.abs(logits[active] - arrays["repeat"][0][active]))) if active.any() else 0.0
    )
    by_role = {}
    for role, state in roles.items():
        role_mask = active & (role_states == state)
        populations = {
            "all_uncommitted": role_mask,
            "actually_corrupted": role_mask & actually_corrupted,
            "unchanged": role_mask & ~actually_corrupted,
        }
        by_role[role] = {}
        for population, mask in populations.items():
            by_role[role][population] = {
                name: classification_summary(probs[mask], targets[mask])
                for name, probs in probabilities.items()
            }
    # Equal role weighting avoids long tails dominating the diagnostic's primary NLL.
    eligible_roles = [
        role for role in roles if by_role[role]["actually_corrupted"]["stale"]["total"]
    ]
    balanced = {
        name: (
            float(
                np.mean(
                    [by_role[role]["actually_corrupted"][name]["nll"] for role in eligible_roles]
                )
            )
            if eligible_roles
            else None
        )
        for name in probabilities
    }
    prediction = {name: value.argmax(axis=-1) for name, value in probabilities.items()}
    corrections = int(
        np.sum(
            actually_corrupted
            & (prediction["stale"] != targets)
            & (prediction["refreshed"] == targets)
        )
    )
    regressions = int(
        np.sum(
            actually_corrupted
            & (prediction["stale"] == targets)
            & (prediction["refreshed"] != targets)
        )
    )
    exact = {}
    for label, mask in (("all_uncommitted", active), ("actually_corrupted", actually_corrupted)):
        eligible = mask.any(axis=1)
        exact[label] = {
            "records": int(eligible.sum()),
            **{
                name: int(np.sum(eligible & np.all((values == targets) | ~mask, axis=1)))
                for name, values in prediction.items()
            },
        }
    repeat_nll_delta = abs(balanced["stale"] - balanced["repeat"]) if eligible_roles else None
    improvement = balanced["stale"] - balanced["refreshed"] if eligible_roles else None
    gate = {
        "all_roles_have_corrupted_uncommitted_bonds": len(eligible_roles) == len(roles),
        "identical_input_repeat_within_tolerance": repeat_delta <= repeat_atol,
        "role_balanced_nll_improves_beyond_repeat": bool(
            improvement is not None and improvement > max(repeat_atol, repeat_nll_delta)
        ),
        "no_role_corrupted_accuracy_loss": all(
            by_role[role]["actually_corrupted"]["refreshed"]["correct"]
            >= by_role[role]["actually_corrupted"]["stale"]["correct"]
            for role in eligible_roles
        ),
        "at_least_one_corrupted_bond_correction": corrections > 0,
    }
    return {
        "by_role": by_role,
        "role_balanced_corrupted_target_nll": balanced,
        "role_balanced_nll_improvement": improvement,
        "identical_input_repeat_max_logit_delta": repeat_delta,
        "identical_input_repeat_role_balanced_nll_delta": repeat_nll_delta,
        "repeat_atol": repeat_atol,
        "corrections": corrections,
        "regressions": regressions,
        "exact_bond_vector_recovery": exact,
        "gate_checks": gate,
        "gate_passed": all(gate.values()),
        "uncommitted_bonds": int(active.sum()),
        "actually_corrupted_uncommitted_bonds": int(actually_corrupted.sum()),
        "excluded_committed_bonds": {key: int(value.sum()) for key, value in commitments.items()},
    }

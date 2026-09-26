"""Family-batch partitioning with global objective denominators."""

import torch
from torch.nn import functional

from forge.model.compose_lipid_training import validate_objective
from forge.model.reaction_program_flow import (
    _endpoint_candidate_mask,
    _parent_candidate_mask,
    _parent_group_cross_entropy,
)
from forge.model.reaction_program_transformer import synthesis_program_offspring_targets


def select(values, selection):
    return {key: value[selection] for key, value in values.items()}


class FamilyLoss:
    """Sum shard contributions to the existing full-family objective, never local means.

    Every denominator depends only on the complete clean family batch. Repeat pairs
    are within records, so partitioning rows cannot divide an admitted pair.
    """

    def __init__(self, clean, maximum_children):
        self.clean = clean
        self.offspring, self.exterior = synthesis_program_offspring_targets(
            clean, maximum_children=maximum_children
        )
        self.roles = clean["role_states"]
        self.closure_roles = self.roles.gather(1, clean["closure_left"])
        self.maximum_role = int(self.roles.max())
        self.repeat_counts = {}
        for field in ("repeat_atom_groups", "repeat_bond_groups"):
            groups = clean[field]
            if bool((groups < 0).any()):
                raise ValueError("Negative repeat group")
            pairs = torch.triu(
                (groups[:, :, None] == groups[:, None, :]) & (groups[:, :, None] > 0),
                diagonal=1,
            )
            self.repeat_counts[field] = pairs.sum()
        morphology = clean["role_morphology_states"]
        self.junctions = []
        for role in range(1, self.maximum_role + 1):
            mask = self.exterior & (self.roles == role)
            conditioned = (morphology[:, :, 1] > 0) & (self.roles == role)
            active = mask.any(1) & conditioned.any(1)
            if bool(active.any()):
                target = morphology[:, :, 1].masked_fill(~conditioned, 0).max(1).values - 1
                self.junctions.append((mask, active, target, mask.sum(1).clamp(min=1)))

    @staticmethod
    def ce(logits, targets, mask, full_mask, *, groups=None, full_groups=None, classes=None):
        index = mask.nonzero(as_tuple=True)
        losses = functional.cross_entropy(logits[index], targets[index], reduction="none")
        if groups is None:
            return losses.sum() / full_mask.sum().clamp(min=1)
        count = classes if classes is not None else int(full_groups.max()) + 1
        sums = logits.new_zeros(count).scatter_add(0, groups[index], losses)
        counts = torch.bincount(full_groups[full_mask], minlength=count)
        present = counts > 0
        return ((sums / counts.clamp(min=1)) * present).sum() / present.sum().clamp(min=1)

    def loss(self, predictions, topology, selection, config):
        full, clean = self.clean, select(self.clean, selection)
        settings = validate_objective(config["objective"])
        weights = config["semantic_weights"]
        by_role = settings["chemistry_loss_balancing"] == "equal_present_role_mass"

        def chemistry(output):
            values = {}
            for name, field, mask, roles in (
                ("node_ce", "nodes", "atom_variable_mask", self.roles),
                ("backbone_bond_ce", "parent_bonds", "parent_bond_variable_mask", self.roles),
                (
                    "closure_bond_ce",
                    "closure_bonds",
                    "closure_bond_variable_mask",
                    self.closure_roles,
                ),
            ):
                values[name] = self.ce(
                    output[field],
                    clean[field],
                    clean[mask],
                    full[mask],
                    groups=roles[selection] if by_role else None,
                    full_groups=roles if by_role else None,
                )
            return values

        metrics = chemistry(predictions)
        candidates = _endpoint_candidate_mask(clean["node_mask"], clean["closure_left"].shape[1])
        for name, field, mask, valid in (
            (
                "parent_pointer_ce",
                "parents",
                "parent_variable_mask",
                _parent_candidate_mask(clean["node_mask"]),
            ),
            ("closure_left_ce", "closure_left", "closure_endpoint_variable_mask", candidates),
            ("closure_right_ce", "closure_right", "closure_endpoint_variable_mask", candidates),
        ):
            metrics[name] = self.ce(
                predictions[field].masked_fill(~valid, -1e9), clean[field], clean[mask], full[mask]
            )
        group_weight = settings.get("parent_group_loss_weight", 0.0)
        if group_weight > 0:
            categorical = metrics["parent_pointer_ce"]
            group = _parent_group_cross_entropy(
                predictions["parents"], clean, full_mask=full["parent_variable_mask"]
            )
            metrics["parent_categorical_ce"] = categorical
            metrics["parent_group_ce"] = group
            metrics["parent_pointer_ce"] = (categorical + group_weight * group) / (1 + group_weight)
        # Match the reference's addition order.
        base = sum(
            (
                metrics[k]
                for k in (
                    "node_ce",
                    "parent_pointer_ce",
                    "backbone_bond_ce",
                    "closure_left_ce",
                    "closure_right_ce",
                    "closure_bond_ce",
                )
            ),
            start=predictions["nodes"].new_zeros(()),
        )
        metrics["total"] = base
        for name, field in (
            ("role_consistency_ce", "role_states"),
            ("core_consistency_ce", "core_position_states"),
        ):
            metrics[name] = self.ce(
                predictions[field],
                clean[field],
                clean["node_mask"],
                full["node_mask"],
                groups=clean[field],
                full_groups=full[field],
                classes=predictions[field].shape[-1],
            )
        offspring, exterior = self.offspring[selection], self.exterior[selection]
        metrics["offspring_ce"] = self.ce(
            predictions["offspring"],
            offspring,
            exterior,
            self.exterior,
            groups=offspring,
            full_groups=self.offspring,
            classes=predictions["offspring"].shape[-1],
        )
        repeat = base.new_zeros(())
        for field, groups_name, metric in (
            ("nodes", "repeat_atom_groups", "repeat_consistency_pairs"),
            ("parent_bonds", "repeat_bond_groups", "repeat_bond_pairs"),
        ):
            groups = clean[groups_name]
            pairs = torch.triu(
                (groups[:, :, None] == groups[:, None, :]) & (groups[:, :, None] > 0), diagonal=1
            )
            batch, left, right = pairs.nonzero(as_tuple=True)
            probability = predictions[field].softmax(-1)
            difference = (probability[batch, left] - probability[batch, right]).square().mean(-1)
            repeat = repeat + difference.sum() / self.repeat_counts[groups_name].clamp(min=1)
            metrics[metric] = pairs.sum()
        metrics["repeat_consistency_mse"] = repeat
        logits = predictions["offspring"]
        states = torch.arange(logits.shape[-1], dtype=logits.dtype, device=logits.device)
        expected = torch.einsum("bnc,c->bn", logits.softmax(-1), (states - 1).clamp(min=0))
        junction = base.new_zeros(())
        comparisons = base.new_zeros((), dtype=torch.int64)
        for mask, active, target, nodes in self.junctions:
            local = active[selection]
            predicted = (expected * mask[selection]).sum(1)
            n = nodes[selection][local].to(logits.dtype)
            numerator = functional.smooth_l1_loss(
                predicted[local] / n, target[selection][local].to(logits.dtype) / n, reduction="sum"
            )
            junction = junction + numerator / active.sum().clamp(min=1)
            comparisons = comparisons + local.sum()
        junction = junction / max(1, len(self.junctions))
        metrics["junction_budget_consistency"] = junction
        metrics["junction_budget_comparisons"] = comparisons
        topology_loss = sum(chemistry(topology).values(), start=base.new_zeros(()))
        metrics["topology_conditioned_chemistry_ce"] = topology_loss
        loss = (
            base
            + weights["role_weight"] * metrics["role_consistency_ce"]
            + weights["core_weight"] * metrics["core_consistency_ce"]
            + settings["offspring_weight"] * metrics["offspring_ce"]
            + settings["junction_consistency_weight"] * junction
            + settings["topology_conditioned_chemistry_weight"] * topology_loss
            + weights["repeat_consistency_weight"] * repeat
        )
        metrics["semantic_total"] = loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite partitioned objective")
        return loss, {key: value.detach() for key, value in metrics.items()}

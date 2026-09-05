"""Rank-weighted semantic guidance for the frozen Ugi decoder.

This is a bounded, graph-specific adaptation of the transition-ranking idea used by MOG-DFM.  It
does not reproduce that sequence model or introduce a learned property oracle.  Hard chemistry and
exact-assembly support are applied before this policy sees a candidate.  The policy only ranks the
remaining supported choices by Transformer score, distance from one identity-free semantic target,
and optional train-fold realism objectives, then mixes the ranked law with a declared uniform
component to preserve residual entropy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from forge.core.hashing import sha256_json
from forge.model.ugi_all_role_semantic_program import (
    UgiAllRoleSemanticTarget,
    UgiMeasuredJointAllRoleSemanticPrior,
    UgiTailPairSemanticTarget,
)
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.model.ugi_role_chemistry_prior import UgiRoleChemistryPrior


class UgiMogSemanticGuidanceError(ValueError):
    """A semantic-guidance request would weaken or underdefine the decoder contract."""


JOINT_REALISM_FEATURE_NAMES = (
    *(f"program.node_count.{role}" for role in ("amine", "aldehyde", "isocyanide")),
    *(f"program.junction_budget.{role}" for role in ("amine", "aldehyde", "isocyanide")),
    *(f"program.cycle_rank.{role}" for role in ("amine", "aldehyde", "isocyanide")),
    *(f"program.attachment_count.{role}" for role in ("amine", "aldehyde", "isocyanide")),
    "amine.heavy_atom_graph_diameter",
    "amine.carbon_skeleton_diameter",
    "amine.nitrogen_atoms",
    "amine.oxygen_atoms",
    "aldehyde.ester_short_side_carbons",
    "aldehyde.ester_long_side_carbons",
    "aldehyde.carbon_carbon_double_bonds",
    "aldehyde.carbon_carbon_triple_bonds",
    "isocyanide.carbon_carbon_double_bonds",
    "isocyanide.carbon_carbon_triple_bonds",
    "isocyanide.carbon_skeleton_diameter",
    "aldehyde.alkoxy_handle_side_carbons",
    "aldehyde.acyl_side_carbons",
)

WHOLE_HEAD_SCORE_MODES = frozenset(
    {
        "whole_head_support_tier",
        "whole_head_support_binary",
        "whole_head_support_distance",
    }
)


def _joint_realism_vector(
    program: UgiMorphologyProgram,
    target: UgiAllRoleSemanticTarget,
) -> np.ndarray:
    """Return the complete identity-free program/semantics vector used for guidance."""

    values = (
        *program.node_counts,
        *program.junction_budgets,
        *program.cycle_ranks,
        *program.attachment_counts,
        *target.amine.key,
        *target.tail_pair.key,
    )
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(JOINT_REALISM_FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise UgiMogSemanticGuidanceError("joint-realism feature vector changed")
    return vector


def replace_amine_semantics(
    target: UgiAllRoleSemanticTarget,
    amine: UgiAmineSemanticTarget,
) -> UgiAllRoleSemanticTarget:
    """Replace only the candidate head semantics in one complete target."""

    return UgiAllRoleSemanticTarget(amine=amine, tail_pair=target.tail_pair)


def replace_directional_ester_semantics(
    target: UgiAllRoleSemanticTarget,
    directional_pair: tuple[int, int],
) -> UgiAllRoleSemanticTarget:
    """Replace only the directional ester split while conserving all other tail semantics."""

    alkoxy, acyl = (int(value) for value in directional_pair)
    tail = target.tail_pair
    return UgiAllRoleSemanticTarget(
        amine=target.amine,
        tail_pair=UgiTailPairSemanticTarget(
            aldehyde_ester_short_side_carbons=min(alkoxy, acyl),
            aldehyde_ester_long_side_carbons=max(alkoxy, acyl),
            aldehyde_carbon_carbon_double_bonds=(tail.aldehyde_carbon_carbon_double_bonds),
            aldehyde_carbon_carbon_triple_bonds=(tail.aldehyde_carbon_carbon_triple_bonds),
            isocyanide_carbon_carbon_double_bonds=(tail.isocyanide_carbon_carbon_double_bonds),
            isocyanide_carbon_carbon_triple_bonds=(tail.isocyanide_carbon_carbon_triple_bonds),
            isocyanide_carbon_skeleton_diameter=(tail.isocyanide_carbon_skeleton_diameter),
            aldehyde_alkoxy_handle_side_carbons=alkoxy,
            aldehyde_acyl_side_carbons=acyl,
        ),
    )


def replace_tail_unsaturation_semantics(
    target: UgiAllRoleSemanticTarget,
    *,
    role: str,
    double_count: int,
    triple_count: int,
) -> UgiAllRoleSemanticTarget:
    """Replace one role's unsaturation counts while preserving all other semantics."""

    tail = target.tail_pair
    if role == "oxoester_aldehyde_body_tail":
        aldehyde_double, aldehyde_triple = int(double_count), int(triple_count)
        isocyanide_double = tail.isocyanide_carbon_carbon_double_bonds
        isocyanide_triple = tail.isocyanide_carbon_carbon_triple_bonds
    elif role == "isocyanide_tail":
        aldehyde_double = tail.aldehyde_carbon_carbon_double_bonds
        aldehyde_triple = tail.aldehyde_carbon_carbon_triple_bonds
        isocyanide_double, isocyanide_triple = int(double_count), int(triple_count)
    else:
        raise UgiMogSemanticGuidanceError(f"unsupported Ugi tail role: {role!r}")
    return UgiAllRoleSemanticTarget(
        amine=target.amine,
        tail_pair=UgiTailPairSemanticTarget(
            aldehyde_ester_short_side_carbons=tail.aldehyde_ester_short_side_carbons,
            aldehyde_ester_long_side_carbons=tail.aldehyde_ester_long_side_carbons,
            aldehyde_carbon_carbon_double_bonds=aldehyde_double,
            aldehyde_carbon_carbon_triple_bonds=aldehyde_triple,
            isocyanide_carbon_carbon_double_bonds=isocyanide_double,
            isocyanide_carbon_carbon_triple_bonds=isocyanide_triple,
            isocyanide_carbon_skeleton_diameter=tail.isocyanide_carbon_skeleton_diameter,
            aldehyde_alkoxy_handle_side_carbons=tail.aldehyde_alkoxy_handle_side_carbons,
            aldehyde_acyl_side_carbons=tail.aldehyde_acyl_side_carbons,
        ),
    )


@dataclass(frozen=True)
class UgiJointSemanticRealismScorer:
    """Train-fold kernel density over complete identity-free Ugi program semantics.

    The scorer is deliberately small and nonparametric.  It learns the correlations among the
    coarse program, head compactness/composition, directional ester geometry and tail
    unsaturation from the measured Ugi training fold.  It never stores a component identity or
    molecular graph and it never accesses calibration or held-out products.
    """

    support: np.ndarray = field(repr=False, compare=False)
    probabilities: np.ndarray = field(repr=False, compare=False)
    center: np.ndarray = field(repr=False, compare=False)
    scale: np.ndarray = field(repr=False, compare=False)
    bandwidth: float
    audit: Mapping[str, Any] = field(compare=False)

    @classmethod
    def from_prior(
        cls,
        prior: UgiMeasuredJointAllRoleSemanticPrior,
        *,
        bandwidth: float,
    ) -> UgiJointSemanticRealismScorer:
        """Bind the scorer to a frozen measured-train joint prior."""

        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise UgiMogSemanticGuidanceError("joint-realism bandwidth must be positive")
        if (
            prior.audit.get("training_fold") != "train"
            or "source-adjudicated measured Ugi train products only"
            not in str(prior.audit.get("reference", ""))
            or prior.audit.get("component_or_family_ids_in_sampled_target") is not False
            or prior.audit.get("stored_component_graphs_in_sampled_target") is not False
            or prior.audit.get("smiles_or_fragment_tokens_in_sampled_target") is not False
        ):
            raise UgiMogSemanticGuidanceError(
                "joint-realism prior is not a train-only identity-free measured reference"
            )
        if not prior.support or len(prior.support) != len(prior.probabilities):
            raise UgiMogSemanticGuidanceError("joint-realism prior support is malformed")
        support = np.asarray(
            [_joint_realism_vector(program, target) for program, target in prior.support],
            dtype=np.float64,
        )
        probabilities = np.asarray(prior.probabilities, dtype=np.float64)
        if (
            support.ndim != 2
            or support.shape[1] != len(JOINT_REALISM_FEATURE_NAMES)
            or np.any(probabilities <= 0)
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise UgiMogSemanticGuidanceError("joint-realism prior weights are invalid")
        center = probabilities @ support
        variance = probabilities @ np.square(support - center)
        scale = np.sqrt(variance)
        scale = np.where(scale > 1e-12, scale, 1.0)
        payload = [
            {
                "features": row.tolist(),
                "probability": float(probability),
            }
            for row, probability in zip(support, probabilities, strict=True)
        ]
        return cls(
            support=support,
            probabilities=probabilities,
            center=center,
            scale=scale,
            bandwidth=float(bandwidth),
            audit={
                "schema_version": "forge.ugi_joint_semantic_realism_reference.v1",
                "reference": "source-adjudicated measured Ugi train products only",
                "training_fold": "train",
                "feature_names": list(JOINT_REALISM_FEATURE_NAMES),
                "support_rows": len(support),
                "support_sha256": str(sha256_json(payload)),
                "source_prior_sha256": str(prior.audit["joint_support_sha256"]),
                "bandwidth": float(bandwidth),
                "component_identity_conditioning": False,
                "stored_component_graphs": False,
                "calibration_or_heldout_access": False,
                "score": (
                    "log probability under a weighted Gaussian kernel density after frozen "
                    "measured-train standardization"
                ),
            },
        )

    def score(
        self,
        program: UgiMorphologyProgram,
        target: UgiAllRoleSemanticTarget,
    ) -> float:
        """Return a stable log-density score; larger values are more train-like."""

        query = (_joint_realism_vector(program, target) - self.center) / self.scale
        reference = (self.support - self.center) / self.scale
        squared = np.square(reference - query).mean(axis=1)
        logits = np.log(self.probabilities) - squared / (2.0 * self.bandwidth**2)
        maximum = float(np.max(logits))
        value = maximum + float(np.log(np.exp(logits - maximum).sum()))
        if not np.isfinite(value):
            raise UgiMogSemanticGuidanceError("joint-realism score is non-finite")
        return value


@dataclass(frozen=True)
class UgiMogSemanticGuidancePolicy:
    """Bounded semantic bands and rank weights for supported decoder choices."""

    amine_heavy_atom_graph_diameter_tolerance: int
    amine_carbon_skeleton_diameter_tolerance: int
    aldehyde_ester_side_carbons_tolerance: int
    model_rank_weight: float
    semantic_rank_weight: float
    rank_temperature: float
    uniform_probability_mass: float
    amine_hydrogen_bond_donors_tolerance: int = 0
    amine_heavy_branch_atoms_tolerance: int = 0
    joint_realism_rank_weight: float = 0.0
    joint_realism_bandwidth: float = 1.0
    local_chemistry_rank_weight: float = 0.0
    # ``None`` preserves the original policy, where one local-chemistry weight ranks both atoms
    # and bonds.  An explicit zero keeps train-fold atom-context ranking while leaving bond and
    # unsaturation placement to the Transformer.  This separation is needed because those two
    # decisions had opposite effects in the blinded morphology diagnostic.
    local_chemistry_bond_rank_weight: float | None = None
    # Tail-bond placement and whole-head selection have different entropy requirements.  Keeping
    # this unset preserves the historical shared mixture.  An explicit value changes only
    # local bond decisions; atom and whole-head choices continue to use
    # ``uniform_probability_mass``.
    local_chemistry_bond_uniform_probability_mass: float | None = None
    # Preserve fine position information for tail bond orders while retaining the deliberately
    # pooled atom-context depth.  ``None`` reproduces the historical shared depth bucket.
    local_chemistry_bond_maximum_depth_bucket: int | None = None
    # Core depth is the historical coordinate.  ``terminal_offset`` transfers measured terminal
    # and internal unsaturation placement across hydrophobic chains of different total length.
    local_chemistry_bond_position_basis: str = "core_depth"
    # Tail-unsaturation placement is one global choice.  When enabled, its local score is computed
    # from the selected C=C/C#C edges rather than taking the minimum over the many invariant single
    # bonds, which would mathematically erase the position-specific signal.
    local_chemistry_unsaturation_position_only: bool = False
    # Optionally require every selected C=C/C#C edge to have measured support at or above this
    # novelty-neutral specificity tier.  Tier four is the exact role/depth/ring/symbol context;
    # lower backoff tiers deliberately ignore one or more of those coordinates.  This is an
    # eligibility rule over local bond contexts, not a component lookup or completed-product filter.
    local_chemistry_unsaturation_minimum_support_tier: int | None = None
    # Permit a bounded change in C=C/C#C count only when that complete count pair occurs in the
    # measured training fold for the same precursor role.  Zero preserves exact target counts.
    tail_unsaturation_count_tolerance: int = 0
    # Historical slack treats every nearby count pair and every positional assignment as one flat
    # candidate set.  ``grouped_downward`` first chooses among count pairs using model evidence;
    # ``frequency_downward`` instead uses unique measured train-fold component frequencies.  Both
    # downward strategies permit only removal of requested unsaturation and never add an
    # unsaturation to a saturated program.  ``frequency_resampled`` draws the count state directly
    # from those measured component frequencies; ``smoothed_frequency_resampled`` applies the
    # prior's declared symmetric pseudocount first.  The Transformer still chooses bond positions.
    tail_unsaturation_count_strategy: str = "flat_symmetric"
    # Coordinatewise bond support cannot express that the measured two-alkene tail uses one joint
    # pair of terminal offsets.  This optional role-level law samples one complete measured pattern
    # (C=C offsets and C#C offsets together), then lets the Transformer choose only among topology
    # assignments realizing that anonymous pattern.
    tail_unsaturation_position_strategy: str = "independent_ranked"
    # Optional explicit pseudocount for the smoothed frequency law.  ``None`` uses the local
    # chemistry prior's declared smoothing value.  Exposing it makes sparse-count calibration
    # auditable without altering atom- or bond-context smoothing.
    tail_unsaturation_frequency_pseudocount: float | None = None
    # ``None`` preserves unrestricted atom ranking.  Otherwise, mix the guided law back toward
    # the Transformer/semantic law until the per-decision total-variation distance is no greater
    # than this radius.  This bounds steering without filtering or changing hard support.
    local_chemistry_atom_total_variation_radius: float | None = None
    # A whole-head score is applied once to the finite set of complete feasible atom assignments.
    # This radius therefore bounds the treatment law relative to the Transformer/semantic law at
    # the component-trajectory level, rather than compounding a local bound over many atom draws.
    whole_head_total_variation_radius: float | None = None
    # A total-variation bound alone can move a small amount of mass onto a few common heads and
    # sharply reduce effective component count.  When configured, flatten the guided complete-head
    # law just enough to retain this fraction of the baseline candidate-law effective count before
    # applying the total-variation bound.  This is a one-shot distribution constraint, not output
    # filtering or component lookup.
    whole_head_candidate_effective_count_retention: float | None = None
    # Candidate serialization is not chemical identity: symmetric assignments can decode to the
    # same head arrangement.  This optional guard applies the same entropy constraint after
    # aggregating candidates by an identity-free arrangement signature supplied by the decoder.
    whole_head_group_effective_count_retention: float | None = None
    # Legacy whole-head policies hard-mask amine bond orders without measured support.  Set false
    # only for diagnostics or a joint atom-and-bond trajectory law that controls this shift softly.
    whole_head_hard_bond_support: bool = True
    # Frequency scores reproduce the frozen historical intervention.  Support tiers use only the
    # specificity of measured-training evidence and therefore do not reward common modes more than
    # rare modes at the same supported context.  Whole-head support scores one completed amine
    # arrangement rather than accumulating rewards over sequential atom decisions.  Binary
    # whole-head support additionally treats every measured-supported arrangement equally, instead
    # of preferring the small exact-signature set over coarse or basic supported arrangements.
    local_chemistry_score_mode: str = "frequency_log_probability"
    # Score each complete feasible rooted head topology before atom and bond identities are drawn.
    # The score uses only identity-free topology signatures from measured Ugi training components.
    # It is intentionally separable from the atom-assignment score: novelty-neutral coordinate-local
    # support can therefore preserve chemical diversity while topology is ranked as one object.
    whole_head_topology_support: bool = False
    # Historical topology support collapses exact and coarse measured matches to one binary score.
    # ``tier`` preserves the identity-free exact-versus-coarse evidence distinction, allowing the
    # decoder to prefer the measured rooted shape without selecting a stored component graph.
    whole_head_topology_score_mode: str = "binary"
    # ``None`` preserves the historical coupling to ``local_chemistry_rank_weight``.  An explicit
    # positive value lets topology use the measured reference while atom and bond ranking remain
    # disabled with a zero local-chemistry weight.
    whole_head_topology_rank_weight: float | None = None
    # ``ranked`` preserves the historical MOG terminal decoder.  ``model_argmax`` confines the
    # stochastic intervention to complete topology selection and restores the frozen Transformer's
    # deterministic atom- and bond-identity readout.  This makes topology support independently
    # testable without post-hoc filtering, repair, retry, or component lookup.
    terminal_chemistry_selection_mode: str = "ranked"
    # Optionally keep stochastic atom/head selection while making every terminal bond-order
    # decision by Transformer argmax.  This isolates the visually problematic unsaturation law
    # without removing the atom and topology entropy that supports component diversity.
    terminal_bond_selection_mode: str = "ranked"
    joint_realism_scorer: UgiJointSemanticRealismScorer | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    local_chemistry_prior: UgiRoleChemistryPrior | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        tolerances = (
            self.amine_heavy_atom_graph_diameter_tolerance,
            self.amine_carbon_skeleton_diameter_tolerance,
            self.aldehyde_ester_side_carbons_tolerance,
            self.amine_hydrogen_bond_donors_tolerance,
            self.amine_heavy_branch_atoms_tolerance,
        )
        weights = (
            self.model_rank_weight,
            self.semantic_rank_weight,
            self.joint_realism_rank_weight,
            self.local_chemistry_rank_weight,
            *(
                ()
                if self.local_chemistry_bond_rank_weight is None
                else (self.local_chemistry_bond_rank_weight,)
            ),
        )
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in tolerances)
            or any(value < 0 or value > 2 for value in tolerances)
            or not all(np.isfinite(value) and value >= 0 for value in weights)
            or sum(weights) <= 0
            or not np.isfinite(self.rank_temperature)
            or self.rank_temperature <= 0
            or not np.isfinite(self.uniform_probability_mass)
            or not 0 < self.uniform_probability_mass < 1
            or (
                self.local_chemistry_bond_uniform_probability_mass is not None
                and (
                    not np.isfinite(self.local_chemistry_bond_uniform_probability_mass)
                    or not 0 < self.local_chemistry_bond_uniform_probability_mass < 1
                    or not self.uses_local_chemistry_bonds
                )
            )
            or not np.isfinite(self.joint_realism_bandwidth)
            or self.joint_realism_bandwidth <= 0
            or (self.joint_realism_scorer is not None and self.joint_realism_rank_weight <= 0)
            or (
                self.local_chemistry_bond_rank_weight is not None
                and self.local_chemistry_rank_weight <= 0
            )
            or (
                self.local_chemistry_bond_maximum_depth_bucket is not None
                and (
                    isinstance(self.local_chemistry_bond_maximum_depth_bucket, bool)
                    or not isinstance(self.local_chemistry_bond_maximum_depth_bucket, int)
                    or not 2 <= self.local_chemistry_bond_maximum_depth_bucket <= 194
                    or not self.uses_local_chemistry_bonds
                )
            )
            or not isinstance(self.local_chemistry_unsaturation_position_only, bool)
            or self.local_chemistry_bond_position_basis
            not in {"core_depth", "terminal_offset"}
            or (
                self.local_chemistry_bond_position_basis != "core_depth"
                and (
                    not self.local_chemistry_unsaturation_position_only
                    or not self.uses_local_chemistry_bonds
                )
            )
            or (
                self.local_chemistry_unsaturation_position_only
                and (
                    self.local_chemistry_bond_maximum_depth_bucket is None
                    or not self.uses_local_chemistry_bonds
                )
            )
            or (
                self.local_chemistry_unsaturation_minimum_support_tier is not None
                and (
                    isinstance(self.local_chemistry_unsaturation_minimum_support_tier, bool)
                    or not isinstance(self.local_chemistry_unsaturation_minimum_support_tier, int)
                    or not 1 <= self.local_chemistry_unsaturation_minimum_support_tier <= 4
                    or not self.local_chemistry_unsaturation_position_only
                )
            )
            or isinstance(self.tail_unsaturation_count_tolerance, bool)
            or not isinstance(self.tail_unsaturation_count_tolerance, int)
            or not 0 <= self.tail_unsaturation_count_tolerance <= 2
            or (
                self.tail_unsaturation_count_tolerance > 0
                and not self.local_chemistry_unsaturation_position_only
            )
            or self.tail_unsaturation_count_strategy
            not in {
                "flat_symmetric",
                "flat_downward",
                "grouped_downward",
                "frequency_downward",
                "frequency_resampled",
                "smoothed_frequency_resampled",
            }
            or self.tail_unsaturation_position_strategy
            not in {
                "independent_ranked",
                "measured_joint_terminal_pattern",
                "measured_joint_terminal_pattern_ranked",
            }
            or (
                self.tail_unsaturation_position_strategy
                in {
                    "measured_joint_terminal_pattern",
                    "measured_joint_terminal_pattern_ranked",
                }
                and (
                    self.local_chemistry_bond_position_basis != "terminal_offset"
                    or not self.local_chemistry_unsaturation_position_only
                    or self.tail_unsaturation_count_strategy
                    not in {"frequency_resampled", "smoothed_frequency_resampled"}
                )
            )
            or (
                self.tail_unsaturation_count_strategy
                not in {
                    "flat_symmetric",
                    "frequency_resampled",
                    "smoothed_frequency_resampled",
                }
                and self.tail_unsaturation_count_tolerance == 0
            )
            or (
                self.tail_unsaturation_frequency_pseudocount is not None
                and (
                    not np.isfinite(self.tail_unsaturation_frequency_pseudocount)
                    or self.tail_unsaturation_frequency_pseudocount <= 0
                    or self.tail_unsaturation_count_strategy
                    != "smoothed_frequency_resampled"
                )
            )
            or (
                self.local_chemistry_atom_total_variation_radius is not None
                and (
                    not np.isfinite(self.local_chemistry_atom_total_variation_radius)
                    or not 0 < self.local_chemistry_atom_total_variation_radius <= 1
                    or self.local_chemistry_rank_weight <= 0
                    or self.local_chemistry_score_mode in WHOLE_HEAD_SCORE_MODES
                )
            )
            or (
                self.whole_head_total_variation_radius is not None
                and (
                    not np.isfinite(self.whole_head_total_variation_radius)
                    or not 0 < self.whole_head_total_variation_radius <= 1
                    or self.local_chemistry_score_mode not in WHOLE_HEAD_SCORE_MODES
                )
            )
            or (
                self.whole_head_candidate_effective_count_retention is not None
                and (
                    not np.isfinite(self.whole_head_candidate_effective_count_retention)
                    or not 0 < self.whole_head_candidate_effective_count_retention <= 1
                    or self.local_chemistry_score_mode not in WHOLE_HEAD_SCORE_MODES
                )
            )
            or (
                self.whole_head_group_effective_count_retention is not None
                and (
                    not np.isfinite(self.whole_head_group_effective_count_retention)
                    or not 0 < self.whole_head_group_effective_count_retention <= 1
                    or self.local_chemistry_score_mode not in WHOLE_HEAD_SCORE_MODES
                )
            )
            or not isinstance(self.whole_head_hard_bond_support, bool)
            or (
                self.whole_head_hard_bond_support is False
                and self.local_chemistry_score_mode not in WHOLE_HEAD_SCORE_MODES
            )
            or not isinstance(self.whole_head_topology_support, bool)
            or self.whole_head_topology_score_mode not in {"binary", "tier"}
            or (
                self.whole_head_topology_score_mode != "binary"
                and not self.whole_head_topology_support
            )
            or (
                self.whole_head_topology_rank_weight is not None
                and (
                    not np.isfinite(self.whole_head_topology_rank_weight)
                    or self.whole_head_topology_rank_weight <= 0
                    or not self.whole_head_topology_support
                )
            )
            or (
                self.whole_head_topology_support
                and self.effective_whole_head_topology_rank_weight <= 0
            )
            or self.terminal_chemistry_selection_mode not in {"ranked", "model_argmax"}
            or self.terminal_bond_selection_mode not in {"ranked", "model_argmax"}
            or (
                self.terminal_chemistry_selection_mode == "model_argmax"
                and self.terminal_bond_selection_mode != "ranked"
            )
            or (
                self.terminal_chemistry_selection_mode == "model_argmax"
                and not self.whole_head_topology_support
            )
            or (
                self.whole_head_topology_support
                and self.local_chemistry_score_mode
                not in ({"support_tier"} | WHOLE_HEAD_SCORE_MODES)
            )
            or (self.local_chemistry_prior is not None and not self.uses_local_reference)
            or self.local_chemistry_score_mode
            not in {
                "frequency_log_probability",
                "support_tier",
                "whole_head_support_tier",
                "whole_head_support_binary",
                "whole_head_support_distance",
            }
            or (
                self.local_chemistry_score_mode != "frequency_log_probability"
                and not self.uses_local_reference
            )
        ):
            raise UgiMogSemanticGuidanceError("invalid MOG-style semantic-guidance policy")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> UgiMogSemanticGuidancePolicy:
        legacy = {
            "amine_heavy_atom_graph_diameter_tolerance",
            "amine_carbon_skeleton_diameter_tolerance",
            "aldehyde_ester_side_carbons_tolerance",
            "model_rank_weight",
            "semantic_rank_weight",
            "rank_temperature",
            "uniform_probability_mass",
        }
        joint = legacy | {"joint_realism_rank_weight", "joint_realism_bandwidth"}
        substitution_slack = legacy | {
            "amine_hydrogen_bond_donors_tolerance",
            "amine_heavy_branch_atoms_tolerance",
        }
        substitution_slack_joint = joint | {
            "amine_hydrogen_bond_donors_tolerance",
            "amine_heavy_branch_atoms_tolerance",
        }
        local = legacy | {"local_chemistry_rank_weight"}
        joint_local = joint | {"local_chemistry_rank_weight"}
        substitution_slack_local = substitution_slack | {"local_chemistry_rank_weight"}
        substitution_slack_joint_local = substitution_slack_joint | {"local_chemistry_rank_weight"}
        accepted = {
            frozenset(legacy),
            frozenset(joint),
            frozenset(substitution_slack),
            frozenset(substitution_slack_joint),
            frozenset(local),
            frozenset(joint_local),
            frozenset(substitution_slack_local),
            frozenset(substitution_slack_joint_local),
        }
        accepted.update(
            fields | {"local_chemistry_bond_rank_weight"}
            for fields in tuple(accepted)
            if "local_chemistry_rank_weight" in fields
        )
        accepted.update(
            fields | {"local_chemistry_bond_uniform_probability_mass"}
            for fields in tuple(accepted)
            if "local_chemistry_rank_weight" in fields
        )
        accepted.update(
            fields | {"local_chemistry_bond_maximum_depth_bucket"}
            for fields in tuple(accepted)
            if "local_chemistry_rank_weight" in fields
        )
        accepted.update(
            fields | {"local_chemistry_unsaturation_position_only"}
            for fields in tuple(accepted)
            if "local_chemistry_bond_maximum_depth_bucket" in fields
        )
        accepted.update(
            fields | {"local_chemistry_bond_position_basis"}
            for fields in tuple(accepted)
            if "local_chemistry_unsaturation_position_only" in fields
        )
        accepted.update(
            fields | {"local_chemistry_unsaturation_minimum_support_tier"}
            for fields in tuple(accepted)
            if "local_chemistry_unsaturation_position_only" in fields
        )
        accepted.update(
            fields | {"tail_unsaturation_count_tolerance"}
            for fields in tuple(accepted)
            if "local_chemistry_unsaturation_position_only" in fields
        )
        accepted.update(
            fields | {"tail_unsaturation_count_strategy"}
            for fields in tuple(accepted)
            if "tail_unsaturation_count_tolerance" in fields
        )
        accepted.update(
            fields | {"tail_unsaturation_frequency_pseudocount"}
            for fields in tuple(accepted)
            if "tail_unsaturation_count_strategy" in fields
        )
        accepted.update(
            fields | {"tail_unsaturation_position_strategy"}
            for fields in tuple(accepted)
            if "tail_unsaturation_count_strategy" in fields
        )
        accepted.update(
            fields | {"local_chemistry_atom_total_variation_radius"}
            for fields in tuple(accepted)
            if "local_chemistry_rank_weight" in fields
        )
        accepted.update(
            fields | {"whole_head_total_variation_radius"}
            for fields in tuple(accepted)
            if "local_chemistry_rank_weight" in fields
        )
        accepted.update(
            fields | {"whole_head_candidate_effective_count_retention"}
            for fields in tuple(accepted)
            if "local_chemistry_rank_weight" in fields
        )
        accepted.update(
            fields | {"whole_head_group_effective_count_retention"}
            for fields in tuple(accepted)
            if "local_chemistry_rank_weight" in fields
        )
        accepted.update(
            fields | {"whole_head_hard_bond_support"}
            for fields in tuple(accepted)
            if "local_chemistry_rank_weight" in fields
        )
        accepted.update(
            fields | {"local_chemistry_score_mode"}
            for fields in tuple(accepted)
            if "local_chemistry_rank_weight" in fields
        )
        accepted.update(
            fields | {"whole_head_topology_support"}
            for fields in tuple(accepted)
            if "local_chemistry_score_mode" in fields
        )
        accepted.update(
            fields | {"whole_head_topology_rank_weight"}
            for fields in tuple(accepted)
            if "whole_head_topology_support" in fields
        )
        accepted.update(
            fields | {"whole_head_topology_score_mode"}
            for fields in tuple(accepted)
            if "whole_head_topology_support" in fields
        )
        accepted.update(
            fields | {"terminal_chemistry_selection_mode"}
            for fields in tuple(accepted)
            if "whole_head_topology_support" in fields
        )
        accepted.update(
            fields | {"terminal_bond_selection_mode"} for fields in tuple(accepted)
        )
        if set(value) not in accepted:
            raise UgiMogSemanticGuidanceError("semantic-guidance policy fields changed")
        if set(value) in {
            frozenset(substitution_slack),
            frozenset(substitution_slack_joint),
            frozenset(substitution_slack_local),
            frozenset(substitution_slack_joint_local),
        } and (
            int(value["amine_hydrogen_bond_donors_tolerance"])
            + int(value["amine_heavy_branch_atoms_tolerance"])
            <= 0
        ):
            raise UgiMogSemanticGuidanceError(
                "substitution-semantic slack must enable at least one tolerance"
            )
        return cls(
            amine_heavy_atom_graph_diameter_tolerance=int(
                value["amine_heavy_atom_graph_diameter_tolerance"]
            ),
            amine_carbon_skeleton_diameter_tolerance=int(
                value["amine_carbon_skeleton_diameter_tolerance"]
            ),
            aldehyde_ester_side_carbons_tolerance=int(
                value["aldehyde_ester_side_carbons_tolerance"]
            ),
            model_rank_weight=float(value["model_rank_weight"]),
            semantic_rank_weight=float(value["semantic_rank_weight"]),
            rank_temperature=float(value["rank_temperature"]),
            uniform_probability_mass=float(value["uniform_probability_mass"]),
            amine_hydrogen_bond_donors_tolerance=int(
                value.get("amine_hydrogen_bond_donors_tolerance", 0)
            ),
            amine_heavy_branch_atoms_tolerance=int(
                value.get("amine_heavy_branch_atoms_tolerance", 0)
            ),
            joint_realism_rank_weight=float(value.get("joint_realism_rank_weight", 0.0)),
            joint_realism_bandwidth=float(value.get("joint_realism_bandwidth", 1.0)),
            local_chemistry_rank_weight=float(value.get("local_chemistry_rank_weight", 0.0)),
            local_chemistry_bond_rank_weight=(
                None
                if "local_chemistry_bond_rank_weight" not in value
                else float(value["local_chemistry_bond_rank_weight"])
            ),
            local_chemistry_bond_uniform_probability_mass=(
                None
                if "local_chemistry_bond_uniform_probability_mass" not in value
                else float(value["local_chemistry_bond_uniform_probability_mass"])
            ),
            local_chemistry_bond_maximum_depth_bucket=(
                None
                if "local_chemistry_bond_maximum_depth_bucket" not in value
                else int(value["local_chemistry_bond_maximum_depth_bucket"])
            ),
            local_chemistry_bond_position_basis=str(
                value.get("local_chemistry_bond_position_basis", "core_depth")
            ),
            local_chemistry_unsaturation_position_only=bool(
                value.get("local_chemistry_unsaturation_position_only", False)
            ),
            local_chemistry_unsaturation_minimum_support_tier=(
                None
                if "local_chemistry_unsaturation_minimum_support_tier" not in value
                else int(value["local_chemistry_unsaturation_minimum_support_tier"])
            ),
            tail_unsaturation_count_tolerance=int(
                value.get("tail_unsaturation_count_tolerance", 0)
            ),
            tail_unsaturation_count_strategy=str(
                value.get("tail_unsaturation_count_strategy", "flat_symmetric")
            ),
            tail_unsaturation_position_strategy=str(
                value.get("tail_unsaturation_position_strategy", "independent_ranked")
            ),
            tail_unsaturation_frequency_pseudocount=(
                None
                if "tail_unsaturation_frequency_pseudocount" not in value
                else float(value["tail_unsaturation_frequency_pseudocount"])
            ),
            local_chemistry_atom_total_variation_radius=(
                None
                if "local_chemistry_atom_total_variation_radius" not in value
                else float(value["local_chemistry_atom_total_variation_radius"])
            ),
            whole_head_total_variation_radius=(
                None
                if "whole_head_total_variation_radius" not in value
                else float(value["whole_head_total_variation_radius"])
            ),
            whole_head_candidate_effective_count_retention=(
                None
                if "whole_head_candidate_effective_count_retention" not in value
                else float(value["whole_head_candidate_effective_count_retention"])
            ),
            whole_head_group_effective_count_retention=(
                None
                if "whole_head_group_effective_count_retention" not in value
                else float(value["whole_head_group_effective_count_retention"])
            ),
            whole_head_hard_bond_support=value.get("whole_head_hard_bond_support", True),
            local_chemistry_score_mode=str(
                value.get("local_chemistry_score_mode", "frequency_log_probability")
            ),
            whole_head_topology_support=value.get("whole_head_topology_support", False),
            whole_head_topology_score_mode=str(
                value.get("whole_head_topology_score_mode", "binary")
            ),
            whole_head_topology_rank_weight=(
                None
                if "whole_head_topology_rank_weight" not in value
                else float(value["whole_head_topology_rank_weight"])
            ),
            terminal_chemistry_selection_mode=str(
                value.get("terminal_chemistry_selection_mode", "ranked")
            ),
            terminal_bond_selection_mode=str(
                value.get("terminal_bond_selection_mode", "ranked")
            ),
        )

    def to_mapping(self) -> dict[str, int | float | str | bool]:
        output = {
            "amine_heavy_atom_graph_diameter_tolerance": (
                self.amine_heavy_atom_graph_diameter_tolerance
            ),
            "amine_carbon_skeleton_diameter_tolerance": (
                self.amine_carbon_skeleton_diameter_tolerance
            ),
            "aldehyde_ester_side_carbons_tolerance": (self.aldehyde_ester_side_carbons_tolerance),
            "model_rank_weight": self.model_rank_weight,
            "semantic_rank_weight": self.semantic_rank_weight,
            "rank_temperature": self.rank_temperature,
            "uniform_probability_mass": self.uniform_probability_mass,
        }
        if self.joint_realism_rank_weight > 0:
            output.update(
                {
                    "joint_realism_rank_weight": self.joint_realism_rank_weight,
                    "joint_realism_bandwidth": self.joint_realism_bandwidth,
                }
            )
        if self.local_chemistry_rank_weight > 0 or self.whole_head_topology_support:
            output["local_chemistry_rank_weight"] = self.local_chemistry_rank_weight
        if self.local_chemistry_bond_rank_weight is not None:
            output["local_chemistry_bond_rank_weight"] = self.local_chemistry_bond_rank_weight
        if self.local_chemistry_bond_uniform_probability_mass is not None:
            output["local_chemistry_bond_uniform_probability_mass"] = (
                self.local_chemistry_bond_uniform_probability_mass
            )
        if self.local_chemistry_bond_maximum_depth_bucket is not None:
            output["local_chemistry_bond_maximum_depth_bucket"] = (
                self.local_chemistry_bond_maximum_depth_bucket
            )
        if self.local_chemistry_bond_position_basis != "core_depth":
            output["local_chemistry_bond_position_basis"] = (
                self.local_chemistry_bond_position_basis
            )
        if self.local_chemistry_unsaturation_position_only:
            output["local_chemistry_unsaturation_position_only"] = True
        if self.local_chemistry_unsaturation_minimum_support_tier is not None:
            output["local_chemistry_unsaturation_minimum_support_tier"] = (
                self.local_chemistry_unsaturation_minimum_support_tier
            )
        if (
            self.tail_unsaturation_count_tolerance > 0
            or self.tail_unsaturation_count_strategy != "flat_symmetric"
        ):
            output["tail_unsaturation_count_tolerance"] = self.tail_unsaturation_count_tolerance
        if self.tail_unsaturation_count_strategy != "flat_symmetric":
            output["tail_unsaturation_count_strategy"] = self.tail_unsaturation_count_strategy
        if self.tail_unsaturation_position_strategy != "independent_ranked":
            output["tail_unsaturation_position_strategy"] = (
                self.tail_unsaturation_position_strategy
            )
        if self.tail_unsaturation_frequency_pseudocount is not None:
            output["tail_unsaturation_frequency_pseudocount"] = (
                self.tail_unsaturation_frequency_pseudocount
            )
        if self.local_chemistry_atom_total_variation_radius is not None:
            output["local_chemistry_atom_total_variation_radius"] = (
                self.local_chemistry_atom_total_variation_radius
            )
        if self.whole_head_total_variation_radius is not None:
            output["whole_head_total_variation_radius"] = self.whole_head_total_variation_radius
        if self.whole_head_candidate_effective_count_retention is not None:
            output["whole_head_candidate_effective_count_retention"] = (
                self.whole_head_candidate_effective_count_retention
            )
        if self.whole_head_group_effective_count_retention is not None:
            output["whole_head_group_effective_count_retention"] = (
                self.whole_head_group_effective_count_retention
            )
        if not self.whole_head_hard_bond_support:
            output["whole_head_hard_bond_support"] = False
        if self.local_chemistry_score_mode != "frequency_log_probability":
            output["local_chemistry_score_mode"] = self.local_chemistry_score_mode
        if self.whole_head_topology_support:
            output["whole_head_topology_support"] = True
        if self.whole_head_topology_score_mode != "binary":
            output["whole_head_topology_score_mode"] = self.whole_head_topology_score_mode
        if self.whole_head_topology_rank_weight is not None:
            output["whole_head_topology_rank_weight"] = self.whole_head_topology_rank_weight
        if self.terminal_chemistry_selection_mode != "ranked":
            output["terminal_chemistry_selection_mode"] = (
                self.terminal_chemistry_selection_mode
            )
        if self.terminal_bond_selection_mode != "ranked":
            output["terminal_bond_selection_mode"] = self.terminal_bond_selection_mode
        if (
            self.amine_hydrogen_bond_donors_tolerance > 0
            or self.amine_heavy_branch_atoms_tolerance > 0
        ):
            output.update(
                {
                    "amine_hydrogen_bond_donors_tolerance": (
                        self.amine_hydrogen_bond_donors_tolerance
                    ),
                    "amine_heavy_branch_atoms_tolerance": (self.amine_heavy_branch_atoms_tolerance),
                }
            )
        return output

    @property
    def uses_joint_realism(self) -> bool:
        return self.joint_realism_rank_weight > 0

    @property
    def uses_local_chemistry(self) -> bool:
        return self.uses_local_chemistry_atoms or self.uses_local_chemistry_bonds

    @property
    def uses_ranked_terminal_chemistry(self) -> bool:
        return self.terminal_chemistry_selection_mode == "ranked"

    @property
    def uses_ranked_terminal_bonds(self) -> bool:
        return (
            self.uses_ranked_terminal_chemistry
            and self.terminal_bond_selection_mode == "ranked"
        )

    @property
    def uses_local_reference(self) -> bool:
        """Whether any active intervention requires the measured train-fold local prior."""

        return self.uses_local_chemistry or self.whole_head_topology_support

    @property
    def effective_whole_head_topology_rank_weight(self) -> float:
        return (
            self.local_chemistry_rank_weight
            if self.whole_head_topology_rank_weight is None
            else self.whole_head_topology_rank_weight
        )

    @property
    def uses_local_chemistry_atoms(self) -> bool:
        return self.uses_ranked_terminal_chemistry and self.local_chemistry_rank_weight > 0

    @property
    def uses_coordinate_local_chemistry_atoms(self) -> bool:
        """Whether individual atom decisions, rather than one whole head, receive scores."""

        return self.uses_local_chemistry_atoms and (
            self.local_chemistry_score_mode not in WHOLE_HEAD_SCORE_MODES
        )

    @property
    def effective_local_chemistry_bond_rank_weight(self) -> float:
        return (
            self.local_chemistry_rank_weight
            if self.local_chemistry_bond_rank_weight is None
            else self.local_chemistry_bond_rank_weight
        )

    @property
    def uses_local_chemistry_bonds(self) -> bool:
        return (
            self.uses_ranked_terminal_bonds
            and self.effective_local_chemistry_bond_rank_weight > 0
        )

    def bind_joint_realism(
        self,
        prior: UgiMeasuredJointAllRoleSemanticPrior,
    ) -> UgiMogSemanticGuidancePolicy:
        """Return a policy bound to one hash-audited measured-train reference."""

        if not self.uses_joint_realism:
            raise UgiMogSemanticGuidanceError(
                "joint-realism binding requires a positive rank weight"
            )
        return replace(
            self,
            joint_realism_scorer=UgiJointSemanticRealismScorer.from_prior(
                prior,
                bandwidth=self.joint_realism_bandwidth,
            ),
        )

    def joint_realism_scores(
        self,
        program: UgiMorphologyProgram,
        targets: Sequence[UgiAllRoleSemanticTarget],
    ) -> tuple[float, ...] | None:
        if not self.uses_joint_realism:
            return None
        if self.joint_realism_scorer is None:
            raise UgiMogSemanticGuidanceError(
                "joint-realism guidance was requested without a bound train reference"
            )
        return tuple(self.joint_realism_scorer.score(program, target) for target in targets)

    def joint_realism_audit(self) -> Mapping[str, Any] | None:
        return None if self.joint_realism_scorer is None else self.joint_realism_scorer.audit

    def bind_local_chemistry(
        self,
        prior: UgiRoleChemistryPrior,
    ) -> UgiMogSemanticGuidancePolicy:
        """Bind the rank objective to one hash-audited measured-train local prior."""

        if not self.uses_local_reference:
            raise UgiMogSemanticGuidanceError(
                "local-chemistry binding requires an active measured-reference intervention"
            )
        audit = prior.to_mapping()
        if (
            audit.get("training_reference")
            != "unique source-adjudicated role components from the measured Ugi train fold"
            or audit.get("component_identity_conditioning") is not False
            or audit.get("component_graph_conditioning") is not False
            or audit.get("fragment_vocabulary_conditioning") is not False
            or audit.get("hard_support_changed") is not False
            or (
                self.local_chemistry_bond_maximum_depth_bucket is not None
                and audit.get("maximum_bond_depth_bucket")
                != self.local_chemistry_bond_maximum_depth_bucket
            )
        ):
            raise UgiMogSemanticGuidanceError(
                "local-chemistry prior is not a train-only identity-free reference"
            )
        return replace(self, local_chemistry_prior=prior)

    def local_chemistry_audit(self) -> Mapping[str, Any] | None:
        if self.local_chemistry_prior is None:
            return None
        return {
            **self.local_chemistry_prior.to_mapping(),
            "score_mode": self.local_chemistry_score_mode,
            "minimum_whole_head_support_tier": (
                self.minimum_whole_head_support_tier
                if self.local_chemistry_score_mode in WHOLE_HEAD_SCORE_MODES
                else None
            ),
            "unsupported_whole_head_candidates_eligible": (
                True if self.local_chemistry_score_mode in WHOLE_HEAD_SCORE_MODES else None
            ),
            "unsupported_amine_bond_orders_eligible": (
                False if self.local_chemistry_score_mode in WHOLE_HEAD_SCORE_MODES else None
            ),
            "component_score_aggregation": (
                (
                    "one binary measured-support indicator for the completed amine-head arrangement"
                    if self.local_chemistry_score_mode == "whole_head_support_binary"
                    else (
                        "one graded distance to measured whole-head morphology"
                        if self.local_chemistry_score_mode == "whole_head_support_distance"
                        else "one novelty-neutral support tier for the completed amine-head arrangement"
                    )
                )
                if self.local_chemistry_score_mode in WHOLE_HEAD_SCORE_MODES
                else (
                    "minimum support tier across local atom and adjacency evidence"
                    if self.local_chemistry_score_mode == "support_tier"
                    else "sum of smoothed local log probabilities"
                )
            ),
            "whole_head_total_variation_radius": self.whole_head_total_variation_radius,
            "whole_head_candidate_effective_count_retention": (
                self.whole_head_candidate_effective_count_retention
            ),
            "whole_head_group_effective_count_retention": (
                self.whole_head_group_effective_count_retention
            ),
            "whole_head_hard_bond_support": self.whole_head_hard_bond_support,
            "whole_head_topology_support": self.whole_head_topology_support,
            "whole_head_topology_score_mode": (
                self.whole_head_topology_score_mode
                if self.whole_head_topology_support
                else None
            ),
            "whole_head_topology_rank_weight": (
                self.effective_whole_head_topology_rank_weight
                if self.whole_head_topology_support
                else None
            ),
            "terminal_chemistry_selection_mode": self.terminal_chemistry_selection_mode,
            "terminal_bond_selection_mode": self.terminal_bond_selection_mode,
            "local_chemistry_bond_uniform_probability_mass": (
                self.local_chemistry_bond_uniform_probability_mass
            ),
            "local_chemistry_bond_position_basis": self.local_chemistry_bond_position_basis,
            "tail_unsaturation_count_tolerance": self.tail_unsaturation_count_tolerance,
            "tail_unsaturation_count_strategy": self.tail_unsaturation_count_strategy,
            "tail_unsaturation_position_strategy": self.tail_unsaturation_position_strategy,
            "tail_unsaturation_frequency_pseudocount": (
                self.tail_unsaturation_frequency_pseudocount
            ),
        }

    def atom_local_scores(
        self,
        *,
        role: str,
        depth: int,
        degree: int,
        in_ring: bool,
        atom_vocabulary: Sequence[Any],
    ) -> np.ndarray:
        """Return the configured measured-training atom objective."""

        if self.local_chemistry_prior is None:
            raise UgiMogSemanticGuidanceError("local-chemistry prior is not bound")
        if self.local_chemistry_score_mode in {
            "whole_head_support_tier",
            "whole_head_support_binary",
            "whole_head_support_distance",
        }:
            raise UgiMogSemanticGuidanceError(
                "whole-head support cannot score an isolated atom decision"
            )
        if self.local_chemistry_score_mode == "support_tier":
            return self.local_chemistry_prior.atom_support_tiers(
                role=role,
                depth=depth,
                degree=degree,
                in_ring=in_ring,
                atom_vocabulary=atom_vocabulary,
            )
        return self.local_chemistry_prior.atom_log_bias(
            role=role,
            depth=depth,
            degree=degree,
            in_ring=in_ring,
            atom_vocabulary=atom_vocabulary,
        )

    def bond_local_scores(
        self,
        *,
        role: str,
        depth: int,
        in_ring: bool,
        symbols: tuple[str, str],
        bond_classes: int,
        terminal_offset: int | None = None,
    ) -> np.ndarray:
        """Return the configured measured-training bond objective."""

        if self.local_chemistry_prior is None:
            raise UgiMogSemanticGuidanceError("local-chemistry prior is not bound")
        if self.local_chemistry_score_mode in {
            "support_tier",
            "whole_head_support_tier",
            "whole_head_support_binary",
            "whole_head_support_distance",
        }:
            # Only the global tail-unsaturation assignment has the completed role topology needed
            # to define this coordinate.  Fixed ester/core and amine bonds retain the historical
            # core-depth score rather than receiving a fabricated terminal position.
            if (
                self.local_chemistry_bond_position_basis == "terminal_offset"
                and terminal_offset is not None
            ):
                return self.local_chemistry_prior.bond_terminal_support_tiers(
                    role=role,
                    terminal_offset=terminal_offset,
                    in_ring=in_ring,
                    symbols=symbols,
                    bond_classes=bond_classes,
                )
            return self.local_chemistry_prior.bond_support_tiers(
                role=role,
                depth=depth,
                in_ring=in_ring,
                symbols=symbols,
                bond_classes=bond_classes,
            )
        return self.local_chemistry_prior.bond_log_bias(
            role=role,
            depth=depth,
            in_ring=in_ring,
            symbols=symbols,
            bond_classes=bond_classes,
        )

    def edge_symbol_local_score(
        self,
        *,
        role: str,
        depth: int,
        in_ring: bool,
        symbols: tuple[str, str],
    ) -> float | None:
        """Score adjacency only for the novelty-neutral support-tier intervention."""

        if self.local_chemistry_score_mode != "support_tier":
            return None
        if self.local_chemistry_prior is None:
            raise UgiMogSemanticGuidanceError("local-chemistry prior is not bound")
        return self.local_chemistry_prior.edge_symbol_support_tier(
            role=role,
            depth=depth,
            in_ring=in_ring,
            symbols=symbols,
        )

    def amine_head_arrangement_support_score(
        self,
        *,
        nodes: Sequence[int],
        symbols_by_node: Mapping[int, str],
        depths_by_node: Mapping[int, int],
        neighbors: Mapping[int, Sequence[int] | set[int]] | Sequence[set[int]],
        ring_nodes: frozenset[int] | set[int],
    ) -> float:
        """Score one completed head without favoring frequent training components."""

        if self.local_chemistry_score_mode not in {
            "whole_head_support_tier",
            "whole_head_support_binary",
            "whole_head_support_distance",
        }:
            raise UgiMogSemanticGuidanceError("whole-head support mode is not active")
        if self.local_chemistry_prior is None:
            raise UgiMogSemanticGuidanceError("local-chemistry prior is not bound")
        if self.local_chemistry_score_mode == "whole_head_support_distance":
            return self.local_chemistry_prior.amine_head_arrangement_similarity(
                nodes=nodes,
                symbols_by_node=symbols_by_node,
                depths_by_node=depths_by_node,
                neighbors=neighbors,
                ring_nodes=ring_nodes,
            )
        tier = self.local_chemistry_prior.amine_head_arrangement_support_tier(
            nodes=nodes,
            symbols_by_node=symbols_by_node,
            depths_by_node=depths_by_node,
            neighbors=neighbors,
            ring_nodes=ring_nodes,
        )
        return (
            float(tier > 0.0)
            if self.local_chemistry_score_mode == "whole_head_support_binary"
            else tier
        )

    def amine_head_topology_support_score(
        self,
        *,
        nodes: Sequence[int],
        depths_by_node: Mapping[int, int],
        neighbors: Mapping[int, Sequence[int] | set[int]] | Sequence[set[int]],
        ring_nodes: frozenset[int] | set[int],
    ) -> float:
        """Score one complete rooted head topology without atom or component identities."""

        if not self.whole_head_topology_support:
            raise UgiMogSemanticGuidanceError("whole-head topology support is not active")
        if self.local_chemistry_prior is None:
            raise UgiMogSemanticGuidanceError("local-chemistry prior is not bound")
        tier = self.local_chemistry_prior.amine_head_topology_support_tier(
            nodes=nodes,
            depths_by_node=depths_by_node,
            neighbors=neighbors,
            ring_nodes=ring_nodes,
        )
        return tier if self.whole_head_topology_score_mode == "tier" else float(tier > 0.0)

    def tail_unsaturation_count_options(
        self,
        *,
        role: str,
        requested_double_count: int,
        requested_triple_count: int,
    ) -> tuple[tuple[int, int], ...]:
        """Return the bounded measured-support count band for one tail role."""

        if self.local_chemistry_prior is None:
            raise UgiMogSemanticGuidanceError("local-chemistry prior is not bound")
        requested = (int(requested_double_count), int(requested_triple_count))
        supported = self.local_chemistry_prior.supported_unsaturation_counts(role)
        if self.tail_unsaturation_count_strategy in {
            "frequency_resampled",
            "smoothed_frequency_resampled",
        }:
            if not supported:
                raise UgiMogSemanticGuidanceError(
                    f"no measured unsaturation-count support for role {role!r}"
                )
            return supported
        options = tuple(
            value
            for value in supported
            if sum(abs(value[index] - requested[index]) for index in (0, 1))
            <= self.tail_unsaturation_count_tolerance
            and (
                self.tail_unsaturation_count_strategy
                not in {"flat_downward", "grouped_downward", "frequency_downward"}
                or (value[0] <= requested[0] and value[1] <= requested[1])
            )
        )
        if not options:
            raise UgiMogSemanticGuidanceError(
                f"no measured unsaturation-count support for role {role!r} near {requested}"
            )
        return options

    @property
    def minimum_whole_head_support_tier(self) -> float:
        """Minimum measured specificity admitted by whole-head constrained generation."""

        return 1.0

    def aggregate_local_scores(self, scores: Sequence[float]) -> float:
        """Aggregate without turning repeated supported coordinates into extra reward."""

        if not scores or not all(np.isfinite(value) for value in scores):
            raise UgiMogSemanticGuidanceError("local-chemistry evidence scores are invalid")
        return (
            float(min(scores))
            if self.local_chemistry_score_mode in ({"support_tier"} | WHOLE_HEAD_SCORE_MODES)
            else float(sum(scores))
        )

    def amine_targets_within_band(
        self, target: UgiAmineSemanticTarget
    ) -> tuple[UgiAmineSemanticTarget, ...]:
        """Enumerate bounded supported slack; measured N/O composition remains exact."""

        heavy = range(
            max(
                1, target.heavy_atom_graph_diameter - self.amine_heavy_atom_graph_diameter_tolerance
            ),
            target.heavy_atom_graph_diameter + self.amine_heavy_atom_graph_diameter_tolerance + 1,
        )
        carbon = range(
            max(0, target.carbon_skeleton_diameter - self.amine_carbon_skeleton_diameter_tolerance),
            target.carbon_skeleton_diameter + self.amine_carbon_skeleton_diameter_tolerance + 1,
        )
        donors: Sequence[int | None]
        branches: Sequence[int | None]
        if target.hydrogen_bond_donors is None:
            donors = (None,)
            branches = (None,)
        else:
            donors = range(
                max(
                    0,
                    target.hydrogen_bond_donors - self.amine_hydrogen_bond_donors_tolerance,
                ),
                min(
                    target.nitrogen_atoms,
                    target.hydrogen_bond_donors + self.amine_hydrogen_bond_donors_tolerance,
                )
                + 1,
            )
            branches = range(
                max(0, target.heavy_branch_atoms - self.amine_heavy_branch_atoms_tolerance),
                target.heavy_branch_atoms + self.amine_heavy_branch_atoms_tolerance + 1,
            )
        return tuple(
            UgiAmineSemanticTarget(
                heavy_atom_graph_diameter=heavy_diameter,
                carbon_skeleton_diameter=carbon_diameter,
                nitrogen_atoms=target.nitrogen_atoms,
                oxygen_atoms=target.oxygen_atoms,
                hydrogen_bond_donors=donor_count,
                heavy_branch_atoms=branch_count,
            )
            for heavy_diameter in heavy
            for carbon_diameter in carbon
            for donor_count in donors
            for branch_count in branches
        )

    def amine_distance(
        self, candidate: UgiAmineSemanticTarget, target: UgiAmineSemanticTarget
    ) -> float:
        distance = float(
            abs(candidate.heavy_atom_graph_diameter - target.heavy_atom_graph_diameter)
            + abs(candidate.carbon_skeleton_diameter - target.carbon_skeleton_diameter)
        )
        if target.hydrogen_bond_donors is not None:
            if candidate.hydrogen_bond_donors is None or candidate.heavy_branch_atoms is None:
                raise UgiMogSemanticGuidanceError(
                    "substitution-semantic candidate dropped a requested coordinate"
                )
            distance += float(
                abs(candidate.hydrogen_bond_donors - target.hydrogen_bond_donors)
                + abs(candidate.heavy_branch_atoms - target.heavy_branch_atoms)
            )
        return distance

    def aldehyde_directional_pairs_within_band(
        self, *, alkoxy_handle_carbons: int, acyl_carbons: int
    ) -> tuple[tuple[int, int], ...]:
        """Move the ester cut within a small band while conserving total carbon count."""

        tolerance = self.aldehyde_ester_side_carbons_tolerance
        output = {
            (alkoxy_handle_carbons + delta, acyl_carbons - delta)
            for delta in range(-tolerance, tolerance + 1)
            if alkoxy_handle_carbons + delta > 0 and acyl_carbons - delta > 0
        }
        return tuple(sorted(output))

    def aldehyde_distance(
        self,
        candidate: tuple[int, int],
        target: tuple[int, int],
    ) -> float:
        return float(abs(candidate[0] - target[0]) + abs(candidate[1] - target[1])) / 2.0

    def probabilities(
        self,
        model_scores: Sequence[float],
        semantic_distances: Sequence[float],
        joint_realism_scores: Sequence[float] | None = None,
        local_chemistry_scores: Sequence[float] | None = None,
        *,
        local_chemistry_rank_weight: float | None = None,
        uniform_probability_mass: float | None = None,
    ) -> np.ndarray:
        """Return a rank-weighted categorical law with an explicit entropy floor."""

        model = np.asarray(model_scores, dtype=np.float64)
        semantic = np.asarray(semantic_distances, dtype=np.float64)
        realism = (
            None
            if joint_realism_scores is None
            else np.asarray(joint_realism_scores, dtype=np.float64)
        )
        chemistry = (
            None
            if local_chemistry_scores is None
            else np.asarray(local_chemistry_scores, dtype=np.float64)
        )
        chemistry_weight = (
            self.local_chemistry_rank_weight
            if local_chemistry_rank_weight is None
            else float(local_chemistry_rank_weight)
        )
        uniform_mass = (
            self.uniform_probability_mass
            if uniform_probability_mass is None
            else float(uniform_probability_mass)
        )
        if (
            model.ndim != 1
            or semantic.shape != model.shape
            or model.size < 1
            or not np.all(np.isfinite(model))
            or not np.all(np.isfinite(semantic))
            or np.any(semantic < 0)
            or (
                realism is not None
                and (realism.shape != model.shape or not np.isfinite(realism).all())
            )
            or (
                chemistry is not None
                and (chemistry.shape != model.shape or not np.isfinite(chemistry).all())
            )
            or (self.uses_joint_realism != (realism is not None))
            or not np.isfinite(chemistry_weight)
            or chemistry_weight < 0
            or (chemistry is not None and chemistry_weight <= 0)
            or not np.isfinite(uniform_mass)
            or not 0 < uniform_mass < 1
        ):
            raise UgiMogSemanticGuidanceError("ranked candidate scores are invalid")

        def dense_cost(values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
            ordered = np.unique(values)
            if higher_is_better:
                ordered = ordered[::-1]
            rank_by_value = {float(value): rank for rank, value in enumerate(ordered.tolist())}
            denominator = max(len(ordered) - 1, 1)
            return np.asarray(
                [rank_by_value[float(value)] / denominator for value in values],
                dtype=np.float64,
            )

        cost = self.model_rank_weight * dense_cost(
            model, higher_is_better=True
        ) + self.semantic_rank_weight * dense_cost(semantic, higher_is_better=False)
        if realism is not None:
            cost += self.joint_realism_rank_weight * dense_cost(
                realism,
                higher_is_better=True,
            )
        if chemistry is not None:
            cost += chemistry_weight * dense_cost(
                chemistry,
                higher_is_better=True,
            )
        logits = -cost / self.rank_temperature
        logits -= np.max(logits)
        ranked = np.exp(logits)
        ranked /= ranked.sum()
        uniform = np.full(model.size, 1.0 / model.size, dtype=np.float64)
        probabilities = (1.0 - uniform_mass) * ranked + uniform_mass * uniform
        probabilities /= probabilities.sum()
        return probabilities

    def chemistry_probabilities(
        self,
        model_scores: Sequence[float],
        semantic_distances: Sequence[float],
        local_chemistry_scores: Sequence[float],
        joint_realism_scores: Sequence[float] | None = None,
        *,
        rank_weight: float | None = None,
        uniform_probability_mass: float | None = None,
    ) -> np.ndarray:
        """Rank one chemistry decision after hard validity and support masking."""

        effective_weight = self.local_chemistry_rank_weight if rank_weight is None else rank_weight
        if effective_weight <= 0 or self.local_chemistry_prior is None:
            raise UgiMogSemanticGuidanceError(
                "local-chemistry ranking requires one bound measured-train prior"
            )
        return self.probabilities(
            model_scores,
            semantic_distances,
            joint_realism_scores,
            local_chemistry_scores,
            local_chemistry_rank_weight=effective_weight,
            uniform_probability_mass=uniform_probability_mass,
        )

    def atom_chemistry_probabilities(
        self,
        model_scores: Sequence[float],
        semantic_distances: Sequence[float],
        local_chemistry_scores: Sequence[float],
        joint_realism_scores: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Rank an atom choice inside the configured Transformer trust region."""

        guided = self.chemistry_probabilities(
            model_scores,
            semantic_distances,
            local_chemistry_scores,
            joint_realism_scores,
        )
        radius = self.local_chemistry_atom_total_variation_radius
        if radius is None:
            return guided
        baseline = self.probabilities(
            model_scores,
            semantic_distances,
            joint_realism_scores,
        )
        total_variation = 0.5 * float(np.abs(guided - baseline).sum())
        if total_variation <= radius:
            return guided
        mixture_weight = float(radius) / total_variation
        probabilities = baseline + mixture_weight * (guided - baseline)
        probabilities /= probabilities.sum()
        return probabilities

    def whole_head_chemistry_probabilities(
        self,
        model_scores: Sequence[float],
        semantic_distances: Sequence[float],
        local_chemistry_scores: Sequence[float],
        joint_realism_scores: Sequence[float] | None = None,
        *,
        group_labels: Sequence[object] | None = None,
    ) -> np.ndarray:
        """Rank complete head assignments inside one component-level trust region."""

        if self.local_chemistry_score_mode not in {
            "whole_head_support_tier",
            "whole_head_support_binary",
            "whole_head_support_distance",
        }:
            raise UgiMogSemanticGuidanceError("whole-head chemistry mode is not active")
        guided = self.chemistry_probabilities(
            model_scores,
            semantic_distances,
            local_chemistry_scores,
            joint_realism_scores,
        )
        radius = self.whole_head_total_variation_radius
        baseline = self.probabilities(
            model_scores,
            semantic_distances,
            joint_realism_scores,
        )
        retention = self.whole_head_candidate_effective_count_retention
        if retention is not None:
            baseline_entropy = -float(np.sum(baseline * np.log(baseline)))
            minimum_entropy = baseline_entropy + float(np.log(retention))

            def entropy(probabilities: np.ndarray) -> float:
                return -float(np.sum(probabilities * np.log(probabilities)))

            if entropy(guided) < minimum_entropy:
                uniform = np.full(guided.size, 1.0 / guided.size, dtype=np.float64)
                lower = 0.0
                upper = 1.0
                for _ in range(64):
                    mixture = 0.5 * (lower + upper)
                    candidate = (1.0 - mixture) * guided + mixture * uniform
                    if entropy(candidate) >= minimum_entropy:
                        upper = mixture
                    else:
                        lower = mixture
                guided = (1.0 - upper) * guided + upper * uniform
                guided /= guided.sum()
        group_retention = self.whole_head_group_effective_count_retention
        if group_retention is not None:
            if group_labels is None or len(group_labels) != guided.size:
                raise UgiMogSemanticGuidanceError(
                    "whole-head group entropy guard requires one label per candidate"
                )
            group_index: dict[object, int] = {}
            candidate_groups = np.empty(guided.size, dtype=np.int64)
            for candidate_index, label in enumerate(group_labels):
                if label not in group_index:
                    group_index[label] = len(group_index)
                candidate_groups[candidate_index] = group_index[label]
            group_count = len(group_index)
            baseline_groups = np.bincount(
                candidate_groups,
                weights=baseline,
                minlength=group_count,
            )
            guided_groups = np.bincount(
                candidate_groups,
                weights=guided,
                minlength=group_count,
            )

            def group_entropy(probabilities: np.ndarray) -> float:
                positive = probabilities[probabilities > 0]
                return -float(np.sum(positive * np.log(positive)))

            minimum_group_entropy = group_entropy(baseline_groups) + float(np.log(group_retention))
            if group_entropy(guided_groups) < minimum_group_entropy:
                uniform_groups = np.full(group_count, 1.0 / group_count, dtype=np.float64)
                lower = 0.0
                upper = 1.0
                for _ in range(64):
                    mixture = 0.5 * (lower + upper)
                    candidate_groups_probability = (
                        1.0 - mixture
                    ) * guided_groups + mixture * uniform_groups
                    if group_entropy(candidate_groups_probability) >= minimum_group_entropy:
                        upper = mixture
                    else:
                        lower = mixture
                target_groups = (1.0 - upper) * guided_groups + upper * uniform_groups
                guided = guided * (
                    target_groups[candidate_groups] / guided_groups[candidate_groups]
                )
                guided /= guided.sum()
        if radius is None:
            return guided
        total_variation = 0.5 * float(np.abs(guided - baseline).sum())
        if total_variation <= radius:
            return guided
        mixture_weight = float(radius) / total_variation
        probabilities = baseline + mixture_weight * (guided - baseline)
        probabilities /= probabilities.sum()
        return probabilities

    def bond_chemistry_probabilities(
        self,
        model_scores: Sequence[float],
        semantic_distances: Sequence[float],
        local_chemistry_scores: Sequence[float],
        joint_realism_scores: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Rank a bond decision only when the policy explicitly enables that objective."""

        return self.chemistry_probabilities(
            model_scores,
            semantic_distances,
            local_chemistry_scores,
            joint_realism_scores,
            rank_weight=self.effective_local_chemistry_bond_rank_weight,
            uniform_probability_mass=(
                self.uniform_probability_mass
                if self.local_chemistry_bond_uniform_probability_mass is None
                else self.local_chemistry_bond_uniform_probability_mass
            ),
        )


__all__ = [
    "JOINT_REALISM_FEATURE_NAMES",
    "UgiJointSemanticRealismScorer",
    "UgiMogSemanticGuidanceError",
    "UgiMogSemanticGuidancePolicy",
    "replace_amine_semantics",
    "replace_directional_ester_semantics",
    "replace_tail_unsaturation_semantics",
]

"""Reaction-program specialization stage adapters and preflight variants."""

from __future__ import annotations

from experiments._runtime.registry import stage
from experiments._runtime.stage import (
    ProducedArtifact,
    RunContext,
    StageResult,
    require_config_inputs,
)


def _reaction_program_specialist_stage(
    context: RunContext,
    *,
    profile: str,
    topology_specialist: bool,
    chemistry_specialist: bool = False,
    contextual_chemistry_specialist: bool = False,
    joint_lipid_specialist: bool = False,
    role_local_decoder: bool = False,
    measured_only_full_model: bool = False,
    structured_topology: bool = False,
) -> StageResult:
    from experiments.phase1.multireaction.reaction_specialization import (
        run_reaction_program_specialization,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_reaction_program_specialization(
        context.config_path,
        context.repo,
        context.input("production_cache"),
        context.output_path("."),
        work_dir=context.work_dir,
        profile=profile,
        allocated_device=context.resources.device,
        resume=context.resume,
    )
    if (
        sum(
            (
                bool(topology_specialist),
                bool(chemistry_specialist),
                bool(contextual_chemistry_specialist),
                bool(joint_lipid_specialist),
                bool(role_local_decoder),
                bool(measured_only_full_model),
                bool(structured_topology),
            )
        )
        > 1
    ):
        raise ValueError("a specialist stage must declare exactly one delta type")
    suffix = (
        "topology_specialization"
        if topology_specialist
        else (
            "measured_only_full_finetune"
            if measured_only_full_model
            else (
                "structured_topology_specialization"
                if structured_topology
                else (
                    "role_local_decoder_specialization"
                    if role_local_decoder
                    else (
                        "contextual_chemistry_specialization"
                        if contextual_chemistry_specialist
                        else (
                            "chemistry_specialization"
                            if chemistry_specialist
                            else (
                                "joint_lipid_specialization"
                                if joint_lipid_specialist
                                else "specialization"
                            )
                        )
                    )
                )
            )
        )
    )
    checkpoint_schema = (
        "forge.reaction_program_topology_specialist_checkpoint.v2"
        if topology_specialist
        else (
            "forge.reaction_program_measured_only_full_finetune_checkpoint.v7"
            if measured_only_full_model
            else (
                "forge.reaction_program_structured_topology_specialist_checkpoint.v8"
                if structured_topology
                else (
                    "forge.reaction_program_role_local_decoder_specialist_checkpoint.v6"
                    if role_local_decoder
                    else (
                        "forge.reaction_program_contextual_chemistry_specialist_checkpoint.v4"
                        if contextual_chemistry_specialist
                        else (
                            "forge.reaction_program_chemistry_specialist_checkpoint.v3"
                            if chemistry_specialist
                            else (
                                "forge.reaction_program_joint_lipid_specialist_checkpoint.v5"
                                if joint_lipid_specialist
                                else "forge.reaction_program_specialist_checkpoint.v1"
                            )
                        )
                    )
                )
            )
        )
    )
    schema_version = (
        "v2"
        if topology_specialist
        else (
            "v7"
            if measured_only_full_model
            else (
                "v8"
                if structured_topology
                else (
                    "v6"
                    if role_local_decoder
                    else (
                        "v4"
                        if contextual_chemistry_specialist
                        else (
                            "v3"
                            if chemistry_specialist
                            else "v5" if joint_lipid_specialist else "v1"
                        )
                    )
                )
            )
        )
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "checkpoint",
                "specialist_checkpoint.pt",
                checkpoint_schema,
            ),
            ProducedArtifact(
                "progress",
                "progress.json",
                f"forge.reaction_program_{suffix}_progress.{schema_version}",
            ),
            ProducedArtifact(
                "result",
                "result.json",
                f"forge.reaction_program_{suffix}_result.{schema_version}",
            ),
        ),
        metrics={
            "optimizer_steps": int(result["observed"]["optimizer_steps"]),
            "additional_examples": int(result["observed"]["additional_examples"]),
            "cumulative_examples": int(result["observed"]["cumulative_examples"]),
            "specialist_parameters": int(result["specialist"]["specialist_trainable_parameters"]),
        },
        summary={
            "status": result["status"],
            "target_program": result["target_program"],
            "preflight_only": profile != "full",
            "shared_parameters_frozen": not measured_only_full_model,
            "full_model_trainable": measured_only_full_model,
            "topology_head_supervised": topology_specialist,
            "structured_topology_head": structured_topology,
            "role_local_tree_decoder": role_local_decoder,
            "measured_only_full_model": measured_only_full_model,
            "chemistry_outputs_only": (chemistry_specialist or contextual_chemistry_specialist),
            "contextual_chemistry_refinement": contextual_chemistry_specialist,
            "joint_lipid_training_measure": joint_lipid_specialist,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("model.reaction-program-specialization.v1")
def train_reaction_program_specialist(context: RunContext) -> StageResult:
    """Fine-tune one lightweight family adapter from an authenticated shared checkpoint."""

    return _reaction_program_specialist_stage(
        context,
        profile=context.profile,
        topology_specialist=False,
    )


@stage("model.reaction-program-specialization-h100-preflight.v1")
def preflight_reaction_program_specialist_h100(context: RunContext) -> StageResult:
    """Exercise authenticated specialist loading, a short final batch and checkpointing on H100."""

    return _reaction_program_specialist_stage(
        context,
        profile="smoke",
        topology_specialist=False,
    )


@stage("model.reaction-program-topology-specialization.v2")
def train_reaction_program_topology_specialist(context: RunContext) -> StageResult:
    """Fine-tune a family adapter and explicit child-count head from one frozen base."""

    return _reaction_program_specialist_stage(
        context,
        profile=context.profile,
        topology_specialist=True,
    )


@stage("model.reaction-program-topology-specialization-h100-preflight.v2")
def preflight_reaction_program_topology_specialist_h100(
    context: RunContext,
) -> StageResult:
    """Qualify the topology-specialist delta and direct objective on the target H100."""

    return _reaction_program_specialist_stage(
        context,
        profile="h100_preflight",
        topology_specialist=True,
    )


@stage("model.reaction-program-chemistry-specialization.v3")
def train_reaction_program_chemistry_specialist(context: RunContext) -> StageResult:
    """Fine-tune only atom and bond output adapters under frozen target topology."""

    return _reaction_program_specialist_stage(
        context,
        profile=context.profile,
        topology_specialist=False,
        chemistry_specialist=True,
    )


@stage("model.reaction-program-chemistry-specialization-h100-preflight.v3")
def preflight_reaction_program_chemistry_specialist_h100(
    context: RunContext,
) -> StageResult:
    """Qualify chemistry-only gradients and checkpointing on the target H100."""

    return _reaction_program_specialist_stage(
        context,
        profile="h100_preflight",
        topology_specialist=False,
        chemistry_specialist=True,
    )


@stage("model.reaction-program-contextual-chemistry-specialization.v4")
def train_reaction_program_contextual_chemistry_specialist(
    context: RunContext,
) -> StageResult:
    """Train chemistry-only local graph refiners while preserving every topology output."""

    return _reaction_program_specialist_stage(
        context,
        profile=context.profile,
        topology_specialist=False,
        contextual_chemistry_specialist=True,
    )


@stage("model.reaction-program-contextual-chemistry-specialization-h100-preflight.v4")
def preflight_reaction_program_contextual_chemistry_specialist_h100(
    context: RunContext,
) -> StageResult:
    """Qualify contextual chemistry gradients and checkpointing on the target H100."""

    return _reaction_program_specialist_stage(
        context,
        profile="h100_preflight",
        topology_specialist=False,
        contextual_chemistry_specialist=True,
    )


@stage("model.reaction-program-joint-lipid-specialization.v5")
def train_reaction_program_joint_lipid_specialist(context: RunContext) -> StageResult:
    """Train a full-output Ugi adapter under the frozen measured/exploration joint measure."""

    return _reaction_program_specialist_stage(
        context,
        profile=context.profile,
        topology_specialist=False,
        joint_lipid_specialist=True,
    )


@stage("model.reaction-program-joint-lipid-specialization-h100-preflight.v5")
def preflight_reaction_program_joint_lipid_specialist_h100(
    context: RunContext,
) -> StageResult:
    """Qualify the identity-free joint Ugi measure and adapter update on one exact H100."""

    return _reaction_program_specialist_stage(
        context,
        profile="h100_preflight",
        topology_specialist=False,
        joint_lipid_specialist=True,
    )


@stage("model.reaction-program-role-local-decoder-specialization.v6")
def train_reaction_program_role_local_decoder_specialist(
    context: RunContext,
) -> StageResult:
    """Train only the Ugi role-local tree decoder from the authenticated shared model."""

    return _reaction_program_specialist_stage(
        context,
        profile=context.profile,
        topology_specialist=False,
        role_local_decoder=True,
    )


@stage("model.reaction-program-role-local-decoder-specialization-h100-preflight.v6")
def preflight_reaction_program_role_local_decoder_specialist_h100(
    context: RunContext,
) -> StageResult:
    """Qualify role-local topology/chemistry updates and delta checkpointing on H100."""

    return _reaction_program_specialist_stage(
        context,
        profile="h100_preflight",
        topology_specialist=False,
        role_local_decoder=True,
    )


@stage("model.reaction-program-measured-only-full-finetune.v7")
def train_reaction_program_measured_only_full_model(context: RunContext) -> StageResult:
    """Fine-tune the complete shared model only on measured Ugi train-fold products."""

    return _reaction_program_specialist_stage(
        context,
        profile=context.profile,
        topology_specialist=False,
        measured_only_full_model=True,
    )


@stage("model.reaction-program-measured-only-full-finetune-h100-preflight.v7")
def preflight_reaction_program_measured_only_full_model_h100(
    context: RunContext,
) -> StageResult:
    """Qualify full-model measured-only updates and restart persistence on H100."""

    return _reaction_program_specialist_stage(
        context,
        profile="h100_preflight",
        topology_specialist=False,
        measured_only_full_model=True,
    )


@stage("model.reaction-program-structured-topology-specialization.v8")
def train_reaction_program_structured_topology_specialist(
    context: RunContext,
) -> StageResult:
    """Train only grammar-normalized Ugi tree and closure residual scores."""

    return _reaction_program_specialist_stage(
        context,
        profile=context.profile,
        topology_specialist=False,
        structured_topology=True,
    )


@stage("model.reaction-program-structured-topology-specialization-h100-preflight.v8")
def preflight_reaction_program_structured_topology_specialist_h100(
    context: RunContext,
) -> StageResult:
    """Qualify the frozen-backbone structured Ugi topology objective on H100."""

    return _reaction_program_specialist_stage(
        context,
        profile="h100_preflight",
        topology_specialist=False,
        structured_topology=True,
    )


__all__ = [
    "train_reaction_program_specialist",
    "preflight_reaction_program_specialist_h100",
    "train_reaction_program_topology_specialist",
    "preflight_reaction_program_topology_specialist_h100",
    "train_reaction_program_chemistry_specialist",
    "preflight_reaction_program_chemistry_specialist_h100",
    "train_reaction_program_contextual_chemistry_specialist",
    "preflight_reaction_program_contextual_chemistry_specialist_h100",
    "train_reaction_program_joint_lipid_specialist",
    "preflight_reaction_program_joint_lipid_specialist_h100",
    "train_reaction_program_role_local_decoder_specialist",
    "preflight_reaction_program_role_local_decoder_specialist_h100",
    "train_reaction_program_measured_only_full_model",
    "preflight_reaction_program_measured_only_full_model_h100",
    "train_reaction_program_structured_topology_specialist",
    "preflight_reaction_program_structured_topology_specialist_h100",
]

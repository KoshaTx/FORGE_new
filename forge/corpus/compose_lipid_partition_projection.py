"""Extend frozen corrected split protections without extending training admission."""

from __future__ import annotations

from forge.assembly.compose_lipid import ComposeLipidError
from forge.corpus.compose_lipid_partition_signatures import (
    FrozenGroupProjection,
    SourceSplitSignatures,
)


def project_preparation_record(
    source: dict,
    preparation: dict,
    *,
    signatures: SourceSplitSignatures,
    projection: FrozenGroupProjection,
    source_pmids: list,
) -> dict:
    """Add known exclusions; a projected TRAIN alone cannot release an unassigned row."""
    target, family = preparation["target_id"], preparation["family"]
    if (
        target != source["target_id"]
        or family != source["primary_family"]
        or preparation["source_anchor"] != source["source_anchor"]
        or preparation["source_declared_heavy_atoms"] != source["heavy_atoms"]
        or preparation.get("training_admitted") is not False
    ):
        raise ComposeLipidError("Frozen projection source/preparation binding differs")
    prior = preparation["disposition"]
    if prior not in {
        "protected",
        "eligible_for_program_preparation",
        "unresolved_partition_or_study",
    }:
        raise ComposeLipidError("Unknown prior preparation disposition")
    description = signatures.describe(
        family=family,
        smiles=source["constitution"],
        metadata=source["primary_metadata"],
        source_anchor=source["source_anchor"],
        instances=preparation["component_instances"],
    )
    projected = projection.project(
        family=family,
        component_ids=sorted({i[1] for i in preparation["component_instances"]}),
        source_pmids=source_pmids,
        morphology=description["morphology_group_signature"],
        combination=description["combination_signature"],
    )
    missing_study = source["source_anchor"] and not source_pmids
    if prior == "protected" or projected["split"] != "train":
        disposition = "protected"
    elif missing_study or prior == "unresolved_partition_or_study":
        disposition = "unresolved_partition_or_study"
    else:
        disposition = "eligible_for_program_preparation"
    return {
        "target_id": target,
        "family": family,
        "morphology_group_signature": description["morphology_group_signature"],
        "combination_signature": description["combination_signature"],
        "corrected_projection": projected,
        "prior_disposition": prior,
        "disposition": disposition,
        "newly_protected": prior != "protected" and disposition == "protected",
        "source_study_identity_unresolved": missing_study,
        "full_universe_partition_qualified": False,
        "training_admitted": False,
    }

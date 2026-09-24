"""Resolve exact v8 construction inputs into canonical component structures.

The v8 release stores compact family-specific metadata.  Those identifiers are
adequate for replay, but they are not uniformly structural: notably the
Vitamin-B5 splitter used a ``series + axis`` label for many distinct complete
reagents.  Model-facing grouped splits need the complete structures instead.

This module is chemistry-only.  It converts an already-bound exact task into a
multiset of complete precursor structures and stable, role-independent
constitutional identities.  Roles remain attached to instances for regional
conditioning and exact-combination grouping.
"""

from __future__ import annotations

from rdkit import Chem

from compose_lipid.data.source_aema_checks import AEMA
from compose_lipid.data.training_corpus import digest


def canonical_component(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid component structure: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def component_identity(smiles: str) -> str:
    return digest(
        ["post_instruction_component_constitution_v8_1", canonical_component(smiles)]
    )


def component(role: str, smiles: str, multiplicity: int = 1) -> list[dict]:
    if not role or type(multiplicity) is not int or multiplicity < 1:
        raise ValueError("invalid component role or multiplicity")
    constitution = canonical_component(smiles)
    identity = component_identity(constitution)
    return [
        {
            "role": role,
            "component_id": identity,
            "constitution": constitution,
        }
        for _ in range(multiplicity)
    ]


def _direct(task: dict, specifications: tuple[tuple[str, str, int | str], ...]):
    result = []
    for role, field, multiplicity in specifications:
        count = task[multiplicity] if isinstance(multiplicity, str) else multiplicity
        result.extend(component(role, task[field], count))
    return result


def _vitamin_b5(task: dict) -> list[dict]:
    reagents = task["reagents"]
    series = task["series"]
    if series == "I7" and len(reagents) == 3:
        roles = ("vitamin_b5_core", "paired_tail_alcohol", "head_acid")
    elif series == "I8" and len(reagents) == 4:
        roles = (
            "vitamin_b5_core",
            "r2_tail_alcohol",
            "head_acid",
            "tail_acid",
        )
    elif series == "I9" and len(reagents) == 4:
        roles = (
            "vitamin_b5_core",
            "tail_acid_1",
            "tail_acid_2",
            "head_nucleophile",
        )
    else:
        raise ValueError("unregistered Vitamin-B5 task architecture")
    return [
        component(role, reagent["smiles"])[0]
        for role, reagent in zip(roles, reagents, strict=True)
    ]


def virtual_task_components(
    family: str, task: dict, ugi4_precursors: dict[str, dict] | None = None
) -> list[dict]:
    """Return complete precursor instances for one exact virtual task."""

    if family == "preassembled_thiol_yne_tail_amidation":
        return _direct(
            task,
            (
                ("amine_head", "head_smiles", 1),
                ("alkynoic_linker", "linker_smiles", 1),
                ("thiol_tail", "tail_smiles", "repeated_tail_count"),
            ),
        )
    if family == "o_esterification":
        return _direct(
            task,
            (
                ("aminoalcohol_head", "head_smiles", 1),
                ("acid_tail", "acid_smiles", "occupancy"),
            ),
        )
    if family == "aldehyde_ugi4":
        if ugi4_precursors is None:
            raise ValueError("Ugi-4 precursor catalogue required")
        result = []
        for identifier in task["precursor_ids"]:
            precursor = ugi4_precursors.get(identifier)
            if precursor is None:
                raise ValueError("unresolved Ugi-4 precursor: " + identifier)
            result.extend(component(precursor["role"], precursor["smiles"]))
        return result
    if family == "ketone_ugi4":
        return _direct(
            task,
            (
                ("amine_head", "head_smiles", 1),
                ("coupled_ketone", "ketone_smiles", 1),
                ("isocyanide", "isocyanide_smiles", 1),
                ("carboxylic_acid", "acid_smiles", 1),
            ),
        )
    if family == "aldehyde_ugi3":
        if task.get("complete_subcomponents"):
            result = []
            for item in task["complete_subcomponents"]:
                result.extend(
                    component(item["reaction_role"], item["constitutional_smiles"])
                )
            return result
        if task.get("precursors"):
            result = []
            for item in task["precursors"]:
                result.extend(component(item["role"], item["smiles"]))
            return result
        raise ValueError("unresolved Ugi-3 task components")
    if family == "a3_amine_aldehyde_alkyne":
        return _direct(
            task,
            (
                ("amine_head", "head_smiles", 1),
                ("aldehyde", "aldehyde_smiles", "events"),
                ("alkyne", "alkyne_smiles", "events"),
            ),
        )
    if family == "aema_aza_thiol_addition":
        return (
            component("amine_core", task["core_smiles"])
            + component("AEMA", AEMA, task["branches"])
            + component("thiol_periphery", task["thiol_smiles"], task["branches"])
        )
    if family in {"aza_michael_acrylate", "aza_michael_acrylamide"}:
        return _direct(
            task,
            (
                ("amine_head", "head_smiles", 1),
                ("acceptor_tail", "tail_smiles", "occupancy"),
            ),
        )
    if family == "amine_epoxide_opening":
        return _direct(
            task,
            (
                ("amine_head", "head_smiles", 1),
                ("epoxide_tail", "tail_smiles", "occupancy"),
            ),
        )
    if family == "epoxide_opening_o_acylation":
        return _direct(
            task,
            (
                ("amine_head", "head_smiles", 1),
                ("epoxide_tail", "epoxide_smiles", 2),
                ("acyl_tail", "acyl_smiles", 2),
            ),
        )
    if family in {
        "alpha_isocyanoester_dihydroimidazole",
        "ketone_isocyanide_amide",
    }:
        return _direct(
            task,
            (
                ("amine_head", "head_smiles", 1),
                ("coupled_ketone", "ketone_smiles", 1),
                ("isocyanide", "isocyanide_smiles", 1),
            ),
        )
    if family == "amine_alkylation":
        return _direct(
            task,
            (
                ("amine_head", "head_smiles", 1),
                ("bromoester_arm", "arm_smiles", 2),
            ),
        )
    if family in {"reductive_amination", "aryl_reductive_amination"}:
        return _direct(
            task,
            (
                ("amine_head", "head_smiles", 1),
                ("coupled_aldehyde", "aldehyde_smiles", 1),
            ),
        )
    if family == "disulfide_michael":
        return _direct(
            task,
            (
                ("amine_head", "head_smiles", 1),
                ("disulfide_acceptor", "arm_smiles", "occupancy"),
            ),
        )
    if family == "thiolactone_aminolysis_michael":
        return _direct(
            task,
            (
                ("amine_head", "head_smiles", 1),
                ("thiolactone_region", "thiolactone_smiles", 1),
                ("acrylate_tail", "acrylate_smiles", 1),
            ),
        )
    if family == "iphos_ring_opening":
        return _direct(
            task,
            (
                ("amine_head", "head_smiles", 1),
                ("phosphate_tail", "phosphate_smiles", "event_count"),
            ),
        )
    if family == "maleate_addition":
        return _direct(
            task,
            (
                ("amine_head", "head_smiles", 1),
                ("maleate", "arm_smiles", "occupancy"),
            ),
        )
    if family == "acid_epoxide_diester_multistep":
        return _direct(
            task,
            (
                ("amine_acid_head", "head_smiles", 1),
                ("epoxide", "epoxide_smiles", 1),
                ("hydrophobic_acid", "hydrophobic_acid_smiles", 1),
            ),
        )
    if family == "passerini_3cr":
        return _direct(
            task,
            (
                ("amine_acid_head", "head_smiles", 1),
                ("aldehyde_tail", "aldehyde_smiles", 1),
                ("isocyanide_tail", "isocyanide_smiles", 1),
            ),
        )
    if family == "vitamin_b5_multistep":
        return _vitamin_b5(task)
    raise ValueError("unregistered v8 family task: " + family)


__all__ = [
    "canonical_component",
    "component_identity",
    "virtual_task_components",
]

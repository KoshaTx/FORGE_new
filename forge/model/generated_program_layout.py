"""Compile exact generated products into semantics without a source training layout.

The graph supplies its own program witnesses. Each witness must have unambiguous atom semantics;
distinct exact programs remain explicit alternatives rather than a fictitious unique route.
This is recycling generated graph context, not a claim of unconditional layout generation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from forge.assembly.families import LibraryAssemblyError, constitutional_molecule
from forge.assembly.library_generation import check_generated_program
from forge.assembly.library_programs import LibraryProgramLimits
from forge.assembly.library_semantics import trace_library_atom_semantics
from forge.model.reaction_program_flow import collate_synthesis_program_layouts
from forge.model.synthesis_program_graph import (
    SynthesisProgramGraphError,
    SynthesisProgramGraphRecord,
    tensorize_synthesis_program_product,
)
from forge.potency.annotations import UgiSemanticAnnotationError, annotate_qualified_ugi_product


class GeneratedLayoutError(ValueError):
    """The requested compiler contract is invalid."""


@dataclass(frozen=True)
class GeneratedLayouts:
    status: str
    records: tuple[SynthesisProgramGraphRecord, ...] = ()
    witnesses: tuple[dict, ...] = ()
    witness_count: int = 0
    reason: str | None = None

    def __post_init__(self):
        if (self.status == "qualified") != bool(self.records) or (
            len(self.records) != len(self.witnesses)
            or (self.records and len(self.records) != self.witness_count)
        ):
            raise GeneratedLayoutError("layout status and record disagree")


def compile_generated_layouts(
    *,
    smiles,
    depth,
    adapter,
    program_policy,
    semantic_policy,
    limits,
    support,
    vocabulary,
    atoms,
    ugi_reaction=None,
) -> GeneratedLayouts:
    """Reconstruct and tensorize a generated graph under unchanged library semantics."""
    family = adapter.reaction_id
    if family not in vocabulary.program_to_index or depth not in semantic_policy["allowed_depths"]:
        raise GeneratedLayoutError("family/depth lies outside the semantic contract")
    if type(semantic_policy["fix_core_atoms"]) is not bool:
        raise GeneratedLayoutError("fix_core_atoms must be Boolean")
    try:
        canonical, molecule = constitutional_molecule(smiles)
    except LibraryAssemblyError as error:
        return GeneratedLayouts("invalid_graph", reason=str(error))
    if molecule.GetNumAtoms() > support["maximum_heavy_atoms"] or (
        molecule.GetNumBonds() - molecule.GetNumAtoms() + 1 > support["maximum_closures"]
    ):
        return GeneratedLayouts("outside_graph_support")
    check = check_generated_program(
        adapter,
        canonical,
        depth=depth,
        accumulator_role=program_policy["accumulator_role"],
        limits=LibraryProgramLimits(program_policy["maximum_steps"], **limits),
    )
    if not check.exact:
        return GeneratedLayouts(check.status, reason=check.reason)
    product_id = hashlib.sha256(canonical.encode()).hexdigest()
    assignments = []
    for witness in check.programs:
        try:
            if family == "ugi_3cr_agile":
                if ugi_reaction is None:
                    raise GeneratedLayoutError("Ugi requires its qualified semantic annotator")
                product, rows, *_ = annotate_qualified_ugi_product(
                    ugi_reaction,
                    product_id=product_id,
                    target_smiles=canonical,
                    component_smiles_by_role=witness["components"],
                    source_evidence_record_id="generated_computed_transform_consistency",
                    max_outcomes=limits["maximum_outcomes"],
                )
                if product["semantic_signature_multiplicity"] != 1:
                    return GeneratedLayouts(
                        "ambiguous_semantics", witness_count=len(check.programs)
                    )
                roles = tuple(str(row["origin_role"]) for row in rows)
                positions = tuple(str(row["core_position"]) for row in rows)
                expected_fixed = tuple(
                    int(row["product_atom_index"]) for row in rows if row["is_ugi_core"]
                )
                if product["product_smiles"] != canonical:
                    raise GeneratedLayoutError("Ugi annotation changed product identity")
            else:
                annotation = trace_library_atom_semantics(
                    adapter,
                    witness["components"],
                    witness["intermediate_products"],
                    accumulator_role=program_policy["accumulator_role"],
                    maximum_outcomes=limits["maximum_outcomes"],
                    core_position_aliases=semantic_policy.get("core_position_aliases", {}),
                )
                if annotation.canonical_product_smiles != canonical:
                    raise GeneratedLayoutError("annotation changed generated product identity")
                roles, positions = annotation.atom_origins, annotation.core_positions
                expected_fixed = None
            if set(roles) != set(semantic_policy["roles"]) or not {
                p for p in positions if p
            }.issubset(semantic_policy["core_positions"]):
                return GeneratedLayouts(
                    "outside_semantic_support", witness_count=len(check.programs)
                )
            fixed = tuple(
                i
                for i, position in enumerate(positions)
                if position and semantic_policy["fix_core_atoms"]
            )
            if expected_fixed is not None and (fixed != expected_fixed or len(fixed) != 5):
                raise GeneratedLayoutError("settled Ugi fixed core changed")
            assignments.append((roles, positions, fixed))
        except (
            LibraryAssemblyError,
            UgiSemanticAnnotationError,
            ValueError,
            RuntimeError,
        ) as error:
            if isinstance(error, GeneratedLayoutError):
                raise
            return GeneratedLayouts(
                "semantic_abstention", witness_count=len(check.programs), reason=str(error)
            )
    records = []
    for roles, positions, fixed in assignments:
        try:
            record = tensorize_synthesis_program_product(
                record_id=product_id,
                program_id=family,
                canonical_product_smiles=canonical,
                atom_roles=roles,
                atom_core_positions=[f"{family}:{p}" if p else "exterior" for p in positions],
                program_depth=depth,
                vocabulary=vocabulary,
                atom_vocabulary=atoms,
                fixed_atom_indices=fixed,
            )
        except SynthesisProgramGraphError as error:
            return GeneratedLayouts(
                "representation_abstention", witness_count=len(check.programs), reason=str(error)
            )
        records.append(record)
    return GeneratedLayouts("qualified", tuple(records), check.programs, len(check.programs))


# The pinned sparse flow explicitly discards these four inputs. Keep a separate full sampler
# contract signature so ignored metadata changes cannot be called novel neural context.
SPARSE_FLOW_IGNORED_CONTEXT = frozenset(
    {
        "repeat_group_states",
        "component_position_states",
        "component_instance_states",
        "role_morphology_states",
    }
)


def layout_signatures(
    record: SynthesisProgramGraphRecord, *, maximum_closures: int
) -> dict[str, str]:
    """Hash unpadded, target-masked context for the current sparse flow and sampler."""
    layout = collate_synthesis_program_layouts([record], maximum_closures=maximum_closures)
    payload = {key: value.tolist() for key, value in sorted(layout.items())}

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    return {
        "sampler_context": digest(payload),
        "sparse_flow_context": digest(
            {k: v for k, v in payload.items() if k not in SPARSE_FLOW_IGNORED_CONTEXT}
        ),
    }

"""Verified ordinary-molecule reaction schematics with unchanged side chains capped by R."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions

from forge.assembly.families import RegistryAssemblyAdapter, constitutional_molecule

ROOT = Path(__file__).resolve().parents[4]
RECORDS = ROOT / "results/phase1/compose_lipid_iclr22_reaction_figures_v1/reaction_records.json"


def _step_identity(step):
    return (
        step["reaction_id"],
        step["registry_pin"]["sha256"],
        step["forward_smarts"],
        tuple((item["role"], item["smiles"]) for item in step["reactants"]),
        step["product_smiles"],
        step["stage_index"],
        step["event_within_stage"],
    )


@lru_cache(maxsize=1)
def _registered_programs():
    contents = RECORDS.read_bytes()
    return json.loads(contents)["records"], {
        "path": str(RECORDS.relative_to(ROOT)),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


def _future_templates(step):
    """Read later registered sites from the same exact exemplar's ordered program."""
    records, provenance = _registered_programs()
    future, matches = {}, 0
    for record in records:
        for position, item in enumerate(record["steps"]):
            if _step_identity(item) != _step_identity(step):
                continue
            matches += 1
            for later in record["steps"][position + 1 :]:
                key = (later["registry_pin"]["sha256"], later["reaction_id"])
                future[key] = later
    if not matches:
        raise ValueError("Step is absent from the exact exemplar program; future sites are unknown")
    templates, pins = [], []
    for key, later in sorted(future.items()):
        adapter = RegistryAssemblyAdapter.from_registry(
            ROOT / later["registry_pin"]["path"],
            reaction_id=later["reaction_id"],
            expected_sha256=later["registry_pin"]["sha256"],
        )
        templates.extend(adapter.reaction.forward.GetReactants())
        pins.append({"reaction_id": key[1], "registry_pin": later["registry_pin"]})
    return templates, {"program_records": provenance, "subsequent_reactions": pins}


def _attributes(atom):
    return (
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic(),
        atom.GetTotalNumHs(),
        atom.GetIsotope(),
        atom.GetNumRadicalElectrons(),
    )


def _components(molecule, keep):
    remaining = set(range(molecule.GetNumAtoms())) - keep
    result = []
    while remaining:
        pending = [min(remaining)]
        component = set()
        while pending:
            index = pending.pop()
            if index not in remaining:
                continue
            remaining.remove(index)
            component.add(index)
            pending.extend(a.GetIdx() for a in molecule.GetAtomWithIdx(index).GetNeighbors())
        attachments = [
            (index, neighbor.GetIdx())
            for index in sorted(component)
            for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
            if neighbor.GetIdx() in keep
        ]
        result.append((component, attachments))
    return result


def _context(molecule, core, future_templates):
    """Keep family context and later registered sites, allowing remote tails to be R."""
    if not core:
        raise ValueError("No explicit reactive-core witness")
    keep = set(core)
    context = {
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() != 6
        or atom.IsInRing()
        or atom.GetFormalCharge() != 0
        or atom.GetNumRadicalElectrons() != 0
    }
    future_sites = set()
    for template in future_templates:
        for match in molecule.GetSubstructMatches(template):
            future_sites.update(match)
    context.update(future_sites)
    for index in sorted(context - core):
        paths = [Chem.GetShortestPath(molecule, index, target) for target in sorted(core)]
        keep.update(min(paths, key=lambda path: (len(path), tuple(path))))
    return keep, future_sites


def _capped(molecule, keep, labels):
    """Induced real graph plus one real-order dummy attachment for each removed component."""
    result = Chem.RWMol()
    mapping = {}
    for index in sorted(keep):
        atom = Chem.Atom(molecule.GetAtomWithIdx(index))
        atom.SetAtomMapNum(0)
        mapping[index] = result.AddAtom(atom)
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in keep and b in keep:
            result.AddBond(mapping[a], mapping[b], bond.GetBondType())
    attachments = []
    for component, boundary in _components(molecule, keep):
        if len(boundary) != 1:
            raise ValueError("A hidden substituent is not singly attached")
        outside, inside = boundary[0]
        label = labels[frozenset(component)]
        atom = Chem.Atom(0)
        atom.SetNoImplicit(True)
        atom.SetProp("atomLabel", f"R{label}")
        atom.SetIntProp("_MolFileRLabel", label)
        new_index = result.AddAtom(atom)
        bond = molecule.GetBondBetweenAtoms(outside, inside)
        result.AddBond(mapping[inside], new_index, bond.GetBondType())
        attachments.append(
            {
                "label": f"R{label}",
                "retained_source_index": inside,
                "hidden_source_index": outside,
                "bond_type": str(bond.GetBondType()),
                "shown_dummy_index": new_index,
            }
        )
    shown = result.GetMol()
    Chem.SanitizeMol(shown)
    if any(atom.HasQuery() for atom in shown.GetAtoms()) or any(
        bond.HasQuery() for bond in shown.GetBonds()
    ):
        raise ValueError("Query state leaked into an ordinary-molecule drawing")
    for source, target in mapping.items():
        if _attributes(molecule.GetAtomWithIdx(source)) != _attributes(
            shown.GetAtomWithIdx(target)
        ):
            raise ValueError("Capping changed a retained atom, hydrogen count, or charge")
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in keep and b in keep:
            actual = shown.GetBondBetweenAtoms(mapping[a], mapping[b])
            if actual is None or actual.GetBondType() != bond.GetBondType():
                raise ValueError("Capping changed a retained bond")
    return shown, attachments


def _signature(molecule, component, origins):
    atoms = sorted((origins[i], _attributes(molecule.GetAtomWithIdx(i))) for i in component)
    bonds = sorted(
        (
            tuple(sorted((origins[bond.GetBeginAtomIdx()], origins[bond.GetEndAtomIdx()]))),
            str(bond.GetBondType()),
        )
        for bond in molecule.GetBonds()
        if bond.GetBeginAtomIdx() in component and bond.GetEndAtomIdx() in component
    )
    return atoms, bonds


def _ordinary_reaction(reactants, product):
    reaction = rdChemReactions.ChemicalReaction()
    for molecule in reactants:
        reaction.AddReactantTemplate(molecule)
    reaction.AddProductTemplate(product)
    return reaction


def schematic_reaction(step):
    """Return (ordinary RDKit reaction, JSON audit); never mutate the source step.

    R labels are local to one arrow and represent exactly matched, unchanged source
    atom sets. All heteroatoms, rings, later registered reaction sites, and paths
    connecting them to the reactive core remain visible. A remote hydrocarbon tail
    may contain unchanged unsaturation. Unverified abstraction falls back to full molecules.
    """
    adapter = RegistryAssemblyAdapter.from_registry(
        ROOT / step["registry_pin"]["path"],
        reaction_id=step["reaction_id"],
        expected_sha256=step["registry_pin"]["sha256"],
    )
    raw = next(
        value
        for value in json.loads(adapter.registry_path.read_text())["reactions"]
        if value["reaction_id"] == adapter.reaction_id
    )
    if raw["atom_mapped_reaction_smarts"] != step["forward_smarts"]:
        raise ValueError("Figure step differs from the pinned registry SMARTS")
    if [item["role"] for item in step["reactants"]] != list(adapter.roles):
        raise ValueError("Figure reactant order differs from the registered roles")
    reactants = [Chem.MolFromSmiles(item["smiles"]) for item in step["reactants"]]
    if any(molecule is None for molecule in reactants):
        raise ValueError("Invalid source reactant")
    target = constitutional_molecule(step["product_smiles"])[0]
    with rdBase.BlockLogs():
        outcomes = adapter.reaction.forward.RunReactants(tuple(reactants), maxProducts=4096)
    if len(outcomes) >= 4096:
        raise ValueError("Figure witness search saturated")
    product = None
    for outcome in outcomes:
        if len(outcome) != 1:
            continue
        candidate = Chem.Mol(outcome[0])
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(candidate)
            if constitutional_molecule(Chem.MolToSmiles(candidate))[0] == target:
                product = candidate
                break
        except (ValueError, RuntimeError):
            continue
    if product is None:
        raise ValueError("Exact registered product could not be replayed for the drawing")
    audit = {
        "reaction_id": adapter.reaction_id,
        "registry_pin": step["registry_pin"],
        "actual_reactant_smiles": [item["smiles"] for item in step["reactants"]],
        "actual_product_smiles": target,
        "context_policy": "registry core, all heteroatoms and rings, later registered sites and shortest connecting paths; unchanged remote hydrocarbon tails may be R",
        "r_label_scope": "local to one reaction arrow",
        "checks": {"exact_registry_replay": True, "source_inputs_unchanged": True},
    }
    try:
        molecules = [*reactants, product]
        origins = [
            {i: (role, i) for i in range(m.GetNumAtoms())} for role, m in enumerate(reactants)
        ]
        product_origins = {}
        anchors = [{} for _ in reactants]
        for atom in product.GetAtoms():
            index = atom.GetIdx()
            if atom.HasProp("react_idx") and atom.HasProp("react_atom_idx"):
                role, source = atom.GetIntProp("react_idx"), atom.GetIntProp("react_atom_idx")
                if (
                    not 0 <= role < len(reactants)
                    or not 0 <= source < reactants[role].GetNumAtoms()
                ):
                    raise ValueError("Invalid replay origin")
                product_origins[index] = (role, source)
                if atom.HasProp("old_mapno"):
                    anchors[role][atom.GetIntProp("old_mapno")] = source
            else:
                product_origins[index] = (-1, index)
        if len(set(product_origins.values())) != product.GetNumAtoms():
            raise ValueError("Product source atom origins are not one-to-one")
        origins.append(product_origins)
        core = []
        for role, (molecule, template) in enumerate(
            zip(reactants, adapter.reaction.forward.GetReactants(), strict=True)
        ):
            matches = [
                match
                for match in molecule.GetSubstructMatches(template)
                if all(
                    atom.GetAtomMapNum() not in anchors[role]
                    or match[atom.GetIdx()] == anchors[role][atom.GetAtomMapNum()]
                    for atom in template.GetAtoms()
                )
            ]
            if not matches:
                raise ValueError("No core match agrees with the exact product origin witness")
            core.append(set(min(matches)))
        core_origins = {origins[role][i] for role, indices in enumerate(core) for i in indices}
        core.append(
            {
                atom.GetIdx()
                for atom in product.GetAtoms()
                if product_origins[atom.GetIdx()] in core_origins
                or product_origins[atom.GetIdx()][0] == -1
                or atom.HasProp("old_mapno")
            }
        )
        future_templates, context_provenance = _future_templates(step)
        contexts = [
            _context(molecule, indices, future_templates)
            for molecule, indices in zip(molecules, core, strict=True)
        ]
        keep = [indices for indices, _ in contexts]
        future_sites = [sites for _, sites in contexts]
        # Synchronize retained source identities across the arrow and expand complete
        # multi-attachment components. In particular, never cap a ring into a chain.
        while True:
            before = [set(indices) for indices in keep]
            kept_origins = {o[i] for o, k in zip(origins, keep, strict=True) for i in k}
            for molecule, ids, retained in zip(molecules, origins, keep, strict=True):
                retained.update(i for i, identity in ids.items() if identity in kept_origins)
                for component, boundary in _components(molecule, retained):
                    if len(boundary) != 1:
                        retained.update(component)
            if before == keep:
                break
        left = {}
        for role, molecule in enumerate(reactants):
            for component, boundary in _components(molecule, keep[role]):
                key = frozenset(origins[role][i] for i in component)
                left[key] = (role, component, boundary)
        right = {
            frozenset(product_origins[i] for i in component): (component, boundary)
            for component, boundary in _components(product, keep[-1])
        }
        if left.keys() != right.keys():
            raise ValueError("Hidden source fragments differ across the arrow")
        groups = []
        labels = [{} for _ in molecules]
        for label, key in enumerate(sorted(left, key=lambda value: tuple(sorted(value))), 1):
            role, component, boundary = left[key]
            product_component, product_boundary = right[key]
            if _signature(reactants[role], component, origins[role]) != _signature(
                product, product_component, product_origins
            ):
                raise ValueError("An R group changed atom, charge, hydrogen, or bond state")
            source_bond = reactants[role].GetBondBetweenAtoms(*boundary[0])
            product_bond = product.GetBondBetweenAtoms(*product_boundary[0])
            if (
                source_bond.GetBondType() != product_bond.GetBondType()
                or origins[role][boundary[0][1]] != product_origins[product_boundary[0][1]]
                or origins[role][boundary[0][0]] != product_origins[product_boundary[0][0]]
            ):
                raise ValueError("An R-group attachment changed")
            labels[role][frozenset(component)] = label
            labels[-1][frozenset(product_component)] = label
            groups.append(
                {
                    "label": f"R{label}",
                    "reactant_role": adapter.roles[role],
                    "source_atom_identities": sorted(key),
                    "reactant_indices": sorted(component),
                    "product_indices": sorted(product_component),
                    "attachment_bond_type": str(source_bond.GetBondType()),
                    "identical_atom_hydrogen_charge_bond_state": True,
                }
            )
        shown, boundaries = [], []
        for molecule, retained, label_map in zip(molecules, keep, labels, strict=True):
            display, attachments = _capped(molecule, retained, label_map)
            shown.append(display)
            boundaries.append(attachments)
        audit.update(
            mode="verified_r_group_abstraction",
            retained_atom_indices=[sorted(indices) for indices in keep],
            reactive_core_atom_indices=[sorted(indices) for indices in core],
            future_reaction_site_indices=[sorted(indices) for indices in future_sites],
            future_context_provenance=context_provenance,
            r_groups=groups,
            attachments=boundaries,
            checks={
                **audit["checks"],
                "ordinary_atoms_and_bonds_only": True,
                "retained_atoms_hydrogens_charges_exact": True,
                "retained_bonds_exact": True,
                "all_hidden_fragments_identical": True,
                "all_hidden_attachments_identical": True,
                "all_heteroatoms_and_rings_retained": True,
                "all_matched_subsequent_reaction_sites_retained": all(
                    sites <= retained for sites, retained in zip(future_sites, keep, strict=True)
                ),
            },
        )
    except (ValueError, RuntimeError) as error:
        shown = [Chem.Mol(molecule) for molecule in [*reactants, product]]
        for molecule in shown:
            for atom in molecule.GetAtoms():
                atom.SetAtomMapNum(0)
        audit.update(mode="full_molecule_fallback", fallback_reason=str(error), r_groups=[])
        audit["checks"]["ordinary_atoms_and_bonds_only"] = all(
            not atom.HasQuery() for molecule in shown for atom in molecule.GetAtoms()
        ) and all(not bond.HasQuery() for molecule in shown for bond in molecule.GetBonds())
    audit["shown_reactant_smiles"] = [Chem.MolToSmiles(molecule) for molecule in shown[:-1]]
    audit["shown_product_smiles"] = Chem.MolToSmiles(shown[-1])
    audit["shown_atom_counts"] = [molecule.GetNumAtoms() for molecule in shown]
    audit["actual_formal_charges"] = [
        sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
        for molecule in [*reactants, product]
    ]
    audit["shown_formal_charges"] = [
        sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()) for molecule in shown
    ]
    if audit["actual_formal_charges"] != audit["shown_formal_charges"]:
        raise ValueError("Molecular formal charge changed in the drawing")
    audit["checks"]["all_molecule_formal_charges_exact"] = True
    return _ordinary_reaction(shown[:-1], shown[-1]), audit

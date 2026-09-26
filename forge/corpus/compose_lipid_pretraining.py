"""Fail-closed pre-training checks over the protected COMPOSE TRAIN population.

The related registry transforms are diagnostic leads, not substitutes for missing source
programs. No source admission flag or split is changed, and no training is launched here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, rdBase

from forge.assembly.compose_lipid import ComposeLipidError, role_metadata
from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid import _frozen_guards, verify_compose_lipid
from forge.model._synthesis_sampling import SAMPLING_SOURCE_FILES
from forge.model.defog_feasibility import AtomState
from forge.model.qualified_vocabulary import QualifiedAtomVocabulary
from forge.model.sparse_topology_feasibility import (
    BOND_VALENCE_UNITS,
    sparse_constitutional_roundtrip_exact,
    sparse_roundtrip_exact,
    tensorize_sparse_row,
)
from forge.model.vocabulary import load_atom_vocabulary

CONFIG_SCHEMA = "forge.compose_lipid_pretraining_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_pretraining.v1"


IMPLEMENTATION_SOURCES = (
    "forge/corpus/compose_lipid_pretraining.py",
    "forge/model/qualified_vocabulary.py",
    *SAMPLING_SOURCE_FILES,
    "forge/assembly/families.py",
    "forge/assembly/registry.py",
    "forge/assembly/program.py",
    "forge/chemistry/reactive_sites.py",
    "forge/model/sparse_topology_feasibility.py",
    "forge/model/defog_feasibility.py",
    "forge/core/hashing.py",
)


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _pin(repo: Path, path: Path) -> dict:
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def _rows(db: sqlite3.Connection, split: str):
    for payload in db.execute(
        "SELECT t.payload FROM assignments a JOIN targets t USING(target_id) "
        "WHERE a.forge_split=? ORDER BY a.family,a.target_id",
        (split,),
    ):
        yield json.loads(payload[0])


def atom_state(atom: Chem.Atom) -> AtomState:
    return AtomState(
        atom.GetSymbol(), atom.GetFormalCharge(), atom.GetIsAromatic(), atom.GetNumExplicitHs()
    )


def component_labels(row: dict, program: dict) -> tuple[dict[str, str], bool]:
    """Opaque source labels, scoped by role; these are NOT chemical identity hashes."""
    roles = role_metadata(row, program)
    labels = {
        role: json.dumps(value["metadata"], sort_keys=True, separators=(",", ":"))
        for role, value in roles["roles"].items()
        if value["metadata"]
    }
    return labels, not roles["missing_role_metadata"]


def audit_split_labels(db: sqlite3.Connection, catalogue: dict) -> dict:
    """Inspect labels only in held-out rows: no graph parsing or held-out component fitting."""
    labels = {name: defaultdict(set) for name in ("provider_train", "protected_train")}
    combinations = {name: defaultdict(set) for name in labels}
    for payload, assignment, fold in db.execute(
        "SELECT t.payload,a.payload,a.forge_split FROM assignments a "
        "JOIN targets t USING(target_id) WHERE a.provider_split='train'"
    ):
        row, assignment = json.loads(payload), json.loads(assignment)
        family = row["primary_family"]
        row_labels, _ = component_labels(row, catalogue[family])
        for population in ("provider_train", "protected_train"):
            if population == "protected_train" and fold != "train":
                continue
            combinations[population][family].add(assignment["combination_signature"])
            for role, label in row_labels.items():
                labels[population][(family, role)].add(label)
    counts = {name: defaultdict(Counter) for name in labels}
    malformed = []
    for payload, assignment in db.execute(
        "SELECT t.payload,a.payload FROM assignments a JOIN targets t USING(target_id) "
        "WHERE a.provider_split='test'"
    ):
        row, assignment = json.loads(payload), json.loads(assignment)
        family = row["primary_family"]
        try:
            row_labels, complete = component_labels(row, catalogue[family])
        except ComposeLipidError as exc:
            # Held-out source metadata has not passed the TRAIN-only import validator.
            # Preserve the defect, and never count that row as a successful identity check.
            malformed.append({"target_id": row["target_id"], "reason": str(exc)})
            row_labels, complete = {}, False
        for population in labels:
            all_seen = complete and all(
                label in labels[population][(family, role)] for role, label in row_labels.items()
            )
            for panel in assignment["test_panels"]:
                counter = counts[population][panel]
                counter["rows"] += 1
                counter["complete_role_labels"] += int(complete)
                counter["all_role_labels_train_seen"] += int(all_seen)
                counter["at_least_one_unseen_role_label"] += int(complete and not all_seen)
                counter["combination_signature_train_seen"] += int(
                    assignment["combination_signature"] in combinations[population][family]
                )
    return {
        "populations": {key: dict(value) for key, value in counts.items()},
        "evidence_basis": "source_metadata_labels_only_not_chemical_identity",
        "heldout_graphs_parsed": False,
        "precursor_identity_holdout_qualified": False,
        "malformed_heldout_metadata": malformed,
    }


def probe_reconstruction(row: dict, program: dict, adapter, binding: dict, guards) -> dict:
    """Bounded fixed-arity reconstruction under unchanged registry role/site policies."""
    if adapter is None:
        return {"status": "missing_executable_program"}
    metadata = role_metadata(row, program)
    if metadata["missing_role_metadata"] or metadata["missing_variable_metadata"]:
        return {"status": "source_role_or_multiplicity_metadata_missing"}
    # Repeated chemistry needs an explicit staged adapter; one local cut is not its program.
    variables = metadata["variable_multiplicity_metadata"]
    if any(type(value) is not int or value != 1 for value in variables.values()):
        return {"status": "repeated_program_adapter_required"}
    try:
        outcomes = adapter.decompose(row["constitution"], maximum_outcomes=binding["limit"])
    except LibraryAssemblyError as exc:
        return {"status": "bounded_search_abstention", "reason": str(exc)}
    if not outcomes:
        return {"status": "no_exact_related_transform_reconstruction"}
    if len(outcomes) != 1:
        return {"status": "ambiguous_related_transform", "exact_decompositions": len(outcomes)}
    components = dict(outcomes[0].components)
    replay = adapter.forward_products(components, maximum_outcomes=binding["limit"])
    if replay.saturated or replay.products != (row["constitution"],):
        return {"status": "ambiguous_forward_site_class"}
    labels, _ = component_labels(row, program)
    mapped = []
    for role, smiles in components.items():
        source_role = binding["registry_to_source_roles"][role]
        identity = hashlib.sha256(smiles.encode()).hexdigest()
        mapped.append(
            {
                "role": source_role,
                "source_label": labels[source_role],
                "canonical_smiles": smiles,
                "constitution_id": identity,
                "historical_fold": guards.effective(identity),
            }
        )
    protected = any(c["historical_fold"] in ("calibration", "heldout") for c in mapped)
    return {
        "status": (
            "exact_related_transform_protected_precursor"
            if protected
            else "exact_related_transform"
        ),
        "components": mapped,
        "evidence_basis": "computed_transform_consistency",
        "source_program_equivalence_qualified": False,
    }


def run_pretraining_checks(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path = (repo / config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ComposeLipidError("unsupported pretraining configuration")
    started = time.monotonic()
    import_path = resolve_pin(config["import_result"], repo, label="import result")
    imported = verify_compose_lipid(repo, import_path)
    import_config = json.loads((repo / imported["config"]["path"]).read_text())
    paths = {key: repo / pin["path"] for key, pin in imported["inputs"].items()}
    catalogue = json.loads(
        (repo / imported["artifacts"]["program_catalogue.json"]["path"]).read_text()
    )
    if set(config["related_transforms"]) - set(catalogue):
        raise ComposeLipidError("unknown family in related transform bindings")
    guards = _frozen_guards(import_config["historical_identity_guards"], paths)
    old_states = load_atom_vocabulary(paths["atom_vocabulary"])
    adapters = {}
    for family, binding in config["related_transforms"].items():
        if binding["reaction_id"] not in catalogue[family]["related_registry_reactions"]:
            raise ComposeLipidError("transform is absent from pinned import catalogue")
        adapter = RegistryAssemblyAdapter.from_registry(
            paths[binding["registry_input"]],
            reaction_id=binding["reaction_id"],
            expected_sha256=imported["inputs"][binding["registry_input"]]["sha256"],
        )
        role_map = binding["registry_to_source_roles"]
        if set(role_map) != set(adapter.roles) or set(role_map.values()) != set(
            catalogue[family]["source_definition"]["roles"]
        ):
            raise ComposeLipidError(f"{family}: related-transform role mapping mismatch")
        if type(binding["limit"]) is not int or binding["limit"] < 2:
            raise ComposeLipidError("invalid reaction search bound")
        adapters[family] = adapter
    output = (repo / output_dir).resolve()
    output.relative_to(repo)
    if output.exists():
        raise ComposeLipidError("pretraining output already exists; choose a fresh version")
    output.parent.mkdir(parents=True, exist_ok=True)
    db_path = repo / imported["artifacts"]["corpus.sqlite"]["path"]
    db = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".pretraining-", dir=output.parent) as temporary:
            work = Path(temporary) / "output"
            work.mkdir()
            states = Counter()
            bonds = Counter()
            family_counts = Counter()
            maxima = {"atoms": 0, "closures": 0}
            invalid = []
            for row in _rows(db, "train"):
                family_counts[row["primary_family"]] += 1
                mol = Chem.MolFromSmiles(row["constitution"])
                if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                    invalid.append(row["target_id"])
                    continue
                states.update(atom_state(atom) for atom in mol.GetAtoms())
                bonds.update(str(bond.GetBondType()) for bond in mol.GetBonds())
                maxima["atoms"] = max(maxima["atoms"], mol.GetNumAtoms())
                maxima["closures"] = max(
                    maxima["closures"], mol.GetNumBonds() - mol.GetNumAtoms() + 1
                )
            if sum(family_counts.values()) != config["expected_train_rows"] or invalid:
                raise ComposeLipidError(
                    f"TRAIN count/graph inventory failed; invalid={invalid[:5]}"
                )
            ordered = tuple(sorted(states, key=AtomState.key))
            extensions = tuple(
                symbol
                for symbol in config["neutral_monovalent_extensions"]
                if any(state.symbol == symbol for state in ordered)
            )
            vocabulary = QualifiedAtomVocabulary(ordered, extensions)
            capacities = vocabulary.capacities()
            mapping = {state: index for index, state in enumerate(vocabulary)}
            print(
                f"Inventory: {sum(family_counts.values())} TRAIN graphs, {len(states)} atom states",
                flush=True,
            )
            split_audit = audit_split_labels(db, catalogue)
            family_checks = {family: Counter() for family in catalogue}
            label_identities = defaultdict(set)
            bond_units = BOND_VALENCE_UNITS.tolist()
            with (work / "train_checks.jsonl.gz").open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as ledger:
                    for number, row in enumerate(_rows(db, "train"), 1):
                        family = row["primary_family"]
                        checks = family_checks[family]
                        checks["rows"] += 1
                        tensor = tensorize_sparse_row(
                            {
                                "r0_structure_id": row["target_id"],
                                "canonical_isomeric_smiles": row["constitution"],
                            },
                            mapping,
                            preserve_aromaticity=True,
                        )
                        exact = sparse_roundtrip_exact(
                            tensor
                        ) and sparse_constitutional_roundtrip_exact(tensor, vocabulary)
                        checks["exact_graph_roundtrips"] += int(exact)
                        used = [0] * tensor.node_count
                        for child in range(1, tensor.node_count):
                            units = int(bond_units[tensor.parent_bonds[child]])
                            used[child] += units
                            used[tensor.parents[child]] += units
                        for left, right, bond in zip(
                            tensor.closure_left,
                            tensor.closure_right,
                            tensor.closure_bonds,
                            strict=True,
                        ):
                            units = int(bond_units[bond])
                            used[left] += units
                            used[right] += units
                        valence_ok = all(
                            u <= capacities[s]
                            for u, s in zip(used, tensor.node_states, strict=True)
                        )
                        checks["valence_policy_covered"] += int(valence_ok)
                        checks["rows_with_new_atom_states"] += int(
                            any(vocabulary[s] not in old_states for s in tensor.node_states)
                        )
                        replay = probe_reconstruction(
                            row,
                            catalogue[family],
                            adapters.get(family),
                            config["related_transforms"].get(family, {}),
                            guards,
                        )
                        checks[replay["status"]] += 1
                        for component in replay.get("components", []):
                            label_identities[
                                (family, component["role"], component["source_label"])
                            ].add(component["constitution_id"])
                        record = {
                            "target_id": row["target_id"],
                            "family": family,
                            "graph_roundtrip_exact": exact,
                            "valence_policy_covered": valence_ok,
                            "related_transform_probe": replay,
                            "training_admitted": False,
                        }
                        ledger.write(
                            (
                                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                            ).encode()
                        )
                        if number % 10000 == 0:
                            print(f"Checked {number} protected TRAIN graphs", flush=True)
            conflicts = [
                {
                    "family": key[0],
                    "role": key[1],
                    "source_label": key[2],
                    "constitution_ids": sorted(value),
                }
                for key, value in sorted(label_identities.items())
                if len(value) > 1
            ]
            graph_pass = all(
                v["rows"] == v["exact_graph_roundtrips"] == v["valence_policy_covered"]
                for v in family_checks.values()
            )
            vocab_artifact = {
                "schema_version": "forge.compose_lipid_atom_vocabulary.v1",
                "status": "pass" if graph_pass else "fail",
                "scope": "protected_TRAIN_graph_representation_only_not_training_admission",
                "import_result": config["import_result"],
                "config": _pin(repo, config_path),
                "fit_split": "train",
                "training_rows": sum(family_counts.values()),
                "atom_vocabulary": [
                    {"index": i, **asdict(state), "train_atoms": states[state]}
                    for i, state in enumerate(vocabulary)
                ],
                "neutral_monovalent_extensions": extensions,
                "heavy_atom_valence_capacities": capacities,
                "bond_counts": dict(bonds),
                "maximum_observed": maxima,
                "rdkit_version": rdBase.rdkitVersion,
                "checkpoint_compatibility": "new_vocabulary_requires_new_model_configuration",
            }
            _dump(work / "atom_vocabulary.json", vocab_artifact)
            _dump(work / "split_label_audit.json", split_audit)
            _dump(work / "component_label_conflicts.json", conflicts)
            implementation = IMPLEMENTATION_SOURCES
            result = {
                "schema_version": RESULT_SCHEMA,
                "status": "training_unqualified",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": _pin(repo, config_path),
                "import_result": config["import_result"],
                "implementation": {p: _pin(repo, repo / p) for p in implementation},
                "artifacts": {
                    p.name: {
                        "path": (output / p.name).relative_to(repo).as_posix(),
                        "sha256": str(sha256_file(p)),
                    }
                    for p in sorted(work.iterdir())
                },
                "by_family": family_checks,
                "train_rows": sum(family_counts.values()),
                "all_train_graph_roundtrips_and_valences_pass": graph_pass,
                "source_label_structure_conflicts": len(conflicts),
                "source_programs_qualified": 0,
                "training_calls": 0,
                "generation_calls": 0,
                "heldout_graphs_parsed": False,
                "random_sampling_used": False,
                "blockers": [
                    "v8_executable_program_registry_not_published_at_named_source_branch",
                    "related_transform_replays_do_not_establish_source_program_equivalence",
                    "precursor_structure_guards_incomplete_for_unreconstructed_programs",
                ],
                "duration_seconds": time.monotonic() - started,
            }
            _dump(work / "result.json", result)
            os.rename(work, output)
    finally:
        db.close()
    return result


def verify_pretraining_checks(repo_root: Path, result_path: Path) -> dict:
    repo = repo_root.resolve()
    result = json.loads((repo / result_path).read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "training_unqualified"
    ):
        raise ComposeLipidError("unsupported pretraining receipt; this audit grants no admission")
    config = json.loads(resolve_pin(result["config"], repo, label="config").read_text())
    if config["import_result"] != result["import_result"]:
        raise ComposeLipidError("import receipt substitution")
    verify_compose_lipid(repo, resolve_pin(result["import_result"], repo, label="import result"))
    for group in ("implementation", "artifacts"):
        for name, pin in result[group].items():
            resolve_pin(pin, repo, label=name)
    if result["training_calls"] != 0 or result["source_programs_qualified"] != 0:
        raise ComposeLipidError(
            "pretraining audit cannot claim model training or program admission"
        )
    return result

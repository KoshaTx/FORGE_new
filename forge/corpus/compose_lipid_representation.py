"""Full-TRAIN Kekule representation gate, retaining unchanged chemical valence limits."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from rdkit import Chem, rdBase

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_pretraining import _dump, _pin, _rows, verify_pretraining_checks
from forge.model.defog_feasibility import AtomState
from forge.model.qualified_vocabulary import QualifiedAtomVocabulary
from forge.model.sparse_topology_feasibility import (
    BOND_VALENCE_UNITS,
    SparseTopologyFlowProbe,
    collate_sparse_records,
    sparse_constitutional_roundtrip_exact,
    sparse_roundtrip_exact,
    tensorize_sparse_row,
)


def kekule_states(smiles: str) -> tuple[AtomState, ...]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ComposeLipidError("Kekule input must be a valid connected graph")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    # Match the existing tensorizer's constitutional Kekule mode exactly. Recovery below
    # must re-perceive the original aromaticity and implicit hydrogen count for every row.
    return tuple(AtomState(a.GetSymbol(), a.GetFormalCharge(), False, 0) for a in mol.GetAtoms())


def capacity_covered(graph, vocabulary: QualifiedAtomVocabulary) -> bool:
    used = [0] * graph.node_count
    units = BOND_VALENCE_UNITS.tolist()
    for child in range(1, graph.node_count):
        value = int(units[graph.parent_bonds[child]])
        used[child] += value
        used[graph.parents[child]] += value
    for left, right, bond in zip(
        graph.closure_left, graph.closure_right, graph.closure_bonds, strict=True
    ):
        used[left] += int(units[bond])
        used[right] += int(units[bond])
    capacities = vocabulary.capacities()
    return all(
        value <= capacities[state] for value, state in zip(used, graph.node_states, strict=True)
    )


def _model_smoke(records: list, vocabulary: QualifiedAtomVocabulary, maximum: dict) -> dict:
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = SparseTopologyFlowProbe(
                len(vocabulary),
                3,
                hidden_dim=16,
                layers=1,
                maximum_closures=max(1, maximum["closures"]),
                dropout=0.0,
            ).eval()
            batch = collate_sparse_records(records, maximum["atoms"], max(1, maximum["closures"]))
            with torch.no_grad():
                prediction = model(
                    **{
                        k: batch[k]
                        for k in (
                            "nodes",
                            "parents",
                            "parent_bonds",
                            "closure_left",
                            "closure_right",
                            "node_mask",
                            "child_mask",
                        )
                    },
                    t=torch.full((len(records),), 0.5),
                )
            finite = all(bool(torch.isfinite(value).all()) for value in prediction.values())
    finally:
        torch.set_num_threads(threads)
    return {
        "finite": finite,
        "records": len(records),
        "device": "cpu",
        "precision": "float32",
        "seed": 0,
        "training_steps": 0,
        "output_shapes": {k: list(v.shape) for k, v in prediction.items()},
    }


def run_representation_check(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path = (repo / config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "forge.compose_lipid_representation_config.v1":
        raise ComposeLipidError("unsupported representation configuration")
    audit_path = resolve_pin(config["pretraining_audit"], repo, label="pretraining audit")
    audit = verify_pretraining_checks(repo, audit_path)
    imported = json.loads((repo / audit["import_result"]["path"]).read_text())
    db_path = repo / imported["artifacts"]["corpus.sqlite"]["path"]
    output = (repo / output_dir).resolve()
    output.relative_to(repo)
    if output.exists():
        raise ComposeLipidError("representation output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    try:
        counts = Counter()
        for row in _rows(db, "train"):
            counts.update(kekule_states(row["constitution"]))
        states = tuple(sorted(counts, key=AtomState.key))
        extensions = tuple(
            s for s in config["neutral_monovalent_extensions"] if s in {a.symbol for a in states}
        )
        vocabulary = QualifiedAtomVocabulary(states, extensions)
        mapping = {s: i for i, s in enumerate(vocabulary)}
        families = {f: Counter() for f in audit["by_family"]}
        maximum = {"atoms": 0, "closures": 0}
        failures = []
        smoke = {}
        for number, row in enumerate(_rows(db, "train"), 1):
            graph = tensorize_sparse_row(
                {
                    "r0_structure_id": row["target_id"],
                    "canonical_isomeric_smiles": row["constitution"],
                },
                mapping,
                preserve_aromaticity=False,
            )
            exact = sparse_roundtrip_exact(graph) and sparse_constitutional_roundtrip_exact(
                graph, vocabulary
            )
            covered = capacity_covered(graph, vocabulary)
            bonds = list(graph.parent_bonds[1:]) + list(graph.closure_bonds)
            bond_support = all(0 <= b < 3 for b in bonds)
            family = families[row["primary_family"]]
            family.update(
                {
                    "rows": 1,
                    "exact_roundtrips": int(exact),
                    "valence_covered": int(covered),
                    "bond_support_covered": int(bond_support),
                }
            )
            if not (exact and covered and bond_support):
                failures.append(
                    {
                        "target_id": row["target_id"],
                        "exact": exact,
                        "valence_covered": covered,
                        "bond_support_covered": bond_support,
                    }
                )
            smoke.setdefault(row["primary_family"], graph)
            for key, value in (("atoms", graph.node_count), ("closures", graph.closure_count)):
                if value > maximum[key]:
                    maximum[key] = value
                    smoke[f"maximum_{key}"] = graph
            if "Br" in row["constitution"]:
                smoke.setdefault("bromine", graph)
            if number % 25000 == 0:
                print(f"Kekule gate: {number} protected TRAIN graphs", flush=True)
        if number != audit["train_rows"]:
            raise ComposeLipidError("representation population differs from pretraining audit")
        model_smoke = _model_smoke(list(smoke.values()), vocabulary, maximum)
        passed = not failures and model_smoke["finite"]
        with tempfile.TemporaryDirectory(prefix=".representation-", dir=output.parent) as tmp:
            work = Path(tmp) / "output"
            work.mkdir()
            _dump(
                work / "atom_vocabulary.json",
                {
                    "schema_version": "forge.compose_lipid_kekule_vocabulary.v1",
                    "status": "pass" if passed else "fail",
                    "scope": "representation_only",
                    "pretraining_audit": config["pretraining_audit"],
                    "config": _pin(repo, config_path),
                    "fit_split": "protected_train_pending_precursor_qualification",
                    "preserve_aromaticity": False,
                    "active_sparse_bond_indices": [0, 1, 2],
                    "atom_vocabulary": [
                        {"index": i, **asdict(s), "train_atoms": counts[s]}
                        for i, s in enumerate(states)
                    ],
                    "neutral_monovalent_extensions": extensions,
                    "heavy_atom_valence_capacities": vocabulary.capacities(),
                    "maximum_observed": maximum,
                    "rdkit_version": rdBase.rdkitVersion,
                    "training_admitted": False,
                },
            )
            _dump(work / "failures.json", failures)
            implementation = (
                "forge/corpus/compose_lipid_representation.py",
                "forge/model/qualified_vocabulary.py",
                "forge/model/sparse_topology_feasibility.py",
                "forge/model/defog_feasibility.py",
            )
            result = {
                "schema_version": "forge.compose_lipid_representation.v1",
                "status": (
                    "representation_pass_training_unqualified"
                    if passed
                    else "representation_failed"
                ),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": _pin(repo, config_path),
                "pretraining_audit": config["pretraining_audit"],
                "implementation": {p: _pin(repo, repo / p) for p in implementation},
                "artifacts": {
                    p.name: {
                        "path": (output / p.name).relative_to(repo).as_posix(),
                        "sha256": str(sha256_file(p)),
                    }
                    for p in sorted(work.iterdir())
                },
                "by_family": families,
                "maximum_observed": maximum,
                "train_rows": number,
                "model_smoke": model_smoke,
                "training_admitted": False,
                "training_calls": 0,
                "generation_calls": 0,
                "heldout_graphs_parsed": False,
                "seed": 0,
                "random_sampling_used": False,
                "remaining_gates": audit["blockers"],
                "refit_on_final_precursor_protected_train_required": True,
            }
            _dump(work / "result.json", result)
            os.rename(work, output)
    finally:
        db.close()
    return result

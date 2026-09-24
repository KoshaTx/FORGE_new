"""Deterministic sparse noise marginals under the admitted formal-family graph measure."""

from __future__ import annotations

import json
import math
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_training_data import ComposeLipidTrainingData

SCHEMA = "forge.compose_lipid_noise_marginals.v1"
POLICY = "normalized_graph_probability_weighted_sparse_token_counts"


def compile_noise_marginals(
    repo: Path,
    output: Path,
    *,
    population: Mapping[str, str],
    verification: Mapping[str, str],
    measure: Mapping[str, str],
    admission: Mapping[str, str],
    maximum_cached_shards: int = 4,
    mapped_cache: Mapping[str, str] | None = None,
) -> Path:
    """Fit global atom/bond sources after admission; publish nothing on failure.

    For token state k, accumulate sum_g p(g) count_g(k), then normalize by
    the weighted token total. Each parent bond except the root sentinel and
    each closure bond contributes exactly once. No padding, nonedges, source
    role reweighting, smoothing, size filter or extra training draw is applied.
    As for the existing sparse-marginal fitter, every declared state must occur.
    """
    repo, output = repo.resolve(), output.resolve()
    output.relative_to(repo)
    if output.exists():
        raise FileExistsError(output)
    own_path = repo / "forge/corpus/compose_lipid_noise_marginals.py"
    if str(sha256_file(own_path)) != str(sha256_file(Path(__file__))):
        raise ValueError("Recorded noise compiler differs from the executing code")
    data_inputs = {
        "population": dict(population),
        "verification": dict(verification),
        "measure": dict(measure),
        "admission": dict(admission),
    }
    if mapped_cache is not None:
        data_inputs["mapped_cache"] = dict(mapped_cache)
    implementation = {
        name: pin(repo, repo / name)
        for name in (
            "forge/corpus/compose_lipid_noise_marginals.py",
            "forge/corpus/compose_lipid_training_data.py",
            "forge/corpus/compose_lipid_training_measure.py",
            "forge/corpus/qualified_program_cache.py",
            "forge/model/sparse_topology_feasibility.py",
        )
    }
    with ComposeLipidTrainingData(
        repo, **data_inputs, maximum_cached_shards=maximum_cached_shards
    ) as data:
        node_counts = np.zeros(len(data.atom_vocabulary), dtype=np.float64)
        bond_counts = np.zeros(3, dtype=np.float64)  # Constitutional single/double/triple states.
        families, probabilities = Counter(), {}
        count, maximum_atoms = 0, 0
        for example, probability in data.iter_weighted_examples():
            graph = example.record.graph
            node_counts += probability * np.bincount(graph.node_states, minlength=len(node_counts))
            bond_counts += probability * np.bincount(
                np.concatenate((graph.parent_bonds[1:], graph.closure_bonds)),
                minlength=len(bond_counts),
            )
            families[example.family] += 1
            probabilities[example.family] = probability
            count += 1
            maximum_atoms = max(maximum_atoms, example.record.node_count)
        if count != len(data) or maximum_atoms != data.maximum_heavy_atoms:
            raise ValueError("Noise fitting omitted admitted records or molecular-size support")
        mass = math.fsum(n * probabilities[f] for f, n in families.items())
        if not math.isclose(mass, 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Noise fitting changed the admitted graph measure")
        marginals = {}
        for name, counts in (("node", node_counts), ("bond", bond_counts)):
            if not np.isfinite(counts).all() or np.any(counts <= 0):
                raise ValueError(f"Every declared {name} state must occur in admitted data")
            marginals[name] = (counts / counts.sum()).tolist()
    # Reauthenticate mutable files before publishing the completed numeric artifact.
    for name, value in {**data_inputs, **implementation}.items():
        resolve_pin(value, repo, label="unchanged noise fitting input " + name)
    for name in ("population", "measure"):
        doc = json.loads(resolve_pin(data_inputs[name], repo, label=name).read_text())
        dependencies = doc["artifacts"].values() if name == "population" else (doc["artifact"],)
        for value in dependencies:
            resolve_pin(value, repo, label="unchanged fitted population")
    document = {
        "schema_version": SCHEMA,
        "seed": 0,
        "inputs": {key: data_inputs[key] for key in ("population", "measure")},
        "qualification_inputs": {key: data_inputs[key] for key in ("verification", "admission")},
        "implementation": implementation,
        "policy": POLICY,
        **marginals,
        "weighted_counts": {"node": node_counts.tolist(), "bond": bond_counts.tolist()},
        "records": count,
        "by_family": dict(families),
        "graph_probability_mass": mass,
        "maximum_heavy_atoms": maximum_atoms,
        "runtime": {
            "device": "cpu",
            "precision": "float64",
            "workers": 0,
            "maximum_cached_shards": maximum_cached_shards,
            "numpy": np.__version__,
        },
        "training_calls": 0,
        "model_allocations": 0,
        "scope": "Noise-source fit only; consumes existing admission and does not issue admission.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".compose-noise-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        dump(staged, document)
        # Exclusive publication refuses to replace another completed compilation.
        output.hardlink_to(staged)
    return output

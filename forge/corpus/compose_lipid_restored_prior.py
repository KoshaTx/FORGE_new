"""Full-population TRAIN core and program/role noise priors for restored training."""

import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from forge.core.hashing import resolve_pin
from forge.core.io import write_json
from forge.corpus.compose_lipid_source_view import pin
from forge.corpus.mapped_program_cache import MappedProgramCache
from forge.model.compose_lipid_core import core_condition


def smooth_sources(counts, *, floor=1e-5, backoff=1.0):
    """Positive global -> program -> role backoff, including unsupported cells."""
    if floor <= 0 or backoff <= 0 or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Invalid source counts or smoothing")
    global_counts = counts.sum((0, 1)) + floor
    global_p = global_counts / global_counts.sum()
    output = np.broadcast_to(global_p, counts.shape).copy()
    for program in range(1, len(counts)):
        parent = counts[program].sum(0) + backoff * global_p
        parent /= parent.sum()
        output[program, 0] = parent
        for role in range(1, counts.shape[1]):
            local = counts[program, role] + backoff * parent
            output[program, role] = local / local.sum()
    return output


def compile_restored_prior(repo: Path, config_path: Path, output: Path):
    """Use every admitted graph with its frozen probability; no CAL/TEST or size cap."""
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    inputs = config["inputs"]
    weights = json.loads(resolve_pin(inputs["measure"], repo, label="measure").read_text())
    weight_path = resolve_pin(weights["artifact"], repo, label="weighted graphs")
    with sqlite3.connect(weight_path.as_uri() + "?mode=ro", uri=True) as db:
        probabilities = np.fromiter(
            (r[0] for r in db.execute("SELECT probability FROM weights ORDER BY record_index")),
            dtype=np.float64,
        )
    bank = defaultdict(lambda: defaultdict(float))
    start = time.monotonic()
    maximum_children = 0
    with MappedProgramCache(repo, manifest=inputs["mapped_cache"]) as cache:
        if len(probabilities) != len(cache) or not np.isclose(probabilities.sum(), 1):
            raise ValueError("Mapped population and admitted probabilities differ")
        programs, roles = len(cache.vocabulary.program_states), len(cache.vocabulary.role_states)
        nodes = np.zeros((programs, roles, len(cache.atom_vocabulary)), dtype=np.float64)
        bonds = np.zeros((programs, roles, 3), dtype=np.float64)
        arrays = cache.arrays
        for begin in range(0, len(cache), 4096):
            end = min(begin + 4096, len(cache))
            offsets = arrays["node_offsets"][begin : end + 1]
            a, b = offsets[0], offsets[-1]
            lengths = np.diff(offsets)
            program = np.repeat(arrays["program_states"][begin:end], lengths).astype(np.int64)
            weight = np.repeat(probabilities[begin:end], lengths)
            role = arrays["role_states"][a:b].astype(np.int64)
            core = arrays["core_position_states"][a:b] > 1
            parent = arrays["parents"][a:b] + np.repeat(offsets[:-1] - a, lengths)
            variable_atom = ~core & ~arrays["fixed_atom_mask"][a:b].astype(bool)
            variable_parent = ~(core & core[parent]) & ~arrays["fixed_parent_bond_mask"][
                a:b
            ].astype(bool)
            variable_parent[offsets[:-1] - a] = False
            for target, values, mask in (
                (nodes, arrays["node_states"][a:b], variable_atom),
                (bonds, arrays["parent_bonds"][a:b], variable_parent),
            ):
                codes = (program * roles + role) * target.shape[-1] + values
                target += np.bincount(
                    codes[mask], weights=weight[mask], minlength=target.size
                ).reshape(target.shape)
            for index in range(begin, end):
                example = cache.record(index)
                record = example.record
                key, value = core_condition(record)
                bank[json.dumps(key, separators=(",", ":"))][
                    json.dumps(value, separators=(",", ":"))
                ] += float(probabilities[index])
                graph = record.graph
                c = record.core_position_states > 1
                cs = (
                    ~(c[graph.closure_left] & c[graph.closure_right])
                    & ~record.fixed_closure_bond_mask
                )
                bonds[record.program_state, 0] += probabilities[index] * np.bincount(
                    graph.closure_bonds[cs], minlength=3
                )
                maximum_children = max(
                    maximum_children,
                    int(np.bincount(graph.parents[1:], minlength=record.node_count).max()),
                )
            if begin % (4096 * 32) == 0:
                print(f"Fitted {end}/{len(cache)} qualified TRAIN records", flush=True)
        result = {
            "schema_version": "forge.compose_lipid_noise_marginals.v1",
            "inputs": {k: inputs[k] for k in ("population", "measure")},
            "mapped_cache": inputs["mapped_cache"],
            "node": smooth_sources(nodes).tolist(),
            "bond": smooth_sources(bonds).tolist(),
            "core_bank": {
                key: [dict(json.loads(core), mass=mass) for core, mass in sorted(values.items())]
                for key, values in sorted(bank.items())
            },
            "records": len(cache),
            "by_family": cache.metadata["by_family"],
            "maximum_children": maximum_children,
            "graph_probability_mass": float(probabilities.sum()),
            "policy": "qualified_train_typed_core_conditions_and_full_support_program_role_noise",
            "source_floor": 1e-5,
            "backoff_strength": 1.0,
            "seed": 0,
            "seconds": time.monotonic() - start,
            "implementation": {
                str(p.relative_to(repo)): pin(repo, p)
                for p in (Path(__file__).resolve(), repo / "forge/model/compose_lipid_core.py")
            },
            "qualification": "No new admission. Typed cores and valence are sampled TRAIN conditions, not universal chemistry laws.",
        }
    write_json(output, result)
    return result

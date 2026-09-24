"""Stream every admitted TRAIN graph into anonymous count and assembly-core laws."""

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
from forge.model.compose_lipid_layout import encoded, summarize_layout


def compile_layout_prior(repo: Path, config_path: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    inputs = config["inputs"]
    measure = json.loads(resolve_pin(inputs["measure"], repo, label="measure").read_text())
    weights = resolve_pin(measure["artifact"], repo, label="graph probabilities")
    sources = [Path(__file__).resolve(), repo / "forge/model/compose_lipid_layout.py"]
    implementation = {str(p.relative_to(repo)): pin(repo, p) for p in sources}
    bank = {}
    started = time.monotonic()
    with (
        sqlite3.connect(weights.as_uri() + "?mode=ro", uri=True) as db,
        MappedProgramCache(repo, manifest=inputs["mapped_cache"]) as cache,
    ):
        mass, records = 0.0, 0
        for index, family, probability in db.execute(
            "SELECT record_index,family,probability FROM weights ORDER BY record_index"
        ):
            if index != records or not np.isfinite(probability) or probability <= 0:
                raise ValueError("Invalid admitted graph order or probability")
            example = cache.record(index)
            if example.family != family:
                raise ValueError("Family changed between weights and mapped source")
            bundle, options, rings = summarize_layout(example)
            key = encoded(bundle)
            if key not in bank:
                bank[key] = dict(
                    mass=0.0, roles=defaultdict(lambda: defaultdict(float)), rings=defaultdict(set)
                )
            item = bank[key]
            item["mass"] += probability
            for role, shape in options.items():
                item["roles"][role][encoded(shape)] += probability
            for role, values in rings.items():
                item["rings"][role].update(values)
            mass += probability
            records += 1
            if records % 50000 == 0:
                print(f"Compiled {records}/{len(cache)} layouts; {len(bank)} bundles", flush=True)
        if records != len(cache) or not np.isclose(mass, 1):
            raise ValueError("Layout fit did not cover the entire admitted measure")
        if implementation != {str(p.relative_to(repo)): pin(repo, p) for p in sources}:
            raise ValueError("Layout compiler source changed during execution")
        result = dict(
            schema_version="forge.compose_lipid_layout_prior.v1",
            records=records,
            graph_probability_mass=mass,
            by_family=cache.metadata["by_family"],
            support=dict(
                maximum_atoms=config["model"]["maximum_heavy_atoms"],
                maximum_closures=config["model"]["maximum_closures"],
            ),
            bundles=[
                dict(
                    bundle=json.loads(key),
                    mass=item["mass"],
                    roles={
                        str(role): [
                            dict(shape=json.loads(shape), mass=w)
                            for shape, w in sorted(values.items())
                        ]
                        for role, values in sorted(item["roles"].items())
                    },
                    rings={
                        str(role): sorted(values) for role, values in sorted(item["rings"].items())
                    },
                )
                for key, item in sorted(bank.items())
            ],
            inputs={k: inputs[k] for k in ("population", "measure", "mapped_cache")},
            implementation=implementation,
            seed=0,
            seconds=time.monotonic() - started,
            policy="TRAIN-weighted assembly bundles; independent role count laws conditioned on full atom/closure support; no exterior chemistry or source identity",
        )
    write_json(output, result)
    return result

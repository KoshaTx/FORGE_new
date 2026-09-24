"""Versioned B5 replay with exact input-pair routing; frozen v1 remains reproducible."""

from __future__ import annotations

import gzip
import json
import os
import platform
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import rdBase

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus import compose_lipid_b5_replay as parent
from forge.corpus.compose_lipid_b5_routing import B5SourcePairProfiles
from forge.corpus.compose_lipid_full_preparation import FullPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

CONFIG_SCHEMA = "forge.compose_lipid_b5_replay_config.v2"
RESULT_SCHEMA = "forge.compose_lipid_b5_replay.v2"
POLICY = {
    **parent.POLICY,
    "profile_selection": "original_inputs_with_exact_source_pair_routing_no_target",
}
IMPLEMENTATION = (
    *parent.IMPLEMENTATION,
    "forge/corpus/compose_lipid_b5_routing.py",
    "forge/corpus/compose_lipid_b5_replay_v2.py",
)
authenticated_originals = parent.authenticated_originals
replay_record = parent.replay_record


def load_contract(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("B5 v2 replay schema or policy changed")
    expected = {
        "parent_config",
        "source_pair_routing",
        "preparation",
        "registry",
        "family_definitions",
        "source_controls",
        "original_tasks",
    }
    if set(config.get("inputs", {})) != expected:
        raise ComposeLipidError("B5 v2 replay input set changed")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    old_config, _, _, programs, controls = parent.load_contract(repo, paths["parent_config"])
    if (
        any(config["inputs"][k] != v for k, v in old_config["inputs"].items())
        or config["search_bounds"] != old_config["search_bounds"]
    ):
        raise ComposeLipidError("B5 v2 routing changed parent source inputs or search bounds")
    overlay = json.loads(paths["source_pair_routing"].read_text())
    if overlay["parent_registry"] != config["inputs"]["registry"]:
        raise ComposeLipidError("B5 v2 routing substituted the parent registry")
    profiles = B5SourcePairProfiles.from_registry(
        repo,
        paths["source_pair_routing"],
        expected_sha256=config["inputs"]["source_pair_routing"]["sha256"],
    )
    return config, paths, profiles, programs, controls


def run_b5_replay(repo_root: Path, config_path: Path, output_dir: Path):
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("B5 replay output must be fresh and inside the repository")
    start = time.monotonic()
    config, paths, profiles, programs, controls = load_contract(repo, config_path)
    implementation = {name: pin(repo, repo / name) for name in IMPLEMENTATION}
    reader = FullPreparationCorpus(repo, paths["preparation"])
    family = profiles.specification["family"]
    prepared = {t: r for t, r in reader.preparation.items() if r["family"] == family}
    originals = authenticated_originals(repo, paths["original_tasks"], config, prepared)
    structures = {r["component_id"]: r["constitution"] for r in rows(reader.precursors)}
    inputs = {**config["inputs"], "precursors": pin(repo, reader.precursors)}
    print(
        f"B5 protected preparation and {len(prepared):,} original tasks authenticated", flush=True
    )
    counts, lanes, failures = Counter(), defaultdict(Counter), Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".b5-replay-", dir=output.parent) as tmp:
        stage = Path(tmp)
        ledger = stage / "replay.jsonl.gz"
        with (
            ledger.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream,
        ):
            for number, item in enumerate(reader.iter_preparation_records(family=family), 1):
                p = item["preparation"]
                original = originals[p["target_id"]]
                result = replay_record(item, original, structures, profiles, programs)
                record = {
                    k: p[k]
                    for k in (
                        "target_id",
                        "family",
                        "constitution_id",
                        "component_instances",
                        "construction_basis",
                    )
                }
                record.update(
                    result=result,
                    training_admitted=False,
                    evidence_basis="computed_transform_consistency",
                    original_task_reference=original["construction"]["provenance"]["task"],
                )
                stream.write((compact(record) + "\n").encode())
                lane = original["primary_metadata"]["design_lane"]
                for counter in (counts, lanes[lane]):
                    counter["rows"] += 1
                    counter["profile_qualified"] += result["binding"]["profile_qualified"]
                    counter["exact"] += result["computed_consistency_pass"]
                    counter[result["disposition"]] += 1
                failures.update(k for k, v in result["binding"]["checks"].items() if not v)
                if number % 250 == 0:
                    print(
                        f"B5 replay {number:,}/{len(prepared):,}; exact={counts['exact']:,}",
                        flush=True,
                    )
        if counts["rows"] != len(prepared):
            raise ComposeLipidError("B5 replay lost eligible records")
        for name, value in implementation.items():
            resolve_pin(value, repo, label=name)
        report = {
            "schema_version": RESULT_SCHEMA,
            "status": "eligible_source_profile_replay_training_unqualified",
            "policy": POLICY,
            "config": pin(repo, config_path),
            "inputs": inputs,
            "implementation": implementation,
            "source_controls": controls,
            "summary": {
                "family": family,
                **dict(counts),
                "by_design_lane": {k: dict(v) for k, v in sorted(lanes.items())},
                "failed_profile_checks": dict(failures),
            },
            "artifact": {
                "path": str((output / ledger.name).relative_to(repo)),
                "sha256": sha256_file(ledger),
            },
            "environment": {
                "python": platform.python_version(),
                "rdkit": rdBase.rdkitVersion,
                "device": "cpu",
                "seed": 0,
            },
            "elapsed_seconds": time.monotonic() - start,
        }
        dump(stage / "result.json", report)
        os.rename(stage, output)
    return report

"""Merge authenticated B5 profile evidence into unchanged full-universe protection."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus import compose_lipid_b5_replay as engine
from forge.corpus.compose_lipid_grouped_readiness import validate_replay_row
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

CONFIG_SCHEMA = "forge.compose_lipid_b5_readiness_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_profiled_staged_readiness.v1"


def validate_profiled_row(row, prepared, expected_binding):
    """Recomputed input qualification and unmodified staged scientific gates are both required."""
    result = row["result"]
    if (
        row.get("training_admitted") is not False
        or row.get("evidence_basis") != "computed_transform_consistency"
        or result.get("experimental_execution_admitted") is not False
        or result.get("binding") != expected_binding
    ):
        raise ComposeLipidError("B5 replay changed its source profile or evidence tier")
    exact = result.get("computed_consistency_pass")
    if type(exact) is not bool:
        raise ComposeLipidError("B5 replay exact flag must be boolean")
    if exact and (
        expected_binding["profile_qualified"] is not True
        or not expected_binding["checks"]
        or not all(v is True for v in expected_binding["checks"].values())
        or result.get("verified_target_constitution_id") != prepared["constitution_id"]
        or result.get("disposition") != "exact_source_profile_transform_consistency"
    ):
        raise ComposeLipidError("Exact B5 replay lacks complete source-profile qualification")
    if not expected_binding["profile_qualified"]:
        if "replay" in result or result["disposition"] != expected_binding["disposition"]:
            raise ComposeLipidError("Unqualified B5 profile reached a molecular replay")
    replay = result.get("replay", {"computed_consistency_pass": False})
    if replay["computed_consistency_pass"] is not exact:
        raise ComposeLipidError("B5 profile and staged exactness disagree")
    # RegistryStagedProgram returns mechanical fields only. Translate this wrapper's
    # already checked disposition into the older validator's wire schema; retain
    # every original forward, inverse and balance payload for its strict checks.
    adapted_replay = {
        **replay,
        "disposition": "exact_computed_reconstruction" if exact else result["disposition"],
        "verified_target_constitution_id": result.get("verified_target_constitution_id"),
    }
    adapted = {**row, "replay": adapted_replay, "experimental_execution_admitted": False}
    return validate_replay_row(adapted, prepared)


def merge_b5_readiness(repo_root: Path, config_path: Path, output_dir: Path):
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("B5 readiness output must be fresh and inside the repository")
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or set(config.get("inputs", {})) != {
        "preparation",
        "previous_report",
        "b5_replay",
    }:
        raise ComposeLipidError("B5 readiness configuration changed")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    previous = json.loads(paths["previous_report"].read_text())
    preparation = json.loads(paths["preparation"].read_text())
    if (
        previous.get("schema_version") != "forge.compose_lipid_grouped_readiness.v1"
        or previous["inputs"]["preparation"] != config["inputs"]["preparation"]
        or previous["training_admitted"] is not False
        or previous["training_calls"] != 0
    ):
        raise ComposeLipidError("Previous readiness scope or population differs")
    resolve_pin(previous["implementation"], repo, label="prior merger")
    resolve_pin(previous["config"], repo, label="prior merge configuration")
    for name, value in previous["inputs"].items():
        resolve_pin(value, repo, label=name)
    prior_ledger = resolve_pin(previous["artifact"], repo, label="previous full readiness ledger")
    full_ledger = resolve_pin(preparation["artifact"], repo, label="full preparation ledger")
    report = json.loads(paths["b5_replay"].read_text())
    if (
        report.get("schema_version") != engine.RESULT_SCHEMA
        or report.get("policy") != engine.POLICY
        or report["inputs"]["preparation"] != config["inputs"]["preparation"]
        or set(report["implementation"]) != set(engine.IMPLEMENTATION)
    ):
        raise ComposeLipidError("B5 replay scope or population differs")
    for name, value in report["implementation"].items():
        if resolve_pin(value, repo, label=name) != (repo / name).resolve():
            raise ComposeLipidError("B5 replay implementation substitution")
    loaded = engine.load_contract(
        repo, resolve_pin(report["config"], repo, label="B5 replay config")
    )
    cfg, replay_paths, profiles, _, controls = loaded
    if {k: v for k, v in report["inputs"].items() if k != "precursors"} != cfg["inputs"] or report[
        "source_controls"
    ] != controls:
        raise ComposeLipidError("B5 replay inputs or source controls differ")
    family = profiles.specification["family"]
    eligible = {
        r["target_id"]: r
        for r in rows(full_ledger)
        if r["family"] == family and r["eligible_for_program_preparation"]
    }
    originals = engine.authenticated_originals(repo, replay_paths["original_tasks"], cfg, eligible)
    structures = {
        r["component_id"]: r["constitution"]
        for r in rows(
            resolve_pin(
                report["inputs"]["precursors"], repo, label="complete global precursor catalog"
            )
        )
    }
    updates, counts, lanes, failures = {}, Counter(), defaultdict(Counter), Counter()
    for row in rows(resolve_pin(report["artifact"], repo, label="B5 replay ledger")):
        target = row["target_id"]
        if target not in eligible or target in updates:
            raise ComposeLipidError("Duplicate, protected or non-B5 replay target")
        original = originals[target]
        binding = profiles.assess(
            eligible[target],
            original["construction"],
            original["primary_metadata"],
            original["original_task"],
            structures,
        )
        if row["original_task_reference"] != original["construction"]["provenance"]["task"]:
            raise ComposeLipidError("B5 replay changed original task reference")
        exact = validate_profiled_row(row, eligible[target], binding)
        result = row["result"]
        updates[target] = {
            "receipt": "b5_replay",
            "program_id": binding.get("program_id"),
            "profile_id": binding.get("profile_id"),
            "disposition": result["disposition"],
            "exact": exact,
        }
        for counter in (counts, lanes[original["primary_metadata"]["design_lane"]]):
            counter["rows"] += 1
            counter["profile_qualified"] += binding["profile_qualified"]
            counter["exact"] += exact
            counter[result["disposition"]] += 1
        failures.update(k for k, v in binding["checks"].items() if not v)
    summary = {
        "family": family,
        **dict(counts),
        "by_design_lane": {k: dict(v) for k, v in sorted(lanes.items())},
        "failed_profile_checks": dict(failures),
    }
    if set(updates) != set(eligible) or summary != report["summary"]:
        raise ComposeLipidError("B5 replay coverage or reported counts do not reproduce")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, dispositions = defaultdict(Counter), Counter()
    total, before_count, newly_exact = 0, 0, 0
    with tempfile.TemporaryDirectory(prefix=".b5-readiness-", dir=output.parent) as tmp:
        stage = Path(tmp)
        ledger = stage / "readiness.jsonl.gz"
        with (
            ledger.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream,
        ):
            for old, prepared in zip_longest(rows(prior_ledger), rows(full_ledger)):
                if (
                    old is None
                    or prepared is None
                    or any(old[k] != prepared[k] for k in ("target_id", "family"))
                    or old["preparation_disposition"] != prepared["disposition"]
                    or old["training_admitted"] is not False
                ):
                    raise ComposeLipidError("Full-universe order or protection changed")
                evidence = [updates[old["target_id"]]] if old["target_id"] in updates else []
                prior_exact = old["passes_completed_recipe_checks"]
                if type(prior_exact) is not bool:
                    raise ComposeLipidError("Prior exactness must be boolean")
                exact = prior_exact or any(e["exact"] for e in evidence)
                if exact and not prepared["eligible_for_program_preparation"]:
                    raise ComposeLipidError("B5 evidence released a protected or unassigned row")
                updated = {
                    **old,
                    "additional_replays": old.get("additional_replays", []) + evidence,
                    "passes_completed_recipe_checks": exact,
                }
                if exact:
                    updated["reconstruction_disposition"] = "exact_computed_reconstruction"
                elif evidence:
                    updated["reconstruction_disposition"] = "unresolved_computed_reconstruction"
                stream.write((compact(updated) + "\n").encode())
                count = counts[prepared["family"]]
                count["source_rows"] += 1
                count["eligible_preparation_rows"] += prepared["eligible_for_program_preparation"]
                count["exact_computed_reconstructions"] += exact
                count["new_exact_computed_reconstructions"] += exact and not prior_exact
                count["pending_eligible_reconstructions"] += (
                    prepared["eligible_for_program_preparation"] and not exact
                )
                dispositions[prepared["disposition"]] += 1
                total += 1
                before_count += prior_exact
                newly_exact += exact and not prior_exact
        prior_summary = previous["summary"]
        if (
            total != prior_summary["source_records"]
            or before_count != prior_summary["exact_reconstructions"]
            or dict(dispositions) != prior_summary["preparation_dispositions"]
        ):
            raise ComposeLipidError("Previous full-universe counts or protection changed")
        result = {
            "schema_version": RESULT_SCHEMA,
            "seed": 0,
            "status": "additional_computed_evidence_training_unqualified",
            "config": pin(repo, config_path),
            "inputs": config["inputs"],
            "implementation": {
                p: pin(repo, repo / p)
                for p in (
                    "forge/corpus/compose_lipid_b5_readiness.py",
                    "forge/corpus/compose_lipid_grouped_readiness.py",
                )
            },
            "replay_summaries": {"b5_replay": report["summary"]},
            "summary": {
                "source_records": total,
                "new_exact_reconstructions": newly_exact,
                "exact_reconstructions": before_count + newly_exact,
                "preparation_dispositions": dict(dispositions),
                "by_family": {k: dict(v) for k, v in sorted(counts.items())},
            },
            "remaining_gates": previous["remaining_gates"],
            "architecture_qualification": "Input-only complete source profiles plus all staged forward, inverse and balance gates; incorrect reported source labels abstain. No exact-execution or training admission.",
            "training_admitted": False,
            "training_calls": 0,
            "artifact": {
                "path": str((output / ledger.name).relative_to(repo)),
                "sha256": sha256_file(ledger),
            },
        }
        dump(stage / "result.json", result)
        os.rename(stage, output)
    return result

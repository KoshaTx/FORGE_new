"""Restartable twelve-library generation using the frozen, admitted completion stages."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from experiments._runtime.historical import resolve_pinned_input
from experiments.phase1.multireaction import (
    combinatorial_connection_reuse as connection,
)
from experiments.phase1.multireaction import (
    combinatorial_graph_reuse as graph,
)
from experiments.phase1.multireaction import (
    combinatorial_graph_reuse_verify as graph_verify,
)
from experiments.phase1.multireaction import (
    combinatorial_occurrence_reuse as occurrence,
)
from experiments.phase1.multireaction import (
    combinatorial_reuse_admission as admission,
)
from experiments.phase1.multireaction import (
    combinatorial_source_core_completion as source_core,
)
from experiments.phase1.multireaction.combinatorial_generation import _pin
from forge.core.hashing import resolve_pin

SCHEMA = "forge.combinatorial_generation_pipeline.v1"
STAGES = ("graph", "admission", "occurrence", "connection", "source_core")
MODULES = dict(zip(STAGES, (graph, admission, occurrence, connection, source_core), strict=True))
SELF = "experiments/phase1/multireaction/combinatorial_generation_pipeline.py"
POLICY = {
    "stages": list(STAGES),
    "family_count": 12,
    "training_calls": 0,
    "remote_compute": False,
    "heldout_access": False,
    "gate_changes": False,
    "source_context": "TRAIN_layout_occurrences_connections_and_typed_core",
    "all_attempts_retained": True,
    "all_qualified_products_exported_without_ranking": True,
    "no_outcome_dependent_budget_extension": True,
    "promotion": False,
}
ACCEPTANCE = {
    "positive_exact_gain_vs_raw": True,
    "positive_exact_gain_vs_connection": True,
    "all_eight_metrics_preserved_in_each_family": True,
    "novel_exact_products_in_each_family": True,
}


def _read(path):
    return json.loads(path.read_text())


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _local(repo, path):
    path = (repo / path).resolve()
    if not path.is_relative_to(repo):
        raise ValueError("pipeline path must remain inside the repository")
    return path


def contract(repo, path):
    config = _read(path)
    if (
        config.get("schema_version") != SCHEMA + "_config"
        or config.get("policy") != POLICY
        or config.get("acceptance") != ACCEPTANCE
        or set(config.get("reference_results", {})) != set(STAGES)
        or config.get("population_mode") not in {"fresh_train", "reference_replay"}
        or config.get("cpu_threads") != 2
    ):
        raise ValueError("pipeline contract differs")
    references, templates, source_paths = {}, {}, {SELF}
    reference_source_pins = {}
    for stage in STAGES:
        reference = _read(resolve_pin(config["reference_results"][stage], repo, label=stage))
        if reference["schema_version"] != MODULES[stage].SCHEMA:
            raise ValueError("pipeline reference has the wrong stage schema: " + stage)
        template = _read(resolve_pin(reference["config"], repo, label=stage + " config"))
        references[stage], templates[stage] = reference, template
        for p in reference["sources"]:
            resolve_pinned_input(repo, p["path"], p["sha256"])
            source_paths.add(p["path"])
            if p["path"] in reference_source_pins and reference_source_pins[p["path"]] != p:
                raise ValueError("pipeline reference source versions differ")
            reference_source_pins[p["path"]] = p
    source_pins = config.get("source_pins", [])
    if (
        any(set(p) != {"path", "sha256"} for p in source_pins)
        or [p["path"] for p in source_pins] != sorted(source_paths)
        or any(p != reference_source_pins.get(p["path"], p) for p in source_pins)
    ):
        raise ValueError("pipeline frozen source manifest differs")
    for p in source_pins:
        resolve_pinned_input(repo, p["path"], p["sha256"])
    parent_keys = {"admission": "parent_result", **{s: "baseline_result" for s in STAGES[2:]}}
    for i, stage in enumerate(STAGES[1:], 1):
        if references[stage][parent_keys[stage]] != config["reference_results"][STAGES[i - 1]]:
            raise ValueError("pipeline references do not form the declared stage chain")
    sampling = config["sampling"]
    if (
        set(sampling) != set(templates["graph"]["sampling"])
        or sampling.get("flow_steps") != 32
        or sampling.get("batch_size") != 16
    ):
        raise ValueError("pipeline flow budget differs")
    proposed_graph = {**templates["graph"], "sampling": sampling, "cpu_threads": 2}
    graph.validate(proposed_graph)
    if config["population_mode"] == "reference_replay":
        if sampling != templates["graph"]["sampling"]:
            raise ValueError("reference replay sampling changed")
    elif any(
        sampling[k] == templates["graph"]["sampling"][k] for k in ("layout_seed", "flow_seed")
    ):
        raise ValueError("fresh population reuses a reference seed")
    return config, templates


def stage_config(stage, config, templates, prior):
    template = templates[stage]
    value = {
        "schema_version": template["schema_version"],
        "policy": template["policy"],
        "scientific_question": "Apply the frozen bounded completion pipeline to all twelve families.",
        "stopping_rule": "Fixed requested population; no redraw, tuning or gate change after outcomes.",
    }
    if stage == "graph":
        value.update(
            inputs=template["inputs"],
            sampling=config["sampling"],
            cpu_threads=2,
            use_saved_layouts=(
                template["use_saved_layouts"]
                if config["population_mode"] == "reference_replay"
                else False
            ),
        )
    elif stage == "admission":
        value["parent_result"] = prior
    else:
        value["baseline_result"] = prior
        if stage == "source_core":
            value.update(cpu_threads=2, source_pins=template["source_pins"])
        else:
            value["maximum_canonical_matches"] = template["maximum_canonical_matches"]
    return value


def _verify_stage(stage, repo, path):
    function = graph_verify.verify if stage == "graph" else MODULES[stage].verify
    return function(repo, path)


def _artifact(repo, result, name):
    content = resolve_pin(result["artifacts"][name], repo, label=name).read_text()
    return [json.loads(line) for line in content.splitlines()]


def summarize(repo, config, stage_results):
    raw = [
        r
        for r in _artifact(repo, stage_results["graph"], "attempts.jsonl")
        if r["arm"] == "baseline"
    ]
    rows = _artifact(repo, stage_results["source_core"], "attempts.jsonl")
    final = [r for r in rows if r["arm"] == "graph_reuse"]
    families = sorted(stage_results["source_core"]["per_arm"]["graph_reuse"])
    count = config["sampling"]["attempts_per_family"]
    if len(families) != 12 or len(raw) != len(final) or len(final) != 12 * count:
        raise ValueError("pipeline population accounting differs")
    if any(sum(r["program_id"] == f for r in final) != count for f in families):
        raise ValueError("pipeline family population changed")
    for a, b in zip(raw, final, strict=True):
        if any(
            a[k] != b[k]
            for k in ("sample_index", "program_id", "requested_depth", "layout_record_id")
        ):
            raise ValueError("pipeline changed attempt identity")
        if not a["valid_connected"] or a["assembly"]["status"] == "exact_computed_program":
            if a["canonical_smiles"] != b["canonical_smiles"]:
                raise ValueError("pipeline changed an original invalid or exact output")
    comparison = graph.compare(raw + final, families)
    core = stage_results["source_core"]
    checks = {
        "positive_exact_gain_vs_raw": comparison["total_exact_gain"] > 0,
        "positive_exact_gain_vs_connection": core["total_exact_gain"] > 0,
        "all_eight_metrics_preserved_in_each_family": comparison[
            "all_family_preservation_screen_passed"
        ]
        and all(stage_results[s]["all_family_preservation_screen_passed"] for s in STAGES[1:]),
        "novel_exact_products_in_each_family": all(
            f["unique_novel_exact_products"] > 0
            for f in comparison["per_arm"]["graph_reuse"].values()
        ),
    }
    products = [r for r in final if r["assembly"]["status"] == "exact_computed_program"]
    metrics = {
        "checks": checks,
        "acceptance_passed": all(checks.values()),
        "comparison_vs_raw": comparison,
        "source_core_increment": core["total_exact_gain"],
        "stage_exact_gains": {s: stage_results[s]["total_exact_gain"] for s in STAGES[1:]},
        "qualified_product_count": len(products),
        "generated_attempts": len(raw),
        "source_core_attempts_replayed": core["neural_attempts_replayed"],
        "original_neural_terminals_reproduced": core["frozen_neural_terminal_states_identical"],
    }
    return metrics, products


@contextmanager
def _lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("pipeline output already has an active writer") from exc
        yield


def require_current_execution_sources(repo, config):
    """Historical authentication does not authorize execution with different active code."""
    for pin in config["source_pins"]:
        resolve_pin(pin, repo, label="pipeline execution source")


def run(repo, config_path, output_dir):
    repo = repo.resolve()
    config_path, output = _local(repo, config_path), _local(repo, output_dir)
    config, templates = contract(repo, config_path)
    require_current_execution_sources(repo, config)
    config_pin = _pin(config_path, repo)
    with _lock(output):
        request = output / "request.json"
        if request.exists() and _read(request) != config_pin:
            raise ValueError("existing pipeline belongs to a different request")
        _write(request, config_pin)
        if (output / "result.json").exists():
            verify(repo, output / "result.json")
            return _read(output / "result.json")
        stages, results, prior = {}, {}, None
        for stage in STAGES:
            expected = stage_config(stage, config, templates, prior)
            directory = output / "stages" / stage
            receipt = directory / "complete.json"
            if receipt.exists():
                saved = _read(receipt)
                if _read(resolve_pin(saved["config"], repo, label=stage)) != expected:
                    raise ValueError("saved stage config differs: " + stage)
                path = resolve_pin(saved["result"], repo, label=stage)
                _verify_stage(stage, repo, path)
            else:
                directory.mkdir(parents=True, exist_ok=True)
                attempt = directory / f"attempt_{len(list(directory.glob('attempt_*'))) + 1:04d}"
                attempt.mkdir()
                stage_path = attempt / "config.json"
                _write(stage_path, expected)
                print(json.dumps({"stage": stage, "status": "started"}), flush=True)
                try:
                    MODULES[stage].run(repo, stage_path, attempt / "output")
                    path = attempt / "output" / "result.json"
                    _verify_stage(stage, repo, path)
                except Exception as exc:
                    _write(
                        attempt / "failure.json",
                        {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "config": _pin(stage_path, repo),
                        },
                    )
                    raise
                saved = {"config": _pin(stage_path, repo), "result": _pin(path, repo)}
                _write(receipt, saved)
                print(json.dumps({"stage": stage, "status": "verified"}), flush=True)
            stages[stage], results[stage], prior = saved, _read(path), saved["result"]
            if results[stage]["config"] != saved["config"]:
                raise ValueError("saved stage result belongs to another config: " + stage)
        metrics, products = summarize(repo, config, results)
        products_path = output / "qualified_products.jsonl"
        products_path.write_text(
            "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in products)
        )
        contract(repo, config_path)
        resolve_pin(config_pin, repo, label="final pipeline request")
        result = {
            "schema_version": SCHEMA,
            "status": "numerical_complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config_pin,
            "stages": stages,
            "qualified_products": _pin(products_path, repo),
            **metrics,
            "limits": [
                "Source-derived TRAIN layout, occurrence, connection and typed core context is used. This is not source-independent or heldout generation.",
                "Every original attempt is retained in the stage ledgers. The export includes every exactly reconstructed product without ranking or prospective selection.",
                "Neutral-product and precursor library scopes remain explicit. Exact computed programs do not establish realism, ionizability or L2/L3 route closure.",
                "Generation and completion have additional compute costs. The source-core stage replays the neural trajectories; verification also recomputes them.",
            ],
        }
        _write(output / "result.json", result)
        return result


def verify(repo, result_path):
    repo = repo.resolve()
    path = _local(repo, result_path)
    result = _read(path)
    if result.get("schema_version") != SCHEMA or result.get("status") != "numerical_complete":
        raise ValueError("pipeline result contract differs")
    config, templates = contract(
        repo, resolve_pin(result["config"], repo, label="pipeline request")
    )
    if set(result["stages"]) != set(STAGES):
        raise ValueError("pipeline stage set differs")
    results, prior = {}, None
    for stage in STAGES:
        saved = result["stages"][stage]
        stage_path = resolve_pin(saved["config"], repo, label=stage + " config")
        if _read(stage_path) != stage_config(stage, config, templates, prior):
            raise ValueError("pipeline stage config substituted: " + stage)
        stage_result = resolve_pin(saved["result"], repo, label=stage)
        results[stage] = _read(stage_result)
        if results[stage]["config"] != saved["config"]:
            raise ValueError("pipeline stage result belongs to another config")
        prior = saved["result"]
    # The final stage's verifier recursively authenticates and recomputes every predecessor.
    _verify_stage("source_core", repo, repo / prior["path"])
    metrics, products = summarize(repo, config, results)
    if any(result.get(k) != v for k, v in metrics.items()):
        raise ValueError("pipeline metrics differ from reconstruction")
    stored = resolve_pin(result["qualified_products"], repo, label="qualified products")
    if [json.loads(line) for line in stored.read_text().splitlines()] != products:
        raise ValueError("pipeline product export differs from full qualified population")
    return {
        "status": "verified",
        "result": _pin(path, repo),
        "acceptance_passed": result["acceptance_passed"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.repo_root, args.verify)
    elif args.config and args.output_dir:
        result = run(args.repo_root, args.config, args.output_dir)
    else:
        parser.error("provide --verify or --config and --output-dir")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "acceptance_passed", "qualified_product_count")
                if k in result
            }
        )
    )
    if not result["acceptance_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

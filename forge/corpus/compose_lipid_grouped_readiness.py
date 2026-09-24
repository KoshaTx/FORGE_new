"""Append exact replay evidence to full-universe accounting without admitting training."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin
from forge.corpus import compose_lipid_current_replay as repeated
from forge.corpus import compose_lipid_fixed_profiles as profiles
from forge.corpus import compose_lipid_fixed_replay as fixed
from forge.corpus import compose_lipid_grouped_replay as grouped
from forge.corpus import compose_lipid_staged_replay as staged
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows


def validate_replay_row(row: dict, prepared: dict) -> bool:
    """Authenticate population and identity before admitting even computed evidence."""
    if (
        prepared.get("eligible_for_program_preparation") is not True
        or prepared.get("old_split") != "train"
        or prepared.get("corrected_split") != "train"
        or prepared.get("exclusion_reasons")
        or prepared.get("pending_reasons")
    ):
        raise ComposeLipidError("Replay reached protected or unresolved preparation")
    for key in (
        "target_id",
        "family",
        "constitution_id",
        "component_instances",
        "construction_basis",
    ):
        if row[key] != prepared[key]:
            raise ComposeLipidError(f"Replay changed source {key}")
    if row["training_admitted"] is not False or row["experimental_execution_admitted"] is not False:
        raise ComposeLipidError("Replay promoted unsupported admission")
    result = row["replay"]
    exact = result["computed_consistency_pass"]
    if type(exact) is not bool:
        raise ComposeLipidError("Replay pass must be boolean")
    if exact:
        if "event_searches" in result:
            required_checks = {
                "complete_search",
                "every_declared_event_replayed",
                "unique_each_completed_source_stage",
                "unique_forward_exact",
                "unique_complete_inverse",
                "all_event_inventories_balance",
                "full_element_hydrogen_charge_balance",
                "terminal_constraints",
                "product_constraints",
            }
            stages = result["event_searches"]
            if (
                not stages
                or len(result["forward_layers"]) != len(stages) + 1
                or not result["event_balance_checks"]
                or not all(v is True for v in result["event_balance_checks"])
                or result["inverse"]["complete_search"] is not True
                or result["inverse"]["bound_reasons"]
                or len(result["inverse"]["candidate_components"]) != 1
                or any(len(layer) != 1 for layer in result["forward_layers"])
            ):
                raise ComposeLipidError("Grouped replay lost stage, inverse or inventory evidence")
            for index, stage in enumerate(stages):
                if (
                    stage["stage"] != index + 1
                    or type(stage["declared_events"]) is not int
                    or stage["declared_events"] < 1
                    or len(stage["layers"]) != stage["declared_events"] + 1
                    or not all(stage["layers"])
                    or stage["layers"][0] != result["forward_layers"][index]
                    or stage["layers"][-1] != result["forward_layers"][index + 1]
                ):
                    raise ComposeLipidError("Grouped replay changed a declared source stage")
        elif "stage_balance_checks" in result:
            required_checks = {
                "declared_component_equalities",
                "complete_search",
                "every_stage_replayed",
                "unique_each_forward_stage",
                "unique_forward_exact",
                "unique_complete_inverse",
                "all_stage_inventories_balance",
                "full_element_hydrogen_charge_balance",
                "terminal_constraints",
                "product_constraints",
            }
            if (
                not result["stage_balance_checks"]
                or not all(v is True for v in result["stage_balance_checks"])
                or result["inverse"]["complete_search"] is not True
                or result["inverse"]["bound_reasons"]
                or len(result["inverse"]["candidate_components"]) != 1
                or not all(len(layer) == 1 for layer in result["forward_layers"])
            ):
                raise ComposeLipidError("Staged replay lost stage, inverse or inventory evidence")
        else:
            required_checks = (
                {
                    "complete_search",
                    "declared_event_count_replayed",
                    "unique_forward_exact",
                    "unique_declared_side_inverse",
                    "full_element_hydrogen_charge_balance",
                }
                if "forward_layers" in result
                else {
                    "unique_unfiltered_forward_exact",
                    "unique_unfiltered_inverse_exact",
                    "source_reactive_site_witness",
                    "full_element_hydrogen_charge_balance",
                    "distinct_source_role_handles_retained",
                }
            )
        if not required_checks.issubset(result.get("checks", {})):
            raise ComposeLipidError("Exact replay omitted a required scientific check")
        products = (
            result["forward_layers"][-1]
            if "forward_layers" in result
            else result.get("forward_products", [])
        )
        if (
            not result["checks"]
            or not all(v is True for v in result["checks"].values())
            or result.get("bound_reasons")
            or result.get("verified_target_constitution_id", row["constitution_id"])
            != row["constitution_id"]
            or result["disposition"] != "exact_computed_reconstruction"
            or len(products) != 1
            or hashlib.sha256(products[0].encode()).hexdigest() != row["constitution_id"]
        ):
            raise ComposeLipidError("Exact replay lacks complete checks or exact target identity")
    return exact


def merge_readiness(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("Readiness output must be fresh and inside the repository")
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "forge.compose_lipid_grouped_readiness_config.v1":
        raise ComposeLipidError("Readiness configuration schema differs")
    inputs = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    previous = json.loads(inputs["previous_report"].read_text())
    preparation = json.loads(inputs["preparation"].read_text())
    if previous["inputs"]["preparation"] != config["inputs"]["preparation"]:
        raise ComposeLipidError("Previous accounting uses another preparation")
    if previous["schema_version"] in (
        "forge.compose_lipid_readiness_extension.v1",
        "forge.compose_lipid_readiness_merge.v1",
        "forge.compose_lipid_staged_readiness.v1",
        "forge.compose_lipid_grouped_readiness.v1",
    ):
        totals = {
            "supplied_exact_reconstructions": previous["summary"]["exact_reconstructions"],
            "source_records": previous["summary"]["source_records"],
            "preparation_dispositions": previous["summary"]["preparation_dispositions"],
        }
        old_pin = previous["artifact"]
    else:
        raise ComposeLipidError("Unsupported prior readiness schema")
    old_ledger = resolve_pin(old_pin, repo, label="old readiness")
    full_ledger = resolve_pin(preparation["artifact"], repo, label="preparation ledger")
    allowed = {
        r["target_id"]: r for r in rows(full_ledger) if r["eligible_for_program_preparation"]
    }
    updates, replay_summaries = defaultdict(list), {}
    for name in config["replay_inputs"]:
        result = json.loads(inputs[name].read_text())
        engines = {
            repeated.RESULT_SCHEMA: repeated,
            fixed.RESULT_SCHEMA: fixed,
            profiles.RESULT_SCHEMA: profiles,
            staged.RESULT_SCHEMA: staged,
            grouped.RESULT_SCHEMA: grouped,
        }
        engine = engines.get(result["schema_version"])
        if (
            engine is None
            or result["policy"] != engine.POLICY
            or result["inputs"]["preparation"] != config["inputs"]["preparation"]
            or set(result["implementation"]) != set(engine.IMPLEMENTATION)
        ):
            raise ComposeLipidError("Replay scope or population differs")
        for label, value in result["implementation"].items():
            if resolve_pin(value, repo, label=label) != (repo / label).resolve():
                raise ComposeLipidError("Replay implementation substitution")
        loaded = engine.load_contract(
            repo, resolve_pin(result["config"], repo, label="replay config")
        )
        cfg = loaded[0]
        if cfg["inputs"] != result["inputs"] or loaded[-1] != result["controls"]:
            raise ComposeLipidError("Replay result/config/control substitution")
        ledger = resolve_pin(result["artifact"], repo, label=name + " ledger")
        seen, counted = set(), defaultdict(Counter)
        for row in rows(ledger):
            target, family = row["target_id"], row["family"]
            if target in seen or target not in allowed or family not in cfg["families"]:
                raise ComposeLipidError("Duplicate, protected or out-of-scope replay target")
            seen.add(target)
            identifier = "program_id" if engine in (staged, grouped) else "reaction_id"
            if row[identifier] != cfg["families"][family][identifier]:
                raise ComposeLipidError("Replay reaction substitution")
            exact = validate_replay_row(row, allowed[target])
            if engine is grouped and exact:
                program = loaded[2][family]["program"]
                if row["replay"]["declared_quantities"] != program.quantities or [
                    stage["declared_events"] for stage in row["replay"]["event_searches"]
                ] != [stage["events"] for stage in program.specification["stages"]]:
                    raise ComposeLipidError("Grouped source quantities or event counts differ")

            counted[family]["rows"] += 1
            counted[family][row["replay"]["disposition"]] += 1
            counted[family].update(
                "failed_" + k for k, v in row["replay"].get("checks", {}).items() if not v
            )
            updates[target].append(
                {
                    "receipt": name,
                    identifier: row[identifier],
                    "disposition": row["replay"]["disposition"],
                    "exact": exact,
                }
            )
        expected = {t for t, r in allowed.items() if r["family"] in cfg["families"]}
        if (
            seen != expected
            or {k: dict(v) for k, v in counted.items()} != result["summary"]["by_family"]
            or dict(sum(counted.values(), Counter())) != result["summary"]["totals"]
        ):
            raise ComposeLipidError("Replay family coverage or reported counts differ")
        replay_summaries[name] = result["summary"]
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, dispositions = defaultdict(Counter), Counter()
    before_count, newly_exact, total = 0, 0, 0
    with tempfile.TemporaryDirectory(prefix=".readiness-", dir=output.parent) as temporary:
        stage = Path(temporary)
        ledger = stage / "readiness.jsonl.gz"
        with (
            ledger.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream,
        ):
            for old, prepared in zip_longest(rows(old_ledger), rows(full_ledger)):
                if (
                    old is None
                    or prepared is None
                    or any(old[k] != prepared[k] for k in ("target_id", "family"))
                ):
                    raise ComposeLipidError("Readiness/preparation ordering or universe differs")
                if (
                    old["preparation_disposition"] != prepared["disposition"]
                    or old["training_admitted"]
                ):
                    raise ComposeLipidError("Readiness changed protection or admitted training")
                evidence = updates.get(old["target_id"], [])
                prior_exact = old["passes_completed_recipe_checks"]
                exact = prior_exact or any(v["exact"] for v in evidence)
                if exact and not prepared["eligible_for_program_preparation"]:
                    raise ComposeLipidError("Exact evidence released protected/unassigned row")
                new = {
                    **old,
                    "additional_replays": old.get("additional_replays", []) + evidence,
                    "passes_completed_recipe_checks": exact,
                }
                if exact:
                    new["reconstruction_disposition"] = "exact_computed_reconstruction"
                elif evidence:
                    new["reconstruction_disposition"] = "unresolved_computed_reconstruction"
                stream.write((compact(new) + "\n").encode())
                family = old["family"]
                counts[family]["source_rows"] += 1
                counts[family]["eligible_preparation_rows"] += prepared[
                    "eligible_for_program_preparation"
                ]
                counts[family]["exact_computed_reconstructions"] += exact
                counts[family]["new_exact_computed_reconstructions"] += exact and not prior_exact
                counts[family]["pending_eligible_reconstructions"] += (
                    prepared["eligible_for_program_preparation"] and not exact
                )
                dispositions[prepared["disposition"]] += 1
                before_count += prior_exact
                newly_exact += exact and not prior_exact
                total += 1
        if (
            before_count != totals["supplied_exact_reconstructions"]
            or total != totals["source_records"]
            or dict(dispositions) != totals["preparation_dispositions"]
        ):
            raise ComposeLipidError("Previous counts or full-universe protection changed")
        result = {
            "schema_version": "forge.compose_lipid_grouped_readiness.v1",
            "seed": 0,
            "status": "additional_computed_evidence_training_unqualified",
            "config": pin(repo, config_path),
            "inputs": config["inputs"],
            "implementation": pin(repo, Path(__file__).resolve()),
            "replay_summaries": replay_summaries,
            "summary": {
                "source_records": total,
                "new_exact_reconstructions": newly_exact,
                "exact_reconstructions": before_count + newly_exact,
                "preparation_dispositions": dict(dispositions),
                "by_family": {k: dict(v) for k, v in sorted(counts.items())},
            },
            "remaining_gates": previous["remaining_gates"],
            "architecture_qualification": "Separate source conflicts and scope limits remain in pinned adjudications; computed exactness does not resolve them.",
            "training_admitted": False,
            "training_calls": 0,
            "artifact": {
                "path": str((output / ledger.name).relative_to(repo)),
                "sha256": pin(repo, ledger)["sha256"],
            },
        }
        dump(stage / "result.json", result)
        os.rename(stage, output)
    return result

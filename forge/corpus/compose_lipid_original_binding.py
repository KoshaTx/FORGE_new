"""Bind eligible supplied recipes to original task/context bytes without admitting training.

Selection precedes JSON decoding of molecular records. Source role/quantity/structure
agreement is separate from forward chemistry and from historical experimental execution.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import constitutional_molecule
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_full_preparation import FullPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

CONFIG_SCHEMA = "forge.compose_lipid_original_binding_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_original_binding.v1"
POLICY = {
    "seed": 0,
    "population": "current_full_preparation_eligible_only",
    "selection_before_molecular_record_decoding": True,
    "product_matching_used_to_choose_components": False,
    "source_payload_hash": "sha256_of_original_line_without_line_ending",
    "upstream_atom_annotations_used_as_supervision": False,
    "reaction_calls": 0,
    "training_calls": 0,
    "training_admitted": False,
    "experimental_execution_admitted": False,
}
FAMILIES = {
    "aldehyde_ugi3",
    "aldehyde_ugi4",
    "thiolactone_aminolysis_michael",
    "acid_epoxide_diester_multistep",
}
IMPLEMENTATION = (
    "forge/corpus/compose_lipid_original_binding.py",
    "forge/assembly/families.py",
    "forge/core/hashing.py",
    "forge/corpus/compose_lipid_full_preparation.py",
)


def selected_lines(path: Path, requested: set[int]):
    """Unselected molecular payloads are never decoded, even if they contain invalid JSON."""
    if any(type(n) is not int or n < 1 for n in requested):
        raise ComposeLipidError("Source line numbers must be positive integers")
    found = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for number, payload in enumerate(stream, 1):
            if number not in requested:
                continue
            row = json.loads(payload)
            if not isinstance(row, dict):
                raise ComposeLipidError(f"Original source line is not an object: {path}:{number}")
            found.add(number)
            yield number, row, hashlib.sha256(payload.rstrip(b"\r\n")).hexdigest()
    if found != requested:
        raise ComposeLipidError(
            f"Original source lines missing in {path}: {sorted(requested-found)}"
        )


def authenticate_reference(reference: dict, source: dict, number: int, payload_sha: str) -> None:
    expected = {
        "path": source["source"],
        "sha256": source["sha256"],
        "line": number,
        "payload_sha256": payload_sha,
    }
    if reference != expected:
        raise ComposeLipidError("Original source path, file, line or payload binding differs")


def require_eligible(prepared: dict) -> None:
    if prepared.get("eligible_for_program_preparation") is not True:
        raise ComposeLipidError("Protected or unassigned target reached original recipe binding")


def compare_recipe(
    prepared: dict, construction: dict, components: list[dict], structures: dict
) -> dict[str, bool]:
    """Compare supplied inputs without inspecting or using a product to choose a recipe."""
    require_eligible(prepared)
    expected = Counter()
    actual = Counter()
    source_ids = Counter()
    for role, identity, quantity in prepared["component_instances"]:
        if identity not in structures or type(quantity) is not int or quantity < 1:
            raise ComposeLipidError("Invalid or unresolved supplied component instance")
        expected[(role, structures[identity])] += quantity
    for component in components:
        role, identity, smiles, quantity = (
            component[k] for k in ("role", "source_id", "smiles", "quantity")
        )
        if (
            any(not isinstance(v, str) or not v for v in (role, identity, smiles))
            or type(quantity) is not int
            or quantity < 1
        ):
            raise ComposeLipidError("Original recipe has incomplete IDs, roles or quantities")
        canonical = constitutional_molecule(smiles)[0]
        actual[(role, canonical)] += quantity
        source_ids[(role, identity)] += 1
    declared_ids = Counter()
    for alias in construction["declared_precursor_identifiers"]:
        for role in alias["roles"]:
            declared_ids[(role, alias["value"])] += 1
    return {
        "complete_role_quantity_structure_agreement": actual == expected,
        "declared_precursor_ids_agree": declared_ids == source_ids,
    }


def source_components(source: dict, binding: dict, definitions: dict) -> list[dict]:
    """Use explicit source fields and registry role counts; no name-based alias guessing."""
    if binding["kind"] == "context":
        components = source["complete_subcomponents"]
        if source["precursor_ids"] != [c["subcomponent_id"] for c in components]:
            raise ComposeLipidError("Original context precursor IDs disagree with its structures")
        return [
            {
                "role": c["reaction_role"],
                "source_id": c["subcomponent_id"],
                "smiles": c["constitutional_smiles"],
                "quantity": definitions["roles"][
                    binding["source_to_registry_role"][c["reaction_role"]]
                ],
            }
            for c in components
        ]
    result = []
    for role, fields in binding["role_fields"].items():
        if definitions["field_to_role"].get(fields["id"]) != [role]:
            raise ComposeLipidError("Original-task role fields differ from the family registry")
        result.append(
            {
                "role": role,
                "source_id": source[fields["id"]],
                "smiles": source[fields["smiles"]],
                "quantity": definitions["roles"][role],
            }
        )
    return result


def bind_original(
    prepared: dict,
    construction: dict,
    source: dict,
    binding: dict,
    definitions: dict,
    structures: dict,
) -> dict:
    require_eligible(prepared)
    if source.get("training_admissible") is not False:
        raise ComposeLipidError("Unexpected upstream training-admission flag")
    components = source_components(source, binding, definitions)
    checks = compare_recipe(prepared, construction, components, structures)
    checks["source_family_agrees"] = source["family"] == prepared["family"]
    if binding["kind"] == "task":
        checks["source_task_id_agrees"] = source["task_id"] == construction["task_id"]
        checks["source_stage_order_declared"] = source.get(binding["stage_order_field"]) is True
        product_verified = False
    else:
        original = constitutional_molecule(source["constitutional_smiles"])[0]
        checks["source_product_identity_agrees"] = (
            hashlib.sha256(original.encode()).hexdigest() == prepared["constitution_id"]
        )
        product_verified = checks["source_product_identity_agrees"]
    return {
        "checks": checks,
        "original_recipe_verified": all(checks.values()),
        "original_product_graph_verified": product_verified,
        "source_components": components,
        "source_training_admissible": source["training_admissible"],
        "disposition": (
            "original_recipe_verified" if all(checks.values()) else "source_recipe_disagreement"
        ),
        "upstream_atom_annotations_used_as_supervision": False,
    }


def bind_regional(
    prepared: dict, construction: dict, source: dict, catalog: dict, structures: dict
) -> dict:
    require_eligible(prepared)
    if source.get("training_admissible") is not False:
        raise ComposeLipidError("Unexpected regional-ledger training-admission flag")
    components = []
    for identity in source["precursor_ids"]:
        if identity not in catalog:
            raise ComposeLipidError(f"Unresolved regional precursor ID: {identity}")
        record = catalog[identity]
        components.append(
            {
                "source_id": identity,
                "role": record["role"],
                "smiles": record["smiles"],
                "quantity": 1,
            }
        )
    checks = compare_recipe(prepared, construction, components, structures)
    checks["source_tuple_id_agrees"] = source["tuple_id"] == construction["task_id"]
    checks["source_product_id_agrees"] = source["product_id"] == construction["target_id"]
    return {
        "checks": checks,
        "original_recipe_verified": all(checks.values()),
        "original_product_graph_verified": False,
        "original_product_program_payload_verified": False,
        "source_components": components,
        "regional_disposition": source["disposition"],
        "reported_released_tuple": source["reported_released_tuple"],
        "source_parent_precursor_ids": source["source_parent_precursor_ids"],
        "source_training_admissible": False,
        "disposition": (
            "regional_recipe_verified_original_program_payload_unavailable"
            if all(checks.values())
            else "source_recipe_disagreement"
        ),
    }


def _load(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("Original-binding configuration or scientific policy changed")
    if set(config.get("families", {})) != FAMILIES:
        raise ComposeLipidError("Original-binding family scope changed")
    if set(config.get("inputs", {})) != {
        "preparation",
        "constructions",
        "task_manifest",
        "family_definitions",
        "ugi4_catalog",
        "role_binding_result",
    }:
        raise ComposeLipidError("Original-binding input set changed")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    reader = FullPreparationCorpus(repo, paths["preparation"])
    intake = reader.reader.result["inputs"]["intake"]
    source_intake = json.loads(resolve_pin(intake, repo, label="source intake").read_text())
    if source_intake["inputs"]["constructions"] != config["inputs"]["constructions"]:
        raise ComposeLipidError("Original binding uses a different full construction export")
    prior_binding = json.loads(paths["role_binding_result"].read_text())
    if prior_binding["inputs"]["preparation"] != config["inputs"]["preparation"]:
        raise ComposeLipidError("Role-namespace evidence uses a different eligible population")
    for group in ("implementation", "inputs", "artifacts"):
        for name, value in prior_binding[group].items():
            resolve_pin(value, repo, label=name)
    saved_binding = json.loads(
        resolve_pin(
            prior_binding["artifacts"]["binding.json"], repo, label="role binding"
        ).read_text()
    )
    if (
        config["families"]["aldehyde_ugi3"]["source_to_registry_role"]
        != saved_binding["source_to_qualified_source_role"]
    ):
        raise ComposeLipidError("Original context role namespace differs from qualified witnesses")
    preparation_config = json.loads(
        resolve_pin(reader.result["config"], repo, label="preparation config").read_text()
    )
    if (
        preparation_config["inputs"]["family_definitions"]["sha256"]
        != config["inputs"]["family_definitions"]["sha256"]
    ):
        raise ComposeLipidError("Original binding uses a different family-definition registry")
    manifest = json.loads(paths["task_manifest"].read_text())
    sources = {}
    for item in manifest["files"]:
        if not item["path"].startswith("original_generator_tasks/"):
            continue
        local = (paths["task_manifest"].parent / item["path"]).resolve()
        if not local.is_relative_to(paths["task_manifest"].parent.resolve()):
            raise ComposeLipidError("Original task path escapes the supplied bundle")
        current = {"path": str(local.relative_to(repo)), "sha256": item["sha256"]}
        resolve_pin(current, repo, label=item["source"])
        if item["source"] in sources:
            raise ComposeLipidError("Duplicate original source path")
        sources[item["source"]] = {**item, "local": local, "pin": current}
    definitions = json.loads(paths["family_definitions"].read_text())
    for family in FAMILIES:
        binding = config["families"][family]
        expected_kind = (
            "context"
            if family == "aldehyde_ugi3"
            else "regional" if family == "aldehyde_ugi4" else "task"
        )
        if binding.get("kind") != expected_kind:
            raise ComposeLipidError(f"Unexpected original binding kind for {family}")
        if expected_kind == "task" and set(binding["role_fields"]) != set(
            definitions[family]["roles"]
        ):
            raise ComposeLipidError(f"Original task role coverage differs for {family}")
        if any(type(q) is not int or q != 1 for q in definitions[family]["roles"].values()):
            raise ComposeLipidError(
                "Original binding requires explicit unit quantities for these families"
            )
    return config, paths, reader, sources, definitions


def run_original_binding(repo_root: Path, config_path: Path, output_dir: Path):
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("Original-binding output must be fresh and inside the repository")
    config, paths, reader, sources, definitions = _load(repo, config_path)
    prepared = {k: v for k, v in reader.preparation.items() if v["family"] in FAMILIES}
    by_line = {v["construction_source_line"]: k for k, v in prepared.items()}
    if len(by_line) != len(prepared):
        raise ComposeLipidError("Preparation has duplicate construction line references")
    structures = {r["component_id"]: r["constitution"] for r in rows(reader.precursors)}
    constructions = {}
    requests = defaultdict(lambda: defaultdict(list))
    regional = defaultdict(list)
    results = {}
    for number, construction, _ in selected_lines(paths["constructions"], set(by_line)):
        target = by_line[number]
        current = prepared[target]
        normalized = sorted(
            [c["role"], c["component_id"], c["quantity"]]
            for c in construction["component_instances"]
        )
        if (
            construction["target_id"] != target
            or construction["primary_family"] != current["family"]
            or normalized != current["component_instances"]
            or construction["construction_basis"] != current["construction_basis"]
        ):
            raise ComposeLipidError("Selected construction differs from authenticated preparation")
        constructions[target] = construction
        if construction["construction_basis"] == "lnpdb_compatible_decomposition_replay":
            results[target] = {
                "original_recipe_verified": False,
                "disposition": "compatible_decomposition_has_no_original_generator_task",
            }
            continue
        binding = config["families"][current["family"]]
        if binding["kind"] == "regional":
            regional[construction["task_id"]].append(target)
            continue
        reference = construction["provenance"]
        if binding["kind"] == "task":
            reference = reference["task"]
        if reference["path"] not in sources:
            results[target] = {
                "original_recipe_verified": False,
                "disposition": "original_source_record_not_in_restored_bundle",
                "unresolved_reference": reference,
            }
            continue
        if type(reference["line"]) is not int or reference["line"] < 1:
            raise ComposeLipidError("Invalid original task line number")
        requests[reference["path"]][reference["line"]].append((target, reference))
    for original_path, selected in sorted(requests.items()):
        source_file = sources[original_path]
        for number, row, payload_sha in selected_lines(source_file["local"], set(selected)):
            for target, reference in selected[number]:
                authenticate_reference(reference, source_file, number, payload_sha)
                family = prepared[target]["family"]
                results[target] = bind_original(
                    prepared[target],
                    constructions[target],
                    row,
                    config["families"][family],
                    definitions[family],
                    structures,
                )
                results[target]["original_source"] = {**reference, "local_file": source_file["pin"]}
    family = "aldehyde_ugi4"
    regional_source = sources[config["families"][family]["regional_source"]]
    catalog = {}
    with paths["ugi4_catalog"].open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in catalog:
                raise ComposeLipidError("Duplicate Ugi-4 catalogue identity")
            catalog[row["id"]] = row
    seen = set()
    for number, row in enumerate(rows(regional_source["local"]), 1):
        if row["tuple_id"] not in regional:
            continue
        if row["tuple_id"] in seen:
            raise ComposeLipidError("Duplicate selected regional tuple")
        seen.add(row["tuple_id"])
        for target in regional[row["tuple_id"]]:
            results[target] = bind_regional(
                prepared[target], constructions[target], row, catalog, structures
            )
            results[target]["original_source"] = {
                "local_file": regional_source["pin"],
                "line": number,
                "original_program_reference": constructions[target]["provenance"]["program"],
            }
    for identity in set(regional) - seen:
        for target in regional[identity]:
            results[target] = {
                "original_recipe_verified": False,
                "disposition": "regional_tuple_not_in_restored_ledger",
                "task_id": identity,
            }
    if set(results) != set(prepared):
        raise ComposeLipidError("Original binding lost eligible preparation targets")
    counts = defaultdict(Counter)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".original-binding-", dir=output.parent) as temporary:
        stage = Path(temporary)
        ledger = stage / "bindings.jsonl.gz"
        with (
            ledger.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream,
        ):
            for target in sorted(prepared):
                current, binding = prepared[target], results[target]
                record = {
                    k: current[k]
                    for k in (
                        "target_id",
                        "family",
                        "constitution_id",
                        "component_instances",
                        "construction_basis",
                    )
                }
                record.update(
                    binding=binding, training_admitted=False, experimental_execution_admitted=False
                )
                stream.write((compact(record) + "\n").encode())
                counts[current["family"]]["rows"] += 1
                counts[current["family"]][binding["disposition"]] += 1
                counts[current["family"]]["verified_recipe_rows"] += binding[
                    "original_recipe_verified"
                ]
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "original_recipe_bindings_checked_training_unqualified",
            "policy": POLICY,
            "config": pin(repo, config_path),
            "inputs": config["inputs"],
            "implementation": {name: pin(repo, repo / name) for name in IMPLEMENTATION},
            "summary": {k: dict(v) for k, v in sorted(counts.items())},
            "artifacts": {
                ledger.name: {
                    "path": str((output / ledger.name).relative_to(repo)),
                    "sha256": str(sha256_file(ledger)),
                }
            },
        }
        dump(stage / "result.json", result)
        os.rename(stage, output)
    return result

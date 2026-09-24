"""Qualify a supplied role namespace by exact precursor agreement with prior evidence.

The binding is fixed across the population before forward replay. It cannot select new
components, discard quantities, or choose a role assignment by matching a product.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.source_event import check_source_event
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus import compose_lipid_source_event as event
from forge.corpus.compose_lipid_family_replay import replay_record
from forge.corpus.compose_lipid_full_preparation import FullPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

CONFIG_SCHEMA = "forge.compose_lipid_role_binding_config.v1"
POLICY = {
    "seed": 0,
    "family": "aldehyde_ugi3",
    "binding_evidence": "prior_exact_precursor_structures_on_current_eligible_population",
    "prior_authentication": "saved_bytes_only_no_population_replay",
    "binding_fixed_before_forward_replay": True,
    "product_matching_used_to_choose_binding": False,
    "component_quantities": "exactly_one_per_declared_single_event_role",
    "unresolved_or_conflicting_binding": "fail_closed",
    "training_admitted": False,
    "training_calls": 0,
    "experimental_execution_admitted": False,
}
IMPLEMENTATION = (
    "forge/corpus/compose_lipid_role_binding.py",
    "forge/corpus/compose_lipid_family_replay.py",
    "forge/corpus/compose_lipid_full_preparation.py",
    *event.IMPLEMENTATION,
)


def bind_recipe(instances, structures, prior_components, expected_roles):
    """Use only precursor identities; product structure is deliberately not an argument."""
    by_structure = defaultdict(set)
    observed_roles = []
    for component in prior_components:
        role, structure = component["role"], component["canonical_smiles"]
        by_structure[structure].add(role)
        observed_roles.append(role)
    if set(observed_roles) != set(expected_roles) or len(observed_roles) != len(expected_roles):
        raise ComposeLipidError("Prior role witness is incomplete or duplicated")
    if len(instances) != len(expected_roles):
        raise ComposeLipidError("Supplied role witness has a different precursor count")
    binding = {}
    for source_role, identity, quantity in instances:
        if identity not in structures:
            raise ComposeLipidError("Source role witness has an unresolved global identity")
        if type(quantity) is not int or quantity != 1 or source_role in binding:
            raise ComposeLipidError("Source role witness changes role multiplicity")
        matches = by_structure.get(structures[identity], set())
        if len(matches) != 1:
            raise ComposeLipidError("Source role witness lacks a unique exact precursor match")
        binding[source_role] = next(iter(matches))
    if set(binding.values()) != set(expected_roles):
        raise ComposeLipidError("Source role witness does not cover the complete program")
    return binding


def fixed_binding(witnesses):
    """Any disagreement rejects the binding; no majority vote or selected witness subset."""
    binding = None
    count = 0
    for witness in witnesses:
        if binding is None:
            binding = witness
        elif binding != witness:
            raise ComposeLipidError("Source role namespace conflicts across exact witnesses")
        count += 1
    if not count:
        raise ComposeLipidError("Source role binding needs exact precursor witnesses")
    return binding, count


def authenticate_prior(repo: Path, path: Path) -> dict:
    """Authenticate saved evidence without rerunning its superseded population."""
    result = json.loads(path.read_text())
    if (
        result.get("schema_version") != event.RESULT_SCHEMA
        or result.get("status") != "source_event_evaluated_training_unqualified"
        or result.get("policy") != event.POLICY
        or set(result.get("implementation", {})) != set(event.IMPLEMENTATION)
        or set(result.get("artifacts", {})) != {"programs.jsonl.gz"}
    ):
        raise ComposeLipidError("Prior source-event receipt scope changed")
    for name, value in result["implementation"].items():
        if resolve_pin(value, repo, label=name) != repo / name:
            raise ComposeLipidError("Prior source-event implementation substitution")
    config = json.loads(resolve_pin(result["config"], repo, label="prior config").read_text())
    if config["inputs"] != result["inputs"]:
        raise ComposeLipidError("Prior source-event receipt/config input disagreement")
    for group in ("inputs", "artifacts"):
        for name, value in result[group].items():
            resolve_pin(value, repo, label=name)
    return result


def run_role_binding(repo_root: Path, config_path: Path, output_dir: Path):
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("Role-binding output must be fresh and inside the repository")
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("Role-binding scientific policy changed")
    if set(config.get("inputs", {})) != {"preparation", "prior_program", "source_config"}:
        raise ComposeLipidError("Role-binding input set changed")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    previous = authenticate_prior(repo, paths["prior_program"])
    if previous["config"] != config["inputs"]["source_config"]:
        raise ComposeLipidError("Prior role witnesses and source contract differ")
    loaded = event._load(repo, paths["source_config"])
    cfg, evidence, adapter, _, _, _, kwargs = loaded
    if cfg["family"] != POLICY["family"]:
        raise ComposeLipidError("Role binding cannot change reaction-family scope")
    mapping = evidence["source_contract"]["registry_to_source_roles"]
    controls = {}
    for control in evidence["source_controls"]:
        checked = check_source_event(
            adapter, control["components"], control["expected_product"], **kwargs
        )
        formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(control["expected_product"]))
        if not checked["computed_consistency_pass"] or formula != control["expected_formula"]:
            raise ComposeLipidError("Role-binding source control failed")
        controls[control["label"]] = checked
    if not controls:
        raise ComposeLipidError("Role-binding source controls are required")
    reader = FullPreparationCorpus(repo, paths["preparation"])
    structures = {r["component_id"]: r["constitution"] for r in rows(reader.precursors)}
    prior_path = resolve_pin(
        previous["artifacts"]["programs.jsonl.gz"], repo, label="prior programs"
    )
    prior = {r["target_id"]: r for r in rows(prior_path)}
    witness_rows = []
    for target, prepared in sorted(reader.preparation.items()):
        if prepared["family"] != cfg["family"]:
            continue
        old = prior.get(target)
        if (
            old is None
            or not old["computed_consistency_pass"]
            or old["training_admitted"]
            or old["constitution_id"] != prepared["constitution_id"]
            or old["family"] != prepared["family"]
            or not all(old["checks"].values())
        ):
            raise ComposeLipidError(
                "Source role binding requires exact prior evidence for every row"
            )
        binding = bind_recipe(
            prepared["component_instances"], structures, old["components"], mapping.values()
        )
        witness_rows.append(
            {
                "target_id": target,
                "binding": binding,
                "component_instances": prepared["component_instances"],
                "prior_components": old["components"],
                "constitution_id": prepared["constitution_id"],
            }
        )
    binding, witness_count = fixed_binding(r["binding"] for r in witness_rows)
    inverse = {value: key for key, value in binding.items()}
    registry_mapping = {role: inverse[source_role] for role, source_role in mapping.items()}
    executor = {
        "kind": "fixed",
        "mapping": registry_mapping,
        "run": lambda components, target: check_source_event(adapter, components, target, **kwargs),
    }
    counts = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".role-binding-", dir=output.parent) as temporary:
        stage = Path(temporary)
        dump(
            stage / "binding.json",
            {
                "source_to_qualified_source_role": binding,
                "registry_to_supplied_role": registry_mapping,
                "witness_count": witness_count,
                "witnesses": witness_rows,
                "controls": controls,
                "scope": "structural_role_concordance_and_computed_replay_not_historical_route",
            },
        )
        ledger = stage / "replay.jsonl.gz"
        with (
            ledger.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream,
        ):
            for item in reader.iter_preparation_records(family=cfg["family"]):
                replay = replay_record(item, structures, executor)
                row = {
                    k: item["preparation"][k]
                    for k in (
                        "target_id",
                        "family",
                        "constitution_id",
                        "component_instances",
                        "construction_basis",
                    )
                }
                row.update(
                    replay=replay, training_admitted=False, experimental_execution_admitted=False
                )
                stream.write((compact(row) + "\n").encode())
                counts[replay["disposition"]] += 1
        if sum(counts.values()) != witness_count:
            raise ComposeLipidError("Role-binding replay did not cover every witness")
        result = {
            "schema_version": "forge.compose_lipid_role_binding.v1",
            "status": "role_concordance_checked_and_supplied_recipes_replayed",
            "config": pin(repo, config_path),
            "inputs": config["inputs"],
            "policy": POLICY,
            "implementation": {name: pin(repo, repo / name) for name in IMPLEMENTATION},
            "summary": {
                "family": cfg["family"],
                "witness_rows": witness_count,
                "replay": dict(counts),
            },
            "artifacts": {
                name: {
                    "path": str((output / name).relative_to(repo)),
                    "sha256": str(sha256_file(stage / name)),
                }
                for name in ("binding.json", "replay.jsonl.gz")
            },
        }
        dump(stage / "result.json", result)
        os.rename(stage, output)
    return result

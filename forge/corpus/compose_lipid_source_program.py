"""Qualify a source-adjudicated, single-event program without admitting training rows."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, rdBase

from forge.assembly.atom_map_variant import derive_product_atom_map_variant
from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import RegistryAssemblyAdapter, constitutional_molecule
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_pretraining import verify_pretraining_checks

CONFIG_SCHEMA = "forge.compose_lipid_source_program_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_source_program.v1"
IMPLEMENTATION = (
    "forge/assembly/atom_map_variant.py",
    "forge/assembly/families.py",
    "forge/assembly/registry.py",
    "forge/assembly/program.py",
    "forge/corpus/compose_lipid_source_program.py",
    "forge/core/hashing.py",
    "forge/corpus/compose_lipid_pretraining.py",
)
CHECKS = {
    "unique_forward_exact",
    "unique_inverse_exact",
    "retained_amine_in_acid_head",
    "full_element_hydrogen_charge_balance",
    "source_map_bonds_match",
}
SCOPE = (
    "single-event constitutional program on inspected protected TRAIN; "
    "no experimental outcome, isotope tracing, L2 or procurement claim"
)
REMAINING_HOLDS = [
    "global_precursor_protection_incomplete",
    "precursor_holdouts_unqualified",
    "remaining_families_unqualified",
    "repository_full_tests_fail",
]


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _pin(repo: Path, path: Path) -> dict:
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def retained_amine(molecule: Chem.Mol, policy: dict) -> tuple[int, ...]:
    """Structural preservation only; the pinned source policy does not estimate pKa."""
    found = []
    for atom in molecule.GetAtoms():
        if (
            atom.GetSymbol() != policy["symbol"]
            or atom.GetFormalCharge() != policy["formal_charge"]
            or atom.GetIsAromatic() != policy["aromatic"]
            or atom.GetDegree() != policy["degree"]
            or any(str(b.GetBondType()) != policy["bond_type"] for b in atom.GetBonds())
        ):
            continue
        neighbors = list(atom.GetNeighbors())
        if any(n.GetSymbol() != policy["neighbor_symbol"] for n in neighbors):
            continue
        if any(
            bond.GetBondTypeAsDouble() > 1
            and bond.GetOtherAtom(neighbor).GetSymbol()
            in policy["forbidden_neighbor_multiple_bond_symbols"]
            for neighbor in neighbors
            for bond in neighbor.GetBonds()
        ):
            continue
        found.append(atom.GetIdx())
    return tuple(found)


def element_inventory(molecules) -> Counter:
    counts = Counter()
    for molecule in molecules:
        for atom in molecule.GetAtoms():
            counts[atom.GetSymbol()] += 1
            counts["H"] += atom.GetTotalNumHs()
            counts["formal_charge"] += atom.GetFormalCharge()
    return counts


def map_bonds_match(adapter: RegistryAssemblyAdapter, expected: list) -> bool:
    template = adapter.reaction.forward.GetProductTemplate(0)
    mapped = {a.GetAtomMapNum(): a.GetIdx() for a in template.GetAtoms() if a.GetAtomMapNum()}
    for left, right, kind in expected:
        if left not in mapped or right not in mapped:
            return False
        bond = template.GetBondBetweenAtoms(mapped[left], mapped[right])
        if bond is None or str(bond.GetBondType()) != kind:
            return False
    return True


def check_complete_program(adapter, components: dict, target: str, contract: dict) -> dict:
    """Require unique constitutional site class, full replay, retained head and atom balance."""
    target, molecule = constitutional_molecule(target)
    bound = contract["bounded_search_limit"]
    canonical = {role: constitutional_molecule(smi)[0] for role, smi in components.items()}
    forward = adapter.forward_products(canonical, maximum_outcomes=bound)
    inverse = adapter.decompose(target, maximum_outcomes=bound)
    head = Chem.MolFromSmiles(canonical[contract["retained_amine_role"]])
    checks = {
        "unique_forward_exact": not forward.saturated and forward.products == (target,),
        "unique_inverse_exact": len(inverse) == 1 and dict(inverse[0].components) == canonical,
        "retained_amine_in_acid_head": bool(
            retained_amine(head, contract["retained_amine_policy"])
        ),
        "full_element_hydrogen_charge_balance": element_inventory(
            Chem.MolFromSmiles(smi) for smi in canonical.values()
        )
        == element_inventory([molecule]),
        "source_map_bonds_match": map_bonds_match(adapter, contract["expected_product_map_bonds"]),
    }
    return {"checks": checks, "qualified": all(checks.values())}


def _load(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != {
        "training_calls": 0,
        "generation_calls": 0,
        "heldout_graphs_parsed": False,
        "source_flags_changed": False,
        "fit_population": "protected_train",
        "seed": 0,
    }:
        raise ComposeLipidError("source-program schema or scope changed")
    paths = {key: resolve_pin(pin, repo, label=key) for key, pin in config["inputs"].items()}
    key = json.loads(paths["decomposition_key"].read_text())[config["family"]]
    contract = config["source_contract"]
    if (
        contract["roles"] != key["roles"]
        or set(key["roles"].values()) != {1}
        or key["variable"]
        or contract["events"] != 1
        or contract["stage_order"] != [config["original_reaction_id"]]
    ):
        raise ComposeLipidError("source program is not the declared complete single-event contract")
    original = next(
        r
        for r in json.loads(paths["registry"].read_text())["reactions"]
        if r["reaction_id"] == config["original_reaction_id"]
    )
    roles = {r["name"] for r in original["reactant_roles"]}
    mapping = contract["registry_to_source_roles"]
    if set(mapping) != roles or set(mapping.values()) != set(key["roles"]):
        raise ComposeLipidError("source program roles do not match the registry")
    if type(contract["bounded_search_limit"]) is not int or contract["bounded_search_limit"] < 2:
        raise ComposeLipidError("source program search bound is invalid")
    variant = derive_product_atom_map_variant(original, config["variant"])
    return config, paths, variant


def _controls(adapter, config):
    controls = {
        row["label"]: check_complete_program(
            adapter, row["components"], row["expected_product"], config["source_contract"]
        )
        for row in config["source_controls"]
    }
    if not controls or len(controls) != len(config["source_controls"]):
        raise ComposeLipidError("independent source controls missing or duplicated")
    if not all(row["qualified"] for row in controls.values()):
        raise ComposeLipidError(f"source controls failed: {controls}")
    return controls


def _observations(repo: Path, pretraining: dict, family: str) -> dict:
    observations = {}
    path = repo / pretraining["artifacts"]["train_checks.jsonl.gz"]["path"]
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row["family"] != family:
                continue
            if row["target_id"] in observations:
                raise ComposeLipidError("duplicate target in pinned TRAIN inspection")
            observations[row["target_id"]] = row["related_transform_probe"]
    return observations


def _summary(rows: list[dict], config: dict, observations: dict) -> dict:
    """Validate scope and protected-component attribution before counting outcomes."""
    counts = Counter()
    seen = set()
    for row in rows:
        identity = row["target_id"]
        observed = observations.get(identity)
        if identity in seen or observed is None:
            raise ComposeLipidError("source-program target is duplicate or outside pinned TRAIN")
        seen.add(identity)
        if (
            row.get("family") != config["family"]
            or row.get("training_admitted") is not False
            or row.get("components") != observed.get("components")
            or observed.get("status")
            not in ("exact_related_transform", "exact_related_transform_protected_precursor")
        ):
            raise ComposeLipidError("source-program component attribution or admission changed")
        components = row["components"]
        if Counter(c["role"] for c in components) != config["source_contract"]["roles"]:
            raise ComposeLipidError("source-program component roles are not complete")
        for component in components:
            canonical, _ = constitutional_molecule(component["canonical_smiles"])
            if (
                canonical != component["canonical_smiles"]
                or hashlib.sha256(canonical.encode()).hexdigest() != component["constitution_id"]
                or component["historical_fold"] not in (None, "train", "calibration", "heldout")
            ):
                raise ComposeLipidError("source-program component identity or fold is invalid")
        checks = row.get("checks", {})
        protected = any(c["historical_fold"] in ("calibration", "heldout") for c in components)
        if (
            set(checks) != CHECKS
            or any(type(value) is not bool for value in checks.values())
            or row.get("qualified") is not all(checks.values())
            or row.get("historical_protected_precursor") is not protected
        ):
            raise ComposeLipidError("source-program check or protection accounting changed")
        counts["rows"] += 1
        counts["complete_program_checks_pass"] += int(row["qualified"])
        counts["historical_protected_precursor"] += int(row["historical_protected_precursor"])
        counts["clear_of_known_historical_precursors"] += int(
            row["qualified"] and not row["historical_protected_precursor"]
        )
    if seen != set(observations):
        raise ComposeLipidError("source-program ledger does not cover pinned TRAIN")
    return dict(counts)


def run_source_program(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path = (repo / config_path).resolve()
    config, paths, variant = _load(repo, config_path)
    pretraining = verify_pretraining_checks(
        repo, resolve_pin(config["pretraining_result"], repo, label="pretraining receipt")
    )
    imported = json.loads((repo / pretraining["import_result"]["path"]).read_text())
    output = (repo / output_dir).resolve()
    output.relative_to(repo)
    if output.exists():
        raise ComposeLipidError("source-program output exists; use a fresh version")
    output.parent.mkdir(parents=True, exist_ok=True)
    observations = _observations(repo, pretraining, config["family"])
    db_path = repo / imported["artifacts"]["corpus.sqlite"]["path"]
    db = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".source-program-", dir=output.parent) as temporary:
            work = Path(temporary)
            registry = work / "registry.json"
            _dump(
                registry,
                {"registry_version": config["variant"]["reaction_id"], "reactions": [variant]},
            )
            adapter = RegistryAssemblyAdapter.from_registry(
                registry,
                reaction_id=variant["reaction_id"],
                expected_sha256=str(sha256_file(registry)),
            )
            controls = _controls(adapter, config)
            reverse_roles = {
                v: k for k, v in config["source_contract"]["registry_to_source_roles"].items()
            }
            ledger = []
            for (payload,) in db.execute(
                "SELECT t.payload FROM assignments a JOIN targets t USING(target_id) "
                "WHERE a.forge_split='train' AND a.family=? ORDER BY a.target_id",
                (config["family"],),
            ):
                target = json.loads(payload)
                observed = observations[target["target_id"]]
                if observed["status"] not in (
                    "exact_related_transform",
                    "exact_related_transform_protected_precursor",
                ):
                    raise ComposeLipidError(
                        "source program requires a fresh ambiguous-component audit"
                    )
                components = {
                    reverse_roles[c["role"]]: c["canonical_smiles"] for c in observed["components"]
                }
                checked = check_complete_program(
                    adapter, components, target["constitution"], config["source_contract"]
                )
                ledger.append(
                    {
                        "target_id": target["target_id"],
                        "family": config["family"],
                        "components": observed["components"],
                        **checked,
                        "historical_protected_precursor": any(
                            c["historical_fold"] in ("calibration", "heldout")
                            for c in observed["components"]
                        ),
                        "training_admitted": False,
                    }
                )
            summary = _summary(ledger, config, observations)
            if summary["rows"] != config["expected_train_rows"]:
                raise ComposeLipidError(
                    "source-program population differs from pinned TRAIN inspection"
                )
            if summary["complete_program_checks_pass"] != summary["rows"]:
                raise ComposeLipidError("complete source-program qualification failed")
            payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger).encode()
            (work / "program_checks.jsonl.gz").write_bytes(gzip.compress(payload, mtime=0))
            result = {
                "schema_version": RESULT_SCHEMA,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "single_family_program_qualified_training_unqualified",
                "family": config["family"],
                "summary": summary,
                "source_controls": controls,
                "evidence_basis": "computed_transform_consistency",
                "source_event_program_qualified": True,
                "scope": SCOPE,
                "training_ready": False,
                "training_rows_admitted": 0,
                "other_source_families_qualified_by_this_receipt": [],
                "remaining_holds": REMAINING_HOLDS,
                "config": _pin(repo, config_path),
                "pretraining_result": config["pretraining_result"],
                "implementation": {p: _pin(repo, repo / p) for p in IMPLEMENTATION},
                "artifacts": {
                    p.name: {
                        "path": (output / p.name).relative_to(repo).as_posix(),
                        "sha256": str(sha256_file(p)),
                    }
                    for p in sorted(work.iterdir())
                },
                "policy": config["policy"],
                "rdkit_version": rdBase.rdkitVersion,
            }
            _dump(work / "result.json", result)
            os.rename(work, output)
    finally:
        db.close()
    return result


def verify_source_program(repo_root: Path, result_path: Path) -> dict:
    """Authenticate execution, recount the pinned ledger and replay primary controls.

    This does not rerun all TRAIN chemistry. A fresh run_source_program invocation does that.
    """
    repo = repo_root.resolve()
    result = json.loads((repo / result_path).read_text())
    config, _, variant = _load(repo, resolve_pin(result["config"], repo, label="config"))
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "single_family_program_qualified_training_unqualified"
        or result.get("training_ready") is not False
        or result.get("training_rows_admitted") != 0
        or result.get("policy") != config["policy"]
        or result.get("family") != config["family"]
        or result.get("pretraining_result") != config["pretraining_result"]
        or result.get("other_source_families_qualified_by_this_receipt") != []
        or result.get("evidence_basis") != "computed_transform_consistency"
        or result.get("scope") != SCOPE
        or result.get("remaining_holds") != REMAINING_HOLDS
        or result.get("source_event_program_qualified") is not True
        or set(result.get("implementation", {})) != set(IMPLEMENTATION)
        or set(result.get("artifacts", {})) != {"registry.json", "program_checks.jsonl.gz"}
    ):
        raise ComposeLipidError("source-program receipt scope changed")
    pretraining = verify_pretraining_checks(
        repo, resolve_pin(config["pretraining_result"], repo, label="pretraining")
    )
    for name, pin in result["implementation"].items():
        if pin["path"] != name:
            raise ComposeLipidError("source-program implementation path changed")
        resolve_pin(pin, repo, label=name)
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in result["artifacts"].items()}
    registry = json.loads(paths["registry.json"].read_text())
    if registry != {"registry_version": config["variant"]["reaction_id"], "reactions": [variant]}:
        raise ComposeLipidError(
            "source-program registry differs from source-adjudicated derivation"
        )
    adapter = RegistryAssemblyAdapter.from_registry(
        paths["registry.json"],
        reaction_id=variant["reaction_id"],
        expected_sha256=result["artifacts"]["registry.json"]["sha256"],
    )
    with gzip.open(paths["program_checks.jsonl.gz"], "rt") as handle:
        ledger = [json.loads(line) for line in handle]
    if (
        _summary(ledger, config, _observations(repo, pretraining, config["family"]))
        != result["summary"]
        or result["summary"]["rows"] != config["expected_train_rows"]
        or result["summary"]["rows"] != result["summary"]["complete_program_checks_pass"]
        or result["source_controls"] != _controls(adapter, config)
    ):
        raise ComposeLipidError("source-program summary is not reproduced by its ledger")
    return result

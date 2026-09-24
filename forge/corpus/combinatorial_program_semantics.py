"""Qualify exact atom semantics for the twelve-library computed-program view."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import rdBase

from forge.assembly.families import LibraryAssemblyError, load_assembly_libraries
from forge.assembly.library_semantics import trace_library_atom_semantics
from forge.core.hashing import resolve_pin, sha256_file
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.synthesis_program_graph import (
    SynthesisProgramGraphError,
    tensorize_synthesis_program_product,
)
from forge.model.vocabulary import load_atom_vocabulary
from forge.potency.annotations import UgiSemanticAnnotationError, annotate_qualified_ugi_product
from forge.synthesis.engine.qualified_forward import load_qualified_forward_reaction

CONFIG_SCHEMA = "forge.combinatorial_program_semantics_config.v1"
RESULT_SCHEMA = "forge.combinatorial_program_semantics.v1"
SEMANTIC_SCHEMA = "forge.combinatorial_program_semantic_record.v1"
_FOLDS = ("train", "calibration", "heldout")


class CombinatorialProgramSemanticError(ValueError):
    """A selected program or semantic artifact violates the pinned contract."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@contextmanager
def _gzip_text(path: Path):
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as handle,
    ):
        yield handle


_ADAPTERS: dict = {}
_UGI = None
_MAX_OUTCOMES = 0
_POSITION_ALIASES: dict[str, dict[str, str]] = {}


def _initialize_workers(
    registries: list,
    families: Sequence[str],
    position_aliases: Mapping[str, Mapping[str, str]],
    ugi_registry: Path,
    ugi_variant: Path,
    maximum: int,
) -> None:
    global _ADAPTERS, _UGI, _MAX_OUTCOMES, _POSITION_ALIASES
    _ADAPTERS = load_assembly_libraries(registries, expected_families=families)
    _POSITION_ALIASES = {
        str(family): {str(key): str(value) for key, value in aliases.items()}
        for family, aliases in position_aliases.items()
    }
    _UGI = load_qualified_forward_reaction(
        ugi_registry,
        ugi_variant,
        reaction_id="ugi_3cr_agile",
    )
    _MAX_OUTCOMES = maximum


def _reason(error: Exception) -> str:
    message = str(error)
    if "ambiguous" in message or "disagree" in message:
        return "semantic_assignment_ambiguous"
    if "saturated" in message:
        return "semantic_enumeration_saturated"
    if "policy" in message:
        return "registry_policy_changed_during_semantic_replay"
    if "reconstruct" in message:
        return "semantic_exact_replay_failed"
    return "semantic_annotation_error"


def _annotate(job: tuple[str, str, str, str, str]) -> dict[str, Any]:
    pid, family, components_json, witness_json, accumulator_role = job
    components = dict(json.loads(components_json))
    witness = json.loads(witness_json)
    target = witness[-1]
    try:
        if family == "ugi_3cr_agile":
            product, atoms, _, _ = annotate_qualified_ugi_product(
                _UGI,
                product_id=pid,
                target_smiles=target,
                component_smiles_by_role=components,
                source_evidence_record_id="computed_transform_consistency",
                max_outcomes=_MAX_OUTCOMES,
            )
            if int(product["semantic_signature_multiplicity"]) != 1:
                raise CombinatorialProgramSemanticError("Ugi semantic signature is ambiguous")
            roles = [str(row["origin_role"]) for row in atoms]
            positions = [str(row["core_position"]) for row in atoms]
            fixed = [int(row["product_atom_index"]) for row in atoms if bool(row["is_ugi_core"])]
            if len(fixed) != 5:
                raise CombinatorialProgramSemanticError("Ugi fixed core no longer has five atoms")
            canonical = str(product["product_smiles"])
            ugi_source_mapping_multiplicity = int(product["source_mapping_multiplicity"])
        else:
            semantics = trace_library_atom_semantics(
                _ADAPTERS[family],
                components,
                witness,
                accumulator_role=accumulator_role or None,
                maximum_outcomes=_MAX_OUTCOMES,
                core_position_aliases=_POSITION_ALIASES[family],
            )
            canonical = semantics.canonical_product_smiles
            roles = list(semantics.atom_origins)
            positions = list(semantics.core_positions)
            fixed = []
            ugi_source_mapping_multiplicity = None
        if hashlib.sha256(canonical.encode()).hexdigest() != pid:
            raise CombinatorialProgramSemanticError("semantic product identity changed")
        return {
            "ok": True,
            "product_id": pid,
            "canonical_smiles": canonical,
            "origin_roles": roles,
            "core_positions": positions,
            "ugi_fixed_atom_indices": fixed,
            "ugi_source_mapping_multiplicity": ugi_source_mapping_multiplicity,
        }
    except (
        CombinatorialProgramSemanticError,
        LibraryAssemblyError,
        UgiSemanticAnnotationError,
        ValueError,
        RuntimeError,
    ) as error:
        return {"ok": False, "product_id": pid, "reason": _reason(error), "detail": str(error)}


def _results(jobs: Iterator[tuple[str, str, str, str, str]], workers: int, args: tuple):
    if workers == 1:
        _initialize_workers(*args)
        yield from map(_annotate, jobs)
        return
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_initialize_workers, initargs=args
    ) as pool:
        pending = deque()
        for _ in range(workers * 2):
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(_annotate, job))
        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(_annotate, job))


def _contracts(
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], ReactionProgramVocabulary]:
    contracts: dict[str, dict[str, Any]] = {}
    for family, raw in config["programs"].items():
        roles = tuple(str(value) for value in raw["roles"])
        positions = tuple(str(value) for value in raw["core_positions"])
        aliases = {
            str(key): str(value) for key, value in raw.get("core_position_aliases", {}).items()
        }
        depths = tuple(int(value) for value in raw["allowed_depths"])
        if (
            not roles
            or len(set(roles)) != len(roles)
            or not positions
            or len(set(positions)) != len(positions)
            or not depths
            or min(depths) < 1
            or type(raw["fix_core_atoms"]) is not bool
            or any(not key or value not in positions for key, value in aliases.items())
        ):
            raise CombinatorialProgramSemanticError(f"invalid semantic contract: {family}")
        contracts[family] = {
            "roles": roles,
            "core_positions": positions,
            "allowed_depths": depths,
            "fix_core_atoms": raw["fix_core_atoms"],
            "position_aliases": aliases,
        }
    vocabulary = ReactionProgramVocabulary.from_semantics(
        program_ids=tuple(sorted(contracts)),
        roles=tuple(sorted({role for value in contracts.values() for role in value["roles"]})),
        core_positions=tuple(
            sorted(
                f"{family}:{position}"
                for family, value in contracts.items()
                for position in value["core_positions"]
            )
        ),
        maximum_steps=max(max(value["allowed_depths"]) for value in contracts.values()),
    )
    return contracts, vocabulary


def _selected(
    db: sqlite3.Connection, path: Path, expected: Mapping[str, Mapping[str, int]]
) -> Counter:
    db.executescript("""
        CREATE TABLE selected(pid TEXT PRIMARY KEY,gid TEXT,family TEXT,fold TEXT,smiles TEXT,
                              source_weight REAL,training_weight REAL);
        CREATE INDEX selected_gid ON selected(gid);
        CREATE TABLE jobs(pid TEXT PRIMARY KEY,family TEXT,components TEXT,witness TEXT,
                          accumulator_role TEXT,depth INTEGER);
    """)
    counts: Counter[tuple[str, str]] = Counter()
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["fold"] == "quarantine"
                or row["unique_policy_qualified_program_available"] != "1"
                or float(row["mean_realism_weight"]) <= 0
            ):
                continue
            family, fold = row["reaction_family"], row["fold"]
            if family not in expected or fold not in _FOLDS:
                raise CombinatorialProgramSemanticError("selected product family/fold changed")
            source_weight = float(row["mean_realism_weight"])
            training_weight = float(row["training_sampling_weight"])
            if (
                not np.isfinite(source_weight)
                or not np.isfinite(training_weight)
                or training_weight < 0
                or (fold == "train") != (training_weight > 0)
            ):
                raise CombinatorialProgramSemanticError("selected product weight changed")
            pid = row["product_id"]
            if pid != hashlib.sha256(row["canonical_smiles"].encode()).hexdigest():
                raise CombinatorialProgramSemanticError("selected product identity changed")
            db.execute(
                "INSERT INTO selected VALUES (?,?,?,?,?,?,?)",
                (
                    pid,
                    row["representative_program_group"],
                    family,
                    fold,
                    row["canonical_smiles"],
                    source_weight,
                    training_weight,
                ),
            )
            counts[(family, fold)] += 1
    observed = {
        family: {fold: counts[(family, fold)] for fold in _FOLDS} for family in sorted(expected)
    }
    if observed != {
        family: {fold: int(raw[fold]) for fold in _FOLDS}
        for family, raw in sorted(expected.items())
    }:
        raise CombinatorialProgramSemanticError("selected program population changed")
    db.commit()
    return counts


def _load_jobs(db: sqlite3.Connection, path: Path) -> None:
    needed = {row[0] for row in db.execute("SELECT DISTINCT gid FROM selected")}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            group = json.loads(line)
            gid = group["group_id"]
            if gid not in needed:
                continue
            selected = {
                row[0] for row in db.execute("SELECT pid FROM selected WHERE gid=?", (gid,))
            }
            for target in group["targets"]:
                pid = hashlib.sha256(target["product_smiles"].encode()).hexdigest()
                if pid not in selected:
                    continue
                if (
                    not target["search_complete_through_minimum_depth"]
                    or target["policy_path_count_capped_at_two"] != 1
                    or not target["policy_intermediate_products"]
                ):
                    raise CombinatorialProgramSemanticError(
                        "selected product lacks its unique policy-qualified witness"
                    )
                db.execute(
                    "INSERT INTO jobs VALUES (?,?,?,?,?,?)",
                    (
                        pid,
                        group["reaction_id"],
                        _json(group["components"]),
                        _json(target["policy_intermediate_products"]),
                        group["accumulator_role"] or "",
                        target["minimum_steps"],
                    ),
                )
    selected_count = db.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
    job_count = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    if job_count != selected_count:
        raise CombinatorialProgramSemanticError(
            f"selected/program ledgers differ: selected={selected_count}, jobs={job_count}"
        )
    db.commit()


def qualify_combinatorial_program_semantics(
    repo_root: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise CombinatorialProgramSemanticError(
            "config/output must be inside the repository and output must be fresh"
        )
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise CombinatorialProgramSemanticError("unsupported semantic config schema")
    if type(config.get("workers")) is not int or config["workers"] < 1:
        raise CombinatorialProgramSemanticError("workers must be a positive integer")
    maximum = config.get("maximum_outcomes")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise CombinatorialProgramSemanticError("maximum_outcomes must be a positive integer")
    required_folds = tuple(str(value) for value in config.get("required_folds", ()))
    if (
        not required_folds
        or len(set(required_folds)) != len(required_folds)
        or not set(required_folds).issubset(_FOLDS)
    ):
        raise CombinatorialProgramSemanticError("required_folds must be unique active fold names")
    policy = config.get("policy")
    if policy != {
        "semantic_ambiguity": "abstain",
        "evidence_basis": "computed_transform_consistency",
        "component_identity_in_model_state": False,
        "raw_family_count_sampling": False,
    }:
        raise CombinatorialProgramSemanticError("semantic policy changed")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    dataset = json.loads(paths["program_dataset_result"].read_text())
    if dataset.get("schema_version") != "forge.combinatorial_program_dataset.v1":
        raise CombinatorialProgramSemanticError("program dataset result schema changed")
    for label, artifact in (("products", "products.csv.gz"), ("programs", "programs.jsonl.gz")):
        if dataset["artifacts"][artifact] != config["inputs"][label]:
            raise CombinatorialProgramSemanticError(
                f"program dataset does not authenticate {label}"
            )
    contracts, vocabulary = _contracts(config)
    if set(config["expected_selected_counts"]) != set(contracts):
        raise CombinatorialProgramSemanticError("selected population and contracts differ")
    registry_pins = [
        (paths[name], config["inputs"][name]["sha256"]) for name in config["registries"]
    ]
    adapters = load_assembly_libraries(registry_pins, expected_families=contracts)
    for family, contract in contracts.items():
        expected_roles = list(adapters[family].roles)
        if family == "ugi_3cr_agile":
            expected_roles.append("assembly_introduced")
        mapped = sorted(
            {
                atom.GetAtomMapNum()
                for atom in adapters[family].reaction.forward.GetProductTemplate(0).GetAtoms()
                if atom.GetAtomMapNum()
            }
        )
        raw_positions = [f"map_{value}" for value in mapped]
        if family == "ugi_3cr_agile":
            raw_positions.append("template_introduced_0")
        aliases = contract["position_aliases"] or {value: value for value in raw_positions}
        if (
            set(contract["roles"]) != set(expected_roles)
            or set(aliases) != set(raw_positions)
            or set(aliases.values()) != set(contract["core_positions"])
        ):
            raise CombinatorialProgramSemanticError(
                f"semantic contract differs from authenticated registry: {family}"
            )
    atom_vocabulary = load_atom_vocabulary(paths["atom_vocabulary"])
    source_root = Path(__file__).resolve().parents[2]
    source_names = (
        "forge/corpus/combinatorial_program_semantics.py",
        "forge/assembly/library_semantics.py",
        "forge/assembly/library_programs.py",
        "forge/assembly/families.py",
        "forge/model/synthesis_program_graph.py",
        "forge/model/reaction_program_conditioning.py",
        "forge/potency/annotations.py",
        "forge/synthesis/engine/qualified_forward.py",
        "experiments/phase1/multireaction/combinatorial_program_semantics.py",
    )
    sources = {name: str(sha256_file(source_root / name)) for name in source_names}
    config_hash = str(sha256_file(config_path))
    start = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".library-semantics-", dir=output.parent) as temporary:
        work = Path(temporary)
        db = sqlite3.connect(work / "work.sqlite")
        selected_counts = _selected(db, paths["products"], config["expected_selected_counts"])
        _load_jobs(db, paths["programs"])
        total = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        success: Counter[tuple[str, str]] = Counter()
        failure: Counter[tuple[str, str, str]] = Counter()
        depths: Counter[tuple[str, int]] = Counter()
        observed_roles: dict[str, set[str]] = {family: set() for family in contracts}
        observed_positions: dict[str, set[str]] = {family: set() for family in contracts}
        maximum_atoms = maximum_closures = maximum_components = 0
        representation_digest = hashlib.sha256()
        ugi_source_mappings: Counter[int] = Counter()
        jobs = (
            (pid, family, components, witness, accumulator)
            for pid, family, components, witness, accumulator in db.execute(
                "SELECT pid,family,components,witness,accumulator_role FROM jobs ORDER BY pid"
            )
        )
        worker_args = (
            registry_pins,
            tuple(sorted(contracts)),
            {family: value["position_aliases"] for family, value in contracts.items()},
            paths["ugi_registry"],
            paths["ugi_variant"],
            maximum,
        )
        with (
            _gzip_text(work / "semantic_records.jsonl.gz") as semantic_handle,
            _gzip_text(work / "semantic_failures.jsonl.gz") as failure_handle,
        ):
            for index, annotation in enumerate(
                _results(jobs, config["workers"], worker_args), start=1
            ):
                pid = annotation["product_id"]
                family, fold, smiles, source_weight, training_weight, depth = db.execute(
                    """SELECT s.family,s.fold,s.smiles,s.source_weight,s.training_weight,j.depth
                    FROM selected s JOIN jobs j ON s.pid=j.pid WHERE s.pid=?""",
                    (pid,),
                ).fetchone()
                if not annotation["ok"]:
                    record = {
                        "schema_version": SEMANTIC_SCHEMA,
                        "product_id": pid,
                        "reaction_family": family,
                        "fold": fold,
                        "program_depth": depth,
                        "disposition": "abstain",
                        "reason": annotation["reason"],
                        "detail": annotation["detail"],
                    }
                    failure[(family, fold, annotation["reason"])] += 1
                    failure_handle.write(_json(record) + "\n")
                    continue
                if annotation["canonical_smiles"] != smiles:
                    raise CombinatorialProgramSemanticError("semantic/source SMILES changed")
                contract = contracts[family]
                roles = annotation["origin_roles"]
                raw_positions = annotation["core_positions"]
                namespaced = [
                    f"{family}:{value}" if value else "exterior" for value in raw_positions
                ]
                if (
                    set(roles) != set(contract["roles"])
                    or not {value for value in raw_positions if value}.issubset(
                        contract["core_positions"]
                    )
                    or depth not in contract["allowed_depths"]
                ):
                    raise CombinatorialProgramSemanticError(
                        f"observed semantics differ from contract: {pid}"
                    )
                fixed = (
                    [index for index, value in enumerate(raw_positions) if value]
                    if contract["fix_core_atoms"]
                    else []
                )
                if family == "ugi_3cr_agile" and fixed != annotation["ugi_fixed_atom_indices"]:
                    raise CombinatorialProgramSemanticError("settled Ugi fixed core changed")
                try:
                    graph = tensorize_synthesis_program_product(
                        record_id=pid,
                        program_id=family,
                        canonical_product_smiles=smiles,
                        atom_roles=roles,
                        atom_core_positions=namespaced,
                        program_depth=depth,
                        vocabulary=vocabulary,
                        atom_vocabulary=atom_vocabulary,
                        fixed_atom_indices=fixed,
                    )
                except SynthesisProgramGraphError as error:
                    record = {
                        "schema_version": SEMANTIC_SCHEMA,
                        "product_id": pid,
                        "reaction_family": family,
                        "fold": fold,
                        "program_depth": depth,
                        "disposition": "abstain",
                        "reason": "shared_tensor_representation_failed",
                        "detail": str(error),
                    }
                    failure[(family, fold, record["reason"])] += 1
                    failure_handle.write(_json(record) + "\n")
                    continue
                record = {
                    "schema_version": SEMANTIC_SCHEMA,
                    "product_id": pid,
                    "reaction_family": family,
                    "fold": fold,
                    "canonical_smiles": smiles,
                    "program_depth": depth,
                    "origin_roles": roles,
                    "core_positions": raw_positions,
                    "fixed_atom_indices": fixed,
                    "source_weight": source_weight,
                    "training_sampling_weight": training_weight,
                    "evidence_basis": "computed_transform_consistency",
                    "disposition": "admit_transform_consistency",
                }
                if annotation["ugi_source_mapping_multiplicity"] is not None:
                    record["ugi_source_mapping_multiplicity"] = annotation[
                        "ugi_source_mapping_multiplicity"
                    ]
                encoded = (_json(record) + "\n").encode()
                representation_digest.update(encoded)
                semantic_handle.write(encoded.decode())
                success[(family, fold)] += 1
                depths[(family, depth)] += 1
                observed_roles[family].update(roles)
                observed_positions[family].update(value for value in raw_positions if value)
                maximum_atoms = max(maximum_atoms, graph.node_count)
                maximum_closures = max(maximum_closures, graph.graph.closure_count)
                maximum_components = max(maximum_components, graph.component_count)
                if family == "ugi_3cr_agile":
                    ugi_source_mappings[annotation["ugi_source_mapping_multiplicity"]] += 1
                if index % 5000 == 0:
                    print(f"qualified {index}/{total} semantic programs", flush=True)
        accounted = sum(success.values()) + sum(failure.values())
        fold_counts = {
            family: {fold: success[(family, fold)] for fold in _FOLDS}
            for family in sorted(contracts)
        }
        selected_by_family = {
            family: {fold: selected_counts[(family, fold)] for fold in _FOLDS}
            for family in sorted(contracts)
        }
        gates = {
            "all_selected_products_accounted": accounted == total,
            "every_family_has_exact_required_fold_semantics": all(
                all(folds[fold] > 0 for fold in required_folds) for folds in fold_counts.values()
            ),
            "all_observed_roles_equal_contract": all(
                observed_roles[family] == set(contract["roles"])
                for family, contract in contracts.items()
            ),
            "all_declared_core_positions_observed": all(
                observed_positions[family] == set(contract["core_positions"])
                for family, contract in contracts.items()
            ),
            "full_graph_support_preserved": maximum_atoms
            <= int(config["support_bounds"]["maximum_heavy_atoms"])
            and maximum_closures <= int(config["support_bounds"]["maximum_closures"]),
            "raw_family_count_sampling_absent": True,
        }
        status = "pass" if all(gates.values()) else "blocked"
        db.close()
        (work / "work.sqlite").unlink()
        for name, pin in config["inputs"].items():
            resolve_pin(pin, repo, label=name)
        if str(sha256_file(config_path)) != config_hash or any(
            str(sha256_file(source_root / name)) != digest for name, digest in sources.items()
        ):
            raise CombinatorialProgramSemanticError(
                "code or configuration changed during semantic qualification"
            )
        artifacts = {
            path.name: {
                "path": str((output / path.name).relative_to(repo)),
                "sha256": str(sha256_file(path)),
            }
            for path in sorted(work.iterdir())
        }
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {"path": str(config_path.relative_to(repo)), "sha256": config_hash},
            "inputs": config["inputs"],
            "sources": sources,
            "artifacts": artifacts,
            "environment": {"rdkit": rdBase.rdkitVersion, "workers": config["workers"]},
            "duration_seconds": time.monotonic() - start,
            "selected_fold_counts": selected_by_family,
            "semantic_fold_counts": fold_counts,
            "semantic_depth_counts": {
                family: {
                    str(depth): count
                    for (observed_family, depth), count in sorted(depths.items())
                    if observed_family == family
                }
                for family in sorted(contracts)
            },
            "failures": {
                family: {
                    fold: {
                        reason: count
                        for (observed_family, observed_fold, reason), count in sorted(
                            failure.items()
                        )
                        if observed_family == family and observed_fold == fold
                    }
                    for fold in _FOLDS
                }
                for family in sorted(contracts)
            },
            "representation": {
                "records": sum(success.values()),
                "selected_records": total,
                "abstentions": sum(failure.values()),
                "sha256": representation_digest.hexdigest(),
                "maximum_heavy_atoms": maximum_atoms,
                "maximum_closures": maximum_closures,
                "maximum_origin_components": maximum_components,
                "program_vocabulary": {
                    "program_states": list(vocabulary.program_states),
                    "role_states": list(vocabulary.role_states),
                    "core_position_states": list(vocabulary.core_position_states),
                    "maximum_steps": vocabulary.maximum_steps,
                },
                "fixed_core_programs": sorted(
                    family for family, value in contracts.items() if value["fix_core_atoms"]
                ),
                "universal_selected_mapping": sum(failure.values()) == 0,
                "ugi_source_mapping_multiplicity_distribution": {
                    str(value): count for value, count in sorted(ugi_source_mappings.items())
                },
            },
            "gates": gates,
            "training_calls": 0,
            "generator_sampling_calls": 0,
            "nonclaims": [
                "Exact atom semantics are computed transform consistency, not source-executed synthesis or route closure.",
                "Ambiguous origin/core assignments abstain; no arbitrary atom mapping enters model state.",
                "The semantic cache source contains no component identifiers, component SMILES, fingerprints or biological labels.",
                "Passing this representation gate is not evidence of trained twelve-family performance or improved realism.",
            ],
        }
        (work / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.rename(work, output)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "SEMANTIC_SCHEMA",
    "CombinatorialProgramSemanticError",
    "qualify_combinatorial_program_semantics",
]

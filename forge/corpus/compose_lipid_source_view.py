"""Bind supplied components to the intersection of frozen preparation populations.

This metadata-only view cannot admit training rows. Both split versions, prior product
exclusions and the global union of known protected components are enforced before a
consumer receives a product. Unassigned full-universe targets remain unassigned.
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_protection import verify_product_protection
from forge.corpus.compose_lipid_supplement import compact, rows

CONFIG_SCHEMA = "forge.compose_lipid_source_view_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_source_view.v1"
POLICY = {
    "seed": 0,
    "required_old_split": "train",
    "required_corrected_split": "train",
    "known_component_protection": "historical_union_corrected_selected_test",
    "unassigned_targets_admitted": False,
    "product_molecules_parsed": 0,
    "training_admitted": False,
}
IMPLEMENTATION = (
    "forge/corpus/compose_lipid_source_view.py",
    "forge/corpus/compose_lipid_supplement.py",
    "forge/corpus/compose_lipid_protection.py",
    "forge/corpus/compose_lipid.py",
    "forge/core/hashing.py",
)


def pin(repo: Path, path: Path) -> dict:
    return {"path": str(path.resolve().relative_to(repo)), "sha256": str(sha256_file(path))}


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def classify(
    target: str,
    family: str,
    old_split: str,
    corrected_split: str,
    components: list,
    protected_components: set[str],
    prior: dict | None,
) -> dict:
    """Classify metadata; no molecular graph or reagent-code joins are needed."""
    if old_split not in {"train", "calibration", "heldout", "quarantine", "reference"}:
        raise ComposeLipidError(f"Unknown old split: {old_split}")
    if corrected_split not in {"train", "calibration", "test", "reference"}:
        raise ComposeLipidError(f"Unknown corrected split: {corrected_split}")
    if not components:
        raise ComposeLipidError("Missing complete source components")
    seen = set()
    for item in components:
        if (
            not isinstance(item, list)
            or len(item) != 3
            or not all(isinstance(v, str) and v for v in item[:2])
            or type(item[2]) is not int
            or item[2] < 1
            or tuple(item[:2]) in seen
        ):
            raise ComposeLipidError("Invalid collapsed source component instance")
        seen.add(tuple(item[:2]))
    if old_split == "train" and (
        prior is None
        or prior.get("target_id") != target
        or prior.get("family") != family
        or type(prior.get("eligible_for_program_preparation")) is not bool
        or not prior.get("constitution_id")
        or prior.get("training_admitted") is not False
    ):
        raise ComposeLipidError("Missing or inconsistent authenticated prior product protection")
    reasons = []
    if old_split != "train":
        reasons.append("old_split_not_train")
    if corrected_split != "train":
        reasons.append("corrected_split_not_train")
    if old_split == "train" and not prior["eligible_for_program_preparation"]:
        reasons.append("prior_protected_product")
    blocked = sorted({identity for _, identity, _ in components} & protected_components)
    if blocked:
        reasons.append("known_protected_component")
    return {
        "target_id": target,
        "family": family,
        "old_split": old_split,
        "corrected_split": corrected_split,
        "component_instances": components,
        "protected_component_ids": blocked,
        "exclusion_reasons": reasons,
        "eligible_for_program_preparation": not reasons,
        "constitution_id": prior["constitution_id"] if old_split == "train" else None,
        "training_admitted": False,
    }


def _load(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("Source-view schema or scientific scope changed")
    paths = {name: resolve_pin(value, repo, label=name) for name, value in config["inputs"].items()}
    if set(paths) != {"intake", "component_policy", "product_protection"}:
        raise ComposeLipidError("Source-view input set changed")
    intake = json.loads(paths["intake"].read_text())
    if (
        intake.get("schema_version") != "forge.compose_lipid_supplement_intake.v1"
        or intake.get("status") != "target_component_joins_verified_training_unqualified"
        or any(intake["constructions"]["join_failures"].values())
        or any(intake["corrected_split"]["panel_failures"].values())
    ):
        raise ComposeLipidError("Supplement joins or corrected split are not authenticated")
    resolve_pin(intake["implementation"], repo, label="intake implementation")
    intake_config = json.loads(
        resolve_pin(intake["config"], repo, label="intake config").read_text()
    )
    if intake_config["inputs"] != intake["inputs"]:
        raise ComposeLipidError("Intake receipt/config input disagreement")
    authenticated = {
        name: resolve_pin(value, repo, label=f"intake.{name}")
        for name, value in intake["inputs"].items()
    }
    joins = resolve_pin(intake["artifacts"]["joins.sqlite"], repo, label="construction joins")
    component_policy = json.loads(paths["component_policy"].read_text())
    if (
        component_policy.get("schema_version")
        != "forge.compose_lipid_supplement_component_exclusion_policy.v1"
        or component_policy.get("training_admitted") is not False
        or any(
            component_policy["inputs"][name] != intake["inputs"][name]
            for name in ("precursors", "selected_groups")
        )
    ):
        raise ComposeLipidError("Component exclusion policy disagrees with supplemental inputs")
    for group in ("inputs", "implementation"):
        for name, value in component_policy[group].items():
            resolve_pin(value, repo, label=f"component_policy.{group}.{name}")
    protected = set()
    for item in component_policy["blocked_components"]:
        identity = item["component_id"]
        if identity in protected or not (
            item["historically_protected"] or item["corrected_test_component"]
        ):
            raise ComposeLipidError("Invalid protected component policy entry")
        protected.add(identity)
    if len(protected) != component_policy["totals"]["union_protected_components"]:
        raise ComposeLipidError("Protected component policy count mismatch")
    protection = verify_product_protection(repo, paths["product_protection"])
    if protection["inputs"]["import_result"] != component_policy["inputs"]["import_result"]:
        raise ComposeLipidError(
            "Historical product and component protections use different imports"
        )
    imported = json.loads(
        resolve_pin(protection["inputs"]["import_result"], repo, label="import").read_text()
    )
    if imported["artifacts"]["corpus.sqlite"] != intake["inputs"]["corpus"]:
        raise ComposeLipidError("Product protection and supplemental joins use different corpora")
    prior = {
        row["target_id"]: row
        for row in rows(repo / protection["artifacts"]["preparation_view.jsonl.gz"]["path"])
    }
    return config, authenticated, joins, protected, prior, intake


def _evaluate(loaded):
    _, paths, joins, protected, prior, intake = loaded
    counts = defaultdict(Counter)
    with closing(sqlite3.connect(joins.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("ATTACH DATABASE ? AS original", (paths["corpus"].as_uri() + "?mode=ro",))
        query = (
            "SELECT a.target_id,a.family,o.forge_split,a.split,c.instances,c.basis,c.source_line "
            "FROM assignments a JOIN original.assignments o USING(target_id) "
            "JOIN constructions c USING(target_id) ORDER BY a.family,a.target_id"
        )
        for target, family, old, corrected, instances, basis, line in db.execute(query):
            row = classify(
                target, family, old, corrected, json.loads(instances), protected, prior.get(target)
            )
            row.update(construction_basis=basis, construction_source_line=line)
            counts[family]["rows"] += 1
            counts[family]["eligible_for_program_preparation"] += row[
                "eligible_for_program_preparation"
            ]
            counts[family].update(row["exclusion_reasons"])
            yield row
    expected = sum(intake["corrected_split"]["split_counts"].values())
    if sum(c["rows"] for c in counts.values()) != expected:
        raise ComposeLipidError("Source view lost selected targets")


def _summary(records):
    counts = defaultdict(Counter)
    for row in records:
        counts[row["family"]]["rows"] += 1
        counts[row["family"]]["eligible_for_program_preparation"] += row[
            "eligible_for_program_preparation"
        ]
        counts[row["family"]].update(row["exclusion_reasons"])
    return {
        "by_family": {family: dict(count) for family, count in sorted(counts.items())},
        "totals": dict(sum(counts.values(), Counter())),
        "reason_counts_overlap": True,
        "all_family_training_ready": False,
        "global_partition_complete": False,
    }


def build_source_view(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("Source view output must be fresh and inside the repository")
    loaded = _load(repo, config_path)
    records = list(_evaluate(loaded))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".source-view-", dir=output.parent) as temporary:
        stage = Path(temporary)
        ledger = stage / "preparation.jsonl.gz"
        with ledger.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
                for row in records:
                    stream.write((compact(row) + "\n").encode())
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "protected_source_preparation_only",
            "config": pin(repo, config_path),
            "inputs": loaded[0]["inputs"],
            "implementation": {name: pin(repo, repo / name) for name in IMPLEMENTATION},
            "policy": POLICY,
            "summary": _summary(records),
            "artifact": {
                "path": str((output / ledger.name).relative_to(repo)),
                "sha256": str(sha256_file(ledger)),
            },
        }
        dump(stage / "result.json", result)
        os.rename(stage, output)
    return result


class SourcePreparationCorpus:
    """Only expose product strings after both split and identity exclusions pass."""

    def __init__(self, repo_root: Path, result_path: Path):
        self.repo = repo_root.resolve()
        result = json.loads((self.repo / result_path).read_text())
        if (
            result.get("schema_version") != RESULT_SCHEMA
            or result.get("status") != "protected_source_preparation_only"
            or result.get("policy") != POLICY
            or set(result.get("implementation", {})) != set(IMPLEMENTATION)
        ):
            raise ComposeLipidError("Source preparation receipt scope changed")
        for name, value in result["implementation"].items():
            if resolve_pin(value, self.repo, label=name) != self.repo / name:
                raise ComposeLipidError("Source-view implementation path substitution")
        loaded = _load(self.repo, resolve_pin(result["config"], self.repo, label="config"))
        ledger = resolve_pin(result["artifact"], self.repo, label="preparation ledger")
        observed = list(rows(ledger))
        if (
            result["inputs"] != loaded[0]["inputs"]
            or observed != list(_evaluate(loaded))
            or result["summary"] != _summary(observed)
        ):
            raise ComposeLipidError("Source preparation ledger differs from authenticated inputs")
        self.result = result
        self.corpus = loaded[1]["corpus"]
        self.precursors = loaded[1]["precursors"]
        self.preparation = {
            row["target_id"]: row for row in observed if row["eligible_for_program_preparation"]
        }

    def iter_preparation_records(self, *, family: str):
        if family not in self.result["summary"]["by_family"]:
            raise ComposeLipidError(f"Unknown source family: {family}")
        with closing(sqlite3.connect(self.corpus.as_uri() + "?mode=ro", uri=True)) as db:
            for target, payload in db.execute(
                "SELECT t.target_id,t.payload FROM assignments a JOIN targets t USING(target_id) "
                "WHERE a.forge_split='train' AND a.family=? ORDER BY t.target_id",
                (family,),
            ):
                if target in self.preparation:
                    yield {"source": json.loads(payload), "preparation": self.preparation[target]}

    def iter_training_records(self):
        raise ComposeLipidError("Source preparation does not establish training qualification")

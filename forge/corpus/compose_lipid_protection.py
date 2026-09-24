"""Enforced prior-product exclusions for COMPOSE training preparation.

The provider corpus remains immutable. Exact string matches establish overlap and are
propagated to every TRAIN alias of the already-authenticated constitutional identity.
Nonmatches do not establish graph disjointness or grant training admission.
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid import verify_compose_lipid

CONFIG_SCHEMA = "forge.compose_lipid_product_protection_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_product_protection.v1"
PROTECTED_FOLDS = frozenset({"validation", "calibration", "val", "heldout", "test"})
KNOWN_FOLDS = PROTECTED_FOLDS | {"train", "unassigned"}
IMPLEMENTATION = (
    "forge/corpus/compose_lipid_protection.py",
    "forge/corpus/compose_lipid.py",
    "forge/core/hashing.py",
)
POLICY = {
    "identity_matching": "exact_strings_propagated_to_authenticated_train_identity_aliases",
    "nonmatches": "unresolved_not_disjoint",
    "graph_parsing": False,
    "source_splits_changed": False,
    "training_calls": 0,
    "seed": 0,
}


def _pin(repo: Path, path: Path) -> dict:
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read(path: Path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def _load(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("product-protection configuration or scientific scope changed")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    if set(paths) != {"import_result", "prior_construction"}:
        raise ComposeLipidError("product-protection input set changed")
    imported = verify_compose_lipid(repo, paths["import_result"])
    catalogue = json.loads(
        (repo / imported["artifacts"]["program_catalogue.json"]["path"]).read_text()
    )
    if set(catalogue) != set(config["expected_families"]):
        raise ComposeLipidError("product-protection view must cover every imported program family")
    return config, paths, imported


def classify_prior_products(train_rows: list[dict], prior_rows) -> list[dict]:
    """Protect known overlaps without equating provider IDs with chemical identity.

    TRAIN constitution IDs come from the authenticated import's train_checks table.
    A single positive overlap protects every alias, including aliases in other families.
    """
    targets = set()
    strings = {}
    identities = set()
    for row in train_rows:
        if row["target_id"] in targets or not row["constitution_id"]:
            raise ComposeLipidError("duplicate or uninspected target in preparation population")
        targets.add(row["target_id"])
        identities.add(row["constitution_id"])
        prior = strings.setdefault(row["constitution"], row["constitution_id"])
        if prior != row["constitution_id"]:
            raise ComposeLipidError("one source string has conflicting authenticated identities")
    matches = defaultdict(set)
    for row in prior_rows:
        fold = row.get("prior_development_split") or "unassigned"
        if fold not in KNOWN_FOLDS:
            raise ComposeLipidError(f"unsupported prior development fold: {fold!r}")
        identity = strings.get(row["constitution"])
        if identity is not None:
            matches[identity].add((row["target_id"], fold))
    output = []
    for row in sorted(train_rows, key=lambda item: (item["family"], item["target_id"])):
        observed = matches[row["constitution_id"]]
        protected = any(fold in PROTECTED_FOLDS for _, fold in observed)
        output.append(
            {
                "target_id": row["target_id"],
                "family": row["family"],
                "constitution_id": row["constitution_id"],
                "prior_matches": [
                    {"source_target_id": key, "fold": fold} for key, fold in sorted(observed)
                ],
                "preparation_disposition": (
                    "excluded_prior_protected_product"
                    if protected
                    else "pending_program_and_global_holdout_qualification"
                ),
                "eligible_for_program_preparation": not protected,
                "training_admitted": False,
            }
        )
    return output


def _evaluate(repo: Path, loaded):
    config, paths, imported = loaded
    db_path = repo / imported["artifacts"]["corpus.sqlite"]["path"]
    with sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True) as db:
        # A left join preserves missing inspection records so validation can reject them.
        rows = [
            dict(zip(("target_id", "family", "constitution", "constitution_id"), row, strict=True))
            for row in db.execute(
                "SELECT a.target_id,a.family,t.constitution,c.constitution_id "
                "FROM assignments a JOIN targets t USING(target_id) "
                "LEFT JOIN train_checks c USING(target_id) WHERE a.forge_split='train' "
                "ORDER BY a.family,a.target_id"
            )
        ]
    if set(row["family"] for row in rows) != set(config["expected_families"]):
        raise ComposeLipidError("preparation population lost an imported family")
    return classify_prior_products(rows, _read(paths["prior_construction"]))


def _summary(rows: list[dict]) -> dict:
    counts = defaultdict(Counter)
    for row in rows:
        value = counts[row["family"]]
        value["rows"] += 1
        value["excluded_prior_protected_product"] += not row["eligible_for_program_preparation"]
        value["eligible_for_program_preparation"] += row["eligible_for_program_preparation"]
        value["prior_exact_identity_overlap"] += bool(row["prior_matches"])
    return {
        "by_family": {
            family: dict(sorted(values.items())) for family, values in sorted(counts.items())
        },
        "totals": dict(sorted(sum(counts.values(), Counter()).items())),
        "training_admitted": 0,
        "global_protection_complete": False,
        "training_ready": False,
    }


def build_product_protection(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo) or not config_path.is_relative_to(repo):
        raise ComposeLipidError("protection output must be fresh and inside the repository")
    loaded = _load(repo, config_path)
    rows = _evaluate(repo, loaded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".protection-", dir=output.parent) as temporary:
        stage = Path(temporary)
        ledger = stage / "preparation_view.jsonl.gz"
        with ledger.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
                for row in rows:
                    stream.write(
                        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                    )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "prior_product_exclusions_enforced_training_unqualified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": _pin(repo, config_path),
            "inputs": loaded[0]["inputs"],
            "implementation": {name: _pin(repo, repo / name) for name in IMPLEMENTATION},
            "policy": POLICY,
            "summary": _summary(rows),
            "artifacts": {
                ledger.name: {
                    "path": (output / ledger.name).relative_to(repo).as_posix(),
                    "sha256": str(sha256_file(ledger)),
                }
            },
        }
        _dump(stage / "result.json", result)
        os.rename(stage, output)
    return result


def verify_product_protection(repo_root: Path, result_path: Path) -> dict:
    repo = repo_root.resolve()
    result = json.loads((repo / result_path).read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "prior_product_exclusions_enforced_training_unqualified"
        or result.get("policy") != POLICY
        or set(result.get("implementation", {})) != set(IMPLEMENTATION)
        or set(result.get("artifacts", {})) != {"preparation_view.jsonl.gz"}
    ):
        raise ComposeLipidError("product-protection receipt scope or provenance changed")
    for name, pin in result["implementation"].items():
        if resolve_pin(pin, repo, label=name) != (repo / name).resolve():
            raise ComposeLipidError("protection implementation path substitution")
    ledger = resolve_pin(
        result["artifacts"]["preparation_view.jsonl.gz"], repo, label="protection ledger"
    )
    loaded = _load(repo, resolve_pin(result["config"], repo, label="config"))
    expected = _evaluate(repo, loaded)
    if (
        result["inputs"] != loaded[0]["inputs"]
        or list(_read(ledger)) != expected
        or result["summary"] != _summary(expected)
    ):
        raise ComposeLipidError("protection ledger or exclusion accounting differs from source")
    return result


class ProtectedPreparationCorpus:
    """Read only records eligible for program preparation; never expose a training iterator."""

    def __init__(self, repo_root: Path, result_path: Path):
        self.repo = repo_root.resolve()
        self.result = verify_product_protection(self.repo, result_path)
        self.imported = json.loads(
            resolve_pin(
                self.result["inputs"]["import_result"], self.repo, label="import"
            ).read_text()
        )
        ledger = self.repo / self.result["artifacts"]["preparation_view.jsonl.gz"]["path"]
        self.preparation = {row["target_id"]: row for row in _read(ledger)}

    def iter_preparation_records(self, *, family: str):
        if family not in self.result["summary"]["by_family"]:
            raise ComposeLipidError(f"unknown program family: {family}")
        path = self.repo / self.imported["artifacts"]["corpus.sqlite"]["path"]
        with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as db:
            for target, payload in db.execute(
                "SELECT t.target_id,t.payload FROM assignments a JOIN targets t USING(target_id) "
                "WHERE a.forge_split='train' AND a.family=? ORDER BY t.target_id",
                (family,),
            ):
                row = self.preparation[target]
                if row["eligible_for_program_preparation"]:
                    yield {"source": json.loads(payload), "preparation": row}

    def iter_training_records(self):
        raise ComposeLipidError(
            "product exclusions are not complete program, precursor or training qualification"
        )

"""Bind every v8 target to its exact canonical precursor structures.

The output repairs a model-split metadata problem without changing corpus
membership.  It resolves source precursors from the bound source dictionary and
virtual precursors from the exact task referenced by the supervision manifest.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from compose_lipid.data.assets import sha256_file
from compose_lipid.data.post_instruction_component_resolver_v8_1 import (
    canonical_component,
    component_identity,
    virtual_task_components,
)
from compose_lipid.data.source_inventory import publish_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/corpus/post_instruction_component_manifest_v8_1.json"


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        yield from map(json.loads, handle)


def output_sha(receipt: dict, name: str) -> str:
    value = receipt["outputs"][name]
    return value["sha256"] if isinstance(value, dict) else value


def payload_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def deterministic_gzip(path: Path, records) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as raw:
        temporary = Path(raw.name)
        with gzip.GzipFile(
            fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6
        ) as compressed, io.TextIOWrapper(compressed, encoding="utf-8") as stream:
            count = 0
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
                count += 1
        raw.flush()
        os.fsync(raw.fileno())
    if path.exists():
        if sha256_file(path) != sha256_file(temporary):
            temporary.unlink()
            raise ValueError("refusing to overwrite changed component manifest")
        temporary.unlink()
    else:
        os.replace(temporary, path)
    return count, sha256_file(path)


def bind_receipt(config: dict, key: str, bindings: dict) -> tuple[Path, dict]:
    path = ROOT / config[f"{key}_receipt"]
    expected = config[f"{key}_receipt_sha256"]
    if sha256_file(path) != expected:
        raise ValueError("changed bound receipt: " + key)
    receipt = json.loads(path.read_text())
    if receipt.get("complete") is not True:
        raise ValueError("incomplete bound receipt: " + key)
    bindings[str(path.relative_to(ROOT))] = expected
    return path.parent, receipt


def source_instance(instance: dict, source_precursors: dict[str, str]) -> dict:
    identifier = instance["precursor_id"]
    constitution = source_precursors.get(identifier)
    if constitution is None:
        raise ValueError("unresolved source precursor: " + identifier)
    roles = instance.get("roles") or [instance.get("role")]
    role = "+".join(sorted(value for value in roles if value))
    if not role:
        raise ValueError("source precursor lacks role")
    canonical = canonical_component(constitution)
    return {
        "role": role,
        "component_id": component_identity(canonical),
        "constitution": canonical,
    }


def save(connection: sqlite3.Connection, record: dict) -> None:
    connection.execute(
        "INSERT INTO resolved VALUES (?,?)",
        (
            record["target_id"],
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        ),
    )


def build() -> dict:
    config = json.loads(CONFIG.read_text())
    if any(config[key] for key in ("biology_used", "beae_outcomes_used", "gpu_used")):
        raise ValueError("component binding must remain chemistry-only and CPU-only")
    if config["training_admissible"]:
        raise ValueError("component binding cannot admit training")

    bindings = {str(CONFIG.relative_to(ROOT)): sha256_file(CONFIG)}
    release_dir, release_receipt = bind_receipt(config, "release", bindings)
    supervision_dir, supervision_receipt = bind_receipt(
        config, "supervision", bindings
    )
    source_dir, source_receipt = bind_receipt(
        config, "source_supervision", bindings
    )
    split_dir, split_receipt = bind_receipt(config, "prior_split", bindings)
    ugi_dir, ugi_receipt = bind_receipt(config, "ugi_catalogue", bindings)

    release_path = release_dir / "accepted_targets.jsonl.gz"
    supervision_path = supervision_dir / "manifest.jsonl.gz"
    source_precursor_path = source_dir / "precursors.jsonl"
    split_groups_path = split_dir / "split_groups.jsonl.gz"
    ugi_path = ugi_dir / "precursors.jsonl"
    for path, receipt in (
        (release_path, release_receipt),
        (supervision_path, supervision_receipt),
        (source_precursor_path, source_receipt),
        (split_groups_path, split_receipt),
        (ugi_path, ugi_receipt),
    ):
        expected = output_sha(receipt, path.name)
        if sha256_file(path) != expected:
            raise ValueError("changed bound input: " + str(path))
        bindings[str(path.relative_to(ROOT))] = expected

    source_precursors = {
        row["precursor_id"]: row["constitution"] for row in rows(source_precursor_path)
    }
    ugi_precursors = {row["id"]: row for row in rows(ugi_path)}
    source_metadata = {
        row["target_id"]: {
            "source_pmids": row.get("source_pmids") or [],
            "source_names": row.get("source_names") or [],
        }
        for row in rows(split_groups_path)
        if row["source_anchor"]
    }

    out = ROOT / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    database = out / ".component_manifest.sqlite"
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE locators (path TEXT, line INTEGER, target_id TEXT PRIMARY KEY, "
        "family TEXT, file_sha TEXT, payload_sha TEXT)"
    )
    connection.execute(
        "CREATE TABLE resolved (target_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
    )

    source_count = virtual_count = 0
    with gzip.open(supervision_path, "rt") as handle:
        for supervision in map(json.loads, handle):
            target_id = supervision["target_id"]
            family = supervision["family"]
            if supervision["source_anchor"]:
                components = [
                    source_instance(instance, source_precursors)
                    for instance in supervision["precursor_instances"]
                ]
                metadata = source_metadata.get(target_id, {})
                save(
                    connection,
                    {
                        "target_id": target_id,
                        "family": family,
                        "source_anchor": True,
                        "component_instances": components,
                        "component_ids": sorted(
                            {item["component_id"] for item in components}
                        ),
                        "source_pmids": metadata.get("source_pmids", []),
                        "source_names": metadata.get("source_names", []),
                        "binding_basis": "source_precursor_dictionary",
                    },
                )
                source_count += 1
                continue
            reference = supervision.get("task_reference") or supervision.get(
                "program_reference"
            )
            if not reference:
                raise ValueError("virtual supervision lacks exact input reference")
            connection.execute(
                "INSERT INTO locators VALUES (?,?,?,?,?,?)",
                (
                    reference["path"],
                    reference["line"],
                    target_id,
                    family,
                    reference["sha256"],
                    reference["payload_sha256"],
                ),
            )
            virtual_count += 1
    connection.commit()

    for (relative,) in connection.execute("SELECT DISTINCT path FROM locators"):
        path = ROOT / relative
        requested = defaultdict(list)
        for line, target_id, family, file_sha, payload_sha in connection.execute(
            "SELECT line,target_id,family,file_sha,payload_sha FROM locators "
            "WHERE path=?",
            (relative,),
        ):
            requested[line].append((target_id, family, file_sha, payload_sha))
        file_hashes = {
            row[2] for values in requested.values() for row in values
        }
        if len(file_hashes) != 1 or sha256_file(path) != next(iter(file_hashes)):
            raise ValueError("changed exact task/program input: " + relative)
        bindings[relative] = next(iter(file_hashes))
        found = set()
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            for line_number, line in enumerate(handle, 1):
                requested_rows = requested.get(line_number)
                if requested_rows is None:
                    continue
                task = json.loads(line)
                actual_payload = payload_digest(task)
                families = {row[1] for row in requested_rows}
                if len(families) != 1:
                    raise ValueError("one task row bound to multiple families")
                family = next(iter(families))
                components = virtual_task_components(family, task, ugi_precursors)
                for target_id, _, _, expected_payload in requested_rows:
                    if actual_payload != expected_payload:
                        raise ValueError(
                            "changed referenced task payload: " + target_id
                        )
                    save(
                        connection,
                        {
                            "target_id": target_id,
                            "family": family,
                            "source_anchor": False,
                            "component_instances": components,
                            "component_ids": sorted(
                                {item["component_id"] for item in components}
                            ),
                            "source_pmids": [],
                            "source_names": [],
                            "binding_basis": "exact_virtual_task_components",
                        },
                    )
                found.add(line_number)
        if found != set(requested):
            raise ValueError("missing referenced task rows in " + relative)
        connection.commit()

    resolved = connection.execute("SELECT COUNT(*) FROM resolved").fetchone()[0]
    if resolved != config["expected_targets"]:
        raise ValueError(f"component manifest incomplete: {resolved}")
    if source_count != config["expected_source_anchors"]:
        raise ValueError("source anchor count changed")

    release_members = {
        row["target_id"]: (row["primary_family"], row["source_anchor"])
        for row in rows(release_path)
    }
    if len(release_members) != config["expected_targets"]:
        raise ValueError("release membership changed")

    family_targets = Counter()
    family_components = defaultdict(set)
    global_components = set()
    role_components = defaultdict(set)
    instance_counts = Counter()

    def output_records():
        for target_id, payload in connection.execute(
            "SELECT target_id,payload FROM resolved ORDER BY target_id"
        ):
            record = json.loads(payload)
            expected = release_members.get(target_id)
            if expected != (record["family"], record["source_anchor"]):
                raise ValueError("component/release membership mismatch: " + target_id)
            family_targets[record["family"]] += 1
            instance_counts[record["family"]] += len(record["component_instances"])
            for item in record["component_instances"]:
                family_components[record["family"]].add(item["component_id"])
                global_components.add(item["component_id"])
                role_components[(record["family"], item["role"])].add(
                    item["component_id"]
                )
            yield record

    manifest_path = out / "components.jsonl.gz"
    count, manifest_sha = deterministic_gzip(manifest_path, output_records())
    if count != config["expected_targets"]:
        raise ValueError("published component count changed")

    summary = {
        "schema": "post_instruction_component_manifest_v8_1_summary",
        "targets": count,
        "source_anchors": source_count,
        "virtual_targets": virtual_count,
        "families": len(family_targets),
        "unique_component_constitutions": len(global_components),
        "family_target_counts": dict(sorted(family_targets.items())),
        "family_unique_component_counts": {
            family: len(values) for family, values in sorted(family_components.items())
        },
        "family_component_instance_counts": dict(sorted(instance_counts.items())),
        "family_role_unique_component_counts": [
            {"family": family, "role": role, "components": len(values)}
            for (family, role), values in sorted(role_components.items())
        ],
        "component_identity": "canonical_stereo_free_complete_precursor_constitution",
        "component_ids_globally_scoped": True,
        "complete_component_resolution": True,
        "biology_used": False,
        "beae_outcomes_used": False,
        "gpu_used": False,
        "training_admissible": False,
    }
    publish_json(out / "summary.json", summary)
    producers = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (
            Path(__file__),
            ROOT
            / "src/compose_lipid/data/post_instruction_component_resolver_v8_1.py",
        )
    }
    receipt = {
        "schema": "post_instruction_component_manifest_v8_1_receipt",
        "complete": True,
        "inputs": dict(sorted(bindings.items())),
        "producers": producers,
        "outputs": {
            "components.jsonl.gz": manifest_sha,
            "summary.json": sha256_file(out / "summary.json"),
        },
        "source_anchors": source_count,
        "virtual_targets": virtual_count,
        "biology_used": False,
        "beae_outcomes_used": False,
        "gpu_used": False,
        "training_admissible": False,
    }
    publish_json(out / "receipt.json", receipt)
    connection.close()
    database.unlink(missing_ok=True)
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))

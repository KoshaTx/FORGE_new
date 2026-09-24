"""Compile the v8 family-balanced graph measure after complete data qualification.

The unit is one unique constitutional graph. Source-role namespaces do not receive
separate mass. This v8 measure does not replace the older R1 realism weights and
does not authorize fitting or provide a training launcher.
"""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_source_view import dump, pin

POLICY = "equal_formal_family_mass_uniform_unique_constitutions"
COHORT_POLICY = "equal_qualified_family_mass_uniform_unique_constitutions"
SCHEMA = "forge.compose_lipid_training_measure.v1"


def qualified_cohort_families(
    repo: Path,
    cohort: Mapping[str, str],
    *,
    population: Mapping[str, str],
    verification: Mapping[str, str],
    evidence_index: Mapping[str, str],
    prepared: dict,
    evidence: dict,
    formal: list[str],
) -> list[str]:
    """Authenticate an explicit, frozen all-qualified cohort without admitting pending rows."""
    document = json.loads(resolve_pin(cohort, repo, label="qualified training cohort").read_text())
    families = sorted(prepared["by_family"])
    if (
        document.get("schema_version") != "forge.compose_lipid_qualified_cohort.v1"
        or document.get("selection") != "all_exact_eligible_records_in_pinned_population"
        or document.get("policy") != COHORT_POLICY
        or document.get("holdouts") != "unchanged"
        or document.get("authorization", {}).get("qualified_cohort_training") is not True
        or not document["authorization"].get("user_instruction")
        or document.get("inputs")
        != {
            "population": dict(population),
            "verification": dict(verification),
            "evidence_index": dict(evidence_index),
        }
        or not families
        or not set(families) <= set(formal)
        or document.get("families") != families
        or document.get("records") != prepared["totals"]["records"]
        or document.get("by_family") != prepared["by_family"]
        or document.get("records") != evidence["summary"]["exact"]
        or document.get("pending_records_excluded") != evidence["summary"]["pending"]
        or document.get("formal_families_without_training_support")
        != sorted(set(formal) - set(families))
        or any(type(n) is not int or n < 1 for n in prepared["by_family"].values())
    ):
        raise ValueError("Qualified cohort authorization, population or exclusions differ")
    return families


class TrainingMeasureUnavailableError(ValueError):
    """The complete source population cannot yet support the requested measure."""

    def __init__(self, blockers: list[str]):
        self.blockers = tuple(blockers)
        super().__init__("Training measure unavailable: " + "; ".join(blockers))


def compile_training_measure(
    repo: Path,
    output: Path,
    *,
    universe_config: Mapping[str, str],
    population: Mapping[str, str],
    verification: Mapping[str, str],
    policy: str,
    cohort: Mapping[str, str] | None = None,
) -> Path:
    """Write probabilities in qualified-loader order, or publish nothing on failure.

    The default requires all 23 formal families and all eligible rows. An explicit
    authenticated qualified cohort instead balances its represented families and
    records every exclusion. Both paths include every exact eligible graph once,
    preserve holdouts and reject source duplicates. Compilation is not admission.
    """
    repo, output = repo.resolve(), output.resolve()
    output.relative_to(repo)
    if output.exists():
        raise FileExistsError(output)
    if policy not in {POLICY, COHORT_POLICY} or (policy == COHORT_POLICY) != (cohort is not None):
        raise ValueError(f"Unsupported v8 weighting policy: {policy}")
    implementation = pin(repo, repo / "forge/corpus/compose_lipid_training_measure.py")
    if implementation["sha256"] != str(sha256_file(Path(__file__))):
        raise ValueError("The recorded measure implementation differs from the executing code")

    def read(value: Mapping[str, str]) -> dict:
        return json.loads(resolve_pin(value, repo, label="v8 training measure input").read_text())

    universe, prepared, verified = (read(p) for p in (universe_config, population, verification))
    if (
        universe.get("schema_version") != "forge.compose_lipid_universe_config.v1"
        or universe.get("policy", {}).get("intended_training_measure")
        != "equal_family_mass_after_all_admission_gates"
    ):
        raise ValueError("The v8 source contract does not specify equal formal-family mass")
    expected = universe["expected_family_rows"]
    references = universe["reference_families"]
    if (
        not isinstance(expected, dict)
        or any(
            not isinstance(k, str) or not k or type(v) is not int or v <= 0
            for k, v in expected.items()
        )
        or not isinstance(references, list)
        or any(not isinstance(k, str) for k in references)
        or len(set(references)) != len(references)
        or not set(references) <= set(expected)
    ):
        raise ValueError("Invalid complete-universe family contract")
    formal = sorted(set(expected) - set(references))
    if len(formal) != 23:
        raise ValueError("The v8 measure requires all 23 formal source families")
    imported = read(universe["inputs"]["import_result"])
    if (
        imported.get("schema_version") != "forge.compose_lipid_import.v1"
        or imported.get("sampling_policy") != "equal_family_mass_after_qualification"
    ):
        raise ValueError("The graph measure applies only to the qualified v8 source contract")
    if imported["summary"]["universe_by_family"] != expected or imported["summary"][
        "universe_rows"
    ] != sum(expected.values()):
        raise ValueError("Formal families differ from the authenticated full source import")
    if (
        prepared.get("schema_version") != "forge.unified_preparation_index.v1"
        or prepared.get("training_admitted") is not False
        or prepared.get("sampling_weights_fitted") is not False
        or verified.get("schema_version") != "forge.unified_preparation_verification.v1"
        or verified.get("inputs", {}).get("population") != dict(population)
        or verified.get("totals") != prepared["totals"]
        or verified.get("verified_tensor_shards") != prepared["tensor_shards"]
        or any(
            verified.get(name) is not True
            for name in (
                "all_exact_eligible_records_present_once",
                "source_identity_family_size_and_constitution_preserved",
                "shard_row_indices_complete",
                "vocabulary_remaps_lossless",
            )
        )
    ):
        raise ValueError(
            "Preparation lacks complete independent identity/representation verification"
        )
    request = read(prepared["request"])
    for value in (request["implementation"], verified["implementation"]):
        resolve_pin(value, repo, label="qualified representation implementation")
    evidence_pin = request["inputs"]["evidence_index"]
    evidence = read(evidence_pin)
    if (
        evidence.get("schema_version") != "forge.compose_lipid_evidence_index.v1"
        or evidence.get("training_admitted") is not False
        or request["inputs"]["evidence_database"] != evidence["artifact"]
    ):
        raise ValueError("Preparation lacks an authenticated exact-evidence index")
    evidence_path = resolve_pin(evidence["artifact"], repo, label="exact source evidence")
    if evidence["partition_inputs"]["corpus"] != imported["artifacts"]["corpus.sqlite"]:
        raise ValueError("Prepared graphs belong to a different source universe")
    summary = evidence["summary"]
    if (
        any(
            type(summary.get(k)) is not int or summary[k] < 0
            for k in ("eligible", "exact", "pending")
        )
        or summary["eligible"] != summary["exact"] + summary["pending"]
        or summary["exact"] != prepared["totals"]["records"]
    ):
        raise ValueError("Eligible, exact and represented record accounting differs")
    source = resolve_pin(
        prepared["artifacts"]["preparation.sqlite"], repo, label="qualified lookup"
    )
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("ATTACH DATABASE ? AS evidence", (evidence_path.as_uri() + "?mode=ro",))
        counts = dict(db.execute("SELECT family,count(*) FROM records GROUP BY family"))
        total, identities, constitutions = db.execute(
            "SELECT count(*),count(DISTINCT target_id),count(DISTINCT constitution_id) FROM records"
        ).fetchone()
        if total != identities or total != constitutions:
            raise ValueError("Source duplicates would receive extra model probability")
        if total != summary["exact"] or counts != prepared["by_family"]:
            raise ValueError("Lookup differs from the authenticated preparation census")
        if (
            db.execute(
                "SELECT count(*) FROM records r LEFT JOIN evidence.exact e USING(target_id) "
                "WHERE e.target_id IS NULL OR r.constitution_id!=e.constitution_id OR r.family!=e.family"
            ).fetchone()[0]
            or db.execute(
                "SELECT count(*) FROM evidence.exact e LEFT JOIN records r USING(target_id) "
                "WHERE r.target_id IS NULL"
            ).fetchone()[0]
        ):
            raise ValueError("Lookup changes or omits qualified exact source identities")
    extra = sorted(set(counts) - set(formal))
    if extra:
        raise ValueError(f"Nonformal or unknown training families: {extra}")
    source_formal = formal
    if cohort is not None:
        formal = qualified_cohort_families(
            repo,
            cohort,
            population=population,
            verification=verification,
            evidence_index=evidence_pin,
            prepared=prepared,
            evidence=evidence,
            formal=formal,
        )
    else:
        blockers = []
        if summary["pending"]:
            blockers.append(f"{summary['pending']} eligible records lack exact chemistry")
        missing = sorted(set(formal) - set(counts))
        if missing:
            blockers.append(
                "formal families without qualified eligible support: " + ", ".join(missing)
            )
        if blockers:
            raise TrainingMeasureUnavailableError(blockers)

    # Only the fully qualified population reaches weight fitting or output creation.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".training-measure-", dir=output.parent) as temporary:
        stage = Path(temporary)
        database = stage / "weights.sqlite"
        with closing(sqlite3.connect(database.as_uri(), uri=True)) as db:
            db.execute("ATTACH DATABASE ? AS source", (source.as_uri() + "?mode=ro",))
            db.executescript(
                "CREATE TABLE weights(record_index INTEGER PRIMARY KEY,target_id TEXT UNIQUE NOT NULL,"
                "constitution_id TEXT UNIQUE NOT NULL,family TEXT NOT NULL,"
                "probability REAL NOT NULL CHECK(probability>0));"
                "CREATE TABLE families(family TEXT PRIMARY KEY,records INTEGER NOT NULL CHECK(records>0));"
            )
            db.executemany("INSERT INTO families VALUES (?,?)", sorted(counts.items()))
            db.execute(
                "INSERT INTO weights SELECT row_number() OVER (ORDER BY r.shard_id,r.row_index)-1,"
                "r.target_id,r.constitution_id,r.family,1.0/(?*f.records) "
                "FROM source.records r JOIN families f USING(family) ORDER BY r.shard_id,r.row_index",
                (len(formal),),
            )
            db.execute("CREATE INDEX weights_family ON weights(family,record_index)")
            db.commit()
            observed = list(
                db.execute(
                    "SELECT family,count(*),min(probability),max(probability) FROM weights GROUP BY family"
                )
            )
            for family, count, smallest, largest in observed:
                probability = 1.0 / (len(formal) * counts[family])
                if count != counts[family] or smallest != probability or largest != probability:
                    raise ValueError("Compiled family mass or per-graph probability changed")
            mass = math.fsum(count * smallest for _, count, smallest, _ in observed)
            if not math.isclose(mass, 1.0, rel_tol=0, abs_tol=1e-14):
                raise ValueError("Compiled graph measure is not normalized")
            if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("Compiled graph measure failed database integrity")
        dump(
            stage / "result.json",
            {
                "schema_version": SCHEMA,
                "seed": 0,
                "inputs": {
                    "universe_config": dict(universe_config),
                    "population": dict(population),
                    "verification": dict(verification),
                    "evidence_index": evidence_pin,
                    **({"cohort": dict(cohort)} if cohort is not None else {}),
                },
                "implementation": implementation,
                "policy": policy,
                "formal_families": formal,
                "source_formal_families": source_formal,
                "pending_records_excluded": summary["pending"],
                "by_family": {
                    f: {
                        "records": counts[f],
                        "probability_per_graph": 1.0 / (len(formal) * counts[f]),
                        "total_probability": 1.0 / len(formal),
                    }
                    for f in formal
                },
                "records": total,
                "record_order": "qualified_preparation_shard_id_then_row_index",
                "artifact": {
                    **pin(repo, database),
                    "path": str((output / database.name).relative_to(repo)),
                },
                "raw_family_frequency_sampling": False,
                "source_role_bindings_receive_separate_mass": False,
                "training_admitted": False,
                "training_calls": 0,
            },
        )
        stage.rename(output)
    return output / "result.json"

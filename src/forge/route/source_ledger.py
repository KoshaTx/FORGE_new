"""Source-linked LNPDB component inventory for the M0-09 addendum.

The inventory deliberately distinguishes four facts that are easy to conflate:

1. a component structure is annotated in LNPDB;
2. a publication is linked to that annotation;
3. a route has been extracted and verified from a primary source;
4. terminal materials have current procurement evidence.

Only the first two are inferred by this module. Route and procurement closure remain unknown
until explicit evidence is added.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
import platform
import tempfile
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.route.supervision_inventory import sha256_file

CONFIG_SCHEMA_VERSION = "m0_09_lnpdb_route_source_config.v1"
PUBMED_SNAPSHOT_SCHEMA_VERSION = "m0_09_lnpdb_pubmed_snapshot.v1"
RESULT_SCHEMA_VERSION = "m0_09_lnpdb_route_source_ledger.v1"

LNPDB_SOURCE = "lnpdb_v1"
ROLE_FIELDS = (
    "IL_head_SMILES",
    "IL_linker_SMILES",
    "IL_tail1_SMILES",
    "IL_tail2_SMILES",
)
ROLE_CODES = {
    "IL_head_SMILES": "head",
    "IL_linker_SMILES": "linker",
    "IL_tail1_SMILES": "tail1",
    "IL_tail2_SMILES": "tail2",
}

SOURCE_COLUMNS = (
    "review_rank",
    "review_bucket",
    "source_id",
    "source_kind",
    "pmid",
    "doi",
    "pmc_id",
    "title",
    "journal",
    "pubdate",
    "pubmed_url",
    "full_text_url",
    "acquisition_status",
    "route_review_status",
    "lnpdb_record_count",
    "unique_lipid_count",
    "experiment_count",
    "il_name_count",
    "unique_component_count",
    "unique_head_count",
    "unique_linker_count",
    "unique_tail1_count",
    "unique_tail2_count",
    "experiment_ids_json",
    "reported_pmids_json",
    "publication_links_json",
    "source_identity_qa_json",
    "local_source_assets_json",
)

COMPONENT_COLUMNS = (
    "component_id",
    "role",
    "parse_status",
    "canonical_smiles",
    "raw_smiles_json",
    "unique_lipid_count",
    "lnpdb_source_record_count",
    "source_count",
    "source_ids_json",
    "publication_count",
    "source_pmids_json",
    "experiment_ids_json",
    "route_evidence_status",
    "procurement_evidence_status",
    "execution_closure_status",
)


class SourceLedgerError(ValueError):
    """Raised when an input violates the source-ledger evidence contract."""


def _as_clean_strings(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return sorted({str(item).strip() for item in values if str(item).strip()})


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SourceLedgerError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SourceLedgerError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceLedgerError(f"{label} must contain a JSON object: {path}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the M0-09 LNPDB source-ledger configuration."""

    config = _load_json(path, "source-ledger config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SourceLedgerError(
            f"unsupported config schema {config.get('schema_version')!r}; "
            f"expected {CONFIG_SCHEMA_VERSION!r}"
        )
    if tuple(config.get("role_fields", ())) != ROLE_FIELDS:
        raise SourceLedgerError("config role_fields are missing, reordered, or unsupported")

    inputs = config.get("inputs", {})
    for name in ("r0", "lnpdb", "pubmed_snapshot"):
        entry = inputs.get(name, {})
        if not entry.get("asset") or not entry.get("expected_sha256"):
            raise SourceLedgerError(f"config inputs.{name} must define asset and expected_sha256")

    local_pmids: set[str] = set()
    for source in config.get("known_local_sources", []):
        required = ("pmid", "asset", "expected_sha256", "route_review_status")
        if any(not source.get(field) for field in required):
            raise SourceLedgerError(
                f"every known_local_sources entry must define {', '.join(required)}"
            )
        pmid = str(source["pmid"])
        if pmid in local_pmids:
            raise SourceLedgerError(f"duplicate known local PMID: {pmid}")
        local_pmids.add(pmid)

    overrides = config.get("publication_identity_overrides", [])
    reported_pmids: set[str] = set()
    for override in overrides:
        required = (
            "reported_pmid",
            "publication_link",
            "resolved_pmid",
            "reason",
        )
        if any(not override.get(field) for field in required):
            raise SourceLedgerError(
                f"every publication_identity_overrides entry must define {', '.join(required)}"
            )
        reported = str(override["reported_pmid"])
        if reported in reported_pmids:
            raise SourceLedgerError(f"duplicate reported PMID override: {reported}")
        reported_pmids.add(reported)

    non_publication_sources = config.get("non_publication_sources", [])
    source_ids: set[str] = set()
    for source in non_publication_sources:
        required = ("source_id", "reported_pmid", "publication_link", "source_kind")
        if any(not source.get(field) for field in required):
            raise SourceLedgerError(
                f"every non_publication_sources entry must define {', '.join(required)}"
            )
        if source["source_id"] in source_ids:
            raise SourceLedgerError(f"duplicate non-publication source_id: {source['source_id']}")
        source_ids.add(source["source_id"])
    return config


def _verified_input(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.exists():
        raise SourceLedgerError(f"{label} not found: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise SourceLedgerError(
            f"hash mismatch for {label}: expected {expected_sha256}, observed {actual_sha256}"
        )
    return {
        "asset": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual_sha256,
    }


def extract_lnpdb_pmids(source_path: Path) -> list[str]:
    """Return sorted PubMed identifiers from raw LNPDB or preserved R0 provenance."""

    pmids: set[str] = set()
    try:
        handle = source_path.open(newline="", encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise SourceLedgerError(f"LNPDB source asset not found: {source_path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SourceLedgerError(f"LNPDB source CSV has no header: {source_path}")
        if "Publication_PMID" in reader.fieldnames:
            for row in reader:
                reported = row["Publication_PMID"].strip()
                if reported.isdigit():
                    pmids.add(reported)
        elif "provenance_json" in reader.fieldnames:
            for row_number, row in enumerate(reader, start=2):
                try:
                    provenance = json.loads(row["provenance_json"])
                except json.JSONDecodeError as exc:
                    raise SourceLedgerError(
                        f"R0 row {row_number} has invalid provenance_json"
                    ) from exc
                metadata = provenance.get(LNPDB_SOURCE, {}).get("metadata", {})
                pmids.update(
                    pmid
                    for pmid in _as_clean_strings(metadata.get("Publication_PMID"))
                    if pmid.isdigit()
                )
        else:
            raise SourceLedgerError(
                "LNPDB source CSV must contain Publication_PMID or provenance_json"
            )
    return sorted(pmids, key=int)


def normalize_pubmed_esummary(
    payload: dict[str, Any],
    expected_pmids: Iterable[str],
    *,
    retrieved_utc: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Normalize an NCBI PubMed ESummary response into a stable minimal snapshot."""

    result = payload.get("result")
    if not isinstance(result, dict):
        raise SourceLedgerError("PubMed ESummary response has no result object")

    expected = sorted({str(pmid) for pmid in expected_pmids}, key=int)
    observed = sorted({str(pmid) for pmid in result.get("uids", [])}, key=int)
    if observed != expected:
        missing = sorted(set(expected) - set(observed), key=int)
        extra = sorted(set(observed) - set(expected), key=int)
        raise SourceLedgerError(f"PubMed ESummary PMID mismatch; missing={missing}, extra={extra}")

    records: list[dict[str, Any]] = []
    for pmid in expected:
        raw = result.get(pmid)
        if not isinstance(raw, dict):
            raise SourceLedgerError(f"PubMed ESummary is missing record {pmid}")
        article_ids = {
            str(item.get("idtype")): str(item.get("value"))
            for item in raw.get("articleids", [])
            if item.get("idtype") and item.get("value")
        }
        records.append(
            {
                "pmid": pmid,
                "doi": article_ids.get("doi", ""),
                "pmc_id": article_ids.get("pmc", ""),
                "title": str(raw.get("title", "")).strip(),
                "journal": str(raw.get("fulljournalname") or raw.get("source") or "").strip(),
                "pubdate": str(raw.get("pubdate", "")).strip(),
            }
        )
    if any(not record["doi"] or not record["title"] for record in records):
        incomplete = [
            record["pmid"] for record in records if not record["doi"] or not record["title"]
        ]
        raise SourceLedgerError(f"PubMed records missing DOI or title: {incomplete}")

    return {
        "schema_version": PUBMED_SNAPSHOT_SCHEMA_VERSION,
        "retrieved_utc": retrieved_utc,
        "source": {
            "provider": "NCBI PubMed E-utilities",
            "endpoint": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            "database": "pubmed",
            "return_mode": "json",
        },
        "source_sha256": source_sha256,
        "records": records,
    }


def fetch_pubmed_snapshot(
    source_path: Path,
    *,
    additional_pmids: Iterable[str] = (),
    retrieved_utc: str | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Fetch one normalized PubMed metadata snapshot for all LNPDB-linked PMIDs."""

    pmids = sorted(
        set(extract_lnpdb_pmids(source_path)) | {str(pmid) for pmid in additional_pmids},
        key=int,
    )
    if not pmids:
        raise SourceLedgerError("R0 contains no LNPDB Publication_PMID values")
    query = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "retmode": "json",
            "tool": "forge_lnpdb_route_inventory",
            "id": ",".join(pmids),
        }
    )
    request = urllib.request.Request(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{query}",
        headers={"User-Agent": "FORGE-M0-09-source-ledger/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceLedgerError(f"PubMed ESummary request failed: {exc}") from exc

    timestamp = retrieved_utc or dt.datetime.now(dt.timezone.utc).isoformat()
    parsed_timestamp = dt.datetime.fromisoformat(timestamp)
    if parsed_timestamp.tzinfo is None:
        raise SourceLedgerError("retrieved_utc must be timezone-aware")
    return normalize_pubmed_esummary(
        payload,
        pmids,
        retrieved_utc=timestamp,
        source_sha256=sha256_file(source_path),
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write stable formatted JSON atomically."""

    _atomic_write(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_pubmed_snapshot(path: Path) -> dict[str, Any]:
    """Load and structurally validate a normalized PubMed metadata snapshot."""

    snapshot = _load_json(path, "PubMed snapshot")
    if snapshot.get("schema_version") != PUBMED_SNAPSHOT_SCHEMA_VERSION:
        raise SourceLedgerError(
            f"unsupported PubMed snapshot schema {snapshot.get('schema_version')!r}; "
            f"expected {PUBMED_SNAPSHOT_SCHEMA_VERSION!r}"
        )
    records = snapshot.get("records")
    if not isinstance(records, list) or not records:
        raise SourceLedgerError("PubMed snapshot records must be a non-empty list")
    pmids = [str(record.get("pmid", "")) for record in records]
    if any(not pmid for pmid in pmids) or len(pmids) != len(set(pmids)):
        raise SourceLedgerError("PubMed snapshot PMIDs must be non-empty and unique")
    return snapshot


def _canonical_component(raw_smiles: str) -> tuple[str, str]:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(raw_smiles)
    if molecule is None:
        return "invalid", ""
    return "parsed", Chem.MolToSmiles(molecule, isomericSmiles=True)


def _component_id(role: str, identity: str) -> str:
    digest = hashlib.sha256(f"{role}\0{identity}".encode()).hexdigest()[:16]
    return f"lnpdb-{ROLE_CODES[role]}-{digest}"


def _read_r0_context(r0_path: Path) -> dict[str, set[str]]:
    r0_ids: set[str] = set()
    canonical_smiles: set[str] = set()
    source_record_ids: set[str] = set()
    try:
        handle = r0_path.open(newline="")
    except FileNotFoundError as exc:
        raise SourceLedgerError(f"R0 asset not found: {r0_path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {
            "r0_structure_id",
            "canonical_isomeric_smiles",
            "provenance_json",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SourceLedgerError(f"R0 CSV must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                provenance = json.loads(row["provenance_json"])
            except json.JSONDecodeError as exc:
                raise SourceLedgerError(f"R0 row {row_number} has invalid provenance_json") from exc
            lnpdb = provenance.get(LNPDB_SOURCE)
            if not lnpdb:
                continue
            r0_ids.add(row["r0_structure_id"])
            canonical_smiles.add(row["canonical_isomeric_smiles"])
            source_record_ids.update(_as_clean_strings(lnpdb.get("source_record_ids")))
    return {
        "r0_ids": r0_ids,
        "canonical_smiles": canonical_smiles,
        "source_record_ids": source_record_ids,
    }


def _read_lnpdb_inventory(lnpdb_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "source_kind": "",
            "pmid": "",
            "lnp_ids": set(),
            "lipid_smiles": set(),
            "experiment_ids": set(),
            "il_names": set(),
            "reported_pmids": set(),
            "publication_links": set(),
            "source_identity_qa": set(),
            **{role: set() for role in ROLE_FIELDS},
        }
    )
    components: dict[tuple[str, str], dict[str, Any]] = {}
    all_lnp_ids: set[str] = set()
    all_lipid_smiles: set[str] = set()
    raw_values_by_role: dict[str, set[str]] = defaultdict(set)
    row_count = 0

    overrides = {
        str(override["reported_pmid"]): override
        for override in config.get("publication_identity_overrides", [])
    }
    non_publication_sources = {
        (str(source["reported_pmid"]), source["publication_link"]): source
        for source in config.get("non_publication_sources", [])
    }

    try:
        handle = lnpdb_path.open(newline="", encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise SourceLedgerError(f"raw LNPDB asset not found: {lnpdb_path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {
            "LNP_ID",
            "Experiment_ID",
            "IL_name",
            "IL_SMILES",
            "Publication_PMID",
            "Publication_link",
            *ROLE_FIELDS,
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SourceLedgerError(f"raw LNPDB CSV must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            lnp_id = row["LNP_ID"].strip()
            if not lnp_id:
                raise SourceLedgerError(f"raw LNPDB row {row_number} has an empty LNP_ID")
            if lnp_id in all_lnp_ids:
                raise SourceLedgerError(f"duplicate raw LNPDB LNP_ID: {lnp_id}")
            all_lnp_ids.add(lnp_id)

            parse_status, lipid_smiles = _canonical_component(row["IL_SMILES"].strip())
            if parse_status != "parsed":
                raise SourceLedgerError(f"raw LNPDB row {row_number} has invalid IL_SMILES")
            all_lipid_smiles.add(lipid_smiles)

            reported_pmid = row["Publication_PMID"].strip()
            publication_link = row["Publication_link"].strip()
            identity_qa = ""
            if reported_pmid.isdigit():
                override = overrides.get(reported_pmid)
                if override:
                    if publication_link != override["publication_link"]:
                        raise SourceLedgerError(
                            f"publication override link mismatch for reported PMID {reported_pmid}"
                        )
                    source_id = str(override["resolved_pmid"])
                    identity_qa = (
                        f"reported PMID {reported_pmid} corrected to {source_id}: "
                        f"{override['reason']}"
                    )
                else:
                    source_id = reported_pmid
                source_kind = "publication"
                resolved_pmid = source_id
            else:
                source = non_publication_sources.get((reported_pmid, publication_link))
                if source is None:
                    raise SourceLedgerError(
                        "unmapped non-publication LNPDB source at row "
                        f"{row_number}: PMID={reported_pmid!r}, link={publication_link!r}"
                    )
                source_id = source["source_id"]
                source_kind = source["source_kind"]
                resolved_pmid = ""

            source_evidence = sources[source_id]
            if source_evidence["source_kind"] not in ("", source_kind):
                raise SourceLedgerError(f"conflicting source kinds for {source_id}")
            if source_evidence["pmid"] not in ("", resolved_pmid):
                raise SourceLedgerError(f"conflicting resolved PMIDs for {source_id}")
            source_evidence["source_kind"] = source_kind
            source_evidence["pmid"] = resolved_pmid
            source_evidence["lnp_ids"].add(lnp_id)
            source_evidence["lipid_smiles"].add(lipid_smiles)
            source_evidence["experiment_ids"].update(_as_clean_strings(row["Experiment_ID"]))
            source_evidence["il_names"].update(_as_clean_strings(row["IL_name"]))
            source_evidence["reported_pmids"].add(reported_pmid)
            source_evidence["publication_links"].add(publication_link)
            if identity_qa:
                source_evidence["source_identity_qa"].add(identity_qa)

            normalized_row_components: dict[str, set[str]] = defaultdict(set)
            for role in ROLE_FIELDS:
                for raw_smiles in _as_clean_strings(row.get(role)):
                    if raw_smiles.upper() == "NA":
                        continue
                    raw_values_by_role[role].add(raw_smiles)
                    parse_status, canonical_smiles = _canonical_component(raw_smiles)
                    identity = (
                        f"canonical:{canonical_smiles}"
                        if parse_status == "parsed"
                        else f"raw:{raw_smiles}"
                    )
                    normalized_row_components[role].add(identity)
                    key = (role, identity)
                    component = components.setdefault(
                        key,
                        {
                            "component_id": _component_id(role, identity),
                            "role": ROLE_CODES[role],
                            "parse_status": parse_status,
                            "canonical_smiles": canonical_smiles,
                            "raw_smiles": set(),
                            "lipid_smiles": set(),
                            "source_record_ids": set(),
                            "source_ids": set(),
                            "pmids": set(),
                            "experiment_ids": set(),
                        },
                    )
                    component["raw_smiles"].add(raw_smiles)
                    component["lipid_smiles"].add(lipid_smiles)
                    component["source_record_ids"].add(lnp_id)
                    component["source_ids"].add(source_id)
                    if resolved_pmid:
                        component["pmids"].add(resolved_pmid)
                    component["experiment_ids"].update(_as_clean_strings(row["Experiment_ID"]))

            for role, identities in normalized_row_components.items():
                source_evidence[role].update(identities)

    component_rows = []
    for component in components.values():
        component_rows.append(
            {
                "component_id": component["component_id"],
                "role": component["role"],
                "parse_status": component["parse_status"],
                "canonical_smiles": component["canonical_smiles"],
                "raw_smiles_json": json.dumps(sorted(component["raw_smiles"])),
                "unique_lipid_count": len(component["lipid_smiles"]),
                "lnpdb_source_record_count": len(component["source_record_ids"]),
                "source_count": len(component["source_ids"]),
                "source_ids_json": json.dumps(sorted(component["source_ids"])),
                "publication_count": len(component["pmids"]),
                "source_pmids_json": json.dumps(sorted(component["pmids"], key=int)),
                "experiment_ids_json": json.dumps(sorted(component["experiment_ids"])),
                "route_evidence_status": "not_assessed",
                "procurement_evidence_status": "not_assessed",
                "execution_closure_status": "unknown",
            }
        )
    component_rows.sort(key=lambda row: (row["role"], row["component_id"]))

    return {
        "sources": sources,
        "components": component_rows,
        "row_count": row_count,
        "lnp_ids": all_lnp_ids,
        "lipid_smiles": all_lipid_smiles,
        "raw_values_by_role": raw_values_by_role,
    }


def _publication_review_bucket(
    pmid: str,
    pmc_id: str,
    local_sources: dict[str, list[dict[str, Any]]],
    source_kind: str,
) -> tuple[str, str, str]:
    if source_kind == "commercial_catalog":
        return "commercial_procurement_review", "live_source_requires_snapshot", "not_reviewed"
    if pmid in local_sources:
        statuses = {source["route_review_status"] for source in local_sources[pmid]}
        if len(statuses) != 1:
            raise SourceLedgerError(f"conflicting local route review statuses for PMID {pmid}")
        return "local_partial_followup", "local_source_available", statuses.pop()
    if pmc_id:
        return "pmc_first_pass", "pmc_identifier_available", "not_reviewed"
    return "publisher_or_author_acquisition", "metadata_only", "not_reviewed"


def _csv_text(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_source_ledger(
    config_path: Path,
    vendor_dir: Path,
    reference_dir: Path,
    *,
    generated_utc: str | None = None,
    seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the LNPDB source queue and component evidence ledger."""

    config = load_config(config_path)
    r0_config = config["inputs"]["r0"]
    lnpdb_config = config["inputs"]["lnpdb"]
    pubmed_config = config["inputs"]["pubmed_snapshot"]
    r0_path = vendor_dir / r0_config["asset"]
    lnpdb_path = vendor_dir / lnpdb_config["asset"]
    pubmed_path = reference_dir / pubmed_config["asset"]

    inputs = [
        _verified_input(config_path, sha256_file(config_path), "source-ledger config"),
        _verified_input(r0_path, r0_config["expected_sha256"], "R0"),
        _verified_input(lnpdb_path, lnpdb_config["expected_sha256"], "raw LNPDB"),
        _verified_input(
            pubmed_path,
            pubmed_config["expected_sha256"],
            "PubMed metadata snapshot",
        ),
    ]

    local_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in config.get("known_local_sources", []):
        local_path = vendor_dir / source["asset"]
        inputs.append(
            _verified_input(
                local_path,
                source["expected_sha256"],
                f"known local source for PMID {source['pmid']}",
            )
        )
        local_sources[str(source["pmid"])].append(source)

    snapshot = load_pubmed_snapshot(pubmed_path)
    inventory = _read_lnpdb_inventory(lnpdb_path, config)
    r0_context = _read_r0_context(r0_path)
    if inventory["lnp_ids"] != r0_context["source_record_ids"]:
        missing = sorted(inventory["lnp_ids"] - r0_context["source_record_ids"])[:10]
        extra = sorted(r0_context["source_record_ids"] - inventory["lnp_ids"])[:10]
        raise SourceLedgerError(
            "raw LNPDB records do not exactly match R0 provenance; "
            f"raw_only_examples={missing}, r0_only_examples={extra}"
        )
    if inventory["lipid_smiles"] != r0_context["canonical_smiles"]:
        missing = sorted(inventory["lipid_smiles"] - r0_context["canonical_smiles"])[:10]
        extra = sorted(r0_context["canonical_smiles"] - inventory["lipid_smiles"])[:10]
        raise SourceLedgerError(
            "raw LNPDB structures do not exactly match the LNPDB subset of R0; "
            f"raw_only_examples={missing}, r0_only_examples={extra}"
        )

    sources = inventory["sources"]
    metadata_by_pmid = {record["pmid"]: record for record in snapshot["records"]}
    observed_pmids = {
        reported
        for evidence in sources.values()
        for reported in evidence["reported_pmids"]
        if reported.isdigit()
    }
    observed_pmids.update(evidence["pmid"] for evidence in sources.values() if evidence["pmid"])
    metadata_pmids = set(metadata_by_pmid)
    if observed_pmids != metadata_pmids:
        missing = sorted(observed_pmids - metadata_pmids, key=int)
        extra = sorted(metadata_pmids - observed_pmids, key=int)
        raise SourceLedgerError(
            f"PubMed snapshot does not exactly match LNPDB identities; missing={missing}, extra={extra}"
        )
    if snapshot.get("source_sha256") != lnpdb_config["expected_sha256"]:
        raise SourceLedgerError(
            "PubMed snapshot was not derived from the configured raw LNPDB hash"
        )

    non_publication_by_id = {
        source["source_id"]: source for source in config.get("non_publication_sources", [])
    }
    source_rows = []
    for source_id, evidence in sources.items():
        source_kind = evidence["source_kind"]
        pmid = evidence["pmid"]
        if source_kind == "publication":
            metadata = metadata_by_pmid[pmid]
            title = metadata["title"]
            doi = metadata["doi"]
            pmc_id = metadata["pmc_id"]
            journal = metadata["journal"]
            pubdate = metadata["pubdate"]
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            full_text_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/" if pmc_id else ""
        else:
            source_config = non_publication_by_id[source_id]
            title = source_config["title"]
            doi = ""
            pmc_id = ""
            journal = ""
            pubdate = ""
            pubmed_url = source_config["publication_link"]
            full_text_url = source_config["publication_link"]

        bucket, acquisition_status, route_review_status = _publication_review_bucket(
            pmid,
            pmc_id,
            local_sources,
            source_kind,
        )
        role_counts = {role: len(evidence[role]) for role in ROLE_FIELDS}
        source_rows.append(
            {
                "review_rank": 0,
                "review_bucket": bucket,
                "source_id": source_id,
                "source_kind": source_kind,
                "pmid": pmid,
                "doi": doi,
                "pmc_id": pmc_id,
                "title": title,
                "journal": journal,
                "pubdate": pubdate,
                "pubmed_url": pubmed_url,
                "full_text_url": full_text_url,
                "acquisition_status": acquisition_status,
                "route_review_status": route_review_status,
                "lnpdb_record_count": len(evidence["lnp_ids"]),
                "unique_lipid_count": len(evidence["lipid_smiles"]),
                "experiment_count": len(evidence["experiment_ids"]),
                "il_name_count": len(evidence["il_names"]),
                "unique_component_count": sum(role_counts.values()),
                "unique_head_count": role_counts["IL_head_SMILES"],
                "unique_linker_count": role_counts["IL_linker_SMILES"],
                "unique_tail1_count": role_counts["IL_tail1_SMILES"],
                "unique_tail2_count": role_counts["IL_tail2_SMILES"],
                "experiment_ids_json": json.dumps(sorted(evidence["experiment_ids"])),
                "reported_pmids_json": json.dumps(sorted(evidence["reported_pmids"])),
                "publication_links_json": json.dumps(sorted(evidence["publication_links"])),
                "source_identity_qa_json": json.dumps(sorted(evidence["source_identity_qa"])),
                "local_source_assets_json": json.dumps(
                    sorted(source["asset"] for source in local_sources.get(pmid, []))
                ),
            }
        )

    bucket_order = {
        "local_partial_followup": 0,
        "pmc_first_pass": 1,
        "commercial_procurement_review": 2,
        "publisher_or_author_acquisition": 3,
    }
    source_rows.sort(
        key=lambda row: (
            bucket_order[row["review_bucket"]],
            -row["unique_component_count"],
            -row["unique_lipid_count"],
            row["source_id"],
        )
    )
    for rank, row in enumerate(source_rows, start=1):
        row["review_rank"] = rank

    component_rows = inventory["components"]
    role_summary = {}
    for role, code in ROLE_CODES.items():
        rows = [row for row in component_rows if row["role"] == code]
        role_summary[code] = {
            "raw_values": len(inventory["raw_values_by_role"][role]),
            "normalized_components": len(rows),
            "parsed_components": sum(row["parse_status"] == "parsed" for row in rows),
            "invalid_components": sum(row["parse_status"] == "invalid" for row in rows),
            "components_without_source": sum(row["source_count"] == 0 for row in rows),
            "components_without_publication": sum(row["publication_count"] == 0 for row in rows),
        }

    source_csv = _csv_text(source_rows, SOURCE_COLUMNS)
    component_csv = _csv_text(component_rows, COMPONENT_COLUMNS)
    timestamp = generated_utc or dt.datetime.now(dt.timezone.utc).isoformat()
    parsed_timestamp = dt.datetime.fromisoformat(timestamp)
    if parsed_timestamp.tzinfo is None:
        raise SourceLedgerError("generated_utc must be timezone-aware")

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-09 addendum",
        "generated_utc": timestamp,
        "inputs": inputs,
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
        "randomness": {"seed": seed, "used": False},
        "scope": {
            "purpose": (
                "Prioritize primary-source route curation for LNPDB-annotated heads, "
                "linkers, and tails."
            ),
            "model_training_authorized": False,
            "all_lnpdb_used_for_inventory_only": True,
            "split_before_extraction_required_for_later_model_data": True,
        },
        "summary": {
            "lnpdb_records": inventory["row_count"],
            "lnpdb_unique_lipids": len(inventory["lipid_smiles"]),
            "r0_lnpdb_structures": len(r0_context["canonical_smiles"]),
            "raw_lnpdb_to_r0_exact_match": True,
            "unique_source_families": len(source_rows),
            "unique_publications": sum(row["source_kind"] == "publication" for row in source_rows),
            "commercial_sources": sum(
                row["source_kind"] == "commercial_catalog" for row in source_rows
            ),
            "publication_identity_corrections": sum(
                bool(json.loads(row["source_identity_qa_json"])) for row in source_rows
            ),
            "publications_with_pmc_identifier": sum(bool(row["pmc_id"]) for row in source_rows),
            "publications_with_local_source": sum(
                row["acquisition_status"] == "local_source_available" for row in source_rows
            ),
            "publications_requiring_publisher_or_author_acquisition": sum(
                row["review_bucket"] == "publisher_or_author_acquisition" for row in source_rows
            ),
            "normalized_role_components": len(component_rows),
            "components_without_source": sum(row["source_count"] == 0 for row in component_rows),
            "components_without_publication": sum(
                row["publication_count"] == 0 for row in component_rows
            ),
            "role_components": role_summary,
        },
        "evidence_contract": {
            "route_statuses": {
                "not_assessed": "No route claim has been reviewed for this exact component.",
                "source_identified": "A primary source is linked, but no complete route is extracted.",
                "route_extracted": "Steps and conditions are transcribed with source locations.",
                "forward_verified": (
                    "Every non-terminal step passes deterministic forward reconstruction."
                ),
                "internally_validated": "The route also has an internal experimental success record.",
            },
            "procurement_statuses": {
                "not_assessed": "No current supplier evidence has been reviewed.",
                "vendor_claim_only": "A supplier listing exists but has not been time-stamped and checked.",
                "vendor_verified": (
                    "Identity, supplier, catalog identifier, region, date, and backup are recorded."
                ),
                "internal_stock_verified": "Identity and current internal stock are recorded.",
            },
            "execution_acceptance_rule": (
                "A component is executable only when it is procurement-verified or has a "
                "forward-verified route whose leaves are procurement-verified or internally stocked."
            ),
            "non_inference_rule": (
                "Publication linkage, structural synthesizability, or supplier plausibility alone "
                "does not establish route or procurement closure."
            ),
        },
        "outputs": {
            "source_queue": {
                "records": len(source_rows),
                "sha256": _sha256_text(source_csv),
            },
            "component_source_ledger": {
                "records": len(component_rows),
                "sha256": _sha256_text(component_csv),
            },
        },
        "decision": {
            "curation_surface_is_finite": True,
            "next_action": (
                "Review locally available and PMC-linked sources first, extract exact component "
                "routes with source locations, then acquire the remaining publisher or author files."
            ),
            "architecture_impact": (
                "Retain generic neural proposals plus deterministic source-linked template search. "
                "This inventory does not authorize a standalone lipid-specific route model."
            ),
        },
    }
    return result, source_rows, component_rows


def write_source_ledger(
    result: dict[str, Any],
    source_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write the result JSON and both deterministic CSV ledgers atomically."""

    source_csv = _csv_text(source_rows, SOURCE_COLUMNS)
    component_csv = _csv_text(component_rows, COMPONENT_COLUMNS)
    if _sha256_text(source_csv) != result["outputs"]["source_queue"]["sha256"]:
        raise SourceLedgerError("source queue changed after result construction")
    if _sha256_text(component_csv) != result["outputs"]["component_source_ledger"]["sha256"]:
        raise SourceLedgerError("component ledger changed after result construction")

    _atomic_write(output_dir / "source_queue.csv", source_csv)
    _atomic_write(output_dir / "component_source_ledger.csv", component_csv)
    write_json(result, output_dir / "route_source_ledger.json")

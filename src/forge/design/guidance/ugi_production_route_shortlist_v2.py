"""Replace the preliminary branch cohort with corrected full-support candidates.

The matched linear potency cohort from v1 is preserved byte-for-byte at the
molecular-record level.  The old short/non-ester branch rows are discarded and
replaced by two route-blinded branch cohorts drawn from the corrected support:
frozen-policy oracle-scored candidates and potency-neutral synthesis exploration.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from forge.core.io import stable_json as _stable_json
from forge.data.r1_prime_audit import sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_production_route_shortlist_config.v2"
RESULT_SCHEMA_VERSION = "phase1_ugi_production_route_shortlist.v2"
LEDGER_SCHEMA_VERSION = "phase1_ugi_production_route_shortlist_ledger.v2"


class UgiProductionRouteShortlistV2Error(RuntimeError):
    """Raised when the corrected route-blinded shortlist cannot be reproduced."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiProductionRouteShortlistV2Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiProductionRouteShortlistV2Error(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiProductionRouteShortlistV2Error(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiProductionRouteShortlistV2Error(f"pin escapes repository: {label}") from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiProductionRouteShortlistV2Error(f"pin changed: {label}")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise UgiProductionRouteShortlistV2Error("JSONL ledger is malformed")
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str, *, label: str) -> bool:
    if value not in {"True", "False"}:
        raise UgiProductionRouteShortlistV2Error(f"{label} is not a serialized boolean")
    return value == "True"


def _eligible_aldehyde_branch(source: Mapping[str, Any], *, require_corpus_absent: bool) -> bool:
    """Recognize one well-spaced, long, ester-bearing aldehyde branch."""

    descriptors = source.get("tail_component_descriptors") or {}
    aldehyde = descriptors.get("oxoester_aldehyde_body_tail") or {}
    admission = source.get("terminal_chemical_admission") or {}
    return bool(
        admission.get("admitted")
        and source.get("realized_carbon_branch_class") == "aldehyde_origin_branched"
        and (not require_corpus_absent or not source.get("exact_refit_corpus_product"))
        and int(aldehyde.get("carbon_atoms", 0)) >= 14
        and int(aldehyde.get("carbon_branch_points", 0)) == 1
        and int(aldehyde.get("ester_carbonyls", 0)) >= 1
        and int(aldehyde.get("adjacent_carbon_branch_edges", 0)) == 0
    )


def _fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise UgiProductionRouteShortlistV2Error(f"invalid product SMILES: {smiles}")
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(molecule)


def _deduplicate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        product = str(row["canonical_product"])
        current = chosen.get(product)
        if current is None or (
            not bool(current.get("conservative_high_potency")),
            -float(current.get("lcb90") or -1e9),
            int(current["draw_index"]),
        ) > (
            not bool(row.get("conservative_high_potency")),
            -float(row.get("lcb90") or -1e9),
            int(row["draw_index"]),
        ):
            chosen[product] = row
    return [chosen[key] for key in sorted(chosen)]


def _select_diverse(
    rows: Sequence[dict[str, Any]],
    count: int,
    *,
    required: Sequence[dict[str, Any]] = (),
    reference: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Max-min select unique products while retaining every required row."""

    candidates = _deduplicate(rows)
    required_unique = _deduplicate(required)
    required_products = {str(row["canonical_product"]) for row in required_unique}
    if not required_products.issubset({str(row["canonical_product"]) for row in candidates}):
        raise UgiProductionRouteShortlistV2Error("required rows are absent from selection pool")
    if len(required_unique) > count or len(candidates) < count:
        raise UgiProductionRouteShortlistV2Error("shortlist quota exceeds its candidate pool")
    fingerprints = {
        str(row["canonical_product"]): _fingerprint(str(row["canonical_product"]))
        for row in [*candidates, *reference]
    }
    selected = sorted(
        required_unique,
        key=lambda row: (
            -float(row.get("lcb90") or -1e9),
            str(row["canonical_product"]),
        ),
    )
    remaining = [
        row for row in candidates if str(row["canonical_product"]) not in required_products
    ]
    anchors = [*selected, *reference]
    if not anchors and remaining:
        selected.append(remaining.pop(0))
        anchors = list(selected)
    while len(selected) < count:
        scored = []
        for row in remaining:
            product = str(row["canonical_product"])
            distance = min(
                1.0
                - DataStructs.TanimotoSimilarity(
                    fingerprints[product], fingerprints[str(anchor["canonical_product"])]
                )
                for anchor in anchors
            )
            scored.append((distance, product, int(row["draw_index"]), row))
        scored.sort(key=lambda value: (-value[0], value[1], value[2]))
        chosen = scored[0][-1]
        selected.append(chosen)
        anchors.append(chosen)
        remaining.remove(chosen)
    return selected


def _gzip_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as handle:
        for row in rows:
            handle.write((_stable_json(row) + "\n").encode())
    return output.getvalue()


def build_production_route_shortlist_v2(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Freeze the corrected, route-blinded production shortlist."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="route shortlist v2 config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status") != "frozen_before_corrected_route_blinded_shortlist"
    ):
        raise UgiProductionRouteShortlistV2Error("unsupported route shortlist v2 config")
    paths = {label: _pin(repo, record, label=label) for label, record in config["inputs"].items()}
    implementation_paths = {
        label: _pin(repo, record, label=f"implementation.{label}")
        for label, record in config["implementation"].items()
    }
    old_result = _load_json(paths["previous_shortlist_result"], label="previous shortlist")
    generation_result = _load_json(paths["branch_generation_result"], label="branch generation")
    applicability_result = _load_json(
        paths["branch_applicability_result"], label="branch applicability"
    )
    if (
        old_result.get("status") != "route_blinded_balanced_production_shortlist_complete"
        or generation_result.get("status") != "complete_full_corpus_branch_exploration_generation"
        or applicability_result.get("status")
        != "complete_branch_exploration_frozen_policy_applicability_rescoring"
    ):
        raise UgiProductionRouteShortlistV2Error("shortlist source status changed")
    for result, key, path in (
        (old_result, "shortlist.jsonl.gz", paths["previous_shortlist_ledger"]),
        (generation_result, "terminal_ledger.jsonl.gz", paths["branch_generation_ledger"]),
        (
            applicability_result,
            "terminal_rescoring.csv.gz",
            paths["branch_applicability_ledger"],
        ),
    ):
        if result.get("artifacts", {}).get(key, {}).get("sha256") != sha256_file(path):
            raise UgiProductionRouteShortlistV2Error(f"source result does not pin {key}")

    previous = _read_jsonl(paths["previous_shortlist_ledger"])
    linear = [row for row in previous if row.get("cohort") == "potency_exploitation"]
    expected_linear = int(config["quotas"]["preserved_linear_potency"])
    if len(linear) != expected_linear:
        raise UgiProductionRouteShortlistV2Error("preserved linear cohort changed")
    generation = {
        int(row["draw_index"]): row for row in _read_jsonl(paths["branch_generation_ledger"])
    }
    applicability = {
        int(row["draw_index"]): row for row in _read_csv(paths["branch_applicability_ledger"])
    }
    if len(generation) != 4096 or set(generation) != set(applicability):
        raise UgiProductionRouteShortlistV2Error("branch ledgers do not align")

    branch_rows = []
    for draw, source in generation.items():
        score = applicability[draw]
        descriptors = source.get("tail_component_descriptors") or {}
        aldehyde = descriptors.get("oxoester_aldehyde_body_tail", {})
        if not _eligible_aldehyde_branch(source, require_corpus_absent=False):
            continue
        product = str(source["canonical_admitted_product"])
        if product != score["canonical_product"] and score["canonical_product"]:
            raise UgiProductionRouteShortlistV2Error("branch product identities differ")
        branch_rows.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "arm_id": "branch_exploration",
                "draw_index": draw,
                "canonical_product": product,
                "components": dict(source["native_terminal"]["component_smiles_by_role"]),
                "program": dict(source["program"]),
                "program_sha256": str(source["program_sha256"]),
                "realized_carbon_branch_class": str(source["realized_carbon_branch_class"]),
                "exact_refit_corpus_product": bool(source["exact_refit_corpus_product"]),
                "aldehyde_carbon_atoms": int(aldehyde["carbon_atoms"]),
                "aldehyde_carbon_branch_points": int(aldehyde["carbon_branch_points"]),
                "aldehyde_ester_carbonyls": int(aldehyde["ester_carbonyls"]),
                "oracle_scored": _bool(score["oracle_scored"], label="oracle_scored"),
                "pattern_id": str(score["pattern_id"]),
                "authority_tier": str(score["authority_tier"]),
                "calibration_scale": str(score["calibration_scale"]),
                "oracle_mean": float(score["oracle_mean"]) if score["oracle_mean"] else None,
                "lcb90": float(score["lcb90"]) if score["lcb90"] else None,
                "potency_utility": (
                    float(score["potency_utility"]) if score["oracle_scored"] == "True" else None
                ),
                "conservative_high_potency": _bool(
                    score["conservative_high_potency"], label="conservative_high"
                ),
                "biological_reason": str(score["reason"]),
            }
        )

    scored_pool = [row for row in branch_rows if row["oracle_scored"]]
    required_high = [row for row in scored_pool if row["conservative_high_potency"]]
    selected_scored = _select_diverse(
        scored_pool,
        int(config["quotas"]["branch_oracle_scored"]),
        required=required_high,
    )
    scored_products = {str(row["canonical_product"]) for row in selected_scored}
    exploration_pool = [
        row
        for row in branch_rows
        if (
            not row["oracle_scored"]
            and not row["exact_refit_corpus_product"]
            and str(row["canonical_product"]) not in scored_products
        )
    ]
    selected_exploration = _select_diverse(
        exploration_pool,
        int(config["quotas"]["branch_potency_neutral_exploration"]),
        reference=selected_scored,
    )

    output: list[dict[str, Any]] = []
    for row in linear:
        copied = dict(row)
        copied["schema_version"] = LEDGER_SCHEMA_VERSION
        copied["source_shortlist_version"] = "v1_preserved_linear_potency"
        output.append(copied)
    for cohort, selected in (
        ("branched_oracle_scored_candidate", selected_scored),
        ("branched_synthesis_exploration_no_potency_claim", selected_exploration),
    ):
        for rank, row in enumerate(selected, start=1):
            copied = dict(row)
            copied.update({"cohort": cohort, "stratum": cohort, "stratum_rank": rank})
            if cohort.endswith("no_potency_claim"):
                copied.update(
                    {
                        "oracle_mean": None,
                        "lcb90": None,
                        "potency_utility": None,
                        "conservative_high_potency": False,
                    }
                )
            output.append(copied)
    products = [str(row["canonical_product"]) for row in output]
    if len(products) != len(set(products)):
        raise UgiProductionRouteShortlistV2Error("corrected shortlist contains duplicate products")
    output.sort(
        key=lambda row: (
            str(row["arm_id"]),
            str(row["cohort"]),
            str(row["stratum"]),
            int(row["stratum_rank"]),
        )
    )
    ledger = _gzip_jsonl(output)
    pins = {
        label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
        for label, path in sorted(paths.items())
    }
    pins["config"] = {
        "path": str(config_path.relative_to(repo)),
        "sha256": sha256_file(config_path),
    }
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "corrected_route_blinded_production_shortlist_complete",
        "inputs": pins,
        "implementation": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(implementation_paths.items())
        },
        "policy": {
            "route_outcomes_used": False,
            "matched_linear_potency_cohort_preserved": True,
            "preliminary_branch_cohort_discarded": True,
            "all_unique_conservative_high_branch_products_retained": True,
            "oracle_scored_branch_rows_may_be_exact_refit_corpus_products": True,
            "potency_neutral_branch_exploration_requires_corpus_absence": True,
            "branching_selection_requires_one_nonadjacent_aldehyde_carbon_branch": True,
            "branching_selection_requires_long_ester_aldehyde": True,
            "potency_neutral_exploration_scores_hidden": True,
            "quotas": dict(config["quotas"]),
        },
        "summary": {
            "rows": len(output),
            "cohorts": dict(sorted(Counter(str(row["cohort"]) for row in output).items())),
            "arms": dict(sorted(Counter(str(row["arm_id"]) for row in output).items())),
            "unique_products": len(set(products)),
            "branch_oracle_scored_conservative_high": sum(
                bool(row.get("conservative_high_potency")) for row in selected_scored
            ),
            "branch_oracle_scored_exact_refit_corpus_products": sum(
                bool(row["exact_refit_corpus_product"]) for row in selected_scored
            ),
            "branch_exploration_exact_refit_corpus_products": sum(
                bool(row["exact_refit_corpus_product"]) for row in selected_exploration
            ),
            "branch_oracle_scored_authority_tiers": dict(
                sorted(Counter(str(row["authority_tier"]) for row in selected_scored).items())
            ),
            "branch_unique_aldehydes": len(
                {
                    row["components"]["oxoester_aldehyde_body_tail"]
                    for row in [*selected_scored, *selected_exploration]
                }
            ),
            "branch_unique_isocyanides": len(
                {
                    row["components"]["isocyanide_tail"]
                    for row in [*selected_scored, *selected_exploration]
                }
            ),
            "branch_unique_amines": len(
                {
                    row["components"]["amine_head"]
                    for row in [*selected_scored, *selected_exploration]
                }
            ),
        },
        "artifacts": {
            "shortlist.jsonl.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(output),
                "sha256": hashlib.sha256(ledger).hexdigest(),
            }
        },
        "decision": {
            "previous_branching_shortlist_superseded": True,
            "next_gate": "identical_graph2edits_aizynthfinder_and_evidence_route_assessment_before_panel_lock",
        },
        "nonclaims": [
            "Shortlisting does not establish complete L2/L3 route closure.",
            "The branched oracle-scored cohort does not establish broad branched-tail oracle generalization.",
            "The potency-neutral branch cohort carries no biological claim.",
            "Some oracle-scored branch candidates occur in the enumerated structural-training corpus and are reported separately.",
            "This artifact is not a locked synthesis panel.",
        ],
    }
    result = {
        **content,
        "result_sha256": hashlib.sha256(_stable_json(content).encode()).hexdigest(),
    }
    return result, ledger


__all__ = [
    "UgiProductionRouteShortlistV2Error",
    "_eligible_aldehyde_branch",
    "build_production_route_shortlist_v2",
]

"""Build a route-blinded, diversity-balanced shortlist from production v2.

The shortlist has two explicitly separate cohorts:

* potency exploitation uses only chemically admitted, non-measured,
  multiview-interpolative terminals with frozen conservative utility; and
* morphology exploration uses admitted, refit-corpus-absent branched products
  without allowing an unsupported potency score to influence selection.

No route outcome is read by this module.  It therefore freezes the molecular
shortlist before Graph2Edits, AiZynthFinder, or evidence-based route assessment.
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
from rdkit.Chem import AllChem, Descriptors

from forge.data.r1_prime_audit import sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_production_route_shortlist_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_production_route_shortlist.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_production_route_shortlist_ledger.v1"
ARMS = ("broad_prior", "support_enriched")


class UgiProductionRouteShortlistError(RuntimeError):
    """Raised when the frozen pre-route shortlist contract changes."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise UgiProductionRouteShortlistError(f"JSON object required: {path}")
    return value


def _pin(repo: Path, record: Mapping[str, Any], *, label: str) -> Path:
    if set(record) != {"path", "sha256"}:
        raise UgiProductionRouteShortlistError(f"malformed input pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiProductionRouteShortlistError(f"input escapes repository: {label}") from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiProductionRouteShortlistError(f"input pin changed: {label}")
    return path


def _generation_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise UgiProductionRouteShortlistError("generation ledger is malformed")
    return rows


def _ranking_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str, *, label: str) -> bool:
    if value not in {"True", "False"}:
        raise UgiProductionRouteShortlistError(f"{label} is not a serialized boolean")
    return value == "True"


def _fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise UgiProductionRouteShortlistError(f"invalid product SMILES: {smiles}")
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(molecule)


def _chemistry_descriptors(source: Mapping[str, Any]) -> dict[str, Any]:
    native = source["native_terminal"]
    product = Chem.MolFromSmiles(str(native["smiles"]))
    if product is None:
        raise UgiProductionRouteShortlistError("admitted product no longer parses")
    tail_roles = ("oxoester_aldehyde_body_tail", "isocyanide_tail")
    unsaturated_bonds = 0
    tail_carbon_branch_points = 0
    for role in tail_roles:
        component = Chem.MolFromSmiles(str(native["component_smiles_by_role"][role]))
        if component is None:
            raise UgiProductionRouteShortlistError("admitted component no longer parses")
        # Lipid-tail unsaturation means C=C.  Carbonyls, the aldehyde handle,
        # and the isocyanide handle are deliberately excluded.
        unsaturated_bonds += sum(
            bond.GetBondTypeAsDouble() == 2.0
            and bond.GetBeginAtom().GetAtomicNum() == 6
            and bond.GetEndAtom().GetAtomicNum() == 6
            for bond in component.GetBonds()
        )
        tail_carbon_branch_points += sum(
            atom.GetAtomicNum() == 6
            and sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) >= 3
            for atom in component.GetAtoms()
        )
    return {
        "molecular_weight": Descriptors.MolWt(product),
        "heavy_atoms": product.GetNumHeavyAtoms(),
        "unsaturated_tail_bonds": unsaturated_bonds,
        "tail_carbon_branch_points": tail_carbon_branch_points,
    }


def _greedy_select(
    rows: Sequence[dict[str, Any]],
    count: int,
    *,
    potency_weight: float,
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        product = str(row["canonical_product"])
        current = unique.get(product)
        if current is None or (
            -float(row["potency_utility"] or 0.0),
            int(row["draw_index"]),
        ) < (
            -float(current["potency_utility"] or 0.0),
            int(current["draw_index"]),
        ):
            unique[product] = row
    rows = list(unique.values())
    if len(rows) < count:
        raise UgiProductionRouteShortlistError(
            f"shortlist stratum has only {len(rows)} rows for quota {count}"
        )
    candidates = sorted(
        rows,
        key=lambda row: (
            -float(row["potency_utility"] or 0.0),
            str(row["canonical_product"]),
            int(row["draw_index"]),
        ),
    )
    fingerprints = {id(row): _fingerprint(str(row["canonical_product"])) for row in candidates}
    selected = [candidates.pop(0)]
    while len(selected) < count:
        scored = []
        for row in candidates:
            fp = fingerprints[id(row)]
            minimum_distance = min(
                1.0 - DataStructs.TanimotoSimilarity(fp, fingerprints[id(chosen)])
                for chosen in selected
            )
            utility = float(row["potency_utility"] or 0.0)
            objective = (1.0 - potency_weight) * minimum_distance + potency_weight * utility
            scored.append(
                (
                    objective,
                    minimum_distance,
                    utility,
                    str(row["canonical_product"]),
                    row,
                )
            )
        scored.sort(key=lambda value: (-value[0], -value[1], -value[2], value[3]))
        chosen = scored[0][-1]
        selected.append(chosen)
        candidates.remove(chosen)
    return selected


def _gzip_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    payload = io.BytesIO()
    with gzip.GzipFile(fileobj=payload, mode="wb", mtime=0) as handle:
        handle.write(b"".join((_stable_json(dict(row)) + "\n").encode() for row in rows))
    return payload.getvalue()


def build_production_route_shortlist(repo: Path, config_path: Path) -> tuple[dict[str, Any], bytes]:
    """Freeze a matched, route-blinded production shortlist."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path)
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status") != "frozen_before_route_blinded_shortlist_selection"
    ):
        raise UgiProductionRouteShortlistError("unsupported shortlist config")
    paths = {label: _pin(repo, record, label=label) for label, record in config["inputs"].items()}
    generation_result = _load_json(paths["generation_result"])
    ranking_result = _load_json(paths["ranking_result"])
    if (
        generation_result.get("status")
        != "complete_constrained_stochastic_production_candidate_generation"
        or ranking_result.get("status")
        != "complete_chemically_admitted_full_support_terminal_rescoring"
    ):
        raise UgiProductionRouteShortlistError("production inputs are not final")
    if generation_result.get("artifacts", {}).get("terminal_ledger.jsonl.gz", {}).get(
        "sha256"
    ) != sha256_file(paths["generation_ledger"]) or ranking_result.get("artifacts", {}).get(
        "terminal_rescoring.csv.gz", {}
    ).get(
        "sha256"
    ) != sha256_file(
        paths["ranking_ledger"]
    ):
        raise UgiProductionRouteShortlistError("ledger is not pinned by its result")

    generated = _generation_rows(paths["generation_ledger"])
    ranked = _ranking_rows(paths["ranking_ledger"])
    by_key = {(str(row["arm_id"]), int(row["draw_index"])): row for row in generated}
    if len(by_key) != len(generated):
        raise UgiProductionRouteShortlistError("generation keys are not unique")
    ranked_by_key = {(str(row["arm_id"]), int(row["draw_index"])): row for row in ranked}
    if len(ranked_by_key) != len(ranked) or set(ranked_by_key) != set(by_key):
        raise UgiProductionRouteShortlistError("ranking and generation keys differ")

    products_by_arm = {
        arm: {
            str(row["canonical_admitted_product"])
            for row in generated
            if row["arm_id"] == arm and row["terminal_chemical_admission"]["admitted"]
        }
        for arm in ARMS
    }
    cross_arm_overlap = products_by_arm[ARMS[0]] & products_by_arm[ARMS[1]]
    exploit_quotas = {str(k): int(v) for k, v in config["quotas"]["exploitation"].items()}
    explore_quotas = {str(k): int(v) for k, v in config["quotas"]["exploration"].items()}
    output: list[dict[str, Any]] = []
    selected_products: set[str] = set()

    for arm in ARMS:
        exploit_pool: dict[str, list[dict[str, Any]]] = {pattern: [] for pattern in exploit_quotas}
        explore_pool: dict[str, list[dict[str, Any]]] = {
            branch_class: [] for branch_class in explore_quotas
        }
        for key, source in by_key.items():
            if key[0] != arm or not source["terminal_chemical_admission"]["admitted"]:
                continue
            product = str(source["canonical_admitted_product"])
            if product in cross_arm_overlap:
                continue
            rank = ranked_by_key[key]
            base = {
                "arm_id": arm,
                "draw_index": key[1],
                "canonical_product": product,
                "components": dict(source["native_terminal"]["component_smiles_by_role"]),
                "branch_class": str(source["native_terminal"]["branch_class"]),
                "program": dict(source["program"]),
                "program_sha256": str(source["program_sha256"]),
                "exact_refit_corpus_product": bool(source["exact_refit_corpus_product"]),
                "pattern_id": str(rank["pattern_id"]),
                "unseen_roles": str(rank["unseen_roles"]),
                "authority_tier": str(rank["authority_tier"]),
                "potency_utility": (
                    None if not rank["potency_utility"] else float(rank["potency_utility"])
                ),
                "lcb90": None if not rank["lcb90"] else float(rank["lcb90"]),
                "oracle_mean": (None if not rank["oracle_mean"] else float(rank["oracle_mean"])),
                **_chemistry_descriptors(source),
            }
            if (
                _bool(rank["conservative_high_potency"], label="conservative high")
                and rank["pattern_id"] in exploit_pool
            ):
                exploit_pool[rank["pattern_id"]].append(base)
            branch_class = str(source["native_terminal"]["branch_class"])
            if not source["exact_refit_corpus_product"] and branch_class in explore_pool:
                exploration = dict(base)
                exploration.update(
                    {
                        "pattern_id": "",
                        "authority_tier": "unguided_morphology_exploration",
                        "potency_utility": None,
                        "lcb90": None,
                        "oracle_mean": None,
                    }
                )
                explore_pool[branch_class].append(exploration)

        for pattern, quota in exploit_quotas.items():
            chosen = _greedy_select(
                exploit_pool[pattern],
                quota,
                potency_weight=float(config["selection"]["exploitation_potency_weight"]),
            )
            for rank_index, row in enumerate(chosen, start=1):
                if row["canonical_product"] in selected_products:
                    raise UgiProductionRouteShortlistError("shortlist product duplicated")
                selected_products.add(row["canonical_product"])
                output.append(
                    {
                        "schema_version": LEDGER_SCHEMA_VERSION,
                        "cohort": "potency_exploitation",
                        "stratum": pattern,
                        "stratum_rank": rank_index,
                        **row,
                    }
                )
        for branch_class, quota in explore_quotas.items():
            chosen = _greedy_select(explore_pool[branch_class], quota, potency_weight=0.0)
            for rank_index, row in enumerate(chosen, start=1):
                if row["canonical_product"] in selected_products:
                    raise UgiProductionRouteShortlistError("shortlist product duplicated")
                selected_products.add(row["canonical_product"])
                output.append(
                    {
                        "schema_version": LEDGER_SCHEMA_VERSION,
                        "cohort": "morphology_exploration_no_potency_claim",
                        "stratum": branch_class,
                        "stratum_rank": rank_index,
                        **row,
                    }
                )

    output.sort(
        key=lambda row: (
            ARMS.index(str(row["arm_id"])),
            str(row["cohort"]),
            str(row["stratum"]),
            int(row["stratum_rank"]),
        )
    )
    ledger = _gzip_jsonl(output)
    summary: dict[str, Any] = {}
    for arm in ARMS:
        selected = [row for row in output if row["arm_id"] == arm]
        summary[arm] = {
            "rows": len(selected),
            "cohorts": dict(Counter(str(row["cohort"]) for row in selected)),
            "strata": dict(Counter(str(row["stratum"]) for row in selected)),
            "refit_corpus_absent": sum(
                not bool(row["exact_refit_corpus_product"]) for row in selected
            ),
            "unsaturated": sum(int(row["unsaturated_tail_bonds"]) > 0 for row in selected),
            "branched": sum(int(row["tail_carbon_branch_points"]) > 0 for row in selected),
        }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "route_blinded_balanced_production_shortlist_complete",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "policy": {
            "route_outcomes_used": False,
            "cross_arm_product_overlap_excluded_from_both_arms": True,
            "potency_authority_limited_to_exploitation_cohort": True,
            "exploration_potency_neutral": True,
            "matched_quotas": config["quotas"],
            "selection": config["selection"],
        },
        "summary": summary,
        "artifacts": {
            "shortlist.jsonl.gz": {
                "rows": len(output),
                "sha256": hashlib.sha256(ledger).hexdigest(),
                "schema_version": LEDGER_SCHEMA_VERSION,
            }
        },
        "next_gate": "matched_graph2edits_aizynthfinder_and_evidence_route_assessment",
        "nonclaims": [
            "Shortlisting does not establish route closure or experimental synthesis success.",
            "Exploration candidates carry no potency-generalization claim.",
            "This artifact is a route-assessment shortlist, not a locked synthesis panel.",
        ],
    }
    result["result_sha256"] = hashlib.sha256(_stable_json(result).encode()).hexdigest()
    return result, ledger


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiProductionRouteShortlistError",
    "build_production_route_shortlist",
]

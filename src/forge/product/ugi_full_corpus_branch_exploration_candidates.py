"""Generate and seal the full-corpus Ugi branch-exploration candidate pool."""

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

from rdkit import Chem, rdBase

from forge.core.io import stable_json as _stable_json
from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_constrained_stochastic_production_candidates import (
    exact_terminal_admission,
)
from forge.product.ugi_held_component_gate import _reaction_contract

CONFIG_SCHEMA_VERSION = "phase1_ugi_full_corpus_branch_exploration_candidates_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_full_corpus_branch_exploration_candidates.v1"
LEDGER_SCHEMA_VERSION = "forge.ugi_full_corpus_branch_exploration_candidate_ledger.v1"
# Keep the generation-time schema check local. Importing the schedule builder
# would pull its analysis-only morphology scorer (and scikit-learn) into the
# minimal sampler image even though generation consumes only frozen JSON.
SCHEDULE_SCHEMA_VERSION = "forge.ugi_branch_exploration_schedule.v1"


class UgiFullCorpusBranchExplorationCandidatesError(RuntimeError):
    """Raised when fresh branch generation violates its frozen contract."""


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiFullCorpusBranchExplorationCandidatesError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiFullCorpusBranchExplorationCandidatesError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiFullCorpusBranchExplorationCandidatesError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiFullCorpusBranchExplorationCandidatesError(
            f"pin escapes repository: {label}"
        ) from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiFullCorpusBranchExplorationCandidatesError(f"pin changed: {label}")
    return path


def _canonical(smiles: str) -> str:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiFullCorpusBranchExplorationCandidatesError(f"invalid admitted molecule: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _carbon_branch_points(molecule: Chem.Mol) -> list[Chem.Atom]:
    return [
        atom
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 6
        and sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) >= 3
    ]


def _component_descriptors(smiles: str) -> dict[str, Any]:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiFullCorpusBranchExplorationCandidatesError(
            f"generated component no longer parses: {smiles}"
        )
    branch_points = _carbon_branch_points(molecule)
    branch_indices = {atom.GetIdx() for atom in branch_points}
    adjacent_branch_edges = sum(
        bond.GetBeginAtomIdx() in branch_indices and bond.GetEndAtomIdx() in branch_indices
        for bond in molecule.GetBonds()
    )
    carbon_carbon_double_bonds = sum(
        bond.GetBondTypeAsDouble() == 2.0
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        for bond in molecule.GetBonds()
    )
    ester_carbonyls = 0
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        has_double_oxygen = any(
            bond.GetBondTypeAsDouble() == 2.0 and bond.GetOtherAtom(atom).GetAtomicNum() == 8
            for bond in atom.GetBonds()
        )
        has_single_alkoxy = any(
            bond.GetBondTypeAsDouble() == 1.0
            and bond.GetOtherAtom(atom).GetAtomicNum() == 8
            and any(
                neighbor.GetAtomicNum() == 6 and neighbor.GetIdx() != atom.GetIdx()
                for neighbor in bond.GetOtherAtom(atom).GetNeighbors()
            )
            for bond in atom.GetBonds()
        )
        ester_carbonyls += int(has_double_oxygen and has_single_alkoxy)
    return {
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "carbon_atoms": sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()),
        "carbon_branch_points": len(branch_points),
        "adjacent_carbon_branch_edges": adjacent_branch_edges,
        "carbon_carbon_double_bonds": carbon_carbon_double_bonds,
        "ester_carbonyls": ester_carbonyls,
    }


def realized_branch_class(components: Mapping[str, str]) -> tuple[str, dict[str, Any]]:
    """Classify realized carbon branching and return role-level descriptors."""

    roles = ("oxoester_aldehyde_body_tail", "isocyanide_tail")
    if not set(roles).issubset(components):
        raise UgiFullCorpusBranchExplorationCandidatesError("terminal components lack tail roles")
    descriptors = {role: _component_descriptors(str(components[role])) for role in roles}
    aldehyde = descriptors[roles[0]]["carbon_branch_points"] > 0
    isocyanide = descriptors[roles[1]]["carbon_branch_points"] > 0
    if aldehyde and isocyanide:
        branch = "both_tail_origins_branched"
    elif aldehyde:
        branch = "aldehyde_origin_branched"
    elif isocyanide:
        branch = "isocyanide_origin_branched"
    else:
        branch = "linear_tail_origins"
    return branch, descriptors


def program_shard(
    schedule: Mapping[str, Any], *, shard_index: int, shard_draws: int
) -> dict[str, Any]:
    """Materialize one sampler-compatible branch exploration shard."""

    if shard_index < 0 or shard_draws < 1:
        raise UgiFullCorpusBranchExplorationCandidatesError("invalid branch shard request")
    if (
        schedule.get("schema_version") != SCHEDULE_SCHEMA_VERSION
        or schedule.get("status") != "frozen_before_branch_exploration_generation"
    ):
        raise UgiFullCorpusBranchExplorationCandidatesError("branch schedule is not frozen")
    records = schedule.get("records")
    if not isinstance(records, list):
        raise UgiFullCorpusBranchExplorationCandidatesError("branch schedule records are malformed")
    start = shard_index * shard_draws
    selected = records[start : start + shard_draws]
    if len(selected) != shard_draws:
        raise UgiFullCorpusBranchExplorationCandidatesError(
            "branch schedule does not contain the requested shard"
        )
    samples = []
    for row in selected:
        draw_index = int(row["draw_index"])
        samples.append(
            {
                "product_id": f"branch-exploration-v1-{draw_index:05d}",
                "source_stratum": "full_corpus_branch_conditional_program_prior",
                "branch_class": str(row["branch_class"]),
                "component_novelty_class": "generated_complete_components",
                "program": dict(row["program"]),
            }
        )
    return {
        "schema_version": "phase1_ugi_program_probe.v1",
        "fold": "frozen_full_corpus_branch_exploration_schedule_v1",
        "seed": int(schedule["design"]["seed"]),
        "input_prior": {
            "schedule_sha256": str(schedule["schedule_sha256"]),
            "shard_index": shard_index,
        },
        "samples": samples,
        "stratum_branch_counts": dict(
            sorted(Counter(sample["branch_class"] for sample in samples).items())
        ),
    }


def _gzip_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as handle:
        for row in rows:
            handle.write((_stable_json(row) + "\n").encode())
    return output.getvalue()


def collect_branch_exploration_candidates(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Verify immutable raw shards and seal exact terminal branch diagnostics."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="branch generation config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status") != "frozen_before_fresh_branch_exploration_generation"
    ):
        raise UgiFullCorpusBranchExplorationCandidatesError("unsupported generation config")
    paths = {label: _pin(repo, record, label=label) for label, record in config["inputs"].items()}
    implementation_paths = {
        label: _pin(repo, record, label=f"implementation.{label}")
        for label, record in config["implementation"].items()
    }
    schedule = _load_json(paths["schedule"], label="branch schedule")
    schedule_result = _load_json(paths["schedule_result"], label="branch schedule result")
    if schedule_result.get(
        "status"
    ) != "full_corpus_morphology_support_and_branch_schedule_frozen" or schedule_result.get(
        "artifacts", {}
    ).get("schedule.json", {}).get("schedule_sha256") != schedule.get("schedule_sha256"):
        raise UgiFullCorpusBranchExplorationCandidatesError("branch schedule receipt changed")
    reaction = _reaction_contract(paths["qualified_reactions"])
    reference_products = set()
    with gzip.open(paths["assignments"], "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            reference_products.add(str(row["canonical_product_smiles"]))
    if len(reference_products) != 112386:
        raise UgiFullCorpusBranchExplorationCandidatesError(
            "all-fold reference product population changed"
        )

    design = config["design"]
    shard_draws = int(design["shard_draws"])
    shard_count = int(design["shards"])
    records: list[dict[str, Any]] = []
    receipts = []
    raw_root = repo / str(design["raw_root"])
    for shard in range(shard_count):
        expected = program_shard(schedule, shard_index=shard, shard_draws=shard_draws)
        result_path = raw_root / f"shard_{shard:02d}/result.json"
        result = _load_json(result_path, label=f"raw branch result {shard}")
        sampling = result.get("sampling", {})
        decoder = sampling.get("terminal_decoder", {})
        terminals = result.get("samples")
        expected_flow_seed = int(design["flow_seed_base"]) + shard
        expected_terminal_seed = int(design["terminal_seed_base"]) + shard
        if (
            int(result.get("seed", -1)) != expected_flow_seed
            or decoder.get("mode") != "stochastic"
            or int(decoder.get("seed", -1)) != expected_terminal_seed
            or float(decoder.get("temperature", -1.0)) != 1.0
            or int(sampling.get("sample_steps", -1)) != int(design["sample_steps"])
            or sampling.get("maximum_adjacent_branch_runs")
            != design["maximum_adjacent_branch_runs"]
            or sampling.get("terminal_tree_repairs") != 0
            or not isinstance(terminals, list)
            or len(terminals) != shard_draws
        ):
            raise UgiFullCorpusBranchExplorationCandidatesError(
                f"raw branch sampling contract changed: shard {shard}"
            )
        start = shard * shard_draws
        scheduled = schedule["records"][start : start + shard_draws]
        for local_index, (terminal, probe, coordinate) in enumerate(
            zip(terminals, expected["samples"], scheduled, strict=True)
        ):
            draw_index = int(coordinate["draw_index"])
            if (
                terminal.get("program") != probe["program"]
                or terminal.get("product_id") != probe["product_id"]
                or terminal.get("program") != coordinate["program"]
                or draw_index != start + local_index
            ):
                raise UgiFullCorpusBranchExplorationCandidatesError(
                    f"raw branch alignment changed: {shard}/{local_index}"
                )
            admission = exact_terminal_admission(terminal, reaction)
            canonical = _canonical(str(terminal["smiles"])) if admission["admitted"] else None
            actual_branch = None
            descriptors = None
            if admission["admitted"]:
                actual_branch, descriptors = realized_branch_class(
                    terminal["component_smiles_by_role"]
                )
            records.append(
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "draw_index": draw_index,
                    "shard_index": shard,
                    "shard_local_index": local_index,
                    "scheduled_branch_class": str(coordinate["branch_class"]),
                    "realized_carbon_branch_class": actual_branch,
                    "program_sha256": str(coordinate["program_sha256"]),
                    "program": dict(coordinate["program"]),
                    "broad_prior_probability": float(coordinate["broad_prior_probability"]),
                    "branch_conditional_probability": float(
                        coordinate["branch_conditional_probability"]
                    ),
                    "stratified_schedule_probability": float(
                        coordinate["stratified_schedule_probability"]
                    ),
                    "importance_ratio_conditional_over_schedule": float(
                        coordinate["importance_ratio_conditional_over_schedule"]
                    ),
                    "flow_seed": expected_flow_seed,
                    "terminal_seed": expected_terminal_seed,
                    "native_terminal": terminal,
                    "terminal_chemical_admission": admission,
                    "canonical_admitted_product": canonical,
                    "exact_refit_corpus_product": bool(
                        canonical is not None and canonical in reference_products
                    ),
                    "tail_component_descriptors": descriptors,
                }
            )
        receipts.append(
            {
                "shard": shard,
                "result": {
                    "path": str(result_path.relative_to(repo)),
                    "sha256": sha256_file(result_path),
                },
                "attempted": len(terminals),
                "raw_valid": sum(row.get("valid") is True for row in terminals),
            }
        )
    if len(records) != int(design["draws"]):
        raise UgiFullCorpusBranchExplorationCandidatesError(
            "combined branch ledger has the wrong denominator"
        )
    admitted = [row for row in records if row["terminal_chemical_admission"]["admitted"]]
    realized = Counter(str(row["realized_carbon_branch_class"]) for row in admitted)
    realized_branched = [
        row for row in admitted if row["realized_carbon_branch_class"] != "linear_tail_origins"
    ]
    long_ester_branched_aldehydes = 0
    adjacent_branch_products = 0
    for row in realized_branched:
        descriptors = row["tail_component_descriptors"]
        aldehyde = descriptors["oxoester_aldehyde_body_tail"]
        if (
            aldehyde["carbon_branch_points"] > 0
            and aldehyde["carbon_atoms"] >= 14
            and aldehyde["ester_carbonyls"] > 0
        ):
            long_ester_branched_aldehydes += 1
        adjacent_branch_products += int(
            any(value["adjacent_carbon_branch_edges"] > 0 for value in descriptors.values())
        )
    ledger = _gzip_jsonl(records)
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
        "status": "complete_full_corpus_branch_exploration_generation",
        "generator_identity": (
            "constrained stochastic graph-flow generator for complete "
            "Ugi-compatible ionizable lipids"
        ),
        "scope": dict(config["scope"]),
        "inputs": pins,
        "implementation": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(implementation_paths.items())
        },
        "design": dict(design),
        "counts": {
            "raw_terminal_attempts": len(records),
            "raw_valid": sum(
                row["terminal_chemical_admission"]["raw_molecule_valid"] for row in records
            ),
            "exact_l1": sum(row["terminal_chemical_admission"]["exact_l1"] for row in records),
            "terminal_chemical_admitted": len(admitted),
            "terminal_chemical_admitted_fraction": len(admitted) / len(records),
            "unique_admitted_products": len(
                {row["canonical_admitted_product"] for row in admitted}
            ),
            "corpus_absent_admitted": sum(
                not row["exact_refit_corpus_product"] for row in admitted
            ),
            "realized_carbon_branch_classes": dict(sorted(realized.items())),
            "realized_carbon_branched": len(realized_branched),
            "realized_long_ester_branched_aldehydes": long_ester_branched_aldehydes,
            "realized_products_with_adjacent_carbon_branch_points": adjacent_branch_products,
            "admission_reasons": dict(
                sorted(
                    Counter(row["terminal_chemical_admission"]["reason"] for row in records).items()
                )
            ),
        },
        "raw_shard_receipts": receipts,
        "artifacts": {
            "terminal_ledger.jsonl.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(records),
                "sha256": hashlib.sha256(ledger).hexdigest(),
            }
        },
        "next_gate": "realized_branch_chemotype_fidelity_then_route_blinded_exploration_shortlist",
        "nonclaims": [
            "Scheduled morphology branching is reported separately from realized carbon branching.",
            "This branch exploration pool carries no potency claim.",
            "Terminal chemical admission is not complete L2/L3 route closure.",
            "No biological or synthesis score altered generation.",
        ],
    }
    result = {**content, "result_sha256": _logical_sha256(content)}
    return result, ledger


__all__ = [
    "UgiFullCorpusBranchExplorationCandidatesError",
    "collect_branch_exploration_candidates",
    "program_shard",
    "realized_branch_class",
]

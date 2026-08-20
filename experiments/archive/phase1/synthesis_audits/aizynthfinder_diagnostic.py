"""Fail-closed helpers for an AiZynthFinder route-hypothesis diagnostic.

The public AiZynthFinder planner is used here only to test whether a broader
retrosynthesis engine recovers hypotheses for FORGE components that the current
evidence ledger leaves unresolved.  Its outputs are deliberately segregated
from evidence-bearing route records: a solved public-stock search is a planner
result, not proof that a route works for the exact substrate.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from forge.core.io import atomic_write as _atomic_write

AUDIT_RESULT_SCHEMA_VERSION = "phase1_ugi_high_potency_molecule_route_audit_result.v1"
TARGET_SCHEMA_VERSION = "phase1_aizynthfinder_component_target.v1"
RESULT_SCHEMA_VERSION = "phase1_aizynthfinder_component_route_diagnostic_result.v1"


class AiZynthFinderDiagnosticError(ValueError):
    """Raised when diagnostic inputs or worker outputs violate their contract."""


def content_sha256(value: Any) -> str:
    """Return a stable SHA-256 over canonical JSON."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AiZynthFinderDiagnosticError(f"{label} must be an object")
    return value


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AiZynthFinderDiagnosticError(f"{label} must be a nonempty string")
    return value


def _canonical_smiles(value: Any, *, label: str) -> str:
    smiles = _nonempty(value, label=label)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise AiZynthFinderDiagnosticError(f"{label} is not valid SMILES")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise AiZynthFinderDiagnosticError(f"{label} must describe one connected component")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _heavy_atoms(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:  # pragma: no cover - canonicalized before this boundary
        raise AiZynthFinderDiagnosticError(f"could not parse canonical target {smiles}")
    return molecule.GetNumHeavyAtoms()


@dataclass(frozen=True)
class DiagnosticTarget:
    """One unresolved component or matched route-ready control."""

    cohort: str
    role: str
    canonical_smiles: str
    paired_smiles: tuple[str, ...]
    nearest_similarity: float | None

    @property
    def target_id(self) -> str:
        return content_sha256(
            {
                "schema_version": TARGET_SCHEMA_VERSION,
                "cohort": self.cohort,
                "role": self.role,
                "canonical_smiles": self.canonical_smiles,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TARGET_SCHEMA_VERSION,
            "target_id": self.target_id,
            "cohort": self.cohort,
            "role": self.role,
            "canonical_smiles": self.canonical_smiles,
            "heavy_atoms": _heavy_atoms(self.canonical_smiles),
            "paired_smiles": list(self.paired_smiles),
            "nearest_similarity": self.nearest_similarity,
        }


def targets_from_molecule_audit(audit_result: dict[str, Any]) -> list[DiagnosticTarget]:
    """Build unresolved and matched-control targets from the frozen molecule audit."""

    if audit_result.get("schema_version") != AUDIT_RESULT_SCHEMA_VERSION:
        raise AiZynthFinderDiagnosticError("molecule-audit result schema is unsupported")
    summary = _mapping(audit_result.get("summary"), label="audit summary")
    diagnostics = _mapping(
        summary.get("unresolved_component_diagnostics"),
        label="unresolved component diagnostics",
    )
    by_role = _mapping(diagnostics.get("by_role"), label="diagnostics by_role")

    pairings: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    similarities: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for role, raw_role_record in sorted(by_role.items()):
        _nonempty(role, label="component role")
        role_record = _mapping(raw_role_record, label=f"diagnostics for {role}")
        nearest_records = role_record.get("nearest_reference_records")
        if not isinstance(nearest_records, list) or not nearest_records:
            raise AiZynthFinderDiagnosticError(f"diagnostics for {role} have no nearest records")
        for index, raw_record in enumerate(nearest_records):
            record = _mapping(raw_record, label=f"nearest record {role}[{index}]")
            unresolved = _canonical_smiles(
                record.get("unresolved_component"), label="unresolved component"
            )
            reference = _canonical_smiles(
                record.get("nearest_route_ready_component"),
                label="nearest route-ready component",
            )
            try:
                similarity = float(record.get("morgan_tanimoto"))
            except (TypeError, ValueError) as exc:
                raise AiZynthFinderDiagnosticError("nearest similarity must be numeric") from exc
            if not 0.0 <= similarity <= 1.0:
                raise AiZynthFinderDiagnosticError(
                    "nearest similarity must be between zero and one"
                )

            unresolved_key = ("unresolved", role, unresolved)
            control_key = ("matched_route_ready_control", role, reference)
            pairings[unresolved_key].add(reference)
            pairings[control_key].add(unresolved)
            similarities[unresolved_key].append(similarity)
            similarities[control_key].append(similarity)

    targets = [
        DiagnosticTarget(
            cohort=cohort,
            role=role,
            canonical_smiles=smiles,
            paired_smiles=tuple(sorted(paired)),
            nearest_similarity=max(similarities[(cohort, role, smiles)]),
        )
        for (cohort, role, smiles), paired in pairings.items()
    ]
    return sorted(targets, key=lambda target: (target.cohort, target.role, target.canonical_smiles))


def select_full_search_targets(
    targets: Iterable[DiagnosticTarget],
    *,
    maximum_unresolved_per_role: int,
) -> set[str]:
    """Select a size-spanning unresolved panel and every paired matched control."""

    if maximum_unresolved_per_role < 1:
        raise AiZynthFinderDiagnosticError("maximum_unresolved_per_role must be positive")
    records = list(targets)
    unresolved_by_role: dict[str, list[DiagnosticTarget]] = defaultdict(list)
    target_by_identity = {
        (target.cohort, target.role, target.canonical_smiles): target for target in records
    }
    for target in records:
        if target.cohort == "unresolved":
            unresolved_by_role[target.role].append(target)

    selected: set[str] = set()
    for role, role_targets in sorted(unresolved_by_role.items()):
        ordered = sorted(
            role_targets,
            key=lambda target: (_heavy_atoms(target.canonical_smiles), target.canonical_smiles),
        )
        if len(ordered) <= maximum_unresolved_per_role:
            chosen = ordered
        elif maximum_unresolved_per_role == 1:
            chosen = [ordered[len(ordered) // 2]]
        else:
            indices = {
                round(index * (len(ordered) - 1) / (maximum_unresolved_per_role - 1))
                for index in range(maximum_unresolved_per_role)
            }
            chosen = [ordered[index] for index in sorted(indices)]
        for target in chosen:
            selected.add(target.target_id)
            for paired_smiles in target.paired_smiles:
                control = target_by_identity.get(
                    ("matched_route_ready_control", role, paired_smiles)
                )
                if control is None:
                    raise AiZynthFinderDiagnosticError(
                        f"missing matched control for {role}: {paired_smiles}"
                    )
                selected.add(control.target_id)
    return selected


def summarize_worker_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize proposal yield and public-stock planner outcomes by cohort and role."""

    rows = list(records)
    if not rows:
        raise AiZynthFinderDiagnosticError("worker ledger is empty")
    seen: set[str] = set()
    counters: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "targets": 0,
            "targets_with_single_step_proposals": 0,
            "single_step_proposals": 0,
            "planner_attempted": 0,
            "planner_solved_to_public_stock": 0,
        }
    )
    for row in rows:
        target = _mapping(row.get("target"), label="worker target")
        target_id = _nonempty(target.get("target_id"), label="worker target_id")
        if target_id in seen:
            raise AiZynthFinderDiagnosticError(f"duplicate worker target_id: {target_id}")
        seen.add(target_id)
        cohort = _nonempty(target.get("cohort"), label="worker cohort")
        role = _nonempty(target.get("role"), label="worker role")
        proposals = row.get("single_step_proposals")
        if not isinstance(proposals, list):
            raise AiZynthFinderDiagnosticError("single_step_proposals must be a list")
        bucket = counters[(cohort, role)]
        bucket["targets"] += 1
        bucket["single_step_proposals"] += len(proposals)
        bucket["targets_with_single_step_proposals"] += int(bool(proposals))
        planner = row.get("full_search")
        if planner is not None:
            planner_record = _mapping(planner, label="full_search")
            bucket["planner_attempted"] += 1
            bucket["planner_solved_to_public_stock"] += int(
                bool(planner_record.get("is_solved_to_public_stock"))
            )

    by_cohort_role = {
        f"{cohort}:{role}": values for (cohort, role), values in sorted(counters.items())
    }
    total = {
        key: sum(bucket[key] for bucket in counters.values())
        for key in next(iter(counters.values()))
    }
    return {"total": total, "by_cohort_and_role": by_cohort_role}


def _molecule_png(smiles: str, *, width: int, height: int) -> bytes:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise AiZynthFinderDiagnosticError(f"could not render molecule: {smiles}")
    rdMolDraw2D.PrepareMolForDrawing(molecule)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.padding = 0.07
    options.minFontSize = 11
    options.maxFontSize = 21
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _stock_leaf_smiles(route: dict[str, Any]) -> list[str]:
    leaves: list[str] = []

    def visit(node: dict[str, Any]) -> None:
        children = node.get("children")
        if not isinstance(children, list) or not children:
            if node.get("is_chemical") and node.get("in_stock"):
                smiles = node.get("smiles")
                if isinstance(smiles, str) and smiles:
                    leaves.append(smiles)
            return
        for child in children:
            if isinstance(child, dict):
                visit(child)

    visit(route)
    return sorted(set(leaves))


def render_route_hypothesis_pdf(
    *,
    result: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    output_path: Path,
) -> None:
    """Render planner-attempted targets and proposal hypotheses for human review."""

    try:
        from reportlab.lib.colors import HexColor, black, white
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - optional report dependency
        raise AiZynthFinderDiagnosticError(
            "PDF rendering requires the optional report dependencies"
        ) from exc

    planner_rows = [row for row in rows if row.get("full_search") is not None]
    planner_rows.sort(
        key=lambda row: (
            row["target"]["cohort"],
            row["target"]["role"],
            row["target"]["canonical_smiles"],
        )
    )
    if not planner_rows:
        raise AiZynthFinderDiagnosticError("no planner-attempted rows are available to render")

    page_width, page_height = landscape(letter)
    buffer = io.BytesIO()
    document = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    document.setTitle("FORGE independent retrosynthesis route-hypothesis audit")
    navy = HexColor("#284F7A")
    blue = HexColor("#C7DCEF")
    rose = HexColor("#E7BBC2")
    green = HexColor("#B9D8C2")
    gray = HexColor("#F5F7F9")

    document.setFillColor(navy)
    document.rect(0, page_height - 96, page_width, 96, fill=1, stroke=0)
    document.setFillColor(white)
    document.setFont("Helvetica-Bold", 22)
    document.drawString(36, page_height - 49, "Independent retrosynthesis hypothesis audit")
    document.setFont("Helvetica", 10.5)
    document.drawString(
        36,
        page_height - 72,
        "Public AiZynthFinder v4.4.1 policy; proposals quarantined from route evidence and synthesis value",
    )
    summary = result["summary"]["total"]
    document.setFillColor(black)
    document.setFont("Helvetica-Bold", 15)
    document.drawString(36, page_height - 140, "Matched diagnostic result")
    metrics = [
        ("Component targets", summary["targets"]),
        ("Targets with top-10 hypotheses", summary["targets_with_single_step_proposals"]),
        ("Bounded full searches", summary["planner_attempted"]),
        ("Solved to public stock", summary["planner_solved_to_public_stock"]),
    ]
    x_positions = [36, 228, 420, 612]
    for x, (label, value) in zip(x_positions, metrics, strict=True):
        document.setFillColor(gray)
        document.roundRect(x, page_height - 232, 160, 70, 8, fill=1, stroke=0)
        document.setFillColor(navy)
        document.setFont("Helvetica-Bold", 22)
        document.drawCentredString(x + 80, page_height - 194, str(value))
        document.setFillColor(black)
        document.setFont("Helvetica", 8.5)
        document.drawCentredString(x + 80, page_height - 216, label)
    document.setFont("Helvetica-Bold", 12)
    document.drawString(36, page_height - 282, "What this diagnostic can establish")
    document.setFont("Helvetica", 10)
    statements = [
        "- Whether a broader learned policy can propose disconnections for components our current ledger misses.",
        "- Whether public-stock search discriminates unresolved generated components from matched accepted controls.",
        "- It cannot certify substrate scope, conditions, purification, procurement, or experimental success.",
        "- A failed bounded search is an abstention, not evidence that the target is unsynthesizable.",
    ]
    y = page_height - 310
    for statement in statements:
        document.drawString(48, y, statement)
        y -= 24
    document.showPage()

    for page_index, row in enumerate(planner_rows, start=1):
        target = row["target"]
        planner = row["full_search"]
        solved = bool(planner["is_solved_to_public_stock"])
        cohort_color = rose if target["cohort"] == "unresolved" else blue
        document.setFillColor(cohort_color)
        document.rect(0, page_height - 54, page_width, 54, fill=1, stroke=0)
        document.setFillColor(black)
        document.setFont("Helvetica-Bold", 15)
        document.drawString(
            28,
            page_height - 33,
            f"{target['cohort'].replace('_', ' ')} - {target['role'].replace('_', ' ')}",
        )
        document.setFillColor(green if solved else white)
        document.roundRect(page_width - 214, page_height - 43, 184, 26, 7, fill=1, stroke=0)
        document.setFillColor(black)
        document.setFont("Helvetica-Bold", 9)
        document.drawCentredString(
            page_width - 122,
            page_height - 34,
            "SOLVED TO PUBLIC STOCK" if solved else "NO SOLVED ROUTE IN BOUNDED SEARCH",
        )

        target_png = _molecule_png(target["canonical_smiles"], width=1300, height=320)
        document.drawImage(
            ImageReader(io.BytesIO(target_png)),
            32,
            405,
            width=728,
            height=118,
            preserveAspectRatio=True,
            anchor="c",
        )
        document.setFont("Helvetica", 7.2)
        document.drawString(
            32,
            395,
            f"Target: {target['canonical_smiles'][:122]}",
        )
        stats = planner["statistics"]
        document.drawRightString(
            page_width - 32,
            395,
            f"iterations {stats.get('profiling', {}).get('iterations', 0)} | "
            f"routes {stats.get('number_of_routes', 0)} | "
            f"solved {stats.get('number_of_solved_routes', 0)}",
        )

        if solved and planner["top_route_hypotheses"]:
            stock_leaves = _stock_leaf_smiles(planner["top_route_hypotheses"][0])
            document.setFont("Helvetica-Bold", 7.2)
            document.drawString(
                32,
                381,
                "First solved route - public-stock leaves: " + " + ".join(stock_leaves)[:112],
            )

        document.setFont("Helvetica-Bold", 10)
        document.drawString(32, 365, "Highest-ranked single-step hypotheses")
        proposals = row["single_step_proposals"][:3]
        panel_width = 238
        for proposal_index, proposal in enumerate(proposals):
            x = 30 + proposal_index * 253
            document.setFillColor(gray)
            document.roundRect(x, 116, panel_width, 244, 8, fill=1, stroke=0)
            reactants_smiles = ".".join(proposal["canonical_reactants"])
            proposal_png = _molecule_png(reactants_smiles, width=720, height=430)
            document.drawImage(
                ImageReader(io.BytesIO(proposal_png)),
                x + 7,
                190,
                width=panel_width - 14,
                height=156,
                preserveAspectRatio=True,
                anchor="c",
            )
            metadata = proposal["metadata"]
            probability = metadata.get("policy_probability")
            feasibility = metadata.get("feasibility")
            document.setFillColor(black)
            document.setFont("Helvetica-Bold", 9)
            document.drawString(x + 9, 176, f"Hypothesis {proposal['rank']}")
            document.setFont("Helvetica", 7.5)
            document.drawString(
                x + 9,
                163,
                f"policy {probability:.3f} | model feasibility {feasibility:.3f}",
            )
            document.drawString(
                x + 9,
                150,
                f"template {str(metadata.get('template_code'))} | proposal only",
            )
            smiles = reactants_smiles
            document.setFont("Courier", 5.8)
            for line_number in range(3):
                start = line_number * 62
                if start >= len(smiles):
                    break
                document.drawString(x + 9, 137 - 9 * line_number, smiles[start : start + 62])

        document.setFont("Helvetica", 7.5)
        document.drawString(
            30,
            91,
            "Interpretation: model probabilities and public-stock closure are diagnostic outputs, not synthesis-success probabilities.",
        )
        document.drawString(
            30,
            76,
            "Every proposed step still requires independent forward verification, evidence review, operational assessment, and terminal-material closure.",
        )
        document.drawRightString(
            page_width - 30,
            30,
            f"audit page {page_index + 1}/{len(planner_rows) + 1}",
        )
        document.showPage()

    document.save()
    _atomic_write(output_path, buffer.getvalue())


__all__ = [
    "AiZynthFinderDiagnosticError",
    "DiagnosticTarget",
    "content_sha256",
    "render_route_hypothesis_pdf",
    "select_full_search_targets",
    "summarize_worker_records",
    "targets_from_molecule_audit",
]

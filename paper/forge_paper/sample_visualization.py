"""Render deterministic paper figures of FORGE-generated ionizable lipids.

The displays are illustrative and nonselecting. They choose exact-L1, open-ended seed-0 products
that fall inside the frozen descriptor manifold and deduplicate constitutional identities. The
current figure shows the generated graph, exact precursor-origin coordinates, and exact L1 building
blocks. The superseded v1 display with post-hoc conformers remains reproducible for provenance.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdDepictor, rdDistGeom, rdFingerprintGenerator, rdForceFieldHelpers
from rdkit.Chem.Draw import rdMolDraw2D

from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import atomic_write, iter_jsonl, read_json_object, write_json
from forge.potency.annotations import annotate_qualified_ugi_product
from forge.synthesis.engine.qualified_forward import load_qualified_forward_reaction

CONFIG_SCHEMA_V1 = "forge.paper.forge_generated_sample_figure_config.v1"
CONFIG_SCHEMA_V2 = "forge.paper.forge_generated_sample_figure_config.v2"
RESULT_SCHEMA_V1 = "forge.paper.forge_generated_sample_figure.v1"
RESULT_SCHEMA_V2 = "forge.paper.forge_generated_sample_figure.v2"
ATLAS_CONFIG_SCHEMA = "forge.paper.forge_generated_sample_atlas_config.v2"
ATLAS_RESULT_SCHEMA = "forge.paper.forge_generated_sample_atlas.v2"

_COMMON_LEDGER_SCHEMA = "forge.common_ugi_assessed_attempts.v1"
_REALISM_LEDGER_SCHEMA = "forge.common_lipid_realism_attempt.v1"
_ROLE_ORDER = (
    "amine_head",
    "oxoester_aldehyde_body_tail",
    "isocyanide_tail",
)
_ROLE_NAMES = {
    "amine_head": "Amine",
    "oxoester_aldehyde_body_tail": "Aldehyde",
    "isocyanide_tail": "Isocyanide",
}
_BACKGROUND = (250, 250, 253, 255)
_PANEL = (255, 255, 255, 255)
_PANEL_TINT = (246, 246, 252, 255)
_DIVIDER = (205, 207, 218, 255)
_TEXT = (32, 35, 48, 255)
_MUTED = (102, 106, 124, 255)
_ELEMENT_RGB: dict[str, tuple[int, int, int, int]] = {
    "C": (126, 132, 210, 255),
    "N": (42, 59, 224, 255),
    "O": (225, 83, 76, 255),
    "S": (215, 170, 42, 255),
    "P": (236, 139, 48, 255),
    "F": (74, 176, 92, 255),
    "Cl": (74, 176, 92, 255),
    "Br": (156, 77, 59, 255),
    "I": (111, 73, 156, 255),
}

# Atlas conformer weights, expressed as fractions of one drawn C--C bond. Keeping them relative
# means the canvas aspect can change without the drawing turning spindly or clotted.
_TYPICAL_BOND_ANGSTROM = 1.5
_CONFORMER_BOND_STROKE = 0.082
_CONFORMER_ATOM_RADIUS = 0.091
_CONFORMER_ATOM_DEPTH_GAIN = 0.045
_CONFORMER_HETEROATOM_GAIN = 0.027

# The appendix atlas renders each row as a wide constitutional graph beside a squarer decorated
# view; the two canvases carry the aspect ratio the LaTeX row expects.
_ATLAS_GRAPH_CANVAS = (1800, 680)
_ATLAS_VIEW_CANVAS = (1360, 900)
_ATLAS_TRIM_MARGIN = 0.025


class ForgeSampleFigureError(ValueError):
    """The pinned display population or rendering contract changed."""


def _read_ledger(path: Path, schema: str) -> list[dict[str, Any]]:
    records = list(iter_jsonl(path))
    if (
        not records
        or records[0] != {"schema_version": schema, "rows": len(records) - 1}
        or not all(isinstance(row, Mapping) for row in records[1:])
    ):
        raise ForgeSampleFigureError(f"figure input ledger changed: {path}")
    return [dict(row) for row in records[1:]]


def _font_candidates(*, serif: bool, bold: bool) -> tuple[Path, ...]:
    family = "DejaVuSerif" if serif else "DejaVuSans"
    suffix = "-Bold" if bold else ""
    name = f"{family}{suffix}.ttf"
    return (
        Path("/usr/local/texlive/2026/texmf-dist/fonts/truetype/public/dejavu") / name,
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf"
        ),
    )


def _font(size: int, *, serif: bool = False, bold: bool = False) -> ImageFont.FreeTypeFont:
    for candidate in _font_candidates(serif=serif, bold=bold):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    raise ForgeSampleFigureError("no supported TrueType font is available for figure rendering")


def _mono_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "-Bold" if bold else ""
    name = f"DejaVuSansMono{suffix}.ttf"
    candidates = (
        Path("/usr/local/texlive/2026/texmf-dist/fonts/truetype/public/dejavu") / name,
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path(
            "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Courier New.ttf"
        ),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    raise ForgeSampleFigureError("no supported monospaced font is available for figure rendering")


def _centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int] = _TEXT,
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - width) // 2
    y = box[1] + (box[3] - box[1] - height) // 2 - bounds[1]
    draw.text((x, y), text, font=font, fill=fill)


def _paste_contain(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    padding: int = 10,
) -> None:
    available = (box[2] - box[0] - 2 * padding, box[3] - box[1] - 2 * padding)
    rendered = image.copy()
    rendered.thumbnail(available, Image.Resampling.LANCZOS)
    x = box[0] + (box[2] - box[0] - rendered.width) // 2
    y = box[1] + (box[3] - box[1] - rendered.height) // 2
    canvas.alpha_composite(rendered.convert("RGBA"), (x, y))


def _rdkit_2d(smiles: str, width: int, height: int, *, font_size: int) -> Image.Image:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ForgeSampleFigureError(f"RDKit could not parse selected constitution: {smiles}")
    rdDepictor.Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    setattr(options, "padding", 0.06)
    setattr(options, "bondLineWidth", 3.2)
    setattr(options, "fixedFontSize", font_size)
    options.setAtomPalette(
        {
            6: tuple(channel / 255.0 for channel in _ELEMENT_RGB["C"][:3]),
            7: tuple(channel / 255.0 for channel in _ELEMENT_RGB["N"][:3]),
            8: tuple(channel / 255.0 for channel in _ELEMENT_RGB["O"][:3]),
            15: tuple(channel / 255.0 for channel in _ELEMENT_RGB["P"][:3]),
            16: tuple(channel / 255.0 for channel in _ELEMENT_RGB["S"][:3]),
            17: tuple(channel / 255.0 for channel in _ELEMENT_RGB["Cl"][:3]),
            35: tuple(channel / 255.0 for channel in _ELEMENT_RGB["Br"][:3]),
        }
    )
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")


def _hex_rgb(value: str) -> tuple[float, float, float]:
    if len(value) != 7 or not value.startswith("#"):
        raise ForgeSampleFigureError(f"semantic-map color must be #RRGGBB: {value!r}")
    try:
        channels = tuple(int(value[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
    except ValueError as exc:
        raise ForgeSampleFigureError(f"semantic-map color is invalid: {value!r}") from exc
    return channels[0], channels[1], channels[2]


def _embed_conformers(
    smiles: str,
    *,
    count: int,
    seed: int,
    max_iterations: int,
) -> tuple[Chem.Mol, list[int], str, list[dict[str, float | int]]]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ForgeSampleFigureError("selected product no longer parses for conformer generation")
    molecule = Chem.AddHs(molecule)
    parameters = rdDistGeom.ETKDGv3()
    setattr(parameters, "randomSeed", seed)
    setattr(parameters, "numThreads", 1)
    setattr(parameters, "pruneRmsThresh", -1.0)
    conformer_ids = [
        int(value)
        for value in rdDistGeom.EmbedMultipleConfs(molecule, numConfs=count, params=parameters)
    ]
    if len(conformer_ids) != count:
        raise ForgeSampleFigureError(
            f"ETKDG generated {len(conformer_ids)} of {count} requested conformers"
        )
    properties = rdForceFieldHelpers.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94s")
    if properties is not None:
        raw = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            molecule,
            numThreads=1,
            maxIters=max_iterations,
            mmffVariant="MMFF94s",
        )
        optimizer = "MMFF94s"
    else:
        raw = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(
            molecule,
            numThreads=1,
            maxIters=max_iterations,
        )
        optimizer = "UFF"
    optimization = [
        {"status": int(status), "energy_kcal_mol": float(energy)} for status, energy in raw
    ]
    if len(optimization) != count or any(
        not math.isfinite(float(row["energy_kcal_mol"])) for row in optimization
    ):
        raise ForgeSampleFigureError("conformer optimization result is incomplete")
    return molecule, conformer_ids, optimizer, optimization


def _embed_atlas_conformer(
    smiles: str,
    *,
    seed: int,
    max_iterations: int,
    maximum_embedding_iterations: int,
) -> tuple[Chem.Mol, int, str, dict[str, float | int], str]:
    try:
        molecule, conformer_ids, optimizer, optimizations = _embed_conformers(
            smiles,
            count=1,
            seed=seed,
            max_iterations=max_iterations,
        )
        return molecule, conformer_ids[0], optimizer, optimizations[0], "etkdgv3"
    except ForgeSampleFigureError as exc:
        if not str(exc).startswith("ETKDG generated 0 of 1 requested conformers"):
            raise

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ForgeSampleFigureError("selected atlas product no longer parses")
    molecule = Chem.AddHs(molecule)
    parameters = rdDistGeom.ETKDGv3()
    setattr(parameters, "randomSeed", seed)
    setattr(parameters, "numThreads", 1)
    setattr(parameters, "pruneRmsThresh", -1.0)
    setattr(parameters, "useRandomCoords", True)
    setattr(parameters, "maxIterations", maximum_embedding_iterations)
    conformer_id = int(rdDistGeom.EmbedMolecule(molecule, parameters))
    if conformer_id < 0:
        raise ForgeSampleFigureError("atlas ETKDG random-coordinate fallback failed")
    properties = rdForceFieldHelpers.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94s")
    if properties is not None:
        raw = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            molecule,
            numThreads=1,
            maxIters=max_iterations,
            mmffVariant="MMFF94s",
        )
        optimizer = "MMFF94s"
    else:
        raw = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(
            molecule,
            numThreads=1,
            maxIters=max_iterations,
        )
        optimizer = "UFF"
    if len(raw) != 1 or not math.isfinite(float(raw[0][1])):
        raise ForgeSampleFigureError("atlas fallback conformer optimization is incomplete")
    optimization_row = {"status": int(raw[0][0]), "energy_kcal_mol": float(raw[0][1])}
    return molecule, conformer_id, optimizer, optimization_row, "etkdgv3_random_coordinates"


def _rotation(angle_z: float, angle_x: float) -> np.ndarray[Any, np.dtype[np.float64]]:
    z = math.radians(angle_z)
    x = math.radians(angle_x)
    rotate_z = np.asarray(
        ((math.cos(z), -math.sin(z), 0.0), (math.sin(z), math.cos(z), 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    rotate_x = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, math.cos(x), -math.sin(x)), (0.0, math.sin(x), math.cos(x))),
        dtype=np.float64,
    )
    return rotate_z @ rotate_x


def _project_heavy_atoms(
    molecule: Chem.Mol,
    conformer_id: int,
    *,
    view_index: int,
) -> tuple[list[int], np.ndarray[Any, np.dtype[np.float64]]]:
    heavy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1]
    conformer = molecule.GetConformer(conformer_id)
    coordinates = np.asarray(
        [
            (
                conformer.GetAtomPosition(index).x,
                conformer.GetAtomPosition(index).y,
                conformer.GetAtomPosition(index).z,
            )
            for index in heavy
        ],
        dtype=np.float64,
    )
    coordinates -= coordinates.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(coordinates, full_matrices=False)
    oriented = coordinates @ axes.T
    if np.linalg.det(axes) < 0:
        oriented[:, 2] *= -1.0
    oriented = oriented @ _rotation((-18.0, 4.0, 24.0)[view_index], 16.0).T
    return heavy, oriented


def _projected_center(
    coordinates: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    planar = coordinates[:, :2]
    return (planar.min(axis=0) + planar.max(axis=0)) / 2.0


def _draw_conformer(
    canvas: Image.Image,
    molecule: Chem.Mol,
    conformer_id: int,
    box: tuple[int, int, int, int],
    *,
    view_index: int,
    label: str,
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    heavy, coordinates = _project_heavy_atoms(molecule, conformer_id, view_index=view_index)
    index_to_row = {atom_index: row for row, atom_index in enumerate(heavy)}
    padding_x = 28
    padding_y = 54
    width = box[2] - box[0] - 2 * padding_x
    height = box[3] - box[1] - 2 * padding_y
    extent = np.ptp(coordinates[:, :2], axis=0)
    extent = np.maximum(extent, 1.0)
    scale = min(width / float(extent[0]), height / float(extent[1]))
    # Centre on the projected bounding box, not the centroid: a lipid's mass sits in its tail, so
    # centroid centring pushes the head out of the padded box.
    center = _projected_center(coordinates)
    points = coordinates[:, :2] - center
    points[:, 0] = box[0] + (box[2] - box[0]) / 2.0 + scale * points[:, 0]
    points[:, 1] = box[1] + (box[3] - box[1]) / 2.0 - scale * points[:, 1]
    depths = coordinates[:, 2]

    bonds = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in index_to_row and end in index_to_row:
            bonds.append(
                (float(depths[index_to_row[begin]] + depths[index_to_row[end]]), begin, end)
            )
    for _, begin, end in sorted(bonds):
        begin_row = index_to_row[begin]
        end_row = index_to_row[end]
        first = tuple(float(value) for value in points[begin_row])
        second = tuple(float(value) for value in points[end_row])
        middle = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
        first_symbol = molecule.GetAtomWithIdx(begin).GetSymbol()
        second_symbol = molecule.GetAtomWithIdx(end).GetSymbol()
        draw.line((first, middle), fill=_ELEMENT_RGB.get(first_symbol, _MUTED), width=8)
        draw.line((middle, second), fill=_ELEMENT_RGB.get(second_symbol, _MUTED), width=8)
        draw.line((first, second), fill=(255, 255, 255, 90), width=2)

    order = sorted(range(len(heavy)), key=lambda row: float(depths[row]))
    depth_min = float(depths.min())
    depth_range = max(float(np.ptp(depths)), 1.0)
    for row in order:
        atom = molecule.GetAtomWithIdx(heavy[row])
        symbol = atom.GetSymbol()
        relative_depth = (float(depths[row]) - depth_min) / depth_range
        radius = int(10 + 5 * relative_depth + (3 if symbol != "C" else 0))
        x, y = (float(value) for value in points[row])
        draw.ellipse(
            (x - radius + 3, y - radius + 4, x + radius + 3, y + radius + 4),
            fill=(37, 39, 57, 35),
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=_ELEMENT_RGB.get(symbol, _MUTED),
            outline=(255, 255, 255, 210),
            width=2,
        )
        highlight = max(2, radius // 4)
        draw.ellipse(
            (
                x - radius // 2,
                y - radius // 2,
                x - radius // 2 + highlight,
                y - radius // 2 + highlight,
            ),
            fill=(255, 255, 255, 165),
        )
    _centered_text(
        draw,
        (box[0], box[3] - 42, box[2], box[3]),
        label,
        _font(24),
        fill=_MUTED,
    )


def _selection_rows(
    common_rows: Sequence[Mapping[str, Any]],
    realism_rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    realism_by_index: dict[int, Mapping[str, Any]] = {}
    for row in realism_rows:
        index = row.get("attempt_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in realism_by_index:
            raise ForgeSampleFigureError("realism attempt identities changed")
        realism_by_index[index] = row
    eligible_by_smiles: dict[str, dict[str, Any]] = {}
    for raw in common_rows:
        index = raw.get("attempt_index")
        if isinstance(index, bool) or not isinstance(index, int) or index not in realism_by_index:
            raise ForgeSampleFigureError("common and realism ledgers no longer align")
        realism = realism_by_index[index]
        smiles = raw.get("canonical_smiles")
        traces = raw.get("exact_l1_traces")
        if not (
            raw.get("method_id") == selection["method_id"]
            and raw.get("seed") == selection["seed"]
            and raw.get("program_id") == selection["program_id"]
            and raw.get("exact_l1_program") is True
            and raw.get("exact_l1_trace_count") == selection["required_exact_l1_trace_count"]
            and raw.get("method_visible_open_ended_exact_l1")
            is selection["require_method_visible_open_ended"]
            and realism.get("descriptor_manifold_member")
            is selection["require_descriptor_manifold_member"]
            and realism.get("within_declared_support") is selection["require_declared_support"]
            and isinstance(smiles, str)
            and smiles
            and isinstance(traces, list)
            and len(traces) == 1
            and isinstance(traces[0], Mapping)
        ):
            continue
        components = traces[0].get("components_by_role")
        if not isinstance(components, Mapping) or set(components) != set(_ROLE_ORDER):
            raise ForgeSampleFigureError("selected exact-L1 trace roles changed")
        candidate = {
            "attempt_index": index,
            "canonical_smiles": smiles,
            "rank_sha256": hashlib.sha256(smiles.encode("utf-8")).hexdigest(),
            "components_by_role": {role: str(components[role]) for role in _ROLE_ORDER},
            "nearest_reference_descriptor_distance": realism.get(
                "nearest_reference_descriptor_distance"
            ),
            "nearest_reference_tanimoto": realism.get("nearest_reference_tanimoto"),
        }
        existing = eligible_by_smiles.get(smiles)
        if existing is None or index < int(existing["attempt_index"]):
            eligible_by_smiles[smiles] = candidate
    ranked = sorted(
        eligible_by_smiles.values(),
        key=lambda row: (str(row["rank_sha256"]), int(row["attempt_index"])),
    )
    display_count = int(selection["display_count"])
    if selection["rank_by"] == "sha256_canonical_smiles_ascending":
        return ranked[:display_count], len(ranked)
    if selection["rank_by"] != "sha256_seeded_ecfp4_maxmin":
        raise ForgeSampleFigureError("FORGE sample-figure display ranking changed")
    radius = selection.get("fingerprint_radius")
    bits = selection.get("fingerprint_bits")
    if (
        isinstance(radius, bool)
        or not isinstance(radius, int)
        or radius < 1
        or isinstance(bits, bool)
        or not isinstance(bits, int)
        or bits < 128
    ):
        raise ForgeSampleFigureError("ECFP4 display-ranking parameters are invalid")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    fingerprints = []
    for row in ranked:
        molecule = Chem.MolFromSmiles(str(row["canonical_smiles"]))
        if molecule is None:
            raise ForgeSampleFigureError("eligible display molecule no longer parses")
        fingerprints.append(generator.GetFingerprint(molecule))
    if not ranked or display_count > len(ranked):
        raise ForgeSampleFigureError("display count exceeds the eligible population")
    selected_indices = [0]
    while len(selected_indices) < display_count:
        candidates: list[tuple[float, str, int]] = []
        for index, row in enumerate(ranked):
            if index in selected_indices:
                continue
            minimum_distance = min(
                1.0
                - float(
                    DataStructs.TanimotoSimilarity(
                        fingerprints[index], fingerprints[selected_index]
                    )
                )
                for selected_index in selected_indices
            )
            candidates.append((-minimum_distance, str(row["rank_sha256"]), index))
        candidates.sort()
        selected_indices.append(candidates[0][2])
    selected = [ranked[index] for index in selected_indices]
    for display_rank, row in enumerate(selected, start=1):
        row["display_diversity_rank"] = display_rank
    return selected, len(ranked)


def _load_display_population(
    config: Mapping[str, Any],
    repo: Path,
    *,
    extra_input_names: set[str],
    expected_rank: str,
    label: str,
) -> tuple[dict[str, Path], Mapping[str, Any], list[dict[str, Any]], int]:
    raw_inputs = config.get("inputs")
    expected_inputs = {
        "common_assessment_result",
        "common_assessed_attempts",
        "realism_assessment_result",
        "realism_assessed_attempts",
        *extra_input_names,
    }
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != expected_inputs:
        raise ForgeSampleFigureError(f"{label} input set changed")
    inputs = {
        name: resolve_pin(pin, repo, label=f"{label} {name}") for name, pin in raw_inputs.items()
    }
    common_result = read_json_object(
        inputs["common_assessment_result"],
        error=ForgeSampleFigureError,
        label=f"{label} common assessment result",
    )
    realism_result = read_json_object(
        inputs["realism_assessment_result"],
        error=ForgeSampleFigureError,
        label=f"{label} realism assessment result",
    )
    if (
        common_result.get("schema_version") != "forge.common_ugi_complete_assessment.v1"
        or common_result.get("status") != "pass"
        or common_result.get("candidate_selection") is not False
        or common_result.get("assessed_attempts", {}).get("sha256")
        != str(sha256_file(inputs["common_assessed_attempts"]))
        or realism_result.get("schema_version")
        != "forge.common_lipid_realism_complete_assessment.v1"
        or realism_result.get("status") != "pass"
        or realism_result.get("candidate_selection") is not False
        or realism_result.get("assessed_attempts", {}).get("sha256")
        != str(sha256_file(inputs["realism_assessed_attempts"]))
    ):
        raise ForgeSampleFigureError(f"{label} source assessments are inadmissible")
    selection = config.get("selection")
    expected_selection_fields = {
        "method_id",
        "seed",
        "program_id",
        "display_count",
        "required_exact_l1_trace_count",
        "require_method_visible_open_ended",
        "require_descriptor_manifold_member",
        "require_declared_support",
        "deduplicate_by",
        "rank_by",
        "expected_eligible_unique",
        "expected_selected_attempt_indices",
    }
    if expected_rank == "sha256_seeded_ecfp4_maxmin":
        expected_selection_fields.update({"fingerprint_radius", "fingerprint_bits"})
    if (
        not isinstance(selection, Mapping)
        or set(selection) != expected_selection_fields
        or selection.get("deduplicate_by") != "canonical_constitution"
        or selection.get("rank_by") != expected_rank
    ):
        raise ForgeSampleFigureError(f"{label} selection policy changed")
    common_rows = _read_ledger(inputs["common_assessed_attempts"], _COMMON_LEDGER_SCHEMA)
    realism_rows = _read_ledger(inputs["realism_assessed_attempts"], _REALISM_LEDGER_SCHEMA)
    selected, eligible_unique = _selection_rows(common_rows, realism_rows, selection)
    if (
        eligible_unique != selection["expected_eligible_unique"]
        or [row["attempt_index"] for row in selected]
        != selection["expected_selected_attempt_indices"]
        or len(selected) != selection["display_count"]
    ):
        raise ForgeSampleFigureError(f"{label} deterministic display population changed")
    return inputs, selection, selected, eligible_unique


def _render_figure(
    selected: Sequence[dict[str, Any]],
    conformer_config: Mapping[str, Any],
    png_path: Path,
) -> list[dict[str, Any]]:
    width, height = 3600, 2250
    canvas = Image.new("RGBA", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(canvas, "RGBA")
    title_font = _font(48, serif=True)
    row_font = _font(25, bold=True)
    small_font = _font(22)
    role_font = _font(19, bold=True)
    draw.rectangle((0, 0, width, 235), fill=(242, 242, 251, 255))
    _centered_text(draw, (100, 72, 990, 195), "Generated molecular graph", title_font)
    _centered_text(draw, (1035, 72, 2755, 195), "Post-hoc conformer ensemble", title_font)
    _centered_text(draw, (2800, 72, 3520, 195), "Exact L1 precursors", title_font)
    draw.line((1010, 245, 1010, 2120), fill=_DIVIDER, width=4)
    draw.line((2780, 245, 2780, 2120), fill=_DIVIDER, width=4)
    row_top = 280
    row_height = 570
    row_gap = 50
    render_rows: list[dict[str, Any]] = []
    for row_index, sample in enumerate(selected):
        top = row_top + row_index * (row_height + row_gap)
        bottom = top + row_height
        if row_index:
            draw.line((120, top - 26, 3480, top - 26), fill=(224, 225, 234, 255), width=2)
        label = f"FORGE-Ugi-{int(sample['attempt_index']):04d}"
        draw.text((135, top + 8), label, fill=_TEXT, font=row_font)
        draw.text(
            (135, top + 46),
            "seed 20260825 · exact-L1 · open-ended",
            fill=_MUTED,
            font=small_font,
        )
        product_image = _rdkit_2d(str(sample["canonical_smiles"]), 920, 450, font_size=28)
        _paste_contain(canvas, product_image, (120, top + 80, 980, bottom - 8), padding=12)

        conformer_seed = int(conformer_config["random_seed"]) + row_index * 9973
        molecule, conformer_ids, optimizer, optimization = _embed_conformers(
            str(sample["canonical_smiles"]),
            count=int(conformer_config["count_per_molecule"]),
            seed=conformer_seed,
            max_iterations=int(conformer_config["max_iterations"]),
        )
        energy_min = min(float(item["energy_kcal_mol"]) for item in optimization)
        panel_width = 540
        for conformer_index, conformer_id in enumerate(conformer_ids):
            left = 1055 + conformer_index * 565
            panel_box = (left, top + 54, left + panel_width, bottom - 22)
            draw.rounded_rectangle(
                panel_box,
                radius=24,
                fill=_PANEL,
                outline=(229, 230, 239, 255),
                width=2,
            )
            delta = float(optimization[conformer_index]["energy_kcal_mol"]) - energy_min
            _draw_conformer(
                canvas,
                molecule,
                conformer_id,
                panel_box,
                view_index=conformer_index,
                label=f"conf. {conformer_index + 1} · ΔE {delta:.1f} kcal mol⁻¹",
            )

        precursor_width = 222
        for role_index, role in enumerate(_ROLE_ORDER):
            left = 2822 + role_index * 228
            precursor_box = (left, top + 74, left + precursor_width, bottom - 20)
            draw.rounded_rectangle(
                precursor_box,
                radius=18,
                fill=_PANEL_TINT,
                outline=(226, 227, 237, 255),
                width=2,
            )
            _centered_text(
                draw,
                (left, top + 80, left + precursor_width, top + 122),
                _ROLE_NAMES[role],
                role_font,
                fill=_MUTED,
            )
            precursor_image = _rdkit_2d(
                str(sample["components_by_role"][role]), 400, 520, font_size=24
            )
            _paste_contain(
                canvas,
                precursor_image,
                (left + 6, top + 125, left + precursor_width - 6, bottom - 30),
                padding=4,
            )
        render_rows.append(
            {
                **sample,
                "conformer_seed": conformer_seed,
                "conformer_generator": "RDKit ETKDGv3",
                "optimizer_used": optimizer,
                "conformers": [
                    {
                        "conformer_id": conformer_id,
                        **optimization[index],
                    }
                    for index, conformer_id in enumerate(conformer_ids)
                ],
            }
        )
    draw.line((120, 2150, 3480, 2150), fill=_DIVIDER, width=2)
    _centered_text(
        draw,
        (150, 2160, 3450, 2230),
        "FORGE generates constitutional graphs. Conformers are deterministic RDKit ETKDGv3/MMFF94s views; hydrogens are omitted for clarity.",
        _font(24),
        fill=_MUTED,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(png_path, format="PNG", dpi=(300, 300), optimize=True)
    return render_rows


def _semantic_product_image(
    smiles: str,
    atom_origins: Sequence[str],
    core_atom_indices: Sequence[int],
    palette: Mapping[str, str],
    *,
    width: int,
    height: int,
) -> Image.Image:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() != len(atom_origins):
        raise ForgeSampleFigureError("semantic-map atom origins no longer align to the product")
    rdDepictor.Compute2DCoords(molecule)
    colors = {role: _hex_rgb(str(value)) for role, value in palette.items()}
    if set(colors) != {*_ROLE_ORDER, "assembly_introduced"}:
        raise ForgeSampleFigureError("semantic-map origin palette changed")
    core = set(core_atom_indices)
    if not core or any(index < 0 or index >= molecule.GetNumAtoms() for index in core):
        raise ForgeSampleFigureError("semantic-map reaction-core indices are invalid")
    atom_colors = {index: [(*colors[origin], 0.38)] for index, origin in enumerate(atom_origins)}
    atom_radii = {index: (0.46 if index in core else 0.31) for index in range(len(atom_origins))}
    bond_colors: dict[int, list[tuple[float, float, float, float]]] = {}
    for bond in molecule.GetBonds():
        begin_origin = atom_origins[bond.GetBeginAtomIdx()]
        end_origin = atom_origins[bond.GetEndAtomIdx()]
        origins = [begin_origin]
        if end_origin != begin_origin:
            origins.append(end_origin)
        bond_colors[bond.GetIdx()] = [(*colors[origin], 0.56) for origin in origins]
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    setattr(options, "padding", 0.07)
    setattr(options, "bondLineWidth", 3.4)
    setattr(options, "fixedFontSize", 30)
    setattr(options, "fillHighlights", True)
    setattr(options, "continuousHighlight", True)
    setattr(options, "atomHighlightsAreCircles", True)
    setattr(options, "highlightBondWidthMultiplier", 8)
    options.setAtomPalette(
        {
            6: (0.16, 0.18, 0.24),
            7: (0.12, 0.23, 0.82),
            8: (0.86, 0.22, 0.20),
            15: (0.91, 0.48, 0.12),
            16: (0.78, 0.61, 0.08),
            17: (0.14, 0.55, 0.26),
            35: (0.55, 0.24, 0.17),
        }
    )
    drawer.DrawMoleculeWithHighlights(
        molecule,
        "",
        atom_colors,
        bond_colors,
        atom_radii,
        {bond.GetIdx(): 1 for bond in molecule.GetBonds()},
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")


def _annotate_semantic_rows(
    selected: Sequence[dict[str, Any]],
    inputs: Mapping[str, Path],
    semantic_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reaction_id = semantic_config.get("reaction_id")
    maximum_outcomes = semantic_config.get("max_forward_outcomes_per_product")
    if (
        reaction_id != "ugi_3cr_agile"
        or isinstance(maximum_outcomes, bool)
        or not isinstance(maximum_outcomes, int)
        or maximum_outcomes < 1
    ):
        raise ForgeSampleFigureError("semantic-map reaction contract changed")
    compiled = load_qualified_forward_reaction(
        inputs["qualified_ugi_reactions"],
        inputs["ugi_variant"],
        reaction_id=reaction_id,
    )
    annotated: list[dict[str, Any]] = []
    for sample in selected:
        product_row, atom_rows, _, _ = annotate_qualified_ugi_product(
            compiled,
            product_id=f"FORGE-Ugi-{int(sample['attempt_index']):04d}",
            target_smiles=str(sample["canonical_smiles"]),
            component_smiles_by_role=sample["components_by_role"],
            source_evidence_record_id="display_exact_l1_trace",
            max_outcomes=maximum_outcomes,
        )
        if product_row["product_smiles"] != sample["canonical_smiles"]:
            raise ForgeSampleFigureError("semantic-map annotation changed product identity")
        atom_origins = [str(row["origin_role"]) for row in atom_rows]
        core_atom_indices = [
            int(row["product_atom_index"]) for row in atom_rows if row["is_ugi_core"]
        ]
        role_anchor_indices = json.loads(str(product_row["role_anchor_indices_json"]))
        if (
            set(atom_origins) != {*_ROLE_ORDER, "assembly_introduced"}
            or len(core_atom_indices) != 5
            or not isinstance(role_anchor_indices, dict)
            or set(role_anchor_indices) != set(_ROLE_ORDER)
        ):
            raise ForgeSampleFigureError("semantic-map annotation is incomplete")
        annotated.append(
            {
                **sample,
                "atom_origins": atom_origins,
                "reaction_core_atom_indices": core_atom_indices,
                "role_anchor_indices": {
                    role: int(role_anchor_indices[role]) for role in _ROLE_ORDER
                },
                "semantic_annotation_sha256": str(product_row["annotation_sha256"]),
            }
        )
    return annotated


def _draw_role_legend(
    draw: ImageDraw.ImageDraw,
    palette: Mapping[str, str],
    *,
    top: int,
) -> None:
    font = _font(21, bold=True)
    entries = (
        ("amine_head", "amine head"),
        ("oxoester_aldehyde_body_tail", "aldehyde body + tail"),
        ("isocyanide_tail", "isocyanide tail"),
        ("assembly_introduced", "assembly-added O"),
    )
    widths = (220, 330, 250, 260)
    left = 1180
    for (role, label), width in zip(entries, widths, strict=True):
        color = tuple(round(channel * 255) for channel in _hex_rgb(str(palette[role])))
        draw.rounded_rectangle((left, top, left + 28, top + 28), radius=7, fill=(*color, 255))
        draw.text((left + 40, top - 1), label, font=font, fill=_MUTED)
        left += width


def _draw_precursor_card(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    role: str,
    smiles: str,
    palette: Mapping[str, str],
    wide: bool,
) -> None:
    color = tuple(round(channel * 255) for channel in _hex_rgb(str(palette[role])))
    draw.line((box[0], box[1], box[2], box[1]), fill=(*color, 255), width=5)
    label = _ROLE_NAMES[role]
    draw.text((box[0] + 18, box[1] + 20), label, font=_font(22, bold=True), fill=_TEXT)
    image = _rdkit_2d(smiles, 840 if wide else 420, 260, font_size=25)
    _paste_contain(canvas, image, (box[0] + 8, box[1] + 52, box[2] - 8, box[3] - 8), padding=3)


def _render_semantic_figure(
    selected: Sequence[dict[str, Any]],
    semantic_config: Mapping[str, Any],
    png_path: Path,
) -> None:
    width, height = 3600, 2250
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    palette = semantic_config["origin_palette"]
    title_font = _font(47, serif=True)
    subtitle_font = _font(22)
    row_title_font = _font(28, bold=True)
    _centered_text(draw, (110, 50, 980, 128), "Generated lipid", title_font)
    _centered_text(draw, (1015, 50, 2510, 128), "Reaction-program coordinates", title_font)
    _centered_text(draw, (2545, 50, 3500, 128), "Exact L1 building blocks", title_font)
    _centered_text(
        draw,
        (110, 130, 980, 184),
        "one whole constitutional graph",
        subtitle_font,
        fill=_MUTED,
    )
    _centered_text(
        draw,
        (2545, 130, 3500, 184),
        "amine + aldehyde + isocyanide",
        subtitle_font,
        fill=_MUTED,
    )
    _draw_role_legend(draw, palette, top=166)
    draw.line((78, 236, 3522, 236), fill=_DIVIDER, width=2)
    draw.line((1000, 236, 1000, 2155), fill=_DIVIDER, width=2)
    draw.line((2530, 236, 2530, 2155), fill=_DIVIDER, width=2)

    row_top = 270
    row_height = 575
    row_gap = 46
    for row_index, sample in enumerate(selected):
        top = row_top + row_index * (row_height + row_gap)
        bottom = top + row_height
        if row_index:
            draw.line((78, top - row_gap // 2, 3522, top - row_gap // 2), fill=_DIVIDER, width=2)
        sample_number = row_index + 1
        draw.text(
            (112, top + 24),
            f"SAMPLE {sample_number:02d}",
            font=_font(19, bold=True),
            fill=_MUTED,
        )
        draw.text(
            (112, top + 60),
            f"FORGE-Ugi-{int(sample['attempt_index']):04d}",
            font=row_title_font,
            fill=_TEXT,
        )
        draw.text(
            (112, top + 102),
            "open-ended · exact L1 · within declared support",
            font=subtitle_font,
            fill=_MUTED,
        )
        product = _rdkit_2d(str(sample["canonical_smiles"]), 860, 420, font_size=30)
        _paste_contain(canvas, product, (105, top + 138, 970, bottom - 18), padding=10)

        semantic = _semantic_product_image(
            str(sample["canonical_smiles"]),
            sample["atom_origins"],
            sample["reaction_core_atom_indices"],
            palette,
            width=1380,
            height=430,
        )
        _paste_contain(canvas, semantic, (1060, top + 40, 2470, bottom - 72), padding=8)
        _centered_text(
            draw,
            (1674, bottom - 80, 2135, bottom - 38),
            "larger halos mark the five-atom Ugi core",
            _font(19),
            fill=_MUTED,
        )

        precursor_left = 2564
        precursor_right = 3488
        half_gap = 18
        half_width = (precursor_right - precursor_left - half_gap) // 2
        _draw_precursor_card(
            canvas,
            draw,
            box=(precursor_left, top + 24, precursor_left + half_width, top + 266),
            role="amine_head",
            smiles=str(sample["components_by_role"]["amine_head"]),
            palette=palette,
            wide=False,
        )
        _draw_precursor_card(
            canvas,
            draw,
            box=(
                precursor_left + half_width + half_gap,
                top + 24,
                precursor_right,
                top + 266,
            ),
            role="isocyanide_tail",
            smiles=str(sample["components_by_role"]["isocyanide_tail"]),
            palette=palette,
            wide=False,
        )
        _draw_precursor_card(
            canvas,
            draw,
            box=(precursor_left, top + 286, precursor_right, bottom - 24),
            role="oxoester_aldehyde_body_tail",
            smiles=str(sample["components_by_role"]["oxoester_aldehyde_body_tail"]),
            palette=palette,
            wide=True,
        )

    draw.line((110, 2168, 3490, 2168), fill=_DIVIDER, width=2)
    _centered_text(
        draw,
        (140, 2172, 3460, 2234),
        "Colors are recovered by exact atom-mapped forward replay; they are semantic coordinates, not a finite component vocabulary.",
        _font(23, bold=True),
        fill=_MUTED,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(png_path, format="PNG", dpi=(300, 300), optimize=True)


def _draw_origin_conformer(
    canvas: Image.Image,
    molecule: Chem.Mol,
    conformer_id: int,
    box: tuple[int, int, int, int],
    *,
    atom_origins: Sequence[str],
    palette: Mapping[str, str],
    view_index: int,
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    heavy, coordinates = _project_heavy_atoms(molecule, conformer_id, view_index=view_index)
    if heavy != list(range(len(atom_origins))):
        raise ForgeSampleFigureError("atlas conformer atom order changed")
    role_colors = {
        role: tuple(round(channel * 255) for channel in _hex_rgb(str(value)))
        for role, value in palette.items()
    }
    index_to_row = {atom_index: row for row, atom_index in enumerate(heavy)}
    padding_x = 34
    padding_y = 46
    width = box[2] - box[0] - 2 * padding_x
    height = box[3] - box[1] - 2 * padding_y
    extent = np.maximum(np.ptp(coordinates[:, :2], axis=0), 1.0)
    scale = min(width / float(extent[0]), height / float(extent[1]))
    # Centre on the projected bounding box, not the centroid: a lipid's mass sits in its tail, so
    # centroid centring pushes the head out of the padded box.
    center = _projected_center(coordinates)
    points = coordinates[:, :2] - center
    points[:, 0] = box[0] + (box[2] - box[0]) / 2.0 + scale * points[:, 0]
    points[:, 1] = box[1] + (box[3] - box[1]) / 2.0 - scale * points[:, 1]
    depths = coordinates[:, 2]
    # Bond stroke and atom radius are expressed against the length of one drawn C--C bond so the
    # drawing keeps a constant visual weight whatever canvas aspect the caller supplies.
    bond_pixels = _TYPICAL_BOND_ANGSTROM * scale
    bond_width = max(4, round(_CONFORMER_BOND_STROKE * bond_pixels))

    bonds: list[tuple[float, int, int]] = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in index_to_row and end in index_to_row:
            bonds.append(
                (float(depths[index_to_row[begin]] + depths[index_to_row[end]]), begin, end)
            )
    for _, begin, end in sorted(bonds):
        first = tuple(float(value) for value in points[index_to_row[begin]])
        second = tuple(float(value) for value in points[index_to_row[end]])
        middle = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
        begin_color = role_colors[atom_origins[begin]]
        end_color = role_colors[atom_origins[end]]
        draw.line((first, middle), fill=(*begin_color, 255), width=bond_width)
        draw.line((middle, second), fill=(*end_color, 255), width=bond_width)
        draw.line((first, second), fill=(255, 255, 255, 105), width=max(2, bond_width // 5))

    depth_min = float(depths.min())
    depth_range = max(float(np.ptp(depths)), 1.0)
    for row in sorted(range(len(heavy)), key=lambda item: float(depths[item])):
        atom_index = heavy[row]
        atom = molecule.GetAtomWithIdx(atom_index)
        symbol = atom.GetSymbol()
        relative_depth = (float(depths[row]) - depth_min) / depth_range
        radius = max(
            5,
            round(
                bond_pixels
                * (
                    _CONFORMER_ATOM_RADIUS
                    + _CONFORMER_ATOM_DEPTH_GAIN * relative_depth
                    + (_CONFORMER_HETEROATOM_GAIN if symbol != "C" else 0.0)
                )
            ),
        )
        x, y = (float(value) for value in points[row])
        role_color = role_colors[atom_origins[atom_index]]
        element_color = _ELEMENT_RGB.get(symbol, (*role_color, 255))
        fill = (*role_color, 255) if symbol == "C" else element_color
        draw.ellipse(
            (x - radius + 3, y - radius + 4, x + radius + 3, y + radius + 4),
            fill=(37, 39, 57, 34),
        )
        draw.ellipse(
            (x - radius - 3, y - radius - 3, x + radius + 3, y + radius + 3),
            fill=(*role_color, 255),
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
            outline=(255, 255, 255, 215),
            width=2,
        )
        highlight = max(2, radius // 4)
        draw.ellipse(
            (
                x - radius // 2,
                y - radius // 2,
                x - radius // 2 + highlight,
                y - radius // 2 + highlight,
            ),
            fill=(255, 255, 255, 175),
        )


def _trim_to_ink(image: Image.Image, *, margin: float) -> Image.Image:
    """Crop the white letterboxing a fixed-aspect canvas leaves around a drawing.

    Each atlas panel is scaled to its column width in LaTeX, so any margin baked into the canvas
    becomes dead space in the appendix row rather than air around the structure.
    """

    rendered = image.convert("RGB")
    ink = ImageChops.difference(rendered, Image.new("RGB", rendered.size, (255, 255, 255)))
    box = ink.getbbox()
    if box is None:
        return image
    pad_x = round(margin * (box[2] - box[0]))
    pad_y = round(margin * (box[3] - box[1]))
    return image.crop(
        (
            max(0, box[0] - pad_x),
            max(0, box[1] - pad_y),
            min(image.width, box[2] + pad_x),
            min(image.height, box[3] + pad_y),
        )
    )


def _write_png(path: Path, image: Image.Image) -> None:
    payload = io.BytesIO()
    image.convert("RGB").save(payload, format="PNG", dpi=(300, 300), optimize=True)
    atomic_write(path, payload.getvalue())


def _latex_smiles(smiles: str, *, chunk_width: int = 28) -> str:
    if not smiles or chunk_width < 1:
        raise ForgeSampleFigureError("canonical SMILES cannot be safely typeset")
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "#": r"\#",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    chunks = [smiles[index : index + chunk_width] for index in range(0, len(smiles), chunk_width)]
    escaped = [
        "".join(replacements.get(character, character) for character in chunk) for chunk in chunks
    ]
    return r"\allowbreak{}".join(escaped)


def _render_atlas_structure_assets(
    sample: Mapping[str, Any],
    *,
    palette: Mapping[str, str],
    output_dir: Path,
) -> tuple[Path, Path]:
    display_rank = int(sample["display_diversity_rank"])
    stem = f"forge_generated_atlas_row_{display_rank:02d}"
    graph_path = output_dir / f"{stem}_2d.png"
    conformer_path = output_dir / f"{stem}_3d.png"

    graph = _rdkit_2d(
        str(sample["canonical_smiles"]),
        width=_ATLAS_GRAPH_CANVAS[0],
        height=_ATLAS_GRAPH_CANVAS[1],
        font_size=34,
    )
    _write_png(graph_path, _trim_to_ink(graph, margin=_ATLAS_TRIM_MARGIN))

    view_width, view_height = _ATLAS_VIEW_CANVAS
    conformer = Image.new("RGBA", (view_width, view_height), (255, 255, 255, 255))
    _draw_origin_conformer(
        conformer,
        sample["conformer_molecule"],
        int(sample["conformer_id"]),
        (16, 16, view_width - 16, view_height - 16),
        atom_origins=sample["atom_origins"],
        palette=palette,
        view_index=(display_rank - 1) % 3,
    )
    _write_png(conformer_path, _trim_to_ink(conformer, margin=_ATLAS_TRIM_MARGIN))
    return graph_path, conformer_path


def _atlas_latex_row(
    sample: Mapping[str, Any],
    *,
    graph_path: Path,
    conformer_path: Path,
    asset_prefix: str,
) -> str:
    """Emit one atlas row.

    Every length, rule and type style is a macro the manuscript preamble owns, so the appendix
    layout can be retuned without reissuing this hash-pinned artifact.
    """

    display_rank = int(sample["display_diversity_rank"])
    smiles = _latex_smiles(str(sample["canonical_smiles"]))
    graph_asset = f"{asset_prefix.rstrip('/')}/{graph_path.name}"
    conformer_asset = f"{asset_prefix.rstrip('/')}/{conformer_path.name}"
    return "\n".join(
        [
            rf"\atlasrank{{{display_rank}}}",
            r"&",
            r"\begin{minipage}[c][\atlasrowheight][c]{\atlasgraphwidth}",
            r"\centering",
            rf"\includegraphics[width=\linewidth,height=\atlasgraphheight,"
            rf"keepaspectratio]{{{graph_asset}}}",
            r"\par\vspace{\atlassmilesskip}",
            rf"{{\atlassmilesstyle {smiles}\par}}",
            r"\end{minipage}",
            r"&",
            r"\begin{minipage}[c][\atlasrowheight][c]{\atlasviewwidth}",
            r"\centering",
            rf"\includegraphics[width=\linewidth,height=\atlasviewheight,"
            rf"keepaspectratio]{{{conformer_asset}}}",
            r"\end{minipage}",
        ]
    )


def render_forge_generated_sample_atlas(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Render individual atlas structures and paper-native LaTeX table rows."""

    config = read_json_object(
        config_path,
        error=ForgeSampleFigureError,
        label="FORGE generated-sample atlas config",
    )
    if (
        config.get("schema_version") != ATLAS_CONFIG_SCHEMA
        or set(config)
        != {
            "schema_version",
            "status",
            "inputs",
            "selection",
            "semantic_map",
            "conformer",
            "layout",
            "display_only",
            "candidate_selection",
        }
        or config.get("status") != "frozen_display_only"
        or config.get("display_only") is not True
        or config.get("candidate_selection") is not False
    ):
        raise ForgeSampleFigureError("FORGE sample-atlas config changed")
    inputs, selection, selected, eligible_unique = _load_display_population(
        config,
        repo,
        extra_input_names={"qualified_ugi_reactions", "ugi_variant"},
        expected_rank="sha256_seeded_ecfp4_maxmin",
        label="FORGE sample atlas",
    )
    semantic_config = config.get("semantic_map")
    if (
        not isinstance(semantic_config, Mapping)
        or set(semantic_config)
        != {
            "reaction_id",
            "max_forward_outcomes_per_product",
            "origin_palette",
            "highlight_reaction_core",
        }
        or semantic_config.get("reaction_id") != "ugi_3cr_agile"
        or semantic_config.get("highlight_reaction_core") is not True
        or not isinstance(semantic_config.get("origin_palette"), Mapping)
    ):
        raise ForgeSampleFigureError("FORGE sample-atlas semantic-map policy changed")
    conformer_config = config.get("conformer")
    if (
        not isinstance(conformer_config, Mapping)
        or set(conformer_config)
        != {
            "generator",
            "random_seed",
            "optimizer",
            "max_iterations",
            "embedding_fallback",
            "maximum_embedding_iterations",
            "render_hydrogens",
            "views_per_molecule",
            "color_mode",
            "view_rotation_policy",
        }
        or conformer_config.get("generator") != "rdkit_etkdgv3"
        or conformer_config.get("optimizer") != "mmff94s_with_uff_fallback"
        or conformer_config.get("embedding_fallback") != "etkdgv3_random_coordinates"
        or conformer_config.get("render_hydrogens") is not False
        or conformer_config.get("views_per_molecule") != 1
        or conformer_config.get("color_mode") != "origin_bonds_with_element_atoms"
        or conformer_config.get("view_rotation_policy") != "display_rank_modulo_three"
    ):
        raise ForgeSampleFigureError("FORGE sample-atlas conformer policy changed")
    layout = config.get("layout")
    if (
        not isinstance(layout, Mapping)
        or set(layout) != {"include_canonical_smiles", "columns", "latex_asset_prefix"}
        or layout.get("include_canonical_smiles") is not True
        or layout.get("columns")
        != [
            "display_diversity_rank",
            "constitutional_graph_2d_with_canonical_smiles",
            "decorated_3d_view",
        ]
        or layout.get("latex_asset_prefix") != "figures/forge_generated_sample_atlas_v1"
    ):
        raise ForgeSampleFigureError("FORGE sample-atlas layout changed")

    annotated = _annotate_semantic_rows(selected, inputs, semantic_config)
    rendered_rows: list[dict[str, Any]] = []
    base_seed = conformer_config.get("random_seed")
    max_iterations = conformer_config.get("max_iterations")
    maximum_embedding_iterations = conformer_config.get("maximum_embedding_iterations")
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
        or isinstance(maximum_embedding_iterations, bool)
        or not isinstance(maximum_embedding_iterations, int)
        or maximum_embedding_iterations < 1
    ):
        raise ForgeSampleFigureError("FORGE sample-atlas conformer parameters are invalid")
    for row_index, sample in enumerate(annotated):
        conformer_seed = base_seed + row_index * 9973
        molecule, conformer_id, optimizer, optimization, embedding_mode = _embed_atlas_conformer(
            str(sample["canonical_smiles"]),
            seed=conformer_seed,
            max_iterations=max_iterations,
            maximum_embedding_iterations=maximum_embedding_iterations,
        )
        rendered_rows.append(
            {
                **sample,
                "conformer_molecule": molecule,
                "conformer_id": conformer_id,
                "conformer_seed": conformer_seed,
                "embedding_mode": embedding_mode,
                "optimizer_used": optimizer,
                "optimization": optimization,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_paths: list[Path] = []
    conformer_paths: list[Path] = []
    latex_rows: list[str] = []
    asset_prefix = str(layout["latex_asset_prefix"])
    for sample in rendered_rows:
        graph_path, conformer_path = _render_atlas_structure_assets(
            sample,
            palette=semantic_config["origin_palette"],
            output_dir=output_dir,
        )
        graph_paths.append(graph_path)
        conformer_paths.append(conformer_path)
        latex_rows.append(
            _atlas_latex_row(
                sample,
                graph_path=graph_path,
                conformer_path=conformer_path,
                asset_prefix=asset_prefix,
            )
        )
    latex_rows_path = output_dir / "atlas_rows.tex"
    # Rows abut, and their separation comes from the struck row height exceeding the panel heights.
    # Inter-row glue would break the column divider into one segment per row, and a row terminator's
    # optional argument only raises that row's depth, which these full-height cells already exceed.
    body = "\\\\\n".join(latex_rows) + "\\\\\n"
    atomic_write(latex_rows_path, body.encode("utf-8"))

    receipt_rows = []
    for row in rendered_rows:
        receipt_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"conformer_molecule", "conformer_id"}
            }
        )
    receipt_path = output_dir / "atlas_samples.json"
    write_json(
        receipt_path,
        {
            "schema_version": "forge.paper.forge_generated_sample_atlas_selection.v1",
            "eligible_unique": eligible_unique,
            "selection_policy": dict(selection),
            "selected": receipt_rows,
            "display_only": True,
            "candidate_selection": False,
        },
    )
    result = {
        "schema_version": ATLAS_RESULT_SCHEMA,
        "status": "complete",
        "config": pin_record(config_path, repo),
        "inputs": {name: pin_record(path, repo) for name, path in sorted(inputs.items())},
        "eligible_unique": eligible_unique,
        "selected_attempt_indices": [row["attempt_index"] for row in rendered_rows],
        "row_count": len(rendered_rows),
        "columns": list(layout["columns"]),
        "latex_asset_prefix": asset_prefix,
        "software": {"rdkit": rdBase.rdkitVersion, "numpy": np.__version__},
        "artifacts": {
            "structure_images_2d": [artifact_record(path) for path in graph_paths],
            "structure_images_3d": [artifact_record(path) for path in conformer_paths],
            "latex_rows": artifact_record(latex_rows_path),
            "atlas_samples": artifact_record(receipt_path),
        },
        "nonclaims": [
            "The atlas is a deterministic display set, not an experimental candidate panel.",
            "FORGE generated constitutional graphs; RDKit generated the displayed 3D coordinates post hoc.",
            "Exact L1 replay and descriptor-manifold membership are structural diagnostics, not activity or synthesis success.",
        ],
        "display_only": True,
        "candidate_selection": False,
    }
    write_json(result_path, result)
    return result


def render_forge_generated_sample_figure(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Render the frozen display-only FORGE sample figure and its provenance receipt."""

    config = read_json_object(
        config_path,
        error=ForgeSampleFigureError,
        label="FORGE generated-sample figure config",
    )
    schema = config.get("schema_version")
    if schema == CONFIG_SCHEMA_V1:
        figure_mode = "conformer_v1"
        mode_field = "conformers"
    elif schema == CONFIG_SCHEMA_V2:
        figure_mode = "semantic_map_v2"
        mode_field = "semantic_map"
    else:
        raise ForgeSampleFigureError("FORGE sample-figure config schema changed")
    expected_fields = {
        "schema_version",
        "status",
        "inputs",
        "selection",
        mode_field,
        "display_only",
        "candidate_selection",
    }
    if (
        set(config) != expected_fields
        or config.get("status") != "frozen_display_only"
        or config.get("display_only") is not True
        or config.get("candidate_selection") is not False
    ):
        raise ForgeSampleFigureError("FORGE sample-figure config changed")
    expected_rank = "sha256_canonical_smiles_ascending"
    extra_inputs: set[str] = set()
    if figure_mode == "semantic_map_v2":
        extra_inputs.update({"qualified_ugi_reactions", "ugi_variant"})
        expected_rank = "sha256_seeded_ecfp4_maxmin"
    inputs, selection, selected, eligible_unique = _load_display_population(
        config,
        repo,
        extra_input_names=extra_inputs,
        expected_rank=expected_rank,
        label="FORGE sample figure",
    )
    mode_config = config.get(mode_field)
    if figure_mode == "conformer_v1":
        expected_mode_fields = {
            "generator",
            "count_per_molecule",
            "random_seed",
            "optimizer",
            "max_iterations",
            "render_hydrogens",
        }
        if (
            not isinstance(mode_config, Mapping)
            or set(mode_config) != expected_mode_fields
            or mode_config.get("generator") != "rdkit_etkdgv3"
            or mode_config.get("optimizer") != "mmff94s_with_uff_fallback"
            or mode_config.get("render_hydrogens") is not False
        ):
            raise ForgeSampleFigureError("FORGE sample-figure conformer policy changed")
    else:
        expected_mode_fields = {
            "reaction_id",
            "max_forward_outcomes_per_product",
            "origin_palette",
            "highlight_reaction_core",
        }
        if (
            not isinstance(mode_config, Mapping)
            or set(mode_config) != expected_mode_fields
            or mode_config.get("reaction_id") != "ugi_3cr_agile"
            or mode_config.get("highlight_reaction_core") is not True
            or not isinstance(mode_config.get("origin_palette"), Mapping)
        ):
            raise ForgeSampleFigureError("FORGE sample-figure semantic-map policy changed")
    png_path = output_dir / "forge_generated_samples.png"
    if figure_mode == "conformer_v1":
        rendered_rows = _render_figure(selected, mode_config, png_path)
    else:
        rendered_rows = _annotate_semantic_rows(selected, inputs, mode_config)
        _render_semantic_figure(rendered_rows, mode_config, png_path)
    selected_path = output_dir / "selected_samples.json"
    selection_receipt = {
        "schema_version": "forge.paper.forge_generated_sample_selection.v1",
        "eligible_unique": eligible_unique,
        "selection_policy": dict(selection),
        "selected": rendered_rows,
        "display_only": True,
        "candidate_selection": False,
    }
    write_json(selected_path, selection_receipt)
    result = {
        "schema_version": RESULT_SCHEMA_V1 if figure_mode == "conformer_v1" else RESULT_SCHEMA_V2,
        "status": "complete",
        "config": pin_record(config_path, repo),
        "inputs": {name: pin_record(path, repo) for name, path in sorted(inputs.items())},
        "eligible_unique": eligible_unique,
        "selected_attempt_indices": [row["attempt_index"] for row in rendered_rows],
        "software": {"rdkit": rdBase.rdkitVersion, "numpy": np.__version__},
        "artifacts": {
            "figure_png": artifact_record(png_path),
            "selected_samples": artifact_record(selected_path),
        },
        "nonclaims": (
            [
                "The displayed structures are deterministic examples, not selected experimental candidates.",
                "FORGE generated constitutional graphs; RDKit generated the displayed conformers post hoc.",
                "Descriptor-manifold membership is a structural diagnostic, not activity or synthesis success.",
            ]
            if figure_mode == "conformer_v1"
            else [
                "The displayed structures are deterministic examples, not selected experimental candidates.",
                "Precursor-origin colors are recovered by exact atom-mapped forward replay; they are not separately generated molecular regions.",
                "Exact L1 replay is transform consistency, not evidence of activity or synthesis success.",
            ]
        ),
        "display_only": True,
        "candidate_selection": False,
    }
    if figure_mode == "semantic_map_v2":
        result["figure_mode"] = figure_mode
    write_json(result_path, result)
    return result


__all__ = [
    "ForgeSampleFigureError",
    "render_forge_generated_sample_atlas",
    "render_forge_generated_sample_figure",
]

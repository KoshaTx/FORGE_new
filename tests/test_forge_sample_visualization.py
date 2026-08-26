from __future__ import annotations

import json
from pathlib import Path

from forge_paper.sample_visualization import (
    _latex_smiles,
    render_forge_generated_sample_atlas,
    render_forge_generated_sample_figure,
)
from PIL import Image, ImageChops


def test_latex_smiles_escapes_tex_metacharacters_and_retains_breaks() -> None:
    rendered = _latex_smiles(r"C#C%10\\C_C&C", chunk_width=4)
    assert r"\#" in rendered
    assert r"\%" in rendered
    assert r"\textbackslash{}" in rendered
    assert r"\_" in rendered
    assert r"\&" in rendered
    assert r"\allowbreak{}" in rendered


def test_forge_sample_figure_is_deterministic_display_not_candidate_selection(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = render_forge_generated_sample_figure(
        repo / "configs/reproduction/natbiotech_v1_forge_sample_figure_v1.json",
        repo,
        tmp_path / "figure",
        result_path=tmp_path / "result.json",
    )
    assert result["status"] == "complete"
    assert result["display_only"] is True
    assert result["candidate_selection"] is False
    assert result["eligible_unique"] == 23
    assert result["selected_attempt_indices"] == [2256, 2931, 1586]
    with Image.open(tmp_path / "figure/forge_generated_samples.png") as image:
        assert image.size == (3600, 2250)


def test_semantic_sample_figure_maps_exact_origins_without_conformers(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = render_forge_generated_sample_figure(
        repo / "configs/reproduction/natbiotech_v1_forge_sample_figure_v2.json",
        repo,
        tmp_path / "figure",
        result_path=tmp_path / "result.json",
    )
    assert result["schema_version"] == "forge.paper.forge_generated_sample_figure.v2"
    assert result["figure_mode"] == "semantic_map_v2"
    assert result["display_only"] is True
    assert result["candidate_selection"] is False
    assert result["eligible_unique"] == 23
    assert result["selected_attempt_indices"] == [2256, 1194, 2699]
    receipt = (tmp_path / "figure/selected_samples.json").read_text()
    assert "atom_origins" in receipt
    assert "conformer" not in receipt
    with Image.open(tmp_path / "figure/forge_generated_samples.png") as image:
        assert image.size == (3600, 2250)


def test_generated_sample_atlas_emits_structure_assets_and_latex_rows(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = render_forge_generated_sample_atlas(
        repo / "configs/reproduction/natbiotech_v1_forge_sample_atlas_v1.json",
        repo,
        tmp_path / "atlas",
        result_path=tmp_path / "result.json",
    )
    assert result["schema_version"] == "forge.paper.forge_generated_sample_atlas.v2"
    assert result["row_count"] == 12
    assert result["columns"] == [
        "display_diversity_rank",
        "constitutional_graph_2d_with_canonical_smiles",
        "decorated_3d_view",
    ]
    assert result["display_only"] is True
    assert result["candidate_selection"] is False
    assert result["selected_attempt_indices"] == [
        2256,
        1194,
        2699,
        2919,
        329,
        1776,
        1861,
        382,
        2078,
        2546,
        1152,
        1059,
    ]
    receipt = json.loads((tmp_path / "atlas/atlas_samples.json").read_text())
    assert len(receipt["selected"]) == 12
    assert all(row["canonical_smiles"] for row in receipt["selected"])
    assert all(len(row["atom_origins"]) > 0 for row in receipt["selected"])
    assert all("conformer_seed" in row for row in receipt["selected"])
    latex_rows = (tmp_path / "atlas/atlas_rows.tex").read_text()
    assert latex_rows.count("_2d.png") == 12
    assert latex_rows.count("_3d.png") == 12
    assert latex_rows.count(r"\atlasrank{") == 12
    assert latex_rows.count("canonical SMILES") == 0
    assert "amine head" not in latex_rows
    # Layout belongs to the manuscript preamble: the artifact names lengths and styles rather than
    # carrying its own rules, page breaks or hard-coded dimensions.
    for literal in (r"\hline", r"\pagebreak", "cm]", "textwidth}"):
        assert literal not in latex_rows
    for macro in (
        r"\atlasrowheight",
        r"\atlasgraphwidth",
        r"\atlasviewwidth",
        r"\atlassmilesstyle",
    ):
        assert macro in latex_rows
    for row in range(1, 13):
        for suffix, canvas in (("2d", (1800, 680)), ("3d", (1360, 900))):
            path = tmp_path / f"atlas/forge_generated_atlas_row_{row:02d}_{suffix}.png"
            with Image.open(path) as image:
                assert image.width <= canvas[0] and image.height <= canvas[1]
                rendered = image.convert("RGB")
                white = Image.new("RGB", rendered.size, (255, 255, 255))
                ink = ImageChops.difference(rendered, white).getbbox()
            # Trimmed to the drawing: what remains is the intended thin margin, not the
            # letterboxing a fixed-aspect canvas leaves around a molecule.
            assert ink is not None
            assert (ink[2] - ink[0]) >= 0.9 * image.width
            assert (ink[3] - ink[1]) >= 0.9 * image.height

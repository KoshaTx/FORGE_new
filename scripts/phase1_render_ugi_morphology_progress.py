#!/usr/bin/env python3
"""Render matched Ugi morphology samples from saved training evaluations."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forge.bio.ugi_semantic_annotations import ROLE_NAMES
from forge.product.ugi_morphology_program import (
    UgiMorphologyProgram,
    preorder_attached_forest_to_parents,
)

REPO = Path(__file__).resolve().parents[1]
ROLE_COLORS = ("#7c3aed", "#d97706", "#0891b2")
ROLE_DIRECTIONS = ((-1.0, 0.0), (0.68, 0.73), (0.68, -0.73))


def _program(row: dict[str, object]) -> UgiMorphologyProgram:
    return UgiMorphologyProgram(
        node_counts=tuple(int(value) for value in row["node_counts"]),
        junction_budgets=tuple(int(value) for value in row["junction_budgets"]),
        cycle_ranks=tuple(int(value) for value in row["cycle_ranks"]),
        attachment_counts=tuple(int(value) for value in row["attachment_counts"]),
    )


def _tree_coordinates(
    offspring: np.ndarray,
    *,
    attachment_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parents = preorder_attached_forest_to_parents(
        offspring,
        attachment_count=attachment_count,
    )
    children: list[list[int]] = [[] for _ in range(offspring.size)]
    depth = np.ones(offspring.size, dtype=np.float64)
    for node, parent in enumerate(parents):
        if parent >= 0:
            children[int(parent)].append(node)
            depth[node] = depth[int(parent)] + 1.0
    transverse = np.zeros(offspring.size, dtype=np.float64)
    next_leaf = 0

    def assign(node: int) -> float:
        nonlocal next_leaf
        if not children[node]:
            value = float(next_leaf)
            next_leaf += 1
        else:
            values = [assign(child) for child in children[node]]
            value = float(np.mean(values))
        transverse[node] = value
        return value

    roots = np.flatnonzero(parents < 0)
    for root in roots:
        assign(int(root))
    transverse -= float(np.mean(transverse[roots]))
    return parents, depth, transverse


def _draw_sample(
    axis: object,
    program: UgiMorphologyProgram,
    offspring_by_role: list[list[int]],
    *,
    title: str,
) -> None:
    axis.set_aspect("equal")
    axis.axis("off")
    core_angles = np.linspace(0.0, 2.0 * np.pi, 6)[:-1] + np.pi / 2.0
    core_x = 0.18 * np.cos(core_angles)
    core_y = 0.18 * np.sin(core_angles)
    axis.plot(
        np.append(core_x, core_x[0]),
        np.append(core_y, core_y[0]),
        color="#334155",
        linewidth=1.8,
        zorder=1,
    )
    axis.scatter(core_x, core_y, s=13, color="#334155", zorder=3)
    for role_index, values in enumerate(offspring_by_role):
        offspring = np.asarray(values, dtype=np.int64)
        parents, depth, transverse = _tree_coordinates(
            offspring,
            attachment_count=program.attachment_counts[role_index],
        )
        direction = np.asarray(ROLE_DIRECTIONS[role_index], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        perpendicular = np.asarray((-direction[1], direction[0]))
        scale = 0.24
        spread = 0.18
        coordinates = (
            depth[:, None] * scale * direction[None, :]
            + transverse[:, None] * spread * perpendicular[None, :]
        )
        roots = np.flatnonzero(parents < 0)
        for root in roots:
            axis.plot(
                [0.0, coordinates[root, 0]],
                [0.0, coordinates[root, 1]],
                color="#94a3b8",
                linewidth=1.0,
                zorder=0,
            )
        for node, parent in enumerate(parents):
            if parent < 0:
                continue
            axis.plot(
                [coordinates[parent, 0], coordinates[node, 0]],
                [coordinates[parent, 1], coordinates[node, 1]],
                color="#94a3b8",
                linewidth=1.0,
                zorder=0,
            )
        degree = offspring.copy()
        degree[parents >= 0] += 1
        sizes = np.where(degree >= 3, 24.0, 14.0)
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=sizes,
            color=ROLE_COLORS[role_index],
            edgecolors="white",
            linewidths=0.45,
            zorder=2,
        )
    node_counts = "/".join(str(value) for value in program.node_counts)
    junctions = "/".join(str(value) for value in program.junction_budgets)
    cycles = "/".join(str(value) for value in program.cycle_ranks)
    axis.set_title(
        f"{title}\nN={program.node_count + 5}  ext={node_counts}  J={junctions}  K={cycles}",
        fontsize=8,
        loc="left",
    )
    axis.relim()
    axis.autoscale_view()
    axis.margins(0.12)


def _atomic_figure(figure: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    try:
        figure.savefig(temporary, dpi=180, bbox_inches="tight", facecolor="white")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs=2, default=(500, 4000))
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    evaluations = {int(row["step"]): row for row in payload["evaluations"]}
    selected = [evaluations[step] for step in args.steps]
    programs = [_program(row) for row in selected[0]["calibration_programs"]]
    if any(
        row["calibration_programs"] != selected[0]["calibration_programs"] for row in selected[1:]
    ):
        raise SystemExit("requested evaluations do not share calibration programs")
    order = np.argsort([program.node_count for program in programs])
    positions = np.linspace(0, len(order) - 1, args.samples).round().astype(int)
    sample_indices = [int(order[position]) for position in positions]
    figure, axes = plt.subplots(
        args.samples,
        len(selected),
        figsize=(11.0, 2.7 * args.samples),
        squeeze=False,
    )
    for row_index, sample_index in enumerate(sample_indices):
        for column_index, evaluation in enumerate(selected):
            _draw_sample(
                axes[row_index, column_index],
                programs[sample_index],
                evaluation["offspring"][sample_index],
                title=f"step {evaluation['step']} · held program {sample_index}",
            )
    legend = "  ".join(
        f"{role}={color}" for role, color in zip(ROLE_NAMES, ROLE_COLORS, strict=True)
    )
    figure.suptitle(
        "FORGE Ugi morphology checkpoints (fixed five-atom core; topology before chemistry)\n"
        + legend,
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    _atomic_figure(figure, args.output.resolve())
    plt.close(figure)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

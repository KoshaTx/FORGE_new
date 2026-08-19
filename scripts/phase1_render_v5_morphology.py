#!/usr/bin/env python3
"""Render V5 morphology-probe samples as inspectable sparse lipid trees."""

from __future__ import annotations

import argparse
import html
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from forge.product.phase1_tree_topology_flow import preorder_offspring_to_parents

REPO = Path(__file__).resolve().parents[1]
REGION_COLORS = {0: "#7c3aed", 1: "#d97706", 2: "#0891b2"}


def _layout(offspring: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parents = preorder_offspring_to_parents(offspring)
    children: list[list[int]] = [[] for _ in range(offspring.size)]
    depths = np.zeros(offspring.size, dtype=np.int64)
    for child in range(1, offspring.size):
        parent = int(parents[child])
        children[parent].append(child)
        depths[child] = depths[parent] + 1
    y = np.zeros(offspring.size, dtype=np.float64)
    leaf = 0

    def assign(node: int) -> float:
        nonlocal leaf
        if not children[node]:
            value = float(leaf)
            leaf += 1
        else:
            values = [assign(child) for child in children[node]]
            value = float(sum(values) / len(values))
        y[node] = value
        return value

    assign(0)
    return parents, depths.astype(np.float64), y


def _sample_svg(offspring_values: list[int], region_values: list[int]) -> str:
    offspring = np.asarray(offspring_values, dtype=np.int64)
    regions = np.asarray(region_values, dtype=np.int64)
    parents, depths, rows = _layout(offspring)
    x = 24.0 + 25.0 * depths
    y = 24.0 + 22.0 * rows
    width = max(220.0, float(x.max(initial=0)) + 28.0)
    height = max(92.0, float(y.max(initial=0)) + 28.0)
    edges = []
    for child in range(1, offspring.size):
        parent = int(parents[child])
        edges.append(
            f'<line x1="{x[parent]:.1f}" y1="{y[parent]:.1f}" '
            f'x2="{x[child]:.1f}" y2="{y[child]:.1f}" />'
        )
    nodes = []
    degrees = offspring.copy()
    if offspring.size > 1:
        degrees[1:] += 1
    for node in range(offspring.size):
        radius = 5.0 if node == 0 else 4.2 if degrees[node] >= 3 else 3.2
        nodes.append(
            f'<circle cx="{x[node]:.1f}" cy="{y[node]:.1f}" r="{radius:.1f}" '
            f'fill="{REGION_COLORS[int(regions[node])]}" '
            f'data-node="{node}" data-degree="{int(degrees[node])}" />'
        )
    return (
        f'<svg viewBox="0 0 {width:.1f} {height:.1f}" role="img" '
        'aria-label="generated sparse lipid morphology">'
        '<g class="edges">'
        + "".join(edges)
        + '</g><g class="nodes">'
        + "".join(nodes)
        + "</g></svg>"
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO / "results/phase1/v5_morphology_probe_1k/result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/v5_morphology_probe_1k/samples.html",
    )
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    evaluations = payload.get("evaluations", [])
    if not evaluations:
        raise SystemExit("probe result contains no evaluations")
    evaluation = (
        next((row for row in evaluations if row["step"] == args.step), None)
        if args.step is not None
        else evaluations[-1]
    )
    if evaluation is None:
        raise SystemExit(f"probe result contains no evaluation at step {args.step}")
    cards = []
    for index, (program, offspring, regions) in enumerate(
        zip(
            evaluation["sample_programs"],
            evaluation["sample_offspring"],
            evaluation["sample_regions"],
            strict=True,
        ),
        start=1,
    ):
        summary = (
            f"N={sum(program[key] for key in ('n_head', 'n_interface', 'n_tail'))} · "
            f"H/I/T={program['n_head']}/{program['n_interface']}/{program['n_tail']} · "
            f"J(HI/T)={program['junction_budget_head_interface']}/"
            f"{program['junction_budget_tail']} · K={program['cycle_rank']}"
        )
        cards.append(
            '<article class="card">'
            f"<h2>Sample {index}</h2><p>{html.escape(summary)}</p>"
            f"{_sample_svg(offspring, regions)}</article>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>FORGE V5 morphology samples</title>
  <style>
    :root {{ color-scheme: light; font-family: ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0; padding: 24px; background: #f8fafc; color: #172033; }}
    header {{ max-width: 960px; margin: 0 auto 22px; }}
    h1 {{ margin: 0 0 8px; font-size: 25px; }}
    header p {{ line-height: 1.5; margin: 5px 0; color: #475569; }}
    .legend {{ display: flex; gap: 16px; font-size: 14px; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(440px,1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #dbe3ee; border-radius: 12px; padding: 14px; overflow: auto; }}
    .card h2 {{ font-size: 16px; margin: 0 0 4px; }}
    .card p {{ font-size: 12px; margin: 0 0 8px; color: #64748b; }}
    svg {{ min-width: 420px; width: 100%; height: 270px; }}
    .edges line {{ stroke: #9aa8ba; stroke-width: 1.5; stroke-linecap: round; }}
    .nodes circle {{ stroke: white; stroke-width: 1; }}
  </style>
</head>
<body>
  <header>
    <h1>FORGE V5 sparse morphology, checkpoint {evaluation['step']}</h1>
    <p>These are generated atom positions and structural roles before element and bond-order assignment. They are not yet chemical molecules.</p>
    <div class="legend">
      <span><i class="dot" style="background:#7c3aed"></i>head</span>
      <span><i class="dot" style="background:#d97706"></i>interface</span>
      <span><i class="dot" style="background:#0891b2"></i>tail</span>
    </div>
  </header>
  <main>{''.join(cards)}</main>
</body>
</html>
"""
    _atomic_text(args.output.resolve(), document)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

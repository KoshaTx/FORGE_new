"""Fail-closed reachability and historical-pin survey for retiring legacy code."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file, sha256_tree
from forge.core.io import write_json
from forge.core.provenance_archive import HistoricalPinArchive
from forge.paper.contract import PaperContract
from forge.paper.verification import provenance_closure
from forge.provenance.pins import collect_pins


def _candidates(repo: Path) -> tuple[str, ...]:
    paths = [
        path.relative_to(repo).as_posix()
        for root, patterns in (
            (repo / "src/forge", ("*.py",)),
            (repo / "scripts", ("*.py", "*.sh")),
        )
        for pattern in patterns
        for path in root.rglob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return tuple(sorted(set(paths)))


def _module_index(candidates: tuple[str, ...]) -> dict[str, str]:
    index: dict[str, str] = {}
    for relative in candidates:
        path = Path(relative)
        if path.parts[:2] != ("src", "forge") or path.suffix != ".py":
            continue
        module_parts = list(path.with_suffix("").parts[1:])
        if module_parts[-1] == "__init__":
            module_parts.pop()
        index[".".join(module_parts)] = relative
    return index


def _script_index(candidates: tuple[str, ...]) -> dict[str, str]:
    return {
        Path(relative).stem: relative
        for relative in candidates
        if Path(relative).parts[:1] == ("scripts",) and Path(relative).suffix == ".py"
    }


def _static_edges(
    repo: Path,
    relative: str,
    *,
    modules: dict[str, str],
    scripts: dict[str, str],
    candidates: set[str],
) -> tuple[set[str], list[str]]:
    path = repo / relative
    if path.suffix != ".py":
        return set(), []
    try:
        tree = ast.parse(path.read_text(), filename=relative)
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        return set(), [f"{relative}:parse:{error}"]
    edges: set[str] = set()
    unresolved_dynamic: list[str] = []

    def add_module(name: str) -> None:
        parts = name.split(".")
        for length in range(len(parts), 0, -1):
            target = modules.get(".".join(parts[:length]))
            if target is not None:
                edges.add(target)
                return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "forge" or alias.name.startswith("forge."):
                    add_module(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "forge" or node.module.startswith("forge."):
                add_module(node.module)
        elif isinstance(node, ast.Call):
            function = node.func
            dynamic_name = None
            if isinstance(function, ast.Name) and function.id == "__import__":
                dynamic_name = "__import__"
            elif isinstance(function, ast.Attribute) and function.attr in {
                "import_module",
                "spec_from_file_location",
            }:
                dynamic_name = function.attr
            if dynamic_name is not None:
                literal = next(
                    (
                        argument.value
                        for argument in node.args
                        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                    ),
                    None,
                )
                if literal is None:
                    unresolved_dynamic.append(f"{relative}:{node.lineno}:{dynamic_name}")
                elif literal.startswith("forge"):
                    add_module(literal)
                elif literal in scripts:
                    edges.add(scripts[literal])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            literal = node.value.replace("\\", "/")
            if literal in candidates:
                edges.add(literal)
            elif literal.startswith("scripts/") and literal.endswith((".py", ".sh")):
                if literal in candidates:
                    edges.add(literal)
    path_parts = Path(relative).parts
    if path_parts[:2] == ("src", "forge"):
        for depth in range(2, len(path_parts) - 1):
            package_init = Path(*path_parts[: depth + 1], "__init__.py").as_posix()
            if package_init in candidates:
                edges.add(package_init)
    return edges, unresolved_dynamic


def _closure(
    roots: set[str],
    graph: dict[str, set[str]],
) -> set[str]:
    reached: set[str] = set()
    queue = deque(sorted(roots))
    while queue:
        path = queue.popleft()
        if path in reached:
            continue
        reached.add(path)
        queue.extend(sorted(graph.get(path, set()) - reached))
    return reached


def _pin_identities(repo: Path, candidates: set[str]) -> dict[str, set[str]]:
    by_path: dict[str, set[str]] = defaultdict(set)
    roots = (repo / "results", repo / "docs/provenance", repo / "configs")
    for pin in collect_pins(roots):
        if pin.path in candidates:
            by_path[pin.path].add(pin.sha256)
    return by_path


def survey_code(
    repo: Path,
    contract_path: Path,
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    """Classify code against supported CLI and paper roots without deleting anything.

    ``retire_candidate`` means only that the static graph found no supported consumer and every
    historical path/hash identity is already content-addressed. Dynamic imports from reachable code
    are surfaced separately; a non-empty list blocks automated bulk deletion.
    """

    repo = repo.resolve()
    contract = PaperContract.load(contract_path)
    candidate_list = _candidates(repo)
    candidates = set(candidate_list)
    modules = _module_index(candidate_list)
    scripts = _script_index(candidate_list)
    graph: dict[str, set[str]] = {}
    dynamic: dict[str, list[str]] = {}
    for relative in candidate_list:
        edges, unresolved = _static_edges(
            repo,
            relative,
            modules=modules,
            scripts=scripts,
            candidates=candidates,
        )
        graph[relative] = edges
        if unresolved:
            dynamic[relative] = unresolved

    cli_roots = {
        "src/forge/cli.py",
        "src/forge/data/vendor.py",
        "src/forge/experiment/modal_app.py",
        "src/forge/provenance/archive.py",
        "src/forge/provenance/pins.py",
    }
    for experiment in contract.registered_experiments:
        spec_path = repo / "configs/experiments" / f"{experiment}.json"
        if not spec_path.is_file():
            continue
        document = json.loads(spec_path.read_text())
        for stage in document.get("stages", []):
            implementation = stage.get("implementation")
            if isinstance(implementation, str) and implementation.startswith("forge."):
                module = implementation.rsplit(".", 1)[0]
                if module in modules:
                    cli_roots.add(modules[module])

    evidence = provenance_closure(repo, contract)
    evidence_paths = {
        row["path"]
        for row in evidence["rows"]
        if row["path"].startswith(("results/", "configs/", "data/"))
    }
    paper_roots = set(contract.numerical_entrypoints)
    paper_roots.update(
        row["path"]
        for row in evidence["rows"]
        if row["path"] in candidates
    )
    # Older evidence builders did not always pin their own source. A script that names an artifact
    # in the recursive paper closure is therefore also a producer/consumer root. This closes the
    # gap without treating every historical script as live.
    for relative in sorted(path for path in candidates if path.startswith("scripts/")):
        text = (repo / relative).read_text(errors="replace")
        literals = {
            match.group(1).rstrip("/")
            for match in re.finditer(r"[\"']((?:results|configs|data)/[^\"']+)[\"']", text)
        }
        if any(
            literal in evidence_paths
            or any(path.startswith(f"{literal}/") for path in evidence_paths)
            for literal in literals
        ):
            paper_roots.add(relative)
    for producer in contract.publication_producers:
        paper_roots.update(
            item for item in producer.command if item in candidates and item.endswith((".py", ".sh"))
        )
    paper_roots.update(
        {"src/forge/paper/__init__.py", "src/forge/paper/build.py", "src/forge/paper/contract.py", "src/forge/paper/verification.py"}
    )
    cli_reachable = _closure(cli_roots & candidates, graph)
    paper_reachable = _closure(paper_roots & candidates, graph)
    reachable_dynamic = sorted(
        message
        for path in sorted(cli_reachable | paper_reachable)
        for message in dynamic.get(path, [])
    )

    identities = _pin_identities(repo, candidates)
    archive = HistoricalPinArchive.load(repo / "provenance/frozen-code/manifest.json", repo)
    rows = []
    for relative in candidate_list:
        pins = sorted(identities.get(relative, set()))
        archived = [digest for digest in pins if archive.resolve(relative, digest) is not None]
        in_cli = relative in cli_reachable
        in_paper = relative in paper_reachable
        if in_cli and in_paper:
            classification = "keep_shared"
        elif in_paper:
            classification = "keep_paper"
        elif in_cli:
            classification = "keep_cli"
        elif len(archived) == len(pins):
            classification = "retire_candidate"
        else:
            classification = "blocked_historical_pin"
        rows.append(
            {
                "archived_pin_identities": len(archived),
                "classification": classification,
                "historical_pin_identities": len(pins),
                "path": relative,
            }
        )

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        observed_classification = row["classification"]
        if not isinstance(observed_classification, str):
            raise TypeError("internal survey classification is not a string")
        counts[observed_classification] += 1
    document = {
        "classifications": rows,
        "counts": dict(sorted(counts.items())),
        "inputs": {
            "contract": {
                "path": str(contract_path.relative_to(repo)),
                "sha256": str(sha256_file(contract_path)),
            },
            "scripts_tree_sha256": str(sha256_tree(repo / "scripts")),
            "source_tree_sha256": str(sha256_tree(repo / "src/forge")),
        },
        "paper_id": contract.paper_id,
        "reachable_dynamic_imports": reachable_dynamic,
        "safe_for_automated_deletion": not reachable_dynamic,
        "schema_version": "forge.code_retirement_survey.v1",
    }
    if output is not None:
        write_json(output, document)
    return document


__all__ = ["survey_code"]

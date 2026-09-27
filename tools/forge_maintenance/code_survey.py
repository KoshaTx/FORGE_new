"""Single-contract reachability and historical-pin survey for manual code review."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from forge_paper.contract import PaperContract
from forge_paper.verification import provenance_closure
from forge_provenance.pins import collect_pins, load_moves
from forge_provenance.resolver import HistoricalPinArchive

from forge.core.hashing import sha256_file, sha256_tree
from forge.core.io import write_json


def _candidates(repo: Path) -> tuple[str, ...]:
    paths = [
        path.relative_to(repo).as_posix()
        for root, patterns in (
            (repo / "forge", ("*.py",)),
            (repo / "experiments" / "archive", ("*.py", "*.sh")),
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
        if path.parts[:1] != ("forge",) or path.suffix != ".py":
            continue
        module_parts = list(path.with_suffix("").parts)
        if module_parts[-1] == "__init__":
            module_parts.pop()
        index[".".join(module_parts)] = relative
    return index


def _script_index(candidates: tuple[str, ...]) -> dict[str, str]:
    return {
        Path(relative).stem: relative
        for relative in candidates
        if Path(relative).parts[:3] == ("experiments", "archive", "producers")
        and Path(relative).suffix == ".py"
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
            elif literal.startswith("experiments/archive/producers/") and literal.endswith(
                (".py", ".sh")
            ):
                if literal in candidates:
                    edges.add(literal)
    path_parts = Path(relative).parts
    if path_parts[:1] == ("forge",):
        for depth in range(1, len(path_parts) - 1):
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


def _classify_reachability(*, in_cli: bool, in_paper: bool, historical_pins_archived: bool) -> str:
    if in_cli and in_paper:
        return "keep_shared"
    if in_paper:
        return "keep_paper"
    if in_cli:
        return "keep_cli"
    if historical_pins_archived:
        return "review_unreached"
    return "blocked_historical_pin"


def survey_code(
    repo: Path,
    contract_path: Path,
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    """Classify code against CLI and one paper contract without deleting anything.

    ``review_unreached`` means only that this static graph found no consumer under those roots
    and all known historical pins are archived. Other supported studies may still use the file.
    A single-contract survey never authorizes automated deletion.
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

    cli_roots: set[str] = set()
    active_sources = [
        repo / "cli",
        repo / "experiments" / "_runtime",
        repo / "experiments" / "phase1",
    ]
    active_sources.extend(repo / "experiments" / name for name in ("__init__.py", "catalog.py"))
    for source in active_sources:
        paths = source.rglob("*.py") if source.is_dir() else (source,)
        for path in paths:
            if not path.is_file():
                continue
            edges, _ = _static_edges(
                repo,
                path.relative_to(repo).as_posix(),
                modules=modules,
                scripts=scripts,
                candidates=candidates,
            )
            cli_roots.update(edges)

    evidence = provenance_closure(repo, contract)
    evidence_paths = {
        row["path"]
        for row in evidence["rows"]
        if row["path"].startswith(("results/", "configs/", "data/"))
    }
    paper_roots = set(contract.numerical_entrypoints)
    paper_roots.update(row["path"] for row in evidence["rows"] if row["path"] in candidates)
    # Older evidence builders did not always pin their own source. A script that names an artifact
    # in the recursive paper closure is therefore also a producer/consumer root. This closes the
    # gap without treating every historical script as live.
    for relative in sorted(
        path for path in candidates if path.startswith("experiments/archive/producers/")
    ):
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
            item
            for item in producer.command
            if item in candidates and item.endswith((".py", ".sh"))
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
        classification = _classify_reachability(
            in_cli=in_cli,
            in_paper=in_paper,
            historical_pins_archived=len(archived) == len(pins),
        )
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
            "producers_tree_sha256": str(
                sha256_tree(repo / "experiments" / "archive" / "producers")
            ),
            "archive_tree_sha256": str(sha256_tree(repo / "experiments" / "archive")),
            "source_tree_sha256": str(sha256_tree(repo / "forge")),
        },
        "paper_id": contract.paper_id,
        "reachable_dynamic_imports": reachable_dynamic,
        "safe_for_automated_deletion": False,
        "schema_version": "forge.code_retirement_survey.v2",
        "survey_scope": {
            "paper_contract": str(
                contract_path.relative_to(repo)
                if contract_path.is_relative_to(repo)
                else contract_path
            ),
            "single_paper_contract_only": True,
            "other_supported_studies_must_be_reviewed": True,
        },
    }
    if output is not None:
        write_json(output, document)
    return document


__all__ = ["survey_code"]


# This broader inventory intentionally roots every maintained application module: some workflows
# are direct commands rather than catalog stages. Being unreached is never deletion authorization.
_SUPPORTED_SOURCE_ROOTS = ("forge", "cli", "experiments", "tools", "tests", "paper", "results")
_SUPPORTED_DOCUMENT_ROOTS = ("configs", "docs", "paper", "experiments", ".github")


def _supported_inputs(repo: Path) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for relative in _SUPPORTED_SOURCE_ROOTS:
        paths.update((repo / relative).rglob("*.py"))
        paths.update((repo / relative).rglob("*.sh"))
    for relative in _SUPPORTED_DOCUMENT_ROOTS:
        paths.update(
            path
            for path in (repo / relative).rglob("*")
            if path.suffix in {".json", ".md", ".txt", ".toml", ".yaml", ".yml"}
        )
    paths.update(repo / name for name in ("Makefile", "README.md", "pyproject.toml"))
    return tuple(
        sorted(
            path
            for path in paths
            if path.is_file()
            and not {"__pycache__", ".venv", "venv", "site-packages", ".git"}.intersection(
                path.parts
            )
            and not path.is_symlink()
        )
    )


def _supported_module(path: str) -> str:
    parts = list(Path(path).with_suffix("").parts)
    if parts[0] == "tools" or parts[:2] == ["paper", "forge_paper"]:
        parts.pop(0)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _supported_kind(path: str) -> str:
    if path.startswith("tests/"):
        return "reference"
    if path.startswith(("experiments/archive/", "results/")) or (
        path.startswith("paper/") and not path.startswith("paper/forge_paper/")
    ):
        return "historical"
    return "active"


def _supported_edges(
    relative: str, text: str, modules: dict[str, str], candidates: set[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve static edges conservatively; report expressions that cannot be resolved."""
    edges: list[dict[str, Any]] = []
    issues: list[str] = []

    def add(name: str, line: int, kind: str) -> None:
        # Parent initializers can register stages and have other import-time effects.
        parts = name.split(".")
        for depth in range(1, len(parts) + 1):
            target = modules.get(".".join(parts[:depth]))
            if target is not None:
                edges.append({"source": relative, "line": line, "target": target, "kind": kind})

    if relative.endswith(".py"):
        try:
            tree = ast.parse(text, filename=relative)
        except SyntaxError as error:
            return [], [f"{relative}:parse:{error}"]
        module = _supported_module(relative)
        package = module if relative.endswith("/__init__.py") else module.rpartition(".")[0]
        aliases = {
            alias.asname or alias.name: alias.name
            for statement in ast.walk(tree)
            if isinstance(statement, ast.ImportFrom)
            for alias in statement.names
            if alias.name in {"import_module", "spec_from_file_location"}
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name, node.lineno, "import")
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    if node.level > len(parts):
                        issues.append(f"{relative}:{node.lineno}:relative-import")
                        continue
                    base = ".".join(parts[: len(parts) - node.level + 1] + ([base] if base else []))
                add(base, node.lineno, "import")
                for alias in node.names:
                    add(f"{base}.{alias.name}", node.lineno, "import")
                    if alias.name == "*":
                        issues.append(f"{relative}:{node.lineno}:star-import")
            elif isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else (func.attr if isinstance(func, ast.Attribute) else "")
                )
                name = aliases.get(name, name)
                if name in {"__import__", "import_module", "spec_from_file_location"}:
                    index = 1 if name == "spec_from_file_location" else 0
                    arg = node.args[index] if len(node.args) > index else None
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        literal = arg.value
                        if literal in candidates:
                            edges.append(
                                {
                                    "source": relative,
                                    "line": node.lineno,
                                    "target": literal,
                                    "kind": "dynamic-literal",
                                }
                            )
                        elif not literal.startswith(".") and not literal.endswith(".py"):
                            add(literal, node.lineno, "dynamic-literal")
                        else:
                            issues.append(f"{relative}:{node.lineno}:dynamic-relative-or-file")
                    else:
                        issues.append(f"{relative}:{node.lineno}:dynamic-expression")
                elif isinstance(func, ast.Name) and name in {"getattr", "eval", "exec"}:
                    if (
                        name != "getattr"
                        or len(node.args) < 2
                        or not isinstance(node.args[1], ast.Constant)
                    ):
                        issues.append(f"{relative}:{node.lineno}:{name}")
    # Include full literal paths and module names in commands, configs, manifests, and __all__.
    for line, content in enumerate(text.splitlines(), 1):
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]*", content):
            token = token.rstrip("./")
            if token in candidates:
                edges.append({"source": relative, "line": line, "target": token, "kind": "path"})
            elif token in modules:
                add(token, line, "module-reference")
    return edges, sorted(set(issues))


def survey_supported_studies(repo: Path, *, output: Path | None = None) -> dict[str, Any]:
    """Inventory both studies, tools, references, and historical consumers without execution.

    All maintained application files are roots, including direct commands absent from the catalog.
    The report deliberately blocks unreached candidates if dynamic usage cannot be bounded.
    Historical pins use the existing collector; its scan is an evidence supplement, not proof of
    absence (missing/unparseable external artifacts cannot establish that code is unused).
    """
    repo = repo.resolve()
    inputs = _supported_inputs(repo)
    candidates = {
        path.relative_to(repo).as_posix() for path in inputs if path.suffix in {".py", ".sh"}
    }
    modules: dict[str, str] = {}
    moves = load_moves(repo / "docs/artifact_path_moves.json")
    issues: list[str] = []
    for candidate in sorted(candidates):
        if not candidate.endswith(".py"):
            continue
        name = _supported_module(candidate)
        if name in modules:
            issues.append(f"module-collision:{name}:{modules[name]}:{candidate}")
        else:
            modules[name] = candidate
    graph: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roots: dict[str, set[str]] = {key: set() for key in ("active", "reference", "historical")}
    hashes: dict[str, str] = {}
    for path in inputs:
        relative = path.relative_to(repo).as_posix()
        hashes[relative] = str(sha256_file(path))
        kind = _supported_kind(relative)
        if relative in candidates and not relative.startswith("forge/"):
            roots[kind].add(relative)
        try:
            text = path.read_text()
        except (OSError, UnicodeError) as error:
            issues.append(f"{relative}:read:{error}")
            continue
        if path.suffix == ".json":
            try:
                json.loads(text)
            except ValueError as error:
                issues.append(f"{relative}:parse-json:{error}")
        edges, unresolved = _supported_edges(relative, text, modules, candidates | set(moves))
        issues.extend(unresolved)
        for edge in edges:
            target = moves.get(edge["target"], edge["target"])
            if target not in candidates:
                continue
            edge["original_target"] = edge["target"]
            edge["target"] = target
            graph[relative].add(target)
            incoming[target].append(edge)
            if relative not in candidates:
                roots[kind].add(target)
    pins = collect_pins(
        tuple(
            repo / name
            for name in ("results", "docs/provenance", "configs", "experiments", "paper")
        )
    )
    if output is not None:
        output_name = (
            str(output.resolve().relative_to(repo))
            if output.resolve().is_relative_to(repo)
            else str(output.resolve())
        )
        pins = [pin for pin in pins if pin.declared_by != output_name]
    pin_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    archive = HistoricalPinArchive.load(repo / "provenance/frozen-code/manifest.json", repo)
    for pin in pins:
        current = moves.get(pin.path, pin.path)
        if current not in candidates:
            continue
        pin_rows[current].append(
            {"path": pin.path, "sha256": pin.sha256, "declared_by": pin.declared_by}
        )
        roots["historical"].add(current)
    reached = {kind: _closure(paths, graph) for kind, paths in roots.items()}
    rows: list[dict[str, Any]] = []
    for relative in sorted(candidates):
        unresolved_pins = sorted(
            {
                pin["sha256"]
                for pin in pin_rows[relative]
                if hashes[relative] != pin["sha256"]
                and archive.resolve(pin["path"], pin["sha256"]) is None
            }
        )
        roles = [kind for kind in roots if relative in reached[kind]]
        classification = next((f"keep_{kind}" for kind in roles), "review_unreached")
        if unresolved_pins or (not roles and issues):
            classification = "blocked"
        rows.append(
            {
                "path": relative,
                "sha256": hashes[relative],
                "classification": classification,
                "roles": roles,
                "incoming_references": incoming[relative],
                "historical_pins": pin_rows[relative],
                "unresolved_pin_identities": unresolved_pins,
                "requires_manual_symbol_review": True,
            }
        )
    # Hash the pin declarations as well as all inspected source/document bytes. No count-only gate.
    from forge.core.hashing import sha256_json

    pin_digest = str(sha256_json(sorted((pin.path, pin.sha256, pin.declared_by) for pin in pins)))
    archive_path = repo / "provenance/frozen-code/manifest.json"
    if archive_path.is_file():
        hashes["provenance/frozen-code/manifest.json"] = str(sha256_file(archive_path))
    document = {
        "schema_version": "forge.supported_studies_code_survey.v1",
        "scope": "supported-studies",
        "safe_for_automated_deletion": False,
        "root_policy": "all maintained applications, tools, study documents, tests, and historical consumers",
        "inputs": hashes,
        "pin_declarations_sha256": pin_digest,
        "fingerprint": str(sha256_json({"files": hashes, "pins": pin_digest})),
        "unresolved_references": sorted(set(issues)),
        "classifications": rows,
        "counts": dict(sorted(Counter(row["classification"] for row in rows).items())),
        "limitations": [
            "Module reachability does not prove symbol use or authorize retirement.",
            "Dynamic expressions and external consumers require manual review.",
            "The historical pin collector tolerates absent or unreadable artifacts; this is not absence proof.",
            "Every direct application is retained unless separately adjudicated as obsolete.",
            "Installed environments (.venv, venv, site-packages), Git metadata, caches, and symlinks are excluded from source candidates.",
        ],
    }
    if output is not None:
        write_json(output, document)
    return document

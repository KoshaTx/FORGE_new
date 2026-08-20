"""Executable boundaries for the root-package FORGE architecture."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIBRARY = REPO / "forge"
RUNTIME = REPO / "experiments" / "_runtime"
ACTIVE_EXPERIMENTS = REPO / "experiments" / "phase1"

LIBRARY_PACKAGES = {
    "assembly",
    "chemistry",
    "core",
    "corpus",
    "flow",
    "model",
    "potency",
    "synthesis",
}
REMOVED_LIBRARY_PACKAGES = {"audit", "bio", "cli", "data", "design", "experiment", "route", "value"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_scientific_library_has_only_declared_domains() -> None:
    packages = {
        path.name
        for path in LIBRARY.iterdir()
        if path.is_dir() and path.name != "__pycache__" and any(path.rglob("*.py"))
    }
    assert packages == LIBRARY_PACKAGES
    assert not (REPO / "src").exists()
    assert not (REPO / "scripts").exists()


def test_removed_catch_all_namespaces_do_not_exist() -> None:
    survivors = sorted(name for name in REMOVED_LIBRARY_PACKAGES if (LIBRARY / name).exists())
    assert not survivors, f"removed forge namespaces survived: {survivors}"


def test_library_never_depends_on_cli_or_experiment_apps() -> None:
    violations: list[str] = []
    for path in sorted(LIBRARY.rglob("*.py")):
        for imported in sorted(_imports(path)):
            if imported == "cli" or imported.startswith(("cli.", "experiments.")):
                violations.append(f"{path.relative_to(REPO)} -> {imported}")
    assert not violations, "application code leaked into forge:\n" + "\n".join(violations)


def test_generic_runtime_has_no_forge_stage_or_application_imports() -> None:
    """The runner may reuse forge.core records, but cannot know scientific stages or apps."""

    violations: list[str] = []
    for path in sorted(RUNTIME.rglob("*.py")):
        for imported in sorted(_imports(path)):
            if imported.startswith("experiments.phase1") or (
                imported.startswith("forge.") and not imported.startswith("forge.core")
            ):
                violations.append(f"{path.relative_to(REPO)} -> {imported}")
    assert not violations, "scientific code leaked into the generic runtime:\n" + "\n".join(
        violations
    )


def test_active_experiments_never_import_the_historical_archive() -> None:
    violations: list[str] = []
    for path in sorted(ACTIVE_EXPERIMENTS.rglob("*.py")):
        for imported in sorted(_imports(path)):
            if imported == "experiments.archive" or imported.startswith("experiments.archive."):
                violations.append(f"{path.relative_to(REPO)} -> {imported}")
    assert not violations, "active experiments depend on historical code:\n" + "\n".join(violations)


def test_cli_is_an_application_boundary_not_a_library_dependency() -> None:
    cli_imports = _imports(REPO / "cli" / "__init__.py")
    eager_scientific_imports = sorted(
        name for name in cli_imports if name.startswith("forge.") and name != "forge.core"
    )
    assert not eager_scientific_imports
    assert "experiments" in cli_imports
    assert "experiments._runtime" in cli_imports


def test_catalog_owns_every_active_experiment_specification() -> None:
    from experiments.catalog import SPECIFICATIONS

    declared = {Path(path) for path in SPECIFICATIONS.values()}
    discovered = {
        path.relative_to(REPO)
        for path in (REPO / "experiments").rglob("*.json")
        if "configs" not in path.parts and "archive" not in path.parts
    }
    assert declared == discovered
    assert all(path.parts[0] == "experiments" for path in declared)


def test_archived_producers_are_importable_but_not_a_supported_cli_surface() -> None:
    archive = REPO / "experiments" / "archive" / "producers"
    assert (archive / "README.md").is_file()
    assert (archive / "__init__.py").is_file()
    assert any(archive.glob("m0_*.py"))
    assert any(archive.glob("phase1_*.py"))


def test_no_deleted_package_survives_only_as_bytecode() -> None:
    empty_namespaces = [
        path.relative_to(REPO).as_posix()
        for path in LIBRARY.iterdir()
        if path.is_dir() and path.name != "__pycache__" and not any(path.rglob("*.py"))
    ]
    assert not empty_namespaces

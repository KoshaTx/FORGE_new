"""Both study interfaces and their extracted source dependencies remain available."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from experiments._runtime.registry import registry
from experiments._runtime.source import SOURCE_DIRECTORIES, SOURCE_FILES, source_fingerprint
from experiments._runtime.spec import ExperimentSpec
from experiments.catalog import SPECIFICATIONS, load_catalog
from forge.model._synthesis_sampling import SAMPLING_SOURCE_FILES

REPO = Path(__file__).resolve().parents[1]


def test_every_catalog_workflow_resolves_without_execution() -> None:
    load_catalog()
    for relative in SPECIFICATIONS.values():
        spec = ExperimentSpec.load(REPO / relative)
        for step in spec.stages:
            assert callable(registry.resolve(step.implementation))


def test_both_study_stage_interfaces_remain_registered() -> None:
    from experiments.phase1.multireaction import compose_lipid_training, stages
    from experiments.phase1.product_l1 import stages as product_l1

    load_catalog()
    assert registry.resolve("corpus.multireaction.lnpdb.v1") is stages.build_multireaction_corpus
    assert registry.resolve("model.multireaction.training.v2") is stages.train_multireaction
    assert registry.resolve(compose_lipid_training.IMPLEMENTATION) is compose_lipid_training.train
    assert registry.resolve("generate.ugi.joint-train.v1") is product_l1.train_ugi_joint_stage
    assert registry.resolve("generate.ugi.sample-shards.v1") is product_l1.sample_ugi_shards


@pytest.mark.parametrize(
    "module,attribute",
    [
        ("forge.corpus.compose_lipid_pretraining", "IMPLEMENTATION_SOURCES"),
        ("experiments.phase1.multireaction.combinatorial_generation", "IMPLEMENTATION_SOURCES"),
        ("experiments.phase1.multireaction.combinatorial_source_core_completion", "SOURCES"),
        ("experiments.phase1.multireaction.combinatorial_count_layout_generation", "SOURCES"),
    ],
)
def test_direct_producers_pin_every_sampler_module(module: str, attribute: str) -> None:
    expected = {
        path.relative_to(REPO).as_posix()
        for path in (REPO / "forge/model/_synthesis_sampling").glob("*.py")
    } | {"forge/model/synthesis_program_sampling.py"}
    assert set(SAMPLING_SOURCE_FILES) == expected
    inventory = getattr(importlib.import_module(module), attribute)
    assert expected <= set(inventory)
    assert len(inventory) == len(set(inventory))


def test_extracted_code_changes_current_run_identity(tmp_path: Path) -> None:
    for relative in SOURCE_DIRECTORIES:
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    for relative in SOURCE_FILES:
        (tmp_path / relative).write_text("# source fingerprint fixture\n")
    files = [
        *SAMPLING_SOURCE_FILES,
        "forge/synthesis/sources/_acquisition_io.py",
        "experiments/phase1/product_l1/_stage_support.py",
        *(
            path.relative_to(REPO).as_posix()
            for path in (REPO / "experiments/phase1/product_l1").glob("*stages.py")
        ),
    ]
    for relative in files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO / relative).read_bytes())
    original = source_fingerprint(tmp_path)
    for relative in files:
        path = tmp_path / relative
        content = path.read_bytes()
        path.write_bytes(content + b"\n# changed source fixture\n")
        assert source_fingerprint(tmp_path) != original, relative
        path.write_bytes(content)
    assert source_fingerprint(tmp_path) == original

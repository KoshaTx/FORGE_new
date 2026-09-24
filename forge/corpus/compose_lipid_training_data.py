"""Admitted, family-weighted batches from the immutable COMPOSE preparation cache."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_training_measure import (
    COHORT_POLICY,
    POLICY,
    SCHEMA,
    qualified_cohort_families,
)
from forge.corpus.mapped_program_cache import MappedProgramCache
from forge.corpus.qualified_program_cache import (
    QualifiedProgramCache,
    QualifiedProgramExample,
    collate_qualified_program_examples,
)

TRAINING_VALIDATION_SCHEMA = "forge.compose_lipid_training_validation.v1"
TRAINING_TEST_FILES = (
    "tests/test_qualified_program_cache.py",
    "tests/test_mapped_program_cache.py",
    "tests/test_compose_lipid_training_measure.py",
    "tests/test_compose_lipid_training_data.py",
    "tests/test_repeat_supervision.py",
    "tests/test_compose_lipid_restoration.py",
    "tests/test_compose_lipid_family_batches.py",
    "tests/test_compose_lipid_generation.py",
    "tests/test_compose_lipid_noise_marginals.py",
    "tests/test_compose_lipid_run.py",
    "tests/test_compose_lipid_tensor_cache.py",
    "tests/test_compose_lipid_prefetch.py",
    "tests/test_compose_lipid_parallel.py",
    "tests/test_compose_lipid_sharding.py",
    "tests/test_compose_lipid_adaptive.py",
    "tests/test_ordered_relation_embedding.py",
    "tests/test_modal_restart.py",
    "tests/test_compose_lipid_prepared_run.py",
    "tests/test_compose_lipid_gpu_preflight.py",
    "tests/test_training_restart.py",
    "tests/test_experiment_modal.py",
)


class ComposeLipidTrainingDataError(ValueError):
    """The admitted population, probabilities or training support changed."""


def _read(repo: Path, value: Mapping[str, str], label: str) -> dict:
    return json.loads(resolve_pin(value, repo, label=label).read_text())


def require_training_admission(
    repo: Path,
    admission: Mapping[str, str],
    *,
    population: Mapping[str, str],
    verification: Mapping[str, str],
    measure: Mapping[str, str],
) -> None:
    """Require final data admission and passing checks of the current training pipeline.

    A preparation or weight-compilation receipt never authorizes training. This
    consumer does not create admission receipts or change source admission flags.
    """
    document = _read(repo, admission, "COMPOSE training admission")
    if (
        document.get("schema_version") != "forge.compose_lipid_training_admission.v1"
        or document.get("training_admitted") is not True
        or document.get("training_ready") is not True
    ):
        raise ComposeLipidTrainingDataError("Final training admission is required")
    inputs = document.get("inputs", {})
    for name, value in (
        ("population", population),
        ("verification", verification),
        ("measure", measure),
    ):
        if inputs.get(name) != dict(value):
            raise ComposeLipidTrainingDataError(f"Admission belongs to a different {name}")
    weights = _read(repo, measure, "admitted graph measure")
    if weights.get("policy") == COHORT_POLICY:
        if not inputs.get("cohort") or inputs["cohort"] != weights["inputs"].get("cohort"):
            raise ComposeLipidTrainingDataError("Admission must bind the explicit qualified cohort")
        resolve_pin(inputs["cohort"], repo, label="admitted qualified cohort")
    validation = _read(repo, inputs["validation"], "admitted training validation")
    tested = validation.get("tests", {})
    if (
        validation.get("schema_version") != TRAINING_VALIDATION_SCHEMA
        or validation.get("training_checks_passed") is not True
        or validation.get("test_files") != list(TRAINING_TEST_FILES)
        or validation.get("all_test_files_executed") is not True
        or validation.get("pytest_exit_code") != 0
        or validation.get("source_snapshot_unchanged") is not True
        or validation.get("vendor_verify_exit_code") != 0
        or tested.get("failures") != 0
        or tested.get("errors") != 0
        or tested.get("skipped") != 0
        or type(tested.get("passed")) is not int
        or tested["passed"] < 1
    ):
        raise ComposeLipidTrainingDataError("Current training pipeline validation has not passed")
    for name, value in validation["inputs"].items():
        resolve_pin(value, repo, label="training validation " + name)
    snapshot = _read(
        repo, validation["inputs"]["source-snapshot.json"], "validated source snapshot"
    )
    required = set(TRAINING_TEST_FILES) | {
        "forge/corpus/compose_lipid_training_data.py",
        "forge/corpus/compose_lipid_training_measure.py",
        "forge/corpus/qualified_program_cache.py",
        "forge/model/compose_lipid_training.py",
    }
    if not required <= set(snapshot):
        raise ComposeLipidTrainingDataError("Training validation omits current code or tests")
    for path, digest in snapshot.items():
        resolve_pin({"path": path, "sha256": digest}, repo, label="validated source")


class ComposeLipidTrainingData:
    """Sample the compiled graph measure without rebalancing source-role bindings.

    Only scalar probabilities are held for the full corpus. Graphs are loaded via
    the bounded preparation cache. Sampling is with replacement and its NumPy
    generator is owned by the caller so checkpoint code can preserve its state.
    """

    def __init__(
        self,
        repo: Path,
        *,
        population: Mapping[str, str],
        verification: Mapping[str, str],
        measure: Mapping[str, str],
        admission: Mapping[str, str],
        maximum_cached_shards: int = 4,
        mapped_cache: Mapping[str, str] | None = None,
    ) -> None:
        repo = repo.resolve()
        self._database = None
        self.cache = None
        require_training_admission(
            repo, admission, population=population, verification=verification, measure=measure
        )
        self.identity = {
            "population": dict(population),
            "verification": dict(verification),
            "measure": dict(measure),
            "admission": dict(admission),
        }
        weights = _read(repo, measure, "compiled COMPOSE measure")
        if (
            weights.get("schema_version") != SCHEMA
            or weights.get("policy") not in {POLICY, COHORT_POLICY}
            or weights.get("record_order") != "qualified_preparation_shard_id_then_row_index"
            or weights.get("inputs", {}).get("population") != dict(population)
            or weights.get("inputs", {}).get("verification") != dict(verification)
            or weights.get("training_admitted") is not False
            or weights.get("raw_family_frequency_sampling") is not False
            or weights.get("source_role_bindings_receive_separate_mass") is not False
        ):
            raise ComposeLipidTrainingDataError("Compiled graph measure contract changed")
        resolve_pin(weights["implementation"], repo, label="weight compiler")
        for name, value in weights["inputs"].items():
            resolve_pin(value, repo, label="weight input " + name)
        prepared = _read(repo, population, "weighted preparation")
        universe = _read(repo, weights["inputs"]["universe_config"], "source universe")
        evidence = _read(repo, weights["inputs"]["evidence_index"], "source evidence")
        request = _read(repo, prepared["request"], "preparation request")
        if request["inputs"]["evidence_index"] != weights["inputs"]["evidence_index"]:
            raise ComposeLipidTrainingDataError("Weights and preparation use different evidence")
        resolve_pin(evidence["artifact"], repo, label="exact-evidence database")
        formal = sorted(set(universe["expected_family_rows"]) - set(universe["reference_families"]))
        if len(formal) != 23:
            raise ComposeLipidTrainingDataError(
                "The authenticated source universe must retain 23 families"
            )
        scoped = weights["policy"] == COHORT_POLICY
        if scoped:
            formal = qualified_cohort_families(
                repo,
                weights["inputs"]["cohort"],
                population=population,
                verification=verification,
                evidence_index=weights["inputs"]["evidence_index"],
                prepared=prepared,
                evidence=evidence,
                formal=formal,
            )
        count = weights.get("records")
        if (
            weights.get("formal_families") != formal
            or type(count) is not int
            or count < 1
            or prepared["totals"]["records"] != count
            or evidence["summary"].get("exact") != count
            or (
                not scoped
                and any(
                    evidence["summary"].get(key) != value
                    for key, value in (("eligible", count), ("exact", count), ("pending", 0))
                )
            )
            or set(prepared["by_family"]) != set(formal)
        ):
            raise ComposeLipidTrainingDataError(
                "Complete exact support for the admitted training population is required"
            )
        lookup = resolve_pin(
            prepared["artifacts"]["preparation.sqlite"], repo, label="weighted lookup"
        )
        database = resolve_pin(weights["artifact"], repo, label="compiled probabilities")
        try:
            self._database = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
            self._database.execute("PRAGMA query_only=ON")
            self._database.execute("ATTACH DATABASE ? AS source", (lookup.as_uri() + "?mode=ro",))
            actual = self._database.execute(
                "SELECT count(*),count(DISTINCT target_id),count(DISTINCT constitution_id),"
                "count(DISTINCT record_index),min(record_index),max(record_index) FROM weights"
            ).fetchone()
            if actual != (count, count, count, count, 0, count - 1):
                raise ComposeLipidTrainingDataError("Weights omit or duplicate source graphs")
            mismatch = self._database.execute(
                "WITH ordered AS (SELECT row_number() OVER (ORDER BY shard_id,row_index)-1 AS idx,"
                "target_id,constitution_id,family FROM source.records) "
                "SELECT count(*) FROM ordered o LEFT JOIN weights w ON w.record_index=o.idx "
                "WHERE w.target_id IS NULL OR w.target_id!=o.target_id "
                "OR w.constitution_id!=o.constitution_id OR w.family!=o.family"
            ).fetchone()[0]
            if (
                mismatch
                or self._database.execute("SELECT count(*) FROM source.records").fetchone()[0]
                != count
            ):
                raise ComposeLipidTrainingDataError("Weights and source graph order differ")
            counts = prepared["by_family"]
            observed = self._database.execute(
                "SELECT family,count(*),min(probability),max(probability) FROM weights GROUP BY family"
            ).fetchall()
            if len(observed) != len(formal):
                raise ComposeLipidTrainingDataError("Weighted family support changed")
            for family, size, smallest, largest in observed:
                expected = 1.0 / (len(formal) * counts[family])
                if size != counts[family] or smallest != expected or largest != expected:
                    raise ComposeLipidTrainingDataError(
                        "Per-graph or formal-family probability changed"
                    )
            probabilities = np.fromiter(
                (
                    r[0]
                    for r in self._database.execute(
                        "SELECT probability FROM weights ORDER BY record_index"
                    )
                ),
                dtype=np.float64,
                count=count,
            )
            if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0, abs_tol=1e-12):
                raise ComposeLipidTrainingDataError("Compiled probabilities are not normalized")
            self._cumulative = probabilities.cumsum()
            self._cumulative[-1] = 1.0
            self.maximum_heavy_atoms = self._database.execute(
                "SELECT max(atoms) FROM source.records"
            ).fetchone()[0]
            if mapped_cache is None:
                self.cache = QualifiedProgramCache(
                    repo,
                    population=population,
                    verification=verification,
                    maximum_cached_shards=maximum_cached_shards,
                )
            else:
                self.cache = MappedProgramCache(repo, manifest=mapped_cache)
                metadata = self.cache.metadata
                if (
                    metadata["selection"] != "complete_qualified_population"
                    or metadata["inputs"]["population"] != dict(population)
                    or metadata["inputs"]["verification"] != dict(verification)
                    or len(self.cache) != count
                    or metadata["by_family"] != counts
                ):
                    raise ComposeLipidTrainingDataError(
                        "Mapped cache is not the admitted population"
                    )
                mapped_path = resolve_pin(
                    metadata["artifacts"]["sources.sqlite"], repo, label="mapped graph identities"
                )
                self._database.execute(
                    "ATTACH DATABASE ? AS mapped", (mapped_path.as_uri() + "?mode=ro",)
                )
                if self._database.execute(
                    "SELECT count(*) FROM weights w LEFT JOIN mapped.sources m "
                    "ON m.idx=w.record_index WHERE m.target_id IS NULL "
                    "OR m.target_id!=w.target_id OR m.constitution_id!=w.constitution_id "
                    "OR m.family!=w.family"
                ).fetchone()[0]:
                    raise ComposeLipidTrainingDataError("Mapped cache changes weighted graph order")
                self.identity["mapped_cache"] = dict(mapped_cache)
            self.vocabulary, self.atom_vocabulary = (
                self.cache.vocabulary,
                self.cache.atom_vocabulary,
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()
            self.cache = None
        if self._database is not None:
            self._database.close()
            self._database = None

    def __enter__(self) -> ComposeLipidTrainingData:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._cumulative)

    def sample_indices(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        families_per_batch: int | None = None,
        family_selection: Sequence[int] | None = None,
    ) -> np.ndarray:
        if self._database is None:
            raise ComposeLipidTrainingDataError("Training data is closed")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if families_per_batch is not None:
            if not hasattr(self, "_family_indices"):
                groups: dict[str, list[int]] = {}
                for index, family in self._database.execute(
                    "SELECT record_index,family FROM weights ORDER BY record_index"
                ):
                    groups.setdefault(family, []).append(index)
                self._family_indices = tuple(
                    np.asarray(groups[name], dtype=np.int64) for name in sorted(groups)
                )
            if (
                type(families_per_batch) is not int
                or not 1 <= families_per_batch <= len(self._family_indices)
                or batch_size % families_per_batch
            ):
                raise ValueError("Batch size must divide into an admitted number of families")
            # The loader has verified equal family mass and uniform graph mass within
            # each family. Uniformly choosing k distinct families and B/k graphs from
            # each therefore preserves every graph's marginal sampling probability.
            families = (
                rng.choice(len(self._family_indices), families_per_batch, replace=False)
                if family_selection is None
                else np.asarray(family_selection)
            )
            if (
                families.shape != (families_per_batch,)
                or families.dtype.kind not in "iu"
                or len(set(families.tolist())) != families_per_batch
                or np.any(families < 0)
                or np.any(families >= len(self._family_indices))
            ):
                raise ValueError("Explicit family selection must contain distinct admitted indices")
            selected = np.concatenate(
                [
                    rng.choice(
                        self._family_indices[i], batch_size // families_per_batch, replace=True
                    )
                    for i in families
                ]
            )
            rng.shuffle(selected)
            return selected
        if family_selection is not None:
            raise ValueError("Explicit family selection requires family-block sampling")
        return np.searchsorted(self._cumulative, rng.random(batch_size), side="right")

    def iter_weighted_examples(self) -> Iterator[tuple[QualifiedProgramExample, float]]:
        """Stream each admitted graph once with its compiled probability, without sampling."""
        if self.cache is None or self._database is None:
            raise ComposeLipidTrainingDataError("Training data is closed")
        for index, identity, probability in self._database.execute(
            "SELECT record_index,target_id,probability FROM weights ORDER BY record_index"
        ):
            if isinstance(self.cache, MappedProgramCache):
                yield self.cache.record(index), probability
            else:
                yield self.cache.records((identity,))[0], probability

    def batch(
        self,
        indices: Sequence[int],
        *,
        maximum_nodes: int,
        maximum_closures: int,
        node_padding: str = "model",
        repeat_supervision: str = "serialization",
        core_conditioning: str = "adapter",
    ) -> dict[str, Any]:
        if self.cache is None or self._database is None:
            raise ComposeLipidTrainingDataError("Training data is closed")
        if maximum_nodes < self.maximum_heavy_atoms:
            raise ComposeLipidTrainingDataError(
                "Model support would exclude eligible large molecules"
            )
        if node_padding not in ("model", "batch"):
            raise ValueError("node_padding must be 'model' or 'batch'")
        if repeat_supervision not in ("serialization", "exact_fragment"):
            raise ValueError("repeat_supervision must be 'serialization' or 'exact_fragment'")
        if not len(indices):
            raise ValueError("A training batch must be nonempty")
        identities = []
        for index in indices:
            if (
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or not 0 <= index < len(self)
            ):
                raise IndexError(index)
            identities.append(
                int(index)
                if isinstance(self.cache, MappedProgramCache)
                else self._database.execute(
                    "SELECT target_id FROM weights WHERE record_index=?", (int(index),)
                ).fetchone()[0]
            )
        # Keep repeated sampled rows and their order; component IDs remain non-neural.
        # Batch padding trims absent positions only. The model-wide support check above
        # still applies even when this draw contains only small molecules.
        examples = self.cache.records(identities)
        if core_conditioning == "qualified_core":
            from forge.model.compose_lipid_core import condition_on_qualified_core

            examples = tuple(condition_on_qualified_core(e) for e in examples)
        elif core_conditioning != "adapter":
            raise ValueError("Unknown core conditioning")
        batch = collate_qualified_program_examples(
            examples,
            maximum_nodes=None if node_padding == "batch" else maximum_nodes,
            maximum_closures=maximum_closures,
            conditioning_mode="program",
        )
        import torch

        family_index = (
            {name: i for i, name in enumerate(sorted(self.cache.metadata["by_family"]))}
            if isinstance(self.cache, MappedProgramCache)
            else {
                name: i
                for i, (name,) in enumerate(
                    self._database.execute("SELECT DISTINCT family FROM weights ORDER BY family")
                )
            }
        )
        batch["family_states"] = torch.tensor([family_index[e.family] for e in examples])
        if repeat_supervision == "exact_fragment":
            from forge.model.repeat_supervision import align_qualified_repeats

            for name in ("repeat_atom_groups", "repeat_bond_groups"):
                batch[name] = torch.zeros_like(batch["nodes"])
            for index, example in enumerate(examples):
                atoms, bonds, _ = align_qualified_repeats(example)
                count = example.record.node_count
                batch["repeat_atom_groups"][index, :count] = torch.from_numpy(atoms)
                batch["repeat_bond_groups"][index, :count] = torch.from_numpy(bonds)
        return batch

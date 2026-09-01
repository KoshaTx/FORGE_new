"""Training-fold joint-lipid measure for Ugi reaction specialization.

The shared FORGE model needs reaction-enumerated products for coverage, but those rows must not
overwhelm the much smaller measured ionizable-lipid distribution during realism specialization.
This module compiles an explicit mixture over *complete training products*: one stratum is uniform
over unique source-adjudicated measured Ugi products, and the other retains the frozen source
weights over the remaining reaction-enumerated support.

Product identifiers are used only to align the immutable training cache.  They are never returned
as model inputs, component identifiers, stored graphs, or fragment tokens.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import sha256_file
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache


class UgiJointLipidPriorError(ValueError):
    """The measured/exploration training mixture is malformed or cannot align to the cache."""


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true"}


@dataclass(frozen=True)
class UgiJointLipidPrior:
    """Identity-free neural training measure over complete Ugi products."""

    reaction_id: str
    measured_mass: float
    exploration_mass: float
    assignments_path: Path
    assignments_sha256: str
    measured_training_record_ids: tuple[str, ...]

    @classmethod
    def from_training_assignments(
        cls,
        assignments_path: Path,
        *,
        measured_mass: float,
        exploration_mass: float,
        reaction_id: str = "ugi_3cr_agile",
        expected_sha256: str | None = None,
    ) -> UgiJointLipidPrior:
        """Read only unique source-adjudicated product IDs from the frozen training fold."""

        if (
            not reaction_id
            or not math.isfinite(measured_mass)
            or not math.isfinite(exploration_mass)
            or measured_mass <= 0
            or exploration_mass < 0
            or not math.isclose(measured_mass + exploration_mass, 1.0)
        ):
            raise UgiJointLipidPriorError("joint-lipid mixture masses are invalid")
        path = assignments_path.resolve()
        if not path.is_file():
            raise UgiJointLipidPriorError(f"Ugi assignments are missing: {path}")
        observed_sha256 = str(sha256_file(path))
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise UgiJointLipidPriorError(
                f"Ugi assignments changed: expected {expected_sha256}, found {observed_sha256}"
            )
        opener = gzip.open if path.suffix == ".gz" else open
        measured: set[str] = set()
        with opener(path, mode="rt", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "product_id",
                "primary_product_fold",
                "is_source_adjudicated_measured_product",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise UgiJointLipidPriorError("Ugi assignments schema changed")
            for row in reader:
                if row["primary_product_fold"] != "train" or not _truthy(
                    row["is_source_adjudicated_measured_product"]
                ):
                    continue
                record_id = row["product_id"].strip()
                if not record_id:
                    raise UgiJointLipidPriorError(
                        "source-adjudicated measured training row has no product ID"
                    )
                measured.add(record_id)
        if not measured:
            raise UgiJointLipidPriorError(
                "Ugi assignments contain no source-adjudicated measured training products"
            )
        return cls(
            reaction_id=reaction_id,
            measured_mass=float(measured_mass),
            exploration_mass=float(exploration_mass),
            assignments_path=path,
            assignments_sha256=observed_sha256,
            measured_training_record_ids=tuple(sorted(measured)),
        )

    def training_measure(
        self,
        cache: SynthesisProgramProductionCache,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the exact measured/exploration mixture aligned to one production cache."""

        if self.reaction_id not in cache.vocabulary.program_to_index:
            raise UgiJointLipidPriorError(
                f"production cache lacks reaction program {self.reaction_id!r}"
            )
        target = cache.indices(program_id=self.reaction_id, fold="train")
        if not len(target):
            raise UgiJointLipidPriorError("production cache has no Ugi training rows")
        by_id: dict[str, int] = {}
        for raw_index in target:
            index = int(raw_index)
            record_id = str(cache.record_id(index))
            if not record_id:
                raise UgiJointLipidPriorError("production cache contains an empty record ID")
            if record_id in by_id:
                raise UgiJointLipidPriorError(
                    f"production cache repeats Ugi training record ID {record_id!r}"
                )
            by_id[record_id] = index
        requested = set(self.measured_training_record_ids)
        missing = requested.difference(by_id)
        if missing:
            raise UgiJointLipidPriorError(
                f"production cache lacks {len(missing)} measured Ugi training products"
            )
        measured_indices = np.asarray(
            [by_id[record_id] for record_id in self.measured_training_record_ids],
            dtype=np.int64,
        )
        measured_index_set = set(int(index) for index in measured_indices)
        exploration_indices = np.asarray(
            [int(index) for index in target if int(index) not in measured_index_set],
            dtype=np.int64,
        )
        if self.exploration_mass > 0 and not len(exploration_indices):
            raise UgiJointLipidPriorError(
                "joint-lipid mixture requests exploration mass without nonmeasured support"
            )

        measure = np.zeros(len(cache), dtype=np.float64)
        measure[measured_indices] = self.measured_mass / len(measured_indices)
        if self.exploration_mass > 0:
            source = cache.arrays["source_weights"][exploration_indices].astype(np.float64)
            if (
                source.shape != (len(exploration_indices),)
                or not np.isfinite(source).all()
                or np.any(source <= 0)
                or source.sum() <= 0
            ):
                raise UgiJointLipidPriorError("nonmeasured Ugi exploration weights are invalid")
            measure[exploration_indices] = self.exploration_mass * source / source.sum()
        if (
            not np.isfinite(measure).all()
            or np.any(measure < 0)
            or not np.isclose(measure.sum(), 1.0)
        ):
            raise UgiJointLipidPriorError("joint-lipid training measure failed normalization")

        measured_ids_payload = "\n".join(self.measured_training_record_ids).encode("utf-8")
        receipt = {
            "schema_version": "forge.ugi_joint_lipid_training_measure.v1",
            "reaction_id": self.reaction_id,
            "training_fold": "train",
            "training_rows": int(len(target)),
            "measured_rows": int(len(measured_indices)),
            "exploration_rows": int(len(exploration_indices)),
            "requested_mass": {
                "measured": self.measured_mass,
                "exploration": self.exploration_mass,
            },
            "realized_mass": {
                "measured": float(measure[measured_indices].sum()),
                "exploration": float(measure[exploration_indices].sum()),
            },
            "measured_weighting": "uniform_unique_constitutional_product",
            "exploration_weighting": "frozen_source_weights",
            "assignments": {
                "path": str(self.assignments_path),
                "sha256": self.assignments_sha256,
            },
            "measured_record_ids_sha256": hashlib.sha256(measured_ids_payload).hexdigest(),
            "component_ids_enter_neural_tensors": False,
            "product_ids_enter_neural_tensors": False,
            "stored_graphs_enter_neural_tensors": False,
            "fragment_tokens_enter_neural_tensors": False,
        }
        return measure, receipt

    def to_mapping(self) -> dict[str, Any]:
        """Return the public policy without exposing training product identifiers."""

        payload = "\n".join(self.measured_training_record_ids).encode("utf-8")
        return {
            "schema_version": "forge.ugi_joint_lipid_prior.v1",
            "reaction_id": self.reaction_id,
            "measured_mass": self.measured_mass,
            "exploration_mass": self.exploration_mass,
            "assignments": {
                "path": str(self.assignments_path),
                "sha256": self.assignments_sha256,
            },
            "measured_training_products": len(self.measured_training_record_ids),
            "measured_record_ids_sha256": hashlib.sha256(payload).hexdigest(),
            "joint_unit": "complete_constitutional_ugi_product",
            "component_identity_conditioning": False,
            "component_graph_conditioning": False,
            "fragment_vocabulary_conditioning": False,
        }


__all__ = ["UgiJointLipidPrior", "UgiJointLipidPriorError"]

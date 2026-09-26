"""Bounded prepared-batch overlap with explicit committed sampler state."""

from __future__ import annotations

import copy
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from forge.core.hashing import resolve_pin
from forge.model.family_exposure import balanced_families

TAPE_SCHEMA = "forge.compose_lipid_presentation_tape.v1"


def load_presentation_tape(repo, value, *, config, data=None):
    """Authenticate the complete immutable schedule before allocating model state.

    Source indices denote admitted constitutional records, not source-row weights.
    With data present, verify every index's actual family and exclusion membership.
    """
    document = json.loads(resolve_pin(value, repo, label="presentation tape").read_text())
    runtime = config["runtime"]
    expected_inputs = {
        k: config["inputs"][k]
        for k in ("population", "verification", "measure", "mapped_cache")
        if k in config["inputs"]
    }
    families = document.get("families")
    if (
        document.get("schema_version") != TAPE_SCHEMA
        or document.get("inputs") != expected_inputs
        or not isinstance(families, list)
        or not families
        or not all(isinstance(f, str) and f for f in families)
        or families != sorted(set(families))
        or any(
            type(document.get(k)) is not int
            for k in ("optimizer_steps", "batch_size", "families_per_batch")
        )
        or document.get("optimizer_steps") != runtime["optimizer_steps"]
        or document.get("batch_size") != runtime["batch_size"]
        or document.get("families_per_batch") != runtime["families_per_batch"]
    ):
        raise ValueError("Presentation tape schema, source population or dimensions changed")
    path = resolve_pin(document["artifact"], repo, label="presentation indices")
    with np.load(path, allow_pickle=False) as arrays:
        indices = arrays[document["indices_key"]].copy()
        groups = arrays[document["families_key"]].copy()
    steps, size, per_batch = (
        runtime["optimizer_steps"],
        runtime["batch_size"],
        runtime["families_per_batch"],
    )
    if (
        indices.dtype.kind not in "iu"
        or groups.dtype.kind not in "iu"
        or indices.shape != (steps, size)
        or groups.shape != (steps, per_batch)
        or np.any(indices < 0)
        or np.any(indices > np.iinfo(np.int64).max)
        or np.any(groups < 0)
        or np.any(groups >= len(families))
        or any(len(set(row.tolist())) != per_batch for row in groups)
    ):
        raise ValueError("Presentation tape has invalid indices or family groups")
    indices, groups = indices.astype(np.int64), groups.astype(np.int64)
    per_family = size // per_batch
    exposure = np.zeros((steps + 1, len(families)), dtype=np.int64)
    for step, group in enumerate(groups):
        exposure[step + 1] = exposure[step]
        exposure[step + 1, group] += per_family
    if document.get("presentations_by_family") != dict(
        zip(families, exposure[-1].tolist(), strict=True)
    ):
        raise ValueError("Presentation tape exposure declaration changed")
    excluded = document.get("excluded_target_ids")
    if (
        not isinstance(excluded, list)
        or any(not isinstance(v, str) for v in excluded)
        or len(set(excluded)) != len(excluded)
    ):
        raise ValueError("Presentation tape exclusions must be explicit unique target IDs")
    if data is not None:
        actual_families = [
            r[0]
            for r in data._database.execute("SELECT DISTINCT family FROM weights ORDER BY family")
        ]
        if actual_families != families or np.any(indices >= len(data)):
            raise ValueError("Presentation tape references an unadmitted record/family")
        lookup = np.full(len(data), -1, dtype=np.int64)
        family_index = {f: i for i, f in enumerate(families)}
        blocked, excluded = [], set(excluded)
        for index, identity, family in data._database.execute(
            "SELECT record_index,target_id,family FROM weights ORDER BY record_index"
        ):
            lookup[index] = family_index[family]
            if identity in excluded:
                blocked.append(index)
        if np.any(lookup < 0) or np.isin(indices, blocked).any():
            raise ValueError("Presentation tape includes an excluded/unbound record")
        for row, group in zip(indices, groups, strict=True):
            observed = np.bincount(lookup[row], minlength=len(families))
            expected = np.zeros(len(families), dtype=np.int64)
            expected[group] = per_family
            if not np.array_equal(observed, expected):
                raise ValueError("Presentation tape source-family counts disagree")
    for array in (indices, groups, exposure):
        array.flags.writeable = False
    return {"document": document, "indices": indices, "families": groups, "exposure": exposure}


@dataclass(frozen=True)
class PreparedBatch:
    step: int
    indices: np.ndarray
    tensors: dict[str, torch.Tensor]
    sampler_state: dict[str, Any]


class PreparedBatches:
    """Workers only gather tensors; sampling stays ordered on the caller's thread.

    Prefetch never mutates the committed RNG. After a successful optimizer step the
    caller installs that ticket's sampler_state; restart discards all lookahead.
    """

    def __init__(
        self,
        data,
        cache,
        *,
        sampler_state,
        seed: int,
        start: int,
        stop: int,
        families: int,
        families_per_batch: int,
        batch_size: int,
        depth: int = 2,
        workers: int = 1,
        pin_memory: bool = False,
        presentation_indices: np.ndarray | None = None,
    ):
        if depth < 1 or workers < 1 or not 0 <= start <= stop:
            raise ValueError("Invalid bounded prefetch settings")
        self.data, self.cache = data, cache
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(sampler_state)
        self.seed, self.next_step, self.stop = seed, start, stop
        self.families, self.per_batch, self.batch_size = families, families_per_batch, batch_size
        self.depth, self.pin_memory = depth, pin_memory
        if presentation_indices is not None and (
            presentation_indices.dtype.kind not in "iu"
            or presentation_indices.shape != (stop, batch_size)
            or np.any(presentation_indices < 0)
        ):
            raise ValueError("Invalid explicit prefetch presentation indices")
        self.presentation_indices = presentation_indices
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.pending = deque()
        self.closed = False
        self._fill()

    def _gather(self, indices):
        batch = self.cache.batch(indices)
        return (
            {key: value.pin_memory() for key, value in batch.items()} if self.pin_memory else batch
        )

    def _fill(self):
        while self.next_step < self.stop and len(self.pending) < self.depth:
            if self.presentation_indices is None:
                selected = balanced_families(
                    families=self.families,
                    per_batch=self.per_batch,
                    step=self.next_step,
                    seed=self.seed,
                )
                indices = self.data.sample_indices(
                    self.batch_size,
                    self.rng,
                    families_per_batch=self.per_batch,
                    family_selection=selected,
                )
            else:
                indices = self.presentation_indices[self.next_step].copy()
            state = copy.deepcopy(self.rng.bit_generator.state)
            future = self.executor.submit(self._gather, indices)
            self.pending.append((self.next_step, indices, state, future))
            self.next_step += 1

    def __iter__(self):
        return self

    def __next__(self):
        if not self.pending or self.closed:
            raise StopIteration
        step, indices, state, future = self.pending.popleft()
        tensors = future.result()
        self._fill()
        return PreparedBatch(step, indices, tensors, state)

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class CudaBatches:
    """Overlap pinned H2D copies on a separate stream, retaining source ownership."""

    def __init__(self, batches, device):
        self.batches, self.device = iter(batches), torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.current = None
        self.retained = deque()
        self._stage()

    def _stage(self):
        try:
            source = next(self.batches)
        except StopIteration:
            self.current = None
            return
        if not all(value.is_pinned() for value in source.tensors.values()):
            raise ValueError("Asynchronous transfer requires pinned source tensors")
        with torch.cuda.stream(self.stream):
            tensors = {
                key: value.to(self.device, non_blocking=True)
                for key, value in source.tensors.items()
            }
            event = self.stream.record_event()
        self.current = (replace(source, tensors=tensors), source, event)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current is None:
            raise StopIteration
        ticket, source, event = self.current
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(event)
        for value in ticket.tensors.values():
            value.record_stream(stream)
        self.retained.append((source, event))
        while self.retained and self.retained[0][1].query():
            self.retained.popleft()
        self._stage()
        return ticket

    def close(self):
        self.stream.synchronize()
        self.current = None
        self.retained.clear()

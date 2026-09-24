"""Bounded prepared-batch overlap with explicit committed sampler state."""

from __future__ import annotations

import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from forge.model.family_exposure import balanced_families


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
    ):
        if depth < 1 or workers < 1 or not 0 <= start <= stop:
            raise ValueError("Invalid bounded prefetch settings")
        self.data, self.cache = data, cache
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(sampler_state)
        self.seed, self.next_step, self.stop = seed, start, stop
        self.families, self.per_batch, self.batch_size = families, families_per_batch, batch_size
        self.depth, self.pin_memory = depth, pin_memory
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

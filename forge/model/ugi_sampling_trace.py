"""Bounded, passive observation of the existing Ugi sampling implementation.

No scores, policy calls, candidate lists, or random draws are recomputed.  The
observer reads the locals at actual choices and returns.  Its process-global
instrumentation is deliberately restricted to one context and one thread.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import math
import sys
import textwrap
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from types import FrameType
from typing import Any

import numpy as np
import torch

from forge.model import synthesis_program_sampling as sampling
from forge.model import ugi_transformer_topology as topology


class SamplingTraceError(RuntimeError):
    """The requested observation cannot be represented completely and faithfully."""


_LOCK = threading.Lock()
_TOPOLOGY = ("_sample_amine_semantic_topology", "_sample_constructive_ester_offspring")
_TERMINAL = (
    "_ugi_ester_motif_constraints",
    "_select_ugi_amine_atom_states",
    "_select_ugi_all_role_tail_bonds",
)
_COORDINATE = ("_sample_allowed", "_sample_mog_chemistry_allowed")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise SamplingTraceError("trace payload contains a nonfinite score")
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _json_safe(item) for key, item in value.items()}
        # JSON object keys would otherwise destroy tuple-valued bond identities.
        return {"entries": [[_json_safe(key), _json_safe(item)] for key, item in value.items()]}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_json_safe(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True)
        )
    raise SamplingTraceError(f"unsupported trace payload type: {type(value).__name__}")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class _GeneratorProxy:
    def __init__(self, original: np.random.Generator, trace: SamplingTrace):
        self._original = original
        self._trace = trace

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def choice(self, *args: Any, **kwargs: Any) -> Any:
        frame = sys._getframe(1)
        selected = self._original.choice(*args, **kwargs)
        self._trace._numpy_choice(frame, args, kwargs, selected)
        return selected


class SamplingTrace:
    """Observe selected attempt indices without changing sampling or RNG state.

    ``events`` and ``summary`` are JSON serializable. ``max_candidates`` bounds
    each observed candidate list, and ``max_events`` bounds the complete trace.
    Exceeding either limit raises; no partial trace is admitted as complete.
    Standalone selector fixtures are assigned index zero with an explicit
    ``standalone_selector`` context. Production indices are ``offset + index``.

    Use the original sampler normally inside this context. Imported aliases
    work because observation binds code objects and actual RNG call sites.
    """

    def __init__(
        self,
        *,
        selected_attempts: Sequence[int] = tuple(range(16)),
        max_events: int = 20000,
        max_candidates: int = 50000,
        decisions: bool = True,
    ):
        attempts = tuple(selected_attempts)
        if (
            not attempts
            or any(type(value) is not int or value < 0 for value in attempts)
            or len(set(attempts)) != len(attempts)
        ):
            raise SamplingTraceError("selected_attempts must contain distinct nonnegative integers")
        if (
            type(max_events) is not int
            or max_events < 1
            or type(max_candidates) is not int
            or max_candidates < 1
        ):
            raise SamplingTraceError("trace bounds must be positive integers")
        self.selected_attempts = attempts
        self.max_events = max_events
        self.max_candidates = max_candidates
        if type(decisions) is not bool:
            raise SamplingTraceError("decisions must be boolean")
        self.decisions = decisions
        self.events: list[dict[str, Any]] = []
        self.endpoint_state: dict[str, Any] = {"sampler_returns": []}
        self._active = False
        self._entered = False
        self._status = "not_started"
        self._error: str | None = None
        self._codes: dict[Any, str] = {}
        self._coordinate_sites: dict[tuple[Any, int], str] = {}

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": "forge.ugi_sampling_trace.v1",
            "status": self._status,
            "error": self._error,
            "selected_attempts": list(self.selected_attempts),
            "max_events": self.max_events,
            "max_candidates": self.max_candidates,
            "decisions": self.decisions,
            "event_count": len(self.events),
            "event_counts": dict(sorted(Counter(event["kind"] for event in self.events).items())),
            "observed_attempt_indices": sorted(
                {event["context"]["attempt_index"] for event in self.events}
            ),
            "events_sha256": _digest(self.events),
            "endpoint_digest": self.endpoint_digest,
            "extra_model_calls": 0,
            "extra_random_draws": 0,
            "probability_scope": "Actual draw distributions; hierarchical tail draws are recorded separately, not converted into unobserved marginal probabilities.",
        }

    @property
    def endpoint_digest(self) -> str:
        return _digest(self.endpoint_state)

    def __enter__(self) -> SamplingTrace:
        if self._entered or not _LOCK.acquire(blocking=False):
            raise SamplingTraceError(
                "nested, concurrent, or reused SamplingTrace contexts are unsupported"
            )
        self._entered = True
        self._thread = threading.get_ident()
        self._profile_before = sys.getprofile()
        self._rng_before = np.random.default_rng
        self._multinomial_before = torch.multinomial
        try:
            expected = {
                **{name: topology for name in _TOPOLOGY},
                **{
                    name: sampling
                    for name in (
                        *_TERMINAL,
                        *_COORDINATE,
                        "_argmax_allowed",
                        "_strict_terminal_record",
                        "sample_synthesis_program_products",
                    )
                },
                "decode_ugi_exact_topology": topology,
            }
            required_locals = {
                **{
                    name: {"candidates", "candidate_scores", "probabilities", "selected"}
                    for name in _TOPOLOGY
                },
                "_ugi_ester_motif_constraints": {"candidates", "selected", "probabilities"},
                "_select_ugi_amine_atom_states": {"candidates", "choice", "probabilities"},
                "_select_ugi_all_role_tail_bonds": {
                    "candidates",
                    "selected",
                    "selected_forced",
                    "group_keys",
                    "group_probabilities",
                    "selected_indices",
                },
                "sample_synthesis_program_products": {"offset", "local", "index", "outputs"},
            }
            for name, module in expected.items():
                function = getattr(module, name, None)
                code = getattr(function, "__code__", None)
                if code is None or not required_locals.get(name, set()).issubset(
                    set(code.co_varnames) | set(code.co_cellvars)
                ):
                    raise SamplingTraceError(f"unsupported sampling implementation schema: {name}")
                self._codes[code] = name
            self._bind_coordinate_sites()
            self._active = True
            self._status = "recording"
            if self.decisions:
                np.random.default_rng = self._default_rng
                torch.multinomial = self._multinomial
            sys.setprofile(self._profile)
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self) -> None:
        sys.setprofile(self._profile_before)
        np.random.default_rng = self._rng_before
        torch.multinomial = self._multinomial_before
        self._active = False
        _LOCK.release()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self._restore()
        if exc is not None:
            self._status = "failed"
            self._error = f"{type(exc).__name__}: {exc}"
        elif self._error is not None:
            self._status = "failed"
            raise SamplingTraceError(self._error)
        else:
            self._status = "complete"
        return False

    def _fail(self, message: str) -> None:
        self._error = message
        raise SamplingTraceError(message)

    def _context(self, frame: FrameType) -> dict[str, Any]:
        context: dict[str, Any] = {"attempt_index": 0, "context_kind": "standalone_selector"}
        index = None
        cursor: FrameType | None = frame
        while cursor is not None:
            name = self._codes.get(cursor.f_code)
            values = cursor.f_locals
            if (
                name in (*_TERMINAL, "_strict_terminal_record", "decode_ugi_exact_topology")
                and index is None
            ):
                index = int(values["index"])
            record = (
                values.get("record")
                if name in (*_TERMINAL, "_strict_terminal_record", "decode_ugi_exact_topology")
                else None
            )
            if record is not None:
                context.setdefault("layout_record_id", record.graph.structure_id)
                context.setdefault("program_id", record.program_id)
            if name == "sample_synthesis_program_products":
                if index is None:
                    index = int(values.get("index", 0))
                context.update(
                    attempt_index=int(values.get("offset", 0)) + index,
                    context_kind="sampler_attempt",
                )
                break
            cursor = cursor.f_back
        name = self._codes.get(frame.f_code)
        if name == "_sample_amine_semantic_topology" or name == "_select_ugi_amine_atom_states":
            context["role"] = "amine_head"
        elif (
            name == "_sample_constructive_ester_offspring" or name == "_ugi_ester_motif_constraints"
        ):
            context["role"] = "oxoester_aldehyde_body_tail"
        elif name == "_select_ugi_all_role_tail_bonds" and "role" in frame.f_locals:
            context["role"] = frame.f_locals["role"]
        if name in (*_COORDINATE, "_argmax_allowed"):
            context.update(self._coordinate_context(frame))
        return context

    def _bind_coordinate_sites(self) -> None:
        """Bind the existing call expressions, avoiding stale loop locals as coordinates."""
        for name in ("_strict_terminal_record", "_select_ugi_amine_atom_states"):
            function = getattr(sampling, name)
            lines, start = inspect.getsourcelines(function)
            tree = ast.parse(textwrap.dedent("".join(lines)))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in (*_COORDINATE, "_argmax_allowed")
                ):
                    continue
                first = node.args[0]
                if isinstance(first, ast.Name):
                    field = {"node_logits": "nodes", "logits": "bond"}.get(first.id)
                else:
                    keys = {
                        item.value
                        for item in ast.walk(first)
                        if isinstance(item, ast.Constant) and isinstance(item.value, str)
                    }
                    field = next(iter(keys)) if keys in ({"nodes"}, {"parents"}) else None
                if field is None:
                    self._fail(f"unsupported coordinate call expression in {name}")
                for line in range(start + node.lineno - 1, start + node.end_lineno):
                    self._coordinate_sites[(function.__code__, line)] = field

    def _coordinate_context(self, frame: FrameType) -> dict[str, Any]:
        caller = frame.f_back
        if caller is None or self._codes.get(caller.f_code) not in (
            "_strict_terminal_record",
            "_select_ugi_amine_atom_states",
        ):
            return {"selection_stage": "standalone_coordinate"}
        values = caller.f_locals
        field = self._coordinate_sites.get((caller.f_code, caller.f_lineno))
        if field is None:
            self._fail("unrecognized coordinate selector call site")
        context: dict[str, Any] = {"caller": self._codes[caller.f_code]}
        if self._codes[caller.f_code] == "_select_ugi_amine_atom_states":
            context.update(
                selection_stage="candidate_internal_atom_assignment",
                field="nodes",
                node=int(values["node"]),
                role="amine_head",
                requested_symbol=values["symbol"],
                candidate_symbol_assignment_sha256=_digest(_json_safe(values["symbols_by_node"])),
            )
        elif field == "bond":
            left, right = int(values["left"]), int(values["right"])
            context.update(
                selection_stage="terminal_coordinate",
                field=f"{values['kind']}_bonds",
                slot=int(values["slot"]),
                edge=[left, right],
                roles=[values["role_names"][left], values["role_names"][right]],
                forced_bond=values.get("forced_bond"),
            )
        else:
            coordinate = "node" if field == "nodes" else "child"
            position = int(values[coordinate])
            context.update(
                selection_stage="terminal_coordinate",
                field=field,
                **{coordinate: position},
                role=values["role_names"][position],
            )
        return context

    def _selected(self, frame: FrameType) -> bool:
        if threading.get_ident() != self._thread:
            self._fail("concurrent sampling while SamplingTrace is active is unsupported")
        return self._context(frame)["attempt_index"] in self.selected_attempts

    def _append(self, frame: FrameType, kind: str, **payload: Any) -> None:
        context = payload.pop("context", None) or self._context(frame)
        if context["attempt_index"] not in self.selected_attempts:
            return
        if len(self.events) >= self.max_events:
            self._fail(f"trace exceeded max_events={self.max_events}; no events were truncated")
        event = {
            "event_index": len(self.events),
            "kind": kind,
            "selector": self._codes[frame.f_code],
            "context": context,
            **payload,
        }
        try:
            self.events.append(_json_safe(event))
        except SamplingTraceError as error:
            self._error = str(error)
            raise

    def _candidates(self, frame: FrameType) -> list[dict[str, Any]]:
        values = frame.f_locals
        candidates = values.get("candidates", ())
        if len(candidates) > self.max_candidates:
            self._fail(
                f"{frame.f_code.co_name} has {len(candidates)} candidates, exceeding max_candidates={self.max_candidates}"
            )
        name = self._codes[frame.f_code]
        result = []
        for index, candidate in enumerate(candidates):
            if name in _TOPOLOGY:
                row = {
                    "identity": candidate,
                    "neural_score": values["candidate_scores"][index],
                    "semantic_distance": values.get("semantic_distances", [None] * len(candidates))[
                        index
                    ],
                }
                for key, source in (
                    ("joint_realism_score", "joint_realism_scores"),
                    ("local_chemistry_score", "topology_support_scores"),
                ):
                    vector = values.get(source)
                    row[key] = None if vector is None or len(vector) == 0 else vector[index]
                if "oriented_edges" in values:
                    row["oriented_global_closures"] = values["oriented_edges"][index]
            elif name in _TERMINAL:
                if len(candidate) != 6:
                    self._fail(f"unsupported candidate tuple schema for {name}")
                if name == "_select_ugi_all_role_tail_bonds":
                    joint = values.get("joint_scores")
                    row = {
                        "identity": candidate[5],
                        "neural_score": candidate[0],
                        "semantic_distance": candidate[1],
                        "local_chemistry_score": candidate[2],
                        "joint_realism_score": None if joint is None else joint[index],
                        "unsaturation_counts": candidate[3:5],
                    }
                else:
                    row = {
                        "identity": (
                            candidate[4:]
                            if name == "_ugi_ester_motif_constraints"
                            else candidate[5]
                        ),
                        "neural_score": candidate[0],
                        "semantic_distance": candidate[1],
                        "joint_realism_score": candidate[2],
                        "local_chemistry_score": candidate[3],
                    }
                    if name == "_select_ugi_amine_atom_states":
                        row["arrangement_label"] = candidate[4]
            else:
                self._fail(f"unsupported candidate selector: {name}")
            row = _json_safe(row)
            result.append(
                {"candidate_index": index, "identity_sha256": _digest(row["identity"]), **row}
            )
        return result

    def _default_rng(self, *args: Any, **kwargs: Any) -> Any:
        generator = self._rng_before(*args, **kwargs)
        frame = sys._getframe(1)
        if self._active and self._codes.get(frame.f_code) == "sample_synthesis_program_products":
            if threading.get_ident() != self._thread:
                self._fail("concurrent sampling while SamplingTrace is active is unsupported")
            return _GeneratorProxy(generator, self)
        return generator

    def observe_generator(self, generator: np.random.Generator) -> Any:
        """Wrap an existing generator for standalone selector identity fixtures."""
        if not self._active or not isinstance(generator, np.random.Generator):
            raise SamplingTraceError(
                "observe_generator requires an active trace and numpy Generator"
            )
        return _GeneratorProxy(generator, self)

    def _multinomial(self, *args: Any, **kwargs: Any) -> Any:
        frame = sys._getframe(1)
        selected = self._multinomial_before(*args, **kwargs)
        if self._active and self._codes.get(frame.f_code) in _TOPOLOGY and self._selected(frame):
            if selected.numel() != 1:
                self._fail("unsupported topology draw shape")
            probabilities = args[0] if args else kwargs["input"]
            candidates = self._candidates(frame)
            self._append(
                frame,
                "categorical_choice",
                distribution_scope="complete_legal_candidates",
                candidates=candidates,
                final_probabilities=probabilities,
                selected_candidate_index=int(selected.item()),
                selection_law="torch_multinomial",
            )
        return selected

    def _numpy_choice(
        self, frame: FrameType, args: tuple[Any, ...], kwargs: dict[str, Any], selected: Any
    ) -> None:
        name = self._codes.get(frame.f_code)
        if not self._active or name not in (*_TERMINAL, *_COORDINATE) or not self._selected(frame):
            return
        values = frame.f_locals
        population = args[0] if args else kwargs["a"]
        probabilities = kwargs.get("p", args[3] if len(args) > 3 else None)
        if probabilities is None or np.asarray(selected).ndim != 0:
            self._fail(f"unsupported numpy choice schema in {name}")
        if name in _COORDINATE:
            allowed = np.asarray(population).tolist()
            if len(allowed) > self.max_candidates:
                self._fail("coordinate state count exceeds max_candidates")
            self._append(
                frame,
                "coordinate_choice",
                allowed_states=allowed,
                neural_scores=np.asarray(values["logits"])[allowed],
                local_chemistry_scores=values.get("chemistry_scores"),
                final_probabilities=probabilities,
                selected_state=int(selected),
                selection_law="numpy_choice",
            )
            return
        candidates = self._candidates(frame)
        if name == "_select_ugi_all_role_tail_bonds" and probabilities is values.get(
            "group_probabilities"
        ):
            keys = values["group_keys"]
            self._append(
                frame,
                "tail_count_group_choice",
                candidates=candidates,
                group_keys=keys,
                group_model_scores=values["group_model_scores"],
                group_semantic_distances=values["group_semantic_distances"],
                group_local_scores=values["group_local_scores"],
                group_joint_scores=values["group_joint_scores"],
                final_probabilities=probabilities,
                selected_group_index=int(selected),
                selected_group=keys[int(selected)],
                count_strategy=values["semantic_guidance_policy"].tail_unsaturation_count_strategy,
                selection_law="numpy_choice",
                distribution_scope="unsaturation_count_groups",
            )
            return
        if name == "_select_ugi_all_role_tail_bonds" and probabilities is values.get(
            "pattern_probabilities"
        ):
            keys = values["pattern_keys"]
            self._append(
                frame,
                "tail_pattern_group_choice",
                candidates=candidates,
                pattern_keys=keys,
                final_probabilities=probabilities,
                selected_group_index=int(selected),
                selected_group=keys[int(selected)],
                selection_law="numpy_choice",
                distribution_scope="terminal_pattern_groups",
            )
            return
        indices = (
            list(range(int(population)))
            if np.isscalar(population)
            else [int(value) for value in population]
        )
        self._append(
            frame,
            "categorical_choice",
            candidates=candidates,
            draw_candidate_indices=indices,
            final_probabilities=probabilities,
            selected_candidate_index=int(selected),
            selected_group=(
                values.get("selected_group") if name == "_select_ugi_all_role_tail_bonds" else None
            ),
            distribution_scope=(
                "complete_legal_candidates"
                if len(indices) == len(candidates)
                else "conditional_group_candidates"
            ),
            selection_law="numpy_choice",
        )

    def _profile(self, frame: FrameType, event: str, arg: Any) -> None:
        if self._profile_before is not None:
            self._profile_before(frame, event, arg)
        name = self._codes.get(frame.f_code)
        if not self._active or name is None:
            return
        if event == "return" and name == "sample_synthesis_program_products" and arg is not None:
            self._capture_endpoint(frame, arg)
            return
        if not self.decisions or not self._selected(frame):
            return
        values = frame.f_locals
        if (
            event == "c_call"
            and name == "_select_ugi_all_role_tail_bonds"
            and getattr(arg, "__name__", None) == "update"
            and getattr(arg, "__self__", None) is values.get("selected_forced")
        ):
            self._terminal_decision(frame, "selected")
        elif event == "return":
            if (
                name in ("_ugi_ester_motif_constraints", "_select_ugi_amine_atom_states")
                and arg is not None
            ):
                key = "choice" if name == "_select_ugi_amine_atom_states" else "selected"
                if key in values and values.get("candidates"):
                    self._terminal_decision(frame, key)
            if name in (
                *_TOPOLOGY,
                *_TERMINAL,
                "_strict_terminal_record",
                "decode_ugi_exact_topology",
            ):
                self._append(
                    frame,
                    "selector_outcome",
                    returned=arg,
                    filter_stage_counts=values.get("stages"),
                    topology_only=values.get("topology_only"),
                )
            elif name == "_argmax_allowed":
                valid = values["valid"]
                allowed = np.flatnonzero(valid)
                context = self._context(frame)
                self._append(
                    frame,
                    (
                        "candidate_internal_atom_argmax"
                        if context.get("selection_stage") == "candidate_internal_atom_assignment"
                        else "coordinate_argmax"
                    ),
                    context=context,
                    allowed_states=allowed,
                    neural_scores=values["logits"][allowed],
                    selected_state=arg,
                    final_probabilities=(
                        None if arg is None else [float(state == arg) for state in allowed]
                    ),
                    selection_law="first_neural_argmax",
                )

    def _capture_endpoint(self, frame: FrameType, returned: Any) -> None:
        if threading.get_ident() != self._thread:
            self._fail("concurrent sampling while SamplingTrace is active is unsupported")
        values = frame.f_locals
        if "terminal_cpu" not in values or "local" not in values:
            self._fail("sampler endpoint schema lacks final terminal state")
        generators = {}
        for field in (
            "generator",
            "topology_generator",
            "topology_conditioned_chemistry_generator",
            "terminal_generator",
        ):
            generator = values.get(field)
            if generator is None:
                generators[field] = None
            elif isinstance(generator, torch.Generator):
                generators[field] = {
                    "kind": "torch",
                    "device": str(generator.device),
                    "state_hex": bytes(generator.get_state().cpu().tolist()).hex(),
                }
            else:
                generators[field] = {"kind": "numpy", "state": generator.bit_generator.state}
        snapshot = {
            "terminal_state_scope": "final_batch",
            "batch_offset": int(values["offset"]),
            "batch_record_ids": [record.graph.structure_id for record in values["local"]],
            "total_attempts": len(returned[0]),
            "terminal_states": values["terminal_cpu"],
            "generator_states": generators,
        }
        self.endpoint_state["sampler_returns"].append(_json_safe(snapshot))
        if self.decisions:
            for row in returned[0]:
                if row["sample_index"] in self.selected_attempts:
                    context = {
                        "attempt_index": row["sample_index"],
                        "context_kind": "sampler_attempt",
                    }
                    self._append(frame, "attempt_outcome", context=context, output=row)

    def _terminal_decision(self, frame: FrameType, key: str) -> None:
        values = frame.f_locals
        policy = values.get("semantic_guidance_policy")
        ranked = policy is not None and (
            policy.uses_ranked_terminal_bonds
            if self._codes[frame.f_code] == "_select_ugi_all_role_tail_bonds"
            else policy.uses_ranked_terminal_chemistry
        )
        candidates = self._candidates(frame)
        selected = int(values[key])
        probabilities = (
            None if ranked else [float(index == selected) for index in range(len(candidates))]
        )
        self._append(
            frame,
            "terminal_assignment",
            candidates=candidates,
            selected_candidate_index=selected,
            final_probabilities=probabilities,
            selection_law="previous_recorded_numpy_choices" if ranked else "first_neural_argmax",
            distribution_scope="see_actual_draw_events" if ranked else "complete_legal_candidates",
        )

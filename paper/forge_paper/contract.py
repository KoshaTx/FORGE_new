"""Strict, hash-pinned contract for one manuscript and its computational evidence roots."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.hashing import is_sha256

PAPER_CONTRACT_SCHEMA = "forge.paper_reproduction.v1"


class PaperContractError(ValueError):
    """The paper contract is malformed, incomplete, or internally inconsistent."""


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PaperContractError(f"{label} must be a non-empty repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise PaperContractError(f"{label} must stay inside the repository: {value!r}")
    return path.as_posix()


@dataclass(frozen=True)
class PaperPin:
    path: str
    sha256: str
    producer_id: str | None = None

    @classmethod
    def from_mapping(cls, value: object, *, label: str, producer: bool = False) -> PaperPin:
        fields = {"path", "sha256", "producer_id"} if producer else {"path", "sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise PaperContractError(f"{label} must define exactly {sorted(fields)}")
        path = _relative_path(value["path"], label=f"{label}.path")
        digest = value["sha256"]
        if not is_sha256(digest):
            raise PaperContractError(f"{label}.sha256 is not a lowercase SHA-256 digest")
        producer_id = value.get("producer_id")
        if producer and (not isinstance(producer_id, str) or not producer_id):
            raise PaperContractError(f"{label}.producer_id must be a non-empty string")
        return cls(path=path, sha256=digest, producer_id=producer_id)


@dataclass(frozen=True)
class PublicationProducer:
    producer_id: str
    command: tuple[str, ...]
    outputs: tuple[str, ...]
    tools: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> PublicationProducer:
        fields = {"id", "command", "outputs", "tools"}
        if not isinstance(value, dict) or set(value) != fields:
            raise PaperContractError(f"{label} must define exactly {sorted(fields)}")
        producer_id = value["id"]
        command = value["command"]
        outputs = value["outputs"]
        tools = value["tools"]
        if not isinstance(producer_id, str) or not producer_id:
            raise PaperContractError(f"{label}.id must be a non-empty string")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise PaperContractError(f"{label}.command must be a non-empty string array")
        if not isinstance(outputs, list) or not outputs:
            raise PaperContractError(f"{label}.outputs must be a non-empty path array")
        if not isinstance(tools, list) or not all(isinstance(item, str) and item for item in tools):
            raise PaperContractError(f"{label}.tools must be a string array")
        return cls(
            producer_id=producer_id,
            command=tuple(command),
            outputs=tuple(
                _relative_path(item, label=f"{label}.outputs[{index}]")
                for index, item in enumerate(outputs)
            ),
            tools=tuple(tools),
        )


@dataclass(frozen=True)
class PaperContract:
    paper_id: str
    source: PaperPin
    evidence_roots: tuple[PaperPin, ...]
    generated_outputs: tuple[PaperPin, ...]
    figure_outputs: tuple[PaperPin, ...]
    publication_producers: tuple[PublicationProducer, ...]
    numerical_entrypoints: tuple[str, ...]
    registered_experiments: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> PaperContract:
        try:
            document: Any = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PaperContractError(
                f"paper contract could not be read: {path}: {error}"
            ) from error
        fields = {
            "schema_version",
            "paper_id",
            "source",
            "evidence_roots",
            "generated_outputs",
            "figure_outputs",
            "publication_producers",
            "numerical_entrypoints",
            "registered_experiments",
        }
        if not isinstance(document, dict) or set(document) != fields:
            raise PaperContractError(f"paper contract must define exactly {sorted(fields)}")
        if document["schema_version"] != PAPER_CONTRACT_SCHEMA:
            raise PaperContractError(
                f"unsupported paper contract schema: {document['schema_version']!r}"
            )
        paper_id = document["paper_id"]
        if not isinstance(paper_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", paper_id):
            raise PaperContractError("paper_id must contain lowercase letters, numbers, or hyphens")

        def pins(name: str, *, producer: bool = False) -> tuple[PaperPin, ...]:
            values = document[name]
            if not isinstance(values, list) or not values:
                raise PaperContractError(f"{name} must be a non-empty array")
            return tuple(
                PaperPin.from_mapping(item, label=f"{name}[{index}]", producer=producer)
                for index, item in enumerate(values)
            )

        producers_raw = document["publication_producers"]
        if not isinstance(producers_raw, list) or not producers_raw:
            raise PaperContractError("publication_producers must be a non-empty array")
        producers = tuple(
            PublicationProducer.from_mapping(item, label=f"publication_producers[{index}]")
            for index, item in enumerate(producers_raw)
        )
        if len({item.producer_id for item in producers}) != len(producers):
            raise PaperContractError("publication producer ids must be unique")

        numerical = document["numerical_entrypoints"]
        if not isinstance(numerical, list) or not numerical:
            raise PaperContractError("numerical_entrypoints must be a non-empty path array")
        experiments = document["registered_experiments"]
        if (
            not isinstance(experiments, list)
            or not experiments
            or not all(isinstance(item, str) and item for item in experiments)
        ):
            raise PaperContractError("registered_experiments must be a non-empty string array")

        contract = cls(
            paper_id=paper_id,
            source=PaperPin.from_mapping(document["source"], label="source"),
            evidence_roots=pins("evidence_roots"),
            generated_outputs=pins("generated_outputs", producer=True),
            figure_outputs=pins("figure_outputs", producer=True),
            publication_producers=producers,
            numerical_entrypoints=tuple(
                _relative_path(item, label=f"numerical_entrypoints[{index}]")
                for index, item in enumerate(numerical)
            ),
            registered_experiments=tuple(experiments),
        )
        contract._validate_links()
        return contract

    def _validate_links(self) -> None:
        all_pins = (
            self.source,
            *self.evidence_roots,
            *self.generated_outputs,
            *self.figure_outputs,
        )
        paths = [pin.path for pin in all_pins]
        if len(set(paths)) != len(paths):
            raise PaperContractError("paper contract paths must be unique")
        producer_by_id = {item.producer_id: item for item in self.publication_producers}
        generated_by_path = {pin.path: pin for pin in self.generated_outputs}
        figures_by_path = {pin.path: pin for pin in self.figure_outputs}
        declared_outputs = generated_by_path | figures_by_path
        for producer in self.publication_producers:
            for output in producer.outputs:
                pin = declared_outputs.get(output)
                if pin is None:
                    raise PaperContractError(
                        f"producer {producer.producer_id!r} has undeclared output {output!r}"
                    )
                if pin.producer_id != producer.producer_id:
                    raise PaperContractError(
                        f"output {output!r} names producer {pin.producer_id!r}, "
                        f"not {producer.producer_id!r}"
                    )
        # Chemist-packet pages are deliberately frozen-only: regenerating them also rewrites the
        # internal blind key. Every other output must have a runnable publication producer.
        for pin in declared_outputs.values():
            if pin.producer_id != "chemist-packet" and pin.producer_id not in producer_by_id:
                raise PaperContractError(
                    f"output {pin.path!r} names unknown producer {pin.producer_id!r}"
                )

    @property
    def direct_pins(self) -> tuple[PaperPin, ...]:
        return (
            self.source,
            *self.evidence_roots,
            *self.generated_outputs,
            *self.figure_outputs,
        )


__all__ = [
    "PAPER_CONTRACT_SCHEMA",
    "PaperContract",
    "PaperContractError",
    "PaperPin",
    "PublicationProducer",
]

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class InvalidStepLimitError(ValueError):
    pass


class RunStatus(str, Enum):
    COMPLETED = "completed"
    LIMIT_REACHED = "limit_reached"


def validate_max_steps(max_steps: Any) -> int:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise InvalidStepLimitError("max_steps must be a positive integer")
    return max_steps


@dataclass(frozen=True)
class UserMessage:
    content: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantMessage:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    content: str


@dataclass(frozen=True)
class SearchEvidence:
    query: str
    chunks: tuple[EvidenceChunk, ...]


@dataclass(frozen=True)
class ToolResultMessage:
    tool_call_id: str
    name: str
    evidence: SearchEvidence | None
    error: str | None

    @classmethod
    def from_evidence(cls, tool_call: ToolCall, evidence: SearchEvidence) -> ToolResultMessage:
        return cls(tool_call.id, tool_call.name, evidence, None)

    @classmethod
    def invalid(cls, tool_call: ToolCall) -> ToolResultMessage:
        return cls(tool_call.id, tool_call.name, None, "invalid_search_call")


@dataclass(frozen=True)
class EvidenceTrace:
    query: str
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalRunResult:
    status: RunStatus
    answer: str | None
    evidence: tuple[EvidenceTrace, ...]
    messages: tuple[Any, ...]
    model_steps: int

    @classmethod
    def completed(cls, answer, evidence, messages, model_steps) -> RetrievalRunResult:
        return cls(RunStatus.COMPLETED, answer, tuple(evidence), tuple(messages), model_steps)

    @classmethod
    def limit_reached(cls, evidence, messages, model_steps) -> RetrievalRunResult:
        return cls(RunStatus.LIMIT_REACHED, None, tuple(evidence), tuple(messages), model_steps)

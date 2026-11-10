"""Retrieval as a Tool Loop.

A policy assistant may search a supplied local handbook when its model requests
evidence. Complete only search_handbook, execute_search_call, and
run_retrieval_loop.

The deterministic retriever, message/result types, tool schema, Groq-compatible
model client, and limits are supplied. Pytest injects a mocked model and local
retriever, so no API key or network access is required.

Required behavior:
- search_handbook passes the exact query to HANDBOOK_RETRIEVER, preserves result
  order and chunk IDs, and truncates each content field to CONTENT_LIMIT.
- execute_search_call accepts only a search_handbook call with a nonblank string
  query. It always preserves the call ID. Invalid calls return the documented
  error ToolResultMessage without searching.
- run_retrieval_loop validates max_steps as a positive non-boolean integer,
  starts with one UserMessage, calls model.complete(messages, tools), appends
  assistant messages and tool results in order, and supports later refined
  searches.
- A nonblank assistant response without tool calls terminates successfully.
- The evidence trace preserves search order, queries, and retrieved chunk IDs.
- Reaching max_steps returns LIMIT_REACHED instead of looping forever.

Not required: building an index, web search, real model calls during evaluation,
prompt grading, answer-style grading, a UI, or deployment.
"""

from __future__ import annotations

from utils.handbook import CONTENT_LIMIT, HANDBOOK_RETRIEVER
from utils.models import (
    AssistantMessage,
    EvidenceChunk,
    EvidenceTrace,
    RetrievalRunResult,
    SearchEvidence,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    validate_max_steps,
)
from utils.tool_schema import SEARCH_TOOL_SCHEMA


def search_handbook(query: str) -> SearchEvidence:
    documents = HANDBOOK_RETRIEVER.invoke(query)
    chunks = tuple(
        EvidenceChunk(doc.chunk_id, doc.content[:CONTENT_LIMIT]) for doc in documents
    )
    return SearchEvidence(query, chunks)


def execute_search_call(tool_call: ToolCall) -> ToolResultMessage:
    if tool_call.name != "search_handbook" or not isinstance(tool_call.arguments, dict):
        return ToolResultMessage.invalid(tool_call)
    query = tool_call.arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return ToolResultMessage.invalid(tool_call)
    return ToolResultMessage.from_evidence(tool_call, search_handbook(query))


def run_retrieval_loop(model, user_message: str, max_steps: int) -> RetrievalRunResult:
    validate_max_steps(max_steps)
    messages: list = [UserMessage(user_message)]
    trace: list[EvidenceTrace] = []
    tools = (SEARCH_TOOL_SCHEMA,)

    for step in range(1, max_steps + 1):
        response: AssistantMessage = model.complete(tuple(messages), tools)
        messages.append(response)

        if response.tool_calls:
            for call in response.tool_calls:
                result = execute_search_call(call)
                messages.append(result)
                if result.evidence is not None:
                    trace.append(
                        EvidenceTrace(
                            result.evidence.query,
                            tuple(c.chunk_id for c in result.evidence.chunks),
                        )
                    )
        elif isinstance(response.content, str) and response.content.strip():
            return RetrievalRunResult.completed(
                response.content, tuple(trace), tuple(messages), step
            )

    return RetrievalRunResult.limit_reached(tuple(trace), tuple(messages), max_steps)

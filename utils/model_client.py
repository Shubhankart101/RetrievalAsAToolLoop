from __future__ import annotations

import json
import os

from utils.models import AssistantMessage, ToolCall, ToolResultMessage, UserMessage

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment]


class GroqModel:
    """Real application model, replaced by a mock during pytest."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is required to call Groq.")
        if Groq is None:
            raise ImportError("Install groq before running the application.")
        self.client = Groq(api_key=key)

    def complete(self, messages, tools) -> AssistantMessage:
        payload = []
        for message in messages:
            if isinstance(message, UserMessage):
                payload.append({"role": "user", "content": message.content})
            elif isinstance(message, ToolResultMessage):
                payload.append({"role": "tool", "tool_call_id": message.tool_call_id, "content": json.dumps(message, default=lambda value: value.__dict__)})
            elif isinstance(message, AssistantMessage):
                payload.append({"role": "assistant", "content": message.content})
        response = self.client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=payload,
            tools=list(tools),
            tool_choice="auto",
        )
        raw = response.choices[0].message
        calls = tuple(
            ToolCall(call.id, call.function.name, json.loads(call.function.arguments))
            for call in list(raw.tool_calls or [])
        )
        return AssistantMessage(content=raw.content or "", tool_calls=calls)

from __future__ import annotations

import re
from dataclasses import dataclass


CONTENT_LIMIT = 120


@dataclass(frozen=True)
class HandbookDocument:
    chunk_id: str
    content: str


DOCUMENTS = (
    HandbookDocument("leave-01", "Planned leave requests must be submitted five working days in advance."),
    HandbookDocument("leave-02", "Medical leave may be submitted later when supporting documentation is provided."),
    HandbookDocument("security-01", "Security incidents must be reported to the response desk immediately."),
)


class LocalHandbookRetriever:
    def invoke(self, query: str) -> list[HandbookDocument]:
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        ranked = []
        for index, document in enumerate(DOCUMENTS):
            score = len(tokens & set(re.findall(r"[a-z0-9]+", document.content.lower())))
            if score:
                ranked.append((-score, index, document))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in ranked[:2]]


HANDBOOK_RETRIEVER = LocalHandbookRetriever()

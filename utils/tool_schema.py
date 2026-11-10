SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_handbook",
        "description": "Search the supplied local policy handbook.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

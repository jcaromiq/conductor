Added an experimental ``pi`` provider that executes every agent step inside
the user's own pi coding agent as a one-shot ``--mode json`` subprocess.
Model selection is delegated to pi: ``--model`` is only passed when the
workflow or agent pins one (otherwise pi's own default applies), and the
actual model used is read back from the final assistant message. MCP
servers come from pi's own configuration (``~/.pi/agent/mcp.json`` plus
project ``.pi/mcp.json`` after ``--approve``); workflow ``mcp_servers``
still serve ``type: mcp`` steps but are not injected into pi sessions.
Structured output uses prompt injection with the shared parse-recovery
loop, and per-agent tool allowlists are refused (``tools: []`` maps to
``--no-tools``).

# Pi Provider Documentation

> **Experimental Provider** — the pi provider runs each agent step inside the
> user's own [`pi`](https://pi.dev) coding agent. It exists in personal/
> organizational forks of Conductor (upstream has no `pi` provider). See
> [Experimental Providers](./experimental.md) for the stability policy.

The pi provider delegates the *agentic loop* — model calls, tools, MCP
servers, reasoning — to pi, a coding-agent harness the user has already
configured. Conductor keeps doing what it is good at: routing, gates,
tickets, budgets, resume, and metrics. Each `execute()` spawns one
one-shot `pi --mode json --print --no-session` subprocess in the step's
working directory, pipes the rendered prompt through stdin, relays pi's
NDJSON event stream as Conductor events, and normalizes the final
assistant message into `AgentOutput`.

## Model routing is delegated to pi

- Agent-level `model:` and workflow-level `default_model:` are forwarded to
  pi as `--model`, which pi resolves against its own model catalog.
- When **neither** is set, nothing is passed: pi uses its own configured
  default (`defaultModel` in `~/.pi/agent/settings.json`). The same workflow
  therefore runs on whatever providers/models each user's pi is set up with
  (Forge, Antigravity, GitHub Copilot, …).
- `reasoning.effort` maps to pi's `--thinking` (same `low`–`max` vocabulary).
- The **actual** model used (which may be pi's default) is read back from
  the final assistant message and reported in `AgentOutput.model`, the
  dashboard, and token metrics.

## Quick Start

```bash
# pi must be installed and logged in (`pi auth login` / `pi auth login forge`)
conductor run workflow.yaml -i ticket=BLAU-1234   # with workflow provider: pi
```

```yaml
workflow:
  name: my-flow
  runtime:
    provider: pi
    # default_model: glm-5-3   # optional; omit to use pi's own default
```

## MCP

pi connects MCP servers from **its own configuration**:

- user-level: `~/.pi/agent/mcp.json`
- project-level: `.pi/mcp.json` in the step's working directory (loaded
  after project trust — the provider passes `--approve`, see below).

Conductor's `runtime.mcp_servers` keep working for `type: mcp` steps
(those run through the engine's own MCP client regardless of provider),
but they are **not injected into pi sessions**: the servers, transports and
tool allowlists a workflow declares cannot be honored faithfully by a
harness that owns its server set. If your LLM agents need a workflow's
MCP tools, declare those servers in pi's own `mcp.json` as well.

## Project trust (`--approve`)

By default the provider passes pi's `--approve` flag: project-local files —
the worktree's `AGENTS.md`, `.agents/skills` found in the working directory
or an ancestor (note `~/.agents/skills`), `.pi/` resources — load without an
interactive trust decision, which a one-shot subprocess could never answer.
Construct `PiProvider(approve_project=False)` to disable it.

## Limitations

- **No per-agent tool allowlists** (`workflow_tools_passthrough=False`).
  `tools: []` maps to pi's `--no-tools`; a non-empty list fails fast.
- **No structured provider settings** (`base_url`/`api_key`): pi owns its
  endpoints and credentials.
- **No checkpoint resume / session continuity**: each execution is a fresh
  subprocess with `--no-session` (pi sessions are not persisted).
- **Interrupt** kills the subprocess; no partial output is returned.
- **No skills/plugins forwarding**: the executor eager-injects declared
  skill content into the prompt (pi's own skill discovery still applies to
  the working directory).

## Troubleshooting

- `pi provider requires the pi CLI on PATH` — install pi; `which pi`.
- Agent answers look empty / `returned no final assistant message` —
  inspect `raw_response.final_message.stopReason` in the step result; pi
  reports `error`/`aborted` stop reasons as `ProviderError`.
- Wrong model used — the agent or workflow pinned one; pi's default only
  applies when *nothing* is declared in the workflow.

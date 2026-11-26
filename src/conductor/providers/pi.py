"""Pi coding-agent provider (subprocess, JSON event stream).

This module implements ``PiProvider`` — a host-side transport that executes
each Conductor agent step inside the user's ``pi`` coding agent
(https://pi.dev), spawned as a one-shot subprocess in JSON mode.

Architecture
------------
Conductor owns orchestration (routing, gates, tickets, budgets, resume);
the *agentic loop* (model calls, tools, MCP, skills) runs inside pi:

- ``pi --mode json --print --no-session`` per execution: the rendered prompt
  enters through piped stdin (prepended to pi's first prompt), pi's NDJSON
  event stream is read from stdout, and pi's own tools operate on the step's
  working directory.
- Model selection is **delegated to pi**. ``--model`` is passed only when
  the workflow or agent pins one; otherwise pi uses its own configured
  default (``defaultModel`` in ``~/.pi/agent/settings.json``), so the same
  fork works for any user and their own pi setup (Forge, Antigravity,
  GitHub Copilot, ...). The actual model used is read back from the final
  assistant message.
- MCP is likewise pi's own: pi connects the servers declared in its
  user-level and project-level config. Conductor's ``runtime.mcp_servers``
  still work for ``type: mcp`` steps (executed by the engine's own MCP
  client) but are **not injected into pi sessions** — workflow-declared
  server names, transport bridges and per-server tool allowlists cannot be
  honored faithfully by a harness that owns its server set, so the provider
  does not claim them.

Error Handling Strategy:
- ValidationError: Invalid inputs or output schema violations.
- ProviderError: subprocess failures, non-zero exits, interrupted or timed
  out executions.

Capabilities honesty (``docs/development/providers.md``):
- ``mcp_tools=True`` describes pi's MCP support, honored at the harness
  level (servers from pi's own config), not by forwarding Conductor's.
- ``workflow_tools_passthrough=False``: Conductor's workflow tool names do
  not map to pi tool names, so a non-empty per-agent allowlist is refused
  rather than silently dropped. ``tools: []`` maps to pi's ``--no-tools``.
- ``skills=False``: the executor eager-injects declared skill content into
  the rendered prompt. (pi additionally discovers ``.agents/skills`` from
  the working directory and its ancestors when project trust is granted —
  see ``approve_project`` in the constructor.)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from conductor.exceptions import ProviderError, ValidationError
from conductor.executor.output import parse_json_output, validate_output
from conductor.providers._event_format import (
    emit_parse_recovery_event,
    extract_tool_result_text,
    format_tool_arguments,
)
from conductor.providers._output_shape import normalize_agent_output
from conductor.providers._recovery_prompt import build_parse_recovery_prompt
from conductor.providers._schema import build_prompt_schema_properties
from conductor.providers.base import AgentOutput, AgentProvider, EventCallback
from conductor.providers.capabilities import ProviderCapabilities
from conductor.providers.reasoning import ReasoningEffort, resolve_reasoning_effort

if TYPE_CHECKING:
    from conductor.config.schema import AgentDef
    from conductor.execution.types import ManagedWorkspaceContext

# Resolved once at import: pi is a user-level CLI, not a Python dependency
# (mirrors the HERMES_SDK_AVAILABLE pattern so diagnostics can report it).
_PI_BINARY = shutil.which("pi")
PI_AVAILABLE = _PI_BINARY is not None

logger = logging.getLogger(__name__)

# Maximum number of recovery attempts when JSON parsing fails (Copilot parity).
_MAX_PARSE_RECOVERY_ATTEMPTS = 3

# Tail of pi's stderr kept for error messages (bounded, like docker.py).
_MAX_STDERR_BYTES = 8192

# Grace period between SIGTERM and SIGKILL when stopping a pi subprocess.
_TERMINATE_GRACE_SECONDS = 10


@dataclass
class _PiRunState:
    """Mutable accumulator for one pi subprocess execution.

    Filled from the NDJSON event stream; consulted when the stream ends.

    Attributes:
        session_id: pi session id from the session header (diagnostics).
        turns: Number of model-call cycles (turn_start events) seen.
        tool_calls: Number of tool executions started.
        final_assistant: The most recent assistant message dict — the
            authoritative final answer with its own ``model``/``usage``/
            ``stopReason`` fields.
        input_tokens: Sum of ``usage.input`` across assistant messages.
        output_tokens: Sum of ``usage.output`` across assistant messages.
        cache_read_tokens: Sum of ``usage.cacheRead`` (subset of input).
        cache_write_tokens: Sum of ``usage.cacheWrite`` (subset of input).
        last_call_input_tokens: ``usage.input`` of the final assistant
            message, or ``None`` when not reported.
    """

    session_id: str | None = None
    turns: int = 0
    tool_calls: int = 0
    final_assistant: dict[str, Any] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    last_call_input_tokens: int | None = None

    def merge_usage_from(self, other: _PiRunState) -> None:
        """Fold another run's usage into this one (recovery runs)."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens


def _fire(event_callback: EventCallback | None, event_type: str, data: dict[str, Any]) -> None:
    """Emit one event, swallowing subscriber errors."""
    if event_callback is None:
        return
    try:
        event_callback(event_type, data)
    except Exception:  # noqa: BLE001 - a failing subscriber must not break the run
        logger.debug("Error in event_callback for %s", event_type, exc_info=True)


def _handle_pi_event(
    event: dict[str, Any],
    state: _PiRunState,
    event_callback: EventCallback | None,
) -> None:
    """Map one pi session event onto Conductor events and run state.

    pi's JSON stream shapes are documented in pi's ``json.md``; the final
    assistant message carries the resolved ``model`` and ``usage``, which is
    where the authoritative answer comes from (deltas are only for display).

    Args:
        event: One parsed NDJSON record from pi's stdout.
        state: Accumulator mutated in place.
        event_callback: Optional subscriber to forward Conductor events to.
    """
    etype = event.get("type")

    if etype == "session":
        state.session_id = event.get("id")

    elif etype == "turn_start":
        state.turns += 1
        # Parity with the other providers: "awaiting_model" before the
        # first model call of the execution, a turn number afterwards.
        if state.turns == 1:
            _fire(event_callback, "agent_turn_start", {"turn": "awaiting_model"})
        else:
            _fire(event_callback, "agent_turn_start", {"turn": state.turns})

    elif etype == "message_update":
        nested = event.get("assistantMessageEvent") or {}
        kind = nested.get("type")
        if kind == "text_delta":
            _fire(event_callback, "agent_message", {"content": nested.get("delta") or ""})
        elif kind == "thinking_delta":
            _fire(event_callback, "agent_reasoning", {"content": nested.get("delta") or ""})

    elif etype == "tool_execution_start":
        state.tool_calls += 1
        _fire(
            event_callback,
            "agent_tool_start",
            {
                "tool_name": event.get("toolName") or "unknown",
                "arguments": format_tool_arguments(event.get("args")),
            },
        )

    elif etype == "tool_execution_end":
        _fire(
            event_callback,
            "agent_tool_complete",
            {
                "tool_name": event.get("toolName") or "unknown",
                "result": extract_tool_result_text(event.get("result")),
            },
        )

    elif etype == "message_end":
        message = event.get("message") or {}
        if message.get("role") == "assistant":
            state.final_assistant = message
            usage = message.get("usage") or {}
            state.input_tokens += usage.get("input") or 0
            state.output_tokens += usage.get("output") or 0
            state.cache_read_tokens += usage.get("cacheRead") or 0
            state.cache_write_tokens += usage.get("cacheWrite") or 0
            # pi reports usage per assistant message, so the last one is
            # exactly the most recent single API call (#412 contract).
            state.last_call_input_tokens = usage.get("input")

    # agent_start / agent_end / agent_settled / queue events carry nothing
    # the run state needs: the assistant message_end events above are the
    # source of truth, and EOF decides when the run is over.


def _assistant_text(message: dict[str, Any] | None) -> str | None:
    """Extract the joined text parts of a pi assistant message.

    Returns:
        The concatenated text content, or ``None`` when there is no message
        or it contains no text parts.
    """
    if not message:
        return None
    parts = [p.get("text") for p in message.get("content") or [] if p.get("type") == "text"]
    return "".join(p for p in parts if p) if any(parts) else None


async def _wait_for_event(event: asyncio.Event) -> None:
    """Await an interrupt event (factored for ``asyncio.wait`` wiring)."""
    await event.wait()


def _parse_and_validate(
    response: str,
    agent: AgentDef,
) -> tuple[dict[str, Any] | None, Exception | None, ValidationError | None]:
    """Parse a response and validate it against the agent's output schema.

    ``parse_json_output`` wraps syntax errors in ``ValidationError`` too,
    so the two failure kinds cannot be told apart by exception type. This
    splits them by which call failed (Copilot/Hermes parity), letting a
    schema failure keep its specific error while a syntax failure stays a
    ``ProviderError``.

    Returns:
        A ``(content, failure, schema_failure)`` tuple. ``content`` is set
        only when both parsing and validation succeeded; ``schema_failure``
        repeats ``failure`` only when it came from schema validation.
    """
    try:
        parsed = parse_json_output(response)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        return (None, exc, None)
    try:
        normalized = normalize_agent_output(parsed, agent.output)  # type: ignore[arg-type]
        validate_output(
            normalized,
            agent.output,  # type: ignore[arg-type]
            warn_undeclared_keys=True,
        )
    except ValidationError as exc:
        return (None, exc, exc)
    return (normalized, None, None)


class PiProvider(AgentProvider):
    """Executes Conductor agent steps inside the user's pi coding agent.

    Each ``execute()`` spawns one ``pi`` subprocess (one-shot JSON mode,
    no persisted session) in the step's working directory with the rendered
    prompt on stdin, relays pi's event stream as Conductor events, and
    normalizes the final assistant message into :class:`AgentOutput`.

    Requires the ``pi`` CLI on PATH (any provider configured inside pi will
    serve the model calls; pi's own default model applies when the workflow
    does not pin one).

    Example:
        >>> provider = PiProvider()
        >>> await provider.validate_connection()
        True
        >>> await provider.close()
    """

    CAPABILITIES = ProviderCapabilities(
        tier="experimental",
        # pi connects MCP servers from its own configuration (user-level
        # ~/.pi/agent/mcp.json and, after project trust, .pi/mcp.json in the
        # working directory). Conductor's runtime.mcp_servers keep working
        # for `type: mcp` steps (engine-side) but are not forwarded into pi
        # sessions; see the module docstring for the honesty trade-off.
        mcp_tools=True,
        workflow_tools_passthrough=False,
        streaming_events=True,
        agent_reasoning_events=True,
        # pi's --thinking accepts the same low..max vocabulary Conductor
        # standardizes on (plus off/minimal, which Conductor never sends).
        reasoning_effort=("low", "medium", "high", "xhigh", "max"),
        structured_output="prompt_injection",
        interrupt=True,
        max_session_seconds=True,
        checkpoint_resume=False,
        usage_tracking=True,
        concurrent_safe=True,
        working_dir=True,
    )

    def __init__(
        self,
        model: str | None = None,
        default_reasoning_effort: ReasoningEffort | None = None,
        max_session_seconds: float | None = None,
        pi_binary: str | None = None,
        approve_project: bool = True,
    ) -> None:
        """Initialize the Pi provider.

        Args:
            model: Default model passed to pi as ``--model`` (pi resolves it
                against its own models config). ``None`` delegates the choice
                to pi's own configured default.
            default_reasoning_effort: Workflow-wide default ``--thinking``
                level applied when an agent does not specify one.
            max_session_seconds: Default wall-clock limit per execution;
                ``None`` leaves pi unbounded.
            pi_binary: Path to the pi CLI. Defaults to the ``pi`` found on
                PATH at import time.
            approve_project: Pass pi's ``--approve`` so project-local files
                (worktree AGENTS.md context, ``.agents/skills`` from the
                working directory's ancestors such as ``~``) load without an
                interactive trust prompt, which a one-shot subprocess cannot
                answer.
        """
        self._default_model = model
        self._default_reasoning_effort = default_reasoning_effort
        self._default_max_session_seconds = max_session_seconds
        self._pi_binary = pi_binary or _PI_BINARY or "pi"
        self._approve_project = approve_project

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        agent: AgentDef,
        context: dict[str, Any],
        rendered_prompt: str,
        *,
        tools: list[str] | None = None,
        interrupt_signal: asyncio.Event | None = None,
        event_callback: EventCallback | None = None,
        skill_directories: list[str] | None = None,
        custom_agents: list[dict[str, Any]] | None = None,
        extra_mcp_servers: dict[str, Any] | None = None,
        continuation_state: object | None = None,
        suppress_mcp_servers: bool = False,
        managed_workspace: ManagedWorkspaceContext | None = None,
    ) -> AgentOutput:
        """Run one agent step inside a fresh pi subprocess.

        Raises:
            ProviderError: pi is not installed, the agent declares an option
                this provider cannot honor, the subprocess fails, exits
                non-zero, is interrupted, or exceeds its time limit.
            ValidationError: The output schema cannot be satisfied after the
                recovery budget is exhausted.
        """
        del context, continuation_state  # Prompts are stateless here; nothing to resume.
        del skill_directories  # skills=False: the executor eager-injects content.
        del custom_agents  # plugins=False: no subagent surface in pi sessions.
        del extra_mcp_servers  # pi owns its MCP server set (see module docstring).
        del suppress_mcp_servers  # Nothing to suppress: no workflow servers are attached.
        if managed_workspace is not None:
            raise ProviderError(
                "The pi provider does not support managed local workspaces",
                provider_name="pi",
                is_retryable=False,
            )
        if tools:
            raise ProviderError(
                f"Agent '{agent.name}' declares tools={tools!r}, but the pi "
                "provider runs pi's own tool set and cannot translate workflow "
                "tool allowlists.",
                suggestion=(
                    "Remove the 'tools:' field to use pi's default tools, or set "
                    "'tools: []' to disable them for this agent."
                ),
            )

        resolved_model = agent.model or self._default_model
        effort = self._resolve_effort(agent)
        resolved_timeout = (
            agent.max_session_seconds
            if agent.max_session_seconds is not None
            else self._default_max_session_seconds
        )

        try:
            cwd = agent.working_dir or Path.cwd()
        except OSError as exc:
            raise ProviderError(
                f"Agent '{agent.name}' declares no working_dir and the process "
                f"working directory could not be resolved: {exc}",
                is_retryable=False,
            ) from exc

        prompt = rendered_prompt
        if agent.output:
            schema_desc = json.dumps(
                {"type": "object", "properties": build_prompt_schema_properties(agent.output)},
                indent=2,
            )
            prompt += (
                f"\n\n**IMPORTANT: You MUST respond with a JSON object matching this schema:**\n"
                f"```json\n{schema_desc}\n```\n"
                "Return ONLY the JSON object, no other text."
            )

        args = self._build_cli_args(agent, resolved_model=resolved_model, effort=effort)

        state = await self._run_pi(
            agent_name=agent.name,
            prompt=prompt,
            args=args,
            cwd=str(cwd),
            resolved_timeout=resolved_timeout,
            interrupt_signal=interrupt_signal,
            event_callback=event_callback,
        )

        final_message = state.final_assistant
        stop_reason = (final_message or {}).get("stopReason")
        if stop_reason in ("error", "aborted"):
            raise ProviderError(
                f"pi agent execution for '{agent.name}' ended with stopReason="
                f"{stop_reason!r} (model={(final_message or {}).get('model')!r}).",
            )
        final_text = _assistant_text(final_message)
        if final_text is None:
            raise ProviderError(
                f"pi agent execution for '{agent.name}' returned no final assistant message.",
            )
        actual_model = (final_message or {}).get("model") or resolved_model

        recovered = False
        if agent.output:
            content, recovered = await self._parse_with_recovery(
                final_text,
                agent,
                agent_name=agent.name,
                args=args,
                cwd=str(cwd),
                resolved_timeout=resolved_timeout,
                interrupt_signal=interrupt_signal,
                event_callback=event_callback,
                state=state,
            )
        else:
            content: dict[str, Any] = {"text": final_text}

        input_tokens = state.input_tokens or None
        output_tokens = state.output_tokens or None
        return AgentOutput(
            content=content,
            raw_response={
                "final_message": final_message,
                "session_id": state.session_id,
                "turns": state.turns,
                "tool_calls": state.tool_calls,
                "model": actual_model,
            },
            tokens_used=(input_tokens + output_tokens)
            if input_tokens is not None and output_tokens is not None
            else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=state.cache_read_tokens or None,
            cache_write_tokens=state.cache_write_tokens or None,
            # A recovery run's usage merges into the aggregate, so the
            # single-call figure is no longer isolated (issue #412).
            last_call_input_tokens=None if recovered else state.last_call_input_tokens,
            model=actual_model,
        )

    def _build_cli_args(
        self,
        agent: AgentDef,
        *,
        resolved_model: str | None,
        effort: ReasoningEffort | None,
    ) -> list[str]:
        """Build the pi command line for one execution.

        Args:
            agent: Agent definition (system prompt, raw ``tools`` list).
            resolved_model: Model to pin, or ``None`` to use pi's default.
            effort: Resolved thinking level, or ``None`` to leave pi's.

        Returns:
            Argument list (the caller prepends the binary and pipes the
            prompt via stdin).

        Note:
            Tool selection follows the RAW ``agent.tools`` signal, not the
            executor-resolved list, because ``resolve_agent_tools`` collapses
            "agent omitted ``tools:``" and "the workflow declares none" into
            the same empty list (the ambiguity is documented on
            ``claude_agent_sdk._DEFAULT_TOOL_PRESET``). For this harness
            provider the split matters: an omitted list means "run pi as the
            user configured it" (its own tool set), while an explicit
            ``tools: []`` is a real disable and maps to ``--no-tools``.
        """
        args = ["--mode", "json", "--print", "--no-session"]
        if resolved_model:
            args += ["--model", resolved_model]
        if effort:
            args += ["--thinking", effort]
        if agent.system_prompt:
            args += ["--system-prompt", agent.system_prompt]
        if agent.tools == []:
            args += ["--no-tools"]
        if self._approve_project:
            args += ["--approve"]
        return args

    def _resolve_effort(self, agent: AgentDef) -> ReasoningEffort | None:
        """Resolve and validate reasoning effort (the run-time half).

        Raises:
            ValidationError: The resolved level is not supported by pi.
        """
        effort = resolve_reasoning_effort(agent, self._default_reasoning_effort)
        if effort is not None:
            supported = self.CAPABILITIES.reasoning_effort
            if supported is None or effort not in supported:
                raise ValidationError(
                    f"Agent {agent.name!r} resolves to reasoning.effort={effort!r}, "
                    f"but the pi provider supports only "
                    f"{sorted(supported) if supported else []}.",
                )
        return effort

    # ------------------------------------------------------------------
    # Subprocess plumbing
    # ------------------------------------------------------------------

    async def _run_pi(
        self,
        *,
        agent_name: str,
        prompt: str,
        args: list[str],
        cwd: str,
        resolved_timeout: float | None,
        interrupt_signal: asyncio.Event | None,
        event_callback: EventCallback | None,
    ) -> _PiRunState:
        """Spawn pi, feed the prompt, and pump the event stream to EOF.

        Args:
            agent_name: Step name, for error messages.
            prompt: Full rendered prompt (schema block included).
            args: pi CLI arguments from :meth:`_build_cli_args`.
            cwd: Working directory for the subprocess.
            resolved_timeout: Wall-clock limit, or ``None`` for none.
            interrupt_signal: When set, the subprocess is killed.
            event_callback: Subscriber for Conductor events.

        Raises:
            ProviderError: pi could not run, was interrupted, timed out, or
                exited non-zero.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                self._pi_binary,
                *args,
                cwd=cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            raise ProviderError(
                f"Could not spawn pi ({self._pi_binary!r}) for agent '{agent_name}': {exc}",
                suggestion="Install pi or set pi_binary on the provider.",
                is_retryable=False,
            ) from exc

        state = _PiRunState()

        async def _feed_stdin() -> None:
            assert proc.stdin is not None
            try:
                proc.stdin.write(prompt.encode("utf-8"))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass  # pi dying early is reported via exit code / stderr
            finally:
                with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
                    proc.stdin.close()

        async def _drain_stderr(tail: list[bytes]) -> None:
            assert proc.stderr is not None
            buf = b""
            while chunk := await proc.stderr.read(4096):
                buf = (buf + chunk)[-_MAX_STDERR_BYTES:]
                tail[0] = buf

        tail: list[bytes] = [b""]
        reader = asyncio.ensure_future(self._pump_events(proc, state, event_callback))
        feeder = asyncio.ensure_future(_feed_stdin())
        err_drain = asyncio.ensure_future(_drain_stderr(tail))
        interrupt_watcher = (
            asyncio.ensure_future(_wait_for_event(interrupt_signal))
            if interrupt_signal is not None
            else None
        )

        try:
            awaitables: list[asyncio.Future[None]] = [reader]
            if interrupt_watcher is not None:
                awaitables.append(interrupt_watcher)
            if resolved_timeout is not None:
                done, _pending = await asyncio.wait(
                    awaitables, timeout=resolved_timeout, return_when=asyncio.FIRST_COMPLETED
                )
            else:
                done, _pending = await asyncio.wait(awaitables, return_when=asyncio.FIRST_COMPLETED)

            interrupted = interrupt_watcher is not None and interrupt_watcher in done
            timed_out = reader not in done and not interrupted
            if timed_out or interrupted:
                await self._terminate(proc)

            await asyncio.gather(reader, feeder, err_drain, return_exceptions=True)
            returncode = await proc.wait()
        finally:
            for task in (interrupt_watcher,):
                if task is not None and not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

        if interrupted:
            raise ProviderError(
                f"pi agent '{agent_name}' was interrupted by user request",
                is_retryable=False,
            )
        if timed_out:
            raise ProviderError(
                f"pi agent '{agent_name}' exceeded maximum session duration of "
                f"{resolved_timeout:.0f}s",
                is_retryable=True,
            )
        if returncode != 0:
            detail = tail[0].decode("utf-8", "replace").strip()
            raise ProviderError(
                f"pi exited with code {returncode} for agent '{agent_name}'"
                + (f": {detail}" if detail else ", no stderr output."),
                is_retryable=True,
            )
        return state

    async def _pump_events(
        self,
        proc: asyncio.subprocess.Process,
        state: _PiRunState,
        event_callback: EventCallback | None,
    ) -> None:
        """Read and dispatch pi's NDJSON stdout until EOF."""
        assert proc.stdout is not None
        while raw := await proc.stdout.readline():
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                logger.debug("Non-JSON line from pi: %.100s", line)
                continue
            if isinstance(event, dict):
                _handle_pi_event(event, state, event_callback)

    @staticmethod
    async def _terminate(proc: asyncio.subprocess.Process) -> None:
        """Stop a still-running pi subprocess: TERM, brief grace, then KILL."""
        if proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_GRACE_SECONDS)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()

    # ------------------------------------------------------------------
    # Structured output: parse with recovery (mirrors Copilot pattern)
    # ------------------------------------------------------------------

    async def _parse_with_recovery(
        self,
        response: str,
        agent: AgentDef,
        *,
        agent_name: str,
        args: list[str],
        cwd: str,
        resolved_timeout: float | None,
        interrupt_signal: asyncio.Event | None,
        event_callback: EventCallback | None,
        state: _PiRunState,
    ) -> tuple[Any, bool]:
        """Parse the response as JSON, re-asking pi on parse failures.

        Recovery runs reuse the original CLI arguments but send the recovery
        prompt; their usage is merged into ``state`` so token accounting
        stays complete, while the caller withholds the single-call figure
        (issue #412).

        Returns:
            The parsed (normalized) content and whether a recovery call ran.

        Raises:
            ValidationError: A schema failure after the budget is spent.
            ProviderError: A syntax failure after the budget is spent, or a
                subprocess failure during recovery.
        """
        last_error: str | None = None
        last_schema_failure: ValidationError | None = None
        schema = build_prompt_schema_properties(agent.output)  # type: ignore[arg-type]
        recovered = False

        for attempt in range(_MAX_PARSE_RECOVERY_ATTEMPTS + 1):
            content, failure, schema_failure = _parse_and_validate(response, agent)
            if content is not None:
                return content, recovered

            last_error = str(failure)
            last_schema_failure = schema_failure
            if attempt >= _MAX_PARSE_RECOVERY_ATTEMPTS:
                break
            recovered = True
            logger.info(
                "Agent '%s' parse recovery attempt %d/%d: %s",
                agent_name,
                attempt + 1,
                _MAX_PARSE_RECOVERY_ATTEMPTS,
                last_error,
            )
            emit_parse_recovery_event(
                event_callback,
                attempt=attempt + 1,
                max_attempts=_MAX_PARSE_RECOVERY_ATTEMPTS,
                is_schema_failure=schema_failure is not None,
                error=last_error,
            )
            recovery_prompt = build_parse_recovery_prompt(
                last_error,
                response,
                schema,
                is_schema_failure=schema_failure is not None,
            )
            run_state = await self._run_pi(
                agent_name=agent_name,
                prompt=recovery_prompt,
                args=args,
                cwd=cwd,
                resolved_timeout=resolved_timeout,
                interrupt_signal=interrupt_signal,
                event_callback=event_callback,
            )
            state.merge_usage_from(run_state)
            response = _assistant_text(run_state.final_assistant) or ""

        if last_schema_failure is not None:
            raise last_schema_failure
        raise ProviderError(
            f"pi agent '{agent_name}' could not produce JSON matching the "
            f"output schema after {_MAX_PARSE_RECOVERY_ATTEMPTS} recovery "
            f"attempts: {last_error}",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def validate_connection(self) -> bool:
        """Confirm the pi CLI is installed and runnable.

        Spawns ``pi --version`` (no model call); credential or model
        failures surface at first execution, as with the other providers.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                self._pi_binary,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError):
            return False
        try:
            await asyncio.wait_for(proc.wait(), timeout=15)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return False
        return proc.returncode == 0

    def pricing_host(self) -> str | None:
        """``None``: pi routes to whatever providers the user configured."""
        return None

    async def close(self) -> None:
        """No-op — the pi provider is stateless (one subprocess per call)."""

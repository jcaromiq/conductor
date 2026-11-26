"""Hermetic unit tests for the pi provider (no pi CLI is spawned)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from conductor.config.schema import AgentDef, OutputField
from conductor.exceptions import ProviderError
from conductor.providers.capabilities import get_capabilities
from conductor.providers.pi import (
    PI_AVAILABLE,
    PiProvider,
    _PiRunState,
    _assistant_text,
    _handle_pi_event,
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _agent(**overrides: Any) -> AgentDef:
    base: dict[str, Any] = {"name": "reviewer"}
    base.update(overrides)
    return AgentDef(**base)


def _state(final_text: str, **message_overrides: Any) -> _PiRunState:
    """Build a finished-run state by replaying one assistant message_end."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": [{"type": "text", "text": final_text}],
        "model": "glm-5-3",
        "stopReason": "stop",
        "usage": {"input": 100, "output": 5, "cacheRead": 10, "cacheWrite": 0},
    }
    message.update(message_overrides)
    state = _PiRunState()
    _handle_pi_event({"type": "message_end", "message": message}, state, None)
    return state


class TestRegistrationAndCapabilities:
    def test_provider_capabilities_are_declared(self) -> None:
        caps = get_capabilities("pi")
        assert caps is not None
        assert caps.tier == "experimental"
        assert caps.mcp_tools is True
        assert caps.workflow_tools_passthrough is False
        assert caps.working_dir is True
        assert caps.concurrent_safe is True

    def test_availability_reflects_path_probe(self) -> None:
        assert isinstance(PI_AVAILABLE, bool)


class TestArgumentBuilding:
    def test_default_args_delegate_model_to_pi(self) -> None:
        provider = PiProvider()
        args = provider._build_cli_args(_agent(), resolved_model=None, effort=None, no_tools=False)
        assert args == ["--mode", "json", "--print", "--no-session", "--approve"]
        assert "--model" not in args and "--thinking" not in args

    def test_pinned_model_and_effort_are_forwarded(self) -> None:
        provider = PiProvider()
        args = provider._build_cli_args(
            _agent(system_prompt="be terse"),
            resolved_model="glm-5-3",
            effort="high",
            no_tools=True,
        )
        assert "--model" in args and args[args.index("--model") + 1] == "glm-5-3"
        assert "--thinking" in args and args[args.index("--thinking") + 1] == "high"
        assert "--no-tools" in args
        assert "--system-prompt" in args and args[args.index("--system-prompt") + 1] == "be terse"

    def test_approve_can_be_disabled(self) -> None:
        provider = PiProvider(approve_project=False)
        args = provider._build_cli_args(_agent(), resolved_model=None, effort=None, no_tools=False)
        assert "--approve" not in args


class TestEventMapping:
    def test_stream_events_relay_and_accumulate(self) -> None:
        state = _PiRunState()
        fired: list[tuple[str, dict[str, Any]]] = []
        events = [
            {"type": "session", "id": "s1"},
            {"type": "turn_start"},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "Ho"}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "la"}},
            {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "bash", "args": {"command": "ls"}},
            {
                "type": "tool_execution_end",
                "toolCallId": "c1",
                "toolName": "bash",
                "result": {"content": [{"type": "text", "text": "ok"}]},
                "isError": False,
            },
            {"type": "turn_start"},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hola"}],
                    "model": "glm-5-3",
                    "stopReason": "stop",
                    "usage": {"input": 90, "output": 4, "cacheRead": 5, "cacheWrite": 1},
                },
            },
        ]
        for event in events:
            _handle_pi_event(event, state, lambda t, d: fired.append((t, d)))

        assert state.session_id == "s1"
        assert state.turns == 2
        assert state.tool_calls == 1
        assert state.final_assistant is not None
        assert _assistant_text(state.final_assistant) == "hola"
        assert state.input_tokens == 90 and state.output_tokens == 4
        assert state.cache_read_tokens == 5 and state.cache_write_tokens == 1
        assert state.last_call_input_tokens == 90

        assert fired[0] == ("agent_turn_start", {"turn": "awaiting_model"})
        assert ("agent_turn_start", {"turn": 2}) in fired
        assert ("agent_message", {"content": "Ho"}) in fired
        tool_start = [d for t, d in fired if t == "agent_tool_start"][0]
        assert tool_start == {"tool_name": "bash", "arguments": '{"command": "ls"}'}
        tool_done = [d for t, d in fired if t == "agent_tool_complete"][0]
        assert tool_done["tool_name"] == "bash"

    def test_usage_sums_across_assistant_messages(self) -> None:
        state = _PiRunState()
        for n in (10, 20):
            _handle_pi_event(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "usage": {"input": n, "output": 1, "cacheRead": 0, "cacheWrite": 0},
                    },
                },
                state,
                None,
            )
        assert state.input_tokens == 30
        assert state.last_call_input_tokens == 20

    def test_assistant_text_none_without_text_parts(self) -> None:
        assert _assistant_text(None) is None
        assert _assistant_text({"role": "assistant", "content": []}) is None


class TestExecuteNormalization:
    def _provider(self, model: str | None = None) -> PiProvider:
        return PiProvider(model=model)

    def test_plain_text_output(self) -> None:
        provider = self._provider()
        with patch.object(PiProvider, "_run_pi", AsyncMock(return_value=_state("hola"))):
            output = _run(provider.execute(_agent(), {}, "prompt"))
        assert output.content == {"text": "hola"}
        assert output.model == "glm-5-3"
        assert output.input_tokens == 100
        assert output.output_tokens == 5
        assert output.cache_read_tokens == 10
        assert output.last_call_input_tokens == 100
        assert output.tokens_used == 105

    def test_schema_output_parses_and_recovers(self) -> None:
        provider = self._provider()
        agent = _agent(output={"source": OutputField(type="string")})
        good = _state('{"source": "jira"}')
        with patch.object(
            PiProvider,
            "_run_pi",
            AsyncMock(side_effect=[_state("no json"), good]),
        ) as run_pi:
            output = _run(provider.execute(agent, {}, "prompt"))
        assert output.content == {"source": "jira"}
        assert run_pi.await_count == 2  # one recovery round-trip
        assert output.last_call_input_tokens is None  # aggregate no longer single-call

    def test_schema_output_first_try_valid(self) -> None:
        provider = self._provider()
        agent = _agent(output={"source": OutputField(type="string")})
        with patch.object(
            PiProvider, "_run_pi", AsyncMock(return_value=_state('{"source": "jira"}'))
        ):
            output = _run(provider.execute(agent, {}, "prompt"))
        assert output.content == {"source": "jira"}

    def test_error_stop_reason_raises(self) -> None:
        provider = self._provider()
        with patch.object(
            PiProvider, "_run_pi", AsyncMock(return_value=_state("x", stopReason="error"))
        ):
            with pytest.raises(ProviderError):
                _run(provider.execute(_agent(), {}, "prompt"))

    def test_nonempty_tools_refused(self) -> None:
        provider = self._provider()
        with pytest.raises(ProviderError, match="tools"):
            _run(provider.execute(_agent(), {}, "prompt", tools=["bash"]))

"""V1 NVIDIA-backed tool-calling agent for NetEval."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI

from src.prompts import V1_SYSTEM_PROMPT, V2_SYSTEM_PROMPT
from src.trajectory_store import TrajectoryStore
from src.tools import (
    check_reachability,
    find_blocking_hop,
    get_device_state,
    inspect_acl,
    trace_path,
)


DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MAX_TOOL_ROUNDS = 5
DEFAULT_AGENT_VERSION = "v1"
EVIDENCE_ORDER = (
    "path",
    "reachability",
    "device_state",
    "policy",
    "root_cause",
)


class ConfigurationError(ValueError):
    """Raised when required NVIDIA configuration is missing."""


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "check_reachability": check_reachability,
    "trace_path": trace_path,
    "inspect_acl": inspect_acl,
    "get_device_state": get_device_state,
    "find_blocking_hop": find_blocking_hop,
}


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "check_reachability",
            "description": (
                "Determine whether a source can communicate with a destination "
                "on a specific port and protocol."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source device ID."},
                    "destination": {
                        "type": "string",
                        "description": "Destination device ID.",
                    },
                    "port": {
                        "type": "integer",
                        "description": "Destination service port.",
                    },
                    "protocol": {
                        "type": "string",
                        "description": "Network protocol, usually tcp.",
                        "default": "tcp",
                    },
                },
                "required": ["source", "destination", "port"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trace_path",
            "description": "Identify the topology path between two network entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source device ID."},
                    "destination": {
                        "type": "string",
                        "description": "Destination device ID.",
                    },
                },
                "required": ["source", "destination"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_acl",
            "description": "Inspect ACL rules configured on a firewall or device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "description": "Device whose ACL rules should be inspected.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional source device filter.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Optional destination device filter.",
                    },
                    "port": {
                        "type": "integer",
                        "description": "Optional destination port filter.",
                    },
                    "protocol": {
                        "type": "string",
                        "description": "Optional protocol filter, usually tcp.",
                        "default": "tcp",
                    },
                },
                "required": ["device"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_device_state",
            "description": "Inspect the status, type, zone, and services of a device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "description": "Device ID to inspect.",
                    }
                },
                "required": ["device"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_blocking_hop",
            "description": (
                "Determine whether a path is blocked and identify the blocking "
                "device or ACL rule."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source device ID."},
                    "destination": {
                        "type": "string",
                        "description": "Destination device ID.",
                    },
                    "port": {
                        "type": "integer",
                        "description": "Destination service port.",
                    },
                    "protocol": {
                        "type": "string",
                        "description": "Network protocol, usually tcp.",
                        "default": "tcp",
                    },
                },
                "required": ["source", "destination", "port"],
                "additionalProperties": False,
            },
        },
    },
]

# Keep the original V1 definitions available under both names for compatibility.
V1_TOOL_DEFINITIONS = TOOL_DEFINITIONS
V2_TOOL_DEFINITIONS = deepcopy(V1_TOOL_DEFINITIONS)

V2_TOOL_DESCRIPTIONS = {
    "trace_path": (
        "Return the physical/logical topology path between source and destination. "
        "Use for path/hop questions. For a pure path question, report only the returned "
        "hops and stop. A returned path does NOT prove reachability or policy permission."
    ),
    "check_reachability": (
        "Determine whether source can communicate with destination on a specific "
        "TCP/UDP port using topology, device state, and network policy. Preserve the "
        "user's exact source, destination, and port. Use first for direct reachability "
        "and root-cause questions. Its reason is a diagnostic signal. For a diagnostic "
        "endpoint-versus-policy question, verify destination_down with "
        "get_device_state(destination); a simple can-reach question may stop here. "
        "Report reachable=true/false as reachability evidence, not as an ACL allow/deny "
        "action unless an ACL result provides that action."
    ),
    "inspect_acl": (
        "Inspect ACL rules on a specific firewall/device, optionally filtered by "
        "source, destination, port, and protocol. Use the blocking device and exact "
        "traffic tuple reported by diagnostics when ACL/policy evidence is required. "
        "Do not pass a rule ID as an argument."
    ),
    "get_device_state": (
        "Return status and service metadata for one device. Use for health/up/down "
        "questions or to verify the destination after check_reachability reports "
        "destination_down. A direct health question needs only this tool. Do not use "
        "repeatedly for unrelated devices."
    ),
    "find_blocking_hop": (
        "Identify whether traffic is blocked and, when possible, the blocking "
        "device/rule. Use for root-cause or policy troubleshooting when supporting "
        "blocking evidence is still needed."
    ),
}

for tool_definition in V2_TOOL_DEFINITIONS:
    function = tool_definition["function"]
    function["description"] = V2_TOOL_DESCRIPTIONS[function["name"]]

AGENT_CONFIGS = {
    "v1": {
        "system_prompt": V1_SYSTEM_PROMPT,
        "tool_definitions": V1_TOOL_DEFINITIONS,
    },
    "v2": {
        "system_prompt": V2_SYSTEM_PROMPT,
        "tool_definitions": V2_TOOL_DEFINITIONS,
    },
}


class NetworkAgent:
    """Bounded, versioned agent that dispatches model tool calls explicitly."""

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        trajectory_store: TrajectoryStore | None = None,
        agent_version: str = DEFAULT_AGENT_VERSION,
    ) -> None:
        load_dotenv()

        self.model = model or os.getenv("NVIDIA_MODEL")
        if not self.model:
            raise ConfigurationError(
                "NVIDIA_MODEL is not configured. Set it in the environment or .env file."
            )
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1.")
        if not agent_version:
            raise ValueError("agent_version must not be empty.")
        if agent_version not in AGENT_CONFIGS:
            supported_versions = ", ".join(sorted(AGENT_CONFIGS))
            raise ValueError(
                f"Unsupported agent_version '{agent_version}'. "
                f"Expected one of: {supported_versions}."
            )

        if client is None:
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "NVIDIA_API_KEY is not configured. Set it in the environment or .env file."
                )
            base_url = os.getenv("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL
            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

        self.client = client
        self.max_tool_rounds = max_tool_rounds
        self.trajectory_store = trajectory_store
        self.agent_version = agent_version
        self.system_prompt = AGENT_CONFIGS[agent_version]["system_prompt"]
        self.tool_definitions = AGENT_CONFIGS[agent_version]["tool_definitions"]

    def run(self, query: str) -> dict[str, Any]:
        """Run a bounded tool-calling conversation and return in-memory run data."""
        started_at = perf_counter()
        run_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        messages: list[Any] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        steps: list[dict[str, Any]] = []
        token_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        usage_available = False
        tool_rounds = 0
        evidence_requirements = (
            self._derive_evidence_requirements(query)
            if self.agent_version == "v2"
            else set()
        )
        evidence_satisfied: set[str] = set()
        evidence_gate_triggers = 0
        executed_tool_calls: set[tuple[str, str]] | None = (
            set() if self.agent_version == "v2" else None
        )

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_definitions,
                    tool_choice="auto",
                )
                usage_available = self._accumulate_usage(
                    response, token_totals
                ) or usage_available
                message = response.choices[0].message
            except Exception as exc:
                return self._run_result(
                    run_id=run_id,
                    timestamp=timestamp,
                    query=query,
                    final_answer=None,
                    steps=steps,
                    completed=False,
                    status="api_error",
                    error=f"NVIDIA API error: {exc}",
                    token_totals=token_totals,
                    usage_available=usage_available,
                    started_at=started_at,
                    evidence_requirements=evidence_requirements,
                    evidence_satisfied=evidence_satisfied,
                    evidence_gate_triggers=evidence_gate_triggers,
                )

            tool_calls = self._field(message, "tool_calls") or []
            if not tool_calls:
                content = self._field(message, "content")
                missing_evidence = evidence_requirements - evidence_satisfied
                if self.agent_version == "v2" and missing_evidence:
                    evidence_gate_triggers += 1
                    if tool_rounds >= self.max_tool_rounds:
                        return self._run_result(
                            run_id=run_id,
                            timestamp=timestamp,
                            query=query,
                            final_answer=None,
                            steps=steps,
                            completed=False,
                            status="max_tool_rounds_exceeded",
                            error=(
                                f"Maximum tool rounds exceeded ({self.max_tool_rounds}) "
                                "before all required evidence was gathered."
                            ),
                            token_totals=token_totals,
                            usage_available=usage_available,
                            started_at=started_at,
                            evidence_requirements=evidence_requirements,
                            evidence_satisfied=evidence_satisfied,
                            evidence_gate_triggers=evidence_gate_triggers,
                        )
                    messages.append({"role": "assistant", "content": content or ""})
                    messages.append(
                        {
                            "role": "system",
                            "content": self._evidence_gate_message(missing_evidence),
                        }
                    )
                    tool_rounds += 1
                    continue
                return self._run_result(
                    run_id=run_id,
                    timestamp=timestamp,
                    query=query,
                    final_answer=content or "",
                    steps=steps,
                    completed=True,
                    status="completed",
                    error=None,
                    token_totals=token_totals,
                    usage_available=usage_available,
                    started_at=started_at,
                    evidence_requirements=evidence_requirements,
                    evidence_satisfied=evidence_satisfied,
                    evidence_gate_triggers=evidence_gate_triggers,
                )

            if tool_rounds >= self.max_tool_rounds:
                return self._run_result(
                    run_id=run_id,
                    timestamp=timestamp,
                    query=query,
                    final_answer=None,
                    steps=steps,
                    completed=False,
                    status="max_tool_rounds_exceeded",
                    error=(
                        f"Maximum tool rounds exceeded ({self.max_tool_rounds}) "
                        "before a final answer was produced."
                    ),
                    token_totals=token_totals,
                    usage_available=usage_available,
                    started_at=started_at,
                    evidence_requirements=evidence_requirements,
                    evidence_satisfied=evidence_satisfied,
                    evidence_gate_triggers=evidence_gate_triggers,
                )

            messages.append(self._assistant_tool_message(message, tool_calls))
            for tool_call in tool_calls:
                step, tool_call_id = self._execute_tool_call(
                    tool_call,
                    executed_tool_calls=executed_tool_calls,
                )
                step["step_index"] = len(steps) + 1
                steps.append(step)
                if self.agent_version == "v2":
                    self._update_evidence(
                        step,
                        evidence_requirements,
                        evidence_satisfied,
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(step["result"], sort_keys=True),
                    }
                )
            tool_rounds += 1

    @classmethod
    def _derive_evidence_requirements(cls, query: str) -> set[str]:
        """Derive general evidence classes from user language for V2 runs."""
        text = cls._intent_text(query)
        requirements: set[str] = set()

        diagnostic = bool(
            re.search(r"\bwhy\s+(?:can(?:not|'t)|is|are|does|do)\b", text)
            or re.search(r"\bwhat(?:'s| is)?\s+caus(?:e|ing)\b", text)
            or any(
                marker in text
                for marker in (
                    "diagnose",
                    "diagnosis",
                    "troubleshoot",
                    "root cause",
                    "caused by",
                    "responsible",
                    "failure",
                )
            )
        )
        endpoint_policy_comparison = diagnostic and bool(
            re.search(r"\b(endpoint|device|destination)\b", text)
            and re.search(r"\b(policy|acl|firewall|blocked?)\b", text)
        )

        if (
            re.search(r"\b(path|route|routes|routing|hops?|traverse|traverses)\b", text)
            or re.search(r"\bget\s+from\b.+\bto\b", text)
        ):
            requirements.add("path")

        if (
            re.search(r"\bcan\b.+\b(reach|connect|communicate)\b", text)
            or re.search(r"\b(reachable|reachability|communication possible)\b", text)
            or re.search(
                r"\bwhether\b.+\b(connection|traffic|communication)\b.+"
                r"\b(allowed|reachable|possible|works?)\b",
                text,
            )
        ):
            requirements.add("reachability")

        if diagnostic:
            requirements.add("root_cause")

        direct_device_state = bool(
            re.search(r"\b(healthy|device status|endpoint status)\b", text)
            or re.search(r"\b(is|are)\b.+\b(up|down)\b", text)
        )
        if direct_device_state and not diagnostic:
            requirements.add("device_state")

        policy_language = bool(
            re.search(r"\b(acl|firewall rule|deny rule|policy|permit)\b", text)
        )
        if policy_language and not endpoint_policy_comparison:
            requirements.add("policy")

        return requirements

    @staticmethod
    def _intent_text(query: str) -> str:
        return query.casefold().replace("’", "'").replace("‑", "-")

    @classmethod
    def _update_evidence(
        cls,
        step: dict[str, Any],
        requirements: set[str],
        satisfied: set[str],
    ) -> None:
        result = step.get("result")
        if not isinstance(result, dict) or result.get("error"):
            return

        tool = step.get("tool")
        if tool == "trace_path":
            satisfied.add("path")
        elif tool == "get_device_state":
            satisfied.add("device_state")
        elif tool == "inspect_acl":
            satisfied.add("policy")
        elif tool in {"check_reachability", "find_blocking_hop"}:
            if tool == "check_reachability":
                satisfied.add("reachability")
            if "root_cause" in requirements:
                satisfied.add("root_cause")

            reason = result.get("reason")
            if reason == "destination_down" and "root_cause" in requirements:
                requirements.add("device_state")
            elif reason == "explicit_deny" and "root_cause" in requirements:
                requirements.add("policy")

    @classmethod
    def _evidence_gate_message(cls, missing: set[str]) -> str:
        missing_text = ", ".join(cls._ordered_evidence(missing))
        return (
            "You have not yet gathered all evidence required by the user's request. "
            f"Missing evidence: {missing_text}. Continue using the available tools "
            "before answering. Do not fabricate tool results."
        )

    @staticmethod
    def _ordered_evidence(values: set[str]) -> list[str]:
        return [name for name in EVIDENCE_ORDER if name in values]

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def _assistant_tool_message(
        self, message: Any, tool_calls: list[Any]
    ) -> dict[str, Any]:
        serialized_calls = []
        for index, tool_call in enumerate(tool_calls):
            function = self._field(tool_call, "function")
            serialized_calls.append(
                {
                    "id": self._field(tool_call, "id") or f"missing-call-id-{index}",
                    "type": self._field(tool_call, "type", "function"),
                    "function": {
                        "name": self._field(function, "name", ""),
                        "arguments": self._field(function, "arguments", "{}"),
                    },
                }
            )
        return {
            "role": "assistant",
            "content": self._field(message, "content"),
            "tool_calls": serialized_calls,
        }

    def _execute_tool_call(
        self,
        tool_call: Any,
        executed_tool_calls: set[tuple[str, str]] | None = None,
    ) -> tuple[dict[str, Any], str]:
        function = self._field(tool_call, "function")
        name = self._field(function, "name", "")
        raw_arguments = self._field(function, "arguments", "{}")
        tool_call_id = self._field(tool_call, "id") or "missing-call-id"

        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must decode to a JSON object.")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return (
                {
                    "tool": name,
                    "arguments": None,
                    "result": {
                        "error": "malformed_tool_arguments",
                        "message": str(exc),
                    },
                },
                tool_call_id,
            )

        tool = TOOL_REGISTRY.get(name)
        if tool is None:
            return (
                {
                    "tool": name,
                    "arguments": arguments,
                    "result": {
                        "error": "unknown_tool",
                        "message": f"Unknown tool requested: {name}.",
                    },
                },
                tool_call_id,
            )

        try:
            inspect.signature(tool).bind(**arguments)
        except TypeError as exc:
            return (
                {
                    "tool": name,
                    "arguments": arguments,
                    "result": {
                        "error": "malformed_tool_arguments",
                        "message": str(exc),
                    },
                },
                tool_call_id,
            )

        if executed_tool_calls is not None:
            call_key = (
                name,
                json.dumps(arguments, sort_keys=True, separators=(",", ":")),
            )
            if call_key in executed_tool_calls:
                return (
                    {
                        "tool": name,
                        "arguments": arguments,
                        "result": {
                            "error": "duplicate_tool_call",
                            "message": (
                                "This exact tool call has already been executed. "
                                "Use existing evidence or choose a different "
                                "investigation step."
                            ),
                        },
                    },
                    tool_call_id,
                )
            executed_tool_calls.add(call_key)

        try:
            result = tool(**arguments)
        except Exception as exc:
            result = {
                "error": "tool_execution_error",
                "message": str(exc),
            }

        return (
            {
                "tool": name,
                "arguments": arguments,
                "result": result,
            },
            tool_call_id,
        )

    def _accumulate_usage(
        self, response: Any, token_totals: dict[str, int]
    ) -> bool:
        usage = self._field(response, "usage")
        if usage is None:
            return False

        input_tokens = self._field(usage, "prompt_tokens")
        if input_tokens is None:
            input_tokens = self._field(usage, "input_tokens")
        output_tokens = self._field(usage, "completion_tokens")
        if output_tokens is None:
            output_tokens = self._field(usage, "output_tokens")
        total_tokens = self._field(usage, "total_tokens")

        found_usage = any(
            value is not None for value in (input_tokens, output_tokens, total_tokens)
        )
        if not found_usage:
            return False

        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens

        token_totals["input_tokens"] += input_tokens
        token_totals["output_tokens"] += output_tokens
        token_totals["total_tokens"] += int(total_tokens)
        return True

    def _run_result(
        self,
        *,
        run_id: str,
        timestamp: str,
        query: str,
        final_answer: str | None,
        steps: list[dict[str, Any]],
        completed: bool,
        status: str,
        error: str | None,
        token_totals: dict[str, int],
        usage_available: bool,
        started_at: float,
        evidence_requirements: set[str],
        evidence_satisfied: set[str],
        evidence_gate_triggers: int,
    ) -> dict[str, Any]:
        result = {
            "run_id": run_id,
            "timestamp": timestamp,
            "query": query,
            "agent_version": self.agent_version,
            "model": self.model,
            "final_answer": final_answer,
            "steps": steps,
            "tool_call_count": len(steps),
            "completed": completed,
            "status": status,
            "error": error,
            "input_tokens": (
                token_totals["input_tokens"] if usage_available else None
            ),
            "output_tokens": (
                token_totals["output_tokens"] if usage_available else None
            ),
            "total_tokens": (
                token_totals["total_tokens"] if usage_available else None
            ),
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        }

        tokens = {
            "input": result["input_tokens"] or 0,
            "output": result["output_tokens"] or 0,
            "total": result["total_tokens"] or 0,
        }
        result["tokens"] = tokens
        result["trajectory_path"] = None
        if self.agent_version == "v2":
            result["evidence_requirements"] = self._ordered_evidence(
                evidence_requirements
            )
            result["evidence_satisfied"] = self._ordered_evidence(evidence_satisfied)
            result["evidence_gate_triggers"] = evidence_gate_triggers

        if self.trajectory_store is not None:
            trajectory = {
                "run_id": run_id,
                "timestamp": timestamp,
                "query": query,
                "agent_version": self.agent_version,
                "model": self.model,
                "steps": steps,
                "final_answer": final_answer,
                "tool_call_count": len(steps),
                "completed": completed,
                "error": self._trajectory_error(status, error),
                "latency_ms": result["latency_ms"],
                "tokens": tokens,
            }
            if self.agent_version == "v2":
                trajectory["evidence_requirements"] = result[
                    "evidence_requirements"
                ]
                trajectory["evidence_satisfied"] = result["evidence_satisfied"]
                trajectory["evidence_gate_triggers"] = evidence_gate_triggers
            result["trajectory_path"] = self.trajectory_store.save(trajectory)

        return result

    @staticmethod
    def _trajectory_error(status: str, error: str | None) -> dict[str, str] | None:
        if error is None:
            return None
        error_type = (
            "max_tool_rounds"
            if status == "max_tool_rounds_exceeded"
            else status
        )
        return {"type": error_type, "message": error}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a NetEval network agent.")
    parser.add_argument("question", help="Network troubleshooting question to answer.")
    parser.add_argument(
        "--agent-version",
        choices=sorted(AGENT_CONFIGS),
        default=DEFAULT_AGENT_VERSION,
        help="Agent configuration version (default: v1).",
    )
    args = parser.parse_args(argv)

    try:
        agent = NetworkAgent(
            trajectory_store=TrajectoryStore(),
            agent_version=args.agent_version,
        )
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    result = agent.run(args.question)
    print(json.dumps(result, indent=2))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

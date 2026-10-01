"""Deterministic trajectory evaluation for NetEval."""

from __future__ import annotations

import json
import re
from typing import Any


class Evaluator:
    """Score an agent trajectory against one deterministic evaluation case.

    Binary success requires an overall score of at least 0.80 and no critical
    failure. This intentionally makes a high aggregate score insufficient when
    the run used the wrong entity or port, stopped prematurely, misread tool
    output, made an unsupported important claim, or was malformed.
    """

    WEIGHTS = {
        "tool_selection": 0.20,
        "arguments": 0.20,
        "trajectory": 0.20,
        "grounding": 0.20,
        "final_answer": 0.15,
        "efficiency": 0.05,
    }

    FAILURE_ORDER = (
        "wrong_tool",
        "missing_required_tool",
        "wrong_arguments",
        "wrong_entity",
        "wrong_port",
        "wrong_tool_order",
        "premature_answer",
        "unnecessary_tool_call",
        "missing_step",
        "tool_result_misread",
        "unsupported_claim",
        "format_failure",
    )

    CRITICAL_FAILURES = {
        "missing_required_tool",
        "wrong_entity",
        "wrong_port",
        "premature_answer",
        "tool_result_misread",
        "unsupported_claim",
        "format_failure",
    }

    ENTITY_ARGUMENTS = {"source", "destination", "device"}

    def evaluate(
        self, eval_case: dict[str, Any], trajectory: dict[str, Any]
    ) -> dict[str, Any]:
        failures: set[str] = set()

        if not isinstance(eval_case, dict) or not isinstance(trajectory, dict):
            return self._format_failure_result(eval_case, trajectory)

        case_id = str(eval_case.get("id", "unknown"))
        steps_value = trajectory.get("steps", [])
        if not isinstance(steps_value, list):
            return self._format_failure_result(eval_case, trajectory)

        valid_steps: list[dict[str, Any]] = []
        for step in steps_value:
            if not isinstance(step, dict):
                failures.add("format_failure")
                continue
            if not isinstance(step.get("tool"), str):
                failures.add("format_failure")
                continue
            if not isinstance(step.get("arguments", {}), dict):
                failures.add("format_failure")
                continue
            if "result" not in step:
                failures.add("format_failure")
                continue
            valid_steps.append(step)

        tools_called = [step["tool"] for step in valid_steps]
        required_tools, missing_tools, allowed_tools = self._select_required_tools(
            eval_case, tools_called
        )
        unexpected_tools = self._unique(
            tool for tool in tools_called if tool not in allowed_tools
        )

        if missing_tools:
            failures.add("missing_required_tool")
        if unexpected_tools:
            failures.add("wrong_tool")

        required_coverage = self._coverage(required_tools, tools_called)
        tool_selection_score = self._clamp(
            required_coverage - (0.25 * len(unexpected_tools))
        )

        argument_score = self._score_arguments(
            eval_case,
            valid_steps,
            allowed_tools,
            failures,
        )

        actual_tool_call_count = len(valid_steps)
        reported_tool_call_count = trajectory.get(
            "tool_call_count", actual_tool_call_count
        )
        if not isinstance(reported_tool_call_count, int) or isinstance(
            reported_tool_call_count, bool
        ):
            failures.add("format_failure")
            reported_tool_call_count = actual_tool_call_count
        elif reported_tool_call_count != actual_tool_call_count:
            failures.add("format_failure")

        min_tool_calls = int(eval_case.get("min_tool_calls", 0))
        max_reasonable_tool_calls = int(
            eval_case.get("max_reasonable_tool_calls", max(min_tool_calls, 1))
        )
        completed = bool(trajectory.get("completed", True))
        final_answer = trajectory.get("final_answer")
        if final_answer is not None and not isinstance(final_answer, str):
            failures.add("format_failure")
            final_answer = ""
        final_answer = final_answer or ""

        missing_step = bool(missing_tools) or actual_tool_call_count < min_tool_calls
        if missing_step:
            failures.add("missing_step")
        if missing_step and final_answer:
            failures.add("premature_answer")
        if actual_tool_call_count > max_reasonable_tool_calls:
            failures.add("unnecessary_tool_call")

        order_score = self._score_explicit_order(
            eval_case, tools_called, failures
        )
        minimum_score = (
            1.0
            if min_tool_calls == 0
            else min(1.0, actual_tool_call_count / min_tool_calls)
        )
        maximum_score = (
            1.0
            if actual_tool_call_count <= max_reasonable_tool_calls
            else max_reasonable_tool_calls / actual_tool_call_count
        )
        trajectory_score = self._mean(
            [
                required_coverage,
                minimum_score,
                maximum_score,
                order_score,
                1.0 if completed else 0.0,
            ]
        )

        tool_results = [step["result"] for step in valid_steps]
        tool_results_text = self._normalized_json(tool_results)
        tool_outcomes = self._tool_outcomes(tool_results)
        query_text = self._normalize(eval_case.get("query", ""))
        answer_text = self._normalize(final_answer)

        required_evidence = [
            str(item) for item in eval_case.get("required_evidence", [])
        ]
        evidence_found: list[str] = []
        evidence_missing: list[str] = []
        for evidence in required_evidence:
            in_tools = self._evidence_in_tools(
                evidence, tool_results_text, tool_outcomes
            )
            directly_available = (
                self._normalize(evidence) in query_text
                and self._evidence_in_answer(evidence, answer_text)
            )
            if in_tools or directly_available:
                evidence_found.append(evidence)
            else:
                evidence_missing.append(evidence)
                if self._evidence_in_answer(evidence, answer_text):
                    failures.add("unsupported_claim")

        grounding_score = (
            1.0
            if not required_evidence
            else len(evidence_found) / len(required_evidence)
        )

        answer_outcome = self._classify_answer_outcome(final_answer)
        self._evaluate_answer_grounding(
            eval_case=eval_case,
            answer_text=answer_text,
            answer_outcome=answer_outcome,
            tool_results_text=tool_results_text,
            tool_outcomes=tool_outcomes,
            query_text=query_text,
            failures=failures,
        )
        final_answer_score = self._score_final_answer(
            eval_case,
            final_answer,
            answer_text,
            completed,
            failures,
        )

        efficiency_score = self._score_efficiency(
            actual_tool_call_count,
            min_tool_calls,
            max_reasonable_tool_calls,
        )

        scores = {
            "tool_selection": self._rounded(tool_selection_score),
            "arguments": self._rounded(argument_score),
            "trajectory": self._rounded(trajectory_score),
            "grounding": self._rounded(grounding_score),
            "final_answer": self._rounded(final_answer_score),
            "efficiency": self._rounded(efficiency_score),
        }
        overall_score = self._rounded(
            sum(scores[name] * weight for name, weight in self.WEIGHTS.items())
        )

        ordered_failures = [
            failure for failure in self.FAILURE_ORDER if failure in failures
        ]
        has_critical_failure = any(
            failure in self.CRITICAL_FAILURES for failure in ordered_failures
        )

        return {
            "case_id": case_id,
            "success": overall_score >= 0.80 and not has_critical_failure,
            "overall_score": overall_score,
            "scores": scores,
            "failures": ordered_failures,
            "details": {
                "tools_called": tools_called,
                "required_tools": required_tools,
                "missing_tools": missing_tools,
                "unexpected_tools": unexpected_tools,
                "evidence_found": evidence_found,
                "evidence_missing": evidence_missing,
                "tool_call_count": reported_tool_call_count,
                "max_reasonable_tool_calls": max_reasonable_tool_calls,
            },
            "latency_ms": trajectory.get("latency_ms"),
            "tokens": self._trajectory_tokens(trajectory),
            "tool_call_count": reported_tool_call_count,
        }

    def _select_required_tools(
        self, eval_case: dict[str, Any], tools_called: list[str]
    ) -> tuple[list[str], list[str], set[str]]:
        base_required = list(
            eval_case.get("required_tools")
            or eval_case.get("expected_tools")
            or []
        )
        candidate_sets = eval_case.get("required_tool_sets") or [base_required]
        candidates = [list(candidate) for candidate in candidate_sets]
        if not candidates:
            candidates = [[]]

        def candidate_rank(candidate: list[str]) -> tuple[float, int, int]:
            coverage = self._coverage(candidate, tools_called)
            missing_count = sum(tool not in tools_called for tool in candidate)
            return (coverage, -missing_count, -len(candidate))

        selected = max(candidates, key=candidate_rank)
        missing = [tool for tool in selected if tool not in tools_called]

        allowed_tools = set(eval_case.get("expected_tools", []))
        allowed_tools.update(base_required)
        allowed_tools.update(eval_case.get("allowed_optional_tools", []))
        for candidate in candidates:
            allowed_tools.update(candidate)

        return selected, missing, allowed_tools

    def _score_arguments(
        self,
        eval_case: dict[str, Any],
        steps: list[dict[str, Any]],
        allowed_tools: set[str],
        failures: set[str],
    ) -> float:
        required_arguments = eval_case.get("required_arguments", {})
        if not isinstance(required_arguments, dict):
            failures.add("format_failure")
            return 0.0
        if not required_arguments:
            return 1.0

        field_scores: list[float] = []
        relevant_steps = [step for step in steps if step["tool"] in allowed_tools]
        for field, expected_value in required_arguments.items():
            actual_values = [
                step["arguments"][field]
                for step in relevant_steps
                if field in step["arguments"]
            ]
            if not actual_values:
                failures.add("wrong_arguments")
                field_scores.append(0.0)
                continue

            matches = [
                self._argument_equal(actual, expected_value)
                for actual in actual_values
            ]
            if all(matches):
                field_scores.append(1.0)
                continue
            if any(matches):
                field_scores.append(0.5)
            else:
                field_scores.append(0.0)

            if field in self.ENTITY_ARGUMENTS:
                failures.add("wrong_entity")
            elif field == "port":
                failures.add("wrong_port")
            else:
                failures.add("wrong_arguments")

        return self._mean(field_scores)

    def _score_explicit_order(
        self,
        eval_case: dict[str, Any],
        tools_called: list[str],
        failures: set[str],
    ) -> float:
        required_order = eval_case.get("required_tool_order")
        if not required_order:
            return 1.0

        position = 0
        for tool in tools_called:
            if position < len(required_order) and tool == required_order[position]:
                position += 1
        if position == len(required_order):
            return 1.0

        failures.add("wrong_tool_order")
        return position / len(required_order)

    def _evaluate_answer_grounding(
        self,
        *,
        eval_case: dict[str, Any],
        answer_text: str,
        answer_outcome: str | None,
        tool_results_text: str,
        tool_outcomes: set[str],
        query_text: str,
        failures: set[str],
    ) -> None:
        if not answer_text or answer_outcome is None:
            return

        opposites = {
            "reachable": "blocked",
            "blocked": "reachable",
            "up": "down",
            "down": "up",
            "allow": "deny",
            "deny": "allow",
            "path_found": "no_path",
            "no_path": "path_found",
        }
        if opposites.get(answer_outcome) in tool_outcomes:
            failures.add("tool_result_misread")
            return

        supported_outcomes = set(tool_outcomes)
        if "deny" in tool_outcomes or "down" in tool_outcomes:
            supported_outcomes.add("blocked")
        if answer_outcome not in supported_outcomes and not self._query_supports_outcome(
            query_text, answer_outcome
        ):
            failures.add("unsupported_claim")

        for field in ("expected_blocker", "expected_rule_id"):
            expected_value = eval_case.get(field)
            if expected_value is None:
                continue
            normalized_value = self._normalize(expected_value)
            if (
                normalized_value in answer_text
                and normalized_value not in tool_results_text
                and normalized_value not in query_text
            ):
                failures.add("unsupported_claim")

    def _score_final_answer(
        self,
        eval_case: dict[str, Any],
        final_answer: str,
        answer_text: str,
        completed: bool,
        failures: set[str],
    ) -> float:
        if not final_answer.strip():
            if completed and eval_case.get("expected_outcome") is not None:
                failures.add("format_failure")
            return 0.0

        expected_outcome = str(eval_case.get("expected_outcome", ""))
        checks: list[float] = []
        if expected_outcome == "path_found":
            path_evidence = [
                str(item) for item in eval_case.get("required_evidence", [])
            ]
            checks.append(
                1.0
                if not path_evidence
                else sum(
                    self._normalize(item) in answer_text for item in path_evidence
                )
                / len(path_evidence)
            )
        else:
            checks.append(
                1.0
                if self._answer_matches_outcome(final_answer, expected_outcome)
                else 0.0
            )

        expected_blocker = eval_case.get("expected_blocker")
        if expected_blocker is not None:
            checks.append(
                1.0 if self._normalize(expected_blocker) in answer_text else 0.0
            )

        expected_rule_id = eval_case.get("expected_rule_id")
        if expected_rule_id is not None:
            checks.append(
                1.0 if self._normalize(expected_rule_id) in answer_text else 0.0
            )

        return self._mean(checks)

    @staticmethod
    def _score_efficiency(
        tool_call_count: int, min_tool_calls: int, max_reasonable_tool_calls: int
    ) -> float:
        if tool_call_count < min_tool_calls:
            if min_tool_calls == 0:
                return 1.0
            return tool_call_count / min_tool_calls
        if tool_call_count <= max_reasonable_tool_calls:
            return 1.0
        extra_calls = tool_call_count - max_reasonable_tool_calls
        return max(0.0, 1.0 - (0.25 * extra_calls))

    def _tool_outcomes(self, tool_results: list[Any]) -> set[str]:
        outcomes: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("reachable") is True:
                    outcomes.add("reachable")
                elif value.get("reachable") is False:
                    outcomes.add("blocked")
                if value.get("blocked") is True:
                    outcomes.add("blocked")
                elif value.get("blocked") is False:
                    outcomes.add("reachable")
                if value.get("path_found") is True:
                    outcomes.add("path_found")
                elif value.get("path_found") is False:
                    outcomes.add("no_path")

                status = self._normalize(value.get("status", ""))
                if status in {"up", "down"}:
                    outcomes.add(status)
                    if status == "down":
                        outcomes.add("blocked")

                action = self._normalize(value.get("action", ""))
                if action in {"allow", "deny"}:
                    outcomes.add(action)
                    if action == "deny":
                        outcomes.add("blocked")

                reason = self._normalize(value.get("reason", ""))
                if reason in {
                    "explicit_deny",
                    "default_deny",
                    "destination_down",
                    "no_path",
                }:
                    outcomes.add("blocked")
                if reason == "destination_down":
                    outcomes.add("down")
                if reason == "no_path":
                    outcomes.add("no_path")

                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        visit(tool_results)
        return outcomes

    def _evidence_in_tools(
        self, evidence: str, tool_results_text: str, tool_outcomes: set[str]
    ) -> bool:
        normalized = self._normalize(evidence)
        semantic_outcomes = {
            "reachable": "reachable",
            "blocked": "blocked",
            "unreachable": "blocked",
            "up": "up",
            "down": "down",
            "allow": "allow",
            "deny": "deny",
            "path_found": "path_found",
            "no_path": "no_path",
        }
        if normalized in semantic_outcomes:
            return semantic_outcomes[normalized] in tool_outcomes
        return normalized in tool_results_text

    def _evidence_in_answer(self, evidence: str, answer_text: str) -> bool:
        normalized = self._normalize(evidence)
        if normalized in {"reachable", "blocked", "down", "up", "allow", "deny"}:
            return self._answer_matches_outcome(answer_text, normalized)
        return normalized in answer_text

    def _answer_matches_outcome(self, answer: str, expected_outcome: str) -> bool:
        outcome = self._normalize(expected_outcome)
        classified = self._classify_answer_outcome(answer)
        if outcome == "unreachable":
            outcome = "blocked"
        return classified == outcome

    def _classify_answer_outcome(self, answer: str) -> str | None:
        text = self._normalize(answer)
        if not text:
            return None
        words = set(re.findall(r"[a-z0-9_-]+", text))

        if any(marker in text for marker in ("not blocked", "not denied")):
            return "reachable"
        if any(
            marker in text
            for marker in (
                "cannot reach",
                "can't reach",
                "cannot connect",
                "can't connect",
                "unable to",
                "unreachable",
                "blocked",
                "explicit deny",
                "explicit_deny",
                "default deny",
                "default_deny",
                "denied",
            )
        ):
            return "blocked"
        if "down" in words or any(
            marker in text for marker in ("offline", "unavailable")
        ):
            return "down"
        if "up" in words or any(marker in text for marker in ("online", "healthy")):
            return "up"
        if any(
            marker in text
            for marker in (
                "can reach",
                "can connect",
                "is reachable",
                "connection is allowed",
                "traffic is allowed",
                "should work",
            )
        ) or "reachable" in words:
            return "reachable"
        if any(marker in text for marker in ("deny", "denies")):
            return "deny"
        if any(marker in text for marker in ("allow", "permits", "permitted")):
            return "allow"
        return None

    @staticmethod
    def _query_supports_outcome(query_text: str, outcome: str) -> bool:
        if outcome == "blocked":
            return any(
                marker in query_text
                for marker in (
                    "can't",
                    "cannot",
                    "unable",
                    "blocked from",
                    "preventing",
                    "failure",
                )
            )
        return False

    @staticmethod
    def _argument_equal(actual: Any, expected: Any) -> bool:
        if isinstance(expected, str) and isinstance(actual, str):
            return actual.casefold() == expected.casefold()
        return actual == expected and type(actual) is type(expected)

    @staticmethod
    def _coverage(required_tools: list[str], tools_called: list[str]) -> float:
        if not required_tools:
            return 1.0
        return sum(tool in tools_called for tool in required_tools) / len(required_tools)

    @staticmethod
    def _unique(values) -> list[str]:
        return list(dict.fromkeys(values))

    @staticmethod
    def _normalize(value: Any) -> str:
        return str(value).casefold().replace("‑", "-").strip()

    @classmethod
    def _normalized_json(cls, value: Any) -> str:
        return cls._normalize(json.dumps(value, sort_keys=True, default=str))

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 1.0

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _rounded(cls, value: float) -> float:
        return round(cls._clamp(value), 4)

    @staticmethod
    def _trajectory_tokens(trajectory: dict[str, Any]) -> Any:
        if "tokens" in trajectory:
            return trajectory["tokens"]
        if any(
            key in trajectory
            for key in ("input_tokens", "output_tokens", "total_tokens")
        ):
            return {
                "input": trajectory.get("input_tokens"),
                "output": trajectory.get("output_tokens"),
                "total": trajectory.get("total_tokens"),
            }
        return None

    def _format_failure_result(
        self, eval_case: Any, trajectory: Any
    ) -> dict[str, Any]:
        case_id = (
            str(eval_case.get("id", "unknown"))
            if isinstance(eval_case, dict)
            else "unknown"
        )
        scores = {name: 0.0 for name in self.WEIGHTS}
        latency_ms = (
            trajectory.get("latency_ms") if isinstance(trajectory, dict) else None
        )
        tokens = (
            self._trajectory_tokens(trajectory)
            if isinstance(trajectory, dict)
            else None
        )
        return {
            "case_id": case_id,
            "success": False,
            "overall_score": 0.0,
            "scores": scores,
            "failures": ["format_failure"],
            "details": {
                "tools_called": [],
                "required_tools": [],
                "missing_tools": [],
                "unexpected_tools": [],
                "evidence_found": [],
                "evidence_missing": [],
                "tool_call_count": 0,
                "max_reasonable_tool_calls": 0,
            },
            "latency_ms": latency_ms,
            "tokens": tokens,
            "tool_call_count": 0,
        }

import json
from pathlib import Path

import pytest

from src.evaluator import Evaluator


EVAL_CASES_PATH = Path(__file__).parents[1] / "data" / "eval_cases.json"


@pytest.fixture(scope="module")
def cases() -> dict[str, dict]:
    loaded = json.loads(EVAL_CASES_PATH.read_text(encoding="utf-8"))
    return {case["id"]: case for case in loaded}


@pytest.fixture
def evaluator() -> Evaluator:
    return Evaluator()


def trajectory(steps: list[dict], final_answer: str) -> dict:
    return {
        "steps": steps,
        "final_answer": final_answer,
        "tool_call_count": len(steps),
        "completed": True,
        "latency_ms": 125.5,
        "tokens": {"input": 100, "output": 25, "total": 125},
    }


def admin_reachability_step(port: int = 5432) -> dict:
    return {
        "tool": "check_reachability",
        "arguments": {
            "source": "admin-net",
            "destination": "orders-db",
            "port": port,
            "protocol": "tcp",
        },
        "result": {
            "reachable": False,
            "source": "admin-net",
            "destination": "orders-db",
            "port": port,
            "protocol": "tcp",
            "path": ["admin-net", "internal-fw", "orders-db"],
            "blocking_device": "internal-fw",
            "rule_id": "acl-17" if port == 5432 else None,
            "reason": "explicit_deny" if port == 5432 else "default_deny",
        },
    }


def admin_acl_step(port: int = 5432) -> dict:
    matched_rules = []
    if port == 5432:
        matched_rules = [
            {
                "id": "acl-17",
                "source": "admin-net",
                "destination": "orders-db",
                "port": 5432,
                "protocol": "tcp",
                "action": "deny",
                "priority": 50,
            }
        ]
    return {
        "tool": "inspect_acl",
        "arguments": {
            "device": "internal-fw",
            "source": "admin-net",
            "destination": "orders-db",
            "port": port,
            "protocol": "tcp",
        },
        "result": {
            "device": "internal-fw",
            "matched_rules": matched_rules,
        },
    }


def test_perfect_admin_root_cause_trajectory_succeeds(
    evaluator: Evaluator, cases: dict[str, dict]
) -> None:
    run = trajectory(
        [admin_reachability_step(), admin_acl_step()],
        "Traffic is blocked by internal-fw because acl-17 is an explicit deny.",
    )

    result = evaluator.evaluate(cases["diag_01"], run)

    assert result["success"] is True
    assert result["overall_score"] >= 0.9
    assert result["failures"] == []


def test_alternative_blocking_hop_strategy_is_not_penalized(
    evaluator: Evaluator, cases: dict[str, dict]
) -> None:
    blocking_step = {
        "tool": "find_blocking_hop",
        "arguments": {
            "source": "admin-net",
            "destination": "orders-db",
            "port": 5432,
            "protocol": "tcp",
        },
        "result": {
            "blocked": True,
            "reason": "explicit_deny",
            "blocking_device": "internal-fw",
            "rule_id": "acl-17",
        },
    }
    run = trajectory(
        [blocking_step, admin_acl_step()],
        "internal-fw blocks the flow with explicit deny rule acl-17.",
    )

    result = evaluator.evaluate(cases["diag_01"], run)

    assert result["success"] is True
    assert "missing_required_tool" not in result["failures"]
    assert "wrong_tool_order" not in result["failures"]


def test_premature_path_only_answer_is_classified(
    evaluator: Evaluator, cases: dict[str, dict]
) -> None:
    path_step = {
        "tool": "trace_path",
        "arguments": {"source": "admin-net", "destination": "orders-db"},
        "result": {
            "source": "admin-net",
            "destination": "orders-db",
            "path": ["admin-net", "internal-fw", "orders-db"],
            "path_found": True,
        },
    }
    run = trajectory(
        [path_step],
        "A path exists, so traffic should work.",
    )

    result = evaluator.evaluate(cases["diag_01"], run)

    assert result["success"] is False
    assert "missing_required_tool" in result["failures"]
    assert "missing_step" in result["failures"]
    assert "premature_answer" in result["failures"]
    assert "unsupported_claim" in result["failures"]


def test_wrong_port_is_classified(
    evaluator: Evaluator, cases: dict[str, dict]
) -> None:
    run = trajectory(
        [admin_reachability_step(port=443), admin_acl_step(port=443)],
        "Traffic is blocked by internal-fw under the default deny policy.",
    )

    result = evaluator.evaluate(cases["diag_01"], run)

    assert result["success"] is False
    assert "wrong_port" in result["failures"]
    assert result["scores"]["arguments"] < 1.0


def test_unnecessary_calls_reduce_efficiency(
    evaluator: Evaluator, cases: dict[str, dict]
) -> None:
    steps = [
        {
            "tool": "trace_path",
            "arguments": {"source": "billing-api", "destination": "billing-db"},
            "result": {"path": ["billing-api", "billing-db"], "path_found": True},
        },
        {
            "tool": "check_reachability",
            "arguments": {
                "source": "billing-api",
                "destination": "billing-db",
                "port": 5432,
            },
            "result": {"reachable": False, "reason": "destination_down"},
        },
        {
            "tool": "get_device_state",
            "arguments": {"device": "billing-db"},
            "result": {"device": "billing-db", "status": "down"},
        },
        {
            "tool": "inspect_acl",
            "arguments": {"device": "internal-fw"},
            "result": {"device": "internal-fw", "matched_rules": []},
        },
    ]
    run = trajectory(steps, "billing-db is down.")

    result = evaluator.evaluate(cases["state_01"], run)

    assert "unnecessary_tool_call" in result["failures"]
    assert result["scores"]["efficiency"] < 1.0
    assert result["details"]["tool_call_count"] == 4


def test_correct_billing_db_down_diagnosis_succeeds(
    evaluator: Evaluator, cases: dict[str, dict]
) -> None:
    steps = [
        {
            "tool": "check_reachability",
            "arguments": {
                "source": "billing-api",
                "destination": "billing-db",
                "port": 5432,
                "protocol": "tcp",
            },
            "result": {
                "reachable": False,
                "blocking_device": "billing-db",
                "rule_id": None,
                "reason": "destination_down",
            },
        },
        {
            "tool": "get_device_state",
            "arguments": {"device": "billing-db"},
            "result": {
                "device": "billing-db",
                "type": "database",
                "status": "down",
                "zone": "data",
                "services": [],
            },
        },
    ]
    run = trajectory(
        steps,
        "billing-api cannot reach billing-db because billing-db is down.",
    )

    result = evaluator.evaluate(cases["diag_02"], run)

    assert result["success"] is True
    assert result["failures"] == []


def test_pure_path_question_needs_only_trace_path(
    evaluator: Evaluator, cases: dict[str, dict]
) -> None:
    path_step = {
        "tool": "trace_path",
        "arguments": {"source": "admin-net", "destination": "orders-db"},
        "result": {
            "source": "admin-net",
            "destination": "orders-db",
            "path": ["admin-net", "internal-fw", "orders-db"],
            "path_found": True,
        },
    }
    run = trajectory(
        [path_step],
        "The path is admin-net to internal-fw to orders-db.",
    )

    result = evaluator.evaluate(cases["path_01"], run)

    assert result["success"] is True
    assert result["failures"] == []
    assert result["details"]["tools_called"] == ["trace_path"]


def test_latency_tokens_and_tool_count_are_preserved(
    evaluator: Evaluator, cases: dict[str, dict]
) -> None:
    path_step = {
        "tool": "trace_path",
        "arguments": {"source": "admin-net", "destination": "orders-db"},
        "result": {
            "path": ["admin-net", "internal-fw", "orders-db"],
            "path_found": True,
        },
    }
    run = trajectory(
        [path_step],
        "The path is admin-net to internal-fw to orders-db.",
    )

    result = evaluator.evaluate(cases["path_01"], run)

    assert result["latency_ms"] == 125.5
    assert result["tokens"] == {"input": 100, "output": 25, "total": 125}
    assert result["tool_call_count"] == 1


def test_malformed_trajectory_returns_format_failure(
    evaluator: Evaluator, cases: dict[str, dict]
) -> None:
    result = evaluator.evaluate(cases["path_01"], {"steps": "not-a-list"})

    assert result["success"] is False
    assert result["overall_score"] == 0.0
    assert result["failures"] == ["format_failure"]

import pytest

from src.runner import aggregate_benchmark


def case_result(
    *,
    case_id: str,
    category: str,
    success: bool,
    score: float,
    tool_calls: int,
    latency_ms: float,
    tokens: dict[str, int],
    failures: list[str],
    component_value: float,
) -> dict:
    return {
        "case_id": case_id,
        "category": category,
        "query": "test query",
        "success": success,
        "overall_score": score,
        "scores": {
            "tool_selection": component_value,
            "arguments": component_value,
            "trajectory": component_value,
            "grounding": component_value,
            "final_answer": component_value,
            "efficiency": component_value,
        },
        "failures": failures,
        "tools_called": [],
        "tool_call_count": tool_calls,
        "latency_ms": latency_ms,
        "tokens": tokens,
        "final_answer": "answer",
        "trajectory_file_path": "trajectory.json",
    }


@pytest.fixture
def benchmark() -> dict:
    results = [
        case_result(
            case_id="reach_01",
            category="simple_reachability",
            success=True,
            score=1.0,
            tool_calls=1,
            latency_ms=100.0,
            tokens={"input": 100, "output": 20, "total": 120},
            failures=[],
            component_value=1.0,
        ),
        case_result(
            case_id="reach_02",
            category="simple_reachability",
            success=False,
            score=0.5,
            tool_calls=3,
            latency_ms=300.0,
            tokens={"input": 300, "output": 60, "total": 360},
            failures=["missing_required_tool", "premature_answer"],
            component_value=0.5,
        ),
        case_result(
            case_id="state_01",
            category="device_state",
            success=True,
            score=0.8,
            tool_calls=2,
            latency_ms=200.0,
            tokens={"input": 200, "output": 40, "total": 240},
            failures=["unnecessary_tool_call"],
            component_value=0.75,
        ),
    ]
    return aggregate_benchmark(results, agent_version="v1", model="test-model")


def test_success_rate_and_average_score(benchmark: dict) -> None:
    assert benchmark["summary"]["successful_cases"] == 2
    assert benchmark["summary"]["failed_cases"] == 1
    assert benchmark["summary"]["task_success_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert benchmark["summary"]["average_overall_score"] == pytest.approx(
        0.7667, abs=1e-4
    )


def test_component_averages(benchmark: dict) -> None:
    expected = pytest.approx(0.75, abs=1e-4)

    assert all(value == expected for value in benchmark["component_scores"].values())


def test_failure_counts(benchmark: dict) -> None:
    counts = benchmark["failure_counts"]

    assert counts["missing_required_tool"] == 1
    assert counts["premature_answer"] == 1
    assert counts["unnecessary_tool_call"] == 1
    assert counts["wrong_port"] == 0


def test_category_metrics(benchmark: dict) -> None:
    reachability = benchmark["category_metrics"]["simple_reachability"]
    device_state = benchmark["category_metrics"]["device_state"]

    assert reachability["success_rate"] == 0.5
    assert reachability["average_score"] == 0.75
    assert device_state["success_rate"] == 1.0
    assert device_state["average_score"] == 0.8


def test_average_tokens_and_tool_calls(benchmark: dict) -> None:
    summary = benchmark["summary"]

    assert summary["average_tool_calls"] == 2.0
    assert summary["average_latency_ms"] == 200.0
    assert summary["average_tokens"] == {
        "input": 200.0,
        "output": 40.0,
        "total": 240.0,
    }


def test_provider_errors_are_aggregated() -> None:
    provider_failure = case_result(
        case_id="reach_01",
        category="simple_reachability",
        success=False,
        score=0.0,
        tool_calls=0,
        latency_ms=50.0,
        tokens={"input": 0, "output": 0, "total": 0},
        failures=["format_failure"],
        component_value=0.0,
    )
    provider_failure["provider_error"] = True

    result = aggregate_benchmark(
        [provider_failure], agent_version="v2", model="test-model"
    )

    assert result["summary"]["provider_error_count"] == 1

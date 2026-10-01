import hashlib
import json
from copy import deepcopy

import pytest

from src.report import (
    compare_benchmarks,
    compare_case_results,
    create_comparison,
)


def benchmark(version: str, cases: list[dict], *, provider_errors: int) -> dict:
    return {
        "agent_version": version,
        "model": "same-model",
        "total_cases": len(cases),
        "summary": {
            "task_success_rate": sum(case["success"] for case in cases) / len(cases),
            "average_overall_score": sum(case["overall_score"] for case in cases)
            / len(cases),
            "average_tool_calls": 1.5,
            "average_latency_ms": 100.0 if version == "v1" else 90.0,
            "average_tokens": {"input": 100, "output": 20, "total": 120},
            "provider_error_count": provider_errors,
        },
        "component_scores": {
            "tool_selection": 0.5 if version == "v1" else 0.75,
            "arguments": 0.6,
            "trajectory": 0.7,
            "grounding": 0.8,
            "final_answer": 0.9,
            "efficiency": 1.0,
        },
        "failure_counts": {
            "wrong_tool": 2 if version == "v1" else 1,
            "missing_required_tool": 1,
            "wrong_arguments": 0,
            "wrong_entity": 0,
            "wrong_port": 0,
            "wrong_tool_order": 0,
            "premature_answer": 0,
            "unnecessary_tool_call": 0,
            "missing_step": 0,
            "tool_result_misread": 0,
            "unsupported_claim": 0,
            "format_failure": 0,
        },
        "category_metrics": {
            "simple_reachability": {
                "success_rate": 0.5 if version == "v1" else 1.0,
                "average_score": 0.6 if version == "v1" else 0.8,
            }
        },
        "cases": cases,
    }


def case(
    case_id: str,
    *,
    success: bool,
    score: float,
    failures: list[str] | None = None,
    tools: list[str] | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "category": "simple_reachability",
        "success": success,
        "overall_score": score,
        "failures": failures or [],
        "tools_called": tools or ["check_reachability"],
    }


@pytest.fixture
def benchmark_pair() -> tuple[dict, dict]:
    v1_cases = [
        case("improved", success=False, score=0.60, failures=["wrong_tool"]),
        case("unchanged", success=True, score=0.90),
        case("regressed", success=True, score=0.90),
    ]
    v2_cases = [
        case("improved", success=True, score=0.85),
        case("unchanged", success=True, score=0.895),
        case("regressed", success=False, score=0.70, failures=["missing_step"]),
    ]
    return (
        benchmark("v1", v1_cases, provider_errors=2),
        benchmark("v2", v2_cases, provider_errors=1),
    )


def test_benchmark_comparison_calculates_deltas(benchmark_pair: tuple[dict, dict]) -> None:
    comparison = compare_benchmarks(*benchmark_pair)

    assert comparison["component_scores"]["tool_selection"] == {
        "v1": 0.5,
        "v2": 0.75,
        "delta": 0.25,
    }
    assert comparison["operational_metrics"]["average_latency_ms"]["delta"] == -10.0


def test_improved_case_classification() -> None:
    result = compare_case_results(
        case("case_1", success=False, score=0.7),
        case("case_1", success=True, score=0.8),
    )

    assert result["classification"] == "improved"


def test_unchanged_case_classification_with_tolerance() -> None:
    result = compare_case_results(
        case("case_1", success=True, score=0.8),
        case("case_1", success=True, score=0.805),
    )

    assert result["classification"] == "unchanged"


def test_regressed_case_classification() -> None:
    result = compare_case_results(
        case("case_1", success=True, score=0.8),
        case("case_1", success=False, score=0.79),
    )

    assert result["classification"] == "regressed"


def test_provider_error_counts_are_preserved(benchmark_pair: tuple[dict, dict]) -> None:
    comparison = compare_benchmarks(*benchmark_pair)

    assert comparison["provider_errors"] == {"v1": 2.0, "v2": 1.0, "delta": -1.0}


def test_category_deltas_are_correct(benchmark_pair: tuple[dict, dict]) -> None:
    comparison = compare_benchmarks(*benchmark_pair)
    category = comparison["category_metrics"]["simple_reachability"]

    assert category["success_rate"]["delta"] == 0.5
    assert category["average_score"]["delta"] == 0.2


def test_failure_taxonomy_deltas_are_correct(
    benchmark_pair: tuple[dict, dict],
) -> None:
    comparison = compare_benchmarks(*benchmark_pair)

    assert comparison["failure_taxonomy"]["wrong_tool"] == {
        "v1": 2.0,
        "v2": 1.0,
        "delta": -1.0,
    }


def test_comparison_json_writes_successfully(
    tmp_path, benchmark_pair: tuple[dict, dict]
) -> None:
    v1_path = tmp_path / "benchmark_v1.json"
    v2_path = tmp_path / "benchmark_v2.json"
    output_path = tmp_path / "comparison.json"
    v1_path.write_text(json.dumps(benchmark_pair[0]), encoding="utf-8")
    v2_path.write_text(json.dumps(benchmark_pair[1]), encoding="utf-8")

    comparison = create_comparison(v1_path, v2_path, output_path)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["cases"] == comparison["cases"]


def test_v1_benchmark_remains_unchanged(
    tmp_path, benchmark_pair: tuple[dict, dict]
) -> None:
    v1_path = tmp_path / "benchmark_v1.json"
    v2_path = tmp_path / "benchmark_v2.json"
    output_path = tmp_path / "comparison.json"
    v1_path.write_text(json.dumps(benchmark_pair[0], indent=2), encoding="utf-8")
    v2_path.write_text(json.dumps(benchmark_pair[1], indent=2), encoding="utf-8")
    before = hashlib.sha256(v1_path.read_bytes()).hexdigest()

    create_comparison(v1_path, v2_path, output_path)

    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == before
    assert deepcopy(benchmark_pair[0]) == json.loads(v1_path.read_text(encoding="utf-8"))

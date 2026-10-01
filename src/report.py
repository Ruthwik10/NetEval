"""Deterministic benchmark comparison and terminal reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.runner import CATEGORY_ORDER, FAILURE_TYPES


DEFAULT_SCORE_TOLERANCE = 0.01


def load_benchmark(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as benchmark_file:
        benchmark = json.load(benchmark_file)
    if not isinstance(benchmark, dict):
        raise ValueError(f"Benchmark at {path} must be a JSON object.")
    return benchmark


def compare_benchmarks(
    v1: dict[str, Any],
    v2: dict[str, Any],
    *,
    score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
) -> dict[str, Any]:
    """Compare two benchmark dictionaries without changing either input."""
    if score_tolerance < 0:
        raise ValueError("score_tolerance must not be negative.")
    if v1.get("model") != v2.get("model"):
        raise ValueError("V1 and V2 benchmarks must use the same model.")

    v1_cases = _case_map(v1)
    v2_cases = _case_map(v2)
    if set(v1_cases) != set(v2_cases):
        missing_v2 = sorted(set(v1_cases) - set(v2_cases))
        missing_v1 = sorted(set(v2_cases) - set(v1_cases))
        raise ValueError(
            "Benchmarks must contain identical case IDs. "
            f"Missing from V2: {missing_v2}; missing from V1: {missing_v1}."
        )

    case_comparisons = [
        compare_case_results(
            v1_cases[case_id],
            v2_cases[case_id],
            score_tolerance=score_tolerance,
        )
        for case_id in v1_cases
    ]
    classification_counts = {
        classification: sum(
            case["classification"] == classification for case in case_comparisons
        )
        for classification in ("improved", "unchanged", "regressed")
    }

    summary_metrics = {
        "task_success_rate": _metric_delta(
            v1.get("summary", {}).get("task_success_rate"),
            v2.get("summary", {}).get("task_success_rate"),
        ),
        "average_overall_score": _metric_delta(
            v1.get("summary", {}).get("average_overall_score"),
            v2.get("summary", {}).get("average_overall_score"),
        ),
    }
    component_scores = {
        component: _metric_delta(
            v1.get("component_scores", {}).get(component),
            v2.get("component_scores", {}).get(component),
        )
        for component in (
            "tool_selection",
            "arguments",
            "trajectory",
            "grounding",
            "final_answer",
            "efficiency",
        )
    }
    operational_metrics = {
        "average_tool_calls": _metric_delta(
            v1.get("summary", {}).get("average_tool_calls"),
            v2.get("summary", {}).get("average_tool_calls"),
        ),
        "average_latency_ms": _metric_delta(
            v1.get("summary", {}).get("average_latency_ms"),
            v2.get("summary", {}).get("average_latency_ms"),
        ),
        "average_total_tokens": _metric_delta(
            v1.get("summary", {}).get("average_tokens", {}).get("total"),
            v2.get("summary", {}).get("average_tokens", {}).get("total"),
        ),
    }
    provider_errors = _metric_delta(
        _provider_error_count(v1),
        _provider_error_count(v2),
    )
    failure_taxonomy = {
        failure: _metric_delta(
            v1.get("failure_counts", {}).get(failure, 0),
            v2.get("failure_counts", {}).get(failure, 0),
        )
        for failure in FAILURE_TYPES
    }
    category_metrics = {
        category: {
            "success_rate": _metric_delta(
                v1.get("category_metrics", {})
                .get(category, {})
                .get("success_rate"),
                v2.get("category_metrics", {})
                .get(category, {})
                .get("success_rate"),
            ),
            "average_score": _metric_delta(
                v1.get("category_metrics", {})
                .get(category, {})
                .get("average_score"),
                v2.get("category_metrics", {})
                .get(category, {})
                .get("average_score"),
            ),
        }
        for category in CATEGORY_ORDER
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": v1.get("model"),
        "v1_agent_version": v1.get("agent_version"),
        "v2_agent_version": v2.get("agent_version"),
        "total_cases": len(case_comparisons),
        "score_tolerance": score_tolerance,
        "summary_metrics": summary_metrics,
        "component_scores": component_scores,
        "operational_metrics": operational_metrics,
        "provider_errors": provider_errors,
        "failure_taxonomy": failure_taxonomy,
        "category_metrics": category_metrics,
        "case_summary": classification_counts,
        "cases": case_comparisons,
        "known_weak_cases": [
            case for case in case_comparisons if not case["v1_success"]
        ],
    }


def compare_case_results(
    v1_case: dict[str, Any],
    v2_case: dict[str, Any],
    *,
    score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
) -> dict[str, Any]:
    if v1_case.get("case_id") != v2_case.get("case_id"):
        raise ValueError("Cannot compare case results with different IDs.")

    v1_success = bool(v1_case.get("success"))
    v2_success = bool(v2_case.get("success"))
    v1_score = _number(v1_case.get("overall_score"))
    v2_score = _number(v2_case.get("overall_score"))
    delta = _rounded(v2_score - v1_score)

    if v2_success and not v1_success:
        classification = "improved"
    elif v1_success and not v2_success:
        classification = "regressed"
    elif delta > score_tolerance:
        classification = "improved"
    elif delta < -score_tolerance:
        classification = "regressed"
    else:
        classification = "unchanged"

    return {
        "case_id": str(v1_case["case_id"]),
        "category": v2_case.get("category", v1_case.get("category")),
        "v1_success": v1_success,
        "v2_success": v2_success,
        "v1_score": v1_score,
        "v2_score": v2_score,
        "delta": delta,
        "classification": classification,
        "v1_failures": list(v1_case.get("failures", [])),
        "v2_failures": list(v2_case.get("failures", [])),
        "v1_tools": list(v1_case.get("tools_called", [])),
        "v2_tools": list(v2_case.get("tools_called", [])),
    }


def save_comparison(comparison: dict[str, Any], path: str | Path) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as comparison_file:
        json.dump(comparison, comparison_file, indent=2, ensure_ascii=False)
        comparison_file.write("\n")
    return str(output_path.resolve())


def create_comparison(
    v1_path: str | Path,
    v2_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    v1 = load_benchmark(v1_path)
    v2 = load_benchmark(v2_path)
    comparison = compare_benchmarks(v1, v2)
    save_comparison(comparison, output_path)
    return comparison


def print_comparison_report(comparison: dict[str, Any]) -> None:
    print("\nNetEval V1 vs V2")
    print(f"\nModel: {comparison['model']}")
    print("\nMetric                     V1          V2       Delta")
    print("------------------------------------------------------")
    rows = [
        ("Task success", comparison["summary_metrics"]["task_success_rate"], True),
        ("Overall score", comparison["summary_metrics"]["average_overall_score"], False),
        *[
            (label, comparison["component_scores"][key], True)
            for key, label in (
                ("tool_selection", "Tool selection"),
                ("arguments", "Arguments"),
                ("trajectory", "Trajectory"),
                ("grounding", "Grounding"),
                ("final_answer", "Final answer"),
                ("efficiency", "Efficiency"),
            )
        ],
        ("Avg tool calls", comparison["operational_metrics"]["average_tool_calls"], False),
        ("Avg total tokens", comparison["operational_metrics"]["average_total_tokens"], False),
        ("Avg latency ms", comparison["operational_metrics"]["average_latency_ms"], False),
    ]
    for label, metric, percent in rows:
        if percent:
            values = (
                f"{metric['v1']:.1%}",
                f"{metric['v2']:.1%}",
                f"{metric['delta']:+.1%}",
            )
        else:
            values = (
                f"{metric['v1']:.3f}",
                f"{metric['v2']:.3f}",
                f"{metric['delta']:+.3f}",
            )
        print(f"{label:<22} {values[0]:>10} {values[1]:>10} {values[2]:>10}")

    cases = comparison["case_summary"]
    print("\nCases:")
    print(f"Improved: {cases['improved']}")
    print(f"Unchanged: {cases['unchanged']}")
    print(f"Regressed: {cases['regressed']}")

    provider = comparison["provider_errors"]
    print("\nProvider errors:")
    print(f"V1: {int(provider['v1'])}")
    print(f"V2: {int(provider['v2'])}")

    print("\nFailure taxonomy (V1 -> V2, delta):")
    for failure, metric in comparison["failure_taxonomy"].items():
        print(
            f"- {failure}: {int(metric['v1'])} -> {int(metric['v2'])} "
            f"({metric['delta']:+.0f})"
        )

    print("\nCategories (success rate; average score):")
    for category, metrics in comparison["category_metrics"].items():
        success = metrics["success_rate"]
        score = metrics["average_score"]
        print(
            f"- {category}: {success['v1']:.1%} -> {success['v2']:.1%} "
            f"({success['delta']:+.1%}); {score['v1']:.3f} -> "
            f"{score['v2']:.3f} ({score['delta']:+.3f})"
        )

    _print_case_group("Top improvements", comparison["cases"], "improved")
    remaining = [case for case in comparison["cases"] if not case["v2_success"]]
    print("\nRemaining V2 failures:")
    if remaining:
        for case in remaining:
            failures = ", ".join(case["v2_failures"]) or "none"
            print(f"- {case['case_id']}: {case['v2_score']:.3f} ({failures})")
    else:
        print("- none")
    _print_case_group("Regressions", comparison["cases"], "regressed")

    print("\nKnown V1 weak cases:")
    for case in comparison["known_weak_cases"]:
        print(
            f"- {case['case_id']}: {case['classification']}, "
            f"{case['v1_score']:.3f} -> {case['v2_score']:.3f}"
        )


def print_regression_report(regression: dict[str, Any]) -> None:
    print("\nNetEval Regression Suite")
    print(f"\nAgent: {regression['agent_version']}")
    print(f"\nRegression cases: {regression['total_cases']}")
    print(f"Passing: {regression['passing']}")
    print(f"Failing: {regression['failing']}")
    print(f"Previously fixed cases regressed: {regression['regressed']}")

    for case in regression["cases"]:
        original = case["original_v1_failure"]
        status = "PASS" if case["regression_pass"] else "FAIL"
        print(f"\n{case['case_id']}")
        print(f"V1: {'passed' if original['success'] else 'failed'}")
        print(f"{regression['agent_version'].upper()} regression: {status}")
        print(f"Score: {case['score']:.3f}")
        print(f"Original issue: {case['reason']}")
        if not case["regression_pass"]:
            print("REGRESSION DETECTED")
            failures = ", ".join(case["acceptance_failures"]) or "none"
            print(f"Acceptance failures: {failures}")
            if case["missing_required_tools"]:
                print(
                    "Missing required tools: "
                    + ", ".join(case["missing_required_tools"])
                )

    print("\nKnown unresolved / regressed cases:")
    unresolved = regression.get("known_unresolved_or_regressed_cases", [])
    if unresolved:
        for case in unresolved:
            failures = ", ".join(case["v2_failures"]) or "none"
            print(
                f"- {case['case_id']}: V1 "
                f"{'passed' if case['v1_success'] else 'failed'} "
                f"({case['v1_score']:.3f}), V2 "
                f"{'passed' if case['v2_success'] else 'failed'} "
                f"({case['v2_score']:.3f}); failures: {failures}"
            )
    else:
        print("- none")

    print("\nImproved but not yet fixed:")
    improved = regression.get("improved_but_not_fixed", [])
    if improved:
        for case in improved:
            failures = ", ".join(case["v2_failures"]) or "none"
            print(
                f"- {case['case_id']}: {case['v1_score']:.3f} -> "
                f"{case['v2_score']:.3f}; V2 failures: {failures}"
            )
    else:
        print("- none")


def _print_case_group(
    title: str,
    cases: list[dict[str, Any]],
    classification: str,
) -> None:
    selected = [case for case in cases if case["classification"] == classification]
    selected.sort(key=lambda case: case["delta"], reverse=classification == "improved")
    print(f"\n{title}:")
    if selected:
        for case in selected:
            print(
                f"- {case['case_id']}: {case['v1_score']:.3f} -> "
                f"{case['v2_score']:.3f} ({case['delta']:+.3f})"
            )
    else:
        print("- none")


def _case_map(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = benchmark.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Benchmark must contain a cases list.")
    mapped: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id:
            raise ValueError("Every benchmark case must have a case_id.")
        if case_id in mapped:
            raise ValueError(f"Duplicate benchmark case ID: {case_id}.")
        mapped[case_id] = case
    return mapped


def _provider_error_count(benchmark: dict[str, Any]) -> int:
    summary_count = benchmark.get("summary", {}).get("provider_error_count")
    if isinstance(summary_count, (int, float)) and not isinstance(summary_count, bool):
        return int(summary_count)
    return sum(
        bool(case.get("provider_error"))
        or case.get("run_status") == "api_error"
        or _number(case.get("retry_count")) > 0
        for case in benchmark.get("cases", [])
    )


def _metric_delta(v1: Any, v2: Any) -> dict[str, float]:
    v1_value = _number(v1)
    v2_value = _number(v2)
    return {
        "v1": _rounded(v1_value),
        "v2": _rounded(v2_value),
        "delta": _rounded(v2_value - v1_value),
    }


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _rounded(value: float) -> float:
    return round(value, 4)

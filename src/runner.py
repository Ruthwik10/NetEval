"""Sequential, versioned benchmark runner for NetEval."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent import ConfigurationError, NetworkAgent
from src.evaluator import Evaluator
from src.trajectory_store import TrajectoryStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_CASES_PATH = PROJECT_ROOT / "data" / "eval_cases.json"
DEFAULT_REGRESSION_CASES_PATH = PROJECT_ROOT / "data" / "regression_cases.json"
DEFAULT_TRAJECTORY_DIR = PROJECT_ROOT / "results" / "trajectories"
DEFAULT_V1_OUTPUT_PATH = PROJECT_ROOT / "results" / "benchmark_v1.json"
DEFAULT_V2_OUTPUT_PATH = PROJECT_ROOT / "results" / "benchmark_v2.json"
DEFAULT_COMPARISON_PATH = PROJECT_ROOT / "results" / "comparison_v1_v2.json"
DEFAULT_REGRESSION_OUTPUT_PATH = PROJECT_ROOT / "results" / "regression_v2.json"
DEFAULT_OUTPUT_PATH = DEFAULT_V1_OUTPUT_PATH
AGENT_VERSION = "v1"

CATEGORY_ORDER = (
    "simple_reachability",
    "path_analysis",
    "device_state",
    "acl_investigation",
    "root_cause",
    "multi_step_troubleshooting",
)
COMPONENT_NAMES = tuple(Evaluator.WEIGHTS)
FAILURE_TYPES = Evaluator.FAILURE_ORDER


def load_eval_cases(path: str | Path = DEFAULT_EVAL_CASES_PATH) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as cases_file:
        cases = json.load(cases_file)
    if not isinstance(cases, list):
        raise ValueError("Evaluation cases must be a JSON list.")
    return cases


def load_regression_cases(
    path: str | Path = DEFAULT_REGRESSION_CASES_PATH,
) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as cases_file:
        cases = json.load(cases_file)
    if not isinstance(cases, list):
        raise ValueError("Regression cases must be a JSON list.")
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every regression case must be a JSON object.")
        case_id = str(case.get("case_id", ""))
        if not case_id:
            raise ValueError("Every regression case must have a case_id.")
        if case_id in case_ids:
            raise ValueError(f"Duplicate regression case ID: {case_id}.")
        case_ids.add(case_id)
    return cases


def aggregate_benchmark(
    case_results: list[dict[str, Any]],
    *,
    agent_version: str,
    model: str,
) -> dict[str, Any]:
    """Aggregate already-evaluated cases without making model calls."""
    total_cases = len(case_results)
    successful_cases = sum(bool(case.get("success")) for case in case_results)
    failed_cases = total_cases - successful_cases

    component_scores = {
        component: _average(
            [
                _number(case.get("scores", {}).get(component))
                for case in case_results
            ]
        )
        for component in COMPONENT_NAMES
    }
    failure_counts = {
        failure: sum(
            failure in case.get("failures", []) for case in case_results
        )
        for failure in FAILURE_TYPES
    }

    category_metrics: dict[str, dict[str, Any]] = {}
    observed_categories = {
        str(case.get("category", "unknown")) for case in case_results
    }
    ordered_categories = list(CATEGORY_ORDER)
    ordered_categories.extend(
        sorted(observed_categories - set(CATEGORY_ORDER))
    )
    for category in ordered_categories:
        category_cases = [
            case for case in case_results if case.get("category") == category
        ]
        category_total = len(category_cases)
        category_successes = sum(
            bool(case.get("success")) for case in category_cases
        )
        category_metrics[category] = {
            "total_cases": category_total,
            "successful_cases": category_successes,
            "failed_cases": category_total - category_successes,
            "success_rate": _rounded(
                category_successes / category_total if category_total else 0.0
            ),
            "average_score": _average(
                [_number(case.get("overall_score")) for case in category_cases]
            ),
        }

    average_tokens = {
        token_type: _average(
            [
                _number((case.get("tokens") or {}).get(token_type))
                for case in case_results
            ]
        )
        for token_type in ("input", "output", "total")
    }
    provider_error_count = sum(_case_has_provider_error(case) for case in case_results)

    return {
        "agent_version": agent_version,
        "model": model,
        "total_cases": total_cases,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "successful_cases": successful_cases,
            "failed_cases": failed_cases,
            "task_success_rate": _rounded(
                successful_cases / total_cases if total_cases else 0.0
            ),
            "average_overall_score": _average(
                [_number(case.get("overall_score")) for case in case_results]
            ),
            "average_tool_calls": _average(
                [_number(case.get("tool_call_count")) for case in case_results]
            ),
            "average_latency_ms": _average(
                [_number(case.get("latency_ms")) for case in case_results]
            ),
            "average_tokens": average_tokens,
            "provider_error_count": provider_error_count,
        },
        "component_scores": component_scores,
        "failure_counts": failure_counts,
        "category_metrics": category_metrics,
        "cases": case_results,
    }


def run_benchmark(
    *,
    eval_cases_path: str | Path = DEFAULT_EVAL_CASES_PATH,
    output_path: str | Path | None = None,
    trajectory_dir: str | Path = DEFAULT_TRAJECTORY_DIR,
    agent_version: str = AGENT_VERSION,
) -> dict[str, Any]:
    cases = load_eval_cases(eval_cases_path)
    trajectory_store = TrajectoryStore(trajectory_dir)
    agent = NetworkAgent(
        trajectory_store=trajectory_store,
        agent_version=agent_version,
    )
    evaluator = Evaluator()
    case_results: list[dict[str, Any]] = []

    for index, eval_case in enumerate(cases, start=1):
        case_id = str(eval_case["id"])
        print(f"[{index}/{len(cases)}] {case_id}", flush=True)

        trajectory, runtime = _run_agent_with_bounded_retry(
            agent, eval_case["query"]
        )

        evaluation = evaluator.evaluate(eval_case, trajectory)
        case_results.append(
            {
                "case_id": case_id,
                "category": eval_case["category"],
                "query": eval_case["query"],
                "success": evaluation["success"],
                "overall_score": evaluation["overall_score"],
                "scores": evaluation["scores"],
                "failures": evaluation["failures"],
                "tools_called": evaluation["details"]["tools_called"],
                "tool_call_count": evaluation["tool_call_count"],
                "latency_ms": evaluation["latency_ms"],
                "tokens": evaluation["tokens"],
                "final_answer": trajectory.get("final_answer"),
                "trajectory_file_path": trajectory.get("trajectory_path"),
                "completed": trajectory.get("completed", False),
                "run_status": trajectory.get("status"),
                "runtime_error": trajectory.get("error"),
                **runtime,
            }
        )

    benchmark = aggregate_benchmark(
        case_results,
        agent_version=agent.agent_version,
        model=agent.model,
    )
    resolved_output_path = output_path or _default_output_path(agent.agent_version)
    save_benchmark(benchmark, resolved_output_path)
    return benchmark


def evaluate_regression_acceptance(
    regression_case: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Apply deterministic success, score, and required-tool acceptance rules."""
    acceptance = regression_case.get("acceptance") or {}
    must_succeed = bool(acceptance.get("must_succeed", True))
    minimum_score = _number(acceptance.get("minimum_score"))
    required_tools = [str(tool) for tool in acceptance.get("required_tools", [])]
    tools_called = [
        str(tool) for tool in evaluation.get("details", {}).get("tools_called", [])
    ]
    missing_required_tools = [
        tool for tool in required_tools if tool not in tools_called
    ]
    current_success = bool(evaluation.get("success"))
    score = _number(evaluation.get("overall_score"))

    acceptance_failures: list[str] = []
    if must_succeed and not current_success:
        acceptance_failures.append("current_evaluation_failed")
    if score < minimum_score:
        acceptance_failures.append("below_minimum_score")
    if missing_required_tools:
        acceptance_failures.append("missing_required_tools")

    return {
        "regression_pass": not acceptance_failures,
        "must_succeed": must_succeed,
        "minimum_score": minimum_score,
        "required_tools": required_tools,
        "missing_required_tools": missing_required_tools,
        "acceptance_failures": acceptance_failures,
    }


def aggregate_regression_results(
    case_results: list[dict[str, Any]],
    *,
    agent_version: str,
    model: str,
    known_unresolved: list[dict[str, Any]] | None = None,
    improved_not_fixed: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    passing = sum(bool(case.get("regression_pass")) for case in case_results)
    failing = len(case_results) - passing
    regressed = sum(
        not case.get("regression_pass", False)
        and bool(case.get("fixed_in_version"))
        for case in case_results
    )
    return {
        "agent_version": agent_version,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(case_results),
        "passing": passing,
        "failing": failing,
        "regressed": regressed,
        "provider_error_count": sum(
            _case_has_provider_error(case) for case in case_results
        ),
        "cases": case_results,
        "known_unresolved_or_regressed_cases": known_unresolved or [],
        "improved_but_not_fixed": improved_not_fixed or [],
    }


def run_regression_suite(
    *,
    agent_version: str,
    regression_cases_path: str | Path = DEFAULT_REGRESSION_CASES_PATH,
    eval_cases_path: str | Path = DEFAULT_EVAL_CASES_PATH,
    comparison_path: str | Path = DEFAULT_COMPARISON_PATH,
    output_path: str | Path = DEFAULT_REGRESSION_OUTPUT_PATH,
    trajectory_dir: str | Path = DEFAULT_TRAJECTORY_DIR,
) -> dict[str, Any]:
    regression_cases = load_regression_cases(regression_cases_path)
    eval_cases = {case["id"]: case for case in load_eval_cases(eval_cases_path)}
    trajectory_store = TrajectoryStore(trajectory_dir)
    agent = NetworkAgent(
        trajectory_store=trajectory_store,
        agent_version=agent_version,
    )
    evaluator = Evaluator()
    case_results: list[dict[str, Any]] = []

    for index, regression_case in enumerate(regression_cases, start=1):
        case_id = str(regression_case["case_id"])
        source_case_id = str(regression_case.get("source_eval_case_id", ""))
        if source_case_id not in eval_cases:
            raise ValueError(
                f"Regression case {case_id} references unknown eval case "
                f"{source_case_id}."
            )
        eval_case = eval_cases[source_case_id]
        print(f"[{index}/{len(regression_cases)}] {case_id}", flush=True)
        trajectory, runtime = _run_agent_with_bounded_retry(
            agent, eval_case["query"]
        )
        evaluation = evaluator.evaluate(eval_case, trajectory)
        acceptance = evaluate_regression_acceptance(regression_case, evaluation)
        case_results.append(
            {
                "case_id": case_id,
                "source_eval_case_id": source_case_id,
                "success": evaluation["success"],
                "score": evaluation["overall_score"],
                "regression_pass": acceptance["regression_pass"],
                "fixed_in_version": regression_case.get("fixed_in_version"),
                "minimum_score": acceptance["minimum_score"],
                "required_tools": acceptance["required_tools"],
                "missing_required_tools": acceptance["missing_required_tools"],
                "acceptance_failures": acceptance["acceptance_failures"],
                "tools_called": evaluation["details"]["tools_called"],
                "failures": evaluation["failures"],
                "original_v1_failure": regression_case["original_v1_failure"],
                "reason": regression_case["reason"],
                "trajectory_file_path": trajectory.get("trajectory_path"),
                "completed": trajectory.get("completed", False),
                "run_status": trajectory.get("status"),
                "runtime_error": trajectory.get("error"),
                **runtime,
            }
        )

    issue_sections = _load_regression_issue_sections(comparison_path)
    result = aggregate_regression_results(
        case_results,
        agent_version=agent.agent_version,
        model=agent.model,
        known_unresolved=issue_sections["known_unresolved"],
        improved_not_fixed=issue_sections["improved_not_fixed"],
    )
    save_benchmark(result, output_path)
    return result


def save_benchmark(benchmark: dict[str, Any], output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as benchmark_file:
        json.dump(benchmark, benchmark_file, indent=2, ensure_ascii=False)
        benchmark_file.write("\n")
    return str(path.resolve())


def print_terminal_summary(benchmark: dict[str, Any]) -> None:
    summary = benchmark["summary"]
    components = benchmark["component_scores"]

    print(f"\n{benchmark['agent_version'].upper()} Benchmark")
    print(f"\nCases: {benchmark['total_cases']}")
    print(f"Passed: {summary['successful_cases']}")
    print(f"Failed: {summary['failed_cases']}")
    print(f"Task success: {summary['task_success_rate']:.1%}")
    print(f"Average score: {summary['average_overall_score']:.3f}")

    print(f"\nTool selection: {components['tool_selection']:.1%}")
    print(f"Arguments: {components['arguments']:.1%}")
    print(f"Trajectory: {components['trajectory']:.1%}")
    print(f"Grounding: {components['grounding']:.1%}")
    print(f"Final answer: {components['final_answer']:.1%}")
    print(f"Efficiency: {components['efficiency']:.1%}")

    print(f"\nAvg tool calls: {summary['average_tool_calls']:.1f}")
    print(f"Avg latency: {summary['average_latency_ms']:.0f} ms")
    print(f"Avg tokens: {summary['average_tokens']['total']:.0f}")
    print(f"Provider errors: {summary.get('provider_error_count', 0)}")

    nonzero_failures = [
        (failure, count)
        for failure, count in benchmark["failure_counts"].items()
        if count
    ]
    nonzero_failures.sort(key=lambda item: (-item[1], item[0]))
    print("\nTop failures:")
    if nonzero_failures:
        for failure, count in nonzero_failures[:5]:
            print(f"- {failure}: {count}")
    else:
        print("- none")

    populated_categories = [
        (category, metrics)
        for category, metrics in benchmark["category_metrics"].items()
        if metrics["total_cases"]
    ]
    populated_categories.sort(
        key=lambda item: (item[1]["success_rate"], item[1]["average_score"], item[0])
    )
    print("\nWeakest categories:")
    for category, metrics in populated_categories[:3]:
        print(
            f"- {category}: {metrics['success_rate']:.1%} success, "
            f"{metrics['average_score']:.3f} average score"
        )

    failed_cases = [case for case in benchmark["cases"] if not case["success"]]
    print("\nFailed cases:")
    if not failed_cases:
        print("- none")
    for case in failed_cases:
        failures = ", ".join(case["failures"]) or "none"
        tools = ", ".join(case["tools_called"]) or "none"
        print(f"\n{case['case_id']}")
        print(f"Query: {case['query']}")
        print(f"Score: {case['overall_score']:.3f}")
        print(f"Failures: {failures}")
        print(f"Tools: {tools}")


def _is_transient_api_error(error: Any) -> bool:
    text = str(error or "").casefold()
    transient_markers = (
        "connection",
        "timeout",
        "timed out",
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "temporarily",
        "service unavailable",
    )
    return any(marker in text for marker in transient_markers)


def _run_agent_with_bounded_retry(
    agent: NetworkAgent,
    query: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory = agent.run(query)
    retry_count = 0
    provider_error = trajectory.get("status") == "api_error"
    if provider_error and _is_transient_api_error(trajectory.get("error")):
        retry_count = 1
        print("  transient API error; retrying once", flush=True)
        trajectory = agent.run(query)

    final_status = trajectory.get("status")
    provider_error = provider_error or final_status == "api_error"
    provider_error_status = None
    if provider_error:
        provider_error_status = (
            "recovered_after_retry" if final_status != "api_error" else "api_error"
        )
    return trajectory, {
        "retry_count": retry_count,
        "provider_error": provider_error,
        "error_type": "api_error" if provider_error else None,
        "error_status": provider_error_status,
    }


def _load_regression_issue_sections(
    comparison_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    with Path(comparison_path).open(encoding="utf-8") as comparison_file:
        comparison = json.load(comparison_file)
    cases = comparison.get("cases", [])
    known_unresolved = [
        _regression_issue_record(case)
        for case in cases
        if case.get("classification") == "regressed"
    ]
    improved_not_fixed = [
        _regression_issue_record(case)
        for case in cases
        if case.get("classification") == "improved"
        and not case.get("v2_success", False)
    ]
    return {
        "known_unresolved": known_unresolved,
        "improved_not_fixed": improved_not_fixed,
    }


def _regression_issue_record(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "classification": case.get("classification"),
        "v1_success": bool(case.get("v1_success")),
        "v1_score": _number(case.get("v1_score")),
        "v1_failures": list(case.get("v1_failures", [])),
        "v2_success": bool(case.get("v2_success")),
        "v2_score": _number(case.get("v2_score")),
        "v2_failures": list(case.get("v2_failures", [])),
    }


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _case_has_provider_error(case: dict[str, Any]) -> bool:
    """Read explicit V2 metadata or infer it from legacy V1 records."""
    if "provider_error" in case:
        return bool(case["provider_error"])
    return case.get("run_status") == "api_error" or _number(
        case.get("retry_count")
    ) > 0


def _default_output_path(agent_version: str) -> Path:
    if agent_version == "v1":
        return DEFAULT_V1_OUTPUT_PATH
    if agent_version == "v2":
        return DEFAULT_V2_OUTPUT_PATH
    return PROJECT_ROOT / "results" / f"benchmark_{agent_version}.json"


def _average(values: list[float]) -> float:
    return _rounded(sum(values) / len(values) if values else 0.0)


def _rounded(value: float) -> float:
    return round(value, 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a NetEval benchmark.")
    parser.add_argument(
        "--agent-version",
        choices=("v1", "v2"),
        default=AGENT_VERSION,
        help="Agent configuration to benchmark (default: v1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Benchmark output path; defaults to results/benchmark_<version>.json.",
    )
    parser.add_argument(
        "--compare-v1",
        action="store_true",
        help="After a V2 run, compare it with the preserved V1 benchmark.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=DEFAULT_COMPARISON_PATH,
        help="Output path for --compare-v1 results.",
    )
    parser.add_argument(
        "--regression",
        action="store_true",
        help="Run the fixed-case regression suite instead of the full benchmark.",
    )
    args = parser.parse_args(argv)
    output_path = args.output or (
        DEFAULT_REGRESSION_OUTPUT_PATH
        if args.regression
        else _default_output_path(args.agent_version)
    )

    if args.regression:
        if args.compare_v1:
            print(
                "Regression error: --compare-v1 cannot be used with --regression.",
                file=sys.stderr,
            )
            return 1
        try:
            regression = run_regression_suite(
                agent_version=args.agent_version,
                output_path=output_path,
            )
        except ConfigurationError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Regression error: {exc}", file=sys.stderr)
            return 1

        from src.report import print_regression_report

        print_regression_report(regression)
        print(f"\nSaved: {output_path.resolve()}")
        return 0 if regression["failing"] == 0 else 1

    try:
        benchmark = run_benchmark(
            agent_version=args.agent_version,
            output_path=output_path,
        )
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 1

    print_terminal_summary(benchmark)
    print(f"\nSaved: {output_path.resolve()}")

    if args.compare_v1:
        if args.agent_version != "v2":
            print("Comparison error: --compare-v1 requires --agent-version v2.", file=sys.stderr)
            return 1
        from src.report import create_comparison, print_comparison_report

        try:
            comparison = create_comparison(
                DEFAULT_V1_OUTPUT_PATH,
                output_path,
                args.comparison_output,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Comparison error: {exc}", file=sys.stderr)
            return 1
        print_comparison_report(comparison)
        print(f"\nSaved: {args.comparison_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json

from src.runner import (
    DEFAULT_COMPARISON_PATH,
    DEFAULT_V1_OUTPUT_PATH,
    aggregate_regression_results,
    evaluate_regression_acceptance,
    load_regression_cases,
    save_benchmark,
)


def regression_case(
    *, minimum_score: float = 0.8, required_tools: list[str] | None = None
) -> dict:
    return {
        "case_id": "path_03",
        "source_eval_case_id": "path_03",
        "fixed_in_version": "v2",
        "acceptance": {
            "must_succeed": True,
            "minimum_score": minimum_score,
            "required_tools": required_tools or ["trace_path"],
        },
    }


def evaluation(*, success: bool, score: float, tools: list[str]) -> dict:
    return {
        "success": success,
        "overall_score": score,
        "details": {"tools_called": tools},
    }


def result_case(*, regression_pass: bool) -> dict:
    return {
        "case_id": "path_03",
        "source_eval_case_id": "path_03",
        "success": regression_pass,
        "score": 1.0 if regression_pass else 0.5,
        "regression_pass": regression_pass,
        "fixed_in_version": "v2",
        "minimum_score": 0.8,
        "required_tools": ["trace_path"],
        "missing_required_tools": [] if regression_pass else ["trace_path"],
        "acceptance_failures": [] if regression_pass else ["missing_required_tools"],
        "tools_called": ["trace_path"] if regression_pass else [],
        "failures": [],
        "original_v1_failure": {
            "success": False,
            "score": 0.2213,
            "failures": ["wrong_tool"],
            "tools_called": ["get_device_state"],
        },
        "reason": "Original issue.",
        "trajectory_file_path": "trajectory.json",
        "provider_error": False,
    }


def test_regression_cases_load_correctly() -> None:
    cases = load_regression_cases()

    assert len(cases) == 6
    assert {case["case_id"] for case in cases} == {
        "path_01",
        "path_03",
        "state_02",
        "diag_04",
        "multi_02",
        "multi_03",
    }
    assert "diag_03" not in {case["case_id"] for case in cases}
    assert "multi_01" not in {case["case_id"] for case in cases}


def test_original_v1_metadata_is_preserved() -> None:
    regressions = load_regression_cases()
    with DEFAULT_V1_OUTPUT_PATH.open(encoding="utf-8") as benchmark_file:
        v1_cases = {
            case["case_id"]: case for case in json.load(benchmark_file)["cases"]
        }

    for regression in regressions:
        original = regression["original_v1_failure"]
        benchmark_case = v1_cases[regression["source_eval_case_id"]]
        assert original["success"] == benchmark_case["success"]
        assert original["score"] == benchmark_case["overall_score"]
        assert original["failures"] == benchmark_case["failures"]
        assert original["tools_called"] == benchmark_case["tools_called"]


def test_successful_high_score_run_passes_regression() -> None:
    acceptance = evaluate_regression_acceptance(
        regression_case(),
        evaluation(success=True, score=0.95, tools=["trace_path"]),
    )

    assert acceptance["regression_pass"] is True
    assert acceptance["acceptance_failures"] == []


def test_failed_current_run_is_marked_as_regressed() -> None:
    acceptance = evaluate_regression_acceptance(
        regression_case(),
        evaluation(success=False, score=0.95, tools=["trace_path"]),
    )
    aggregate = aggregate_regression_results(
        [result_case(regression_pass=acceptance["regression_pass"])],
        agent_version="v2",
        model="test-model",
    )

    assert acceptance["regression_pass"] is False
    assert "current_evaluation_failed" in acceptance["acceptance_failures"]
    assert aggregate["regressed"] == 1


def test_score_below_threshold_fails_regression() -> None:
    acceptance = evaluate_regression_acceptance(
        regression_case(minimum_score=0.9),
        evaluation(success=True, score=0.89, tools=["trace_path"]),
    )

    assert acceptance["regression_pass"] is False
    assert acceptance["acceptance_failures"] == ["below_minimum_score"]


def test_required_tool_acceptance_is_enforced() -> None:
    acceptance = evaluate_regression_acceptance(
        regression_case(required_tools=["trace_path"]),
        evaluation(success=True, score=1.0, tools=["check_reachability"]),
    )

    assert acceptance["regression_pass"] is False
    assert acceptance["missing_required_tools"] == ["trace_path"]
    assert "missing_required_tools" in acceptance["acceptance_failures"]


def test_unresolved_cases_are_not_counted_as_fixed_regressions() -> None:
    unresolved = [{"case_id": "diag_03"}]
    improved_not_fixed = [{"case_id": "multi_01"}]
    aggregate = aggregate_regression_results(
        [result_case(regression_pass=True)],
        agent_version="v2",
        model="test-model",
        known_unresolved=unresolved,
        improved_not_fixed=improved_not_fixed,
    )

    assert aggregate["total_cases"] == 1
    assert aggregate["passing"] == 1
    assert aggregate["known_unresolved_or_regressed_cases"] == unresolved
    assert aggregate["improved_but_not_fixed"] == improved_not_fixed


def test_comparison_tracks_expected_open_issues() -> None:
    comparison = json.loads(DEFAULT_COMPARISON_PATH.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in comparison["cases"]}

    assert cases["diag_03"]["classification"] == "regressed"
    assert cases["multi_01"]["classification"] == "improved"
    assert cases["multi_01"]["v2_success"] is False


def test_regression_result_schema_writes_valid_json(tmp_path) -> None:
    result = aggregate_regression_results(
        [result_case(regression_pass=True)],
        agent_version="v2",
        model="test-model",
        known_unresolved=[{"case_id": "diag_03"}],
        improved_not_fixed=[{"case_id": "multi_01"}],
    )
    output_path = tmp_path / "regression_v2.json"

    saved_path = save_benchmark(result, output_path)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_path == str(output_path.resolve())
    assert set(
        (
            "agent_version",
            "total_cases",
            "passing",
            "failing",
            "regressed",
            "cases",
            "known_unresolved_or_regressed_cases",
            "improved_but_not_fixed",
        )
    ).issubset(loaded)
    assert set(
        (
            "case_id",
            "success",
            "score",
            "regression_pass",
            "required_tools",
            "tools_called",
            "failures",
            "original_v1_failure",
            "trajectory_file_path",
        )
    ).issubset(loaded["cases"][0])

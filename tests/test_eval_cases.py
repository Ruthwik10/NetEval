import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
EVAL_CASES_PATH = PROJECT_ROOT / "data" / "eval_cases.json"
NETWORK_PATH = PROJECT_ROOT / "data" / "network.json"

VALID_CATEGORIES = {
    "simple_reachability",
    "path_analysis",
    "device_state",
    "acl_investigation",
    "root_cause",
    "multi_step_troubleshooting",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_exactly_18_eval_cases_exist() -> None:
    assert len(load_json(EVAL_CASES_PATH)) == 18


def test_eval_case_ids_are_unique() -> None:
    cases = load_json(EVAL_CASES_PATH)
    ids = [case["id"] for case in cases]

    assert len(ids) == len(set(ids))


def test_all_categories_are_valid() -> None:
    cases = load_json(EVAL_CASES_PATH)

    assert {case["category"] for case in cases} <= VALID_CATEGORIES


def test_all_referenced_devices_exist() -> None:
    cases = load_json(EVAL_CASES_PATH)
    network = load_json(NETWORK_PATH)
    device_ids = {device["id"] for device in network["devices"]}

    for case in cases:
        referenced_devices = set(case["referenced_devices"])
        for field in ("source", "destination", "device"):
            if field in case["required_arguments"]:
                referenced_devices.add(case["required_arguments"][field])
        if case["expected_blocker"] is not None:
            referenced_devices.add(case["expected_blocker"])

        assert referenced_devices <= device_ids, case["id"]


def test_expected_rule_ids_exist() -> None:
    cases = load_json(EVAL_CASES_PATH)
    network = load_json(NETWORK_PATH)
    acl_ids = {rule["id"] for rule in network["acl_rules"]}

    for case in cases:
        if case["expected_rule_id"] is not None:
            assert case["expected_rule_id"] in acl_ids, case["id"]


def test_required_ports_are_integers() -> None:
    cases = load_json(EVAL_CASES_PATH)

    for case in cases:
        port = case["required_arguments"].get("port")
        if port is not None:
            assert isinstance(port, int) and not isinstance(port, bool), case["id"]


def test_at_least_six_cases_require_multiple_tool_calls() -> None:
    cases = load_json(EVAL_CASES_PATH)
    multi_step_cases = [case for case in cases if case["min_tool_calls"] >= 2]

    assert len(multi_step_cases) >= 6
    assert all(len(case["expected_tools"]) >= 2 for case in multi_step_cases)


def test_every_case_has_a_non_empty_query() -> None:
    cases = load_json(EVAL_CASES_PATH)

    assert all(case["query"].strip() for case in cases)


def test_every_case_has_an_expected_outcome() -> None:
    cases = load_json(EVAL_CASES_PATH)

    assert all(case["expected_outcome"] for case in cases)

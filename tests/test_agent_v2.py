import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.agent import (
    TOOL_REGISTRY,
    V1_TOOL_DEFINITIONS,
    V2_TOOL_DEFINITIONS,
    NetworkAgent,
)
from src.evaluator import Evaluator
from src.prompts import V1_SYSTEM_PROMPT, V2_SYSTEM_PROMPT
from src.trajectory_store import TrajectoryStore
from tests.test_agent import make_client, make_response, make_tool_call


def descriptions(tool_definitions: list[dict]) -> dict[str, str]:
    return {
        tool["function"]["name"]: tool["function"]["description"]
        for tool in tool_definitions
    }


def test_v1_prompt_still_exists() -> None:
    assert V1_SYSTEM_PROMPT == """You are a network troubleshooting assistant.
Use the available tools to investigate the user's question.
Base your answer on tool results.
Do not invent network facts.

Network summary:
- The network contains two application stacks: orders and billing.
- Public traffic enters through edge-fw.
- Sensitive systems are protected by internal-fw.
- Retrieve detailed network facts through the available tools.
"""


def test_v2_prompt_exists() -> None:
    assert V2_SYSTEM_PROMPT.strip()


def test_v2_states_path_does_not_prove_reachability() -> None:
    prompt = V2_SYSTEM_PROMPT.casefold()

    assert "a physical path existing does not prove that traffic is allowed" in prompt
    assert "topology path evidence establishes route/hops only" in prompt
    assert "do not claim traffic is allowed or reachable" in prompt


def test_v2_contains_stopping_criteria() -> None:
    prompt = V2_SYSTEM_PROMPT.casefold()

    assert "stopping criteria" in prompt
    assert "stop calling tools when the question is answered" in prompt


def test_v2_distinguishes_downstream_endpoint_from_policy() -> None:
    prompt = V2_SYSTEM_PROMPT.casefold()

    assert "billing-api -> billing-db" in prompt
    assert "destination state and policy" in prompt
    assert "never pass a rule id" in prompt


def test_v2_tool_descriptions_are_more_precise_than_v1() -> None:
    v1 = descriptions(V1_TOOL_DEFINITIONS)
    v2 = descriptions(V2_TOOL_DEFINITIONS)

    assert v1.keys() == v2.keys()
    assert all(v2[name] != v1[name] for name in v1)
    assert "does not prove reachability or policy permission" in v2[
        "trace_path"
    ].casefold()
    assert "do not use repeatedly" in v2["get_device_state"].casefold()


def test_v2_pure_path_question_uses_trace_path_only() -> None:
    client, create = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "trace_path",
                    '{"source":"admin-net","destination":"orders-db"}',
                )
            ]
        ),
        make_response(content="admin-net -> internal-fw -> orders-db"),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("How does traffic from the admin network get to the orders database?")

    assert [step["tool"] for step in result["steps"]] == ["trace_path"]
    assert "reachable" not in result["final_answer"].casefold()
    assert "allowed" not in result["final_answer"].casefold()
    system_prompt = create.call_args_list[0].kwargs["messages"][0]["content"]
    assert "report only the returned hops" in system_prompt


def test_v2_preserves_exact_orders_api_entity() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "check_reachability",
                    '{"source":"internet","destination":"orders-api","port":443}',
                )
            ]
        ),
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-2",
                    "inspect_acl",
                    (
                        '{"device":"edge-fw","source":"internet",'
                        '"destination":"orders-api","port":443}'
                    ),
                )
            ]
        ),
        make_response(content="edge-fw rule acl-02 explicitly denies the request."),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("Why is internet blocked from reaching the orders API on port 443?")

    assert result["steps"][0]["arguments"] == {
        "source": "internet",
        "destination": "orders-api",
        "port": 443,
    }
    assert result["steps"][1]["arguments"]["destination"] == "orders-api"


def test_v2_compound_route_and_policy_collects_both_evidence_types() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "trace_path",
                    '{"source":"admin-net","destination":"orders-db"}',
                )
            ]
        ),
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-2",
                    "check_reachability",
                    '{"source":"admin-net","destination":"orders-db","port":5432}',
                )
            ]
        ),
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-3",
                    "inspect_acl",
                    (
                        '{"device":"internal-fw","source":"admin-net",'
                        '"destination":"orders-db","port":5432}'
                    ),
                )
            ]
        ),
        make_response(content="The route crosses internal-fw, where acl-17 denies it."),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("Show the route and explain the policy failure.")

    assert [step["tool"] for step in result["steps"]] == [
        "trace_path",
        "check_reachability",
        "inspect_acl",
    ]


def test_v2_root_cause_does_not_stop_after_device_state() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1", "get_device_state", '{"device":"billing-api"}'
                )
            ]
        ),
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-2",
                    "check_reachability",
                    '{"source":"internet","destination":"billing-api","port":8443}',
                )
            ]
        ),
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-3",
                    "find_blocking_hop",
                    '{"source":"internet","destination":"billing-api","port":8443}',
                )
            ]
        ),
        make_response(content="edge-fw applies default deny."),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("Why can't internet reach billing-api on port 8443?")

    assert [step["tool"] for step in result["steps"]] == [
        "get_device_state",
        "check_reachability",
        "find_blocking_hop",
    ]
    assert result["final_answer"] == "edge-fw applies default deny."


def test_v2_device_state_question_remains_single_tool() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1", "get_device_state", '{"device":"billing-db"}'
                )
            ]
        ),
        make_response(content="billing-db is down."),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("Is billing-db up?")

    assert [step["tool"] for step in result["steps"]] == ["get_device_state"]


def test_v2_diagnostic_destination_down_verifies_destination_state() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "check_reachability",
                    '{"source":"billing-api","destination":"billing-db","port":5432}',
                )
            ]
        ),
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-2", "get_device_state", '{"device":"billing-db"}'
                )
            ]
        ),
        make_response(content="billing-db is down, so the endpoint caused the failure."),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("Why can't billing-api reach billing-db on port 5432?")

    assert [step["tool"] for step in result["steps"]] == [
        "check_reachability",
        "get_device_state",
    ]
    assert result["steps"][1]["arguments"] == {"device": "billing-db"}


def test_v2_endpoint_vs_policy_question_verifies_diagnosed_endpoint() -> None:
    client, create = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "check_reachability",
                    '{"source":"billing-api","destination":"billing-db","port":5432}',
                )
            ]
        ),
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-2", "get_device_state", '{"device":"billing-db"}'
                )
            ]
        ),
        make_response(content="The billing-db endpoint is down; policy is not the cause."),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("Determine whether the billing failure is caused by its endpoint or network policy.")

    assert [step["tool"] for step in result["steps"]] == [
        "check_reachability",
        "get_device_state",
    ]
    system_prompt = create.call_args_list[0].kwargs["messages"][0]["content"]
    assert "treat the returned reason as a diagnostic signal" in system_prompt.casefold()
    assert "do not inspect acls" in system_prompt.casefold()


def test_v2_simple_reachability_does_not_add_device_state() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "check_reachability",
                    '{"source":"billing-api","destination":"billing-db","port":5432}',
                )
            ]
        ),
        make_response(content="No. billing-db is down."),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("Can billing-api reach billing-db on port 5432?")

    assert [step["tool"] for step in result["steps"]] == ["check_reachability"]


def test_v2_direct_acl_question_remains_single_tool() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "inspect_acl",
                    (
                        '{"device":"internal-fw","source":"admin-net",'
                        '"destination":"orders-db","port":5432}'
                    ),
                )
            ]
        ),
        make_response(content="acl-17 denies the traffic."),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("Which ACL affects admin-net traffic to orders-db on port 5432?")

    assert [step["tool"] for step in result["steps"]] == ["inspect_acl"]


def test_v2_path_and_reachability_answer_stays_within_tool_evidence() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "trace_path",
                    '{"source":"internet","destination":"public-web"}',
                )
            ]
        ),
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-2",
                    "check_reachability",
                    '{"source":"internet","destination":"public-web","port":443}',
                )
            ]
        ),
        make_response(
            content=(
                "Path: internet -> edge-fw -> public-web. "
                "check_reachability returned reachable=true for TCP port 443."
            )
        ),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("Verify the path and whether the connection is reachable on port 443.")
    eval_cases_path = Path(__file__).resolve().parents[1] / "data" / "eval_cases.json"
    eval_cases = json.loads(eval_cases_path.read_text(encoding="utf-8"))
    eval_case = next(case for case in eval_cases if case["id"] == "multi_03")
    evaluation = Evaluator().evaluate(eval_case, result)

    assert [step["tool"] for step in result["steps"]] == [
        "trace_path",
        "check_reachability",
    ]
    answer = result["final_answer"].casefold()
    assert all(
        unsupported not in answer
        for unsupported in ("allowed", "permitted", "acl", "firewall", "health", "cause")
    )
    assert evaluation["success"] is True
    assert evaluation["failures"] == []


def test_v2_evidence_gate_completes_compound_path_and_reachability(
    tmp_path,
) -> None:
    client, create = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "trace_path",
                    '{"source":"internet","destination":"public-web"}',
                )
            ]
        ),
        make_response(content="The path is internet -> edge-fw -> public-web."),
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-2",
                    "check_reachability",
                    '{"source":"internet","destination":"public-web","port":443}',
                )
            ]
        ),
        make_response(content="The path is recorded and reachable=true on TCP 443."),
    )

    result = NetworkAgent(
        client=client,
        model="test-model",
        agent_version="v2",
        trajectory_store=TrajectoryStore(tmp_path),
    ).run("Verify the path and whether HTTPS is reachable.")

    assert [step["tool"] for step in result["steps"]] == [
        "trace_path",
        "check_reachability",
    ]
    assert result["evidence_requirements"] == ["path", "reachability"]
    assert result["evidence_satisfied"] == ["path", "reachability"]
    assert result["evidence_gate_triggers"] == 1
    reminder_messages = create.call_args_list[2].kwargs["messages"]
    assert any(
        message["role"] == "system"
        and "missing evidence: reachability" in message["content"].casefold()
        for message in reminder_messages
    )
    saved = json.loads(Path(result["trajectory_path"]).read_text(encoding="utf-8"))
    assert saved["evidence_requirements"] == ["path", "reachability"]
    assert saved["evidence_satisfied"] == ["path", "reachability"]
    assert saved["evidence_gate_triggers"] == 1


def test_v2_evidence_gate_accepts_complete_simple_path() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "trace_path",
                    '{"source":"admin-net","destination":"orders-db"}',
                )
            ]
        ),
        make_response(content="admin-net -> internal-fw -> orders-db"),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("What path does admin-net take to orders-db?")

    assert result["tool_call_count"] == 1
    assert result["evidence_requirements"] == ["path"]
    assert result["evidence_satisfied"] == ["path"]
    assert result["evidence_gate_triggers"] == 0


def test_v2_evidence_gate_accepts_complete_device_state() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1", "get_device_state", '{"device":"billing-db"}'
                )
            ]
        ),
        make_response(content="billing-db is down."),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("Is billing-db up?")

    assert result["tool_call_count"] == 1
    assert result["evidence_requirements"] == ["device_state"]
    assert result["evidence_satisfied"] == ["device_state"]
    assert result["evidence_gate_triggers"] == 0


def test_v2_evidence_gate_requires_state_after_destination_down() -> None:
    client, create = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "check_reachability",
                    '{"source":"billing-api","destination":"billing-db","port":5432}',
                )
            ]
        ),
        make_response(content="The destination is down."),
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-2", "get_device_state", '{"device":"billing-db"}'
                )
            ]
        ),
        make_response(content="billing-db is down, confirming endpoint failure."),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("Why can't billing-api reach billing-db on port 5432?")

    assert [step["tool"] for step in result["steps"]] == [
        "check_reachability",
        "get_device_state",
    ]
    assert result["evidence_requirements"] == [
        "reachability",
        "device_state",
        "root_cause",
    ]
    assert result["evidence_satisfied"] == [
        "reachability",
        "device_state",
        "root_cause",
    ]
    assert result["evidence_gate_triggers"] == 1
    assert "missing evidence: device_state" in str(
        create.call_args_list[2].kwargs["messages"]
    ).casefold()


def test_v2_evidence_gate_requires_acl_after_explicit_deny() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "check_reachability",
                    '{"source":"admin-net","destination":"orders-db","port":5432}',
                )
            ]
        ),
        make_response(content="The request is explicitly denied."),
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-2",
                    "inspect_acl",
                    (
                        '{"device":"internal-fw","source":"admin-net",'
                        '"destination":"orders-db","port":5432}'
                    ),
                )
            ]
        ),
        make_response(content="acl-17 on internal-fw explicitly denies the request."),
    )

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v2"
    ).run("Why can't admin-net reach orders-db on port 5432?")

    assert [step["tool"] for step in result["steps"]] == [
        "check_reachability",
        "inspect_acl",
    ]
    assert "policy" in result["evidence_requirements"]
    assert "policy" in result["evidence_satisfied"]
    assert result["evidence_gate_triggers"] == 1


def test_v1_does_not_apply_v2_evidence_gate() -> None:
    client, create = make_client(make_response(content="Answer without tools."))

    result = NetworkAgent(
        client=client, model="test-model", agent_version="v1"
    ).run("Verify the path and whether HTTPS is reachable.")

    assert result["completed"] is True
    assert result["tool_call_count"] == 0
    assert create.call_count == 1
    assert "evidence_requirements" not in result
    assert "evidence_satisfied" not in result
    assert "evidence_gate_triggers" not in result


def test_v2_duplicate_tool_call_is_not_executed_twice() -> None:
    repeated_arguments = '{"source":"admin-net","destination":"orders-db"}'
    client, _ = make_client(
        make_response(
            tool_calls=[make_tool_call("call-1", "trace_path", repeated_arguments)]
        ),
        make_response(
            tool_calls=[make_tool_call("call-2", "trace_path", repeated_arguments)]
        ),
        make_response(content="The existing path evidence answers the question."),
    )
    tool = Mock(return_value={"path_found": True})

    with patch.dict(TOOL_REGISTRY, {"trace_path": tool}):
        result = NetworkAgent(
            client=client,
            model="test-model",
            agent_version="v2",
        ).run("Trace the path")

    assert tool.call_count == 1
    assert result["steps"][1]["result"]["error"] == "duplicate_tool_call"


def test_v2_allows_different_tool_calls() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "trace_path",
                    '{"source":"admin-net","destination":"orders-db"}',
                ),
                make_tool_call(
                    "call-2",
                    "trace_path",
                    '{"source":"internet","destination":"public-web"}',
                ),
            ]
        ),
        make_response(content="Both paths were checked."),
    )
    tool = Mock(return_value={"path_found": True})

    with patch.dict(TOOL_REGISTRY, {"trace_path": tool}):
        result = NetworkAgent(
            client=client,
            model="test-model",
            agent_version="v2",
        ).run("Compare paths")

    assert tool.call_count == 2
    assert all("error" not in step["result"] for step in result["steps"])


def test_v1_duplicate_calls_still_execute_normally() -> None:
    repeated_arguments = '{"source":"admin-net","destination":"orders-db"}'
    client, _ = make_client(
        make_response(
            tool_calls=[make_tool_call("call-1", "trace_path", repeated_arguments)]
        ),
        make_response(
            tool_calls=[make_tool_call("call-2", "trace_path", repeated_arguments)]
        ),
        make_response(content="The path was checked twice."),
    )
    tool = Mock(return_value={"path_found": True})

    with patch.dict(TOOL_REGISTRY, {"trace_path": tool}):
        result = NetworkAgent(
            client=client,
            model="test-model",
            agent_version="v1",
        ).run("Trace the path twice")

    assert tool.call_count == 2
    assert all("error" not in step["result"] for step in result["steps"])

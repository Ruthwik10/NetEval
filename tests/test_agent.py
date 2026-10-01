import json
from types import SimpleNamespace
from unittest.mock import Mock

from src.agent import NetworkAgent


def make_tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_response(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    usage = SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=usage,
    )


def make_client(*responses: SimpleNamespace) -> tuple[SimpleNamespace, Mock]:
    create = Mock(side_effect=responses)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, create


def test_trace_path_executes_and_result_is_returned_to_model() -> None:
    first = make_response(
        tool_calls=[
            make_tool_call(
                "call-1",
                "trace_path",
                '{"source":"admin-net","destination":"orders-db"}',
            )
        ],
        input_tokens=10,
        output_tokens=4,
    )
    second = make_response(
        content="The path crosses internal-fw.",
        input_tokens=20,
        output_tokens=6,
    )
    client, create = make_client(first, second)

    result = NetworkAgent(client=client, model="test-model").run("Trace the path")

    assert result["completed"] is True
    assert result["final_answer"] == "The path crosses internal-fw."
    assert result["steps"][0]["tool"] == "trace_path"
    assert result["steps"][0]["result"]["path"] == [
        "admin-net",
        "internal-fw",
        "orders-db",
    ]
    second_messages = create.call_args_list[1].kwargs["messages"]
    tool_message = next(message for message in second_messages if message["role"] == "tool")
    assert json.loads(tool_message["content"])["path_found"] is True
    assert result["input_tokens"] == 30
    assert result["output_tokens"] == 10
    assert result["total_tokens"] == 40


def test_multiple_sequential_tool_rounds_work() -> None:
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

    result = NetworkAgent(client=client, model="test-model").run("Diagnose")

    assert result["completed"] is True
    assert [step["tool"] for step in result["steps"]] == [
        "trace_path",
        "inspect_acl",
    ]
    assert result["steps"][1]["result"]["matched_rules"][0]["id"] == "acl-17"


def test_multiple_tool_calls_in_one_response_are_supported() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[
                make_tool_call(
                    "call-1",
                    "get_device_state",
                    '{"device":"orders-db"}',
                ),
                make_tool_call(
                    "call-2",
                    "get_device_state",
                    '{"device":"billing-db"}',
                ),
            ]
        ),
        make_response(content="The billing database is down."),
    )

    result = NetworkAgent(client=client, model="test-model").run("Compare databases")

    assert result["completed"] is True
    assert result["tool_call_count"] == 2
    assert [step["result"]["status"] for step in result["steps"]] == [
        "up",
        "down",
    ]


def test_unknown_tool_is_returned_as_safe_error() -> None:
    client, create = make_client(
        make_response(
            tool_calls=[make_tool_call("call-1", "delete_network", "{}")]
        ),
        make_response(content="That tool is unavailable."),
    )

    result = NetworkAgent(client=client, model="test-model").run("Use a bad tool")

    assert result["completed"] is True
    assert result["steps"][0]["result"]["error"] == "unknown_tool"
    tool_messages = [
        message
        for message in create.call_args_list[1].kwargs["messages"]
        if message["role"] == "tool"
    ]
    assert json.loads(tool_messages[0]["content"])["error"] == "unknown_tool"


def test_malformed_tool_arguments_are_returned_as_safe_error() -> None:
    client, _ = make_client(
        make_response(
            tool_calls=[make_tool_call("call-1", "trace_path", "not-json")]
        ),
        make_response(content="The arguments were invalid."),
    )

    result = NetworkAgent(client=client, model="test-model").run("Malformed call")

    assert result["completed"] is True
    assert result["steps"][0]["result"]["error"] == "malformed_tool_arguments"


def test_max_tool_round_limit_returns_controlled_status() -> None:
    repeated_call = make_response(
        tool_calls=[
            make_tool_call(
                "call",
                "trace_path",
                '{"source":"admin-net","destination":"orders-db"}',
            )
        ]
    )
    client, create = make_client(repeated_call, repeated_call, repeated_call)

    result = NetworkAgent(
        client=client,
        model="test-model",
        max_tool_rounds=2,
    ).run("Keep calling tools")

    assert result["completed"] is False
    assert result["status"] == "max_tool_rounds_exceeded"
    assert result["tool_call_count"] == 2
    assert create.call_count == 3


def test_api_error_returns_controlled_status() -> None:
    create = Mock(side_effect=RuntimeError("service unavailable"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = NetworkAgent(client=client, model="test-model").run("Diagnose")

    assert result["completed"] is False
    assert result["status"] == "api_error"
    assert result["error"] == "NVIDIA API error: service unavailable"

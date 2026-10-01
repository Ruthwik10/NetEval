import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.agent import NetworkAgent
from src.trajectory_store import TrajectoryStore


def sample_trajectory() -> dict:
    return {
        "run_id": "8fcceea9-7f5b-47ac-9340-dff10560243c",
        "timestamp": "2026-09-30T12:34:56.123456+00:00",
        "query": "Why is traffic blocked?",
        "agent_version": "v1",
        "model": "test-model",
        "steps": [
            {
                "step_index": 1,
                "tool": "trace_path",
                "arguments": {"source": "admin-net", "destination": "orders-db"},
                "result": {"path_found": True},
            },
            {
                "step_index": 2,
                "tool": "inspect_acl",
                "arguments": {"device": "internal-fw"},
                "result": {"matched_rules": [{"id": "acl-17"}]},
            },
        ],
        "final_answer": "acl-17 blocks the traffic.",
        "tool_call_count": 2,
        "completed": True,
        "error": None,
        "latency_ms": 12.5,
        "tokens": {"input": 100, "output": 25, "total": 125},
    }


def load_saved(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_trajectory_directory_is_created_automatically(tmp_path: Path) -> None:
    output_dir = tmp_path / "missing" / "trajectories"

    TrajectoryStore(output_dir).save(sample_trajectory())

    assert output_dir.is_dir()


def test_successful_trajectory_is_valid_json(tmp_path: Path) -> None:
    path = TrajectoryStore(tmp_path).save(sample_trajectory())

    saved = load_saved(path)
    assert saved["completed"] is True
    assert saved["final_answer"] == "acl-17 blocks the traffic."


def test_tool_step_order_is_preserved(tmp_path: Path) -> None:
    path = TrajectoryStore(tmp_path).save(sample_trajectory())

    saved = load_saved(path)
    assert [step["tool"] for step in saved["steps"]] == [
        "trace_path",
        "inspect_acl",
    ]
    assert [step["step_index"] for step in saved["steps"]] == [1, 2]


def test_token_counts_are_persisted(tmp_path: Path) -> None:
    path = TrajectoryStore(tmp_path).save(sample_trajectory())

    assert load_saved(path)["tokens"] == {
        "input": 100,
        "output": 25,
        "total": 125,
    }


def test_agent_version_is_persisted(tmp_path: Path) -> None:
    path = TrajectoryStore(tmp_path).save(sample_trajectory())

    assert load_saved(path)["agent_version"] == "v1"


def test_failed_agent_run_is_persisted(tmp_path: Path) -> None:
    create = Mock(side_effect=RuntimeError("service unavailable"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    store = TrajectoryStore(tmp_path)

    result = NetworkAgent(
        client=client,
        model="test-model",
        trajectory_store=store,
    ).run("Diagnose the outage")

    saved = load_saved(result["trajectory_path"])
    assert saved["completed"] is False
    assert saved["error"] == {
        "type": "api_error",
        "message": "NVIDIA API error: service unavailable",
    }


def test_api_keys_and_authorization_values_are_redacted(
    tmp_path: Path, monkeypatch
) -> None:
    secret = "nvapi-test-secret-value"
    monkeypatch.setenv("NVIDIA_API_KEY", secret)
    trajectory = sample_trajectory()
    trajectory["query"] = f"Never persist {secret}"
    trajectory["debug"] = {
        "NVIDIA_API_KEY": secret,
        "Authorization": f"Bearer {secret}",
    }

    path = TrajectoryStore(tmp_path).save(trajectory)
    serialized = Path(path).read_text(encoding="utf-8")

    assert secret not in serialized
    assert "[REDACTED]" in serialized


def test_save_returns_actual_output_path(tmp_path: Path) -> None:
    path = TrajectoryStore(tmp_path).save(sample_trajectory())

    assert Path(path).is_file()
    assert Path(path).parent == tmp_path.resolve()


def test_successful_agent_run_persists_ordered_steps(tmp_path: Path) -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(
            name="trace_path",
            arguments='{"source":"admin-net","destination":"orders-db"}',
        ),
    )
    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[tool_call])
            )
        ],
        usage=None,
    )
    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="The path uses internal-fw.", tool_calls=[])
            )
        ],
        usage=None,
    )
    create = Mock(side_effect=[first_response, final_response])
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = NetworkAgent(
        client=client,
        model="test-model",
        trajectory_store=TrajectoryStore(tmp_path),
        agent_version="v1",
    ).run("Trace the route")

    saved = load_saved(result["trajectory_path"])
    assert saved["completed"] is True
    assert saved["steps"][0]["step_index"] == 1
    assert saved["steps"][0]["tool"] == "trace_path"

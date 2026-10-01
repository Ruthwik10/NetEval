from pathlib import Path

import pytest

from src.network_engine import NetworkEngine


NETWORK_PATH = Path(__file__).parents[1] / "data" / "network.json"


@pytest.fixture(scope="module")
def engine() -> NetworkEngine:
    return NetworkEngine(NETWORK_PATH)


def test_network_json_loads_successfully(engine: NetworkEngine) -> None:
    assert engine.graph.number_of_nodes() == 11
    assert engine.graph.number_of_edges() == 10


def test_expected_devices_exist(engine: NetworkEngine) -> None:
    expected_devices = {
        "public-web",
        "orders-api",
        "orders-db",
        "billing-api",
        "billing-db",
        "edge-fw",
        "internal-fw",
        "admin-net",
    }

    assert all(engine.device_exists(device_id) for device_id in expected_devices)


def test_orders_db_details(engine: NetworkEngine) -> None:
    orders_db = engine.get_device("orders-db")

    assert orders_db is not None
    assert orders_db["type"] == "database"
    assert orders_db["status"] == "up"
    assert any(
        service["name"] == "postgresql" and service["port"] == 5432
        for service in orders_db["services"]
    )


def test_billing_db_is_down(engine: NetworkEngine) -> None:
    billing_db = engine.get_device("billing-db")

    assert billing_db is not None
    assert billing_db["status"] == "down"


def test_admin_to_orders_db_path_includes_internal_firewall(
    engine: NetworkEngine,
) -> None:
    path = engine.get_path("admin-net", "orders-db")

    assert path is not None
    assert "internal-fw" in path


def test_admin_to_orders_db_acl_lookup_returns_acl_17(
    engine: NetworkEngine,
) -> None:
    rules = engine.get_acl_rules(
        source="admin-net",
        destination="orders-db",
        port=5432,
    )

    assert [rule["id"] for rule in rules] == ["acl-17"]


def test_default_policy_is_deny(engine: NetworkEngine) -> None:
    assert engine.get_default_policy() == "deny"

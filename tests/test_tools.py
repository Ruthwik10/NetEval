import pytest

from src.tools import (
    check_reachability,
    find_blocking_hop,
    get_device_state,
    inspect_acl,
    trace_path,
)


def test_trace_path_includes_internal_firewall() -> None:
    result = trace_path("admin-net", "orders-db")

    assert result["path_found"] is True
    assert "internal-fw" in result["path"]


def test_get_device_state_reports_billing_db_down() -> None:
    result = get_device_state("billing-db")

    assert result["status"] == "down"


def test_inspect_acl_returns_acl_17_deny() -> None:
    result = inspect_acl(
        "internal-fw",
        "admin-net",
        "orders-db",
        5432,
    )

    assert len(result["matched_rules"]) == 1
    assert result["matched_rules"][0]["id"] == "acl-17"
    assert result["matched_rules"][0]["action"] == "deny"


def test_find_blocking_hop_returns_acl_17() -> None:
    result = find_blocking_hop("admin-net", "orders-db", 5432)

    assert result == {
        "blocked": True,
        "reason": "explicit_deny",
        "blocking_device": "internal-fw",
        "rule_id": "acl-17",
    }


def test_admin_net_cannot_reach_orders_db() -> None:
    result = check_reachability("admin-net", "orders-db", 5432)

    assert result["reachable"] is False


def test_billing_db_is_unreachable_when_down() -> None:
    result = check_reachability("billing-api", "billing-db", 5432)

    assert result["reachable"] is False
    assert result["reason"] == "destination_down"


def test_orders_api_can_reach_orders_db() -> None:
    result = check_reachability("orders-api", "orders-db", 5432)

    assert result["reachable"] is True


def test_invalid_device_name_raises_clear_error() -> None:
    with pytest.raises(ValueError, match=r"^Unknown device: missing-device\.$"):
        trace_path("missing-device", "orders-db")

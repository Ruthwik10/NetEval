"""Deterministic network investigation tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.network_engine import NetworkEngine


NETWORK_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "network.json"
_ENGINE = NetworkEngine(NETWORK_DATA_PATH)


def _require_device(device_id: str) -> None:
    if not _ENGINE.device_exists(device_id):
        raise ValueError(f"Unknown device: {device_id}.")


def trace_path(source: str, destination: str) -> dict[str, Any]:
    """Return the shortest topology path without evaluating traffic policy."""
    _require_device(source)
    _require_device(destination)

    path = _ENGINE.get_path(source, destination)
    return {
        "source": source,
        "destination": destination,
        "path": path,
        "path_found": path is not None,
    }


def get_device_state(device: str) -> dict[str, Any]:
    """Return state and service metadata for a device."""
    _require_device(device)
    device_data = _ENGINE.get_device(device)
    if device_data is None:  # Defensive guard after validation above.
        raise ValueError(f"Unknown device: {device}.")

    return {
        "device": device,
        "type": device_data["type"],
        "status": device_data["status"],
        "zone": device_data["zone"],
        "services": device_data["services"],
    }


def inspect_acl(
    device: str,
    source: str | None = None,
    destination: str | None = None,
    port: int | None = None,
    protocol: str | None = "tcp",
) -> dict[str, Any]:
    """Return priority-ordered ACL rules matching the supplied filters."""
    _require_device(device)
    if source is not None:
        _require_device(source)
    if destination is not None:
        _require_device(destination)

    rules = _ENGINE.get_acl_rules(
        device=device,
        source=source,
        destination=destination,
        port=port,
        protocol=protocol,
    )
    rules = sorted(rules, key=lambda rule: (rule["priority"], rule["id"]))

    return {
        "device": device,
        "matched_rules": [
            {key: value for key, value in rule.items() if key != "device"}
            for rule in rules
        ],
    }


def find_blocking_hop(
    source: str,
    destination: str,
    port: int,
    protocol: str = "tcp",
) -> dict[str, Any]:
    """Identify the deterministic reason traffic is blocked, if any."""
    _require_device(source)
    _require_device(destination)

    path = _ENGINE.get_path(source, destination)
    if path is None:
        return _blocking_result(reason="no_path")

    destination_data = _ENGINE.get_device(destination)
    if destination_data is None:  # Defensive guard after validation above.
        raise ValueError(f"Unknown device: {destination}.")
    if destination_data["status"] == "down":
        return _blocking_result(
            reason="destination_down",
            blocking_device=destination,
        )

    firewalls = [
        device_id
        for device_id in path
        if _ENGINE.graph.nodes[device_id]["type"] == "firewall"
    ]
    if not firewalls:
        return _not_blocked_result()

    firewall_decisions: list[tuple[str, dict[str, Any] | None]] = []
    for firewall in firewalls:
        rules = _ENGINE.get_acl_rules(
            device=firewall,
            source=source,
            destination=destination,
            port=port,
            protocol=protocol,
        )
        rules = sorted(rules, key=lambda rule: (rule["priority"], rule["id"]))
        winning_rule = rules[0] if rules else None
        firewall_decisions.append((firewall, winning_rule))

    for firewall, rule in firewall_decisions:
        if rule is not None and rule["action"] == "deny":
            return _blocking_result(
                reason="explicit_deny",
                blocking_device=firewall,
                rule_id=rule["id"],
            )

    if _ENGINE.get_default_policy() == "deny":
        for firewall, rule in firewall_decisions:
            if rule is None or rule["action"] != "allow":
                return _blocking_result(
                    reason="default_deny",
                    blocking_device=firewall,
                )

    return _not_blocked_result()


def check_reachability(
    source: str,
    destination: str,
    port: int,
    protocol: str = "tcp",
) -> dict[str, Any]:
    """Combine topology, destination state, and ACL policy into one result."""
    _require_device(source)
    _require_device(destination)

    path = _ENGINE.get_path(source, destination)
    blocking = find_blocking_hop(source, destination, port, protocol)
    return {
        "reachable": not blocking["blocked"],
        "source": source,
        "destination": destination,
        "port": port,
        "protocol": protocol,
        "path": path,
        "blocking_device": blocking["blocking_device"],
        "rule_id": blocking["rule_id"],
        "reason": blocking["reason"],
    }


def _blocking_result(
    reason: str,
    blocking_device: str | None = None,
    rule_id: str | None = None,
) -> dict[str, Any]:
    return {
        "blocked": True,
        "reason": reason,
        "blocking_device": blocking_device,
        "rule_id": rule_id,
    }


def _not_blocked_result() -> dict[str, Any]:
    return {
        "blocked": False,
        "reason": None,
        "blocking_device": None,
        "rule_id": None,
    }

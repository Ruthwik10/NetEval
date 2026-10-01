"""Deterministic network topology loading and lookup primitives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx


class NetworkEngine:
    """Load, validate, and expose a synthetic network as a NetworkX graph."""

    REQUIRED_TOP_LEVEL_FIELDS = {
        "devices",
        "links",
        "acl_rules",
        "zones",
        "metadata",
    }

    def __init__(self, network_path: str | Path) -> None:
        self.network_path = Path(network_path)
        self.data = self._load_data()
        self._validate_data()

        self.devices = {device["id"]: device for device in self.data["devices"]}
        self.links = self.data["links"]
        self.acl_rules = self.data["acl_rules"]
        self.zones = self.data["zones"]
        self.metadata = self.data["metadata"]
        self.graph = self._build_graph()

    def _load_data(self) -> dict[str, Any]:
        try:
            with self.network_path.open(encoding="utf-8") as network_file:
                data = json.load(network_file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in network data file '{self.network_path}': {exc.msg}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError("Network data must be a JSON object.")

        return data

    def _validate_data(self) -> None:
        missing_fields = self.REQUIRED_TOP_LEVEL_FIELDS - self.data.keys()
        if missing_fields:
            fields = ", ".join(sorted(missing_fields))
            raise ValueError(f"Network data is missing required fields: {fields}.")

        for field in ("devices", "links", "acl_rules", "zones"):
            if not isinstance(self.data[field], list):
                raise ValueError(f"Network field '{field}' must be a list.")
        if not isinstance(self.data["metadata"], dict):
            raise ValueError("Network field 'metadata' must be an object.")
        if "default_policy" not in self.data["metadata"]:
            raise ValueError("Network metadata is missing required field: default_policy.")

        zones = self.data["zones"]
        if any(not isinstance(zone, str) for zone in zones):
            raise ValueError("Every zone must be a string.")

        device_ids: set[str] = set()
        for index, device in enumerate(self.data["devices"]):
            self._require_object_fields(
                device,
                {"id", "type", "status", "zone", "services"},
                f"device at index {index}",
            )
            device_id = device["id"]
            if not isinstance(device_id, str) or not device_id:
                raise ValueError(f"Device at index {index} has an invalid ID.")
            if device_id in device_ids:
                raise ValueError(f"Duplicate device ID: {device_id}.")
            if device["zone"] not in zones:
                raise ValueError(
                    f"Device '{device_id}' references unknown zone: {device['zone']}."
                )
            if not isinstance(device["services"], list):
                raise ValueError(f"Device '{device_id}' field 'services' must be a list.")
            device_ids.add(device_id)

        for index, link in enumerate(self.data["links"]):
            self._require_object_fields(
                link,
                {"source", "destination", "bidirectional"},
                f"link at index {index}",
            )
            source = link["source"]
            destination = link["destination"]
            if source not in device_ids:
                raise ValueError(f"Link references unknown source device: {source}.")
            if destination not in device_ids:
                raise ValueError(
                    f"Link references unknown destination device: {destination}."
                )

        acl_ids: set[str] = set()
        required_acl_fields = {
            "id",
            "device",
            "source",
            "destination",
            "port",
            "protocol",
            "action",
        }
        for index, rule in enumerate(self.data["acl_rules"]):
            self._require_object_fields(
                rule,
                required_acl_fields,
                f"ACL rule at index {index}",
            )
            acl_id = rule["id"]
            if not isinstance(acl_id, str) or not acl_id:
                raise ValueError(f"ACL rule at index {index} has an invalid ID.")
            if acl_id in acl_ids:
                raise ValueError(f"Duplicate ACL ID: {acl_id}.")
            for field in ("source", "destination", "device"):
                referenced_device = rule[field]
                if referenced_device not in device_ids:
                    raise ValueError(
                        f"ACL rule '{acl_id}' references unknown {field} device: "
                        f"{referenced_device}."
                    )
            acl_ids.add(acl_id)

    @staticmethod
    def _require_object_fields(
        value: Any, required_fields: set[str], context: str
    ) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"Network {context} must be an object.")
        missing_fields = required_fields - value.keys()
        if missing_fields:
            fields = ", ".join(sorted(missing_fields))
            raise ValueError(f"Network {context} is missing required fields: {fields}.")

    def _build_graph(self) -> nx.Graph:
        graph = nx.Graph()
        for device in self.data["devices"]:
            graph.add_node(
                device["id"],
                type=device["type"],
                status=device["status"],
                zone=device["zone"],
                services=device["services"],
            )

        for link in self.data["links"]:
            graph.add_edge(
                link["source"],
                link["destination"],
                bidirectional=link["bidirectional"],
            )

        return graph

    def get_device(self, device_id: str) -> dict[str, Any] | None:
        """Return a device record, or None when the device does not exist."""
        return self.devices.get(device_id)

    def device_exists(self, device_id: str) -> bool:
        """Return whether a device is present in the network."""
        return device_id in self.devices

    def get_neighbors(self, device_id: str) -> list[str]:
        """Return devices directly connected to the requested device."""
        if not self.device_exists(device_id):
            return []
        return list(self.graph.neighbors(device_id))

    def get_path(self, source: str, destination: str) -> list[str] | None:
        """Return the shortest physical path, or None when no path exists."""
        if not self.device_exists(source) or not self.device_exists(destination):
            return None
        try:
            return nx.shortest_path(self.graph, source=source, target=destination)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_acl_rules(
        self,
        device: str | None = None,
        source: str | None = None,
        destination: str | None = None,
        port: int | None = None,
        protocol: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return ACL rules matching every supplied filter."""
        filters = {
            "device": device,
            "source": source,
            "destination": destination,
            "port": port,
            "protocol": protocol,
        }
        return [
            rule
            for rule in self.acl_rules
            if all(
                expected is None or rule[field] == expected
                for field, expected in filters.items()
            )
        ]

    def get_default_policy(self) -> str:
        """Return the network's configured default ACL policy."""
        return self.metadata["default_policy"]

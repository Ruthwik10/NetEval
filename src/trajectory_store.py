"""JSON persistence for NetEval agent trajectories."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


class TrajectoryStore:
    """Persist agent trajectories as readable, secret-scrubbed JSON files."""

    def __init__(self, output_dir: str | Path = "results/trajectories") -> None:
        self.output_dir = Path(output_dir)

    def save(self, trajectory: dict[str, Any]) -> str:
        """Write a trajectory and return its resolved output path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self._filename_component(
            str(trajectory.get("timestamp", "unknown-time")).replace(":", "-")
        )
        run_id = self._filename_component(str(trajectory.get("run_id", "unknown-run")))
        version = self._filename_component(
            str(trajectory.get("agent_version", "unknown-version"))
        )
        output_path = self.output_dir / f"{timestamp}_{run_id}_{version}.json"

        safe_trajectory = self._redact_secrets(trajectory)
        with output_path.open("w", encoding="utf-8") as trajectory_file:
            json.dump(safe_trajectory, trajectory_file, indent=2, ensure_ascii=False)
            trajectory_file.write("\n")

        return str(output_path.resolve())

    @staticmethod
    def _filename_component(value: str) -> str:
        safe_value = re.sub(r"[^A-Za-z0-9._-]", "-", value)
        return safe_value.strip(".-") or "unknown"

    @classmethod
    def _redact_secrets(cls, value: Any, key: str | None = None) -> Any:
        sensitive_keys = {"nvidia_api_key", "authorization"}
        if key is not None and key.lower() in sensitive_keys:
            return "[REDACTED]"

        if isinstance(value, dict):
            return {
                item_key: cls._redact_secrets(item_value, str(item_key))
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [cls._redact_secrets(item) for item in value]
        if isinstance(value, tuple):
            return [cls._redact_secrets(item) for item in value]
        if isinstance(value, str):
            redacted = value
            api_key = os.getenv("NVIDIA_API_KEY")
            if api_key:
                redacted = redacted.replace(api_key, "[REDACTED]")
            return re.sub(
                r"(?i)\bBearer\s+[^\s,;]+",
                "Bearer [REDACTED]",
                redacted,
            )
        return value

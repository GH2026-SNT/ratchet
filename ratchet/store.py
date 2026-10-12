"""Where harness versions and attempts are recorded. LocalStore writes a JSON file;
ratchet/live/atlas.py has MongoStore with the same two methods."""
from __future__ import annotations

import json
from typing import Any, Dict


class LocalStore:
    def __init__(self, path: str = "ratchet_store.json"):
        self.path = path
        self.data: Dict[str, Any] = {"harness_versions": [], "attempts": []}

    def _flush(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def save_version(self, version: Dict[str, Any]) -> None:
        self.data["harness_versions"].append(version)
        self._flush()

    def save_attempt(self, attempt: Dict[str, Any]) -> None:
        self.data["attempts"].append(attempt)
        self._flush()

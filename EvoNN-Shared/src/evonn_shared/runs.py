"""Run identity helpers for shared EvoNN artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256

from pydantic import BaseModel, ConfigDict


class RunCoordinates(BaseModel):
    """Stable run identity envelope for shared manifests and exports."""

    model_config = ConfigDict(frozen=True)

    system: str
    run_id: str
    pack_name: str
    seed: int

    def default_artifact_prefix(self) -> str:
        return f"{self.system}/{self.pack_name}/{self.run_id}"

    def data_signature(self) -> str:
        payload = f"{self.system}|{self.pack_name}|{self.seed}|{self.run_id}"
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

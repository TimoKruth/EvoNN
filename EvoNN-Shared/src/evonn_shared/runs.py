"""Run identity helpers for shared EvoNN artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256

from pydantic import BaseModel, ConfigDict


_DATA_SIGNATURE_HEX_LENGTH = 16
_DATA_SIGNATURE_SEPARATOR = "|"


class RunCoordinates(BaseModel):
    """Stable run identity envelope for shared manifests and exports."""

    model_config = ConfigDict(frozen=True)

    system: str
    run_id: str
    pack_name: str
    seed: int

    def default_artifact_prefix(self) -> str:
        """Return the canonical artifact prefix for run-scoped outputs."""

        return f"{self.system}/{self.pack_name}/{self.run_id}"

    def data_signature(self) -> str:
        """Return the stable short hash used to correlate run data artifacts."""

        payload = _DATA_SIGNATURE_SEPARATOR.join(
            (self.system, self.pack_name, str(self.seed), self.run_id)
        )
        return sha256(payload.encode("utf-8")).hexdigest()[:_DATA_SIGNATURE_HEX_LENGTH]


def utc_now_iso() -> str:
    """Return the current UTC timestamp in Python's ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()

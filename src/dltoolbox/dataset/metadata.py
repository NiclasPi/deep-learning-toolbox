from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


@dataclass(frozen=True, kw_only=True)
class DatasetMetadata:
    name: str
    split: Literal["train", "valid", "test"]
    description: str = ""

    origin_path: str
    sample_ids: list[str]
    sample_meta: dict[str, Any] = field(default_factory=dict)

    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_hash: str | None = None

    comment: str | None = None

    @staticmethod
    def from_json_bytes(json_bytes: bytes) -> DatasetMetadata:
        """Parse DatasetMetadata from JSON bytes."""
        data = json.loads(json_bytes.decode("utf-8"))
        return DatasetMetadata(**data)

    def to_json_bytes(self) -> bytes:
        """Return JSON representation as bytes."""
        return json.dumps(
            asdict(self),
            ensure_ascii=False,
            separators=(",", ":"),  # for most compact JSON representation
        ).encode("utf-8")

    @property
    def num_samples(self) -> int:
        return len(self.sample_ids)

    def get_sample_meta(self, sample_id: str) -> Any | None:
        if sample_id in self.sample_meta:
            return self.sample_meta[sample_id]
        return None

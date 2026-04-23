from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class DatasetMetadata:
    """Slim, JSON-serializable header describing a dataset file.

    Intended to live in the HDF5 userblock as a small, fast-to-read summary of the file
    (identity, split, size, provenance) — not per-sample metadata, which is stored separately
    as HDF5 datasets and accessed via a `SampleMetaStore`.

    Immutable by design: once a dataset is built, its header should not change. Serialization
    uses stdlib `json`, so all fields (including anything placed in `extra_data`) must be
    JSON-native; `created_at` is handled specially via ISO-8601.
    """

    # schema version of this header; bump when the on-disk layout changes in an incompatible way
    version: int = 1

    # human-readable dataset name
    name: str
    # which ML split this file represents
    split: Literal["train", "valid", "test"]
    # free-form long description
    description: str = ""

    # total number of samples stored in the file (pre-selection); cross-checked against the data dataset at load
    num_samples: int

    # stable identifier of where the raw inputs came from (e.g. source directory); not a path to this .h5
    origin_path: str

    # escape hatch for project-specific fields that don't warrant a schema change; must be JSON-serializable
    extra_data: dict = field(default_factory=dict)

    # UTC timestamp of when this metadata object was constructed (i.e. dataset build time)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # commit hash of the code that produced the dataset, for reproducibility
    git_hash: str | None = None

    # freeform human note
    comment: str | None = None

    @classmethod
    def from_json_bytes(cls, json_bytes: bytes) -> DatasetMetadata:
        """Parse DatasetMetadata from JSON bytes."""
        data = json.loads(json_bytes)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

    def to_json_bytes(self) -> bytes:
        """Return JSON representation as bytes."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return json.dumps(d).encode("utf-8")

    @property
    def filename(self) -> str:
        """Derive a safe filename from the dataset's name.

        The returned value is not the original source filename but a filesystem-safe identifier
        derived from `self.name`. Can be used to create new files that are derived from this dataset.
        """
        # turn accents into ASCII equivalents and drop non-ASCII
        s = unicodedata.normalize("NFKD", self.name).encode("ascii", "ignore").decode("ascii")
        # replace spaces with underscore
        s = s.replace(" ", "_")
        # remove any slashes (forward or back)
        s = re.sub(r"[\\/]", "", s)
        # add the dataset split indicator
        s += "__" + self.split
        return s

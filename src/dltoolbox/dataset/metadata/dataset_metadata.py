from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from typing import Literal, TypeVar

from attrs import field, frozen
from cattrs import Converter
from cattrs.preconf.json import make_converter

T = TypeVar("T")

_default_converter = make_converter()


@frozen(kw_only=True)
class DatasetMetadata:
    version: int = 1

    name: str
    split: Literal["train", "valid", "test"]
    description: str = ""

    # the number of samples this dataset holds
    num_samples: int

    # a unique identifier of the origin (e.g. a filesystem path to the files' directory)
    origin_path: str

    # additional data fields
    extra_data: dict = field(factory=dict)

    created_at: datetime = field(factory=lambda: datetime.now(UTC))
    git_hash: str | None = None

    comment: str | None = None

    @classmethod
    def from_json_bytes(cls, json_bytes: bytes, converter: Converter | None = None) -> DatasetMetadata:
        """Parse DatasetMetadata from JSON bytes."""

        if converter is None:
            converter = _default_converter
        return converter.loads(json_bytes, cls)

    def to_json_bytes(self, converter: Converter | None = None) -> bytes:
        """Return JSON representation as bytes."""

        if converter is None:
            converter = _default_converter
        return converter.dumps(converter.unstructure(self)).encode("utf-8")

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

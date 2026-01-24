import os
from typing import Optional


class Logger:
    def __init__(self, log_file: Optional[str] = None):
        self._log_file_path = os.path.abspath(log_file) if log_file is not None else ""
        self._debug = __debug__ or log_file is None

    def log(self, *values: object) -> None:
        print(*values, file=open(self._log_file_path, "a") if not self._debug else None, flush=True)

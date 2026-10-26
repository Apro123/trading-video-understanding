"""
Durable content-addressed disk cache and serialization utilities.
Used to prevent redundant API and local inference calls for VLM, LLM, and Grounding.
"""

import os
import re
import json
import time
import hashlib
import tempfile
import random
from pathlib import Path
from typing import Any, Callable, TypeVar, Optional

T = TypeVar("T")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Parse the first JSON value in text, tolerating ``` fences and surrounding prose."""
    if text is None:
        raise ValueError("empty model output")
    text = text.strip()
    for candidate in [m.group(1) for m in _FENCE.finditer(text)] + [text]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in model output: {text[:200]!r}")


def _default(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def stable_hash(*parts: Any) -> str:
    """Produces a deterministic 16-character SHA-256 hash for cache keys."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, bytes):
            h.update(p)
        else:
            h.update(json.dumps(p, sort_keys=True, default=_default).encode())
    return h.hexdigest()[:16]


def retry(
    fn: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 1.5,
    retry_on: tuple[type[BaseException], ...] = (Exception,)
) -> T:
    """Executes a callable with exponential backoff and jitter."""
    for i in range(attempts):
        try:
            return fn()
        except retry_on:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i) + random.random())
    raise RuntimeError("unreachable")


def write_json(path: Path, data: Any) -> None:
    """Atomic write of JSON to disk via temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1, default=_default)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    """Reads JSON file safely."""
    with open(path) as f:
        return json.load(f)


class DiskCache:
    """Content-addressed disk cache partitioned by namespace."""

    def __init__(self, namespace: str, base_dir: Optional[Path] = None):
        if base_dir is None:
            root = Path(__file__).resolve().parent
            base_dir = root / ".cache"
        self.dir = base_dir / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        try:
            write_json(self._path(key), value)
        except Exception:
            pass

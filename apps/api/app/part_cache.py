"""
On-demand lookup + cache: part detail, model metadata, model×part compatibility.
No full crawl; entries expire by TTL and are refetched on next request (DB or future PartSelect API).
"""
from __future__ import annotations

import re
import time
import threading
from typing import Any, Optional

# TTL in seconds: Part 7d, Model 7d, Compatibility 1d, Model-parts list 1d
TTL_PART_SEC = 7 * 24 * 3600
TTL_MODEL_SEC = 7 * 24 * 3600
TTL_COMPAT_SEC = 1 * 24 * 3600
TTL_MODEL_PARTS_SEC = 1 * 24 * 3600


def _normalize(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    s = s.strip().upper()
    s = re.sub(r"[\s\-]+", "", s)
    return s[:80]


def normalize_part_number(part_number: str) -> str:
    """Normalize case and strip spaces/dashes for cache key."""
    return _normalize(part_number or "")


def normalize_model_number(model_number: str) -> str:
    return _normalize(model_number or "")


class _TTLCache:
    def __init__(self):
        self._data: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, ttl_sec: float) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            val, expires = self._data[key]
            if time.monotonic() > expires:
                del self._data[key]
                return None
            return val

    def set(self, key: str, value: Any, ttl_sec: float) -> None:
        with self._lock:
            self._data[key] = (value, time.monotonic() + ttl_sec)


_cache = _TTLCache()


def cache_key_part(part_number: str) -> str:
    return f"part:{normalize_part_number(part_number)}"


def cache_key_model(model_number: str) -> str:
    return f"model:{normalize_model_number(model_number)}"


def cache_key_compat(model_number: str, part_number: str) -> str:
    return f"compat:{normalize_model_number(model_number)}:{normalize_part_number(part_number)}"


def get_part_cached(part_number: str) -> Optional[dict]:
    key = cache_key_part(part_number)
    return _cache.get(key, TTL_PART_SEC)


def set_part_cached(part_number: str, value: dict) -> None:
    key = cache_key_part(part_number)
    _cache.set(key, value, TTL_PART_SEC)


def get_model_cached(model_number: str) -> Optional[dict]:
    key = cache_key_model(model_number)
    return _cache.get(key, TTL_MODEL_SEC)


def set_model_cached(model_number: str, value: dict) -> None:
    key = cache_key_model(model_number)
    _cache.set(key, value, TTL_MODEL_SEC)


def get_compat_cached(model_number: str, part_number: str) -> Optional[dict]:
    key = cache_key_compat(model_number, part_number)
    return _cache.get(key, TTL_COMPAT_SEC)


def set_compat_cached(model_number: str, part_number: str, value: dict) -> None:
    key = cache_key_compat(model_number, part_number)
    _cache.set(key, value, TTL_COMPAT_SEC)


def cache_key_model_parts(model_number: str) -> str:
    return f"model_parts:{normalize_model_number(model_number)}"


def get_model_parts_cached(model_number: str) -> Optional[list]:
    """Cached list of part dicts for a model (from PartSelect or DB)."""
    key = cache_key_model_parts(model_number)
    return _cache.get(key, TTL_MODEL_PARTS_SEC)


def set_model_parts_cached(model_number: str, value: list) -> None:
    key = cache_key_model_parts(model_number)
    _cache.set(key, value, TTL_MODEL_PARTS_SEC)

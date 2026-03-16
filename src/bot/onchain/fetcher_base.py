"""Base class for on-chain data fetchers with rate limiting and TTL cache."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


class BaseFetcher(ABC):
    """Abstract base for API data fetchers.

    Provides:
    - aiohttp session management
    - TTL-based in-memory cache
    - Rate limiting via sleep between requests
    - Graceful degradation (returns None on failure)
    """

    def __init__(
        self,
        base_url: str,
        cache_ttl: float = 300.0,
        min_request_interval: float = 2.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._cache_ttl = cache_ttl
        self._min_request_interval = min_request_interval
        self._cache: dict[str, tuple[float, Any]] = {}
        self._last_request_time: float = 0.0
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"Accept": "application/json"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _get_cached(self, key: str) -> Any | None:
        """Return cached value if not expired, else None."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires_at, data = entry
        if time.monotonic() > expires_at:
            del self._cache[key]
            return None
        return data

    def _set_cached(self, key: str, data: Any) -> None:
        self._cache[key] = (time.monotonic() + self._cache_ttl, data)

    async def _rate_limit(self) -> None:
        """Simple rate limiting via sleep."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            import asyncio
            await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.monotonic()

    async def _get(self, path: str, params: dict | None = None) -> dict | None:
        """Make a GET request with caching and rate limiting.

        Returns parsed JSON dict or None on failure.
        """
        cache_key = f"{path}:{params}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        await self._rate_limit()

        try:
            session = await self._get_session()
            url = f"{self._base_url}{path}"
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning(
                        "api_request_failed",
                        fetcher=self.__class__.__name__,
                        url=url,
                        status=resp.status,
                    )
                    return None
                data = await resp.json()
                self._set_cached(cache_key, data)
                return data
        except Exception as e:
            logger.warning(
                "api_request_error",
                fetcher=self.__class__.__name__,
                path=path,
                error=str(e),
            )
            return None

    @abstractmethod
    async def fetch(self) -> Any:
        """Fetch and return processed data. Returns None on failure."""

"""A minimal stand-in for aiohttp.ClientSession used by the collector tests.

aioresponses 0.7.9 (its newest release) breaks against aiohttp >= 3.12, and pinning
a production dependency down to suit a test tool isn't worth it — this covers the
handful of aiohttp features the collectors actually use.
"""
from __future__ import annotations

from typing import Any

import aiohttp
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self.url = "https://fake.invalid/"  # replaced by FakeSession.get
        self._payload = payload
        self._body = body

    async def json(self) -> Any:
        return self._payload

    async def read(self) -> bytes:
        return self._body

    def raise_for_status(self) -> None:
        if self.status >= 400:
            url = URL(self.url)
            info = aiohttp.RequestInfo(
                url=url, method="GET", headers=CIMultiDictProxy(CIMultiDict()), real_url=url
            )
            raise aiohttp.ClientResponseError(
                info, (), status=self.status, message=f"HTTP {self.status}"
            )

    def release(self) -> None:
        pass

    @property
    def content(self) -> "_Body":
        return _Body(self._body)

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def iter_chunked(self, size: int):
        for start in range(0, len(self._data), size):
            yield self._data[start : start + size]


class _Awaitable:
    """Mimics aiohttp's request context manager: awaitable *and* an async CM."""

    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    def __await__(self):
        async def _inner() -> FakeResponse:
            return self._response

        return _inner().__await__()

    async def __aenter__(self) -> FakeResponse:
        return self._response

    async def __aexit__(self, *exc) -> bool:
        return False


class FakeSession:
    """`routes` maps a URL prefix to a response.

    A list of responses is consumed one per call (the last one repeats), so a retry
    can be given a different outcome than the first attempt.
    """

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.requests: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs) -> _Awaitable:
        self.requests.append((url, kwargs))
        for prefix, response in self.routes.items():
            if url.startswith(prefix):
                if isinstance(response, list):
                    response = response.pop(0) if len(response) > 1 else response[0]
                response.url = url
                return _Awaitable(response)
        raise aiohttp.ClientConnectionError(f"no route for {url}")

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

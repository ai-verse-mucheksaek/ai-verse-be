import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class ProcessingGate:
    def __init__(self, limit: int):
        self._semaphore = asyncio.Semaphore(limit)
        self.active = 0
        self.waiting = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        async with self._lock:
            self.waiting += 1
        await self._semaphore.acquire()
        async with self._lock:
            self.waiting -= 1
            self.active += 1
        return self

    async def __aexit__(self, *_):
        async with self._lock:
            self.active -= 1
        self._semaphore.release()

    def snapshot(self) -> dict[str, int]:
        return {"active": self.active, "waiting": self.waiting}


class RequestControlMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_minute: int, log_path: Path):
        super().__init__(app)
        self.limit = limit_per_minute
        self.requests: dict[str, deque[float]] = defaultdict(deque)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("its_me.access")
        if not self.logger.handlers:
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def dispatch(self, request: Request, call_next):
        request_id = uuid4().hex
        start = time.perf_counter()
        key = request.client.host if request.client else "local"
        now = time.monotonic()
        bucket = self.requests[key]
        while bucket and now - bucket[0] >= 60:
            bucket.popleft()
        if request.url.path.startswith("/api/") and len(bucket) >= self.limit:
            return JSONResponse({"detail": "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.", "request_id": request_id}, status_code=429, headers={"Retry-After": "60"})
        bucket.append(now)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        self.logger.info(json.dumps({"request_id": request_id, "method": request.method, "path": request.url.path, "status": response.status_code, "duration_ms": round((time.perf_counter() - start) * 1000, 2)}, ensure_ascii=False))
        return response

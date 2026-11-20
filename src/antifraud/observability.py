"""Basic API instrumentation. Do not log payment or personal data."""
from __future__ import annotations

import logging
import time
import json

from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

logger = logging.getLogger("antifraud.api")
REQUESTS = Counter(
    "antifraud_http_requests_total",
    "Total HTTP requests",
    ["method", "route", "status"],
)
LATENCY = Histogram(
    "antifraud_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "route"],
)


def install_observability(app: FastAPI) -> None:
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            # Use route templates / allowlist, never raw URLs or identifiers.
            route = request.scope.get("route")
            path = getattr(route, "path", None)
            if path not in {"/health", "/v1/model-info", "/v1/score", "/metrics"}:
                path = "other"
            elapsed = time.perf_counter() - started
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(request.method, path).observe(elapsed)
            logger.info(json.dumps({"event": "request_complete", "method": request.method, "route": path, "status_code": status, "duration_ms": round(elapsed * 1000, 2)}, ensure_ascii=False))

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

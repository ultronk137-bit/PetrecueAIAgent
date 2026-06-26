"""
FastAPI application entry point.
Configures the app, mounts routers, and sets up global exception handlers.
"""
import collections
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import agent, health
from app.config import get_settings
from app.utils.exceptions import PetRescueException
from app.utils.helpers import get_current_timestamp
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

class RateLimiter:
    """A thread-safe IP-based rolling window rate limiter."""
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self.requests = collections.defaultdict(collections.deque)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        queue = self.requests[ip]
        while queue and queue[0] < now - self.window:
            queue.popleft()
        if len(queue) < self.limit:
            queue.append(now)
            return True
        return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle handler."""
    logger.info(
        "PetRescue AI Agent starting up",
        extra={
            "environment": settings.environment,
            "project": settings.gcp_project_id,
            "primary_model": settings.openai_model,
            "fallback_model": settings.gemini_model,
        }
    )
    yield
    # Gracefully shut down HTTP clients
    from app.services.location_service import GoogleMapsLocationProvider
    await GoogleMapsLocationProvider.close_client()
    logger.info("PetRescue AI Agent shutting down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="PetRescue AI Agent",
        description=(
            "An autonomous AI agent that analyzes images of injured animals, "
            "determines rescue priority, locates nearby help, and generates "
            "a complete rescue plan in a single API call."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate Limiter setup
    rate_limiter = RateLimiter(
        limit=settings.rate_limit_requests,
        window=settings.rate_limit_window_seconds
    )

    # ── Request size limit middleware ─────────────────────────────────────────
    @app.middleware("http")
    async def size_limit_middleware(request: Request, call_next) -> Response:
        """Enforce request size limit before parsing the payload."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                # Allow 1MB buffer above max image size for headers/boundaries
                max_allowed = settings.max_image_size_bytes + 1_048_576
                if length > max_allowed:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "status": "error",
                            "message": f"Request payload too large. Max allowed is {settings.max_image_size_mb} MB.",
                            "timestamp": get_current_timestamp(),
                        }
                    )
            except ValueError:
                pass
        return await call_next(request)

    # ── Rate limiting middleware ──────────────────────────────────────────────
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next) -> Response:
        """Enforce rolling window IP-based rate limiting."""
        client_ip = request.client.host if request.client else "unknown"
        if not rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "message": "Too many requests. Please try again later.",
                    "timestamp": get_current_timestamp(),
                }
            )
        return await call_next(request)

    # ── Request logging middleware ────────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        """Log every incoming request and its response time."""
        start = time.time()
        response = await call_next(request)
        elapsed = round(time.time() - start, 3)
        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "elapsed_seconds": elapsed,
            }
        )
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(PetRescueException)
    async def petrescue_exception_handler(
        request: Request,
        exc: PetRescueException
    ) -> JSONResponse:
        """Handle all application-specific exceptions uniformly."""
        logger.error(
            "Application exception",
            extra={
                "path": request.url.path,
                "error": exc.message,
                "status_code": exc.status_code,
            }
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "message": exc.message,
                "details": exc.details,
                "timestamp": get_current_timestamp(),
            }
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Catch-all for unhandled exceptions — prevents raw stack traces leaking."""
        logger.error(
            "Unhandled exception",
            extra={"path": request.url.path, "error": str(exc)},
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An internal server error occurred.",
                "timestamp": get_current_timestamp(),
            }
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(agent.router)

    return app


app = create_app()

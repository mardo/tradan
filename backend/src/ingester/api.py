"""
FastAPI application.

Start with:
  cd backend && uv run uvicorn ingester.api:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Tradan API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


from .routers import klines as klines_router  # noqa: E402
from .routers import jobs as jobs_router      # noqa: E402

app.include_router(klines_router.router, prefix="/api")
app.include_router(jobs_router.router, prefix="/api")

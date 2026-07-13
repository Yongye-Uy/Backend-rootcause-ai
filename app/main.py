from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.config import settings
from app.database import Base, engine
from app.routers import debug, reports, sessions

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(title="RootCause AI")

if settings.cors_origins_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(debug.router)
app.include_router(sessions.router)
app.include_router(reports.router)


@app.on_event("startup")
def on_startup() -> None:
    import app.models  # noqa: F401  (register models on Base before create_all)

    Base.metadata.create_all(bind=engine)


@app.get("/api/health")
def health_check() -> dict:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok", "database": "connected"}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import ai, interview, job_description, rag, resume, speech, user
from app.config import settings
from app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle application startup and shutdown events."""
    init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# Register API routers
app.include_router(user.router)
app.include_router(resume.router)
app.include_router(job_description.router)
app.include_router(rag.router)
app.include_router(ai.router)
app.include_router(interview.router)
app.include_router(speech.router)


@app.get(
    "/",
    tags=["Root"],
    summary="Root API health and status",
)
def read_root():
    return {
        "message": "AI-Powered Interview Coach API",
        "status": "running",
    }


@app.get(
    "/health",
    tags=["Root"],
    summary="Application health check",
)
def read_health():
    return {
        "status": "healthy",
    }

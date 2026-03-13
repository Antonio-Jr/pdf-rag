"""FastAPI application entry point with lifecycle management.

Initializes the database, checkpointer, and logging infrastructure
during startup, and gracefully shuts everything down on termination.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routers.chat import router as chat_router
from src.api.routers.upload import router as upload_router
from src.infrastructure.database import async_pool, build_checkpoint, init_database
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle.

    Initializes the PostgreSQL database (with pgvector extension),
    the LangGraph async checkpointer, and the logging system.
    On shutdown, closes the connection pool gracefully.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the application after infrastructure is ready.
    """
    logger.info("🚀 Starting infrastructure")
    try:
        async with init_database(), build_checkpoint():
            setup_logging()
            logger.info("✅ Infrastructure initialized")

            yield
            await async_pool.close()
        logger.info("🛑 Stopping application.")
    except Exception as e:
        logger.error(f"❌ Critical failure on infra initialization: {e}")


app = FastAPI(
    title="Resume Analyzer API",
    lifespan=lifespan,
)
app.include_router(chat_router)
app.include_router(upload_router)


@app.get("/health", tags=["Infrastructure"])
async def health_check():
    """Return the current health status of the API.

    Returns:
        A dictionary with the current service status.
    """
    return {"status": "online"}

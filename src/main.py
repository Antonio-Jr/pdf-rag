import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routers.chat import router as chat_router
from src.api.routers.upload import router as upload_router
from src.infrastructure.database import build_checkpoint, init_database, async_pool
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting infrastructure")
    try:
        async with init_database(), build_checkpoint():
            setup_logging()
            logger.info("✅ Infrastructure initialized")

            yield
            await async_pool.close()
        logger.info("🛑 Stoping application.")
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
    return {"status": "online"}

from contextlib import asynccontextmanager

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.shared.schemas.universal_discovery import UniversalDiscovery
from src.core.settings import settings
from src.utils.log_wrapper import get_logger, log_execution

POOL_CONNECTION_STRING = settings.DATABASE_URL

if "+asyncpg" in settings.DATABASE_URL:
    ASYNC_CONNECTION_STRING = settings.DATABASE_URL.replace(
        "postgresql+asyncpg://", "postgresql://"
    )
else:
    ASYNC_CONNECTION_STRING = settings.DATABASE_URL

async_pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
    conninfo=ASYNC_CONNECTION_STRING,
    min_size=2,
    max_size=10,
    timeout=30,
    open=False,
    kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
)

async_engine = create_async_engine(settings.DATABASE_URL)

_checkpointer: AsyncPostgresSaver | None = None


@asynccontextmanager
@log_execution
async def init_database():
    engine = create_async_engine(settings.DATABASE_URL)
    logger = get_logger(__name__)

    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            logger.info("🏛️ Extension PGVector was guaranteed (Async)")

        yield engine
    except Exception as e:
        logger.error(f"❌ Error during the db initialization: {e}")
        raise
    finally:
        await engine.dispose()
        logger.info("🛑 Database setup engine disposed.")


@asynccontextmanager
@log_execution
async def build_checkpoint():
    """Initializes LangGraph checkpointer schema in asynchronous way"""
    global _checkpointer
    logger = get_logger(__name__)

    try:
        await async_pool.open()
        await async_pool.wait()

        saver = AsyncPostgresSaver(async_pool)
        saver.serde = JsonPlusSerializer(allowed_msgpack_modules=[UniversalDiscovery])

        async with async_pool.connection():
            await saver.setup()

        _checkpointer = saver
        logger.info("🏛️ AsyncPostgresSaver initialized using global pool.")

        yield saver
    except Exception as e:
        logger.error(f"❌ Error during checkpoint initialization: {e}")
        raise
    finally:
        _checkpointer = None
        await async_pool.close()
        logger.info("🛑 Checkpointer pool closed")


def get_checkpointer():
    if _checkpointer is None:
        raise RuntimeError("Checkpointer was not initialized on lifespan")

    return _checkpointer

"""PostgreSQL database and LangGraph checkpointer initialization.

Manages the async connection pool (psycopg), the SQLAlchemy async engine,
and the LangGraph ``AsyncPostgresSaver`` checkpointer used for
conversation memory persistence.
"""

from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.core.settings import settings
from src.shared.schemas.universal_discovery import UniversalDiscovery
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
    """Initialize the PostgreSQL database with the pgvector extension.

    Creates a short-lived SQLAlchemy async engine, ensures the ``vector``
    extension exists, yields the engine for downstream use, and disposes
    of it on exit.

    Yields:
        The SQLAlchemy ``AsyncEngine`` instance.

    Raises:
        Exception: Propagates any database error after logging.
    """
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
    """Initialize the LangGraph async checkpointer backed by PostgreSQL.

    Opens the global connection pool, creates an ``AsyncPostgresSaver``
    with JSON-plus serialization, runs its schema migration, and exposes
    the saver via the module-level ``_checkpointer`` variable.

    Yields:
        The configured ``AsyncPostgresSaver`` instance.

    Raises:
        Exception: Propagates any initialization error after logging.
    """
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
    """Return the active LangGraph checkpointer.

    Returns:
        The ``AsyncPostgresSaver`` initialized during application startup.

    Raises:
        RuntimeError: If called before the checkpointer has been set up
                      via :func:`build_checkpoint`.
    """
    if _checkpointer is None:
        raise RuntimeError("Checkpointer was not initialized on lifespan")

    return _checkpointer

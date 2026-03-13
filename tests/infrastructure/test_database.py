"""Tests for the Database configuration and checkpointer."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.infrastructure.database import build_checkpoint, get_checkpointer, init_database


@pytest.mark.asyncio
async def test_init_database_success(mocker):
    """Test successful database initialization and extension creation."""
    mock_create_engine = mocker.patch("src.infrastructure.database.create_async_engine")
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine
    
    mock_conn = AsyncMock()
    mock_engine.begin.return_value.__aenter__.return_value = mock_conn
    mock_engine.dispose = AsyncMock()
    
    async with init_database() as engine:
        assert engine == mock_engine
        
    # Verify extension was requested
    mock_conn.execute.assert_called_once()
    assert "CREATE EXTENSION IF NOT EXISTS vector;" in str(mock_conn.execute.call_args[0][0])
    
    # Verify cleanup occurred
    mock_engine.dispose.assert_called_once()


@pytest.mark.asyncio
async def test_init_database_failure(mocker):
    """Test database initialization error propagation and cleanup."""
    mock_create_engine = mocker.patch("src.infrastructure.database.create_async_engine")
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine
    
    mock_conn = AsyncMock()
    # Simulate DB error during extension creation
    mock_conn.execute.side_effect = Exception("DB Connection Refused")
    mock_engine.begin.return_value.__aenter__.return_value = mock_conn
    mock_engine.dispose = AsyncMock()
    
    with pytest.raises(Exception, match="DB Connection Refused"):
        async with init_database():
            pass
            
    # Verify cleanup still triggered
    mock_engine.dispose.assert_called_once()


@pytest.mark.asyncio
async def test_build_checkpoint_success(mocker):
    """Test successful initialization of the LangGraph checkpointer."""
    mock_pool = mocker.patch("src.infrastructure.database.async_pool")
    mock_pool.open = AsyncMock()
    mock_pool.wait = AsyncMock()
    mock_pool.close = AsyncMock()
    mock_pool.connection.return_value.__aenter__.return_value = AsyncMock()
    
    mock_saver_cls = mocker.patch("src.infrastructure.database.AsyncPostgresSaver")
    mock_saver = AsyncMock()
    mock_saver_cls.return_value = mock_saver
    
    async with build_checkpoint() as saver:
        assert saver == mock_saver
        # Confirm it's registered globally
        assert get_checkpointer() == mock_saver
        
    mock_pool.open.assert_called_once()
    mock_saver.setup.assert_called_once()
    mock_pool.close.assert_called_once()
    
    # Assert global registry returns to None
    with pytest.raises(RuntimeError, match="Checkpointer was not initialized"):
        get_checkpointer()

@pytest.mark.asyncio
async def test_build_checkpoint_failure(mocker):
    """Test checkpointer failure propagation."""
    mock_pool = mocker.patch("src.infrastructure.database.async_pool")
    mock_pool.open = AsyncMock(side_effect=Exception("Pool Exhausted"))
    mock_pool.close = AsyncMock()
    
    mock_pool.close = AsyncMock()
    
    with pytest.raises(Exception, match="Pool Exhausted"):
        async with build_checkpoint():
            pass
            
    mock_pool.close.assert_called_once()

def test_database_url_sync_fallback(mocker, monkeypatch):
    """Test database connection string parses successfully when asyncpg is missing."""
    import importlib
    import sys
    
    from src.infrastructure import database
    
    # Store old for cleanup
    old_url = database.settings.DATABASE_URL
    
    # Inject dummy sys modules that natively block SQLAlchemy sync initialization tracking crashes
    mock_engine = mocker.MagicMock()
    mock_lc = mocker.MagicMock()
    
    monkeypatch.setitem(sys.modules, "sqlalchemy.ext.asyncio", mock_engine)
    monkeypatch.setitem(sys.modules, "langchain_postgres", mock_lc)
    
    try:
        mocker.patch("src.infrastructure.database.settings.DATABASE_URL", "postgresql://user:pass@localhost/db")
        importlib.reload(database)
        assert database.ASYNC_CONNECTION_STRING == "postgresql://user:pass@localhost/db"
    finally:
        # Restore old URL and reload to prevent breaking downstream asyncpg tests
        mocker.patch("src.infrastructure.database.settings.DATABASE_URL", old_url)
        importlib.reload(database)

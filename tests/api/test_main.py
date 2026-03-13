"""Tests for the main application lifecycle and healthcheck."""


import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client_main():
    return TestClient(app)


def test_health_check(client_main):
    """Test the /health endpoint."""
    response = client_main.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "online"}


@pytest.mark.asyncio
async def test_lifespan_success(mocker):
    """Test standard lifespan application boot up."""
    mocker.patch("src.main.setup_logging")

    mock_init_db = mocker.patch("src.main.init_database")
    mock_db_context = mocker.AsyncMock()
    mock_init_db.return_value = mock_db_context

    mock_build_checkpoint = mocker.patch("src.main.build_checkpoint")
    mock_checkpoint_context = mocker.AsyncMock()
    mock_build_checkpoint.return_value = mock_checkpoint_context

    from fastapi import FastAPI

    from src.main import lifespan

    app_instance = FastAPI()

    async with lifespan(app_instance):
        pass  # App is running during yield

    mock_init_db.assert_called_once()
    mock_build_checkpoint.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_failure(mocker):
    """Test lifespan application handles exceptions gracefully."""
    mocker.patch("src.main.setup_logging")

    mock_init_db = mocker.patch("src.main.init_database")
    mock_init_db.return_value.__aenter__.side_effect = Exception("DB Timeout")

    mock_logger = mocker.patch("src.main.logger")

    from fastapi import FastAPI

    from src.main import lifespan

    app_instance = FastAPI()

    with pytest.raises(RuntimeError):
        async with lifespan(app_instance):
            pass

    mock_logger.error.assert_called_once()
    assert "DB Timeout" in mock_logger.error.call_args[0][0]

"""Pytest configuration and shared fixtures for the test suite."""

from typing import AsyncGenerator

import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.main import app


@pytest_asyncio.fixture()
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Provide an asynchronous HTTP client for testing the FastAPI app.

    Uses ASGITransport to bypass the network and directly call the ASGI app.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture()
def client() -> TestClient:
    """Provide a synchronous HTTP client for simple endpoint tests."""
    return TestClient(app)

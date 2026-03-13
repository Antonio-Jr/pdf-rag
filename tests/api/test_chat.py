"""Tests for the chat endpoints in routers/chat.py."""

from fastapi.testclient import TestClient


async def mock_analyze_query_generator(*args, **kwargs):
    """Mock async generator to simulate the LangGraph streaming response."""
    yield "Hello"
    yield " "
    yield "World!"


def test_chat_streaming_success(client: TestClient, mocker):
    """Test the chat endpoint ensuring it streams the expected response."""
    mocker.patch(
        "src.api.routers.chat.analyze_query",
        side_effect=mock_analyze_query_generator,
    )

    payload = {
        "query": "What is in the document?",
        "thread_id": "session_123",
        "user_id": "user_456",
    }

    with client.stream("POST", "/chat/", json=payload) as response:
        assert response.status_code == 200
        # Check that the media_type is exactly what the code specified
        assert response.headers["content-type"].startswith("text/plain")

        chunks = list(response.iter_text())
        assert "".join(chunks) == "Hello World!"


def test_chat_internal_error(client: TestClient, mocker):
    """Test the chat endpoint behavior when the agent raises an error."""
    mocker.patch(
        "src.api.routers.chat.analyze_query",
        side_effect=ValueError("LLM API quota exceeded"),
    )

    payload = {
        "query": "Will this fail?",
        "thread_id": "session_error",
        "user_id": "user_error",
    }

    response = client.post("/chat/", json=payload)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Agent Error"}

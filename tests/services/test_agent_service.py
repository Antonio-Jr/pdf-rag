"""Tests for the AgentService."""

import pytest

from src.services.agents.agent_service import analyze_query


@pytest.fixture
def mock_graph_stream(mocker):
    """Mock the LangGraph astream_events method to simulate streaming output."""
    mock_generate = mocker.patch("src.services.agents.agent_service.generate_graph")
    mock_compiled_graph = mocker.AsyncMock()
    mock_generate.return_value.compile.return_value = mock_compiled_graph

    async def mock_astream(*args, **kwargs):
        """Simulate a stream of chunked events from the LangGraph model."""
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "chatbot"},
            "data": {"chunk": mocker.Mock(content="Hello")},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "chatbot"},
            "data": {"chunk": mocker.Mock(content=" World!")},
        }
        # Simulate an ignored event (tool call or internal routing)
        yield {
            "event": "on_tool_start",
            "metadata": {"langgraph_node": "tools"},
            "name": "discovery_tool",
            "data": {},
        }

    mock_compiled_graph.astream_events = mocker.MagicMock(side_effect=mock_astream)

    # We must also mock `get_checkpointer` to bypass DB operations
    mocker.patch("src.services.agents.agent_service.get_checkpointer")

    return mock_compiled_graph


@pytest.mark.asyncio
async def test_analyze_query_streaming(mock_graph_stream):
    """Test that analyze_query correctly yields text chunks from the graph stream."""

    # Create the generator
    generator = analyze_query(user_input="Test query", thread_id="thread_1", user_id="user_1")

    # Consume the async generator
    chunks = [chunk async for chunk in generator]

    # It should only yield the 'content' from 'on_chat_model_stream' events
    # where the node is 'chatbot'
    assert chunks == ["Hello", " World!"]

    # Ensure the underlying graph was called with correct configuration
    mock_graph_stream.astream_events.assert_called_once()

    call_args = mock_graph_stream.astream_events.call_args
    assert call_args[0][0] == {
        "messages": [("user", "Test query")],
        "query": "Test query",
        "user_id": "thread_1",
    }

    config = call_args[1]["config"]
    assert config["configurable"]["thread_id"] == "thread_1"
    assert config["configurable"]["user_id"] == "user_1"


@pytest.mark.asyncio
async def test_analyze_query_streaming_list_chunks(mocker):
    """Test analyze_query parsing list-formatted chunk streams strictly isolated."""
    mock_generate = mocker.patch("src.services.agents.agent_service.generate_graph")
    mock_compiled_graph = mocker.AsyncMock()
    mock_generate.return_value.compile.return_value = mock_compiled_graph

    async def mock_astream(*args, **kwargs):
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "chatbot"},
            "data": {
                "chunk": mocker.Mock(
                    content=[
                        {"type": "text", "text": "Structured"},
                        {"type": "image_url", "image_url": "should_ignore"},
                        " String in list ",
                    ]
                )
            },
        }

    mock_compiled_graph.astream_events = mocker.MagicMock(side_effect=mock_astream)
    mocker.patch("src.services.agents.agent_service.get_checkpointer")

    generator = analyze_query(user_input="List chunks", thread_id="t1", user_id="u1")

    chunks = [chunk async for chunk in generator]
    assert chunks == ["Structured", " String in list "]

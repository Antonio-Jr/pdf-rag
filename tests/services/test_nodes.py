"""Tests for the LangGraph nodes (chatbot and summarizer)."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.services.nodes.chabot_node import chatbot_node
from src.services.nodes.summarizer import summarizer_node
from src.services.states.graph_state import GraphState


@pytest.fixture
def mock_get_model(mocker):
    """Mock the LLM factory to return a controlled model."""
    mock = mocker.patch("src.services.nodes.chabot_node.get_model")
    return mock


@pytest.mark.asyncio
async def test_chatbot_node_success(mock_get_model, mocker):
    """Test the chatbot node invoking the model and updating state."""
    # Mock the prompt registry
    mock_prompt = mocker.patch("src.services.nodes.chabot_node.prompt_registry")
    mock_prompt.get_prompt.return_value.format_messages.return_value = [
        {"role": "system", "content": "You are a helpful AI"}
    ]

    # Setup mock model behavior
    mock_model_instance = MagicMock()
    mock_get_model.return_value.bind_tools.return_value = mock_model_instance

    mock_response = MagicMock(content="I am ready to help.", tool_calls=[])
    mock_model_instance.ainvoke = mocker.AsyncMock(return_value=mock_response)

    # Initial state
    initial_state = GraphState(
        messages=[HumanMessage(content="Hello")], query="Hello", user_id="user_test"
    )

    # Execute node
    new_state = await chatbot_node(initial_state)

    # Assert model was called with system prompt + user message
    mock_model_instance.ainvoke.assert_called_once()
    assert len(mock_model_instance.ainvoke.call_args[0][0]) == 2

    # Assert state was updated
    assert (
        isinstance(new_state.messages[0], AIMessage)
        or new_state.messages[0].content == "I am ready to help."
    )


@pytest.fixture
def mock_summarizer_model(mocker):
    """Mock the LLM factory for the summarizer."""
    mock = mocker.patch("src.services.nodes.summarizer.get_model")
    return mock


@pytest.mark.asyncio
async def test_summarizer_skips_short_history():
    """Test summarizer node bypasses processing if messages < threshold."""
    # State with only 2 messages
    initial_state = GraphState(
        messages=[HumanMessage(content="1"), HumanMessage(content="2")],
        summary="",
        query="test query",
        user_id="test_user",
    )

    new_state = await summarizer_node(initial_state)

    # Should return state completely unchanged
    assert new_state == initial_state


@pytest.mark.asyncio
async def test_summarizer_runs_long_history(mock_summarizer_model, mocker):
    """Test summarizer compresses history when over the threshold."""
    # State with 8 messages (threshold is 7)
    messages = [HumanMessage(content=f"msg_{i}") for i in range(8)]
    initial_state = GraphState(
        messages=messages, summary="Old summary", query="test query", user_id="test_user"
    )

    # Mock prompt registry and chain
    mock_prompt = mocker.patch("src.services.nodes.summarizer.prompt_registry")
    mock_base_prompt = MagicMock()
    mock_prompt.get_prompt.return_value = mock_base_prompt

    # We must patch the `__or__` operator which builds the LCEL chain: `prompt | model`
    mock_chain = MagicMock()
    mock_base_prompt.__add__.return_value.__or__.return_value = mock_chain
    mock_chain.ainvoke = mocker.AsyncMock(return_value=MagicMock(content="New compressed summary"))

    new_state = await summarizer_node(initial_state)

    # Assert state summary was updated
    assert new_state.summary == "New compressed summary"
    mock_chain.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_chatbot_node_with_tool_calls(mock_get_model, mocker):
    """Test that chatbot node gracefully iterates and logs invoked tools."""
    mock_prompt = mocker.patch("src.services.nodes.chabot_node.prompt_registry")
    mock_prompt.get_prompt.return_value.format_messages.return_value = [
        {"role": "sys", "content": "AI"}
    ]

    mock_logger = mocker.patch("src.services.nodes.chabot_node.get_logger")
    logger_instance = mock_logger.return_value

    mock_model_instance = MagicMock()
    mock_get_model.return_value.bind_tools.return_value = mock_model_instance

    mock_response = MagicMock(
        content="", tool_calls=[{"name": "extraction_tool", "args": {"points": ["a"]}}]
    )
    mock_model_instance.ainvoke = mocker.AsyncMock(return_value=mock_response)

    initial_state = GraphState(
        messages=[HumanMessage(content="Hello")], query="Hello", user_id="u1"
    )

    await chatbot_node(initial_state)
    logger_instance.info.assert_called_with(
        "🛠️ [TOOL CALL]: extraction_tool with args: {'points': ['a']}"
    )

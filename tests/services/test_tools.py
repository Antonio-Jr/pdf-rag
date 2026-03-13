"""Tests for the Discovery and Extraction tools."""

import json
from unittest.mock import MagicMock

import pytest

from src.services.tools.discovery import discovery_document_tool
from src.services.tools.extraction import extraction_data_tool


@pytest.fixture
def mock_retriever(mocker):
    """Mock the document retriever."""
    return mocker.patch("src.services.tools.discovery.get_retriever")


@pytest.fixture
def mock_extraction_retriever(mocker):
    return mocker.patch("src.services.tools.extraction.get_retriever")


@pytest.fixture
def mock_discovery_model(mocker):
    """Mock the structured LLM caller."""
    return mocker.patch("src.services.tools.discovery.get_model")


@pytest.fixture
def mock_extraction_model(mocker):
    return mocker.patch("src.services.tools.extraction.get_model")


@pytest.mark.asyncio
async def test_discovery_tool_success(mocker, mock_retriever, mock_discovery_model):
    """Test the discovery tool successfully fetching and structuring info."""
    mocker.patch("src.services.tools.discovery.prompt_registry")

    # Setup mock retriever docs
    mock_retriever_instance = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Document text here"
    mock_retriever_instance.ainvoke = mocker.AsyncMock(return_value=[mock_doc])
    mock_retriever.return_value = mock_retriever_instance

    # Setup mock model dump to return structured JSON
    mock_chain = MagicMock()
    mock_result = MagicMock()
    mock_result.model_dump.return_value = {
        "purpose": "A test doc",
        "entities": ["Company"],
        "trackable_points": ["KPIs"],
        "tone": "Formal",
    }
    mock_chain.ainvoke = mocker.AsyncMock(return_value=mock_result)

    # Mock LCEL `prompt | model`
    mock_prompt_instance = mocker.patch(
        "src.services.tools.discovery.prompt_registry.get_prompt"
    ).return_value
    mock_prompt_instance.__or__.return_value = mock_chain

    # Call the tool function explicitly giving the injected state
    result = await discovery_document_tool.ainvoke(
        {"query": "find overview", "user_id": "test_user_99"}
    )

    # Verify JSON structure
    data = json.loads(result)
    assert data["purpose"] == "A test doc"
    mock_retriever.assert_called_with(user_id="test_user_99")


@pytest.mark.asyncio
async def test_extraction_tool_success(mocker, mock_extraction_retriever, mock_extraction_model):
    """Test the extraction tool returning targeted data."""
    mocker.patch("src.services.tools.extraction.prompt_registry")

    mock_retriever_instance = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "We made $1M in revenue."
    mock_retriever_instance.ainvoke = mocker.AsyncMock(return_value=[mock_doc])
    mock_extraction_retriever.return_value = mock_retriever_instance

    mock_chain = MagicMock()
    mock_result = MagicMock()
    mock_result.content = "Revenue: $1M"
    mock_chain.ainvoke = mocker.AsyncMock(return_value=mock_result)

    mock_prompt_instance = mocker.patch(
        "src.services.tools.extraction.prompt_registry.get_prompt"
    ).return_value
    mock_prompt_instance.__or__.return_value = mock_chain

    result = await extraction_data_tool.ainvoke(
        {
            "discovery_summary": "Financial report",
            "tracking_points": ["revenue"],
            "user_id": "test_user_99",
        }
    )

    assert result == "Revenue: $1M"
    mock_extraction_retriever.assert_called_with(user_id="test_user_99")


@pytest.mark.asyncio
async def test_tool_missing_user_id():
    """Test tools validate missing state injects."""
    with pytest.raises(ValueError, match="was not injected"):
        await discovery_document_tool.ainvoke({"query": "find overview", "user_id": ""})

    with pytest.raises(ValueError, match="was not injected"):
        await extraction_data_tool.ainvoke(
            {"discovery_summary": "sum", "tracking_points": ["a", "b"], "user_id": ""}
        )


@pytest.mark.asyncio
async def test_discovery_tool_empty_docs(mocker):
    """Test discovery tool returning error on blank vector queries (or unassociated id)."""
    mock_retriever = mocker.patch("src.services.tools.discovery.get_retriever")
    instance = mock_retriever.return_value
    instance.ainvoke = mocker.AsyncMock(return_value=[])

    result = await discovery_document_tool.ainvoke({"query": "something", "user_id": "test_user"})

    assert "did not find any information in the docs" in result


@pytest.mark.asyncio
async def test_discovery_tool_blank_page_content(mocker, mock_discovery_model):
    """Test discovery tool parsing docs without valid readable text."""
    mock_retriever = mocker.patch("src.services.tools.discovery.get_retriever")
    instance = mock_retriever.return_value

    # Doc found but is completely blank string
    blank_doc = mocker.MagicMock(page_content="   \n ")
    instance.ainvoke = mocker.AsyncMock(return_value=[blank_doc])

    result = await discovery_document_tool.ainvoke({"query": "something", "user_id": "test"})

    assert "do not seem to contain legible text" in result


@pytest.mark.asyncio
async def test_extraction_tool_empty_docs(mocker):
    """Test extraction preventing processing if vector docs fail to return."""
    mock_retriever = mocker.patch("src.services.tools.extraction.get_retriever")
    instance = mock_retriever.return_value
    instance.ainvoke = mocker.AsyncMock(return_value=[])

    result = await extraction_data_tool.ainvoke(
        {"discovery_summary": "sum", "tracking_points": ["a", "b"], "user_id": "test_user"}
    )

    assert "ERROR: No content found" in result

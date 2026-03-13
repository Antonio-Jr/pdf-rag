"""Tests for the IngestionService."""

from unittest.mock import MagicMock

import pytest

from src.services.ingestion.ingestion_service import IngestionService


from langchain_core.documents import Document

@pytest.fixture(autouse=True)
def mock_embeddings(mocker):
    """Mock get_embeddings to prevent network calls during IngestionService.__init__."""
    return mocker.patch("src.services.ingestion.ingestion_service.get_embeddings")

@pytest.fixture
def mock_loader(mocker):
    """Mock the PyPDFLoader."""
    mock = mocker.patch("src.services.ingestion.ingestion_service.PyPDFLoader")
    instance = mock.return_value
    doc1 = Document(page_content="Page 1 Content", metadata={})
    doc2 = Document(page_content="Page 2 Content", metadata={})
    instance.load.return_value = [doc1, doc2]
    return mock


@pytest.fixture
def mock_splitter(mocker):
    """Mock the RecursiveCharacterTextSplitter."""
    mock = mocker.patch(
        "src.services.ingestion.ingestion_service.RecursiveCharacterTextSplitter"
    )
    instance = mock.return_value
    chunk_mock = Document(page_content="Chunk", metadata={})
    instance.split_documents.return_value = [chunk_mock] * 4
    return mock


@pytest.fixture
def mock_vector_store(mocker):
    """Mock the PGVector insertion call to bypass database."""
    # Mock the specific method being called in ingestion_service.py
    return mocker.patch("src.services.ingestion.ingestion_service.PGVector.afrom_documents")


@pytest.mark.asyncio
async def test_ingest_data_success(mock_loader, mock_splitter, mock_vector_store):
    """Test successful ingestion of a PDF document."""
    service = IngestionService()
    
    # Run the ingestion process
    chunk_count = await service.ingest_data(
        file_path="fake/path/doc.pdf",
        user_id="user_123",
        extra_metadata={"status": "uploaded"}
    )
    
    # Assertions
    assert chunk_count == 4
    
    # Verify loader was called with correct path
    mock_loader.assert_called_once_with("fake/path/doc.pdf")
    
    # Verify splitter was called
    mock_splitter.return_value.split_documents.assert_called_once()
    
    # Verify vector store was called to insert the 4 chunks
    mock_vector_store.assert_called_once()
    
    # Check that metadata was injected using the kwargs signature
    call_kwargs = mock_vector_store.call_args.kwargs
    documents_passed = call_kwargs.get("documents", [])
    assert len(documents_passed) == 4
    
    # Verify the user_id and extra metadata got injected into the first chunk
    first_doc_meta = documents_passed[0].metadata
    assert first_doc_meta["user_id"] == "user_123"
    assert first_doc_meta["status"] == "uploaded"


@pytest.mark.asyncio
async def test_ingest_data_empty_document(mocker, mock_loader):
    """Test behavior when the loader returns no documents."""
    # Override the mock to return empty list
    mock_loader.return_value.load.return_value = []
    
    service = IngestionService()
    
    # Expected to raise ValueError for empty document
    with pytest.raises(ValueError, match="No text could be extracted"):
        await service.ingest_data(file_path="empty.pdf", user_id="user_123")

"""Tests for the ingestion endpoints in routers/upload.py."""

import os

import pytest
from fastapi.testclient import TestClient


def test_upload_success(client: TestClient, mocker):
    """Test successful document upload and ingestion."""
    mock_service = mocker.patch("src.api.routers.upload.service")
    
    async def mock_ingest(*args, **kwargs):
        return 42

    mock_service.ingest_data.side_effect = mock_ingest

    files = {"files": ("test_doc.pdf", b"dummy pdf content", "application/pdf")}
    data = {"user_id": "test_user_123"}

    response = client.post("/upload/", files=files, data=data)

    assert response.status_code == 200
    assert response.json() == {
        "message": "Ingestion successful",
        "chunks": 42,
        "files": ["test_doc.pdf"],
    }

    mock_service.ingest_data.assert_called_once()


def test_upload_service_error(client: TestClient, mocker):
    """Test the upload endpoint when the ingestion service raises an error."""
    mock_service = mocker.patch("src.api.routers.upload.service")
    mock_service.ingest_data = mocker.AsyncMock(side_effect=Exception("Database down"))
    
    with open("dummy.pdf", "wb") as f:
        f.write(b"%PDF-1.4\n...")
        
    try:
        with open("dummy.pdf", "rb") as pdf_file:
            response = client.post(
                "/upload/",
                files=[("files", ("dummy.pdf", pdf_file, "application/pdf"))],
                data={"user_id": "test_user_789"}
            )
    finally:
        os.remove("dummy.pdf")

    assert response.status_code == 500
    assert "Database down" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_invalid_content_type(client):
    """Test uploading a file that is not a PDF."""
    response = client.post(
        "/upload/",
        files=[("files", ("dummy.txt", b"Hello", "text/plain"))],
        data={"user_id": "test_user_789"}
    )
    
    assert response.status_code == 400
    assert "Only PDF files are accepted" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_empty_file(client):
    """Test uploading a fully blank PDF buffer."""
    response = client.post(
        "/upload/",
        files=[("files", ("blank.pdf", b"", "application/pdf"))],
        data={"user_id": "test_user"}
    )
    
    assert response.status_code == 500
    assert "Ingestion failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_too_many_files(client):
    """Test bouncing uploads containing more than 5 files."""
    files = [("files", (f"file{i}.pdf", b"data", "application/pdf")) for i in range(6)]
    response = client.post("/upload/", files=files, data={"user_id": "u"})
    assert response.status_code == 400
    assert "Maximum of 5 files" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_file_too_large(client):
    """Test bouncing uploads exceeding 10MB bounds."""
    # Create an 11MB payload
    large_content = b"0" * (11 * 1024 * 1024)
    response = client.post(
        "/upload/", 
        files=[("files", ("large.pdf", large_content, "application/pdf"))], 
        data={"user_id": "u"}
    )
    assert response.status_code == 400
    assert "Maximum total size" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_missing_filename(mocker):
    """Test upload router gracefully ignores files lacking a filename header."""
    from src.api.routers.upload import upload_files
    
    mock_file = mocker.MagicMock()
    mock_file.filename = ""
    mock_file.size = 100
    
    result = await upload_files(user_id="u", files=[mock_file])
    assert result["files"] == [""]

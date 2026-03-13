"""HTTP client for the Document Intelligence backend API.

Wraps ``requests`` calls to the FastAPI backend, providing methods
for health checks, file uploads, and streaming chat interactions.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)


class APIClient:
    """Client for communicating with the Document Intelligence REST API.

    Reads the backend base URL from the ``API_BASE_URL`` environment
    variable, defaulting to ``http://127.0.0.1:8000`` for local
    development.

    Attributes:
        base_url: Resolved base URL of the backend API.
    """

    def __init__(self):
        self.base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

    def check_health(self):
        """Ping the backend health endpoint.

        Returns:
            True if the API responds with HTTP 200 within 2 seconds,
            False otherwise.
        """
        try:
            return requests.get(f"{self.base_url}/health", timeout=2).status_code == 200
        except Exception as e:
            logger.error(f"ERROR: Something went wrong when trying to check API health: {e}")
            return False

    def upload_files(self, files, user_id):
        """Upload PDF files to the ingestion endpoint.

        Args:
            files: List of tuples in the format expected by
                   ``requests.post(files=...)``.
            user_id: Identifier of the uploading user.

        Returns:
            The ``requests.Response`` from the upload endpoint.
        """
        payload = {"user_id": user_id}
        return requests.post(f"{self.base_url}/upload", files=files, data=payload)

    def chat_stream(self, query, user_id):
        """Open a streaming connection to the chat endpoint.

        Args:
            query: The user's natural-language question.
            user_id: User/thread identifier for scoped document retrieval.

        Returns:
            A streaming ``requests.Response`` that can be iterated
            for real-time text chunks.
        """
        payload = {"query": query, "user_id": user_id, "thread_id": user_id}
        return requests.post(f"{self.base_url}/chat", json=payload, stream=True)

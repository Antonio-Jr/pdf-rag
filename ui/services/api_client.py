import os
import requests
import logging

logger = logging.getLogger(__name__)


class APIClient:
    def __init__(self):
        self.base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

    def check_health(self):
        try:
            return requests.get(f"{self.base_url}/health", timeout=2).status_code == 200
        except Exception as e:
            logger.error(
                f"ERROR: Something went wrong when trying to check API health: {e}"
            )
            return False

    def upload_files(self, files, user_id):
        payload = {"user_id": user_id}
        return requests.post(f"{self.base_url}/upload", files=files, data=payload)

    def chat_stream(self, query, user_id):
        payload = {"query": query, "user_id": user_id, "thread_id": user_id}
        return requests.post(f"{self.base_url}/chat", json=payload, stream=True)

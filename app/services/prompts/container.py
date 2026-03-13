import logging
from typing import Any

logger = logging.getLogger(__name__)

class PromptContainer:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def format(self, key: str, **kwargs) -> str:
        template = self.data.get(key, {}).get("template", "")
        if not template:
            logger.warning(f"Key {key} not found in YAML.")
            return  ""

        return template.format(**kwargs)
"""Prompt template registry backed by YAML files.

Implements a singleton that scans the ``prompts/`` directory tree on
first access, loading every ``.yaml`` file into a nested dictionary
keyed by ``(category, prompt_name)``.  Prompt data is converted to
LangChain ``ChatPromptTemplate`` instances on retrieval.
"""

from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate

from src.utils.log_wrapper import get_logger

logger = get_logger(__name__)


class PromptRegistry:
    """Singleton registry that loads and serves prompt templates from YAML.

    Folder structure expected under the ``prompts/`` directory::

        prompts/
        ├── nodes/
        │   ├── chatbot.yaml
        │   └── summarizer.yaml
        └── tools/
            ├── discovery.yaml
            └── extraction.yaml

    Each YAML file may contain ``system`` and/or ``human`` keys whose
    values are the prompt text strings.

    Attributes:
        _prompts: Nested dictionary ``{category: {name: data}}``.
    """

    _instance = None
    _prompts = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptRegistry, cls).__new__(cls)
            cls._instance._load_all_prompts()
        return cls._instance

    def _load_all_prompts(self):
        """Scan the prompts directory and load all YAML files into memory."""
        base_path = Path(__file__).parent

        for folder in base_path.iterdir():
            if folder.is_dir() and not folder.name.startswith("__"):
                category = folder.name
                self._prompts[category] = {}

                for yaml_file in folder.glob("*.yaml"):
                    prompt_name = yaml_file.stem
                    with open(yaml_file, "r", encoding="utf-8") as f:
                        self._prompts[category][prompt_name] = yaml.safe_load(f)

        logger.info(f"🏛️ Prompt Registry loaded categories: {list(self._prompts.keys())}")

    def get_prompt(self, category: str, name: str) -> ChatPromptTemplate:
        """Retrieve a prompt template by category and name.

        Args:
            category: Top-level grouping (e.g. ``"nodes"``, ``"tools"``).
            name: Prompt identifier within the category (e.g. ``"chatbot"``).

        Returns:
            A ``ChatPromptTemplate`` built from the YAML ``system`` and
            ``human`` message entries.

        Raises:
            KeyError: If no prompt matches the given category and name.
        """
        data = self._prompts.get(category, {}).get(name, {})

        if not data:
            raise KeyError(f"Prompt '{name}' not found in category '{category}'")

        messages = []
        if "system" in data:
            messages.append(("system", data["system"]))
        if "human" in data:
            messages.append(("human", data["human"]))

        return ChatPromptTemplate.from_messages(messages)


prompt_registry = PromptRegistry()

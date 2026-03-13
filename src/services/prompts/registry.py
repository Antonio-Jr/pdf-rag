import yaml
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from src.utils.log_wrapper import get_logger

logger = get_logger(__name__)


class PromptRegistry:
    _instance = None
    _prompts = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptRegistry, cls).__new__(cls)
            cls._instance._load_all_prompts()
        return cls._instance

    def _load_all_prompts(self):
        """Varre a pasta prompts e carrega todos os YAMLs em um dicionário aninhado."""
        base_path = Path(__file__).parent

        # Itera sobre subpastas (nodes, tools, etc)
        for folder in base_path.iterdir():
            if folder.is_dir() and not folder.name.startswith("__"):
                category = folder.name
                self._prompts[category] = {}

                # Itera sobre arquivos .yaml na pasta
                for yaml_file in folder.glob("*.yaml"):
                    prompt_name = yaml_file.stem  # chatbot, discovery, etc.
                    with open(yaml_file, "r", encoding="utf-8") as f:
                        self._prompts[category][prompt_name] = yaml.safe_load(f)

        logger.info(
            f"🏛️ Prompt Registry loaded categories: {list(self._prompts.keys())}"
        )

    def get_prompt(self, category: str, name: str) -> ChatPromptTemplate:
        """Busca o prompt no dicionário carregado."""
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

"""Tests for the Prompt Registry."""

from unittest.mock import MagicMock

import pytest

from src.services.prompts.registry import PromptRegistry


@pytest.fixture(autouse=True)
def reset_singleton():
    PromptRegistry._instance = None
    PromptRegistry._prompts = {}


def test_registry_singleton_initialization(mocker):
    """Test the registry loads the embedded prompts correctly."""
    registry = PromptRegistry()
    registry._load_all_prompts()

    assert registry._prompts is not None
    assert "nodes" in registry._prompts
    assert "tools" in registry._prompts


def test_registry_get_prompt_cache_hit(mocker):
    """Test get_prompt building correctly format messages."""
    registry = PromptRegistry()
    registry._load_all_prompts()

    prompt = registry.get_prompt("nodes", "summarizer")
    assert prompt is not None
    assert len(prompt.messages) >= 1

    # Assert specific formatting triggers caching mechanisms implicitly
    assert registry._prompts.get("nodes").get("summarizer") is not None


def test_registry_get_prompt_missing(mocker):
    """Test registry throwing error for invalid domains."""
    registry = PromptRegistry()
    registry._load_all_prompts()

    with pytest.raises(KeyError, match="Prompt 'p' not found in category 'invalid'"):
        registry.get_prompt("invalid", "p")

    with pytest.raises(KeyError, match="Prompt 'invalid' not found in category 'nodes'"):
        registry.get_prompt("nodes", "invalid")

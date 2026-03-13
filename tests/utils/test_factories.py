"""Tests for model factories (Embeddings and Chat Models)."""

import pytest


def test_get_embeddings_google(mocker):
    """Test fetching a Google GenAI embedding instance."""
    mock_settings = mocker.patch("src.utils.embedding_factory.settings")
    mock_settings.LLM_EMBEDDING_PROVIDER = "google_genai"
    mock_settings.LLM_EMBEDDING_MODEL = "models/embedding-001"
    mock_settings.LLM_API_KEY = "fake_key"
    
    mock_google = mocker.patch("src.utils.embedding_factory.GoogleGenerativeAIEmbeddings")
    
    from src.utils.embedding_factory import get_embeddings
    get_embeddings()
    
    mock_google.assert_called_once_with(model="models/embedding-001", api_key="fake_key")


def test_get_embeddings_ollama(mocker):
    """Test fetching an Ollama embedding instance."""
    mock_settings = mocker.patch("src.utils.embedding_factory.settings")
    mock_settings.LLM_EMBEDDING_PROVIDER = "ollama"
    mock_settings.LLM_EMBEDDING_MODEL = "nomic-embed-text"
    mock_settings.LLM_BASE_URL = "http://localhost:11434"
    
    mock_ollama = mocker.patch("src.utils.embedding_factory.OllamaEmbeddings")
    
    from src.utils.embedding_factory import get_embeddings
    get_embeddings()
    
    mock_ollama.assert_called_once_with(model="nomic-embed-text", base_url="http://localhost:11434")


def test_get_embeddings_openai(mocker):
    """Test fetching an OpenAI embedding instance."""
    mock_settings = mocker.patch("src.utils.embedding_factory.settings")
    mock_settings.LLM_EMBEDDING_PROVIDER = "openai"
    mock_settings.LLM_EMBEDDING_MODEL = "text-embedding-3-small"
    
    mock_openai = mocker.patch("src.utils.embedding_factory.OpenAIEmbeddings")
    
    from src.utils.embedding_factory import get_embeddings
    get_embeddings()
    
    mock_openai.assert_called_once_with(model="text-embedding-3-small")


def test_get_embeddings_unsupported(mocker):
    """Test failure when provider is not matched."""
    mock_settings = mocker.patch("src.utils.embedding_factory.settings")
    mock_settings.LLM_EMBEDDING_PROVIDER = "unknown"
    
    from src.utils.embedding_factory import get_embeddings
    
    with pytest.raises(ValueError, match="Embedding provider 'unknown' not supported"):
        get_embeddings()


def test_get_model_basic(mocker):
    """Test get_model default instantiation without schemas."""
    mocker.patch("src.utils.llm_factory.settings")
    mock_init = mocker.patch("src.utils.llm_factory.init_chat_model")
    mock_model_instance = mock_init.return_value
    
    from src.utils.llm_factory import get_model
    
    # Fallthrough kwargs and missing schema
    result = get_model(model_provider="openai", model_name="gpt-4o", temperature=0.7)
    
    assert result == mock_model_instance
    mock_init.assert_called_once()
    assert mock_init.call_args[1]["model_provider"] == "openai"


def test_get_model_structured(mocker):
    """Test get_model returning a structured binding."""
    mocker.patch("src.utils.llm_factory.settings")
    mock_init = mocker.patch("src.utils.llm_factory.init_chat_model")
    mock_model_instance = mock_init.return_value
    mock_model_instance.with_structured_output.return_value = "StructuredModel"
    
    from src.utils.llm_factory import get_model
    
    class FakeSchema:
        pass
        
    result = get_model(structured_schema=FakeSchema)
    
    assert result == "StructuredModel"
    mock_model_instance.with_structured_output.assert_called_once_with(FakeSchema)

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from src.utils.log_wrapper import log_execution
from src.core.settings import settings


@log_execution
def get_embeddings():
    provider = settings.LLM_EMBEDDING_PROVIDER.lower()

    if provider == "google_genai":
        return GoogleGenerativeAIEmbeddings(
            model=settings.LLM_EMBEDDING_MODEL or "text-embedding-004",
            api_key=settings.LLM_API_KEY,
        )

    elif provider == "ollama":
        return OllamaEmbeddings(
            model=settings.LLM_EMBEDDING_MODEL,
            base_url=settings.LLM_BASE_URL,
        )
    elif provider == "openai":
        return OpenAIEmbeddings(model=settings.LLM_EMBEDDING_MODEL)

    raise ValueError(f"Embedding provider '{provider}' not supported")

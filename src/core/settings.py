"""Application settings loaded from environment variables.

Uses Pydantic Settings to provide typed, validated configuration
for the LLM provider, embedding strategy, database connection,
and document chunking parameters.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the application.

    All values are loaded from a ``.env`` file or environment variables.
    Fields without defaults are required; the application will fail to
    start if they are missing.

    Attributes:
        LLM_API_KEY: API key used to authenticate with the LLM provider.
        LLM_BASE_URL: Base URL of the LLM provider endpoint.
        LLM_PROVIDER: Identifier of the LLM provider (e.g. ``google_genai``, ``ollama``).
        LLM_MODEL_NAME: Name of the chat model to use.
        LLM_TEMPERATURE: Sampling temperature for the chat model.
        LLM_EMBEDDING_PROVIDER: Provider for the embedding model.
        LLM_EMBEDDING_MODEL: Name of the embedding model.
        DATABASE_URL: PostgreSQL connection string (async-compatible).
        DATABASE_COLLECTION_NAME: Name of the vector store collection.
        CHUNK_SIZE: Maximum number of characters per document chunk.
        CHUNK_OVERLAP: Number of overlapping characters between chunks.
        VECTOR_COLLECTION: Default PGVector collection name.
    """

    LLM_API_KEY: str = Field(...)
    LLM_BASE_URL: str = Field(...)
    LLM_PROVIDER: str = Field(...)
    LLM_MODEL_NAME: str = Field(...)
    LLM_TEMPERATURE: float = Field(...)

    LLM_EMBEDDING_PROVIDER: str = Field(...)
    LLM_EMBEDDING_MODEL: str = Field(...)

    DATABASE_URL: str = Field(...)
    DATABASE_COLLECTION_NAME: str = Field(...)
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    VECTOR_COLLECTION: str = "universal_doc_store"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()

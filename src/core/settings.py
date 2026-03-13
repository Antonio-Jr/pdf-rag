from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM Configuration
    LLM_API_KEY: str = Field(...)
    LLM_BASE_URL: str = Field(...)
    LLM_PROVIDER: str = Field(...)
    LLM_MODEL_NAME: str = Field(...)
    LLM_TEMPERATURE: float = Field(...)

    # Embedding Strategy (Agnostic)
    LLM_EMBEDDING_PROVIDER: str = Field(...)
    LLM_EMBEDDING_MODEL: str = Field(...)

    # Ingestion Settings
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

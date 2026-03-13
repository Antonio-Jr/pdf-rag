"""Vector store retriever factory.

Builds a LangChain retriever backed by PGVector, scoped to a specific
user and optional metadata filters, for similarity-based document search.
"""

from langchain_postgres.vectorstores import PGVector

from src.utils.log_wrapper import log_execution
from src.core.settings import settings
from src.utils.embedding_factory import get_embeddings


@log_execution
def get_retriever(user_id: str, k: int = 5, additional_filters: dict | None = None):
    """Create a PGVector retriever filtered by user ownership.

    Args:
        user_id: Identifier of the user whose documents should be searched.
        k: Maximum number of similar documents to return. Defaults to ``5``.
        additional_filters: Extra JSONB metadata filters merged into the
                            base ``user_id`` filter.

    Returns:
        A LangChain ``VectorStoreRetriever`` instance ready for invocation.
    """
    embeddings = get_embeddings()
    vector_store = PGVector(
        connection=settings.DATABASE_URL,
        collection_name=settings.DATABASE_COLLECTION_NAME,
        embeddings=embeddings,
        create_extension=False,
        use_jsonb=True,
        async_mode=True,
    )

    compose_filter = {"user_id": user_id}

    if additional_filters:
        compose_filter.update(additional_filters)

    search_kwargs = {"k": k, "filter": compose_filter}

    return vector_store.as_retriever(search_kwargs=search_kwargs)

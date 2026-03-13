"""Tests for the Vector Store Retriever factory."""


def test_get_retriever(mocker):
    """Test get_retriever correctly configures the LangChain VectorStoreRetriever."""
    mocker.patch("src.infrastructure.retrievers.get_embeddings")
    mock_pgvector = mocker.patch("src.infrastructure.retrievers.PGVector")
    mock_vector_instance = mock_pgvector.return_value
    mock_vector_instance.as_retriever.return_value = "RetrieverInstance"

    from src.infrastructure.retrievers import get_retriever

    # Test with standard configuration + additional filters
    retriever = get_retriever(
        user_id="test_user_777", k=10, additional_filters={"project": "secret"}
    )

    assert retriever == "RetrieverInstance"

    # Ensure as_retriever was called with the correct merged kwargs
    mock_vector_instance.as_retriever.assert_called_once_with(
        search_kwargs={"k": 10, "filter": {"user_id": "test_user_777", "project": "secret"}}
    )


def test_get_retriever_no_filters(mocker):
    """Test get_retriever without additional filters."""
    mocker.patch("src.infrastructure.retrievers.get_embeddings")
    mock_pgvector = mocker.patch("src.infrastructure.retrievers.PGVector")
    mock_vector_instance = mock_pgvector.return_value

    from src.infrastructure.retrievers import get_retriever

    get_retriever(user_id="user_888")

    mock_vector_instance.as_retriever.assert_called_once_with(
        search_kwargs={
            "k": 5,  # default
            "filter": {"user_id": "user_888"},
        }
    )

"""PDF document ingestion service.

Loads PDF files, splits them into overlapping text chunks, attaches
user-scoped metadata, and stores the resulting embeddings in PGVector.
"""

from typing import Any, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.settings import settings
from src.utils.embedding_factory import get_embeddings
from src.utils.log_wrapper import get_logger, log_execution


class IngestionService:
    """Orchestrates the end-to-end PDF ingestion pipeline.

    On instantiation, resolves the embedding model and database
    connection parameters.  The :meth:`ingest_data` method handles
    loading, chunking, metadata enrichment, and vector storage.

    Attributes:
        embeddings: The embedding model instance used for vectorization.
        connection_string: PostgreSQL connection URL.
        collection_name: Target PGVector collection name.
    """

    def __init__(self):
        self.embeddings = get_embeddings()
        self.connection_string = settings.DATABASE_URL
        self.collection_name = settings.DATABASE_COLLECTION_NAME

    @log_execution
    async def ingest_data(
        self, file_path: str, user_id: str, extra_metadata: Dict[str, Any] | None = None
    ) -> int:
        """Ingest a single PDF file into the vector store.

        Args:
            file_path: Absolute path to the PDF file on disk.
            user_id: Owner identifier attached to every chunk's metadata.
            extra_metadata: Optional key-value pairs merged into each
                            chunk's metadata (e.g. filename, status).

        Returns:
            The number of text chunks successfully stored.

        Raises:
            Exception: Propagates any pipeline error after logging.
        """
        logger = get_logger(__name__)
        try:
            loader = PyPDFLoader(file_path)
            raw_documents = loader.load()

            logger.info(f"Loaded {len(raw_documents)} pages from {file_path}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                add_start_index=True,
            )

            chunks = text_splitter.split_documents(raw_documents)

            for chunk in chunks:
                chunk.metadata["user_id"] = user_id
                if extra_metadata:
                    chunk.metadata.update(extra_metadata)

            logger.info(f"Connection String: {self.connection_string}")

            await PGVector.afrom_documents(
                embedding=self.embeddings,
                documents=chunks,
                collection_name=self.collection_name,
                connection=self.connection_string,
                pre_delete_collection=False,
                use_jsonb=True,
                create_extension=False,
            )

            logger.info(f"Successfully ingested {len(chunks)} chunks into {self.collection_name}")
            return len(chunks)
        except Exception as e:
            logger.error(f"Critical error during ingestion pipeline: {str(e)}")
            raise

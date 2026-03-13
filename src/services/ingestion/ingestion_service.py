from typing import Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.log_wrapper import get_logger, log_execution
from src.core.settings import settings
from src.utils.embedding_factory import get_embeddings


class IngestionService:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.connection_string = settings.DATABASE_URL
        self.collection_name = settings.DATABASE_COLLECTION_NAME

    @log_execution
    async def ingest_data(
        self, file_path: str, user_id: str, extra_metadata: Dict[str, Any] | None = None
    ) -> int:
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

            logger.info(
                f"Successfully ingested {len(chunks)} chunks into {self.collection_name}"
            )
            return len(chunks)
        except Exception as e:
            logger.error(f"Critical error during ingestion pipeline: {str(e)}")
            raise

from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Form

import shutil
import os

from src.utils.log_wrapper import log_execution
from src.services.ingestion.ingestion_service import IngestionService

router = APIRouter(prefix="/upload", tags=["upload"])
service = IngestionService()

TEMP_STORAGE = Path("storage/uploads")
TEMP_STORAGE.mkdir(parents=True, exist_ok=True)


@router.post("/")
@log_execution
async def upload_files(user_id: str = Form(...), files: list[UploadFile] = File(...)):
    try:
        if len(files) > 5:
            raise HTTPException(400, "Max 5 arquivos")

        total_size = sum(f.size for f in files if f.size)
        if total_size > 10 * 1024 * 1024:
            raise HTTPException(400, "Max 10MB total")

        saved_paths: list[str] = []
        chunks_created = None
        for file in files:
            if not file.filename:
                continue

            if not file.filename.endswith(".pdf"):
                raise HTTPException(400, "Só PDFs")

            file_path = Path(f"{TEMP_STORAGE}/{file.filename}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_paths.append(str(file_path))

            chunks_created = await service.ingest_data(
                file_path=str(file_path),
                user_id=user_id,
                extra_metadata={"filename": file.filename, "status": "processed"},
            )

        return {
            "message": "Ingestion successful",
            "chunks": chunks_created,
            "files": [f.filename for f in files],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    finally:
        for file in files:
            file_path = Path(f"{TEMP_STORAGE}/{file.filename}")
            if file_path.exists():
                os.remove(file_path)

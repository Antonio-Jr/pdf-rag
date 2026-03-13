from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.schemas.query import ChatQuery
from src.utils.log_wrapper import log_execution
from src.services.agents.agent_service import analyze_query

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/")
@log_execution
async def chat(payload: ChatQuery):
    try:
        return StreamingResponse(
            analyze_query(
                user_input=payload.query,
                thread_id=payload.thread_id,
                user_id=payload.user_id,
            ),
            media_type="text/plain",
        )

    except Exception:
        return HTTPException(500, detail="Internal Agent Error")

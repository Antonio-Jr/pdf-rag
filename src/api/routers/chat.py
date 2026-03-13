"""Chat API endpoint for conversational document analysis.

Exposes a streaming POST route that forwards user queries to the
LangGraph agent and returns the model's response as a real-time
text stream.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas.query import ChatQuery
from src.services.agents.agent_service import analyze_query
from src.utils.log_wrapper import log_execution

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/")
@log_execution
async def chat(payload: ChatQuery):
    """Stream an AI-generated response for a user query.

    Invokes the document analysis agent with the given query,
    thread, and user context, then streams back the LLM output
    token-by-token as ``text/plain``.

    Args:
        payload: The incoming chat request containing the query,
                 thread identifier, and user identifier.

    Returns:
        A ``StreamingResponse`` with the model's answer.

    Raises:
        HTTPException: If an unexpected error occurs during processing.
    """
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

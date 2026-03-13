from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from src.services.prompts.registry import prompt_registry
from src.utils.log_wrapper import log_execution
from src.infrastructure.retrievers import get_retriever
from src.utils.llm_factory import get_model


@tool
@log_execution
async def extraction_data_tool(
    discovery_summary: str,
    tracking_points: list[str],
    user_id: Annotated[str, InjectedState("user_id")],
) -> str:
    """
    Extracts precise and structured data based on specific tracking points.
    Use this tool AFTER discovery to obtain exact values, dates, or clauses.
    """
    if not user_id:
        raise ValueError("user_id was not injected!")

    retriever = get_retriever(user_id=user_id)
    docs = await retriever.ainvoke(", ".join(tracking_points))

    if not docs:
        return "ERROR: No content found for this user in the document database."

    content = "\n\n".join([doc.page_content for doc in docs])

    prompt = prompt_registry.get_prompt("tools", "discovery")
    model = get_model()
    chain = prompt | model

    result = await chain.ainvoke(
        {
            "content": content,
            "discovery_summary": discovery_summary,
            "tracking_points": tracking_points,
        }
    )

    return result.content

import json
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from src.services.prompts.registry import prompt_registry
from src.utils.log_wrapper import get_logger, log_execution
from src.shared.schemas.universal_discovery import UniversalDiscovery
from src.infrastructure.retrievers import get_retriever
from src.utils.llm_factory import get_model


@tool
@log_execution
async def discovery_document_tool(
    query: str, user_id: Annotated[str, InjectedState("user_id")]
) -> str:
    """
    REQUIRED for initial contact with new documents.
    Accesses the database of resumes and PDFs uploaded by the user.
    Use whenever the user mentions that they 'uploaded a file', 'made an upload'
    or wants a 'summary/context' of the document.
    """
    logger = get_logger(__name__)
    if not user_id:
        raise ValueError("user_id was not injected!")

    retriever = get_retriever(user_id=user_id)
    docs = await retriever.ainvoke(query or "General document overview")

    if not docs:
        logger.warning(f"⚠️ No docs found for the user_id: {user_id}")
        return "Sorry, I did not found any information in the docs you have sent previously."

    content = "\n\n".join([doc.page_content for doc in docs])

    if not content.strip():
        return "I found the docs, but they are not seem to be a text legible."

    model = get_model(structured_schema=UniversalDiscovery)
    prompt = prompt_registry.get_prompt("tools", "discovery")
    chain = prompt | model

    result = await chain.ainvoke({"content": content})

    return json.dumps(result.model_dump(), ensure_ascii=False)

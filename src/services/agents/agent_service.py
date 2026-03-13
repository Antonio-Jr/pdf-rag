"""Agent service for streaming document analysis.

Compiles the LangGraph document graph with a checkpointer and
streams LLM events back to the caller, yielding text tokens as
they are generated.
"""

from langchain_core.runnables import RunnableConfig

from src.infrastructure.database import get_checkpointer
from src.services.graphs.document_graph import generate_graph
from src.utils.log_wrapper import get_logger, log_execution


@log_execution
async def analyze_query(user_input: str, thread_id: str, user_id: str):
    """Stream an AI response for a user query against uploaded documents.

    Compiles the document analysis graph with the active checkpointer,
    then iterates over streamed LLM events to yield text content
    incrementally.

    Args:
        user_input: The user's natural-language question.
        thread_id: Conversation thread identifier for memory continuity.
        user_id: Identifier used to scope document retrieval.

    Yields:
        Text fragments (strings) as they are produced by the chat model.
    """
    logger = get_logger(__name__)
    checkpointer = get_checkpointer()
    graph = generate_graph().compile(checkpointer=checkpointer)

    config = RunnableConfig({"configurable": {"thread_id": thread_id, "user_id": user_id}})

    initial_state = {
        "messages": [("user", user_input)],
        "query": user_input,
        "user_id": thread_id,
    }

    async for event in graph.astream_events(initial_state, config=config, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]  # type: ignore

            content = chunk.content
            if isinstance(content, str):
                yield content

            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        yield block.get("text", "")
                    elif isinstance(block, str):
                        yield block

        elif kind == "on_tool_start":
            logger.info(f"  \n> 🔍 *Searching documents for {user_id}...* \n")
            logger.info(f"\n[⚙️ Executing: {event['name']}]\n")

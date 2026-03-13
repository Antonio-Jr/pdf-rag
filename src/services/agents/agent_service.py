from langchain_core.runnables import RunnableConfig

from src.infrastructure.database import get_checkpointer
from src.utils.log_wrapper import log_execution, get_logger
from src.services.graphs.document_graph import generate_graph


@log_execution
async def analyze_query(user_input: str, thread_id: str, user_id: str):
    logger = get_logger(__name__)
    checkpointer = get_checkpointer()
    graph = generate_graph().compile(checkpointer=checkpointer)

    config = RunnableConfig(
        {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    )

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
            logger.info(f"  \n> 🔍 *Pesquisando nos documentos de {user_id}...* \n")
            logger.info(f"\n[⚙️ Executando: {event['name']}]\n")

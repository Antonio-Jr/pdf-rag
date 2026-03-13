from src.services.prompts.registry import prompt_registry
from src.utils.log_wrapper import log_execution
from src.services.states.graph_state import GraphState
from src.utils.llm_factory import get_model
from langchain_core.prompts import MessagesPlaceholder

MINIMUM_MESSAGES_TO_SUMMARIZE = 7

@log_execution
async def summarizer_node(state: GraphState) -> GraphState:
    messages = state.messages or []

    if len(messages) < MINIMUM_MESSAGES_TO_SUMMARIZE:
        return state

    model = get_model()
    history_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in messages[-5:]])

    base_prompt = prompt_registry.get_prompt("nodes", "summarizer")
    prompt = base_prompt + MessagesPlaceholder(variable_name="messages")

    chain = prompt | model
    result = await chain.ainvoke(
        {
            "messages": state.messages,
            "existing_summary": state.summary,
            "new_messages": history_str,
        }
    )

    state.summary = result.content
    return state

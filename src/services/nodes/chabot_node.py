
from src.services.prompts.registry import prompt_registry
from src.utils.log_wrapper import log_execution, get_logger
from src.services.states.graph_state import GraphState
from src.utils.llm_factory import get_model
from src.services.tools import tools


@log_execution
async def chatbot_node(state: GraphState) -> GraphState:
    logger = get_logger(__name__)
    model = get_model().bind_tools(tools)

    prompt_template = prompt_registry.get_prompt("nodes", "chatbot")
    base_messages = prompt_template.format_messages()
    messages = base_messages + state.messages

    response = await model.ainvoke(messages)
    state.messages = [response]

    if response.tool_calls:
        for tool_call in response.tool_calls:
            logger.info(
                f"🛠️ [TOOL CALL]: {tool_call['name']} com args: {tool_call['args']}"
            )

    return state

"""Chatbot graph node for conversational AI responses.

Invokes the LLM with the system prompt and conversation history,
binding available tools so the model can decide whether to call
a tool or reply directly.
"""

from src.services.prompts.registry import prompt_registry
from src.services.states.graph_state import GraphState
from src.services.tools import tools
from src.utils.llm_factory import get_model
from src.utils.log_wrapper import get_logger, log_execution


@log_execution
async def chatbot_node(state: GraphState) -> GraphState:
    """Process the current conversation state and generate a model response.

    Loads the chatbot system prompt, appends existing messages, invokes
    the LLM with tool bindings, and updates the state with the model's
    reply.  Any tool calls requested by the model are logged for
    observability.

    Args:
        state: The current graph state containing messages and context.

    Returns:
        The updated ``GraphState`` with the model's response appended.
    """
    logger = get_logger(__name__)
    model = get_model().bind_tools(tools)

    prompt_template = prompt_registry.get_prompt("nodes", "chatbot")
    base_messages = prompt_template.format_messages()
    messages = base_messages + state.messages

    response = await model.ainvoke(messages)
    state.messages = [response]

    if response.tool_calls:
        for tool_call in response.tool_calls:
            logger.info(f"🛠️ [TOOL CALL]: {tool_call['name']} with args: {tool_call['args']}")

    return state

"""LangGraph state graph definition for document analysis.

Constructs the workflow that routes user messages through a chatbot
node, optional tool calls (discovery / extraction), and a
conversation summarizer before ending.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.services.nodes.chabot_node import chatbot_node
from src.services.nodes.summarizer import summarizer_node
from src.services.states.graph_state import GraphState
from src.services.tools import tools


def generate_graph() -> StateGraph:
    """Build and return the document analysis state graph.

    The graph follows this flow::

        START → chatbot ──(tools_condition)──→ tools → chatbot
                    │                                      │
                    └──────(no tool calls)──→ summarizer → END

    Returns:
        A ``StateGraph`` instance (not yet compiled).
    """
    workflow = StateGraph(GraphState)

    tool_node = ToolNode(tools)

    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("summarizer", summarizer_node)

    workflow.add_edge(START, "chatbot")

    workflow.add_conditional_edges(
        "chatbot", tools_condition, {"tools": "tools", "__end__": "summarizer"}
    )

    workflow.add_edge("tools", "chatbot")
    workflow.add_edge("summarizer", END)

    return workflow

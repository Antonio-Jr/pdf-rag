"""Graph state schema for the document analysis workflow.

Defines the Pydantic model that flows through every node of the
LangGraph state graph, carrying messages, user context, discovery
results, extracted data, and conversation summaries.
"""

from typing import List, Annotated, Dict, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel

from src.shared.schemas.universal_discovery import UniversalDiscovery


class GraphState(BaseModel):
    """Shared state passed between all nodes in the document analysis graph.

    Attributes:
        messages: Conversation message history, accumulated via
                  LangGraph's ``add_messages`` reducer.
        query: The original user query for the current turn.
        user_id: Identifier used to scope document retrieval.
        discovery: Structured summary produced by the discovery tool.
        extracted_data: Key-value pairs extracted from documents.
        raw_content: Unprocessed text content from retrieved documents.
        response: The final formatted response text.
        summary: Condensed conversation summary for context management.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    user_id: str
    discovery: UniversalDiscovery = UniversalDiscovery(
        data_points_to_track=[], doc_purpose="", key_entities=[], tone=""
    )
    extracted_data: Dict[str, Any] = {}
    raw_content: str = ""
    response: str = ""
    summary: str = ""

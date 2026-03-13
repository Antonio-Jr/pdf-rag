from typing import List, Annotated, Dict, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel

from src.shared.schemas.universal_discovery import UniversalDiscovery


class GraphState(BaseModel):
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

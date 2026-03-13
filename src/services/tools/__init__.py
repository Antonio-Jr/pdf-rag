"""Tool registry for the document analysis agent.

Aggregates all LangChain tools available to the chatbot node so they
can be bound to the LLM and used by the LangGraph ``ToolNode``.
"""

from src.services.tools.discovery import discovery_document_tool
from src.services.tools.extraction import extraction_data_tool

tools = [discovery_document_tool, extraction_data_tool]

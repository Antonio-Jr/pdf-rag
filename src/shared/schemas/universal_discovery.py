"""Universal document discovery schema.

Defines the structured output model returned by the discovery tool,
capturing the high-level characteristics of an analyzed document.
"""

from typing import List

from pydantic import BaseModel, Field


class UniversalDiscovery(BaseModel):
    """Structured summary of a document's key characteristics.

    Used as the LLM's structured output schema during discovery to
    capture what kind of document was uploaded and what data can be
    extracted from it.

    Attributes:
        doc_purpose: The primary intent or subject of the document.
        key_entities: Main subjects, companies, or people mentioned.
        data_points_to_track: Relevant fields identified for extraction.
        tone: Overall tone classification (e.g. Technical, Formal, Legal).
    """

    doc_purpose: str = Field(..., description="The primary intent of the document")
    key_entities: List[str] = Field(
        ..., description="Main subjects, companies, or people mentioned"
    )
    data_points_to_track: List[str] = Field(
        ..., description="Relevant fields identified for extraction"
    )
    tone: str = Field(..., description="Overall tone: Technical, Formal, Legal, etc.")

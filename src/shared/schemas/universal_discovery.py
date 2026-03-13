from typing import List

from pydantic import BaseModel, Field


class UniversalDiscovery(BaseModel):
    doc_purpose: str = Field(..., description="The primary intent of the document")
    key_entities: List[str] = Field(
        ..., description="Main subjects, companies, or people mentioned"
    )
    data_points_to_track: List[str] = Field(
        ..., description="Relevant fields identified for extraction"
    )
    tone: str = Field(..., description="Overall tone: Technical, Formal, Legal, etc.")

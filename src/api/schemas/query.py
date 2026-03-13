"""Request schemas for the chat API.

Defines Pydantic models that validate and document the JSON body
expected by chat-related endpoints.
"""

from pydantic import BaseModel


class ChatQuery(BaseModel):
    """Schema for an incoming chat request.

    Attributes:
        query: The user's natural-language question or instruction.
        thread_id: Unique conversation thread identifier used for
                   checkpointed memory continuity.
        user_id: Identifier of the user making the request, used
                 to scope document retrieval.
    """

    query: str
    thread_id: str
    user_id: str

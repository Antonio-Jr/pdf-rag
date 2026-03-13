from pydantic import BaseModel


class ChatQuery(BaseModel):
    query: str
    thread_id: str
    user_id: str

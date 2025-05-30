from typing import Optional, List
from openai import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[ChatMessage]]

class ChatResponse(BaseModel):
    response: str
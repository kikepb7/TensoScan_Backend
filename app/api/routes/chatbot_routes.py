from typing import List
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Depends
from app.domain.models.chatbot_model import ChatRequest, ChatResponse
import logging, os, httpx

from app.infrastructure.services.get_user_service import get_current_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["CHATBOT"])

try:
    logger.info("Initializing Chatbot...")
    logger.info("Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize the Chatbot: {str(e)}")
    raise

load_dotenv()
OPENAI_MODEL = "gpt-3.5-turbo"

async def call_openai(messages: List[dict]) -> str:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url="https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "temperature": 0.7
                }
            )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    except httpx.HTTPStatusError as e:
        logger.error(f"OpenAI API error: {e.response.text}")
        raise HTTPException(status_code=500, detail="Error from OpenAI API")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/chatbot", response_model=ChatResponse)
async def respond_to_prompt(request: ChatRequest, user=Depends(get_current_user)):
    try:
        user_id = str(user["_id"])
        messages = [message.model_dump() for message in request.history] if request.history else []
        messages.append(
            {
                "role": "user",
                "content": request.prompt
            }
        )

        reply = await call_openai(messages)
        logger.info("Successfully got response from OpenAI")
        return ChatResponse(response=reply)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chatbot endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your request")

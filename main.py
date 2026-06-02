from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

app = FastAPI(title="Aether AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": "You are Aether, a helpful, honest, and practical AI agent. The user is from Pakistan and wants to start a scrap metal melting business. Be direct, useful, and give real advice."}
        ] + request.history + [{"role": "user", "content": request.message}]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return {"response": data["choices"][0]["message"]["content"]}

    except Exception as e:
        return {"response": f"Error: {str(e)}"}

@app.get("/")
async def root():
    return {"message": "Aether Backend is running successfully!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

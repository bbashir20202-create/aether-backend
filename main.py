from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
from typing import List

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Aether AI Agent")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

if not os.getenv("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY is not set in .env file!")

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": "You are Aether, a highly intelligent, honest, and practical AI agent. The user is from Pakistan and is planning to start a scrap metal melting business (mainly aluminum). Be direct, useful, and give real business advice. Also support them emotionally if needed."}
        ] + request.history + [{"role": "user", "content": request.message}]

        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=messages,
            temperature: 0.7,
            max_tokens: 1200
        )

        response = completion.choices[0].message.content

        return {"response": response}

    except Exception as e:
        return {"response": f"Error: {str(e)}"}

@app.get("/")
async def root():
    return {"message": "Aether Backend is running. Send POST requests to /api/chat"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

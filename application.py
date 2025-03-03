from importlib import metadata
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from pathlib import Path

# Get absolute path to the .env file
env_path = Path('.env')
load_dotenv(dotenv_path=env_path)

# Debug print
print(f"PINECONE_API_KEY loaded: {'Yes' if os.getenv('PINECONE_API_KEY') else 'No'}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

# Assuming retrieval_chain is already defined in pinecone_script.py
from pinecone_script import retrieval_chain

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = retrieval_chain.invoke({"input": request.query})
        sources = {doc.metadata["source"] for doc in response['context']}

        return ChatResponse(
            answer=response['answer'], 
            sources=list(sources)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


application = app
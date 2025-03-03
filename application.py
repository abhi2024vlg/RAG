from importlib import metadata
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
from fastapi.middleware.cors import CORSMiddleware

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
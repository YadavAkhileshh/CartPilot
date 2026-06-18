import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import tempfile
import uuid
import json

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional

from shopping_agent import agent
import setup_db

if not os.path.exists("store.db") or not os.path.exists("faiss_index"):
    print("Initializing database and FAISS index for the first time...")
    setup_db.create_database()
    setup_db.create_vector_db()

app = FastAPI(title="CartPilot API")

# Serve static files from the "public" directory
app.mount("/static", StaticFiles(directory="public"), name="static")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: str

sessions = {}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("public/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if request.session_id not in sessions:
        sessions[request.session_id] = []
        
    history = sessions[request.session_id]
    
    for msg in request.messages:
        history.append({"role": msg.role, "content": msg.content})

    result = agent.invoke({"messages": history})
    response_content = result["messages"][-1].content.replace("`", "")
    
    history.append({"role": "assistant", "content": response_content})
    
    return {"response": response_content}

@app.post("/api/upload_image")
async def upload_image(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    if session_id not in sessions:
        sessions[session_id] = []
        
    history = sessions[session_id]

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        image_path = tmp.name

    prompt = f"I uploaded a product image. Please analyze it and find similar products in the store. Image path: {image_path}"
    history.append({"role": "user", "content": prompt})
    
    result = agent.invoke({"messages": history})
    response_content = result["messages"][-1].content.replace("`", "")
    
    history.append({"role": "assistant", "content": response_content})
    
    return {"response": response_content, "filename": file.filename}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

import os
from dotenv import load_dotenv
load_dotenv()

import tempfile
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List

try:
    from upstash_redis.asyncio import Redis
    redis = Redis.from_env()
except Exception as e:
    print(f"Warning: Could not initialize Upstash Redis. Falling back to dict. ({e})")
    redis = None

from shopping_agent import agent

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="CartPilot API")
app.mount("/static", StaticFiles(directory="public"), name="static")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: str

local_sessions = {}

async def get_history(session_id: str):
    if redis:
        data = await redis.get(f"session:{session_id}")
        if data:
            return json.loads(data) if isinstance(data, str) else data
        return []
    return local_sessions.get(session_id, [])

async def save_history(session_id: str, history: list):
    if redis:
        await redis.set(f"session:{session_id}", json.dumps(history))
    else:
        local_sessions[session_id] = history

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("public/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat")
async def chat(request: ChatRequest):
    history = await get_history(request.session_id)

    for msg in request.messages:
        history.append({"role": msg.role, "content": msg.content})

    async def event_generator():
        full_response = ""
        try:
            result = await agent.ainvoke({"messages": history})
            full_response = result["messages"][-1].content.replace("`", "")

            # Stream word-by-word for a smooth natural typing effect
            words = full_response.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                await asyncio.sleep(0.03)

        except Exception as e:
            print(f"Chat error: {e}")
            full_response = "Sorry, I encountered an error. Please try again."
            yield f"data: {json.dumps({'chunk': full_response})}\n\n"

        history.append({"role": "assistant", "content": full_response})
        await save_history(request.session_id, history)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/upload_image")
async def upload_image(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    history = await get_history(session_id)

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        image_path = tmp.name

    prompt = f"I uploaded a product image. Please analyze it and find similar products in the store. Image path: {image_path}"
    history.append({"role": "user", "content": prompt})

    result = await agent.ainvoke({"messages": history})
    response_content = result["messages"][-1].content.replace("`", "")

    history.append({"role": "assistant", "content": response_content})
    await save_history(session_id, history)

    return {"response": response_content, "filename": file.filename}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

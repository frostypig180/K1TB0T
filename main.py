import os
from werkzeug.utils import secure_filename
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, status, Depends
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from openai import OpenAI
import shutil
import asyncio
from typing import List, Dict
import threading
import queue

# ===============================================================================================
# Kit Bot: LLM Chatbot using vLLM and Mistral Instruct
# Authors: Eli Gruhlke, Ian Walch, Will Dani, Beaumont Ujlaky, Caleb Schweigert, and Erik Greiner
# ===============================================================================================

USER = "admin"
PASS = "wce2026"

# Connect to local vLLM server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
model = "mistralai/Mistral-7B-Instruct-v0.3"
# Path to instructions directory
instructions_path = "/home/k1tbot/Documents/k1tbot/instructions"
# Path to bot rules file
rules_path = "/home/k1tbot/Documents/k1tbot/bot_rules/BotPrompt.txt"
# Initialize chat histories and locks for multiple users
chat_histories: dict[str, list[dict[str, str]]] = {}
chat_locks: dict[str, asyncio.Lock] = {}
# Initialize FastAPI app
app = FastAPI()
# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://k1tb0t.com", "https://www.k1tb0t.com"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Basic auth for admin endpoints
security = HTTPBasic()
def check_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USER)
    correct_password = secrets.compare_digest(credentials.password, PASS)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
# Uploads configuration
UPLOAD_FOLDER = instructions_path
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.csv', '.txt', '.json', '.mp4'}
# Mount uploads as static files so they can be served at /instructions/{filename}
app.mount('/instructions', StaticFiles(directory=UPLOAD_FOLDER), name='instructions')
# Load bot rules
def load_bot_rules() -> str:
    try:
        return Path(rules_path).read_text(encoding="utf-8").strip()
    except Exception as e:
        return f"Could not read bot rules! ({e})"
# Read all files in instructions directory
def load_instructions() -> str:
    parts = []
    try:
        # read all .txt files from the SAME folder used by uploads/list/delete
        for file in sorted(Path(UPLOAD_FOLDER).glob("*.txt")):
            parts.append(file.read_text(encoding="utf-8").strip())
        combined = "\n\n".join([p for p in parts if p])
        return combined if combined else "No instruction files found. Default behaviour: act like a helpful assistant."
    except Exception as e:
        return f"Could not read instructions! Default behaviour: act like a helpful assistant. ({e})"
# Get or create chat history and lock for a session
def get_history_and_lock(sid: str):
    if sid not in chat_histories:
        chat_histories[sid] = [{"role": "system", "content": load_instructions()}]
    if sid not in chat_locks:
        chat_locks[sid] = asyncio.Lock()
    return chat_histories[sid], chat_locks[sid]    
# admin endpoint
@app.get("/admin")
def admin_root(auth: str = Depends(check_auth)):
    return FileResponse("admin/index.html")
# File upload endpoint
@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    # basic sanitize + validation
    original_name = Path(file.filename).name
    if not original_name:
        raise HTTPException(status_code=400, detail='Invalid filename')
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail='File type not allowed')
    # sanitize and ensure unique filename
    sanitized = secure_filename(original_name)
    base, ext = os.path.splitext(sanitized)
    candidate = sanitized
    i = 1
    while os.path.exists(os.path.join(UPLOAD_FOLDER, candidate)):
        candidate = f"{base}_{i}{ext}"
        i += 1
    safe_name = candidate
    save_path = os.path.join(UPLOAD_FOLDER, safe_name)
    try:
        contents = await file.read()
        with open(save_path, 'wb') as out_f:
            out_f.write(contents)
    finally:
        await file.close()
    # Reload instructions after upload
    bot_prompt = load_bot_rules()
    lesson_notes = load_instructions()
    # Clear chat history, load bot rules as system prompt, load lesson notes as first user message
    for sid in chat_histories.items():
        chat_histories[sid].clear()
        chat_histories[sid] = [
            {"role": "system", "content": bot_prompt},
            {"role": "user", "content": lesson_notes}
        ]
    return {"filename": safe_name}
# File Delete Endpoint
@app.delete("/delete")
async def delete_resources(payload: dict):
    files = payload.get("resources", [])
    if not files:
        raise HTTPException(status_code=400, detail="No files specified for deletion")
    deleted_files = []
    errors = []
    for filename in files:
        # prevent directory traversal
        safe_name = Path(filename).name
        file_path = os.path.join(UPLOAD_FOLDER, safe_name)
        if not os.path.exists(file_path):
            errors.append(f"File not found: {safe_name}")
            continue
        try:
            os.remove(file_path)
            deleted_files.append(safe_name)
        except Exception as e:
            errors.append(f"Error deleting {safe_name}: {str(e)}")
    # Reload instructions after upload
    bot_prompt = load_bot_rules()
    lesson_notes = load_instructions()
    # Clear chat history, load bot rules as system prompt, load lesson notes as first user message
    for sid in chat_histories.items():
        chat_histories[sid].clear()
        chat_histories[sid] = [
            {"role": "system", "content": bot_prompt},
            {"role": "user", "content": lesson_notes}
        ]
    return {
        "deleted": deleted_files,
        "errors": errors
    }

# FastAPI chat endpoint
@app.post("/chat")
async def chat(payload: dict, x_session_id: str = Header(None, alias="X-Session-Id")):
    if not x_session_id:
        raise HTTPException(400, "Missing X-Session-Id header")
    # Get user message and chat history for this session
    user_input = payload["message"]
    chat_history, chat_lock = get_history_and_lock(x_session_id)
    if(user_input.strip() == "Hello my mechanized assistant!"):
        bot_rules = load_bot_rules()
        chat_history.clear()
        chat_history.append({"role": "system", "content": bot_rules})
        user_input = load_instructions()
        print(f"{"\x1b[43m"}[{x_session_id}] New Session Started.{"\x1b[0m"}")
    else:
        print(f"{"\x1b[43m"}[{x_session_id}] User: {user_input}{"\x1b[0m"}")
    # Create a temporary message list for this request (history + current user message)
    user_msg = {"role": "user", "content": user_input}
    temp_messages = list(chat_history) + [user_msg]
    # Define a generator to stream the response from the model
    async def stream():
        async with chat_lock:
            q: "queue.Queue[str|object]" = queue.Queue()
            DONE = object()
            # Worker thread to call the OpenAI client and stream results into the queue
            def worker():
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=temp_messages,
                        temperature=0.7,
                        stream=True,
                    )
                    for chunk in resp:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            q.put(delta.content)
                    q.put(DONE)
                except Exception as e:
                    q.put(f"[ERROR] {e}")
                    q.put(DONE)
            threading.Thread(target=worker, daemon=True).start()
            collected = ""
            # small “kick” helps some proxies/clients flush early
            yield ""
            while True:
                item = await asyncio.to_thread(q.get)  # don’t block event loop
                if item is DONE:
                    break
                collected += item
                yield item
            # commit after the model finishes
            chat_history.append(user_msg)
            chat_history.append({"role": "assistant", "content": collected})
            print(f'{"\x1b[42m"}[{x_session_id}] K1T B0T: {collected}{"\x1b[0m"}')
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",  # harmless even without nginx; some proxies respect it
    }
    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8", headers=headers)

# List to populate delete endpoint
@app.get("/list")
async def list_resources():
    if not os.path.exists(UPLOAD_FOLDER):
        return []
    return sorted([
        f for f in os.listdir(UPLOAD_FOLDER)
        if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))
    ])

@app.get("/debug")
async def debug():
    folder = UPLOAD_FOLDER
    exists = os.path.exists(folder)
    files = []
    if exists:
        files = sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    prompt = load_instructions()
    return {
        "UPLOAD_FOLDER": folder,
        "exists": exists,
        "files": files,
        "prompt_preview": prompt[:300],
        "prompt_length": len(prompt),
    }

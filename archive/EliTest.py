import os
from werkzeug.utils import secure_filename
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import shutil

# ===================================================================
# Kit Bot: LLM Chatbot using vLLM and Mistral Instruct
# Authors: Eli Gruhlke, Ian Walch, Will Dani, Beaumont Ujlaky, and Erik Greiner
# ===================================================================

# Initialize messages array
messages = []

def msg_to_chat(role, content):
    msg = {"role": role, "content": content}
    messages.append(msg)
    return msg

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to your local vLLM server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
model = "mistralai/Mistral-7B-Instruct-v0.3"

# Paths to your topic files
instructions_path = "/home/k1tbot/Documents/k1tbot/instructions"

# Read all files in instructions directory
try:
    for file in Path(instructions_path).glob("*.txt"):
        with open(file, "r") as f:
            bot_prompt = f.read().strip()
except Exception as e:
    bot_prompt = "Could not read instructions! Default behaviour: act like a helpful assistant"
msg_to_chat("system", bot_prompt)
# Uploads configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'instructions')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.csv', '.txt', '.json', '.mp4'}
# Mount uploads as static files so they can be served at /instructions/{filename}
app.mount('/instructions', StaticFiles(directory=UPLOAD_FOLDER), name='instructions')
# Streaming generator
def generate_stream(messages):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            stream=True
        )
        collected = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
                text = delta.content
                collected += text
        msg_to_chat("assistant", collected)
    except Exception as e:
        yield f"[ERROR] {str(e)}"
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
    # Create a sanitized, human-readable filename and avoid collisions by
    # appending numeric suffixes (file, file_1, file_2, ...)
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
    # Read all files in instructions directory
    bot_prompt = ""
    try:
        for file in Path(instructions_path).glob("*.txt"):
            with open(file, "r") as f:
                bot_prompt = f.read().strip()
    except Exception as e:
        bot_prompt = "Could not read instructions! Default behaviour: act like a helpful assistant"
    # Update system message
    messages[0] = {"role": "system", "content": bot_prompt}
    return {"filename": safe_name, "url": f"/instructions/{safe_name}", "status": "uploaded and reloaded"}

# FastAPI chat endpoint
@app.post("/chat")
async def chat(payload: dict):
    user_input = payload["message"]
    msg_to_chat("user", user_input)
    return StreamingResponse(
        generate_stream(messages),
        media_type="text/event-stream"
    )

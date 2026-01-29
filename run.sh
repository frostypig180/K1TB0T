# Activate virtual environment
source "/home/k1tbot/Documents/k1tbot/.venv/bin/activate"

# Start the vLLM server in the background
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 &

cd /home/k1tbot/Documents/k1tbot
# Start the FastAPI application with Uvicorn
uvicorn "main:app" --host 0.0.0.0 --port 8001 --reload &

# Start a simple HTTP server to serve static files
python3 -m http.server 5500
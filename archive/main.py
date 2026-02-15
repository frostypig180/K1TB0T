from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# ===================================================================
# Kit Bot: LLM Chatbot using vLLM and Mistral
# Authors: Eli Gruhlke, Ian Walch, Will Dani, Beau Ujlaky, Erik Greiner
# Project for Wireless Commiunications Enterprise at Michigan Technological University
# ===================================================================
# How to run on server PC:
# "source .venv/bin/activate". If you're already in .venv, you don't need to do this step
# "vllm serve mistralai/Mistral-7B-Instruct-v0.3 --port 8000"
# Once you see "(APIServer pid=24634) INFO:   Application startup complete", press the run button top right of the console or run "python .venv/main.py" in a new console
# Point to your local vLLM server (NOT OpenAI’s cloud)
# ===================================================================
# helper function to stream chat response
# ===================================================================
def stream_chat_response(messages):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        stream=True,
    )
    collected = ""  # final string to store full assistant response
    print("K1t Bot: ", end="", flush=True)
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            text = delta.content
            collected += text
            print(text, end="", flush=True)

    print("\n")  # finish the line
    return collected
# ===================================================================
# initialize kit bot
# ===================================================================
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
model = "mistralai/Mistral-7B-Instruct-v0.3"
file_path = "/home/k1tbot/Documents/k1tbot/.venv/BotPrompt.txt"
with open(file_path, "r") as file:
    bot_prompt = file.read() # K1tBot will read a text file which will give it a specific personality and instructions
messages = [
    {"role": "system", 
     "content": bot_prompt}
]
with open(".venv/Topic.txt", "r") as topic_file:
    topic = topic_file.read() # Initial question when program is run
messages.append(
    {"role": "user", 
     "content": topic}
) 
kit_response = stream_chat_response(messages)
messages.append({"role": "assistant", "content": kit_response})
# ===================================================================
# main chat loop
# ===================================================================
while True:
    content = input("> Enter message: ")
    if content.lower() == "goodbye":
        print("sweet dreams")
        break
    messages.append({"role": "user", "content": content})
    kit_response = stream_chat_response(messages)
    messages.append({"role": "assistant", "content": kit_response})

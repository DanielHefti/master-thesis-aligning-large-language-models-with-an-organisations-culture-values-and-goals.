import requests
import csv
import time
import sys
import json

# Read prompt from file
with open("public_dataset_without_instruction-prompt.txt", "r", encoding="utf-8") as f:
    prompt_without_instruction = f.read()

with open("public_dataset_full-prompt.txt", "r", encoding="utf-8") as f:
    prompt_with_instruction = f.read()

# Ollama API endpoint for chat
url = "https://my-ollama-api-endpoint.net/api/chat" # Replace with your actual Ollama API endpoint

models = [
    {
        "model": "smollm2:135m",
        "num_ctx": 8000 - 10,
        "intervention": False,
        "last_phase": 0
    },
    {
        "model": "llama3.2:1b",
        "num_ctx": 128000 - 10,
        "intervention": True,
        "last_phase": 0
    },
    {
        "model": "llama3.2:3b",
        "num_ctx": 128000 - 10,
        "intervention": False,
        "last_phase": 0
    },
    {
        "model": "qwen3:4b",
        "num_ctx": 256000 - 10,
        "intervention": True,
        "last_phase": 0
    },
    {
        "model": "qwen3:14b",
        "num_ctx": 40000 - 10,
        "intervention": False,
        "last_phase": 0
    },
    {
        "model": "deepseek-r1:14b",
        "num_ctx": 128000 - 10,
        "intervention": True,
        "last_phase": 0
    },
    {
        "model": "mistral-small:22b",
        "num_ctx": 128000 - 10,
        "intervention": False,
        "last_phase": 0
    },
    {
        "model": "deepseek-r1:32b",
        "num_ctx": 128000 - 10,
        "intervention": True,
        "last_phase": 0
    },
    {
        "model": "llama3.3:70b",
        "num_ctx": 128000 - 10,
        "intervention": False,
        "last_phase": 0
    },
]

# Iterate over phases and models
for phase in range(1,4):
    for model in models:
        if model["last_phase"] >= phase:
            print(f"Skipping model {model['model']} for phase {phase} as it was already evaluated in phase {model['last_phase']}.")
            continue
        rows = []
        for attempt in range(1, 11):
            print(f"Model: {model['model']}, Phase: {phase}, Attempt: {attempt}")
            # Select prompt based on phase and intervention
            if phase == 2 and model["intervention"]:
                prompt = prompt_with_instruction
                print("Using prompt with instruction.")
            else:
                prompt = prompt_without_instruction
            data = {
                "model": model["model"],
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repeat_penalty": 1.1,
                    "num_ctx": model["num_ctx"]
                },
                "stream": True
            }
            try:
                with requests.post(url, json=data, stream=True) as response:
                    resp_text = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = line.decode('utf-8')
                                chunk_json = json.loads(chunk)
                                content = chunk_json.get("message", {}).get("content", "")
                                resp_text += content
                            except Exception as e:
                                print(f"Error decoding chunk: {e}", file=sys.stderr)
                    if not resp_text:
                        resp_text = "[No response or empty stream]"
            except Exception as e:
                resp_text = f"Error: {e}"
            print(f"Response: {resp_text[:100]}...")  # Print first 100 characters of response
            rows.append({
                "model-name": model["model"],
                "phase": phase,
                "attempt": attempt,
                "response": resp_text
            })

        filename = f"experiment_{phase}_{model['model'].replace(':', '_')}.csv"
        print(f"Writing {len(rows)} rows to {filename}")
        # Only write the file if there is at least one row
        if rows:
            with open(filename, "w", newline='', encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["model-name", "phase", "attempt", "response"])
                writer.writeheader()
                writer.writerows(rows)
        else:
            print(f"No responses collected for model {model['model']}, file not written.")

        # Wait 5 minutes and 30 seconds before next model
        print(f"Waiting 5 minutes and 30 seconds before next model...")
        time.sleep(5 * 60 + 30)
    phase += 1
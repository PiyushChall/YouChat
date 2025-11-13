import os
import requests
import json

# Set your Gemini API key here
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"

# Simple test prompt
prompt = "Say hello in one sentence."

# API endpoint
url = "https://api.generativeai.google/v1beta/models/gemini-2.5-flash/outputs"

headers = {
    "Authorization": f"Bearer {GOOGLE_API_KEY}",
    "Content-Type": "application/json",
}

data = {
    "prompt": {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    },
    "temperature": 0.2,
    "max_output_tokens": 50
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    result = response.json()
    # Extract generated text
    text_output = result["candidates"][0]["content"][0]["text"]
    print("Gemini API is working! Generated text:")
    print(text_output)
except requests.exceptions.RequestException as e:
    print("Gemini API test failed:", e)
except KeyError:
    print("Unexpected response format:", response.text)

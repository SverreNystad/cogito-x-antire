import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hi!"}],
    },
)

print(f"Status: {response.status_code}")
if response.ok:
    data = response.json()
    print(f"Response: {data['choices'][0]['message']['content']}")
else:
    print(f"Error: {response.text}")

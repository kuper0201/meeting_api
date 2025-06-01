import requests

API_KEY = "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"  # <-- 여기에 OpenRouter API 키 입력

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "google/gemma-3-27b-it:free",  # 또는 "google/gemma-3b-it" 등으로 변경
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "한국어로 인사말 작성"}
    ]
}

response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

if response.status_code == 200:
    print(response.json()['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
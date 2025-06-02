import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests

# 요약용 프롬프트 구성 함수
def build_prompt(text: str) -> str:
    return f"""다음은 회의록입니다. 회의의 주요 주제, 핵심 논의 내용, 결정사항, 그리고 추가 논의가 필요한 사항을 항목별로 정리해서 요약해 주세요.

포맷은 다음과 같이 해주세요:
	- 회의 주제:
	- 논의된 주요 내용:
	- 결정된 사항:
	- 향후 과제 또는 Follow-up:

회의록: {text}
"""

API_KEY = ""

def summarize(text: str, max_new_tokens=1000) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = build_prompt(text)
    
    data = {
        "model": "google/gemma-3-27b-it:free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in summarizing meeting minutes."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        summary = response.json()['choices'][0]['message']['content']
        return summary
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return "요약 생성 중 오류가 발생했습니다."

# # 요약 실행 함수
# def summarize(text: str, max_new_tokens=1000) -> str:
#     print('start summarize')
#     print(text)

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # 모델과 토크나이저 불러오기
#     model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

#     prompt = build_prompt(text)
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

#     with torch.no_grad():
#         output_ids = model.generate(
#             input_ids,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             repetition_penalty=1.1,
#             eos_token_id=tokenizer.eos_token_id
#         )

#     output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     summary = output_text.split("회의록:")[-1].strip()
#     return summary